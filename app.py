import time
import pandas as pd
import threading
import queue
import multiprocessing as mp
import edgeiq


class HistoryTracker:
    def __init__(self):
        self._columns = [
                'ts', 'wrist_x', 'wrist_y', 'wrist_z', 'eye_y', 'neck_y', 'distance_traveled',
                'inst_velocity', 'state']
        self._all_data = pd.DataFrame(columns=self._columns)
        self._history = pd.DataFrame(columns=self._columns)
        self.reset()

    def reset(self):
        self._all_data = self._all_data.append(
                self._history, ignore_index=True)
        self._history = pd.DataFrame(columns=self._columns)
        self.distance_traveled = 0.0
        self.inst_velocity = 0.0
        self.wrist = (0, 0, 0.0)
        self.eye = (0, 0, 0.0)
        self.neck = (0, 0, 0.0)

    def _get_last_data(self):
        if len(self._history) == 0:
            return None
        return self._history.iloc[-1]

    def update(self, data, state, ts=None):
        # Skip all missing or zeroed out points
        if data is None or data['Right Wrist'][2] == 0.0:
            return

        if ts is None:
            ts = time.time()
        self.wrist = data['Right Wrist']
        self.eye = data['Right Eye']
        self.neck = data['Neck']

        last_data = self._get_last_data()
        if last_data is not None:
            self.distance_traveled = last_data['wrist_z'] - self.wrist[2]
            self.inst_velocity = self.distance_traveled / (ts - last_data['ts'])

        new_row = {
                'ts': ts,
                'wrist_x': self.wrist[0],
                'wrist_y': self.wrist[1],
                'wrist_z': self.wrist[2],
                'eye_y': data['Right Eye'][1],
                'neck_y': data['Neck'][1],
                'distance_traveled': self.distance_traveled,
                'inst_velocity': self.inst_velocity,
                'state': state}
        self._history = self._history.append(new_row, ignore_index=True)

    def save(self):
        out_df = self._all_data.append(self._history, ignore_index=True)
        out_df.to_csv('run-{}.csv'.format(time.strftime('%Y.%m.%d-%H.%M.%S')))


class PoseHandler:
    def __init__(self):
        self.pose_estimator = edgeiq.PoseEstimation('alwaysai/human-pose')
        self.pose_estimator.load(engine=edgeiq.Engine.DNN)

        print('Loaded model:\n{}\n'.format(self.pose_estimator.model_id))
        print('Engine: {}'.format(self.pose_estimator.engine))
        print('Accelerator: {}\n'.format(self.pose_estimator.accelerator))

        self.last_data = {
                'Right Eye': (0, 0, 0.0),
                'Neck': (0, 0, 0.0),
                'Right Wrist': (0, 0, 0.0)
                }

        self.batch_thread = None
        self.frame_queue = queue.Queue()
        self.result_queue = queue.Queue()

    def _get_point_data(self, rs_frame, kp, name):
        if kp[name] != (-1, -1):
            depth = rs_frame.compute_object_distance(kp[name])
            return (
                    kp[name][0],
                    kp[name][1],
                    depth)
        else:
            return self.last_data[name]

    def get_pose_data(self, rs_frame):
        results = self.pose_estimator.estimate(rs_frame.image)
        if len(results.poses) == 0:
            return None
        kp = results.poses[0].key_points

        data = {}
        data['Right Eye'] = self._get_point_data(rs_frame, kp, 'Right Eye')
        data['Neck'] = self._get_point_data(rs_frame, kp, 'Neck')
        data['Right Wrist'] = self._get_point_data(rs_frame, kp, 'Right Wrist')
        if data['Right Wrist'][2] > data['Neck'][2]:
            print('Ignoring invalid wrist depth...')
            data['Right Wrist'] = (0, 0, 0.0)
        self.last_data = data

        return data

    def _batch_thread_target(self):
        while True:
            in_data = self.frame_queue.get()
            if in_data is None:
                print('Batch processing complete')
                return

            print('Processing batch frame')

            out_data = self.get_pose_data(in_data['rs_frame'])
            self.result_queue.put({'data': out_data, 'ts': in_data['ts']})

    def submit_for_batch_processing(self, rs_frame):
        if self.batch_thread is None:
            print('Starting batch processing')
            self.batch_thread = threading.Thread(target=self._batch_thread_target, daemon=True)
            # self.batch_thread = mp.Process(target=self._batch_thread_target)
            self.batch_thread.start()

        self.frame_queue.put({'rs_frame': rs_frame.get_portable_realsense_frame(), 'ts': time.time()})

    def complete_batch(self):
        self.frame_queue.put(None)

    def check_batch_finished(self):
        if self.batch_thread.is_alive() is False:
            self.batch_thread.join()
            self.batch_thread = None
            return True
        else:
            return False

    def get_batch_results(self):
        results = []
        while True:
            try:
                result = self.result_queue.get_nowait()
            except queue.Empty:
                break

            results.append(result)
        return results


class Darts:
    def __init__(self):
        self._state = 'waiting'
        self._throwing_ctr = 0
        self._THROWING_CTR_MAX = 15
        self._history = HistoryTracker()
        self._pose_hdlr = PoseHandler()
        self._image = None
        self._info = []
        self._to_go = 5
        self._score = 0

    def _update_state(self, new_state):
        print('State: {} => {}'.format(self._state, new_state))
        self._state = new_state
        self._throwing_ctr = 0

    def update(self, rs_frame):
        self._image = rs_frame.image
        self._info = [
                'Score: {}'.format(self._score),
                'Remaining attempts: {}'.format(self._to_go)
                ]

        if self._state == 'waiting':
            data = self._pose_hdlr.get_pose_data(rs_frame)
            self._history.update(data, self._state)
            self._info = self._info + [
                    'Get into throwing position!',
                    'Wrist pos: ({}, {}) {:2.2f}m'.format(
                        self._history.wrist[0],
                        self._history.wrist[1],
                        self._history.wrist[2]),
                    'Eye pos: {}'.format(self._history.eye[1]),
                    'Neck pos: {}'.format(self._history.neck[1]),
                    'Inst velocity: {:2.2f}m/s'.format(self._history.inst_velocity)
                    ]

            # Transition to ready when wrist is in position between
            # eyes and neck and velocity has stopped
            if data['Right Wrist'][1] > data['Right Eye'][1] and \
                    data['Right Wrist'][1] < data['Neck'][1] and \
                    self._history.inst_velocity < 0.01:
                self._update_state('ready')

        if self._state == 'ready':
            data = self._pose_hdlr.get_pose_data(rs_frame)
            self._history.update(data, self._state)
            self._info = self._info + [
                'Throw!',
                'Wrist depth: {:2.2f}m'.format(self._history.wrist[2]),
                'Inst velocity: {:2.2f}m/s'.format(self._history.inst_velocity)
                ]

            # Transition to throwing when forward movement begins
            if self._history.inst_velocity > 0.3:
                self._history.reset()
                self._update_state('throwing')

        if self._state == 'throwing':
            self._pose_hdlr.submit_for_batch_processing(rs_frame)
            self._info = self._info + ['Throwing']

            # Transition to flying when forward velocity stops
            self._throwing_ctr += 1
            if self._throwing_ctr >= self._THROWING_CTR_MAX:
                self._pose_hdlr.complete_batch()
                self._update_state('flying')

        if self._state == 'flying':
            self._info = self._info + ['The dart is flying!']
            if self._pose_hdlr.check_batch_finished():
                results = self._pose_hdlr.get_batch_results()
                for result in results:
                    self._history.update(result['data'], 'throwing', result['ts'])
                # Process results
                self._score += 10

                # Update attempts
                self._to_go -= 1
                if self._to_go == 0:
                    self._score = 0
                self._update_state('waiting')

    @property
    def image(self):
        return self._image

    @property
    def text(self):
        return self._info

    def save(self):
        self._history.save()


def main():

    fps = edgeiq.FPS()

    darts = Darts()

    try:
        with edgeiq.RealSense() as video_stream, \
                edgeiq.Streamer() as streamer:
            # Allow Webcam to warm up
            time.sleep(2.0)
            fps.start()

            # loop detection
            while True:
                rs_frame = video_stream.read()
                darts.update(rs_frame)

                streamer.send_data(darts.image, darts.text)

                fps.update()

                if streamer.check_exit():
                    break
    finally:
        fps.stop()
        print('elapsed time: {:.2f}'.format(fps.get_elapsed_seconds()))
        print('approx. FPS: {:.2f}'.format(fps.compute_fps()))
        darts.save()

        print('Program Ending')


if __name__ == '__main__':
    main()
