import time
import pandas as pd
import edgeiq


class HistoryTracker:
    def __init__(self):
        self._columns = [
                'ts', 'val', 'distance_traveled',
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
        self.val = 0.0

    def _get_last_val(self):
        if len(self._history) == 0:
            return None
        return self._history.iloc[-1]

    def update(self, new_val, state):
        if new_val is None or new_val == 0.0:
            return

        ts = time.time()
        self.val = new_val

        last_val = self._get_last_val()
        if last_val is not None:
            self.distance_traveled = last_val['val'] - self.val
            self.inst_velocity = self.distance_traveled / (ts - last_val['ts'])

        new_row = {
                'ts': ts,
                'val': self.val,
                'distance_traveled': self.distance_traveled,
                'inst_velocity': self.inst_velocity,
                'state': state}
        self._history = self._history.append(new_row, ignore_index=True)

    def save(self):
        out_df = self._all_data.append(self._history, ignore_index=True)
        out_df.to_csv('run-{}.csv'.format(time.strftime('%Y.%m.%d-%H.%M.%S')))


class Darts:
    def __init__(self):
        self._state = 'line-up'
        self._history = HistoryTracker()
        self._image = None
        self._info = ''
        self._to_go = 5
        self._score = 0

    def _update_state(self, new_state):
        print('State: {} => {}'.format(self._state, new_state))
        self._state = new_state

    def update(self, rs_frame, results):
        self._image = rs_frame.image
        if len(results.poses) > 0:
            if results.poses[0].key_points['Right Wrist'] != (-1, -1):
                wrist_distance = rs_frame.compute_object_distance(
                        results.poses[0].key_points['Right Wrist'])
            else:
                wrist_distance = None

            self._history.update(wrist_distance, self._state)

        if self._state == 'line-up':
            self._info = [
                'Line up!',
                'Last wrist pos: {:2.2f}m'.format(self._history.val),
                'Inst velocity: {:2.2f}m'.format(self._history.inst_velocity)
                ]
            print(self._info)

            # Transition to throwing when forward movement begins
            if self._history.inst_velocity > 0.3:
                self._history.reset()
                self._update_state('throwing')

        if self._state == 'throwing':
            self._info = [
                'Throwing!',
                'Last wrist pos: {:2.2f}m'.format(self._history.val),
                'Inst velocity: {:2.2f}m'.format(self._history.inst_velocity)
                ]
            print(self._info)

            # Transition to flying when forward velocity stops
            if self._history.inst_velocity < 0.1:
                self._update_state('flying')

        if self._state == 'flying':
            self._info = [
                'The dart is flying!',
                'Last wrist pos: {:2.2f}m'.format(self._history.val),
                'Inst velocity: {:2.2f}m'.format(self._history.inst_velocity)
                ]
            print(self._info)
            time.sleep(10)
            self._update_state('line-up')

    @property
    def image(self):
        return self._image

    @property
    def text(self):
        return self._info

    def save(self):
        self._history.save()


def main():
    pose_estimator = edgeiq.PoseEstimation('alwaysai/human-pose')
    pose_estimator.load(engine=edgeiq.Engine.DNN)

    print('Loaded model:\n{}\n'.format(pose_estimator.model_id))
    print('Engine: {}'.format(pose_estimator.engine))
    print('Accelerator: {}\n'.format(pose_estimator.accelerator))

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
                results = pose_estimator.estimate(rs_frame.image)

                darts.update(rs_frame, results)
                # Generate text to display on streamer
                text = ['Model: {}'.format(pose_estimator.model_id)]
                text.append(
                        'Inference time: {:1.3f} s'.format(results.duration))
                text.append(darts.text)

                streamer.send_data(darts.image, text)

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
