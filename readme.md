# Darts

Play darts with alwaysAI and the Intel RealSense depth camera!

## Note

This app is using an experimental and unreleased update to the edgeIQ RealSense API. You will be able to run this app using the QA image in the Dockerfile, but integrate this API into your own apps with caution.

## Setup
This app requires an alwaysAI account. Head to the [Sign up page](https://www.alwaysai.co/dashboard) if you don't have an account yet. Follow the instructions to install the alwaysAI tools on your development machine.

Next, create an empty project to be used with this app. When you clone this repo, you can run `aai app configure` within the repo directory and your new project will appear in the list.

## Usage
Once you have the alwaysAI tools installed and the new project created, run the following CLI commands at the top level of the repo:

To set the project and select the target device, run:

```
aai app configure
```

To build your app and install on the target device:

```
aai app install
```

To start the app:

```
aai app start
```
