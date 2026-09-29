# The DROID Robot Platform

This repository contains the code for setting up your DROID robot platform and using it to collect teleoperated demonstration data. This platform was used to collect the [DROID dataset](https://droid-dataset.github.io), a large, in-the-wild dataset of robot manipulations.

If you are interested in using the DROID dataset for training robot policies, please check out our [policy learning repo](https://github.com/droid-dataset/droid_policy_learning).
For more information about DROID, please see the following links: 

[**[Homepage]**](https://droid-dataset.github.io) &ensp; [**[Documentation]**](https://droid-dataset.github.io/droid) &ensp; [**[Paper]**](https://arxiv.org/abs/2403.12945) &ensp; [**[Dataset Visualizer]**](https://droid-dataset.github.io/dataset.html).

![](https://droid-dataset.github.io/droid/assets/index/droid_teaser.jpg)

---------
## Setup Guide

We assembled a step-by-step guide for setting up the DROID robot platform in our [developer documentation](https://droid-dataset.github.io/droid).
This guide has been used to set up 18 DROID robot platforms over the course of the DROID dataset collection. Please refer to the steps in this guide for setting up your own robot. Specifically, you can follow these key steps:

1. [Hardware Assembly and Setup](https://droid-dataset.github.io/droid/docs/hardware-setup)
2. [Software Installation and Setup](https://droid-dataset.github.io/droid/docs/software-setup)
3. [Example Workflows to collect data or calibrate cameras](https://droid-dataset.github.io/droid/docs/example-workflows)

If you encounter issues during setup, please raise them as issues in this github repo.

---------
## Running the fine-tuned SmolVLA policy

The fine-tuned Fabric-DROID SmolVLA checkpoint is provided separately from this robot-platform repository, in the `droid_smolvla_inference` inference release. That release contains the inference wrapper, the 20,000-step checkpoint, and its matching preprocessing and normalization files. This DROID repository does not contain those files. Clone or copy the inference release and run the commands below from its root directory.

### Install

Python 3.10 is recommended. For CUDA inference, first make sure the machine has a working NVIDIA driver and a compatible PyTorch/CUDA environment.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-inference.txt
```

Checkpoint `.safetensors` files are stored with Git LFS. If the cloned checkpoint files contain only small pointer text files, install Git LFS and run `git lfs pull`.

### Predict an action

```python
import numpy as np
from inference.policy import SmolVLAInference

policy = SmolVLAInference(device="cuda")  # use "cpu" for a slow functional test

# Replace these example arrays with live robot observations.
state = np.zeros(8, dtype=np.float32)
exterior_rgb = np.zeros((480, 640, 3), dtype=np.uint8)
wrist_rgb = np.zeros((480, 640, 3), dtype=np.uint8)

action = policy.predict_action(
    {
        "observation.state": state,
        "observation.images.exterior_1_left": exterior_rgb,
        "observation.images.wrist_left": wrist_rgb,
        "task": "Remove the fabric from the rack and drop it into the basket.",
    }
)
print(action.shape, action)  # one unnormalized 8-D action
```

The state must be a `float32` vector with shape `(8,)`. The two required images are RGB, in HWC or CHW layout, either `uint8` in `[0, 255]` or float in `[0, 1]`. The camera keys `observation.images.camera1` and `observation.images.camera2` are accepted aliases. A third image, `observation.images.camera3`, is optional. If using OpenCV, convert images from BGR to RGB before passing them to the policy.

Use the task instruction that matches the behavior being requested. These are the language instructions used in the training data:

1. `Reach for the fabric, grasp it, and hold it for inspection.`
2. `Leave the fabric on the rack, release it, and return to the ready position.`
3. `Remove the fabric from the rack and drop it into the basket.`

Call `policy.reset()` at the start of a new episode or after resetting the task to clear the policy's cached action chunk.

### Important: validate before connecting a robot

The wrapper only predicts actions; it does not send commands to robot hardware. Its output has 8 dimensions and is postprocessed with the checkpoint's saved action unnormalization. Confirm the meaning, order, units, signs, and gripper convention of every state and action dimension against the training data and your robot driver. In particular, the checkpoint configuration describes `observation.state` as 6-D, while the training dataset metadata and saved normalization statistics are 8-D; the wrapper follows the actual training data and requires 8 state values. Do not pad or truncate values without a verified mapping.

Before hardware execution, also validate camera views and RGB ordering, implement robot-specific limits and rate/force constraints, and ensure an emergency stop and manual override are available. First test with offline data, simulation, or a low-speed controlled setup. The zero-filled observations above are only for checking that model loading and the API work; they are not suitable for robot control.
