# EmbodiedVG

EmbodiedVG is the vision-side pipeline for plug grasping. The current real-machine validation entry takes one RGB image, one registered D2RGB depth PNG, and the current robot end-effector pose, then outputs the plug grasp frame in the robot base frame.

## Create Conda Environment

```bash
conda create -n embodiedvg python=3.11 -y
conda activate embodiedvg
```

## Install PyTorch

For an NVIDIA CUDA 12.8 machine:

```bash
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
```

If the target host uses a different CUDA version or only CPU, install the matching PyTorch build from the official PyTorch selector first, then continue with the project requirements.

## Install Project Requirements

From the project root:

```bash
pip install -r requirement.txt
```

`requirement.txt` installs the local `./ultralytics` source tree in editable mode, so keep the `ultralytics/` directory together with this project when migrating to another host.

## Verify Installation

```bash
python -m py_compile infer_6d_single.py plug_vg/robot_transform.py
python infer_6d_single.py --help
```

Optional import check:

```bash
python - <<'PY'
import torch
import cv2
import numpy as np
import yaml
from infer import YOLO

print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("opencv:", cv2.__version__)
print("numpy:", np.__version__)
print("pyyaml:", yaml.__version__)
print("YOLO:", YOLO)
PY
```

## Single-Frame 6D Inference

```bash
python infer_6d_single.py \
  --rgb plug_dataset_all_20260520/rgbd_test/color/color_20260519_215321_955_7.png \
  --d2rgb plug_dataset_all_20260520/rgbd_test/D2RGB/D2RGB_20260519_215321_955_7.png \
  --robot-pose 0.42 -0.18 0.63 0.3 -0.2 0.5 \
  --output-dir ultralytics/runs/plug_6d_single \
  --save-overlay
```

The output JSON contains `grasp_pose_camera` and `grasp_pose_base`. In the grasp frame, `+X` points from plug tail to head and `+Z` is the approach direction from the camera-visible side into the plug. `grasp_pose_base` is the vision-estimated grasp frame in the robot base frame; it is not a direct TCP motion target unless a grasp-frame-to-TCP transform is added later.
