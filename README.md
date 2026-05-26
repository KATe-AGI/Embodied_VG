# EmbodiedVG

EmbodiedVG is the vision-side pipeline for plug grasping. The current real-machine validation entry takes one RGB image, one registered D2RGB depth PNG, and the current robot end-effector pose, then outputs a plug 6D grasp frame in the robot base frame. Window geometry can be provided when the grasp must be constrained through a known window.

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

Default direct-visual mode:

```bash
python infer_6d_single.py \
  --rgb plug_dataset_all_20260520/rgbd_test/color/color_20260519_215321_955_7.png \
  --d2rgb plug_dataset_all_20260520/rgbd_test/D2RGB/D2RGB_20260519_215321_955_7.png \
  --robot-pose 0.42 -0.18 0.63 0.3 -0.2 0.5 \
  --output-dir ultralytics/runs/plug_6d_single \
  --save-overlay
```

Window-constrained mode:

```bash
python infer_6d_single.py \
  --rgb plug_dataset_all_20260520/rgbd_test/color/color_20260519_215321_955_7.png \
  --d2rgb plug_dataset_all_20260520/rgbd_test/D2RGB/D2RGB_20260519_215321_955_7.png \
  --robot-pose 0.42 -0.18 0.63 0.3 -0.2 0.5 \
  --output-dir ultralytics/runs/plug_6d_single \
  --window-config configs/window/box_window.yaml \
  --save-overlay
```

Window geometry is optional. Provide either `--window-config` or `--window-corners-base` to enable window-constrained candidate generation. If neither is provided, the final grasp pose is the direct visual base-frame pose.

The output JSON contains:

- `grasp_solution_mode`: `direct_visual` without window geometry, or `window_constrained` when window geometry is enabled.
- `grasp_pose_camera`: visual 6D estimate in the RGB camera frame.
- `grasp_pose_base`: the visual estimate transformed to robot base frame. In direct-visual mode this is the final grasp pose; in window-constrained mode it is a `surface_normal_reference`.
- `window_constrained_grasp_candidates`: sorted base-frame grasp poses constrained by the window geometry; present only in window-constrained mode.
- `best_grasp_pose_base`: the first window candidate and the downstream grasp pose to consume; present only in window-constrained mode.
- `grasp_point_base_m`: final grasp point `(x, y, z)` in the robot base frame. It is estimated from a robust midsection 3D center, not from a single surface anchor point.
- `tail_to_head_axis_base`: virtual tail/head endpoints on the plug `tail -> head` axis. The axis uses the visual reference pose `+X`, is centered on `grasp_point_base_m`, and uses the configured plug head-tail distance.
- `quality.grasp_center_estimation`: diagnostics for the robust midsection center, including candidate counts, depth outlier rejection, surface center, and the half-thickness center offset.

In direct-visual mode, the grasp frame is the original visual grasp coordinate system. In window-constrained mode, `best_grasp_pose_base` uses the selected window approach frame while `tail_to_head_axis_base` still describes the plug body's visual `tail -> head` direction.
