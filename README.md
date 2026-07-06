# EmbodiedVG

EmbodiedVG is the vision-side pipeline for plug grasping. The current real-machine validation entry uses visible-plug YOLO segmentation, D2RGB mask back-projection, and grasp-frame CAD point-cloud registration to output a plug 6D grasp frame in the robot base frame.

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
python -m py_compile infer_6d_single.py plug_vg/grasp_model.py plug_vg/model_registration.py plug_vg/visible_points.py
python infer_6d_single.py --help
```

Optional import check:

```bash
python - <<'PY'
import torch
import cv2
import numpy as np
import yaml
from ultralytics import YOLO

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
  --rgb test_20260703/20260705_104219_123_color.png \
  --d2rgb test_20260703/20260705_104219_123_d2rgb.npy \
  --scene-ply test_20260703/20260705_104219_123_pointcloud.ply \
  --robot-pose 0.42 -0.18 0.63 0.3 -0.2 0.5 \
  --output-dir ultralytics/runs/plug_6d_single \
  --model-config configs/plug_models/2175B.yaml \
  --save-overlay \
  --save-ply \
  --save-review
```

`--scene-ply` is optional and is used only for the interactive review. Registration input comes from D2RGB pixels inside the visible YOLO mask.

The output JSON contains:

- `status`: `ok`, `failed`, or `ambiguous`.
- `t_camera_grasp`: CAD grasp frame pose in the RGB camera frame, only emitted for `ok`.
- `grasp_pose_camera` and `grasp_pose_base`: executable grasp pose when registration is accepted.
- `semantic_points_camera` and `semantic_points_base`: grasp, tail, and head semantic points transformed from `configs/plug_models/2175B.yaml`.
- `grasp_point_base_m`: final grasp center in robot base frame.
- `tail_to_head_axis_base`: semantic plug axis in robot base frame.
- `visible_point_cloud`: visible mask/D2RGB extraction diagnostics.
- `registration_quality`: ICP candidate, fitness, RMSE, ambiguity, and quality gate diagnostics.
- `artifacts`: visible mask overlay, visible points PLY, and optional interactive registration review HTML.

When `status` is `ambiguous`, the JSON is not an executable robot command. The review artifact can still show the best registration candidate for manual inspection.

## Stage-1 Segmentation Debug

`infer.py` is a segmentation-only debug helper:

```bash
python infer.py --source test_20260703 --output ultralytics/runs/plug_stage1_seg_debug
```

It writes segmentation JSON and mask overlays for visual inspection.
