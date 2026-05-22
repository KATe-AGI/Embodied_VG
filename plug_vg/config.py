"""Shared project paths, camera loading, and plug physical parameters."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
ULTRALYTICS_DIR = ROOT / "ultralytics"
DATASET = ROOT / "plug_dataset_all_20260520"
RGBD_TEST = DATASET / "rgbd_test"
YOLO_DATASET = DATASET / "yolo_train"

DEFAULT_CAMERA = ROOT / "configs" / "camera" / "plug_rgbd.yaml"
DEFAULT_MANIFEST = RGBD_TEST / "meta" / "frame_manifest.csv"

DEFAULT_SEG_WEIGHTS = ULTRALYTICS_DIR / "runs" / "segment" / "plug_yolo26s_seg_v1" / "weights" / "best.pt"
DEFAULT_POSE_WEIGHTS = ULTRALYTICS_DIR / "runs" / "pose" / "plug_yolo26s_pose_v1" / "weights" / "best.pt"

GRASP_REGION_LENGTH_M = 0.085
GRASP_REGION_WIDTH_M = 0.055
GRASP_REGION_THICKNESS_M = GRASP_REGION_WIDTH_M
HEAD_TAIL_DISTANCE_M = 0.165


def load_camera(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    required = ("fx", "fy", "cx", "cy", "depth_scale", "image_width", "image_height")
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"Camera config missing required key(s): {', '.join(missing)}")
    return data

