"""Shared project paths and camera loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ULTRALYTICS_DIR = ROOT / "ultralytics"
DATASET = ROOT / "yolo_plug_dataset"
YOLO_DATASET = DATASET
YOLO_VAL_IMAGES = YOLO_DATASET / "images" / "val"

RGBD_DATASET = ROOT / "plug_dataset_all_20260529"
RGBD_TEST = RGBD_DATASET / "rgbd_test"

DEFAULT_CAMERA = ROOT / "configs" / "camera" / "plug_rgbd.yaml"
DEFAULT_MANIFEST = RGBD_TEST / "meta" / "frame_manifest.csv"

DEFAULT_SEG_WEIGHTS = ULTRALYTICS_DIR / "runs" / "segment" / "plug_yolo26n_seg_20260814" / "weights" / "best.pt"


def load_camera(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    required = ("fx", "fy", "cx", "cy", "depth_scale", "image_width", "image_height")
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"Camera config missing required key(s): {', '.join(missing)}")
    return data
