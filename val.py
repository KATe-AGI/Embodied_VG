#!/usr/bin/env python3
"""Validate trained YOLO26s plug segmentation and head/tail pose models."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
ULTRALYTICS_DIR = ROOT / "ultralytics"
if str(ULTRALYTICS_DIR) not in sys.path:
    sys.path.insert(0, str(ULTRALYTICS_DIR))

from ultralytics import YOLO  # noqa: E402


DATASET = ROOT / "plug_dataset_all_20260520" / "yolo_train"
DEFAULTS = {
    "segment": {
        "weights": ULTRALYTICS_DIR / "runs" / "segment" / "plug_yolo26s_seg_v1" / "weights" / "best.pt",
        "data": DATASET / "seg" / "plug_seg.yaml",
        "project": ULTRALYTICS_DIR / "runs" / "segment",
        "name": "plug_yolo26s_seg_v1_val",
    },
    "pose": {
        "weights": ULTRALYTICS_DIR / "runs" / "pose" / "plug_yolo26s_pose_v1" / "weights" / "best.pt",
        "data": DATASET / "pose" / "plug_pose.yaml",
        "project": ULTRALYTICS_DIR / "runs" / "pose",
        "name": "plug_yolo26s_pose_v1_val",
    },
}


def absolute_data_yaml(path: Path) -> Path:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    raw_path = Path(data.get("path", "."))
    if raw_path.is_absolute():
        dataset_path = raw_path
    elif (ROOT / raw_path).exists():
        dataset_path = ROOT / raw_path
    else:
        dataset_path = path.parent / raw_path
    data["path"] = str(dataset_path.resolve())

    tmp = tempfile.NamedTemporaryFile("w", suffix=f"_{path.name}", encoding="utf-8", delete=False)
    with tmp:
        yaml.safe_dump(data, tmp, allow_unicode=True, sort_keys=False)
    return Path(tmp.name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("all", "segment", "pose"), default="all", help="Model(s) to validate.")
    parser.add_argument("--imgsz", type=int, default=640, help="Validation image size.")
    parser.add_argument("--batch", type=int, default=8, help="Validation batch size.")
    parser.add_argument("--device", default=None, help="CUDA device, e.g. 0, or cpu.")
    parser.add_argument("--exist-ok", action="store_true", help="Reuse an existing validation directory.")
    parser.add_argument("--seg-weights", type=Path, default=DEFAULTS["segment"]["weights"], help="Segmentation weights.")
    parser.add_argument("--pose-weights", type=Path, default=DEFAULTS["pose"]["weights"], help="Pose weights.")
    parser.add_argument("--seg-data", type=Path, default=DEFAULTS["segment"]["data"], help="Segmentation data YAML.")
    parser.add_argument("--pose-data", type=Path, default=DEFAULTS["pose"]["data"], help="Pose data YAML.")
    return parser.parse_args()


def val_one(task: str, weights: Path, data_yaml: Path, args: argparse.Namespace) -> None:
    run = DEFAULTS[task]
    data_abs = absolute_data_yaml(data_yaml)
    print(f"\n=== Validating {task}: weights={weights}, data={data_abs} ===")
    model = YOLO(str(weights))
    model.val(
        data=str(data_abs),
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(run["project"]),
        name=run["name"],
        device=args.device,
        exist_ok=args.exist_ok,
    )


def main() -> None:
    args = parse_args()
    if args.task in ("all", "segment"):
        val_one("segment", args.seg_weights, args.seg_data, args)
    if args.task in ("all", "pose"):
        val_one("pose", args.pose_weights, args.pose_data, args)


if __name__ == "__main__":
    main()
