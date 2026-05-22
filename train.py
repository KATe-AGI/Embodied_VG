#!/usr/bin/env python3
"""Train YOLO26s plug segmentation and head/tail pose models."""

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
        "model": ULTRALYTICS_DIR / "yolo26s-seg.pt",
        "data": DATASET / "seg" / "plug_seg.yaml",
        "project": ULTRALYTICS_DIR / "runs" / "segment",
        "name": "plug_yolo26s_seg_v1",
    },
    "pose": {
        "model": ULTRALYTICS_DIR / "yolo26s-pose.pt",
        "data": DATASET / "pose" / "plug_pose.yaml",
        "project": ULTRALYTICS_DIR / "runs" / "pose",
        "name": "plug_yolo26s_pose_v1",
    },
}


def absolute_data_yaml(path: Path) -> Path:
    """Return a temporary dataset YAML whose path field is absolute."""
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
    parser.add_argument("--task", choices=("all", "segment", "pose"), default="all", help="Model(s) to train.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs.")
    parser.add_argument("--batch", type=int, default=32, help="Batch size. Use 4 if GPU memory is insufficient.")
    parser.add_argument("--patience", type=int, default=10, help="Early-stopping patience.")
    parser.add_argument("--workers", type=int, default=8, help="Dataloader workers.")
    parser.add_argument("--device", default=None, help="CUDA device, e.g. 0, or cpu.")
    parser.add_argument("--exist-ok", action="store_true", help="Reuse an existing run directory.")
    parser.add_argument("--seg-model", default=DEFAULTS["segment"]["model"], help="Segmentation model or checkpoint.")
    parser.add_argument("--pose-model", default=DEFAULTS["pose"]["model"], help="Pose model or checkpoint.")
    parser.add_argument("--seg-data", type=Path, default=DEFAULTS["segment"]["data"], help="Segmentation data YAML.")
    parser.add_argument("--pose-data", type=Path, default=DEFAULTS["pose"]["data"], help="Pose data YAML.")
    return parser.parse_args()


def train_one(task: str, model_path: str, data_yaml: Path, args: argparse.Namespace) -> None:
    run = DEFAULTS[task]
    data_abs = absolute_data_yaml(data_yaml)
    print(f"\n=== Training {task}: model={model_path}, data={data_abs} ===")
    model = YOLO(model_path)
    model.train(
        data=str(data_abs),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        project=str(run["project"]),
        name=run["name"],
        patience=args.patience,
        workers=args.workers,
        device=args.device,
        exist_ok=args.exist_ok,
    )


def main() -> None:
    args = parse_args()
    if args.task in ("all", "segment"):
        train_one("segment", args.seg_model, args.seg_data, args)
    if args.task in ("all", "pose"):
        train_one("pose", args.pose_model, args.pose_data, args)


if __name__ == "__main__":
    main()
