#!/usr/bin/env python3
"""Train the YOLO26s visible-plug segmentation model."""

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


DATASET = ROOT / "yolo_plug_dataset"
DEFAULT_MODEL = ULTRALYTICS_DIR / "yolo26n-seg.pt"
DEFAULT_DATA = DATASET / "data.yaml"
DEFAULT_PROJECT = ULTRALYTICS_DIR / "runs" / "segment"
DEFAULT_NAME = "plug_yolo26n_seg_20260724"


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
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs.")
    parser.add_argument("--batch", type=int, default=32, help="Batch size. Use 4 if GPU memory is insufficient.")
    parser.add_argument("--patience", type=int, default=10, help="Early-stopping patience.")
    parser.add_argument("--workers", type=int, default=8, help="Dataloader workers.")
    parser.add_argument("--device", default=None, help="CUDA device, e.g. 0, or cpu.")
    parser.add_argument("--exist-ok", action="store_true", help="Reuse an existing run directory.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Segmentation model or checkpoint.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Segmentation data YAML.")
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT, help="Ultralytics output project directory.")
    parser.add_argument("--name", default=DEFAULT_NAME, help="Ultralytics run name.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_abs = absolute_data_yaml(args.data)
    print(f"\n=== Training visible-plug segmentation: model={args.model}, data={data_abs} ===")
    model = YOLO(args.model)
    model.train(
        data=str(data_abs),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        project=str(args.project),
        name=args.name,
        patience=args.patience,
        workers=args.workers,
        device=args.device,
        exist_ok=args.exist_ok,
    )


if __name__ == "__main__":
    main()
