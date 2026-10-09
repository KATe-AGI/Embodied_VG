#!/usr/bin/env python3
"""Train a YOLO model with a task-specific augmentation policy."""

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


DATASET = ROOT / "yolo-plug_head_and_tail_dataset"
DEFAULT_MODEL = ULTRALYTICS_DIR / "yolo26n-seg.pt"
DEFAULT_DATA = DATASET / "data.yaml"
DEFAULT_PROJECT = ULTRALYTICS_DIR / "runs" / "segment"
DEFAULT_NAME = "plug_yolo26n_seg_20260904"
DEFAULT_TASK = "segment"


def absolute_data_yaml(path: Path) -> Path:
    """Return a temporary dataset YAML whose path field is absolute."""
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    raw_path = Path(data.get("path", "."))
    if raw_path.is_absolute():
        dataset_path = raw_path
    else:
        # Dataset YAML paths are relative to the YAML file, as in the YOLO format.
        # Keep a project-root fallback for older project-level YAML files.
        dataset_path = path.parent / raw_path
        if not dataset_path.exists() and (ROOT / raw_path).exists():
            dataset_path = ROOT / raw_path
    data["path"] = str(dataset_path.resolve())

    tmp = tempfile.NamedTemporaryFile("w", suffix=f"_{path.name}", encoding="utf-8", delete=False)
    with tmp:
        yaml.safe_dump(data, tmp, allow_unicode=True, sort_keys=False)
    return Path(tmp.name)


def load_augmentation_policy(task: str, path: Path | None = None) -> dict[str, object]:
    """Load one task policy from the shared augmentation YAML."""
    policy_path = path or (ROOT / "augmentation.yaml")
    if not policy_path.is_file():
        raise FileNotFoundError(
            f"No augmentation policy file: {policy_path}. "
            "Create it next to train.py or pass --augmentation."
        )
    with policy_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    policy = data.get(task)
    if not isinstance(policy, dict):
        available = ", ".join(sorted(str(key) for key in data)) or "none"
        raise ValueError(f"no '{task}' policy in {policy_path}; available tasks: {available}")
    return policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("segment", "detect"), default=DEFAULT_TASK, help="YOLO task type.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs.")
    parser.add_argument("--batch", type=int, default=32, help="Batch size. Use 4 if GPU memory is insufficient.")
    parser.add_argument("--patience", type=int, default=10, help="Early-stopping patience.")
    parser.add_argument("--workers", type=int, default=8, help="Dataloader workers.")
    parser.add_argument("--device", default=None, help="CUDA device, e.g. 0, or cpu.")
    parser.add_argument("--exist-ok", action="store_true", help="Reuse an existing run directory.")
    parser.add_argument(
        "--augmentation",
        type=Path,
        default=None,
        help="Shared task augmentation YAML. Defaults to augmentation.yaml next to train.py.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="YOLO model or checkpoint.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Dataset YAML.")
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT, help="Ultralytics output project directory.")
    parser.add_argument("--name", default=DEFAULT_NAME, help="Ultralytics run name.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_abs = absolute_data_yaml(args.data)
    augmentation = load_augmentation_policy(args.task, args.augmentation)
    policy_path = args.augmentation or (ROOT / "augmentation.yaml")
    print(f"\n=== Training task={args.task}: model={args.model}, data={data_abs} ===")
    print(f"=== Augmentation policy: {policy_path} ===")
    model = YOLO(args.model)
    model.train(
        task=args.task,
        data=str(data_abs),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        project=str(args.project.resolve()),
        name=args.name,
        patience=args.patience,
        workers=args.workers,
        device=args.device,
        exist_ok=args.exist_ok,
        **augmentation,
    )


if __name__ == "__main__":
    main()
