#!/usr/bin/env python3
"""Convert the v01 LabelMe rectangles to a YOLO detection dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080
CLASS_NAMES = {"0": "gripper_with_plug", "1": "gripper_without_plug"}


def split_for(stem: str, train_ratio: float) -> str:
    value = int(hashlib.md5(stem.encode("utf-8")).hexdigest(), 16) / float(1 << 128)
    return "train" if value < train_ratio else "val"


def clipped_box(points: list[list[float]], width: int, height: int) -> tuple[float, float, float, float]:
    if len(points) < 2:
        raise ValueError("rectangle must contain two corner points")
    x_values = [float(point[0]) for point in points]
    y_values = [float(point[1]) for point in points]
    x1 = max(0.0, min(x_values))
    y1 = max(0.0, min(y_values))
    x2 = min(float(width), max(x_values))
    y2 = min(float(height), max(y_values))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"rectangle has no visible area: {points}")
    return x1, y1, x2, y2


def yolo_line(label: str, points: list[list[float]], width: int, height: int) -> str:
    x1, y1, x2, y2 = clipped_box(points, width, height)
    center_x = (x1 + x2) / 2.0 / width
    center_y = (y1 + y2) / 2.0 / height
    box_width = (x2 - x1) / width
    box_height = (y2 - y1) / height
    return f"{int(label)} {center_x:.6f} {center_y:.6f} {box_width:.6f} {box_height:.6f}"


def convert(source: Path, output: Path, train_ratio: float) -> dict[str, int]:
    if not 0.0 <= train_ratio <= 1.0:
        raise ValueError("train_ratio must be between 0 and 1")
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")

    for split in ("train", "val"):
        (output / "images" / split).mkdir(parents=True)
        (output / "labels" / split).mkdir(parents=True)

    counts = {"images": 0, "train": 0, "val": 0, "class_0": 0, "class_1": 0}
    for annotation_path in sorted(source.glob("*.json")):
        data = json.loads(annotation_path.read_text(encoding="utf-8"))
        image_path = source / str(data.get("imagePath", annotation_path.with_suffix(".png").name))
        if not image_path.is_file():
            image_path = annotation_path.with_suffix(".png")
        if not image_path.is_file():
            raise FileNotFoundError(f"missing image for {annotation_path.name}")
        width = int(data.get("imageWidth") or IMAGE_WIDTH)
        height = int(data.get("imageHeight") or IMAGE_HEIGHT)
        shapes = data.get("shapes") or []
        if len(shapes) != 1:
            raise ValueError(f"expected one rectangle in {annotation_path.name}, got {len(shapes)}")
        shape = shapes[0]
        label = str(shape.get("label"))
        if label not in CLASS_NAMES:
            raise ValueError(f"unknown class {label!r} in {annotation_path.name}")
        if shape.get("shape_type") != "rectangle":
            raise ValueError(f"expected rectangle in {annotation_path.name}")
        split = split_for(annotation_path.stem, train_ratio)
        image_out = output / "images" / split / image_path.name
        label_out = output / "labels" / split / f"{image_path.stem}.txt"
        shutil.copy2(image_path, image_out)
        label_out.write_text(yolo_line(label, shape.get("points") or [], width, height) + "\n", encoding="utf-8")
        counts["images"] += 1
        counts[split] += 1
        counts[f"class_{label}"] += 1

    yaml = (
        "path: .\n"
        "train: images/train\n"
        "val: images/val\n\n"
        "names:\n"
        "  0: gripper_with_plug\n"
        "  1: gripper_without_plug\n"
    )
    (output / "data.yaml").write_text(yaml, encoding="utf-8")
    (output / "README.txt").write_text(
        "YOLO detection dataset converted from v01_成功1 LabelMe rectangles.\n"
        "Class 0: gripper_with_plug\n"
        "Class 1: gripper_without_plug\n",
        encoding="utf-8",
    )
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("v01_成功1"))
    parser.add_argument("--output", type=Path, default=Path("v01_成功1_yolo"))
    parser.add_argument("--train-ratio", type=float, default=0.8)
    args = parser.parse_args()
    print(convert(args.source, args.output, args.train_ratio))


if __name__ == "__main__":
    main()
