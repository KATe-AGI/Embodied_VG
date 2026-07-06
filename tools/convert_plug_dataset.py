#!/usr/bin/env python3
"""Convert plug camera captures and LabelMe visible-mask annotations into a segmentation dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


LABEL_PLUG = "plug_grasp_region"
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080
LABELME_JSON_PATTERNS = ("color_*.json", "undistort_color_*.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--camera-dir",
        type=Path,
        default=Path("plug_camera_20260520"),
        help="Optional flat camera capture directory. If it is missing, sibling PNGs beside LabelMe JSON files are used.",
    )
    parser.add_argument("--annotation-dir", type=Path, default=Path("plug_annotation_20260520"), help="Directory containing LabelMe JSON files.")
    parser.add_argument("--rgbd-test-dir", type=Path, default=None, help="Optional camera capture directory for 6D test color/D2RGB files.")
    parser.add_argument("--output", type=Path, default=Path("plug_dataset_20260529"), help="Output dataset root.")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="MD5 split train ratio in [0, 1].")
    parser.add_argument("--force", action="store_true", help="Remove output directory first if it already exists.")
    return parser.parse_args()


def raw_id_from_color(path: Path) -> str:
    stem = path.stem
    for prefix in ("undistort_color_", "color_"):
        if stem.startswith(prefix):
            return stem.removeprefix(prefix)
    return stem


def md5_split(raw_id: str, train_ratio: float) -> str:
    value = int(hashlib.md5(raw_id.encode("utf-8")).hexdigest(), 16) / float(1 << 128)
    return "train" if value < train_ratio else "val"


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def fmt(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


def norm_x(x: float, width: int) -> float:
    return clamp(x, 0.0, width - 1.0) / width


def norm_y(y: float, height: int) -> float:
    return clamp(y, 0.0, height - 1.0) / height


def find_camera_files(camera_dir: Path) -> dict[str, dict[str, Path]]:
    files: dict[str, dict[str, Path]] = {}
    if not camera_dir.is_dir():
        return files
    prefixes = {
        "color": "color_png",
        "undistort_color": "undistort_color_png",
    }
    for path in camera_dir.iterdir():
        if not path.is_file():
            continue
        for prefix, key in prefixes.items():
            marker = f"{prefix}_"
            if path.name.startswith(marker):
                raw_id = path.stem.removeprefix(marker)
                files.setdefault(raw_id, {})[key] = path
                break
    return files


def collect_labelme(annotation_dir: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    duplicates: list[str] = []
    for pattern in LABELME_JSON_PATTERNS:
        for path in sorted(annotation_dir.rglob(pattern)):
            raw_id = raw_id_from_color(path)
            if raw_id in found:
                duplicates.append(raw_id)
                continue
            found[raw_id] = path
    if duplicates:
        names = ", ".join(sorted(set(duplicates))[:10])
        raise ValueError(f"Duplicate LabelMe annotations for raw_id(s): {names}")
    return found


def shape_points(shape: dict[str, Any]) -> list[list[float]]:
    points = shape.get("points") or []
    return [[float(x), float(y)] for x, y in points if len([x, y]) == 2]


def polygons(shapes: list[dict[str, Any]], label: str) -> list[list[list[float]]]:
    items: list[list[list[float]]] = []
    for shape in shapes:
        if shape.get("label") == label and shape.get("shape_type") == "polygon":
            points = shape_points(shape)
            if len(points) >= 3:
                items.append(points)
    return items


def yolo_seg_line(points: list[list[float]], width: int, height: int) -> str:
    coords: list[str] = ["0"]
    for x, y in points:
        coords.extend([fmt(norm_x(x, width)), fmt(norm_y(y, height))])
    return " ".join(coords)


def rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def write_yaml_files(output: Path) -> None:
    seg = output / "yolo_train" / "seg" / "plug_seg.yaml"
    seg.parent.mkdir(parents=True, exist_ok=True)
    seg.write_text(
        f"path: {output.as_posix()}/yolo_train/seg\n"
        "train: images/train\n"
        "val: images/val\n\n"
        "names:\n"
        "  0: visible_plug\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def copy_rgbd_test_dir(src_dir: Path, dst_dir: Path) -> list[dict[str, Any]]:
    if not src_dir.is_dir():
        raise NotADirectoryError(f"RGB-D test source is not a directory: {src_dir}")
    color_dir = dst_dir / "color"
    d2rgb_dir = dst_dir / "D2RGB"
    color_dir.mkdir(parents=True, exist_ok=True)
    d2rgb_dir.mkdir(parents=True, exist_ok=True)

    for src in sorted(src_dir.glob("color_*.png")):
        if src.is_file():
            shutil.copy2(src, color_dir / src.name)
    for pattern in ("D2RGB_*.png", "D2RGB_*.jpg", "*_d2rgb.npy"):
        for src in sorted(src_dir.glob(pattern)):
            if src.is_file():
                shutil.copy2(src, d2rgb_dir / src.name)

    rows: list[dict[str, Any]] = []
    for color in sorted(color_dir.glob("color_*.png")):
        raw_id = raw_id_from_color(color)
        d2png = d2rgb_dir / f"D2RGB_{raw_id}.png"
        d2jpg = d2rgb_dir / f"D2RGB_{raw_id}.jpg"
        d2npy = d2rgb_dir / f"{raw_id}_d2rgb.npy"
        rows.append(
            {
                "raw_id": raw_id,
                "color_png": f"color/{color.name}",
                "D2RGB_png": f"D2RGB/{d2png.name}" if d2png.exists() else "",
                "D2RGB_jpg": f"D2RGB/{d2jpg.name}" if d2jpg.exists() else "",
                "D2RGB_npy": f"D2RGB/{d2npy.name}" if d2npy.exists() else "",
                "has_d2rgb": d2png.exists() or d2jpg.exists() or d2npy.exists(),
            }
        )
    return rows


def convert(args: argparse.Namespace) -> None:
    if not 0.0 <= args.train_ratio <= 1.0:
        raise ValueError("--train-ratio must be in [0, 1]")
    if args.output.exists():
        if not args.force:
            raise FileExistsError(f"Output already exists: {args.output}. Use --force to overwrite.")
        shutil.rmtree(args.output)

    camera_files = find_camera_files(args.camera_dir)
    annotations = collect_labelme(args.annotation_dir)
    yolo = args.output / "yolo_train"
    for path in [
        yolo / "annotations_standard",
        yolo / "meta",
        yolo / "seg" / "images" / "train",
        yolo / "seg" / "images" / "val",
        yolo / "seg" / "labels" / "train",
        yolo / "seg" / "labels" / "val",
    ]:
        path.mkdir(parents=True, exist_ok=True)

    split_rows: list[dict[str, Any]] = []
    label_summary_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    counts = {"standard_annotations": 0, "seg_samples": 0}

    for raw_id, label_path in sorted(annotations.items()):
        split = md5_split(raw_id, args.train_ratio)
        color_name = f"color_{raw_id}.png"
        label_name = f"color_{raw_id}.txt"
        camera_entry = camera_files.get(raw_id, {})
        color_src = camera_entry.get("color_png") or camera_entry.get("undistort_color_png") or label_path.with_suffix(".png")
        if not color_src.exists():
            raise FileNotFoundError(f"Missing color image for {raw_id}: {color_src}")

        data = json.loads(label_path.read_text(encoding="utf-8"))
        width = int(data.get("imageWidth") or IMAGE_WIDTH)
        height = int(data.get("imageHeight") or IMAGE_HEIGHT)
        plug_polys = polygons(data.get("shapes") or [], LABEL_PLUG)
        has_plug = bool(plug_polys)
        label_error = "" if has_plug else "missing_visible_plug_polygon"

        seg_image_rel = ""
        seg_label_rel = ""
        if has_plug:
            seg_image = yolo / "seg" / "images" / split / color_name
            seg_label = yolo / "seg" / "labels" / split / label_name
            shutil.copy2(color_src, seg_image)
            seg_label.write_text("\n".join(yolo_seg_line(poly, width, height) for poly in plug_polys) + "\n", encoding="utf-8")
            seg_image_rel = rel(seg_image, yolo)
            seg_label_rel = rel(seg_label, yolo)
            counts["seg_samples"] += 1

        labels = {LABEL_PLUG: [{"type": "polygon", "points": poly} for poly in plug_polys]}
        standard = {
            "raw_id": raw_id,
            "split": split,
            "files": {
                "color_png": seg_image_rel or None,
                "seg_image": seg_image_rel or None,
                "seg_label": seg_label_rel or None,
            },
            "labels": labels,
            "derived": {
                "has_visible_plug_polygon": has_plug,
                "use_for_yolo_seg": has_plug,
            },
        }
        standard_path = yolo / "annotations_standard" / f"color_{raw_id}.standard.json"
        standard_path.write_text(json.dumps(standard, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        counts["standard_annotations"] += 1

        split_rows.append({"raw_id": raw_id, "split": split})
        label_summary_rows.append(
            {
                "raw_id": raw_id,
                "split": split,
                "has_labelme": True,
                "has_visible_plug_polygon": has_plug,
                "use_for_yolo_seg": has_plug,
                "label_error": label_error,
            }
        )
        manifest_rows.append(
            {
                "raw_id": raw_id,
                "split": split,
                "label_error": label_error,
                "standard_annotation": rel(standard_path, yolo),
                "seg_image": seg_image_rel,
                "seg_label": seg_label_rel,
            }
        )

    write_yaml_files(args.output)
    write_csv(yolo / "meta" / "split.csv", split_rows, ["raw_id", "split"])
    write_csv(
        yolo / "meta" / "label_summary.csv",
        label_summary_rows,
        ["raw_id", "split", "has_labelme", "has_visible_plug_polygon", "use_for_yolo_seg", "label_error"],
    )
    write_csv(
        yolo / "meta" / "frame_manifest.csv",
        manifest_rows,
        ["raw_id", "split", "label_error", "standard_annotation", "seg_image", "seg_label"],
    )

    rgbd_info: dict[str, Any] | None = None
    if args.rgbd_test_dir is not None:
        rgbd = args.output / "rgbd_test"
        rgbd_rows = copy_rgbd_test_dir(args.rgbd_test_dir, rgbd)
        write_csv(rgbd / "meta" / "frame_manifest.csv", rgbd_rows, ["raw_id", "color_png", "D2RGB_png", "D2RGB_jpg", "D2RGB_npy", "has_d2rgb"])
        rgbd_info = {"manifest": "rgbd_test/meta/frame_manifest.csv", "counts": {"manifest_rows": len(rgbd_rows)}}

    split_counts = {
        "train": sum(1 for row in split_rows if row["split"] == "train"),
        "val": sum(1 for row in split_rows if row["split"] == "val"),
    }
    info = {
        "dataset_name": args.output.name,
        "source": {
            "camera_dir": str(args.camera_dir) if args.camera_dir.is_dir() else None,
            "annotation_dir": str(args.annotation_dir),
            "rgbd_test_dir": None if args.rgbd_test_dir is None else str(args.rgbd_test_dir),
        },
        "split_policy": {"method": "md5(raw_id)", "train_ratio": args.train_ratio},
        "yolo_train": {
            "segmentation_yaml": "yolo_train/seg/plug_seg.yaml",
            "annotations_standard": "yolo_train/annotations_standard",
            "meta": "yolo_train/meta",
        },
        "rgbd_test": rgbd_info,
        "label_convention": {LABEL_PLUG: "LabelMe polygon for visible plug pixels."},
        "counts": {**counts, "split": split_counts, "label_errors": sum(1 for row in label_summary_rows if row["label_error"])},
    }
    (yolo / "meta" / "dataset_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Generated {args.output}: {counts['seg_samples']} segmentation samples.")


def main() -> None:
    convert(parse_args())


if __name__ == "__main__":
    main()
