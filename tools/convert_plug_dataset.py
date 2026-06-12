#!/usr/bin/env python3
"""Convert plug camera captures and LabelMe annotations into the project dataset layout."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


LABEL_GRASP = "plug_grasp_region"
LABEL_HEAD = "plug_head"
LABEL_TAIL = "plug_tail"
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080
BBOX_MARGIN_RATIO = 0.1
LABELME_JSON_PATTERNS = ("color_*.json", "undistort_color_*.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--camera-dir",
        type=Path,
        default=Path("plug_camera_20260520"),
        help="Optional flat camera capture directory. If it is missing, sibling PNGs beside LabelMe JSON files are used.",
    )
    parser.add_argument("--annotation-dir", type=Path, default=Path("plug_annotation_20260520"), help="Directory containing one or more LabelMe subdirectories.")
    parser.add_argument(
        "--rgbd-test-dir",
        type=Path,
        default=None,
        help="Optional camera capture directory for 6D test data. Copies color/*.png and D2RGB/*.{png,jpg}.",
    )
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


def norm_x(x: float, width: int) -> float:
    return clamp(x, 0.0, width - 1.0) / width


def norm_y(y: float, height: int) -> float:
    return clamp(y, 0.0, height - 1.0) / height


def fmt(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


def find_camera_files(camera_dir: Path) -> dict[str, dict[str, Path]]:
    files: dict[str, dict[str, Path]] = {}
    if not camera_dir.is_dir():
        return files
    prefixes = {
        "color": "color_png",
        "D2RGB": "D2RGB",
        "depth": "depth",
        "Color3D(map_to_color)": "Color3D_map_to_color_xyz",
        "Color3D": "Color3D_xyz",
        "Point3D": "Point3D_xyz",
        "RGB2D": "RGB2D_png",
        "leftIR": "leftIR_png",
        "undistort_color": "undistort_color_png",
    }
    for path in camera_dir.iterdir():
        if not path.is_file():
            continue
        for prefix, key in prefixes.items():
            marker = f"{prefix}_"
            if path.name.startswith(marker):
                raw_id = path.stem.removeprefix(marker)
                entry = files.setdefault(raw_id, {})
                if key == "D2RGB":
                    if path.suffix.lower() == ".png":
                        entry["D2RGB_png"] = path
                    elif path.suffix.lower() == ".jpg":
                        entry["D2RGB_jpg"] = path
                elif key == "depth":
                    if path.suffix.lower() == ".png":
                        entry["depth_png"] = path
                    elif path.suffix.lower() == ".jpg":
                        entry["depth_jpg"] = path
                else:
                    entry[key] = path
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


def first_point(shapes: list[dict[str, Any]], label: str) -> list[float] | None:
    for shape in shapes:
        if shape.get("label") == label and shape.get("shape_type") == "point":
            points = shape_points(shape)
            if points:
                return points[0]
    return None


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


def bbox_from_points(points: list[list[float]], width: int, height: int) -> tuple[float, float, float, float]:
    xs = [clamp(p[0], 0.0, width - 1.0) for p in points]
    ys = [clamp(p[1], 0.0, height - 1.0) for p in points]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    bw = x2 - x1
    bh = y2 - y1
    margin = max(bw, bh) * BBOX_MARGIN_RATIO
    x1 = clamp(x1 - margin, 0.0, width - 1.0)
    y1 = clamp(y1 - margin, 0.0, height - 1.0)
    x2 = clamp(x2 + margin, 0.0, width - 1.0)
    y2 = clamp(y2 + margin, 0.0, height - 1.0)
    return x1, y1, x2, y2


def yolo_pose_line(polys: list[list[list[float]]], head: list[float], tail: list[float], width: int, height: int) -> str:
    bbox_points = [point for poly in polys for point in poly] + [head, tail]
    x1, y1, x2, y2 = bbox_from_points(bbox_points, width, height)
    xc = ((x1 + x2) / 2.0) / width
    yc = ((y1 + y2) / 2.0) / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    values = [
        "0",
        fmt(xc),
        fmt(yc),
        fmt(bw),
        fmt(bh),
        fmt(norm_x(head[0], width)),
        fmt(norm_y(head[1], height)),
        "2",
        fmt(norm_x(tail[0], width)),
        fmt(norm_y(tail[1], height)),
        "2",
    ]
    return " ".join(values)


def copy_if_exists(src: Path | None, dst: Path) -> str:
    if src is None or not src.exists():
        return ""
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return str(dst)


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
    for pattern in ("D2RGB_*.png", "D2RGB_*.jpg"):
        for src in sorted(src_dir.glob(pattern)):
            if src.is_file():
                shutil.copy2(src, d2rgb_dir / src.name)

    rows: list[dict[str, Any]] = []
    for color in sorted(color_dir.glob("color_*.png")):
        raw_id = raw_id_from_color(color)
        d2png = d2rgb_dir / f"D2RGB_{raw_id}.png"
        d2jpg = d2rgb_dir / f"D2RGB_{raw_id}.jpg"
        rows.append(
            {
                "raw_id": raw_id,
                "color_png": f"color/{color.name}",
                "D2RGB_png": f"D2RGB/{d2png.name}" if d2png.exists() else "",
                "D2RGB_jpg": f"D2RGB/{d2jpg.name}" if d2jpg.exists() else "",
                "has_d2rgb": d2png.exists(),
            }
        )
    return rows


def rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def write_yaml_files(output: Path) -> None:
    seg = output / "yolo_train" / "seg" / "plug_seg.yaml"
    pose = output / "yolo_train" / "pose" / "plug_pose.yaml"
    seg.write_text(
        f"path: {output.as_posix()}/yolo_train/seg\n"
        "train: images/train\n"
        "val: images/val\n\n"
        "names:\n"
        "  0: plug_grasp_region\n",
        encoding="utf-8",
    )
    pose.write_text(
        f"path: {output.as_posix()}/yolo_train/pose\n"
        "train: images/train\n"
        "val: images/val\n\n"
        "kpt_shape: [2, 3]\n"
        "flip_idx: [0, 1]\n\n"
        "names:\n"
        "  0: plug\n\n"
        "kpt_names:\n"
        "  0:\n"
        "    - plug_head\n"
        "    - plug_tail\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def convert(args: argparse.Namespace) -> None:
    if not 0.0 <= args.train_ratio <= 1.0:
        raise ValueError("--train-ratio must be in [0, 1]")
    if args.output.exists():
        if not args.force:
            raise FileExistsError(f"Output already exists: {args.output}. Use --force to overwrite.")
        shutil.rmtree(args.output)

    camera_dir_used = args.camera_dir if args.camera_dir.is_dir() else None
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
        yolo / "pose" / "images" / "train",
        yolo / "pose" / "images" / "val",
        yolo / "pose" / "labels" / "train",
        yolo / "pose" / "labels" / "val",
    ]:
        path.mkdir(parents=True, exist_ok=True)

    split_rows: list[dict[str, Any]] = []
    label_summary_rows: list[dict[str, Any]] = []
    yolo_manifest_rows: list[dict[str, Any]] = []
    counts = {
        "standard_annotations": 0,
        "seg_samples": 0,
        "pose_samples": 0,
    }

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
        shapes = data.get("shapes") or []
        grasp_polys = polygons(shapes, LABEL_GRASP)
        head = first_point(shapes, LABEL_HEAD)
        tail = first_point(shapes, LABEL_TAIL)
        has_grasp = bool(grasp_polys)
        has_head = head is not None
        has_tail = tail is not None
        pose_valid = has_grasp and has_head and has_tail
        errors: list[str] = []
        if not has_grasp:
            errors.append("missing_grasp_region")
        if not has_head:
            errors.append("missing_head")
        if not has_tail:
            errors.append("missing_tail")

        seg_image_rel = seg_label_rel = pose_image_rel = pose_label_rel = ""
        if has_grasp:
            seg_image = yolo / "seg" / "images" / split / color_name
            seg_label = yolo / "seg" / "labels" / split / label_name
            shutil.copy2(color_src, seg_image)
            seg_label.write_text("\n".join(yolo_seg_line(poly, width, height) for poly in grasp_polys) + "\n", encoding="utf-8")
            seg_image_rel = rel(seg_image, yolo)
            seg_label_rel = rel(seg_label, yolo)
            counts["seg_samples"] += 1

        if pose_valid:
            pose_image = yolo / "pose" / "images" / split / color_name
            pose_label = yolo / "pose" / "labels" / split / label_name
            shutil.copy2(color_src, pose_image)
            pose_label.write_text(yolo_pose_line(grasp_polys, head, tail, width, height) + "\n", encoding="utf-8")
            pose_image_rel = rel(pose_image, yolo)
            pose_label_rel = rel(pose_label, yolo)
            counts["pose_samples"] += 1

        labels = {
            LABEL_GRASP: [{"type": "polygon", "points": poly} for poly in grasp_polys],
            LABEL_HEAD: None if head is None else {"type": "point", "point": head, "visible": True},
            LABEL_TAIL: None if tail is None else {"type": "point", "point": tail, "visible": True},
        }
        standard = {
            "raw_id": raw_id,
            "split": split,
            "files": {
                "color_png": seg_image_rel or pose_image_rel or None,
                "color_json": None,
                "D2RGB_png": None,
                "D2RGB_jpg": None,
                "depth_png": None,
                "depth_jpg": None,
                "Color3D_map_to_color_xyz": None,
                "Color3D_xyz": None,
                "Point3D_xyz": None,
                "RGB2D_png": None,
                "leftIR_png": None,
                "undistort_color_png": None,
                "seg_image": seg_image_rel or None,
                "seg_label": seg_label_rel or None,
                "pose_image": pose_image_rel or None,
                "pose_label": pose_label_rel or None,
            },
            "labels": labels,
            "derived": {
                "has_labelme": True,
                "has_grasp_region": has_grasp,
                "has_head": has_head,
                "has_tail": has_tail,
                "pose_valid": pose_valid,
                "use_for_yolo_seg": has_grasp,
                "use_for_yolo_pose": pose_valid,
            },
        }
        standard_path = yolo / "annotations_standard" / f"color_{raw_id}.standard.json"
        standard_path.write_text(json.dumps(standard, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        counts["standard_annotations"] += 1

        split_rows.append({"raw_id": raw_id, "split": split})
        label_error = ";".join(errors)
        label_summary_rows.append(
            {
                "raw_id": raw_id,
                "split": split,
                "has_labelme": True,
                "has_grasp_region": has_grasp,
                "has_head": has_head,
                "has_tail": has_tail,
                "pose_valid": pose_valid,
                "use_for_yolo_seg": has_grasp,
                "use_for_yolo_pose": pose_valid,
                "label_error": label_error,
            }
        )
        yolo_manifest_rows.append(
            {
                "raw_id": raw_id,
                "split": split,
                "label_error": label_error,
                "standard_annotation": rel(standard_path, yolo),
                "seg_image": seg_image_rel,
                "seg_label": seg_label_rel,
                "pose_image": pose_image_rel,
                "pose_label": pose_label_rel,
            }
        )

    write_yaml_files(args.output)
    write_csv(yolo / "meta" / "split.csv", split_rows, ["raw_id", "split"])
    write_csv(
        yolo / "meta" / "label_summary.csv",
        label_summary_rows,
        [
            "raw_id",
            "split",
            "has_labelme",
            "has_grasp_region",
            "has_head",
            "has_tail",
            "pose_valid",
            "use_for_yolo_seg",
            "use_for_yolo_pose",
            "label_error",
        ],
    )
    write_csv(
        yolo / "meta" / "frame_manifest.csv",
        yolo_manifest_rows,
        ["raw_id", "split", "label_error", "standard_annotation", "seg_image", "seg_label", "pose_image", "pose_label"],
    )
    rgbd_info: dict[str, Any] | None = None
    if args.rgbd_test_dir is not None:
        rgbd = args.output / "rgbd_test"
        rgbd_rows = copy_rgbd_test_dir(args.rgbd_test_dir, rgbd)
        write_csv(rgbd / "meta" / "frame_manifest.csv", rgbd_rows, ["raw_id", "color_png", "D2RGB_png", "D2RGB_jpg", "has_d2rgb"])
        rgbd_info = {
            "source_dir": str(args.rgbd_test_dir),
            "layout": "Project RGB-D test layout with color/ and D2RGB/ subdirectories.",
            "manifest": "rgbd_test/meta/frame_manifest.csv",
            "color": "rgbd_test/color/color_*.png",
            "D2RGB_png": "rgbd_test/D2RGB/D2RGB_*.png",
            "D2RGB_jpg": "rgbd_test/D2RGB/D2RGB_*.jpg",
            "counts": {
                "color_images": sum(1 for _ in (rgbd / "color").glob("color_*.png")),
                "d2rgb_png": sum(1 for _ in (rgbd / "D2RGB").glob("D2RGB_*.png")),
                "d2rgb_jpg": sum(1 for _ in (rgbd / "D2RGB").glob("D2RGB_*.jpg")),
                "manifest_rows": len(rgbd_rows),
                "with_d2rgb": sum(1 for row in rgbd_rows if row["has_d2rgb"]),
                "missing_d2rgb": sum(1 for row in rgbd_rows if not row["has_d2rgb"]),
            },
        }

    split_counts = {
        "train": sum(1 for row in split_rows if row["split"] == "train"),
        "val": sum(1 for row in split_rows if row["split"] == "val"),
    }
    info = {
        "dataset_name": args.output.name,
        "source": {
            "camera_dir": None if camera_dir_used is None else str(camera_dir_used),
            "annotation_dir": str(args.annotation_dir),
            "rgbd_test_dir": None if args.rgbd_test_dir is None else str(args.rgbd_test_dir),
        },
        "split_policy": {
            "method": "md5(raw_id)",
            "train_ratio": args.train_ratio,
        },
        "layout": {
            "yolo_train": "YOLO train/val dataset generated from LabelMe annotations.",
            "rgbd_test": "Optional RGB-D test dataset aligned with plug_dataset/rgbd_test; generated only when --rgbd-test-dir is provided.",
        },
        "yolo_train": {
            "segmentation_yaml": "yolo_train/seg/plug_seg.yaml",
            "pose_yaml": "yolo_train/pose/plug_pose.yaml",
            "annotations_standard": "yolo_train/annotations_standard",
            "meta": "yolo_train/meta",
        },
        "rgbd_test": rgbd_info,
        "label_convention": {
            LABEL_GRASP: "LabelMe polygon for the graspable plug body region.",
            LABEL_HEAD: "LabelMe point for the plug head/insertion/outward end.",
            LABEL_TAIL: "LabelMe point for the cable-side end.",
        },
        "counts": {
            **counts,
            "split": split_counts,
            "label_errors": sum(1 for row in label_summary_rows if row["label_error"]),
        },
    }
    (yolo / "meta" / "dataset_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    message = f"Generated {args.output}: {counts['seg_samples']} seg samples, {counts['pose_samples']} pose samples"
    if rgbd_info is not None:
        rgbd_counts = rgbd_info["counts"]
        message += (
            f", {rgbd_counts['color_images']} rgbd test color images "
            f"({rgbd_counts['with_d2rgb']} with D2RGB, {rgbd_counts['missing_d2rgb']} missing D2RGB)"
        )
    else:
        message += ", rgbd_test not generated"
    print(message + ".")


def main() -> None:
    convert(parse_args())


if __name__ == "__main__":
    main()
