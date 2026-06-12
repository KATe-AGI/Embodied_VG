#!/usr/bin/env python3
"""Merge plug datasets that share the project dataset layout."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_INPUTS = [Path("plug_dataset"), Path("plug_dataset_20260520")]
DEFAULT_OUTPUT = Path("plug_dataset_all_20260520")
PARTS = ("yolo_train", "rgbd_test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", default=DEFAULT_INPUTS, help="Input dataset roots to merge.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output merged dataset root.")
    parser.add_argument(
        "--parts",
        nargs="+",
        choices=("all", *PARTS),
        default=["all"],
        help="Dataset parts to merge. Use all, yolo_train, and/or rgbd_test.",
    )
    parser.add_argument("--force", action="store_true", help="Remove output directory first if it already exists.")
    return parser.parse_args()


def selected_parts(parts: list[str]) -> set[str]:
    return set(PARTS) if "all" in parts else set(parts)


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_checked(src: Path, dst: Path) -> bool:
    """Copy src to dst. Return True if copied, False if an identical file already existed."""
    if dst.exists():
        if file_hash(src) == file_hash(dst):
            return False
        raise FileExistsError(f"Conflicting file already exists: {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def merge_csvs(inputs: list[Path], output: Path, key_field: str = "raw_id") -> tuple[int, int]:
    fieldnames: list[str] = []
    rows_by_key: dict[str, dict[str, str]] = {}
    row_count = 0
    duplicate_count = 0

    for path in inputs:
        if not path.exists():
            continue
        fields, rows = read_csv(path)
        for field in fields:
            if field not in fieldnames:
                fieldnames.append(field)
        for row in rows:
            row_count += 1
            key = row.get(key_field, "")
            if key and key in rows_by_key:
                normalized_old = {name: rows_by_key[key].get(name, "") for name in fieldnames}
                normalized_new = {name: row.get(name, "") for name in fieldnames}
                if normalized_old != normalized_new:
                    raise ValueError(f"Conflicting CSV row for {key_field}={key} while merging {path}")
                duplicate_count += 1
                continue
            rows_by_key[key or f"__row_{row_count}"] = row

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows_by_key.values():
            writer.writerow({name: row.get(name, "") for name in fieldnames})
    return len(rows_by_key), duplicate_count


def write_yolo_yaml(output: Path) -> None:
    seg = output / "yolo_train" / "seg" / "plug_seg.yaml"
    pose = output / "yolo_train" / "pose" / "plug_pose.yaml"
    seg.parent.mkdir(parents=True, exist_ok=True)
    pose.parent.mkdir(parents=True, exist_ok=True)
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


def copy_glob(inputs: list[Path], relative_glob: str, output: Path) -> tuple[int, int]:
    copied = 0
    skipped = 0
    for dataset in inputs:
        for src in sorted(dataset.glob(relative_glob)):
            if not src.is_file() or src.suffix == ".cache":
                continue
            dst = output / src.relative_to(dataset)
            if copy_checked(src, dst):
                copied += 1
            else:
                skipped += 1
    return copied, skipped


def copy_yolo_manifest_files(inputs: list[Path], output: Path) -> tuple[Counter, Counter]:
    """Copy only YOLO files referenced by each input frame_manifest.csv."""
    copied = Counter()
    skipped = Counter()
    fields = {
        "standard_annotation": "standard_annotations",
        "seg_image": "seg_{split}_images",
        "seg_label": "seg_{split}_labels",
        "pose_image": "pose_{split}_images",
        "pose_label": "pose_{split}_labels",
    }

    for dataset in inputs:
        yolo_root = dataset / "yolo_train"
        manifest = yolo_root / "meta" / "frame_manifest.csv"
        if not manifest.exists():
            raise FileNotFoundError(f"Input dataset is missing yolo_train frame manifest: {manifest}")
        _, rows = read_csv(manifest)
        for row in rows:
            split = row.get("split", "")
            for field, count_name_template in fields.items():
                rel_value = row.get(field, "")
                if not rel_value:
                    continue
                rel_path = Path(rel_value)
                if rel_path.is_absolute() or ".." in rel_path.parts:
                    raise ValueError(f"Unsafe {field} path in {manifest}: {rel_value}")
                src = yolo_root / rel_path
                if not src.is_file():
                    raise FileNotFoundError(f"Manifest references missing file: {src}")
                dst = output / "yolo_train" / rel_path
                count_name = count_name_template.format(split=split)
                if copy_checked(src, dst):
                    copied[count_name] += 1
                else:
                    skipped[count_name] += 1
    return copied, skipped


def merge_yolo_train(inputs: list[Path], output: Path) -> dict[str, Any]:
    copied, skipped = copy_yolo_manifest_files(inputs, output)

    write_yolo_yaml(output)
    meta_inputs = [dataset / "yolo_train" / "meta" for dataset in inputs]
    split_rows, split_dupes = merge_csvs([meta / "split.csv" for meta in meta_inputs], output / "yolo_train" / "meta" / "split.csv")
    summary_rows, summary_dupes = merge_csvs(
        [meta / "label_summary.csv" for meta in meta_inputs],
        output / "yolo_train" / "meta" / "label_summary.csv",
    )
    manifest_rows, manifest_dupes = merge_csvs(
        [meta / "frame_manifest.csv" for meta in meta_inputs],
        output / "yolo_train" / "meta" / "frame_manifest.csv",
    )

    split_counts = Counter()
    split_csv = output / "yolo_train" / "meta" / "split.csv"
    if split_csv.exists():
        _, rows = read_csv(split_csv)
        split_counts.update(row.get("split", "") for row in rows)

    info = {
        "dataset_name": output.name,
        "source_datasets": [str(path) for path in inputs],
        "merged_parts": ["yolo_train"],
        "layout": {
            "yolo_train": "Merged YOLO train/val dataset.",
        },
        "yolo_train": {
            "segmentation_yaml": "yolo_train/seg/plug_seg.yaml",
            "pose_yaml": "yolo_train/pose/plug_pose.yaml",
            "annotations_standard": "yolo_train/annotations_standard",
            "meta": "yolo_train/meta",
        },
        "counts": {
            **dict(copied),
            "split_rows": split_rows,
            "label_summary_rows": summary_rows,
            "frame_manifest_rows": manifest_rows,
            "split": dict(split_counts),
        },
        "duplicates_skipped": {
            **{name: count for name, count in skipped.items() if count},
            "split_rows": split_dupes,
            "label_summary_rows": summary_dupes,
            "frame_manifest_rows": manifest_dupes,
        },
    }
    info_path = output / "yolo_train" / "meta" / "dataset_info.json"
    info_path.parent.mkdir(parents=True, exist_ok=True)
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return info


def merge_rgbd_test(inputs: list[Path], output: Path) -> dict[str, Any]:
    copied = Counter()
    skipped = Counter()
    patterns = {
        "color_images": "rgbd_test/color/color_*.png",
        "d2rgb_png": "rgbd_test/D2RGB/D2RGB_*.png",
        "d2rgb_jpg": "rgbd_test/D2RGB/D2RGB_*.jpg",
    }
    for name, pattern in patterns.items():
        c, s = copy_glob(inputs, pattern, output)
        copied[name] += c
        skipped[name] += s

    manifest_rows, manifest_dupes = merge_csvs(
        [dataset / "rgbd_test" / "meta" / "frame_manifest.csv" for dataset in inputs],
        output / "rgbd_test" / "meta" / "frame_manifest.csv",
    )

    rgbd_root = output / "rgbd_test"
    return {
        "counts": {
            **dict(copied),
            "manifest_rows": manifest_rows,
            "with_d2rgb": count_manifest_value(rgbd_root / "meta" / "frame_manifest.csv", "has_d2rgb", "True"),
            "missing_d2rgb": count_manifest_value(rgbd_root / "meta" / "frame_manifest.csv", "has_d2rgb", "False"),
        },
        "duplicates_skipped": {
            **{name: count for name, count in skipped.items() if count},
            "manifest_rows": manifest_dupes,
        },
    }


def update_dataset_info(output: Path, inputs: list[Path], parts: set[str], summaries: dict[str, Any]) -> None:
    info_path = output / "yolo_train" / "meta" / "dataset_info.json"
    if not info_path.exists():
        return
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["source_datasets"] = [str(path) for path in inputs]
    info["merged_parts"] = sorted(parts)
    if "rgbd_test" in summaries:
        info.setdefault("layout", {})["rgbd_test"] = "Merged RGB-D test dataset."
        info["rgbd_test"] = {
            "manifest": "rgbd_test/meta/frame_manifest.csv",
            "color": "rgbd_test/color/color_*.png",
            "D2RGB_png": "rgbd_test/D2RGB/D2RGB_*.png",
            "D2RGB_jpg": "rgbd_test/D2RGB/D2RGB_*.jpg",
            "counts": summaries["rgbd_test"]["counts"],
        }
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def count_manifest_value(path: Path, field: str, value: str) -> int:
    if not path.exists():
        return 0
    _, rows = read_csv(path)
    return sum(1 for row in rows if row.get(field) == value)


def validate_inputs(inputs: list[Path], parts: set[str]) -> None:
    for dataset in inputs:
        if not dataset.is_dir():
            raise NotADirectoryError(f"Input dataset not found: {dataset}")
        for part in parts:
            if not (dataset / part).is_dir():
                raise NotADirectoryError(f"Input dataset {dataset} is missing requested part: {part}")


def main() -> None:
    args = parse_args()
    parts = selected_parts(args.parts)
    validate_inputs(args.inputs, parts)

    if args.output.exists():
        if not args.force:
            raise FileExistsError(f"Output already exists: {args.output}. Use --force to overwrite.")
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True, exist_ok=True)

    summaries: dict[str, Any] = {}
    if "yolo_train" in parts:
        summaries["yolo_train"] = merge_yolo_train(args.inputs, args.output)
    if "rgbd_test" in parts:
        summaries["rgbd_test"] = merge_rgbd_test(args.inputs, args.output)
    update_dataset_info(args.output, args.inputs, parts, summaries)

    print(f"Merged {', '.join(str(path) for path in args.inputs)} -> {args.output}")
    for part, summary in summaries.items():
        print(f"\n[{part}]")
        for key, value in summary["counts"].items():
            print(f"  {key}: {value}")
        skipped = {key: value for key, value in summary["duplicates_skipped"].items() if value}
        if skipped:
            print("  duplicates_skipped:")
            for key, value in skipped.items():
                print(f"    {key}: {value}")


if __name__ == "__main__":
    main()
