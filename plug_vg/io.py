"""Shared file discovery and JSON/CSV helpers."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def collect_sources(source: Path) -> tuple[list[Path], list[Path]]:
    source_str = str(source)
    if any(ch in source_str for ch in "*?[]"):
        paths = [Path(p) for p in sorted(source.parent.glob(source.name))]
    elif source.is_dir():
        paths = sorted(p for p in source.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES | VIDEO_SUFFIXES)
    elif source.is_file() and source.suffix.lower() in IMAGE_SUFFIXES | VIDEO_SUFFIXES:
        paths = [source]
    else:
        raise FileNotFoundError(f"No images or videos found for source: {source}")

    images = [p for p in paths if p.suffix.lower() in IMAGE_SUFFIXES]
    videos = [p for p in paths if p.suffix.lower() in VIDEO_SUFFIXES]
    return images, videos


def load_manifest(path: Path, dataset: Path) -> dict[str, Path]:
    if not path.exists():
        return {}

    mapping: dict[str, Path] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_id = (row.get("raw_id") or "").strip()
            d2rgb = (row.get("D2RGB_png") or "").strip()
            if raw_id and d2rgb:
                mapping[raw_id] = dataset / d2rgb
    return mapping


def collect_stage1_records(source: Path) -> list[dict[str, Any]]:
    if source.is_dir():
        records: list[dict[str, Any]] = []
        for path in sorted(source.glob("*.json")):
            if path.name == "summary.json":
                continue
            record = read_json(path)
            record["_stage1_json"] = str(path)
            records.append(record)
        return records

    data = read_json(source)
    if isinstance(data, list):
        records = []
        for record in data:
            if isinstance(record, dict):
                record = dict(record)
                record["_stage1_json"] = str(source)
                records.append(record)
        return records
    if isinstance(data, dict):
        data["_stage1_json"] = str(source)
        return [data]
    raise ValueError(f"Unsupported stage1 JSON shape: {source}")


def raw_id_from_image(image_path: str | None) -> str | None:
    if not image_path:
        return None
    stem = Path(image_path).stem
    if stem.startswith("color_"):
        return stem.removeprefix("color_")
    return stem


def first_detection(record: dict[str, Any], key: str) -> dict[str, Any] | None:
    items = record.get(key)
    if not isinstance(items, list) or not items:
        return None
    item = items[0]
    return item if isinstance(item, dict) else None


def output_stem(record: dict[str, Any], index: int) -> str:
    image = record.get("image")
    if image:
        return Path(image).stem
    raw_id = raw_id_from_image(record.get("video"))
    return raw_id or f"record_{index:06d}"

