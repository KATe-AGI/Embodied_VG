"""Shared file discovery and JSON helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


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


def read_depth_raw(path: Path) -> np.ndarray | None:
    """Read a D2RGB depth image from PNG or NPY format.

    Returns the raw sensor values as a numpy array (typically uint16), or None
    if the file cannot be read.
    """
    suffix = path.suffix.lower()
    if suffix == ".npy":
        try:
            data = np.load(str(path))
        except (ValueError, OSError):
            return None
        if not isinstance(data, np.ndarray) or data.ndim < 2:
            return None
        return data
    # Default: treat as a standard image (PNG, TIFF, etc.)
    import cv2

    return cv2.imread(str(path), cv2.IMREAD_UNCHANGED)


