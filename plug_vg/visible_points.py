"""Visible-object point extraction from a segmentation mask and D2RGB depth."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .geometry import polygon_mask


@dataclass(frozen=True)
class VisiblePointCloudResult:
    status: str
    visible_points_camera_m: np.ndarray
    mask: np.ndarray
    quality: dict[str, Any]
    warnings: list[str]
    reason: str | None = None


def _empty_result(reason: str, mask_shape: tuple[int, int], warnings: list[str], quality: dict[str, Any] | None = None) -> VisiblePointCloudResult:
    return VisiblePointCloudResult(
        status="failed",
        reason=reason,
        visible_points_camera_m=np.empty((0, 3), dtype=np.float64),
        mask=np.zeros(mask_shape, dtype=np.uint8),
        quality=quality or {},
        warnings=warnings,
    )


def _depth_to_camera_points(mask: np.ndarray, depth_raw: np.ndarray, camera: dict[str, Any], min_depth_m: float, max_depth_m: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    depth = np.asarray(depth_raw, dtype=np.float64) * float(camera["depth_scale"])
    ys, xs = np.nonzero(mask)
    z = depth[ys, xs]
    valid = np.isfinite(z) & (z >= float(min_depth_m)) & (z <= float(max_depth_m))
    xs = xs[valid].astype(np.float64)
    ys = ys[valid].astype(np.float64)
    z = z[valid].astype(np.float64)

    x = (xs - float(camera["cx"])) * z / float(camera["fx"])
    y = (ys - float(camera["cy"])) * z / float(camera["fy"])
    return np.column_stack([x, y, z]), np.column_stack([xs, ys]), z


def _robust_depth_keep(z: np.ndarray) -> np.ndarray:
    if len(z) == 0:
        return np.zeros(0, dtype=bool)
    median = float(np.median(z))
    mad = float(np.median(np.abs(z - median)))
    if mad > 1e-9:
        return np.abs(z - median) <= 3.5 * 1.4826 * mad
    q1, q3 = np.percentile(z, [25, 75])
    iqr = float(q3 - q1)
    if iqr > 1e-9:
        return (z >= q1 - 1.5 * iqr) & (z <= q3 + 1.5 * iqr)
    return np.ones(len(z), dtype=bool)


def voxel_downsample(points: np.ndarray, voxel_size_m: float) -> np.ndarray:
    """Deterministically keep one point per voxel."""

    points = np.asarray(points, dtype=np.float64)
    if len(points) == 0 or voxel_size_m <= 0.0:
        return points
    keys = np.floor(points / float(voxel_size_m)).astype(np.int64)
    _, keep_indices = np.unique(keys, axis=0, return_index=True)
    keep_indices.sort()
    return points[keep_indices]


def extract_visible_points_from_mask(
    image_bgr: np.ndarray,
    depth_raw: np.ndarray | None,
    camera: dict[str, Any],
    polygon_xy: list[list[float]] | None,
    min_depth_m: float,
    max_depth_m: float,
    voxel_size_m: float = 0.0,
    min_points: int = 200,
) -> VisiblePointCloudResult:
    """Extract visible object points in the RGB camera frame from a polygon mask."""

    warnings: list[str] = []
    if image_bgr is None or not hasattr(image_bgr, "shape") or len(image_bgr.shape) < 2:
        raise ValueError("image_bgr must be an image array")
    image_shape = (int(image_bgr.shape[0]), int(image_bgr.shape[1]))
    if depth_raw is None:
        return _empty_result("depth_unreadable", image_shape, warnings)
    if depth_raw.ndim == 3:
        warnings.append("depth_image_has_multiple_channels_using_first_channel")
        depth_raw = depth_raw[:, :, 0]
    if not polygon_xy:
        return _empty_result("segmentation_missing", depth_raw.shape[:2], warnings)

    expected_shape = (int(camera["image_height"]), int(camera["image_width"]))
    if depth_raw.shape[:2] != expected_shape:
        warnings.append(f"depth_shape_{depth_raw.shape[1]}x{depth_raw.shape[0]}_differs_from_camera_config")

    mask = polygon_mask(polygon_xy, depth_raw.shape[:2])
    mask_pixels = int(np.count_nonzero(mask))
    if mask_pixels == 0:
        return VisiblePointCloudResult(
            status="failed",
            reason="mask_empty",
            visible_points_camera_m=np.empty((0, 3), dtype=np.float64),
            mask=mask,
            quality={"mask_pixels": 0},
            warnings=warnings,
        )

    points, pixels, z = _depth_to_camera_points(mask, depth_raw, camera, min_depth_m, max_depth_m)
    raw_count = int(len(points))
    quality: dict[str, Any] = {
        "mask_pixels": mask_pixels,
        "visible_raw_points": raw_count,
        "min_depth_m": float(min_depth_m),
        "max_depth_m": float(max_depth_m),
        "voxel_size_m": float(voxel_size_m),
    }
    if raw_count == 0:
        return VisiblePointCloudResult(
            status="failed",
            reason="no_valid_depth_points",
            visible_points_camera_m=np.empty((0, 3), dtype=np.float64),
            mask=mask,
            quality={**quality, "visible_filtered_points": 0},
            warnings=warnings,
        )

    keep = _robust_depth_keep(z)
    filtered = points[keep]
    filtered_pixels = pixels[keep]
    filtered_z = z[keep]
    downsampled = voxel_downsample(filtered, voxel_size_m)
    quality.update(
        {
            "visible_filtered_points": int(len(filtered)),
            "visible_downsampled_points": int(len(downsampled)),
            "depth_min_m": round(float(np.min(filtered_z)), 8) if len(filtered_z) else None,
            "depth_median_m": round(float(np.median(filtered_z)), 8) if len(filtered_z) else None,
            "depth_max_m": round(float(np.max(filtered_z)), 8) if len(filtered_z) else None,
            "depth_rejected_points": int(raw_count - len(filtered)),
            "pixel_bbox_xyxy": [
                int(np.min(filtered_pixels[:, 0])) if len(filtered_pixels) else None,
                int(np.min(filtered_pixels[:, 1])) if len(filtered_pixels) else None,
                int(np.max(filtered_pixels[:, 0])) if len(filtered_pixels) else None,
                int(np.max(filtered_pixels[:, 1])) if len(filtered_pixels) else None,
            ],
        }
    )
    if len(downsampled) < int(min_points):
        return VisiblePointCloudResult(
            status="failed",
            reason="insufficient_visible_points",
            visible_points_camera_m=downsampled,
            mask=mask,
            quality={**quality, "min_visible_points": int(min_points)},
            warnings=warnings,
        )
    return VisiblePointCloudResult(
        status="ok",
        reason=None,
        visible_points_camera_m=downsampled,
        mask=mask,
        quality=quality,
        warnings=warnings,
    )


def save_visible_points_ply(points: np.ndarray, output_path: Path, max_points: int = 200000) -> None:
    points = np.asarray(points, dtype=np.float64)
    if len(points) > max_points:
        idx = np.linspace(0, len(points) - 1, num=max_points, dtype=np.int64)
        points = points[idx]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="ascii") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("end_header\n")
        for x, y, z in points:
            f.write(f"{float(x):.8f} {float(y):.8f} {float(z):.8f}\n")


def save_mask_overlay(image_bgr: np.ndarray, mask: np.ndarray, output_path: Path) -> None:
    import cv2

    canvas = image_bgr.copy()
    if mask.shape[:2] != canvas.shape[:2]:
        mask = cv2.resize(mask, (canvas.shape[1], canvas.shape[0]), interpolation=cv2.INTER_NEAREST)
    layer = canvas.copy()
    layer[mask > 0] = (40, 180, 60)
    canvas = cv2.addWeighted(layer, 0.38, canvas, 0.62, 0)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, contours, -1, (20, 220, 80), 2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)
