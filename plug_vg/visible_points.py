"""Visible-object point extraction from a segmentation mask and D2RGB depth."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .geometry import polygon_mask
from .point_preprocessing import PreprocessingError, bilateral_smooth_points, statistical_inlier_mask


@dataclass(frozen=True)
class VisiblePointCloudResult:
    status: str
    visible_points_camera_m: np.ndarray
    mask: np.ndarray
    quality: dict[str, Any]
    warnings: list[str]
    reason: str | None = None
    full_points_camera_m: np.ndarray | None = None
    cleaned_depth_m: np.ndarray | None = None


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


def global_mad_depth_keep(z: np.ndarray, threshold: float) -> tuple[np.ndarray, dict[str, float | None]]:
    """Return a global median-absolute-deviation inlier mask for camera Z."""

    z = np.asarray(z, dtype=np.float64)
    if len(z) == 0:
        return np.zeros(0, dtype=bool), {
            "median_m": None,
            "mad_m": None,
            "scaled_mad_m": None,
            "lower_m": None,
            "upper_m": None,
        }
    median = float(np.median(z))
    mad = float(np.median(np.abs(z - median)))
    scaled_mad = 1.4826 * mad
    if threshold <= 0.0 or scaled_mad <= 1e-12:
        keep = np.ones(len(z), dtype=bool)
        lower = None
        upper = None
    else:
        radius = float(threshold) * scaled_mad
        lower = median - radius
        upper = median + radius
        keep = np.abs(z - median) <= radius
    return keep, {
        "median_m": median,
        "mad_m": mad,
        "scaled_mad_m": scaled_mad,
        "lower_m": lower,
        "upper_m": upper,
    }


def voxel_downsample(points: np.ndarray, voxel_size_m: float) -> np.ndarray:
    """Deterministically replace each occupied voxel with its centroid."""

    points = np.asarray(points, dtype=np.float64)
    if len(points) == 0 or voxel_size_m <= 0.0:
        return points
    keys = np.floor(points / float(voxel_size_m)).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    counts = np.bincount(inverse)
    return np.column_stack(
        [np.bincount(inverse, weights=points[:, axis]) / counts for axis in range(3)]
    )


def erode_mask(mask: np.ndarray, erosion_px: int) -> np.ndarray:
    """Erode a binary mask inward by approximately ``erosion_px`` pixels."""

    mask = np.asarray(mask, dtype=np.uint8)
    erosion_px = int(erosion_px)
    if erosion_px <= 0 or not np.any(mask):
        return mask.copy()
    import cv2

    diameter = 2 * erosion_px + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))
    return cv2.erode(mask, kernel, iterations=1)


def synthesize_scene_point_cloud(
    image_bgr: np.ndarray,
    depth_raw: np.ndarray,
    camera: dict[str, Any],
    min_depth_m: float,
    max_depth_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Back-project valid D2RGB pixels into a colored RGB-camera-frame cloud."""

    if image_bgr is None or image_bgr.ndim < 3 or image_bgr.shape[2] < 3:
        raise ValueError("image_bgr must be a BGR image array")
    if depth_raw.ndim == 3:
        depth_raw = depth_raw[:, :, 0]
    if image_bgr.shape[:2] != depth_raw.shape[:2]:
        raise ValueError(
            "RGB and D2RGB shapes must match for scene point cloud synthesis, "
            f"got RGB {image_bgr.shape[1]}x{image_bgr.shape[0]} and "
            f"depth {depth_raw.shape[1]}x{depth_raw.shape[0]}"
        )

    mask = np.ones(depth_raw.shape[:2], dtype=np.uint8)
    points, pixels, _z = _depth_to_camera_points(mask, depth_raw, camera, min_depth_m, max_depth_m)
    if not len(points):
        return points, np.empty((0, 3), dtype=np.float64)
    xy = pixels.astype(np.int64)
    colors_bgr = image_bgr[xy[:, 1], xy[:, 0], :3]
    colors_rgb = colors_bgr[:, ::-1].astype(np.float64) / 255.0
    return points, colors_rgb


def extract_visible_points_from_mask(
    image_bgr: np.ndarray,
    depth_raw: np.ndarray | None,
    camera: dict[str, Any],
    polygon_xy: list[list[float]] | None,
    min_depth_m: float,
    max_depth_m: float,
    voxel_size_m: float = 0.0,
    min_points: int = 200,
    mask_erosion_px: int = 5,
    mad_z_threshold: float = 3.5,
    point_filter: str = "none",
    sor_neighbors: int = 20,
    sor_std_ratio: float = 6.0,
    bilateral_neighbors: int = 20,
    bilateral_sharpness_angle: float = 15.0,
) -> VisiblePointCloudResult:
    """Extract visible object points in the RGB camera frame from a polygon mask."""

    if point_filter not in {"none", "sor", "bilateral", "edge_clean", "depth_consistency"}:
        raise ValueError("point_filter must be none, sor, bilateral, edge_clean or depth_consistency")
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

    raw_mask = polygon_mask(polygon_xy, depth_raw.shape[:2])
    raw_mask_pixels = int(np.count_nonzero(raw_mask))
    if raw_mask_pixels == 0:
        return VisiblePointCloudResult(
            status="failed",
            reason="mask_empty",
            visible_points_camera_m=np.empty((0, 3), dtype=np.float64),
            mask=raw_mask,
            quality={"raw_mask_pixels": 0, "mask_pixels": 0},
            warnings=warnings,
        )

    if point_filter == "edge_clean":
        from .organized_preprocessing import clean_depth_cloud
        cleaned = clean_depth_cloud(np.asarray(depth_raw, dtype=float) * camera["depth_scale"],
                                    raw_mask, camera, min_depth_m, max_depth_m)
        downsampled = voxel_downsample(cleaned.points, voxel_size_m)
        from .registration_cleanup import refine_registration_cloud
        downsampled, registration_cleanup = refine_registration_cloud(downsampled, camera, raw_mask, voxel_size_m)
        quality = {
            "raw_mask_pixels": raw_mask_pixels, "mask_pixels": int(cleaned.valid_mask.sum()),
            "mask_erosion_px": 0, "mask_erosion_requested_px": int(mask_erosion_px),
            "depth_filter_strategy": "organized_support_radius_bilateral",
            "mad_z_threshold": 0., "voxel_size_m": float(voxel_size_m),
            "visible_raw_points": cleaned.quality["input_points"],
            "visible_filtered_points": len(cleaned.points),
            "visible_downsampled_points": len(downsampled),
            "point_filter": {"method": "edge_clean", **cleaned.quality},
            "registration_cleanup": registration_cleanup,
        }
        if len(cleaned.points):
            quality.update(depth_min_m=float(cleaned.points[:, 2].min()),
                           depth_median_m=float(np.median(cleaned.points[:, 2])),
                           depth_max_m=float(cleaned.points[:, 2].max()))
        quality.update(min_depth_m=float(min_depth_m), max_depth_m=float(max_depth_m),
                       min_visible_points=int(min_points),
                       invalid_or_out_of_range_depth_pixels=raw_mask_pixels-cleaned.quality["input_points"])
        ok = len(downsampled) >= min_points and len(downsampled) > 0
        return VisiblePointCloudResult(
            status="ok" if ok else "failed",
            reason=None if ok else ("no_valid_depth_points" if cleaned.quality["input_points"] == 0 else "insufficient_visible_points"),
            visible_points_camera_m=downsampled, mask=cleaned.valid_mask.astype(np.uint8),
            quality=quality, warnings=warnings,
            full_points_camera_m=cleaned.points, cleaned_depth_m=cleaned.depth_m,
        )

    mask = erode_mask(raw_mask, mask_erosion_px)
    mask_pixels = int(np.count_nonzero(mask))
    mask_quality = {
        "raw_mask_pixels": raw_mask_pixels,
        "mask_pixels": mask_pixels,
        "mask_erosion_px": int(mask_erosion_px),
        "mask_erosion_removed_pixels": int(raw_mask_pixels - mask_pixels),
    }
    if mask_pixels == 0:
        return VisiblePointCloudResult(
            status="failed",
            reason="mask_empty_after_erosion",
            visible_points_camera_m=np.empty((0, 3), dtype=np.float64),
            mask=mask,
            quality=mask_quality,
            warnings=warnings,
        )

    points, pixels, z = _depth_to_camera_points(mask, depth_raw, camera, min_depth_m, max_depth_m)
    raw_count = int(len(points))
    quality: dict[str, Any] = {
        **mask_quality,
        "visible_raw_points": raw_count,
        "min_depth_m": float(min_depth_m),
        "max_depth_m": float(max_depth_m),
        "voxel_size_m": float(voxel_size_m),
        "invalid_or_out_of_range_depth_pixels": int(mask_pixels - raw_count),
        "depth_filter_strategy": (
            "pass_through_only" if mad_z_threshold <= 0.0 else "pass_through_then_global_mad"
        ),
        "mad_z_threshold": float(mad_z_threshold),
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

    keep, mad_stats = global_mad_depth_keep(z, mad_z_threshold)
    depth_rejected_count = 0
    if point_filter == "depth_consistency":
        from .depth_consistency import depth_consistency_keep
        valid_depth = depth_consistency_keep(
            np.asarray(depth_raw, dtype=float) * camera["depth_scale"], mask)
        xy = pixels.astype(np.int64)
        depth_keep = valid_depth[xy[:, 1], xy[:, 0]]
        depth_rejected_count = int((~depth_keep).sum())
        mad_rejected_after_depth = int((depth_keep & ~keep).sum())
        keep &= depth_keep
        quality["depth_consistency"] = {
            "radius_px": 2, "depth_diff_threshold_m": .001,
            "bad_neighbor_ratio_threshold": .8,
            "neighbor_source": "original_depth_before_masking",
            "input_points": raw_count, "removed_points": depth_rejected_count,
            "output_points": raw_count-depth_rejected_count,
        }
        quality["depth_filter_strategy"] = "pass_through_then_local_depth_consistency"

    filtered_points = points[keep]
    filtered_pixels = pixels[keep]
    filtered_z = z[keep]
    if not len(filtered_points):
        return VisiblePointCloudResult(
            status="failed",
            reason=("no_points_after_depth_consistency" if point_filter == "depth_consistency" and depth_rejected_count == raw_count else "no_points_after_global_mad"),
            visible_points_camera_m=np.empty((0, 3), dtype=np.float64),
            mask=mask,
            quality={
                **quality,
                "visible_filtered_points": 0,
                "mad_rejected_points": (mad_rejected_after_depth if point_filter == "depth_consistency" else raw_count),
                "mad_statistics": mad_stats,
            },
            warnings=warnings,
        )

    mad_rejected_count = (mad_rejected_after_depth if point_filter == "depth_consistency" else raw_count - len(filtered_points))
    sor_input_count = len(filtered_points)
    smoothing_quality = {}
    if point_filter == "sor":
        sor_keep = statistical_inlier_mask(filtered_points, sor_neighbors, sor_std_ratio)
        filtered_points = filtered_points[sor_keep]
        filtered_pixels = filtered_pixels[sor_keep]
        filtered_z = filtered_z[sor_keep]
    elif point_filter == "bilateral":
        try:
            smoothed = bilateral_smooth_points(filtered_points, bilateral_neighbors, bilateral_sharpness_angle)
        except PreprocessingError as exc:
            return VisiblePointCloudResult(
                status="failed", reason="point_preprocessing_failed",
                visible_points_camera_m=np.empty((0, 3)), mask=mask,
                quality={**quality, "point_filter": {"method": point_filter, "error": str(exc)}},
                warnings=warnings,
            )
        movement = np.linalg.norm(smoothed - filtered_points, axis=1)
        smoothing_quality = {
            "neighbors": int(bilateral_neighbors), "sharpness_angle": float(bilateral_sharpness_angle),
            "iterations": 1, "neighbor_radius_m": 0.,
            "movement_mean_m": float(movement.mean()), "movement_max_m": float(movement.max()),
        }
        filtered_points = smoothed
        filtered_z = smoothed[:, 2]
    quality["point_filter"] = {
        "method": point_filter,
        "depth_consistency_removed_points": depth_rejected_count,
        "implementation": {"sor": "open3d.remove_statistical_outlier", "bilateral": "CGAL.bilateral_smooth_point_set", "depth_consistency": "FoundationPose-style depth consistency rejection (NumPy)"}.get(point_filter),
        "position": "after_depth_filter_before_voxelization",
        "neighbors": int(sor_neighbors) if point_filter == "sor" else None,
        "std_ratio": float(sor_std_ratio) if point_filter == "sor" else None,
        "input_points": sor_input_count,
        "output_points": len(filtered_points),
        "removed_points": sor_input_count - len(filtered_points),
        **smoothing_quality,
    }
    if not len(filtered_points):
        return VisiblePointCloudResult(
            status="failed", reason="no_points_after_statistical_filter",
            visible_points_camera_m=filtered_points, mask=mask,
            quality={**quality, "visible_filtered_points": 0,
                     "mad_rejected_points": mad_rejected_count, "mad_statistics": mad_stats},
            warnings=warnings,
        )
    downsampled = voxel_downsample(filtered_points, voxel_size_m)
    quality.update(
        {
            "visible_filtered_points": int(len(filtered_points)),
            "visible_downsampled_points": int(len(downsampled)),
            "mad_rejected_points": int(mad_rejected_count),
            "mad_statistics": mad_stats,
            "pass_through_depth_min_m": round(float(np.min(z)), 8),
            "pass_through_depth_median_m": round(float(np.median(z)), 8),
            "pass_through_depth_max_m": round(float(np.max(z)), 8),
            "depth_min_m": round(float(np.min(filtered_z)), 8),
            "depth_median_m": round(float(np.median(filtered_z)), 8),
            "depth_max_m": round(float(np.max(filtered_z)), 8),
            "pixel_bbox_xyxy": [
                int(np.min(filtered_pixels[:, 0])),
                int(np.min(filtered_pixels[:, 1])),
                int(np.max(filtered_pixels[:, 0])),
                int(np.max(filtered_pixels[:, 1])),
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
