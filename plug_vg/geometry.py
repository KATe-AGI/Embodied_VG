"""Geometry utilities for RGB-D plug grasp pose estimation."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from .config import GRASP_REGION_THICKNESS_M


MIDSECTION_AXIS_RANGE = (0.4, 0.6)
MIDSECTION_MIN_POINTS = 30
MASK_INTERIOR_DISTANCE_PX = 2.0
LOCAL_Z_MIN_TOLERANCE_M = 0.006
FUSED_ANCHOR_RADIUS_FRACTION = 0.18
FUSED_ANCHOR_MIN_RADIUS_PX = 8.0
FUSED_ANCHOR_AXIS_CONFIDENCE_PX = 5.0
FUSED_ANCHOR_MAX_AXIS_WEIGHT = 0.35


def polygon_mask(polygon_xy: list[list[float]], shape: tuple[int, int]) -> np.ndarray:
    import cv2

    mask = np.zeros(shape, dtype=np.uint8)
    if not polygon_xy:
        return mask
    pts = np.asarray(polygon_xy, dtype=np.float32)
    pts[:, 0] = np.clip(pts[:, 0], 0, shape[1] - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, shape[0] - 1)
    cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 1)
    return mask


def derive_grasp_target_2d(mask: np.ndarray, head_xy: list[float], tail_xy: list[float]) -> tuple[list[float] | None, str | None]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None, "missing_mask_pixels"
    points = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
    head = np.asarray(head_xy, dtype=np.float64)
    tail = np.asarray(tail_xy, dtype=np.float64)
    axis = head - tail
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-6:
        target = np.median(points, axis=0)
        return [round(float(target[0]), 3), round(float(target[1]), 3)], "mask_median_fallback"
    axis = axis / axis_norm
    projected = (points - tail) @ axis
    lo, hi = np.percentile(projected, [40, 60])
    candidates = points[(projected >= lo) & (projected <= hi)]
    method = "axis_mid_mask_median"
    if len(candidates) < 10:
        candidates = points
        method = "mask_median_fallback"
    target = np.median(candidates, axis=0)
    return [round(float(target[0]), 3), round(float(target[1]), 3)], method


def depth_to_points(mask: np.ndarray, depth_raw: np.ndarray, camera: dict[str, Any], min_depth: float, max_depth: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    depth = depth_raw.astype(np.float32) * float(camera["depth_scale"])
    ys, xs = np.nonzero(mask)
    z = depth[ys, xs]
    valid = np.isfinite(z) & (z >= min_depth) & (z <= max_depth)
    xs = xs[valid].astype(np.float32)
    ys = ys[valid].astype(np.float32)
    z = z[valid]

    x = (xs - float(camera["cx"])) * z / float(camera["fx"])
    y = (ys - float(camera["cy"])) * z / float(camera["fy"])
    points = np.column_stack([x, y, z]).astype(np.float32)
    pixels = np.column_stack([xs, ys]).astype(np.float32)
    return points, pixels, z.astype(np.float32)


def robust_depth_filter(points: np.ndarray, pixels: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    if len(points) == 0:
        return points, pixels, {"raw_valid_points": 0}

    median = float(np.median(z))
    mad = float(np.median(np.abs(z - median)))
    if mad > 1e-6:
        keep = np.abs(z - median) <= 3.5 * 1.4826 * mad
    else:
        q1, q3 = np.percentile(z, [25, 75])
        iqr = float(q3 - q1)
        if iqr > 1e-6:
            keep = (z >= q1 - 1.5 * iqr) & (z <= q3 + 1.5 * iqr)
        else:
            keep = np.ones_like(z, dtype=bool)

    filtered = points[keep]
    filtered_pixels = pixels[keep]
    stats = {
        "raw_valid_points": int(len(points)),
        "filtered_points": int(len(filtered)),
        "depth_median_m": round(median, 6),
        "depth_min_m": round(float(np.min(z)), 6),
        "depth_max_m": round(float(np.max(z)), 6),
    }
    return filtered, filtered_pixels, stats


def normalize(vector: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm < 1e-9:
        return None
    return vector / norm


def pca_axes(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centered = points - np.mean(points, axis=0)
    cov = np.cov(centered.T)
    values, vectors = np.linalg.eigh(cov)
    order = np.argsort(values)[::-1]
    return values[order], vectors[:, order[0]], vectors[:, order[-1]]


def fit_plane_normal(points: np.ndarray, threshold: float, iterations: int, warnings: list[str]) -> tuple[np.ndarray | None, int]:
    if len(points) < 3:
        return None, 0

    rng = np.random.default_rng(0)
    sample_count = min(len(points), 10000)
    sample_idx = rng.choice(len(points), size=sample_count, replace=False)
    sample = points[sample_idx]

    best_normal: np.ndarray | None = None
    best_inliers: np.ndarray | None = None
    best_count = 0
    for _ in range(max(1, iterations)):
        ids = rng.choice(sample_count, size=3, replace=False)
        p0, p1, p2 = sample[ids]
        normal = normalize(np.cross(p1 - p0, p2 - p0))
        if normal is None:
            continue
        dist = np.abs((sample - p0) @ normal)
        inliers = dist < threshold
        count = int(np.count_nonzero(inliers))
        if count > best_count:
            best_normal = normal
            best_inliers = inliers
            best_count = count

    if best_normal is None or best_inliers is None or best_count < 3:
        warnings.append("plane_ransac_failed_using_pca_normal")
        _, _, normal = pca_axes(points)
        return normalize(normal), len(points)

    inlier_points = sample[best_inliers]
    _, _, normal = pca_axes(inlier_points)
    return normalize(normal), best_count


def keypoint_3d(
    xy: list[float],
    depth_raw: np.ndarray,
    camera: dict[str, Any],
    window_size: int,
    min_depth: float,
    max_depth: float,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    h, w = depth_raw.shape[:2]
    u, v = float(xy[0]), float(xy[1])
    radius = max(1, int(window_size) // 2)
    x0 = max(0, int(round(u)) - radius)
    x1 = min(w, int(round(u)) + radius + 1)
    y0 = max(0, int(round(v)) - radius)
    y1 = min(h, int(round(v)) + radius + 1)
    patch = depth_raw[y0:y1, x0:x1].astype(np.float32) * float(camera["depth_scale"])
    valid = np.isfinite(patch) & (patch >= min_depth) & (patch <= max_depth)
    valid_count = int(np.count_nonzero(valid))
    total = int(patch.size)
    info = {"xy": [round(u, 3), round(v, 3)], "valid_depth_pixels": valid_count, "window_pixels": total}
    if valid_count == 0:
        return None, info

    z = float(np.median(patch[valid]))
    x = (u - float(camera["cx"])) * z / float(camera["fx"])
    y = (v - float(camera["cy"])) * z / float(camera["fy"])
    info["depth_m"] = round(z, 6)
    return np.asarray([x, y, z], dtype=np.float32), info


def build_rotation(
    points: np.ndarray,
    pixels: np.ndarray,
    center: np.ndarray,
    head_3d: np.ndarray | None,
    tail_3d: np.ndarray | None,
    head_xy: list[float],
    tail_xy: list[float],
    plane_threshold: float,
    ransac_iters: int,
    warnings: list[str],
) -> tuple[np.ndarray | None, dict[str, Any]]:
    values, pca_main, pca_normal = pca_axes(points)
    z_axis, plane_inliers = fit_plane_normal(points, plane_threshold, ransac_iters, warnings)
    if z_axis is None:
        z_axis = normalize(pca_normal)
    if z_axis is None:
        return None, {"plane_inliers": plane_inliers}

    approach_side = center
    if float(np.dot(z_axis, approach_side)) < 0:
        z_axis = -z_axis

    x_source = "keypoints_3d"
    if head_3d is not None and tail_3d is not None:
        x_axis = normalize(head_3d - tail_3d)
    else:
        x_source = "mask_pca"
        x_axis = normalize(pca_main)
        if x_axis is not None:
            direction_2d = np.asarray([head_xy[0] - tail_xy[0], head_xy[1] - tail_xy[1]], dtype=np.float32)
            if float(np.dot(x_axis[:2], direction_2d)) < 0:
                x_axis = -x_axis
            warnings.append("keypoint_depth_invalid_using_mask_pca_axis")
    if x_axis is None:
        return None, {"plane_inliers": plane_inliers, "x_source": x_source}

    x_axis = x_axis - float(np.dot(x_axis, z_axis)) * z_axis
    x_axis = normalize(x_axis)
    if x_axis is None:
        x_axis = normalize(pca_main - float(np.dot(pca_main, z_axis)) * z_axis)
    if x_axis is None:
        return None, {"plane_inliers": plane_inliers, "x_source": x_source}

    y_axis = normalize(np.cross(z_axis, x_axis))
    if y_axis is None:
        return None, {"plane_inliers": plane_inliers, "x_source": x_source}
    z_axis = normalize(np.cross(x_axis, y_axis))
    if z_axis is None:
        return None, {"plane_inliers": plane_inliers, "x_source": x_source}

    rotation = np.column_stack([x_axis, y_axis, z_axis]).astype(np.float32)
    return rotation, {
        "x_source": x_source,
        "plane_inliers": int(plane_inliers),
        "pca_eigenvalues": [round(float(v), 10) for v in values.tolist()],
    }


def rotation_to_quaternion_xyzw(rotation: np.ndarray) -> list[float]:
    m = rotation.astype(np.float64)
    trace = float(np.trace(m))
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        qw = (m[2, 1] - m[1, 2]) / s
        qx = 0.25 * s
        qy = (m[0, 1] + m[1, 0]) / s
        qz = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        qw = (m[0, 2] - m[2, 0]) / s
        qx = (m[0, 1] + m[1, 0]) / s
        qy = 0.25 * s
        qz = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        qw = (m[1, 0] - m[0, 1]) / s
        qx = (m[0, 2] + m[2, 0]) / s
        qy = (m[1, 2] + m[2, 1]) / s
        qz = 0.25 * s
    quat = np.asarray([qx, qy, qz, qw], dtype=np.float64)
    quat /= np.linalg.norm(quat)
    return [round(float(v), 8) for v in quat.tolist()]


def project_point(point: np.ndarray, camera: dict[str, Any]) -> tuple[int, int] | None:
    if point[2] <= 1e-6:
        return None
    u = float(camera["fx"]) * point[0] / point[2] + float(camera["cx"])
    v = float(camera["fy"]) * point[1] / point[2] + float(camera["cy"])
    if not np.isfinite(u) or not np.isfinite(v):
        return None
    return int(round(u)), int(round(v))


def project_point_float(point: np.ndarray, camera: dict[str, Any]) -> np.ndarray | None:
    if point[2] <= 1e-6:
        return None
    u = float(camera["fx"]) * point[0] / point[2] + float(camera["cx"])
    v = float(camera["fy"]) * point[1] / point[2] + float(camera["cy"])
    if not np.isfinite(u) or not np.isfinite(v):
        return None
    return np.asarray([u, v], dtype=np.float64)


def robust_extent(points: np.ndarray, center: np.ndarray, rotation: np.ndarray) -> dict[str, float]:
    local = (points - center) @ rotation
    lo = np.percentile(local, 5, axis=0)
    hi = np.percentile(local, 95, axis=0)
    extents = hi - lo
    return {
        "length_x_m": round(float(extents[0]), 6),
        "width_y_m": round(float(extents[1]), 6),
        "thickness_z_m": round(float(extents[2]), 6),
    }


def _round_vector(vector: np.ndarray, digits: int = 8) -> list[float]:
    return [round(float(v), digits) for v in vector.tolist()]


def _robust_scalar_keep(values: np.ndarray, min_tolerance: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return np.zeros(0, dtype=bool)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if mad > 1e-9:
        tolerance = max(float(min_tolerance), 3.5 * 1.4826 * mad)
        return np.abs(values - median) <= tolerance

    q1, q3 = np.percentile(values, [25, 75])
    iqr = float(q3 - q1)
    if iqr > 1e-9:
        tolerance = max(float(min_tolerance), 1.5 * iqr)
        return (values >= q1 - tolerance) & (values <= q3 + tolerance)
    return np.abs(values - median) <= float(min_tolerance)


def _interior_pixel_keep(mask: np.ndarray, pixels: np.ndarray, min_distance_px: float) -> np.ndarray:
    import cv2

    h, w = mask.shape[:2]
    xy = np.round(pixels).astype(np.int32)
    xs = np.clip(xy[:, 0], 0, w - 1)
    ys = np.clip(xy[:, 1], 0, h - 1)
    margin = max(3, int(math.ceil(float(min_distance_px))) + 2)
    x0 = max(0, int(np.min(xs)) - margin)
    x1 = min(w, int(np.max(xs)) + margin + 1)
    y0 = max(0, int(np.min(ys)) - margin)
    y1 = min(h, int(np.max(ys)) + margin + 1)
    cropped = (mask[y0:y1, x0:x1] > 0).astype(np.uint8)
    distance = cv2.distanceTransform(cropped, cv2.DIST_L2, 3)
    return distance[ys - y0, xs - x0] >= float(min_distance_px)


def _stable_axis_from_mask_pixels(pixels: np.ndarray, head_xy: list[float], tail_xy: list[float]) -> tuple[np.ndarray, str]:
    centered = pixels - np.mean(pixels, axis=0)
    if len(centered) >= 3:
        cov = np.cov(centered.T)
        values, vectors = np.linalg.eigh(cov)
        order = np.argsort(values)[::-1]
        if float(values[order[0]]) > 1e-9:
            axis = vectors[:, order[0]]
            source = "mask_pca"
        else:
            axis = np.asarray([1.0, 0.0], dtype=np.float64)
            source = "mask_bbox_fallback"
    else:
        axis = np.asarray([1.0, 0.0], dtype=np.float64)
        source = "mask_bbox_fallback"

    span = np.ptp(pixels, axis=0)
    if source == "mask_bbox_fallback":
        axis = np.asarray([1.0, 0.0], dtype=np.float64) if span[0] >= span[1] else np.asarray([0.0, 1.0], dtype=np.float64)

    keypoint_axis = np.asarray(head_xy, dtype=np.float64) - np.asarray(tail_xy, dtype=np.float64)
    if float(np.linalg.norm(keypoint_axis)) > 1e-6 and float(np.dot(axis, keypoint_axis)) < 0.0:
        axis = -axis
    norm = float(np.linalg.norm(axis))
    if norm < 1e-9:
        return np.asarray([1.0, 0.0], dtype=np.float64), "mask_bbox_fallback"
    return axis / norm, source


def _fused_anchor_keep(
    pixels: np.ndarray,
    candidate: np.ndarray,
    head_xy: list[float],
    tail_xy: list[float],
    min_points: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    candidate_pixels = pixels[candidate]
    if len(candidate_pixels) == 0:
        return np.zeros(len(pixels), dtype=bool), {"source": "empty_candidate"}

    mask_center = np.median(candidate_pixels, axis=0)
    axis_unit, axis_source = _stable_axis_from_mask_pixels(candidate_pixels, head_xy, tail_xy)
    tail = np.asarray(tail_xy, dtype=np.float64)
    axis_point = tail + axis_unit * float(np.dot(mask_center - tail, axis_unit))
    axis_distance = float(np.linalg.norm(mask_center - axis_point))
    axis_weight = FUSED_ANCHOR_MAX_AXIS_WEIGHT / (1.0 + (axis_distance / FUSED_ANCHOR_AXIS_CONFIDENCE_PX) ** 2)
    fused_anchor = mask_center * (1.0 - axis_weight) + axis_point * axis_weight

    projected = (candidate_pixels - mask_center) @ axis_unit
    p05, p95 = np.percentile(projected, [5, 95])
    body_length_px = max(1.0, float(p95 - p05))
    radius_px = max(FUSED_ANCHOR_MIN_RADIUS_PX, body_length_px * FUSED_ANCHOR_RADIUS_FRACTION)
    distances = np.linalg.norm(candidate_pixels - fused_anchor, axis=1)
    local_keep = distances <= radius_px
    if int(np.count_nonzero(local_keep)) < min_points:
        radius_px = max(radius_px, float(np.percentile(distances, min(100.0, 100.0 * min_points / max(1, len(distances))))))
        local_keep = distances <= radius_px

    keep = np.zeros(len(pixels), dtype=bool)
    keep[np.flatnonzero(candidate)] = local_keep
    info = {
        "source": "mask_axis_fused_anchor",
        "midsection_axis_source": axis_source,
        "midsection_axis_2d": _round_vector(axis_unit),
        "mask_center_2d": _round_vector(mask_center, digits=3),
        "axis_projected_center_2d": _round_vector(axis_point, digits=3),
        "fused_anchor_2d": _round_vector(fused_anchor, digits=3),
        "axis_center_distance_px": round(axis_distance, 4),
        "axis_fusion_weight": round(float(axis_weight), 6),
        "local_radius_px": round(float(radius_px), 4),
        "keypoints_used_for_midsection": False,
    }
    return keep, info


def _center_from_indices(
    points: np.ndarray,
    indices: np.ndarray,
    origin: np.ndarray,
    rotation: np.ndarray,
    offset_m: float,
    source: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    selected = points[indices]
    local = (selected - origin) @ rotation
    x_lo, x_hi = np.percentile(local[:, 0], [10, 90])
    y_lo, y_hi = np.percentile(local[:, 1], [10, 90])
    surface_local = np.asarray(
        [
            (float(x_lo) + float(x_hi)) * 0.5,
            (float(y_lo) + float(y_hi)) * 0.5,
            float(np.median(local[:, 2])),
        ],
        dtype=np.float64,
    )
    center_local = surface_local.copy()
    center_local[2] += float(offset_m)
    surface_center = origin + rotation @ surface_local
    center = origin + rotation @ center_local
    info = {
        "source": source,
        "surface_center_camera_m": _round_vector(surface_center),
        "local_center_percentiles": {
            "x_p10_m": round(float(x_lo), 8),
            "x_p90_m": round(float(x_hi), 8),
            "y_p10_m": round(float(y_lo), 8),
            "y_p90_m": round(float(y_hi), 8),
            "surface_z_median_m": round(float(surface_local[2]), 8),
        },
    }
    return center.astype(np.float32), info


def robust_midsection_center(
    points: np.ndarray,
    pixels: np.ndarray,
    mask: np.ndarray,
    head_xy: list[float],
    tail_xy: list[float],
    rotation: np.ndarray,
    warnings: list[str],
    axis_range: tuple[float, float] = MIDSECTION_AXIS_RANGE,
    min_points: int = MIDSECTION_MIN_POINTS,
    interior_distance_px: float = MASK_INTERIOR_DISTANCE_PX,
    axis_offset_m: float = 0.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Estimate a plug midsection center from robust in-mask midsection points."""

    points = np.asarray(points, dtype=np.float64)
    pixels = np.asarray(pixels, dtype=np.float64)
    rotation = np.asarray(rotation, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or pixels.shape != (len(points), 2):
        raise ValueError("points must be Nx3 and pixels must be Nx2 with matching length")
    if rotation.shape != (3, 3):
        raise ValueError(f"rotation must be 3x3, got {rotation.shape}")
    if len(points) < 3:
        raise ValueError("robust_midsection_center requires at least 3 points")

    origin = np.median(points, axis=0)
    offset_m = float(GRASP_REGION_THICKNESS_M) * 0.5
    rejected = {
        "mask_boundary": 0,
        "fused_anchor_region": 0,
        "local_z_outlier": 0,
        "fallback_local_z_outlier": 0,
    }
    anchor_info: dict[str, Any] = {}

    candidate = np.ones(len(points), dtype=bool)
    if mask.size and interior_distance_px > 0:
        interior_keep = _interior_pixel_keep(mask, pixels, interior_distance_px)
        if int(np.count_nonzero(candidate & interior_keep)) >= min_points:
            rejected["mask_boundary"] = int(np.count_nonzero(candidate & ~interior_keep))
            candidate &= interior_keep

    anchor_keep, anchor_info = _fused_anchor_keep(pixels, candidate, head_xy, tail_xy, min_points)
    if int(np.count_nonzero(candidate & anchor_keep)) >= min_points:
        rejected["fused_anchor_region"] = int(np.count_nonzero(candidate & ~anchor_keep))
        candidate &= anchor_keep
    else:
        warnings.append("robust_midsection_center_fused_anchor_fallback_interior_mask")

    midsection_candidate_count = int(np.count_nonzero(candidate))
    local_all = (points - origin) @ rotation
    z_keep = np.zeros(len(points), dtype=bool)
    if midsection_candidate_count:
        candidate_indices = np.flatnonzero(candidate)
        local_z_keep = _robust_scalar_keep(local_all[candidate_indices, 2], LOCAL_Z_MIN_TOLERANCE_M)
        z_keep[candidate_indices] = local_z_keep
        rejected["local_z_outlier"] = int(midsection_candidate_count - np.count_nonzero(local_z_keep))
    final_indices = np.flatnonzero(candidate & z_keep)

    used_fallback = False
    if len(final_indices) < min_points:
        used_fallback = True
        fallback_keep = _robust_scalar_keep(local_all[:, 2], LOCAL_Z_MIN_TOLERANCE_M)
        rejected["fallback_local_z_outlier"] = int(len(points) - np.count_nonzero(fallback_keep))
        final_indices = np.flatnonzero(fallback_keep)
        warnings.append("robust_midsection_center_fallback_full_mask")

    if len(final_indices) < min_points:
        final_indices = np.arange(len(points))
        warnings.append("robust_midsection_center_fallback_all_points")

    auto_center, center_info = _center_from_indices(
        points,
        final_indices,
        origin,
        rotation,
        offset_m,
        "fallback_full_mask" if used_fallback else "midsection",
    )
    axis_offset = float(axis_offset_m)
    axis_offset_vector = rotation[:, 0] * axis_offset
    center = auto_center.astype(np.float64) + axis_offset_vector
    info = {
        "mode": "robust_midsection_center",
        "axis_range": [round(float(axis_range[0]), 4), round(float(axis_range[1]), 4)],
        "mask_interior_distance_px": round(float(interior_distance_px), 4),
        "min_points": int(min_points),
        "midsection_candidate_count": midsection_candidate_count,
        "filtered_count": int(len(final_indices)),
        "rejected_reason_counts": rejected,
        "center_offset_axis": "+z_approach",
        "center_offset_m": round(float(offset_m), 8),
        "thickness_m": round(float(GRASP_REGION_THICKNESS_M), 8),
        "auto_grasp_center_camera_m": _round_vector(auto_center.astype(np.float64)),
        "axis_offset_m": round(axis_offset, 8),
        "axis_offset_direction": "tail_to_head",
        "axis_offset_vector_camera_m": _round_vector(axis_offset_vector),
        "final_grasp_center_camera_m": _round_vector(center),
        **anchor_info,
        **center_info,
    }
    return center.astype(np.float32), info


def draw_overlay(
    record: dict[str, Any],
    mask: np.ndarray,
    center: np.ndarray,
    rotation: np.ndarray,
    camera: dict[str, Any],
    output_path: Path,
    axis_scale: float,
    axis_thickness: int,
) -> None:
    import cv2

    image_path = record.get("image")
    image = cv2.imread(str(image_path)) if image_path else None
    if image is None:
        image = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    if image.shape[:2] != mask.shape:
        image = cv2.resize(image, (mask.shape[1], mask.shape[0]))

    layer = image.copy()
    layer[mask > 0] = (40, 180, 60)
    canvas = cv2.addWeighted(layer, 0.28, image, 0.72, 0)

    center_px = project_point(center, camera)
    if center_px is not None:
        cv2.circle(canvas, center_px, 6, (255, 255, 255), -1)
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
        labels = ["+X", "+Y", "+Z"]
        for axis, color, label in zip(rotation.T, colors, labels):
            end_px = project_point(center + axis * axis_scale, camera)
            if end_px is not None:
                cv2.arrowedLine(canvas, center_px, end_px, color, max(1, int(axis_thickness)), tipLength=0.22)
                cv2.putText(canvas, label, (end_px[0] + 6, end_px[1] + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)


def save_ply(points: np.ndarray, output_path: Path, max_points: int = 20000) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if len(points) > max_points:
        rng = np.random.default_rng(0)
        points = points[rng.choice(len(points), size=max_points, replace=False)]
    with output_path.open("w", encoding="ascii") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("end_header\n")
        for point in points:
            f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")
