"""Depth ROI helpers for line-box inner plane base-X estimation."""

from __future__ import annotations

from typing import Any

import numpy as np


def clamp_roi(roi_xyxy: list[float] | tuple[float, ...], width: int, height: int) -> tuple[int, int, int, int]:
    if len(roi_xyxy) != 4:
        raise ValueError(f"ROI must have 4 values [x1,y1,x2,y2], got {len(roi_xyxy)}")
    x1, y1, x2, y2 = [int(round(float(value))) for value in roi_xyxy]
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    x1 = max(0, min(int(width), x1))
    x2 = max(0, min(int(width), x2))
    y1 = max(0, min(int(height), y1))
    y2 = max(0, min(int(height), y2))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"ROI is empty after clamping to image bounds: {[x1, y1, x2, y2]}")
    return x1, y1, x2, y2


def roi_valid_depth_pixels(
    depth_m: np.ndarray,
    roi_xyxy: tuple[int, int, int, int],
    min_depth: float,
    max_depth: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x1, y1, x2, y2 = roi_xyxy
    roi_depth = np.asarray(depth_m[y1:y2, x1:x2], dtype=np.float64)
    valid = np.isfinite(roi_depth) & (roi_depth >= float(min_depth)) & (roi_depth <= float(max_depth))
    ys_local, xs_local = np.nonzero(valid)
    if len(xs_local) == 0:
        return (
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
        )
    xs = xs_local.astype(np.float64) + float(x1)
    ys = ys_local.astype(np.float64) + float(y1)
    z = roi_depth[ys_local, xs_local].astype(np.float64)
    return xs, ys, z


def _histogram_edges(min_depth: float, max_depth: float, bin_size_m: float) -> np.ndarray:
    if bin_size_m <= 0.0:
        raise ValueError(f"hist-bin-size-m must be positive, got {bin_size_m}")
    start = np.floor(float(min_depth) / float(bin_size_m)) * float(bin_size_m)
    stop = np.ceil(float(max_depth) / float(bin_size_m)) * float(bin_size_m)
    if stop <= start:
        stop = start + float(bin_size_m)
    return np.arange(start, stop + float(bin_size_m) * 0.5, float(bin_size_m), dtype=np.float64)


def choose_farthest_stable_depth_peak(
    depth_values_m: np.ndarray,
    min_depth: float,
    max_depth: float,
    bin_size_m: float,
    min_peak_points: int,
    min_peak_fraction: float,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    values = np.asarray(depth_values_m, dtype=np.float64)
    values = values[np.isfinite(values) & (values >= float(min_depth)) & (values <= float(max_depth))]
    if len(values) == 0:
        return None, {"reason": "no_valid_roi_depth", "valid_depth_count": 0}

    edges = _histogram_edges(min_depth, max_depth, bin_size_m)
    counts, edges = np.histogram(values, bins=edges)
    if len(counts) == 0:
        return None, {"reason": "empty_depth_histogram", "valid_depth_count": int(len(values))}

    smooth = np.convolve(counts.astype(np.float64), np.asarray([1.0, 1.0, 1.0]) / 3.0, mode="same")
    threshold = max(int(min_peak_points), int(np.ceil(float(min_peak_fraction) * len(values))))
    peaks: list[dict[str, Any]] = []
    for index, count in enumerate(counts):
        left = smooth[index - 1] if index > 0 else -1.0
        right = smooth[index + 1] if index + 1 < len(smooth) else -1.0
        raw_left = counts[index - 1] if index > 0 else -1
        raw_right = counts[index + 1] if index + 1 < len(counts) else -1
        raw_peak = count >= raw_left and count >= raw_right and count > 0
        smooth_peak = smooth[index] >= left and smooth[index] >= right and smooth[index] > 0.0
        is_peak = raw_peak or smooth_peak
        support_start = max(0, index - 1)
        support_stop = min(len(counts), index + 2)
        support_count = int(np.sum(counts[support_start:support_stop]))
        if is_peak and support_count >= threshold:
            center = (float(edges[index]) + float(edges[index + 1])) * 0.5
            peaks.append(
                {
                    "bin_index": int(index),
                    "center_m": round(center, 8),
                    "bin_min_m": round(float(edges[index]), 8),
                    "bin_max_m": round(float(edges[index + 1]), 8),
                    "raw_peak": bool(raw_peak),
                    "smooth_peak": bool(smooth_peak),
                    "raw_count": int(count),
                    "smoothed_count": round(float(smooth[index]), 6),
                    "support_count_for_threshold": support_count,
                }
            )

    histogram = {
        "valid_depth_count": int(len(values)),
        "bin_size_m": round(float(bin_size_m), 8),
        "min_depth_m": round(float(min_depth), 8),
        "max_depth_m": round(float(max_depth), 8),
        "stable_peak_threshold_count": int(threshold),
        "bin_edges_m": [round(float(value), 8) for value in edges.tolist()],
        "counts": [int(value) for value in counts.tolist()],
        "smoothed_counts": [round(float(value), 6) for value in smooth.tolist()],
        "candidate_peaks": peaks,
    }
    if not peaks:
        histogram["reason"] = "no_stable_depth_peak"
        return None, histogram

    selected = max(
        peaks,
        key=lambda item: (1 if item.get("raw_peak") else 0, float(item["center_m"]), int(item["support_count_for_threshold"])),
    )
    histogram["selected_peak"] = selected
    return selected, histogram


def select_depth_band(
    xs: np.ndarray,
    ys: np.ndarray,
    z: np.ndarray,
    peak_center_m: float,
    bin_size_m: float,
    half_width_bins: float = 1.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    half_width_m = float(bin_size_m) * float(half_width_bins)
    z_values = np.asarray(z, dtype=np.float64)
    keep = np.isfinite(z_values) & (np.abs(z_values - float(peak_center_m)) <= half_width_m)
    return np.asarray(xs)[keep], np.asarray(ys)[keep], z_values[keep], keep


def depth_pixels_to_camera_points(xs: np.ndarray, ys: np.ndarray, z: np.ndarray, camera: dict[str, Any]) -> np.ndarray:
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    x = (xs - float(camera["cx"])) * z / float(camera["fx"])
    y = (ys - float(camera["cy"])) * z / float(camera["fy"])
    return np.column_stack([x, y, z]).astype(np.float64)


def transform_points(points: np.ndarray, transform_4x4: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    transform = np.asarray(transform_4x4, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must be Nx3, got {points.shape}")
    if transform.shape != (4, 4):
        raise ValueError(f"transform_4x4 must be 4x4, got {transform.shape}")
    ones = np.ones((len(points), 1), dtype=np.float64)
    homogeneous = np.hstack([points, ones])
    return (transform @ homogeneous.T).T[:, :3]


def robust_mad_keep(values: np.ndarray, min_tolerance: float = 0.005) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return np.zeros(0, dtype=bool)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if mad > 1e-12:
        tolerance = max(float(min_tolerance), 3.5 * 1.4826 * mad)
    else:
        q1, q3 = np.percentile(values, [25, 75])
        iqr = float(q3 - q1)
        tolerance = max(float(min_tolerance), 1.5 * iqr)
    return np.abs(values - median) <= tolerance


def robust_axis_stats(values: np.ndarray, min_tolerance: float = 0.005) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        raise ValueError("robust_axis_stats requires at least one value")
    keep = robust_mad_keep(values, min_tolerance)
    filtered = values[keep]
    if len(filtered) == 0:
        filtered = values
        keep = np.ones(len(values), dtype=bool)
    return {
        "count_raw": int(len(values)),
        "count_filtered": int(len(filtered)),
        "rejected_count": int(len(values) - len(filtered)),
        "min_m": round(float(np.min(filtered)), 8),
        "p01_m": round(float(np.percentile(filtered, 1)), 8),
        "p05_m": round(float(np.percentile(filtered, 5)), 8),
        "median_m": round(float(np.median(filtered)), 8),
        "p95_m": round(float(np.percentile(filtered, 95)), 8),
        "max_m": round(float(np.max(filtered)), 8),
        "std_m": round(float(np.std(filtered)), 8),
        "mad_filter_min_tolerance_m": round(float(min_tolerance), 8),
        "filter_keep_ratio": round(float(len(filtered) / max(1, len(values))), 6),
    }


__all__ = [
    "choose_farthest_stable_depth_peak",
    "clamp_roi",
    "depth_pixels_to_camera_points",
    "robust_axis_stats",
    "robust_mad_keep",
    "roi_valid_depth_pixels",
    "select_depth_band",
    "transform_points",
]
