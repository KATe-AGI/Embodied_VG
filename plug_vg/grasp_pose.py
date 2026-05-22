"""Estimate plug grasp 6D poses from stage1 detections and D2RGB depth."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import (
    GRASP_REGION_LENGTH_M,
    GRASP_REGION_THICKNESS_M,
    GRASP_REGION_WIDTH_M,
    HEAD_TAIL_DISTANCE_M,
)
from .geometry import (
    build_rotation,
    depth_to_points,
    derive_grasp_target_2d,
    draw_overlay,
    grasp_center_from_surface,
    keypoint_3d,
    polygon_mask,
    robust_depth_filter,
    robust_extent,
    rotation_to_quaternion_xyzw,
    save_ply,
)
from .io import first_detection, raw_id_from_image


def resolve_d2rgb_path(record: dict[str, Any], manifest: dict[str, Path], dataset: Path) -> tuple[Path | None, str | None]:
    raw_id = raw_id_from_image(record.get("image") or record.get("video"))
    if not raw_id:
        return None, None

    candidates: list[Path] = []
    if raw_id in manifest:
        candidates.append(manifest[raw_id])
    candidates.append(dataset / "D2RGB" / f"D2RGB_{raw_id}.png")

    for candidate in candidates:
        if candidate.exists():
            return candidate, raw_id
    return candidates[0], raw_id


def make_failure(record: dict[str, Any], raw_id: str | None, d2rgb_path: Path | None, reason: str, warnings: list[str]) -> dict[str, Any]:
    return {
        "status": "failed",
        "reason": reason,
        "warnings": warnings,
        "input": {
            "stage1_json": record.get("_stage1_json"),
            "image": record.get("image"),
            "raw_id": raw_id,
            "d2rgb": None if d2rgb_path is None else str(d2rgb_path),
        },
    }


def quality_score(head_tail_error: float | None, length_error: float, width_error: float, valid_points: int, warnings: list[str]) -> float:
    point_term = min(1.0, valid_points / 5000.0)
    ht_term = 0.65 if head_tail_error is None else max(0.0, 1.0 - min(head_tail_error, 1.0) * 0.7)
    length_term = max(0.0, 1.0 - min(length_error, 1.0) * 0.35)
    width_term = max(0.0, 1.0 - min(width_error, 1.0) * 0.35)
    warning_term = max(0.5, 1.0 - 0.05 * len(warnings))
    return round(float(point_term * ht_term * length_term * width_term * warning_term), 4)


def estimate_record(
    record: dict[str, Any],
    camera: dict[str, Any],
    manifest: dict[str, Path],
    args,
) -> tuple[dict[str, Any], np.ndarray | None, np.ndarray | None, np.ndarray | None, list[float] | None, list[float] | None]:
    warnings: list[str] = []
    d2rgb_path, raw_id = resolve_d2rgb_path(record, manifest, args.dataset)
    if d2rgb_path is None or not d2rgb_path.exists():
        return make_failure(record, raw_id, d2rgb_path, "d2rgb_depth_missing", warnings), None, None, None, None, None

    seg = first_detection(record, "segmentation")
    pose = first_detection(record, "pose")
    if seg is None:
        return make_failure(record, raw_id, d2rgb_path, "segmentation_missing", warnings), None, None, None, None, None
    if pose is None:
        return make_failure(record, raw_id, d2rgb_path, "pose_missing", warnings), None, None, None, None, None

    keypoints = pose.get("keypoints") or {}
    head = keypoints.get("plug_head") or {}
    tail = keypoints.get("plug_tail") or {}
    head_xy = head.get("xy")
    tail_xy = tail.get("xy")
    if not head_xy or not tail_xy:
        return make_failure(record, raw_id, d2rgb_path, "head_tail_keypoints_missing", warnings), None, None, None, None, None

    depth_raw = cv2.imread(str(d2rgb_path), cv2.IMREAD_UNCHANGED)
    if depth_raw is None:
        return make_failure(record, raw_id, d2rgb_path, "d2rgb_depth_unreadable", warnings), None, None, None, None, None
    if depth_raw.ndim == 3:
        warnings.append("depth_image_has_multiple_channels_using_first_channel")
        depth_raw = depth_raw[:, :, 0]

    expected_shape = (int(camera["image_height"]), int(camera["image_width"]))
    if depth_raw.shape[:2] != expected_shape:
        warnings.append(f"depth_shape_{depth_raw.shape[1]}x{depth_raw.shape[0]}_differs_from_camera_config")

    polygon = seg.get("polygon_xy") or []
    mask = polygon_mask(polygon, depth_raw.shape[:2])
    if int(np.count_nonzero(mask)) == 0:
        return make_failure(record, raw_id, d2rgb_path, "segmentation_mask_empty", warnings), None, None, None, head_xy, tail_xy

    points, pixels, z = depth_to_points(mask, depth_raw, camera, args.min_depth, args.max_depth)
    points, pixels, depth_stats = robust_depth_filter(points, pixels, z)
    if len(points) < args.min_points:
        result = make_failure(record, raw_id, d2rgb_path, "insufficient_mask_depth_points", warnings)
        result["quality"] = depth_stats
        return result, mask, None, None, head_xy, tail_xy

    center = np.median(points, axis=0)
    mean = np.mean(points, axis=0)

    grasp_target_2d, grasp_target_method = derive_grasp_target_2d(mask, head_xy, tail_xy)
    grasp_target_3d = None
    grasp_target_info: dict[str, Any] = {"method": grasp_target_method, "xy": grasp_target_2d}
    if grasp_target_2d is not None:
        grasp_target_3d, grasp_target_info = keypoint_3d(
            grasp_target_2d, depth_raw, camera, args.keypoint_window, args.min_depth, args.max_depth
        )
        grasp_target_info["method"] = grasp_target_method
    if grasp_target_3d is None:
        grasp_target_3d = center
        grasp_target_info["fallback"] = "mask_point_median"
        warnings.append("grasp_target_depth_missing_using_mask_center")

    head_3d, head_info = keypoint_3d(head_xy, depth_raw, camera, args.keypoint_window, args.min_depth, args.max_depth)
    tail_3d, tail_info = keypoint_3d(tail_xy, depth_raw, camera, args.keypoint_window, args.min_depth, args.max_depth)
    head_tail_distance = None
    head_tail_error = None
    if head_3d is None or tail_3d is None:
        warnings.append("head_or_tail_depth_missing")
    else:
        head_tail_distance = float(np.linalg.norm(head_3d - tail_3d))
        head_tail_error = abs(head_tail_distance - HEAD_TAIL_DISTANCE_M) / HEAD_TAIL_DISTANCE_M
        if head_tail_error > args.head_tail_tolerance:
            warnings.append(
                f"head_tail_distance_out_of_tolerance_{head_tail_distance:.4f}m_expected_{HEAD_TAIL_DISTANCE_M:.4f}m"
            )

    rotation, rotation_info = build_rotation(
        points,
        pixels,
        center,
        head_3d,
        tail_3d,
        head_xy,
        tail_xy,
        args.plane_threshold,
        args.ransac_iters,
        warnings,
    )
    if rotation is None:
        return make_failure(record, raw_id, d2rgb_path, "rotation_estimation_failed", warnings), mask, points, None, head_xy, tail_xy

    grasp_center_3d, grasp_center_info = grasp_center_from_surface(grasp_target_3d, rotation)

    extents = robust_extent(points, center, rotation)
    length_error = abs(extents["length_x_m"] - GRASP_REGION_LENGTH_M) / GRASP_REGION_LENGTH_M
    width_error = abs(extents["width_y_m"] - GRASP_REGION_WIDTH_M) / GRASP_REGION_WIDTH_M
    if length_error > 0.5:
        warnings.append(f"grasp_length_out_of_range_{extents['length_x_m']:.4f}m_expected_{GRASP_REGION_LENGTH_M:.4f}m")
    if width_error > 0.5:
        warnings.append(f"grasp_width_out_of_range_{extents['width_y_m']:.4f}m_expected_{GRASP_REGION_WIDTH_M:.4f}m")

    score = quality_score(head_tail_error, length_error, width_error, len(points), warnings)
    result = {
        "status": "ok",
        "warnings": warnings,
        "input": {
            "stage1_json": record.get("_stage1_json"),
            "image": record.get("image"),
            "raw_id": raw_id,
            "d2rgb": str(d2rgb_path),
        },
        "camera": {
            "frame": camera.get("camera_frame", "camera_rgb"),
            "fx": float(camera["fx"]),
            "fy": float(camera["fy"]),
            "cx": float(camera["cx"]),
            "cy": float(camera["cy"]),
            "depth_scale": float(camera["depth_scale"]),
        },
        "grasp_pose_camera": {
            "translation_m": [round(float(v), 8) for v in grasp_center_3d.tolist()],
            "rotation_matrix": [[round(float(v), 8) for v in row] for row in rotation.tolist()],
            "quaternion_xyzw": rotation_to_quaternion_xyzw(rotation),
        },
        "axes": {
            "x_tail_to_head": [round(float(v), 8) for v in rotation[:, 0].tolist()],
            "y_closing": [round(float(v), 8) for v in rotation[:, 1].tolist()],
            "z_approach": [round(float(v), 8) for v in rotation[:, 2].tolist()],
        },
        "quality": {
            "quality_score": score,
            "valid_mask_points": int(len(points)),
            "grasp_target": grasp_target_info,
            "grasp_target_camera_m": [round(float(v), 8) for v in grasp_center_3d.tolist()],
            "grasp_surface_anchor_camera_m": [round(float(v), 8) for v in grasp_target_3d.tolist()],
            "grasp_center_adjustment": grasp_center_info,
            "mask_point_mean_m": [round(float(v), 8) for v in mean.tolist()],
            "mask_point_median_m": [round(float(v), 8) for v in center.tolist()],
            "depth": depth_stats,
            "head": head_info,
            "tail": tail_info,
            "head_tail_distance_m": None if head_tail_distance is None else round(head_tail_distance, 6),
            "head_tail_distance_expected_m": HEAD_TAIL_DISTANCE_M,
            "head_tail_relative_error": None if head_tail_error is None else round(head_tail_error, 6),
            "grasp_region_extents_m": extents,
            "grasp_region_expected_m": {
                "length_x_m": GRASP_REGION_LENGTH_M,
                "width_y_m": GRASP_REGION_WIDTH_M,
                "thickness_z_m": GRASP_REGION_THICKNESS_M,
            },
            "grasp_region_relative_error": {
                "length_x": round(float(length_error), 6),
                "width_y": round(float(width_error), 6),
            },
            **rotation_info,
        },
    }
    return result, mask, points, rotation, head_xy, tail_xy


__all__ = ["draw_overlay", "estimate_record", "save_ply"]
