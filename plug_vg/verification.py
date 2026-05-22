"""Verification logic for saved 6D plug grasp outputs."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .config import GRASP_REGION_LENGTH_M, GRASP_REGION_WIDTH_M, HEAD_TAIL_DISTANCE_M
from .geometry import (
    depth_to_points,
    derive_grasp_target_2d,
    grasp_center_from_surface,
    keypoint_3d,
    polygon_mask,
    project_point_float,
    robust_depth_filter,
    robust_extent,
    rotation_to_quaternion_xyzw,
)
from .io import first_detection, read_json, write_json


def max_abs_diff(a: Any, b: Any) -> float:
    return float(np.max(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))))


def rel_error(value: float, expected: float) -> float:
    return abs(value - expected) / expected if expected else float("inf")


def result_stem(path: Path) -> str:
    name = path.stem
    return name[:-3] if name.endswith("_3d") else name


def verify_one(path: Path, camera: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    result = read_json(path)
    stem = result_stem(path)
    sample: dict[str, Any] = {
        "stem": stem,
        "result_json": str(path),
        "status": result.get("status"),
        "reason": result.get("reason"),
        "warnings": result.get("warnings") or [],
        "checks": {},
        "math_failures": [],
        "geometry_anomalies": [],
        "input_failures": [],
    }
    if result.get("status") != "ok":
        return sample

    input_info = result.get("input") or {}
    image_path = Path(input_info.get("image") or "")
    depth_path = Path(input_info.get("d2rgb") or "")
    stage1_path = Path(input_info.get("stage1_json") or "")
    for label, file_path in (("image", image_path), ("d2rgb", depth_path), ("stage1_json", stage1_path)):
        if not file_path.exists():
            sample["input_failures"].append(f"{label}_missing:{file_path}")
    if sample["input_failures"]:
        return sample

    stage1 = read_json(stage1_path)
    seg = first_detection(stage1, "segmentation")
    pose = first_detection(stage1, "pose")
    if seg is None:
        sample["input_failures"].append("stage1_segmentation_missing")
        return sample
    if pose is None:
        sample["input_failures"].append("stage1_pose_missing")
        return sample

    depth_raw = np.asarray(Image.open(depth_path))
    if depth_raw.ndim == 3:
        depth_raw = depth_raw[:, :, 0]
        sample["geometry_anomalies"].append("depth_image_has_multiple_channels_using_first_channel")
    image = Image.open(image_path)
    expected_shape = (int(camera["image_height"]), int(camera["image_width"]))
    if depth_raw.shape[:2] != expected_shape:
        sample["input_failures"].append(f"depth_shape_{depth_raw.shape[1]}x{depth_raw.shape[0]}_differs_from_camera_config")
    if image.size != (expected_shape[1], expected_shape[0]):
        sample["input_failures"].append(f"image_shape_{image.size[0]}x{image.size[1]}_differs_from_camera_config")
    if int(stage1.get("width", 0)) != image.size[0] or int(stage1.get("height", 0)) != image.size[1]:
        sample["input_failures"].append("stage1_dimensions_differ_from_rgb")

    keypoints = pose.get("keypoints") or {}
    head_xy = (keypoints.get("plug_head") or {}).get("xy")
    tail_xy = (keypoints.get("plug_tail") or {}).get("xy")
    if not head_xy or not tail_xy:
        sample["input_failures"].append("head_tail_keypoints_missing")
        return sample

    mask = polygon_mask(seg.get("polygon_xy") or [], depth_raw.shape[:2])
    points, pixels, z_values = depth_to_points(mask, depth_raw, camera, 0.1, 1.0)
    points, pixels, depth_stats = robust_depth_filter(points, pixels, z_values)
    if len(points) == 0:
        sample["input_failures"].append("no_valid_mask_depth_points")
        return sample

    target_xy, target_method = derive_grasp_target_2d(mask, head_xy, tail_xy)
    if target_xy is None:
        sample["input_failures"].append("grasp_target_2d_missing")
        return sample
    target_3d, target_info = keypoint_3d(target_xy, depth_raw, camera, 5, 0.1, 1.0)
    if target_3d is None:
        target_3d = np.median(points, axis=0)
        target_info["fallback"] = "mask_point_median"
    target_info["method"] = target_method

    head_3d, head_info = keypoint_3d(head_xy, depth_raw, camera, 5, 0.1, 1.0)
    tail_3d, tail_info = keypoint_3d(tail_xy, depth_raw, camera, 5, 0.1, 1.0)

    pose_camera = result.get("grasp_pose_camera") or {}
    json_translation = np.asarray(pose_camera.get("translation_m") or [float("nan")] * 3, dtype=np.float64)
    json_rotation = np.asarray(pose_camera.get("rotation_matrix") or np.full((3, 3), float("nan")), dtype=np.float64)
    json_quaternion = np.asarray(pose_camera.get("quaternion_xyzw") or [float("nan")] * 4, dtype=np.float64)

    expected_translation, center_adjustment = grasp_center_from_surface(target_3d, json_rotation)
    translation_error_m = float(np.linalg.norm(expected_translation.astype(np.float64) - json_translation))
    projected = project_point_float(json_translation, camera)
    surface_projected = project_point_float(target_3d.astype(np.float64), camera)
    reprojection_error_px = None if projected is None else float(np.linalg.norm(projected - np.asarray(target_xy, dtype=np.float64)))
    surface_reprojection_error_px = (
        None if surface_projected is None else float(np.linalg.norm(surface_projected - np.asarray(target_xy, dtype=np.float64)))
    )
    if not np.isfinite(translation_error_m) or translation_error_m > args.translation_tol_m:
        sample["math_failures"].append(f"translation_error_m={translation_error_m:.9g}")
    if surface_reprojection_error_px is None or surface_reprojection_error_px > args.reprojection_tol_px:
        sample["math_failures"].append(f"surface_reprojection_error_px={surface_reprojection_error_px}")

    rt_r = json_rotation.T @ json_rotation
    orth_error = float(np.max(np.abs(rt_r - np.eye(3))))
    determinant = float(np.linalg.det(json_rotation))
    if not np.isfinite(orth_error) or orth_error > args.rotation_orth_tol:
        sample["math_failures"].append(f"rotation_orth_error={orth_error:.9g}")
    if not np.isfinite(determinant) or determinant < args.det_min or determinant > args.det_max:
        sample["math_failures"].append(f"rotation_det={determinant:.9g}")

    axes = result.get("axes") or {}
    axis_errors = {
        "x_tail_to_head": max_abs_diff(axes.get("x_tail_to_head", [float("nan")] * 3), json_rotation[:, 0]),
        "y_closing": max_abs_diff(axes.get("y_closing", [float("nan")] * 3), json_rotation[:, 1]),
        "z_approach": max_abs_diff(axes.get("z_approach", [float("nan")] * 3), json_rotation[:, 2]),
    }
    for axis_name, axis_error in axis_errors.items():
        if not np.isfinite(axis_error) or axis_error > args.axes_tol:
            sample["math_failures"].append(f"{axis_name}_axis_error={axis_error:.9g}")

    recomputed_quaternion = np.asarray(rotation_to_quaternion_xyzw(json_rotation), dtype=np.float64)
    quaternion_error = min(
        float(np.linalg.norm(recomputed_quaternion - json_quaternion)),
        float(np.linalg.norm(recomputed_quaternion + json_quaternion)),
    )
    if not np.isfinite(quaternion_error) or quaternion_error > args.quaternion_tol:
        sample["math_failures"].append(f"quaternion_error={quaternion_error:.9g}")

    head_tail_distance = None
    head_tail_relative_error = None
    if head_3d is None or tail_3d is None:
        sample["geometry_anomalies"].append("head_or_tail_depth_missing")
    else:
        head_tail_distance = float(np.linalg.norm(head_3d - tail_3d))
        head_tail_relative_error = rel_error(head_tail_distance, HEAD_TAIL_DISTANCE_M)
        if head_tail_relative_error > args.head_tail_tolerance:
            sample["geometry_anomalies"].append(
                f"head_tail_distance_out_of_tolerance_{head_tail_distance:.4f}m_expected_{HEAD_TAIL_DISTANCE_M:.4f}m"
            )

    center = np.median(points, axis=0)
    extents = robust_extent(points, center, json_rotation)
    length_error = rel_error(extents["length_x_m"], GRASP_REGION_LENGTH_M)
    width_error = rel_error(extents["width_y_m"], GRASP_REGION_WIDTH_M)
    if length_error > args.grasp_region_tolerance:
        sample["geometry_anomalies"].append(f"grasp_length_out_of_range_{extents['length_x_m']:.4f}m")
    if width_error > args.grasp_region_tolerance:
        sample["geometry_anomalies"].append(f"grasp_width_out_of_range_{extents['width_y_m']:.4f}m")

    json_quality = result.get("quality") or {}
    sample["checks"] = {
        "quality_score": json_quality.get("quality_score"),
        "mask_pixels": int(np.count_nonzero(mask)),
        "valid_mask_points": int(len(points)),
        "depth": depth_stats,
        "grasp_target": target_info,
        "expected_translation_m": [round(float(v), 8) for v in expected_translation.tolist()],
        "grasp_surface_anchor_camera_m": [round(float(v), 8) for v in target_3d.tolist()],
        "grasp_center_adjustment": center_adjustment,
        "translation_error_m": translation_error_m,
        "reprojection_error_px": reprojection_error_px,
        "surface_reprojection_error_px": surface_reprojection_error_px,
        "rotation_orth_error": orth_error,
        "rotation_det": determinant,
        "axis_errors": axis_errors,
        "quaternion_error": quaternion_error,
        "head": head_info,
        "tail": tail_info,
        "head_tail_distance_m": None if head_tail_distance is None else round(head_tail_distance, 6),
        "head_tail_relative_error": None if head_tail_relative_error is None else round(head_tail_relative_error, 6),
        "grasp_region_extents_m": extents,
        "grasp_region_relative_error": {
            "length_x": round(length_error, 6),
            "width_y": round(width_error, 6),
        },
        "json_quality_comparison": {
            "valid_mask_points_diff": int(len(points)) - int(json_quality.get("valid_mask_points") or 0),
            "head_tail_distance_diff_m": None
            if head_tail_distance is None or json_quality.get("head_tail_distance_m") is None
            else round(head_tail_distance - float(json_quality["head_tail_distance_m"]), 9),
            "length_x_diff_m": round(extents["length_x_m"] - float((json_quality.get("grasp_region_extents_m") or {}).get("length_x_m", 0.0)), 9),
            "width_y_diff_m": round(extents["width_y_m"] - float((json_quality.get("grasp_region_extents_m") or {}).get("width_y_m", 0.0)), 9),
        },
    }
    overlay = args.run_dir / "overlays" / f"{stem}_grasp3d.jpg"
    sample["overlay"] = str(overlay) if overlay.exists() else None
    return sample


def compact_samples(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact = []
    for sample in samples:
        checks = sample.get("checks") or {}
        compact.append(
            {
                "stem": sample["stem"],
                "input_failures": sample.get("input_failures") or [],
                "math_failures": sample.get("math_failures") or [],
                "geometry_anomalies": sample.get("geometry_anomalies") or [],
                "warnings": sample.get("warnings") or [],
                "quality_score": checks.get("quality_score"),
                "head_tail_distance_m": checks.get("head_tail_distance_m"),
                "head_tail_relative_error": checks.get("head_tail_relative_error"),
                "grasp_region_extents_m": checks.get("grasp_region_extents_m"),
                "overlay": sample.get("overlay"),
            }
        )
    return compact


def build_report(samples: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    ok_samples = [sample for sample in samples if sample["status"] == "ok"]
    skipped = [sample for sample in samples if sample["status"] != "ok"]
    warning_counts: Counter[str] = Counter()
    for sample in ok_samples:
        warning_counts.update(sample.get("warnings") or [])

    math_failures = [sample for sample in ok_samples if sample["math_failures"]]
    input_failures = [sample for sample in ok_samples if sample["input_failures"]]
    head_tail_anomalies = [
        sample
        for sample in ok_samples
        if any(str(item).startswith("head_tail_distance_out_of_tolerance") for item in sample["geometry_anomalies"])
    ]
    grasp_region_anomalies = [
        sample
        for sample in ok_samples
        if any(str(item).startswith("grasp_length_out_of_range") or str(item).startswith("grasp_width_out_of_range") for item in sample["geometry_anomalies"])
    ]

    by_quality = sorted(
        [sample for sample in ok_samples if sample["checks"].get("quality_score") is not None],
        key=lambda item: float(item["checks"]["quality_score"]),
    )
    by_head_tail = sorted(
        [sample for sample in ok_samples if sample["checks"].get("head_tail_relative_error") is not None],
        key=lambda item: float(item["checks"]["head_tail_relative_error"]),
        reverse=True,
    )
    warning_samples = [sample for sample in ok_samples if sample.get("warnings")]
    review_map: dict[str, dict[str, Any]] = {}
    for reason, group in (
        ("low_quality", by_quality[: args.review_count]),
        ("high_head_tail_error", by_head_tail[: args.review_count]),
        ("has_warnings", warning_samples),
    ):
        for sample in group:
            entry = review_map.setdefault(
                sample["stem"],
                {
                    "stem": sample["stem"],
                    "overlay": sample.get("overlay"),
                    "quality_score": sample["checks"].get("quality_score"),
                    "head_tail_relative_error": sample["checks"].get("head_tail_relative_error"),
                    "warnings": sample.get("warnings") or [],
                    "reasons": [],
                },
            )
            entry["reasons"].append(reason)

    return {
        "metadata": {
            "run_dir": str(args.run_dir),
            "camera_config": str(args.camera_config),
            "thresholds": {
                "translation_tol_m": args.translation_tol_m,
                "reprojection_tol_px": args.reprojection_tol_px,
                "rotation_orth_tol": args.rotation_orth_tol,
                "det_min": args.det_min,
                "det_max": args.det_max,
                "axes_tol": args.axes_tol,
                "quaternion_tol": args.quaternion_tol,
                "head_tail_tolerance": args.head_tail_tolerance,
                "grasp_region_tolerance": args.grasp_region_tolerance,
            },
        },
        "summary": {
            "total_result_jsons": len(samples),
            "ok_checked": len(ok_samples),
            "skipped_non_ok": len(skipped),
            "skipped_d2rgb_depth_missing": sum(1 for sample in skipped if sample.get("reason") == "d2rgb_depth_missing"),
            "input_failures": len(input_failures),
            "math_failures": len(math_failures),
            "head_tail_anomalies": len(head_tail_anomalies),
            "grasp_region_anomalies": len(grasp_region_anomalies),
            "warning_samples": len(warning_samples),
        },
        "warning_counts": dict(warning_counts.most_common()),
        "input_failures": compact_samples(input_failures),
        "math_failures": compact_samples(math_failures),
        "head_tail_anomalies": compact_samples(head_tail_anomalies),
        "grasp_region_anomalies": compact_samples(grasp_region_anomalies),
        "review_candidates": list(review_map.values()),
        "samples": samples,
    }


def markdown_report(report: dict[str, Any], max_rows: int) -> str:
    summary = report["summary"]
    lines = [
        "# 6D Output Verification Report",
        "",
        "## Summary",
        "",
        f"- Total result JSONs: {summary['total_result_jsons']}",
        f"- OK samples checked: {summary['ok_checked']}",
        f"- Skipped non-OK samples: {summary['skipped_non_ok']}",
        f"- Skipped d2rgb_depth_missing: {summary['skipped_d2rgb_depth_missing']}",
        f"- Input failures among OK samples: {summary['input_failures']}",
        f"- Math consistency failures: {summary['math_failures']}",
        f"- Head-tail geometry anomalies: {summary['head_tail_anomalies']}",
        f"- Grasp-region geometry anomalies: {summary['grasp_region_anomalies']}",
        f"- OK samples with warnings: {summary['warning_samples']}",
        "",
    ]
    if report["warning_counts"]:
        lines.extend(["## Warning Counts", ""])
        for warning, count in report["warning_counts"].items():
            lines.append(f"- {count}: `{warning}`")
        lines.append("")

    for title, key in (
        ("Input Failures", "input_failures"),
        ("Math Failures", "math_failures"),
        ("Head-Tail Anomalies", "head_tail_anomalies"),
        ("Grasp Region Anomalies", "grasp_region_anomalies"),
    ):
        lines.extend([f"## {title}", ""])
        rows = report[key][:max_rows]
        if not rows:
            lines.append("- None")
        for row in rows:
            details = row.get("math_failures") or row.get("input_failures") or row.get("geometry_anomalies") or []
            lines.append(f"- `{row['stem']}`: {', '.join(details) if details else 'flagged'}")
        if len(report[key]) > max_rows:
            lines.append(f"- ... {len(report[key]) - max_rows} more")
        lines.append("")

    lines.extend(["## Review Candidates", ""])
    for row in report["review_candidates"][:max_rows]:
        reasons = ", ".join(row["reasons"])
        overlay = row.get("overlay") or "missing overlay"
        lines.append(
            f"- `{row['stem']}` ({reasons}): quality={row.get('quality_score')}, "
            f"head_tail_relerr={row.get('head_tail_relative_error')}, overlay={overlay}"
        )
    if not report["review_candidates"]:
        lines.append("- None")
    return "\n".join(lines) + "\n"


def print_summary(report: dict[str, Any], max_rows: int) -> None:
    print(markdown_report(report, max_rows))


def verify_outputs(args: argparse.Namespace, camera: dict[str, Any]) -> dict[str, Any]:
    json_dir = args.run_dir / "jsons"
    paths = sorted(path for path in json_dir.glob("*_3d.json") if path.name != "summary.json")
    if not paths:
        raise FileNotFoundError(f"No *_3d.json files found under {json_dir}")
    samples = [verify_one(path, camera, args) for path in paths]
    return build_report(samples, args)


def write_reports(report: dict[str, Any], report_json: Path, report_md: Path, max_print: int) -> None:
    write_json(report_json, report)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_md.write_text(markdown_report(report, max_print), encoding="utf-8")
