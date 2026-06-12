#!/usr/bin/env python3
"""Estimate a line-box inner plane base-frame X value from one RGB-D frame."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from plug_vg.config import DEFAULT_CAMERA, ROOT, load_camera
from plug_vg.geometry import save_ply
from plug_vg.io import write_json
from plug_vg.linebox_depth import (
    choose_farthest_stable_depth_peak,
    clamp_roi,
    depth_pixels_to_camera_points,
    robust_axis_stats,
    roi_valid_depth_pixels,
    select_depth_band,
    transform_points,
)
from plug_vg.robot_transform import load_hand_eye_matrix, robot_pose_to_matrix, round_list

r'''

python infer_linebox_inner_x_single.py \
  --rgb test_20260523/color_20260523_142307_548_0.png \
  --d2rgb test_20260523/D2RGB_20260523_142307_548_0.png \
  --robot-pose -0.58694 -0.03700 0.59149 -2.097 0.000 1.555 \
  --roi 700 300 1250 500 \
  --reference-base-x -0.900 \
  --output-dir ultralytics/runs/linebox_inner_x_single \
  --save-overlay \
  --save-histogram

'''

DEFAULT_OUTPUT = ROOT / "ultralytics" / "runs" / "linebox_inner_x_single"
DEFAULT_HAND_EYE = ROOT / "hand_eye_calibration" / "eye_hand_data" / "calib_20260522" / "hand_eye_result_in-hand.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb", type=Path, required=True, help="RGB image path.")
    parser.add_argument("--d2rgb", type=Path, required=True, help="Registered D2RGB depth PNG path.")
    parser.add_argument(
        "--robot-pose",
        type=float,
        nargs=6,
        required=True,
        metavar=("X", "Y", "Z", "ROLL", "PITCH", "YAW"),
        help="Current robot end-effector pose T_base_end as x y z roll pitch yaw in meters/radians.",
    )
    parser.add_argument(
        "--roi",
        type=float,
        nargs=4,
        required=True,
        metavar=("X1", "Y1", "X2", "Y2"),
        help="Coarse RGB/D2RGB pixel ROI as x1 y1 x2 y2.",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for JSON and optional debug artifacts.")
    parser.add_argument("--camera-config", type=Path, default=DEFAULT_CAMERA, help="RGB-D camera intrinsics YAML.")
    parser.add_argument("--hand-eye-config", type=Path, default=DEFAULT_HAND_EYE, help="Eye-in-hand calibration YAML containing T_end_camera.")
    parser.add_argument("--min-depth", type=float, default=0.1, help="Minimum valid ROI depth in meters.")
    parser.add_argument("--max-depth", type=float, default=2.0, help="Maximum valid ROI depth in meters.")
    parser.add_argument("--hist-bin-size-m", type=float, default=0.01, help="Depth histogram bin size in meters.")
    parser.add_argument("--min-peak-points", type=int, default=200, help="Minimum neighboring-bin histogram support for a stable plane peak.")
    parser.add_argument("--min-peak-fraction", type=float, default=0.03, help="Minimum stable peak support as a fraction of valid ROI depth pixels.")
    parser.add_argument("--reference-base-x", type=float, default=None, help="Optional known window/base X used to compute width along negative base X.")
    parser.add_argument("--save-overlay", action="store_true", help="Save an RGB overlay with ROI and selected target-plane pixels.")
    parser.add_argument("--save-histogram", action="store_true", help="Save a depth histogram PNG with the selected peak.")
    parser.add_argument("--save-ply", action="store_true", help="Save selected target-plane points in robot base frame as ASCII PLY.")
    return parser.parse_args()


def output_json_path(output_dir: Path, rgb_path: Path) -> Path:
    return output_dir / f"{rgb_path.stem}_linebox_inner_x.json"


def make_failure(args: argparse.Namespace, reason: str, warnings: list[str] | None = None) -> dict[str, Any]:
    return {
        "status": "failed",
        "reason": reason,
        "warnings": list(warnings or []),
        "input": input_payload(args),
    }


def input_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "image": str(args.rgb),
        "d2rgb": str(args.d2rgb),
        "robot_pose_xyzrpy_m_rad": [float(v) for v in args.robot_pose],
        "roi_xyxy_input": [float(v) for v in args.roi],
        "reference_base_x_m": None if args.reference_base_x is None else float(args.reference_base_x),
        "camera_config": str(args.camera_config),
        "hand_eye_config": str(args.hand_eye_config),
    }


def print_result(result: dict[str, Any], output_path: Path) -> None:
    print(f"status: {result.get('status')}")
    if result.get("status") == "ok":
        print(f"linebox_inner_plane_base_x_m: {result.get('linebox_inner_plane_base_x_m')}")
        if "linebox_width_along_negative_base_x_m" in result:
            print(f"linebox_width_along_negative_base_x_m: {result.get('linebox_width_along_negative_base_x_m')}")
        depth = result.get("selected_plane_depth_peak_m")
        print(f"selected_plane_depth_peak_m: {depth}")
        warnings = result.get("warnings") or []
        if warnings:
            print(f"warnings: {warnings}")
    else:
        print(f"reason: {result.get('reason')}")
        warnings = result.get("warnings") or []
        if warnings:
            print(f"warnings: {warnings}")
    print(f"json: {output_path}")
    timing = result.get("timing") or {}
    if timing:
        print(f"Timing: single end-to-end = {timing.get('single_end_to_end_s')} (s)")


def read_depth_m(path: Path, camera: dict[str, Any], warnings: list[str]) -> np.ndarray | None:
    depth_raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth_raw is None:
        return None
    if depth_raw.ndim == 3:
        warnings.append("depth_image_has_multiple_channels_using_first_channel")
        depth_raw = depth_raw[:, :, 0]
    return depth_raw.astype(np.float64) * float(camera["depth_scale"])


def selected_pixel_mask(shape: tuple[int, int], xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    if len(xs) == 0:
        return mask
    xi = np.clip(np.round(xs).astype(np.int32), 0, shape[1] - 1)
    yi = np.clip(np.round(ys).astype(np.int32), 0, shape[0] - 1)
    mask[yi, xi] = 1
    return mask


def save_overlay(
    image_bgr: np.ndarray,
    depth_shape: tuple[int, int],
    roi_xyxy: tuple[int, int, int, int],
    selected_xs: np.ndarray,
    selected_ys: np.ndarray,
    output_path: Path,
) -> None:
    canvas = image_bgr.copy()
    mask = selected_pixel_mask(depth_shape, selected_xs, selected_ys)
    if canvas.shape[:2] != mask.shape:
        mask = cv2.resize(mask, (canvas.shape[1], canvas.shape[0]), interpolation=cv2.INTER_NEAREST)

    layer = canvas.copy()
    layer[mask > 0] = (0, 80, 255)
    canvas = cv2.addWeighted(layer, 0.45, canvas, 0.55, 0)
    x1, y1, x2, y2 = roi_xyxy
    if image_bgr.shape[:2] != depth_shape:
        sx = image_bgr.shape[1] / float(depth_shape[1])
        sy = image_bgr.shape[0] / float(depth_shape[0])
        x1, x2 = int(round(x1 * sx)), int(round(x2 * sx))
        y1, y2 = int(round(y1 * sy)), int(round(y2 * sy))
    cv2.rectangle(canvas, (x1, y1), (max(x1, x2 - 1), max(y1, y2 - 1)), (0, 255, 255), 3)
    cv2.putText(canvas, "ROI", (x1 + 8, max(24, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)


def save_histogram(histogram: dict[str, Any], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    edges = np.asarray(histogram["bin_edges_m"], dtype=np.float64)
    counts = np.asarray(histogram["counts"], dtype=np.float64)
    selected = histogram.get("selected_peak") or {}
    centers = (edges[:-1] + edges[1:]) * 0.5

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=140)
    ax.bar(centers, counts, width=float(histogram["bin_size_m"]) * 0.92, color="#5876a6", alpha=0.78)
    if selected:
        ax.axvline(float(selected["center_m"]), color="#d33f2f", linewidth=2.0, label="selected far stable peak")
        ax.legend(loc="upper right")
    ax.set_xlabel("Depth in ROI (m)")
    ax.set_ylabel("Pixel count")
    ax.set_title("Line-box ROI depth histogram")
    ax.grid(True, alpha=0.22)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def build_ok_result(
    args: argparse.Namespace,
    camera: dict[str, Any],
    roi_xyxy: tuple[int, int, int, int],
    depth_shape: tuple[int, int],
    peak: dict[str, Any],
    histogram: dict[str, Any],
    selected_xs: np.ndarray,
    selected_ys: np.ndarray,
    selected_z: np.ndarray,
    points_camera: np.ndarray,
    points_base: np.ndarray,
    t_base_end: np.ndarray,
    t_end_camera: np.ndarray,
    warnings: list[str],
) -> dict[str, Any]:
    x_stats = robust_axis_stats(points_base[:, 0])
    result: dict[str, Any] = {
        "status": "ok",
        "warnings": warnings,
        "input": input_payload(args),
        "roi": {
            "xyxy_clamped": [int(v) for v in roi_xyxy],
            "width_px": int(roi_xyxy[2] - roi_xyxy[0]),
            "height_px": int(roi_xyxy[3] - roi_xyxy[1]),
            "image_width_px": int(depth_shape[1]),
            "image_height_px": int(depth_shape[0]),
        },
        "camera": {
            "frame": camera.get("camera_frame", "camera_rgb"),
            "fx": float(camera["fx"]),
            "fy": float(camera["fy"]),
            "cx": float(camera["cx"]),
            "cy": float(camera["cy"]),
            "depth_scale": float(camera["depth_scale"]),
        },
        "selected_plane_depth_peak_m": float(peak["center_m"]),
        "linebox_inner_plane_base_x_m": x_stats["median_m"],
        "base_x_stats": x_stats,
        "selected_plane": {
            "selected_pixel_count": int(len(selected_z)),
            "depth_band_center_m": float(peak["center_m"]),
            "depth_band_half_width_m": round(float(args.hist_bin_size_m) * 1.5, 8),
            "depth_min_m": round(float(np.min(selected_z)), 8),
            "depth_median_m": round(float(np.median(selected_z)), 8),
            "depth_max_m": round(float(np.max(selected_z)), 8),
            "camera_point_mean_m": [round(float(v), 8) for v in np.mean(points_camera, axis=0).tolist()],
            "base_point_mean_m": [round(float(v), 8) for v in np.mean(points_base, axis=0).tolist()],
        },
        "depth_histogram": histogram,
        "frame_transform": {
            "chain": "T_base_point = T_base_end_current @ T_end_camera @ T_camera_point",
            "hand_eye_config": str(args.hand_eye_config),
            "T_base_end_current": round_list(t_base_end),
            "T_end_camera": round_list(t_end_camera),
            "T_base_camera": round_list(t_base_end @ t_end_camera),
        },
    }
    if args.reference_base_x is not None:
        width = float(args.reference_base_x) - float(result["linebox_inner_plane_base_x_m"])
        result["reference_base_x_m"] = round(float(args.reference_base_x), 8)
        result["linebox_width_along_negative_base_x_m"] = round(width, 8)
        result["linebox_width_formula"] = "reference_base_x_m - linebox_inner_plane_base_x_m"
    return result


def run(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_json_path(args.output_dir, args.rgb)
    warnings: list[str] = []

    if args.min_depth >= args.max_depth:
        result = make_failure(args, "invalid_depth_range", ["min_depth_must_be_less_than_max_depth"])
        write_json(json_path, result)
        return result, json_path
    if args.hist_bin_size_m <= 0.0:
        result = make_failure(args, "invalid_hist_bin_size")
        write_json(json_path, result)
        return result, json_path
    if args.min_peak_fraction < 0.0 or args.min_peak_fraction > 1.0:
        result = make_failure(args, "invalid_min_peak_fraction")
        write_json(json_path, result)
        return result, json_path

    if not args.rgb.is_file():
        result = make_failure(args, "rgb_missing")
        write_json(json_path, result)
        return result, json_path
    if not args.d2rgb.is_file():
        result = make_failure(args, "d2rgb_missing")
        write_json(json_path, result)
        return result, json_path

    camera = load_camera(args.camera_config)
    image = cv2.imread(str(args.rgb))
    if image is None:
        result = make_failure(args, "rgb_unreadable")
        write_json(json_path, result)
        return result, json_path

    depth_m = read_depth_m(args.d2rgb, camera, warnings)
    if depth_m is None:
        result = make_failure(args, "d2rgb_unreadable", warnings)
        write_json(json_path, result)
        return result, json_path
    if image.shape[:2] != depth_m.shape[:2]:
        warnings.append(f"rgb_shape_{image.shape[1]}x{image.shape[0]}_differs_from_d2rgb_shape_{depth_m.shape[1]}x{depth_m.shape[0]}")
    expected_shape = (int(camera["image_height"]), int(camera["image_width"]))
    if depth_m.shape[:2] != expected_shape:
        warnings.append(f"depth_shape_{depth_m.shape[1]}x{depth_m.shape[0]}_differs_from_camera_config")

    try:
        roi_xyxy = clamp_roi(args.roi, depth_m.shape[1], depth_m.shape[0])
    except ValueError as exc:
        result = make_failure(args, "invalid_roi", [str(exc)])
        write_json(json_path, result)
        return result, json_path

    xs, ys, z = roi_valid_depth_pixels(depth_m, roi_xyxy, args.min_depth, args.max_depth)
    peak, histogram = choose_farthest_stable_depth_peak(
        z,
        args.min_depth,
        args.max_depth,
        args.hist_bin_size_m,
        args.min_peak_points,
        args.min_peak_fraction,
    )
    if peak is None:
        result = make_failure(args, histogram.get("reason", "no_stable_depth_peak"), warnings)
        result["roi"] = {
            "xyxy_clamped": [int(v) for v in roi_xyxy],
            "width_px": int(roi_xyxy[2] - roi_xyxy[0]),
            "height_px": int(roi_xyxy[3] - roi_xyxy[1]),
        }
        result["depth_histogram"] = histogram
        write_json(json_path, result)
        return result, json_path

    selected_xs, selected_ys, selected_z, _keep = select_depth_band(xs, ys, z, float(peak["center_m"]), args.hist_bin_size_m)
    if len(selected_z) == 0:
        result = make_failure(args, "selected_depth_band_empty", warnings)
        result["depth_histogram"] = histogram
        write_json(json_path, result)
        return result, json_path

    t_end_camera = load_hand_eye_matrix(args.hand_eye_config)
    t_base_end = robot_pose_to_matrix(args.robot_pose)
    t_base_camera = t_base_end @ t_end_camera
    points_camera = depth_pixels_to_camera_points(selected_xs, selected_ys, selected_z, camera)
    points_base = transform_points(points_camera, t_base_camera)

    result = build_ok_result(
        args,
        camera,
        roi_xyxy,
        depth_m.shape[:2],
        peak,
        histogram,
        selected_xs,
        selected_ys,
        selected_z,
        points_camera,
        points_base,
        t_base_end,
        t_end_camera,
        warnings,
    )

    stem = args.rgb.stem
    artifacts: dict[str, str] = {}
    if args.save_overlay:
        overlay_path = args.output_dir / "overlays" / f"{stem}_linebox_roi_overlay.jpg"
        save_overlay(image, depth_m.shape[:2], roi_xyxy, selected_xs, selected_ys, overlay_path)
        artifacts["overlay"] = str(overlay_path)
    if args.save_histogram:
        hist_path = args.output_dir / "histograms" / f"{stem}_linebox_depth_histogram.png"
        save_histogram(histogram, hist_path)
        artifacts["histogram"] = str(hist_path)
    if args.save_ply:
        ply_path = args.output_dir / "ply" / f"{stem}_linebox_plane_base_points.ply"
        save_ply(points_base, ply_path)
        artifacts["ply_base_points"] = str(ply_path)
    if artifacts:
        result["artifacts"] = artifacts

    write_json(json_path, result)
    return result, json_path


def main() -> None:
    args = parse_args()
    start_time = time.perf_counter()
    try:
        result, json_path = run(args)
    except Exception as exc:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_json_path(args.output_dir, args.rgb)
        result = make_failure(args, type(exc).__name__, [str(exc)])
        elapsed = time.perf_counter() - start_time
        result["timing"] = {
            "single_end_to_end_s": round(float(elapsed), 6),
            "single_end_to_end_ms": round(float(elapsed * 1000.0), 3),
            "scope": "rgbd_roi_to_linebox_inner_plane_base_x",
            "includes_debug_artifact_writes": bool(args.save_overlay or args.save_histogram or args.save_ply),
        }
        write_json(json_path, result)
        print_result(result, json_path)
        raise SystemExit(1) from exc

    elapsed = time.perf_counter() - start_time
    result["timing"] = {
        "single_end_to_end_s": round(float(elapsed), 6),
        "single_end_to_end_ms": round(float(elapsed * 1000.0), 3),
        "scope": "rgbd_roi_to_linebox_inner_plane_base_x",
        "includes_debug_artifact_writes": bool(args.save_overlay or args.save_histogram or args.save_ply),
    }
    write_json(json_path, result)
    print_result(result, json_path)
    if result.get("status") != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
