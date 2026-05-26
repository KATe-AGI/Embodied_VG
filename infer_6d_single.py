#!/usr/bin/env python3
"""Run single-frame plug 6D grasp inference for real-machine validation."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from plug_vg.config import DEFAULT_CAMERA, DEFAULT_POSE_WEIGHTS, DEFAULT_SEG_WEIGHTS, ROOT, load_camera
from plug_vg.geometry import draw_overlay as draw_3d_overlay, save_ply
from plug_vg.grasp_pose import estimate_record
from plug_vg.io import raw_id_from_image, write_json
from plug_vg.robot_transform import convert_camera_grasp_to_base, load_hand_eye_matrix, robot_pose_to_matrix
from plug_vg.vision import draw_overlay as draw_stage1_overlay, run_models
from plug_vg.window_grasp import (
    DEFAULT_MARGIN_M,
    WindowGraspError,
    add_window_candidates,
    attach_direct_visual_grasp,
    build_window_geometry,
    resolve_window_inputs,
)
from tools.visualize_base_pose import build_view_data, default_output_path as default_base_view_path, render_html, write_html

from infer import YOLO

r'''
windows:
python infer_6d_single.py `
  --rgb test_20260525\undistort_color_20260525_152152_943_0.png `
  --d2rgb test_20260525\D2RGB_20260525_152152_943_0.png `
  --robot-pose  -0.58694 -0.03700 0.59149 -2.097 0.000 1.555 `
  --output-dir ultralytics/runs/plug_6d_single `
  --window-config configs/window/box_window.yaml `
  --save-overlay


ubuntu:
python infer_6d_single.py \
  --rgb test_20260523/color_20260523_142307_548_0.png \
  --d2rgb test_20260523/D2RGB_20260523_142307_548_0.png \
  --robot-pose  -0.6119 -0.0641 0.5842 -2.174 0.080 1.449 \
  --output-dir ultralytics/runs/plug_6d_single \
  --window-config configs/window/box_window.yaml \
  --save-overlay \
  --save-base-view \
  --save-ply
'''

DEFAULT_OUTPUT = ROOT / "ultralytics" / "runs" / "plug_6d_single"
DEFAULT_HAND_EYE = ROOT / "hand_eye_calibration" / "eye_hand_data" / "calib_20260522" / "hand_eye_result_in-hand.yaml"
DEFAULT_ROBOT_CONFIG = ROOT / "configs" / "robot" / "cs_robot.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Window geometry is optional. Provide --window-config or --window-corners-base to enable window-constrained grasp candidates.",
    )
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
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for JSON and optional debug artifacts.")
    parser.add_argument("--seg-weights", type=Path, default=DEFAULT_SEG_WEIGHTS, help="Segmentation weights.")
    parser.add_argument("--pose-weights", type=Path, default=DEFAULT_POSE_WEIGHTS, help="Pose weights.")
    parser.add_argument("--camera-config", type=Path, default=DEFAULT_CAMERA, help="RGB-D camera intrinsics YAML.")
    parser.add_argument("--hand-eye-config", type=Path, default=DEFAULT_HAND_EYE, help="Eye-in-hand calibration YAML containing T_end_camera.")
    parser.add_argument("--robot-config", type=Path, default=DEFAULT_ROBOT_CONFIG, help="Robot config YAML.")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.7, help="YOLO IoU threshold.")
    parser.add_argument("--device", default=None, help="CUDA device, e.g. 0, or cpu.")
    parser.add_argument("--max-det", type=int, default=10, help="Maximum detections per YOLO model.")
    parser.add_argument("--min-depth", type=float, default=0.1, help="Minimum valid depth in meters.")
    parser.add_argument("--max-depth", type=float, default=1.0, help="Maximum valid depth in meters.")
    parser.add_argument("--min-points", type=int, default=200, help="Minimum filtered mask point count.")
    parser.add_argument("--keypoint-window", type=int, default=5, help="Odd pixel window size for keypoint depth lookup.")
    parser.add_argument("--plane-threshold", type=float, default=0.004, help="RANSAC plane inlier threshold in meters.")
    parser.add_argument("--ransac-iters", type=int, default=128, help="RANSAC plane iterations.")
    parser.add_argument("--head-tail-tolerance", type=float, default=0.35, help="Relative tolerance for 3D head-tail distance.")
    parser.add_argument("--axis-scale", type=float, default=0.1, help="Overlay XYZ axis length in meters.")
    parser.add_argument("--axis-thickness", type=int, default=5, help="Overlay XYZ axis line thickness in pixels.")
    parser.add_argument("--save-overlay", action="store_true", help="Save YOLO and 3D grasp overlays.")
    parser.add_argument("--save-ply", action="store_true", help="Save filtered mask point cloud as an ASCII PLY file.")
    parser.add_argument("--save-base-view", action="store_true", help="Save an interactive HTML 3D view of the final base-frame grasp pose.")
    parser.add_argument("--window-config", type=Path, default=None, help="Optional YAML file containing base-frame window corners W1-W4. Enables window-constrained grasp candidates.")
    parser.add_argument(
        "--window-corners-base",
        type=float,
        nargs=12,
        default=None,
        metavar=("W1X", "W1Y", "W1Z", "W2X", "W2Y", "W2Z", "W3X", "W3Y", "W3Z", "W4X", "W4Y", "W4Z"),
        help="Optional window corners W1 W2 W3 W4 in robot base frame, meters. Overrides --window-config corners and enables window-constrained grasp candidates.",
    )
    parser.add_argument(
        "--window-margin-m",
        type=float,
        default=None,
        help=f"Window inward sampling margin in meters. Defaults to YAML margin_m or {DEFAULT_MARGIN_M}.",
    )
    return parser.parse_args()


def output_json_path(output_dir: Path, rgb_path: Path) -> Path:
    return output_dir / f"{rgb_path.stem}_6d_base.json"


def window_constraint_requested(args: argparse.Namespace) -> bool:
    return args.window_config is not None or args.window_corners_base is not None


def margin_ignored_without_window(args: argparse.Namespace) -> bool:
    return args.window_margin_m is not None and not window_constraint_requested(args)


def make_failure(args: argparse.Namespace, reason: str, warnings: list[str] | None = None) -> dict[str, Any]:
    all_warnings = list(warnings or [])
    if margin_ignored_without_window(args):
        all_warnings.append("window_margin_ignored_without_window_geometry")
    return {
        "status": "failed",
        "reason": reason,
        "warnings": all_warnings,
        "input": {
            "image": str(args.rgb),
            "d2rgb": str(args.d2rgb),
            "robot_pose_xyzrpy_m_rad": [float(v) for v in args.robot_pose],
            "window_config": None if args.window_config is None else str(args.window_config),
            "window_corners_base_provided": args.window_corners_base is not None,
            "window_constraint_enabled": window_constraint_requested(args),
            "window_margin_m": None if args.window_margin_m is None else float(args.window_margin_m),
        },
    }


def single_stage1_record(image_path: Path, image_bgr: np.ndarray, seg_items: list[dict], pose_items: list[dict], stage1_json: Path) -> dict:
    return {
        "type": "image",
        "image": str(image_path),
        "width": int(image_bgr.shape[1]),
        "height": int(image_bgr.shape[0]),
        "segmentation": seg_items,
        "pose": pose_items,
        "_stage1_json": str(stage1_json),
    }


def print_result(result: dict[str, Any], output_path: Path) -> None:
    print(f"status: {result.get('status')}")
    if result.get("status") == "ok":
        print(f"grasp_solution_mode: {result.get('grasp_solution_mode')}")
        best_pose = result.get("best_grasp_pose_base")
        if isinstance(best_pose, dict):
            candidates = result.get("window_constrained_grasp_candidates") or []
            print(f"window_constrained_grasp_candidates.count: {len(candidates)}")
            print(f"best_grasp_pose_base.xyzrpy_m_rad: {best_pose.get('xyzrpy_m_rad')}")
            print(f"best_grasp_pose_base.xyzrpy_m_deg: {best_pose.get('xyzrpy_m_deg')}")
            print(f"best_grasp_pose_base.score_visual_geometry: {best_pose.get('score_visual_geometry')}")
        else:
            grasp_pose_base = result.get("grasp_pose_base") or {}
            print(f"grasp_pose_base.robot_pose_xyzrpy_m_rad: {grasp_pose_base.get('robot_pose_xyzrpy_m_rad')}")
            print(f"grasp_pose_base.robot_pose_xyzrpy_m_deg: {grasp_pose_base.get('robot_pose_xyzrpy_m_deg')}")
        axis = result.get("tail_to_head_axis_base") or {}
        print(f"grasp_point_base_m: {result.get('grasp_point_base_m')}")
        print(f"tail_to_head_axis_base.tail_point_m: {axis.get('tail_point_m')}")
        print(f"tail_to_head_axis_base.head_point_m: {axis.get('head_point_m')}")
        print(f"grasp_pose_base.role: {result.get('grasp_pose_base_role')}")
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


def save_base_view(result: dict[str, Any], json_path: Path) -> Path:
    view_args = argparse.Namespace(
        json=json_path,
        axis_length=0.1,
        model_length=0.085,
        model_width=0.055,
        model_thickness=0.055,
    )
    output_path = default_base_view_path(json_path)
    write_html(output_path, render_html(build_view_data(result, view_args)))
    return output_path


def run(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_json_path(args.output_dir, args.rgb)
    args.dataset = args.d2rgb.parent.parent
    window_requested = window_constraint_requested(args)

    if window_requested:
        try:
            window_inputs = resolve_window_inputs(args.window_config, args.window_corners_base, args.window_margin_m)
            build_window_geometry(window_inputs.corners, window_inputs.margin_m)
        except WindowGraspError as exc:
            result = make_failure(args, exc.reason, [str(exc)])
            if exc.details:
                result["window_error"] = exc.details
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

    image = cv2.imread(str(args.rgb))
    if image is None:
        result = make_failure(args, "rgb_unreadable")
        write_json(json_path, result)
        return result, json_path

    depth_probe = cv2.imread(str(args.d2rgb), cv2.IMREAD_UNCHANGED)
    if depth_probe is None:
        result = make_failure(args, "d2rgb_unreadable")
        write_json(json_path, result)
        return result, json_path

    stage1_dir = args.output_dir / "stage1_jsons"
    overlay_dir = args.output_dir / "overlays"
    ply_dir = args.output_dir / "ply"
    stage1_json = stage1_dir / f"{args.rgb.stem}.json"

    camera = load_camera(args.camera_config)
    t_end_camera = load_hand_eye_matrix(args.hand_eye_config)
    t_base_end = robot_pose_to_matrix(args.robot_pose)

    seg_model = YOLO(str(args.seg_weights))
    pose_model = YOLO(str(args.pose_weights))
    seg_items, pose_items = run_models(image, seg_model, pose_model, args)

    record = single_stage1_record(args.rgb, image, seg_items, pose_items, stage1_json)
    write_json(stage1_json, record)
    if args.save_overlay:
        overlay_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(overlay_dir / f"{args.rgb.stem}_stage1.jpg"), draw_stage1_overlay(image, seg_items, pose_items))

    raw_id = raw_id_from_image(str(args.rgb)) or args.rgb.stem
    manifest = {raw_id: args.d2rgb}
    result, mask, points, rotation, _head_xy, _tail_xy = estimate_record(record, camera, manifest, args)

    if result.get("status") == "ok":
        result = convert_camera_grasp_to_base(
            result,
            t_base_end,
            t_end_camera,
            args.hand_eye_config,
            args.robot_config,
        )
        if window_requested:
            try:
                result = add_window_candidates(result, args.window_config, args.window_corners_base, args.window_margin_m)
            except WindowGraspError as exc:
                result["status"] = "failed"
                result["reason"] = exc.reason
                result.pop("best_grasp_pose_base", None)
                result.pop("grasp_point_base_m", None)
                result.pop("tail_to_head_axis_base", None)
                result.setdefault("warnings", []).append(str(exc))
                if exc.details:
                    result["window_error"] = exc.details
        else:
            try:
                result = attach_direct_visual_grasp(result)
            except WindowGraspError as exc:
                result["status"] = "failed"
                result["reason"] = exc.reason
                result.setdefault("warnings", []).append(str(exc))
            if margin_ignored_without_window(args):
                result.setdefault("warnings", []).append("window_margin_ignored_without_window_geometry")
        if args.save_overlay and mask is not None and rotation is not None:
            center = np.asarray(result["grasp_pose_camera"]["translation_m"], dtype=np.float32)
            # 相机坐标系下的抓取轴：X=尾→头, Y=夹爪闭合, Z=接近方向
            draw_3d_overlay(record, mask, center, rotation, camera,
                            overlay_dir / f"{args.rgb.stem}_grasp3d.jpg",
                            args.axis_scale, args.axis_thickness)
            # 基坐标系的世界轴 (robot base frame +X/+Y/+Z) 投影到相机图像
            # R_camera_base = inv(R_base_end @ R_end_camera) 将基系向量变换到相机系
            R_base_camera = (t_base_end @ t_end_camera)[:3, :3]
            R_camera_base = R_base_camera.T.astype(np.float32)
            draw_3d_overlay(record, mask, center, R_camera_base, camera,
                            overlay_dir / f"{args.rgb.stem}_grasp3d_world.jpg",
                            args.axis_scale, args.axis_thickness)
        if args.save_ply and points is not None:
            save_ply(points, ply_dir / f"{args.rgb.stem}_points.ply")

    result.setdefault("input", {})
    result["input"].update(
        {
            "image": str(args.rgb),
            "d2rgb": str(args.d2rgb),
            "robot_pose_xyzrpy_m_rad": [float(v) for v in args.robot_pose],
            "window_config": None if args.window_config is None else str(args.window_config),
            "window_corners_base_provided": args.window_corners_base is not None,
            "window_constraint_enabled": window_requested,
            "window_margin_m": None if args.window_margin_m is None else float(args.window_margin_m),
        }
    )
    if margin_ignored_without_window(args):
        warnings = result.setdefault("warnings", [])
        if "window_margin_ignored_without_window_geometry" not in warnings:
            warnings.append("window_margin_ignored_without_window_geometry")
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
            "scope": "rgbd_input_to_base_6d_pose",
            "includes_model_loading": True,
            "includes_debug_artifact_writes": bool(args.save_overlay or args.save_ply or args.save_base_view),
        }
        write_json(json_path, result)
        print_result(result, json_path)
        raise SystemExit(1) from exc

    elapsed = time.perf_counter() - start_time
    result["timing"] = {
        "single_end_to_end_s": round(float(elapsed), 6),
        "single_end_to_end_ms": round(float(elapsed * 1000.0), 3),
        "scope": "rgbd_input_to_base_6d_pose",
        "includes_model_loading": True,
        "includes_debug_artifact_writes": bool(args.save_overlay or args.save_ply or args.save_base_view),
    }
    write_json(json_path, result)
    if args.save_base_view and result.get("status") == "ok":
        base_view_path = save_base_view(result, json_path)
        print(f"base_view_html: {base_view_path}")
    print_result(result, json_path)
    if result.get("status") != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
