#!/usr/bin/env python3
"""Run end-to-end plug 6D grasp pose inference on rgbd_test/color + rgbd_test/D2RGB datasets."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from plug_vg.config import (
    DEFAULT_CAMERA,
    DEFAULT_POSE_WEIGHTS,
    DEFAULT_SEG_WEIGHTS,
    ROOT,
    load_camera,
)
from plug_vg.geometry import draw_overlay as draw_3d_overlay, save_ply
from plug_vg.grasp_pose import estimate_record
from plug_vg.io import collect_sources, output_stem, raw_id_from_image, write_json
from plug_vg.robot_transform import (
    RobotPoseProvider,
    convert_camera_grasp_to_base,
    load_hand_eye_matrix,
    robot_pose_to_matrix,
)
from plug_vg.vision import draw_overlay as draw_stage1_overlay, run_models
from infer import (
    YOLO,
)


DEFAULT_DATASET = ROOT / "plug_dataset_all_20260520" / "rgbd_test"
DEFAULT_OUTPUT = ROOT / "ultralytics" / "runs" / "plug_6d_infer"
DEFAULT_HAND_EYE = ROOT / "hand_eye_calibration" / "eye_hand_data" / "calib_20260522" / "hand_eye_result_in-hand.yaml"
DEFAULT_ROBOT_CONFIG = ROOT / "configs" / "robot" / "cs_robot.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLO stage1 inference and D2RGB-based 6D grasp pose estimation in one command."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="Dataset root containing color/ and D2RGB/.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output directory.")
    parser.add_argument("--seg-weights", type=Path, default=DEFAULT_SEG_WEIGHTS, help="Segmentation weights.")
    parser.add_argument("--pose-weights", type=Path, default=DEFAULT_POSE_WEIGHTS, help="Pose weights.")
    parser.add_argument("--camera-config", type=Path, default=DEFAULT_CAMERA, help="RGB-D camera intrinsics YAML.")
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
    parser.add_argument("--save-ply", action="store_true", help="Save filtered mask point clouds as ASCII PLY files.")
    parser.add_argument("--axis-scale", type=float, default=0.1, help="Overlay XYZ axis length in meters.")
    parser.add_argument("--axis-thickness", type=int, default=5, help="Overlay XYZ axis line thickness in pixels.")
    parser.add_argument("--no-overlay", action="store_true", help="Disable 2D overlay outputs.")
    parser.add_argument("--save-base-pose", action="store_true", help="Also save grasp poses transformed into the robot base frame.")
    parser.add_argument("--hand-eye-config", type=Path, default=DEFAULT_HAND_EYE, help="Eye-in-hand calibration YAML containing T_end_camera.")
    parser.add_argument("--robot-config", type=Path, default=DEFAULT_ROBOT_CONFIG, help="Robot config YAML.")
    parser.add_argument(
        "--robot-pose",
        type=float,
        nargs=6,
        metavar=("X", "Y", "Z", "ROLL", "PITCH", "YAW"),
        default=None,
        help="Current robot end-effector pose T_base_end as x y z roll pitch yaw in meters/radians.",
    )
    return parser.parse_args()


def stage1_record(image_path: Path, image_bgr: np.ndarray, seg_items: list[dict], pose_items: list[dict]) -> dict:
    return {
        "type": "image",
        "image": str(image_path),
        "width": int(image_bgr.shape[1]),
        "height": int(image_bgr.shape[0]),
        "segmentation": seg_items,
        "pose": pose_items,
    }


def depth_candidates(raw_id: str, color_stem: str, depth_dir: Path) -> list[Path]:
    candidates = [
        depth_dir / f"D2RGB_{raw_id}.png",
        depth_dir / f"D2RGB_{raw_id}.tif",
        depth_dir / f"D2RGB_{raw_id}.tiff",
        depth_dir / f"{raw_id}.png",
        depth_dir / f"{raw_id}.tif",
        depth_dir / f"{raw_id}.tiff",
    ]
    if color_stem != raw_id:
        candidates.extend(
            [
                depth_dir / f"D2RGB_{color_stem}.png",
                depth_dir / f"{color_stem}.png",
            ]
        )
    return candidates


def resolve_depth_path(image_path: Path, depth_dir: Path) -> tuple[str, Path]:
    raw_id = raw_id_from_image(str(image_path)) or image_path.stem
    for candidate in depth_candidates(raw_id, image_path.stem, depth_dir):
        if candidate.exists():
            return raw_id, candidate
    return raw_id, depth_dir / f"D2RGB_{raw_id}.png"


def load_base_transform_inputs(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray] | None:
    if not args.save_base_pose:
        return None
    t_end_camera = load_hand_eye_matrix(args.hand_eye_config)
    if args.robot_pose is not None:
        return robot_pose_to_matrix(args.robot_pose), t_end_camera
    provider = RobotPoseProvider(args.robot_config)
    return robot_pose_to_matrix(provider.get_current_pose()), t_end_camera


def main() -> None:
    args = parse_args()
    total_start = time.perf_counter()
    dataset = args.dataset
    color_dir = dataset / "color"
    depth_dir = dataset / "D2RGB"

    images, videos = collect_sources(color_dir)
    if videos:
        print(f"Ignoring {len(videos)} video source(s) under {color_dir}; this script estimates 6D poses for images only.")
    if not images:
        raise FileNotFoundError(f"No RGB images found under {color_dir}")
    if not depth_dir.is_dir():
        raise FileNotFoundError(f"D2RGB directory not found: {depth_dir}")

    stage1_json_dir = args.output / "stage1_jsons"
    stage1_overlay_dir = args.output / "stage1_overlays"
    json_dir = args.output / "jsons"
    overlay_dir = args.output / "overlays"
    ply_dir = args.output / "ply"
    stage1_json_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_overlay:
        stage1_overlay_dir.mkdir(parents=True, exist_ok=True)
        overlay_dir.mkdir(parents=True, exist_ok=True)

    camera = load_camera(args.camera_config)
    base_transform_inputs = load_base_transform_inputs(args)
    seg_model = YOLO(str(args.seg_weights))
    pose_model = YOLO(str(args.pose_weights))

    stage1_summary = []
    summary = []
    sample_timings = []
    for index, image_path in enumerate(images):
        sample_start = time.perf_counter()
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Skipping unreadable RGB image: {image_path}")
            continue

        raw_id, depth_path = resolve_depth_path(image_path, depth_dir)
        manifest = {raw_id: depth_path}
        seg_items, pose_items = run_models(image, seg_model, pose_model, args)
        record = stage1_record(image_path, image, seg_items, pose_items)
        record["_stage1_json"] = str(stage1_json_dir / f"{image_path.stem}.json")

        write_json(stage1_json_dir / f"{image_path.stem}.json", record)
        stage1_summary.append(record)
        if not args.no_overlay:
            stage1_overlay = draw_stage1_overlay(image, seg_items, pose_items)
            cv2.imwrite(str(stage1_overlay_dir / f"{image_path.stem}.jpg"), stage1_overlay)

        result, mask, points, rotation, _head_xy, _tail_xy = estimate_record(record, camera, manifest, args)
        if base_transform_inputs is not None and result.get("status") == "ok":
            t_base_end, t_end_camera = base_transform_inputs
            result = convert_camera_grasp_to_base(
                result,
                t_base_end,
                t_end_camera,
                args.hand_eye_config,
                args.robot_config,
            )
        stem = output_stem(record, index)

        if result.get("status") == "ok" and mask is not None and rotation is not None:
            center = np.asarray(result["grasp_pose_camera"]["translation_m"], dtype=np.float32)
            if not args.no_overlay:
                draw_3d_overlay(record, mask, center, rotation, camera, overlay_dir / f"{stem}_grasp3d.jpg", args.axis_scale, args.axis_thickness)
            if args.save_ply and points is not None:
                save_ply(points, ply_dir / f"{stem}_points.ply")

        sample_elapsed = time.perf_counter() - sample_start
        sample_timings.append(sample_elapsed)
        result["timing"] = {
            "sample_end_to_end_s": round(float(sample_elapsed), 6),
            "sample_end_to_end_ms": round(float(sample_elapsed * 1000.0), 3),
            "scope": "rgbd_input_to_base_6d_pose" if base_transform_inputs is not None else "rgbd_input_to_camera_6d_pose",
            "includes_model_loading": False,
            "includes_debug_artifact_writes": True,
        }
        write_json(json_dir / f"{stem}_3d.json", result)
        summary.append(result)

    write_json(stage1_json_dir / "summary.json", stage1_summary)
    write_json(json_dir / "summary.json", summary)

    ok_count = sum(1 for item in summary if item.get("status") == "ok")
    total_elapsed = time.perf_counter() - total_start
    processed_count = len(summary)
    timing_summary = {
        "total_end_to_end_s": round(float(total_elapsed), 6),
        "total_end_to_end_ms": round(float(total_elapsed * 1000.0), 3),
        "processed_images": processed_count,
        "ok_images": ok_count,
        "failed_images": processed_count - ok_count,
        "average_total_per_processed_image_s": None
        if processed_count == 0
        else round(float(total_elapsed / processed_count), 6),
        "average_total_per_processed_image_ms": None
        if processed_count == 0
        else round(float(total_elapsed * 1000.0 / processed_count), 3),
        "average_sample_pipeline_s": None
        if not sample_timings
        else round(float(sum(sample_timings) / len(sample_timings)), 6),
        "average_sample_pipeline_ms": None
        if not sample_timings
        else round(float(sum(sample_timings) * 1000.0 / len(sample_timings)), 3),
        "scope": "dataset_run_including_setup_model_loading_and_output_summary",
        "sample_scope": "rgbd_input_to_base_6d_pose" if base_transform_inputs is not None else "rgbd_input_to_camera_6d_pose",
    }
    write_json(json_dir / "timing_summary.json", timing_summary)
    print(
        f"Processed {len(summary)} image source(s): {ok_count} ok, "
        f"{len(summary) - ok_count} failed. Results saved to {args.output}"
    )
    print(
        "Timing: "
        f"total = {timing_summary['total_end_to_end_s']} (s), "
        f"average per image = {timing_summary['average_total_per_processed_image_s']} (s), "
        f"average pipeline per image = {timing_summary['average_sample_pipeline_s']} (s)"
    )


if __name__ == "__main__":
    main()
