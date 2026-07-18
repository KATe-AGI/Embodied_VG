#!/usr/bin/env python3
"""Run single-frame plug 6D grasp inference with visible-mask CAD registration."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
ULTRALYTICS_DIR = ROOT / "ultralytics"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ULTRALYTICS_DIR) not in sys.path:
    sys.path.insert(0, str(ULTRALYTICS_DIR))

from ultralytics import YOLO  # noqa: E402

from plug_vg.config import DEFAULT_CAMERA, DEFAULT_SEG_WEIGHTS, load_camera  # noqa: E402
from plug_vg.geometry import project_point_float, rotation_to_quaternion_xyzw  # noqa: E402
from plug_vg.grasp_model import (  # noqa: E402
    DEFAULT_GRASP_MODEL_CONFIG,
    axis_from_semantic_points,
    load_grasp_model,
    semantic_points_camera,
    transform_semantic_points_to_base,
)
from plug_vg.io import read_depth_raw, write_json  # noqa: E402
from plug_vg.model_registration import register_visible_points  # noqa: E402
from plug_vg.registration_review import write_interactive_review_html, write_registration_comparison_ply  # noqa: E402
from plug_vg.robot_transform import convert_camera_grasp_to_base, load_hand_eye_matrix, robot_pose_to_matrix, round_list  # noqa: E402
from plug_vg.visible_points import (  # noqa: E402
    extract_visible_points_from_mask,
    save_mask_overlay,
    save_visible_points_ply,
    synthesize_scene_point_cloud,
)
from plug_vg.vision import serialize_seg  # noqa: E402



r'''
# conda activate embodiedvg

# Ubuntu / bash
python infer_6d_single.py \
  --rgb test_20260701/20260701_155018_359_color.png \
  --d2rgb test_20260701/20260701_155018_359_d2rgb.npy \
  --robot-pose -0.712547 0.000064 0.581025 -2.279 0.216 1.488 \
  --output-dir output/plug_6d_single \
  --save-ply \
  --save-review

# Windows PowerShell
python .\infer_6d_single.py `
  --rgb .\test_20260701\20260701_155018_359_color.png `
  --d2rgb .\test_20260701\20260701_155018_359_d2rgb.npy `
  --robot-pose -0.712547 0.000064 0.581025 -2.279 0.216 1.488 `
  --output-dir .\output\plug_6d_single `
  --save-ply `
  --save-review
'''

DEFAULT_OUTPUT = ROOT / "ultralytics" / "runs" / "plug_6d_single"
DEFAULT_HAND_EYE = ROOT / "hand_eye_calibration" / "eye_hand_data" / "calib_20260618" / "hand_eye_result_in-hand.yaml"
DEFAULT_ROBOT_CONFIG = ROOT / "configs" / "robot" / "cs_robot.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb", type=Path, required=True, help="RGB image path.")
    parser.add_argument("--d2rgb", type=Path, required=True, help="Registered D2RGB depth PNG/NPY path.")
    parser.add_argument(
        "--robot-pose",
        type=float,
        nargs=6,
        required=True,
        metavar=("X", "Y", "Z", "ROLL", "PITCH", "YAW"),
        help="Current robot end-effector pose T_base_end as x y z roll pitch yaw in meters/radians.",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for JSON and debug artifacts.")
    parser.add_argument("--seg-weights", type=Path, default=DEFAULT_SEG_WEIGHTS, help="Visible-plug segmentation weights.")
    parser.add_argument("--model-config", type=Path, default=DEFAULT_GRASP_MODEL_CONFIG, help="Grasp-frame CAD model YAML.")
    parser.add_argument("--camera-config", type=Path, default=DEFAULT_CAMERA, help="RGB-D camera intrinsics YAML.")
    parser.add_argument("--hand-eye-config", type=Path, default=DEFAULT_HAND_EYE, help="Eye-in-hand calibration YAML containing T_end_camera.")
    parser.add_argument("--robot-config", type=Path, default=DEFAULT_ROBOT_CONFIG, help="Robot config YAML.")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.7, help="YOLO IoU threshold.")
    parser.add_argument("--device", default=None, help="CUDA device, e.g. 0, or cpu.")
    parser.add_argument("--max-det", type=int, default=10, help="Maximum YOLO detections.")
    parser.add_argument("--min-depth", type=float, default=0.1, help="Minimum valid depth in meters.")
    parser.add_argument("--max-depth", type=float, default=1.0, help="Maximum valid depth in meters.")
    parser.add_argument("--min-visible-points", type=int, default=200, help="Minimum visible D2RGB mask point count.")
    parser.add_argument("--voxel-size", type=float, default=0.004, help="Voxel size in meters for extraction/registration.")
    parser.add_argument("--outlier-nb-neighbors", type=int, default=20, help="Open3D statistical outlier neighbor count.")
    parser.add_argument("--outlier-std-ratio", type=float, default=2.0, help="Open3D statistical outlier std ratio.")
    parser.add_argument("--icp-threshold", type=float, default=0.015, help="ICP correspondence threshold in meters.")
    parser.add_argument("--icp-iterations", type=int, default=100, help="ICP max iterations.")
    parser.add_argument("--max-model-points", type=int, default=12000, help="Max CAD points before registration sampling.")
    parser.add_argument("--max-scene-points", type=int, default=12000, help="Max visible points before registration sampling.")
    parser.add_argument("--min-registration-fitness", type=float, default=0.35, help="Minimum ICP fitness for ok status.")
    parser.add_argument("--max-inlier-rmse", type=float, default=0.012, help="Maximum ICP inlier RMSE in meters for ok status.")
    parser.add_argument("--seed", type=int, default=7, help="Deterministic sampling seed.")
    parser.add_argument("--save-overlay", action="store_true", help="Save visible mask overlay.")
    parser.add_argument("--save-ply", action="store_true", help="Save visible plug point cloud PLY.")
    parser.add_argument("--save-review", action="store_true", help="Save interactive full-scene/CAD registration review HTML.")
    return parser.parse_args()


def output_json_path(output_dir: Path, rgb_path: Path) -> Path:
    return output_dir / f"{rgb_path.stem}_6d_base.json"


def choose_segmentation(seg_items: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not seg_items:
        return None
    return max(seg_items, key=lambda item: -1.0 if item.get("confidence") is None else float(item["confidence"]))


def run_segmentation(image_bgr: np.ndarray, model: YOLO, args: argparse.Namespace) -> list[dict[str, Any]]:
    result = model.predict(
        source=image_bgr,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        max_det=args.max_det,
        verbose=False,
    )[0]
    return serialize_seg(result)


def input_summary(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "rgb": str(args.rgb),
        "d2rgb": str(args.d2rgb),
        "seg_weights": str(args.seg_weights),
        "model_config": str(args.model_config),
        "robot_pose_xyzrpy_m_rad": [float(v) for v in args.robot_pose],
    }


def make_failure(args: argparse.Namespace, reason: str, warnings: list[str] | None = None) -> dict[str, Any]:
    return {
        "status": "failed",
        "reason": reason,
        "warnings": list(warnings or []),
        "input": input_summary(args),
    }


def _segmentation_summary(item: dict[str, Any] | None) -> dict[str, Any]:
    if item is None:
        return {"status": "failed", "reason": "segmentation_missing"}
    return {
        "status": "ok",
        "class_id": item.get("class_id"),
        "class_name": item.get("class_name"),
        "confidence": item.get("confidence"),
        "polygon_points": len(item.get("polygon") or []),
        "bbox_xyxy": item.get("bbox_xyxy"),
    }


def _visible_summary(extraction) -> dict[str, Any]:
    return {
        "status": extraction.status,
        "reason": extraction.reason,
        **extraction.quality,
    }


def _camera_pose_from_transform(t_camera_grasp: np.ndarray) -> dict[str, Any]:
    rotation = np.asarray(t_camera_grasp[:3, :3], dtype=np.float64)
    translation = np.asarray(t_camera_grasp[:3, 3], dtype=np.float64)
    return {
        "translation_m": round_list(translation),
        "rotation_matrix": round_list(rotation),
        "quaternion_xyzw": rotation_to_quaternion_xyzw(rotation),
    }


def _save_rgb_projection(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    semantic_camera_points: dict[str, list[float]],
    camera: dict[str, Any],
    output_path: Path,
) -> None:
    canvas = image_bgr.copy()
    if mask.shape[:2] != canvas.shape[:2]:
        mask = cv2.resize(mask, (canvas.shape[1], canvas.shape[0]), interpolation=cv2.INTER_NEAREST)
    layer = canvas.copy()
    layer[mask > 0] = (40, 180, 60)
    canvas = cv2.addWeighted(layer, 0.30, canvas, 0.70, 0)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, contours, -1, (20, 220, 80), 2)

    colors = {
        "tail_center": (255, 80, 40),
        "grasp_center": (255, 255, 255),
        "head_center": (30, 30, 255),
    }
    labels = {
        "tail_center": "tail",
        "grasp_center": "grasp",
        "head_center": "head",
    }
    projected: dict[str, tuple[int, int]] = {}
    for name in ("tail_center", "grasp_center", "head_center"):
        point = np.asarray(semantic_camera_points[f"{name}_camera_m"], dtype=np.float64)
        pixel = project_point_float(point, camera)
        if pixel is None:
            continue
        u, v = int(round(float(pixel[0]))), int(round(float(pixel[1])))
        if 0 <= u < canvas.shape[1] and 0 <= v < canvas.shape[0]:
            projected[name] = (u, v)

    if "tail_center" in projected and "head_center" in projected:
        cv2.arrowedLine(canvas, projected["tail_center"], projected["head_center"], (0, 0, 255), 4, tipLength=0.12)
    for name, pt in projected.items():
        color = colors[name]
        cv2.circle(canvas, pt, 10, (0, 0, 0), -1)
        cv2.circle(canvas, pt, 7, color, -1)
        cv2.putText(canvas, labels[name], (pt[0] + 12, pt[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 3, cv2.LINE_AA)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)


def _save_review(
    args: argparse.Namespace,
    html_path: Path,
    image_bgr: np.ndarray,
    depth_raw: np.ndarray,
    visible_points: np.ndarray,
    model_points: np.ndarray,
    t_camera_grasp: np.ndarray | None,
    semantic_camera_points: dict[str, list[float]] | None,
    status: str,
    reason: str | None,
    quality: dict[str, Any],
    camera: dict[str, Any],
) -> None:
    scene_points, scene_colors = synthesize_scene_point_cloud(
        image_bgr,
        depth_raw,
        camera,
        args.min_depth,
        args.max_depth,
    )
    write_interactive_review_html(
        html_path,
        scene_points,
        scene_colors,
        visible_points,
        model_points,
        t_camera_grasp,
        semantic_camera_points,
        status,
        reason,
        quality,
        args.seed,
        camera,
    )


def _register_points(args: argparse.Namespace, model_points: np.ndarray, scene_points: np.ndarray):
    return register_visible_points(
        model_points,
        scene_points,
        voxel_size_m=args.voxel_size,
        outlier_nb_neighbors=args.outlier_nb_neighbors,
        outlier_std_ratio=args.outlier_std_ratio,
        icp_threshold_m=args.icp_threshold,
        icp_iterations=args.icp_iterations,
        max_model_points=args.max_model_points,
        max_scene_points=args.max_scene_points,
        min_registration_fitness=args.min_registration_fitness,
        max_inlier_rmse_m=args.max_inlier_rmse,
        seed=args.seed,
    )


def print_result(result: dict[str, Any], output_path: Path) -> None:
    print(f"status: {result.get('status')}")
    if result.get("status") == "ok":
        grasp_pose_base = result.get("grasp_pose_base") or {}
        print(f"grasp_pose_base.robot_pose_xyzrpy_m_rad: {grasp_pose_base.get('robot_pose_xyzrpy_m_rad')}")
        print(f"grasp_pose_base.robot_pose_xyzrpy_m_deg: {grasp_pose_base.get('robot_pose_xyzrpy_m_deg')}")
        print(f"grasp_point_base_m: {result.get('grasp_point_base_m')}")
        axis = result.get("tail_to_head_axis_base") or {}
        print(f"tail_to_head_axis_base.direction_unit: {axis.get('direction_unit')}")
    else:
        print(f"reason: {result.get('reason')}")
        quality = result.get("registration_quality") or {}
        if quality:
            print(f"registration_quality.reason: {quality.get('reason')}")
            print(f"registration_quality.fitness: {quality.get('fitness')}")
            print(f"registration_quality.inlier_rmse: {quality.get('inlier_rmse')}")
    warnings = result.get("warnings") or []
    if warnings:
        print(f"warnings: {warnings}")
    print(f"json: {output_path}")
    timing = result.get("timing") or {}
    if timing:
        core = (
            f"model load={timing.get('model_load_s')} s, "
            f"YOLO forward={timing.get('segmentation_s')} s, "
            f"point extraction={timing.get('visible_points_s')} s, "
            f"registration={timing.get('registration_s')} s, "
            f"coordinate transform={timing.get('coordinate_transform_s')} s, "
            f"other/review/I/O={timing.get('other_review_io_s')} s"
        )
        print(f"Timing: single end-to-end = {timing.get('single_end_to_end_s')} s ({core})")
        print(
            "Timing: warm online inference = "
            f"{timing.get('warm_online_inference_s')} s "
            "(YOLO forward + point extraction + registration + coordinate transform)"
        )


def _finalize_timing(timing: dict[str, float], t0: float) -> dict[str, float]:
    """Add one-shot and model-resident timing summaries."""

    for key in ("model_load_s", "segmentation_s", "visible_points_s", "registration_s", "coordinate_transform_s"):
        timing.setdefault(key, 0.0)
    warm_keys = ["segmentation_s", "visible_points_s", "registration_s", "coordinate_transform_s"]
    warm = sum(float(timing[key]) for key in warm_keys)
    total = time.perf_counter() - t0
    measured = float(timing["model_load_s"]) + warm
    timing["warm_online_inference_s"] = round(warm, 6)
    timing["other_review_io_s"] = round(max(0.0, total - measured), 6)
    timing["single_end_to_end_s"] = round(total, 6)
    return timing


def run(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    t0 = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_json_path(args.output_dir, args.rgb)
    artifacts: dict[str, str] = {}

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
    depth = read_depth_raw(args.d2rgb)
    if depth is None:
        result = make_failure(args, "d2rgb_unreadable")
        write_json(json_path, result)
        return result, json_path

    camera = load_camera(args.camera_config)
    t_end_camera = load_hand_eye_matrix(args.hand_eye_config)
    t_base_end = robot_pose_to_matrix(args.robot_pose)
    t_base_camera = t_base_end @ t_end_camera

    stage_t0 = time.perf_counter()
    seg_model = YOLO(str(args.seg_weights))
    timing = {"model_load_s": round(time.perf_counter() - stage_t0, 6)}
    stage_t0 = time.perf_counter()
    seg_items = run_segmentation(image, seg_model, args)
    seg_item = choose_segmentation(seg_items)
    timing["segmentation_s"] = round(time.perf_counter() - stage_t0, 6)
    stage1_segmentation = _segmentation_summary(seg_item)

    if seg_item is None:
        result = make_failure(args, "visible_segmentation_missing")
        result.update(
            {
                "stage1_segmentation": stage1_segmentation,
                "artifacts": artifacts,
                "timing": _finalize_timing(timing, t0),
            }
        )
        write_json(json_path, result)
        return result, json_path

    polygon = seg_item.get("polygon_xy") or seg_item.get("polygon")
    stage_t0 = time.perf_counter()
    extraction = extract_visible_points_from_mask(
        image,
        depth,
        camera,
        polygon,
        min_depth_m=args.min_depth,
        max_depth_m=args.max_depth,
        voxel_size_m=args.voxel_size,
        min_points=args.min_visible_points,
    )
    timing["visible_points_s"] = round(time.perf_counter() - stage_t0, 6)
    warnings = list(extraction.warnings)
    if args.save_overlay or args.save_review:
        overlay_path = args.output_dir / f"{args.rgb.stem}_visible_mask.jpg"
        save_mask_overlay(image, extraction.mask, overlay_path)
        artifacts["visible_mask"] = str(overlay_path)
    if args.save_ply or args.save_review:
        ply_path = args.output_dir / f"{args.rgb.stem}_visible_points.ply"
        save_visible_points_ply(extraction.visible_points_camera_m, ply_path)
        artifacts["visible_points_ply"] = str(ply_path)

    if extraction.status != "ok":
        result = make_failure(args, extraction.reason or "visible_point_extraction_failed", warnings)
        result.update({
            "stage1_segmentation": stage1_segmentation,
            "visible_point_cloud": _visible_summary(extraction),
            "artifacts": artifacts,
            "timing": _finalize_timing(timing, t0),
        })
        write_json(json_path, result)
        return result, json_path

    model = load_grasp_model(args.model_config)
    stage_t0 = time.perf_counter()
    reg_status, t_camera_grasp, registration_quality = _register_points(
        args, model.points_grasp_m, extraction.visible_points_camera_m
    )
    timing["registration_s"] = round(time.perf_counter() - stage_t0, 6)

    semantic_camera_points = semantic_points_camera(model.config, t_camera_grasp) if t_camera_grasp is not None else None
    if args.save_review:
        if semantic_camera_points is not None:
            projection_path = args.output_dir / f"{args.rgb.stem}_rgb_projection.jpg"
            _save_rgb_projection(image, extraction.mask, semantic_camera_points, camera, projection_path)
            artifacts["rgb_projection"] = str(projection_path)
        review_path = args.output_dir / f"{args.rgb.stem}_registration_review.html"
        _save_review(
            args,
            review_path,
            image,
            depth,
            extraction.visible_points_camera_m,
            model.points_grasp_m,
            t_camera_grasp,
            semantic_camera_points,
            reg_status,
            registration_quality.get("reason"),
            registration_quality,
            camera,
        )
        artifacts["registration_review_html"] = str(review_path)
        comparison_path = args.output_dir / f"{args.rgb.stem}_point_cloud_comparison.ply"
        write_registration_comparison_ply(
            comparison_path,
            extraction.visible_points_camera_m,
            model.points_grasp_m,
            t_camera_grasp,
        )
        artifacts["point_cloud_comparison_ply"] = str(comparison_path)

    common = {
        "input": input_summary(args),
        "stage1_segmentation": stage1_segmentation,
        "visible_point_cloud": _visible_summary(extraction),
        "registration_quality": registration_quality,
        "artifacts": artifacts,
        "warnings": warnings,
        "timing": timing,
    }
    if reg_status != "ok" or t_camera_grasp is None or semantic_camera_points is None:
        _finalize_timing(timing, t0)
        result = {
            "status": reg_status,
            "reason": registration_quality.get("reason", "registration_failed"),
            **common,
        }
        write_json(json_path, result)
        return result, json_path

    stage_t0 = time.perf_counter()
    semantic_base_points = transform_semantic_points_to_base(semantic_camera_points, t_base_camera)
    semantic_source = f"{model.config.get('model_id', 'plug')}_grasp_frame_semantic_points"
    result = {
        "status": "ok",
        "t_camera_grasp": round_list(t_camera_grasp),
        "grasp_pose_camera": _camera_pose_from_transform(t_camera_grasp),
        "semantic_points_camera": semantic_camera_points,
        "tail_to_head_axis_camera": axis_from_semantic_points(
            semantic_camera_points,
            "tail_center_camera_m",
            "head_center_camera_m",
            semantic_source,
        ),
        "semantic_points_base": semantic_base_points,
        "grasp_point_base_m": semantic_base_points["grasp_center_base_m"],
        "tail_to_head_axis_base": axis_from_semantic_points(
            semantic_base_points,
            "tail_center_base_m",
            "head_center_base_m",
            semantic_source,
        ),
        **common,
    }
    result = convert_camera_grasp_to_base(result, t_base_end, t_end_camera, args.hand_eye_config, args.robot_config)
    timing["coordinate_transform_s"] = round(time.perf_counter() - stage_t0, 6)
    _finalize_timing(timing, t0)
    write_json(json_path, result)
    return result, json_path


def main() -> int:
    args = parse_args()
    result, json_path = run(args)
    print_result(result, json_path)
    return 0 if result.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
