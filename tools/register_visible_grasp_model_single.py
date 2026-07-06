#!/usr/bin/env python3
"""Validate visible-mask plug points against the grasp-frame CAD model."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ULTRALYTICS_DIR = ROOT / "ultralytics"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ULTRALYTICS_DIR) not in sys.path:
    sys.path.insert(0, str(ULTRALYTICS_DIR))

from ultralytics import YOLO  # noqa: E402

from plug_vg.config import DEFAULT_CAMERA, DEFAULT_SEG_WEIGHTS, load_camera  # noqa: E402
from plug_vg.geometry import project_point_float  # noqa: E402
from plug_vg.grasp_model import (  # noqa: E402
    DEFAULT_GRASP_MODEL_CONFIG,
    axis_from_semantic_points,
    load_grasp_model,
    semantic_points_camera,
)
from plug_vg.io import read_depth_raw, write_json  # noqa: E402
from plug_vg.model_registration import register_visible_points  # noqa: E402
from plug_vg.registration_review import read_realsense_binary_ply, write_interactive_review_html  # noqa: E402
from plug_vg.robot_transform import round_list  # noqa: E402
from plug_vg.visible_points import extract_visible_points_from_mask, save_mask_overlay, save_visible_points_ply  # noqa: E402
from plug_vg.vision import run_segmentation  # noqa: E402


DEFAULT_OUTPUT_DIR = ROOT / "ultralytics" / "runs" / "visible_grasp_registration_single"


def make_failure(args: argparse.Namespace, reason: str, warnings: list[str] | None = None) -> dict[str, Any]:
    return {
        "status": "failed",
        "reason": reason,
        "warnings": list(warnings or []),
        "input": {
            "rgb": str(args.rgb),
            "d2rgb": str(args.d2rgb),
            "scene_ply": None if args.scene_ply is None else str(args.scene_ply),
            "seg_weights": str(args.seg_weights),
            "model_config": str(args.model_config),
        },
    }


def choose_segmentation(seg_items: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not seg_items:
        return None
    return max(seg_items, key=lambda item: -1.0 if item.get("confidence") is None else float(item["confidence"]))


def _visible_summary(extraction) -> dict[str, Any]:
    return {"status": extraction.status, "reason": extraction.reason, **extraction.quality}


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


def run(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    start = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.rgb.stem.replace("_color", "")
    json_path = args.output_dir / f"{stem}_visible_grasp_registration.json"
    mask_path = args.output_dir / f"{stem}_visible_mask.jpg"
    points_path = args.output_dir / f"{stem}_visible_points.ply"
    projection_path = args.output_dir / f"{stem}_rgb_projection.jpg"
    review_html = args.output_dir / f"{stem}_registration_review.html"
    warnings: list[str] = []
    artifacts: dict[str, str] = {}

    try:
        if not args.rgb.is_file():
            result = make_failure(args, "rgb_missing", warnings)
            write_json(json_path, result)
            return result, json_path
        if not args.d2rgb.is_file():
            result = make_failure(args, "d2rgb_missing", warnings)
            write_json(json_path, result)
            return result, json_path

        image = cv2.imread(str(args.rgb))
        if image is None:
            result = make_failure(args, "rgb_unreadable", warnings)
            write_json(json_path, result)
            return result, json_path
        depth_raw = read_depth_raw(args.d2rgb)
        if depth_raw is None:
            result = make_failure(args, "d2rgb_unreadable", warnings)
            write_json(json_path, result)
            return result, json_path

        camera = load_camera(args.camera_config)
        seg_model = YOLO(str(args.seg_weights))
        seg_items = run_segmentation(image, seg_model, args)
        seg = choose_segmentation(seg_items)
        if seg is None:
            result = make_failure(args, "visible_segmentation_missing", warnings)
            result["stage1_segmentation"] = []
            write_json(json_path, result)
            return result, json_path

        extraction = extract_visible_points_from_mask(
            image,
            depth_raw,
            camera,
            seg.get("polygon_xy") or seg.get("polygon"),
            args.min_depth,
            args.max_depth,
            args.voxel_size,
            args.min_visible_points,
        )
        warnings.extend(extraction.warnings)
        save_mask_overlay(image, extraction.mask, mask_path)
        artifacts["visible_mask"] = str(mask_path)
        if len(extraction.visible_points_camera_m):
            save_visible_points_ply(extraction.visible_points_camera_m, points_path)
            artifacts["visible_points_ply"] = str(points_path)
        if extraction.status != "ok":
            result = make_failure(args, extraction.reason or "visible_point_extraction_failed", warnings)
            result["stage1_segmentation"] = seg_items
            result["visible_point_cloud"] = _visible_summary(extraction)
            result["artifacts"] = artifacts
            write_json(json_path, result)
            return result, json_path

        model = load_grasp_model(args.model_config)
        reg_status, t_camera_grasp, reg_quality = register_visible_points(
            model.points_grasp_m,
            extraction.visible_points_camera_m,
            args.voxel_size,
            args.outlier_nb_neighbors,
            args.outlier_std_ratio,
            args.icp_threshold,
            args.icp_iterations,
            args.max_model_points,
            args.max_scene_points,
            args.min_registration_fitness,
            args.max_inlier_rmse,
            args.ambiguity_fitness_margin,
            args.ambiguity_rmse_margin,
            args.seed,
        )
        semantic = semantic_points_camera(model.config, t_camera_grasp) if t_camera_grasp is not None else None

        if args.save_review:
            if semantic is not None:
                _save_rgb_projection(image, extraction.mask, semantic, camera, projection_path)
                artifacts["rgb_projection"] = str(projection_path)
            scene_points = extraction.visible_points_camera_m
            scene_colors = None
            if args.scene_ply is not None and args.scene_ply.is_file():
                scene_cloud = read_realsense_binary_ply(args.scene_ply)
                scene_points = scene_cloud.points
                scene_colors = scene_cloud.colors
            write_interactive_review_html(
                review_html,
                scene_points,
                scene_colors,
                extraction.visible_points_camera_m,
                model.points_grasp_m,
                t_camera_grasp,
                semantic,
                reg_status,
                reg_quality.get("reason"),
                reg_quality,
                args.seed,
                camera,
            )
            artifacts["registration_review_html"] = str(review_html)

        result: dict[str, Any] = {
            "status": reg_status,
            "reason": None if reg_status == "ok" else reg_quality.get("reason", f"registration_{reg_status}"),
            "warnings": warnings,
            "input": {
                "rgb": str(args.rgb),
                "d2rgb": str(args.d2rgb),
                "scene_ply": None if args.scene_ply is None else str(args.scene_ply),
                "seg_weights": str(args.seg_weights),
                "model_config": str(args.model_config),
                "model_ply": str(model.pointcloud_path),
            },
            "stage1_segmentation": seg_items,
            "visible_point_cloud": _visible_summary(extraction),
            "registration_quality": reg_quality,
            "artifacts": artifacts,
            "timing": {"visible_grasp_registration_single_s": round(float(time.perf_counter() - start), 6)},
        }
        if reg_status == "ok" and t_camera_grasp is not None and semantic is not None:
            rotation = t_camera_grasp[:3, :3]
            translation = t_camera_grasp[:3, 3]
            result.update(
                {
                    "t_camera_grasp": round_list(t_camera_grasp),
                    "grasp_pose_camera": {
                        "translation_m": round_list(translation),
                        "rotation_matrix": round_list(rotation),
                    },
                    "semantic_points_camera": semantic,
                    "tail_to_head_axis_camera": axis_from_semantic_points(
                        semantic,
                        "tail_center_camera_m",
                        "head_center_camera_m",
                        "2175B_grasp_frame_semantic_points",
                    ),
                }
            )
        write_json(json_path, result)
        return result, json_path
    except Exception as exc:
        result = make_failure(args, type(exc).__name__, [*warnings, str(exc)])
        result["timing"] = {"visible_grasp_registration_single_s": round(float(time.perf_counter() - start), 6)}
        write_json(json_path, result)
        return result, json_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb", type=Path, required=True, help="RGB image path.")
    parser.add_argument("--d2rgb", type=Path, required=True, help="Registered D2RGB depth path, PNG/TIFF/NPY.")
    parser.add_argument("--scene-ply", type=Path, default=None, help="Optional RealSense scene PLY used only for review.")
    parser.add_argument("--seg-weights", type=Path, default=DEFAULT_SEG_WEIGHTS, help="Visible plug segmentation weights.")
    parser.add_argument("--model-config", type=Path, default=DEFAULT_GRASP_MODEL_CONFIG, help="Grasp-frame model YAML config.")
    parser.add_argument("--camera-config", type=Path, default=DEFAULT_CAMERA, help="RGB-D camera intrinsics YAML.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.7, help="YOLO IoU threshold.")
    parser.add_argument("--device", default=None, help="CUDA device, e.g. 0, or cpu.")
    parser.add_argument("--max-det", type=int, default=10, help="Maximum YOLO detections.")
    parser.add_argument("--min-depth", type=float, default=0.1, help="Minimum valid depth in meters.")
    parser.add_argument("--max-depth", type=float, default=1.2, help="Maximum valid depth in meters.")
    parser.add_argument("--min-visible-points", type=int, default=200, help="Minimum visible points required.")
    parser.add_argument("--voxel-size", type=float, default=0.004, help="Voxel size in meters.")
    parser.add_argument("--outlier-nb-neighbors", type=int, default=20, help="Statistical outlier neighbors.")
    parser.add_argument("--outlier-std-ratio", type=float, default=2.0, help="Statistical outlier std ratio.")
    parser.add_argument("--icp-threshold", type=float, default=0.015, help="ICP correspondence threshold in meters.")
    parser.add_argument("--icp-iterations", type=int, default=100, help="ICP iterations per candidate.")
    parser.add_argument("--max-model-points", type=int, default=30000, help="Maximum model points used.")
    parser.add_argument("--max-scene-points", type=int, default=50000, help="Maximum visible scene points used.")
    parser.add_argument("--min-registration-fitness", type=float, default=0.35, help="Minimum accepted ICP fitness.")
    parser.add_argument("--max-inlier-rmse", type=float, default=0.012, help="Maximum accepted ICP inlier RMSE in meters.")
    parser.add_argument("--ambiguity-fitness-margin", type=float, default=0.05, help="Ambiguity threshold for candidate fitness gap.")
    parser.add_argument("--ambiguity-rmse-margin", type=float, default=0.003, help="Ambiguity threshold for candidate RMSE gap in meters.")
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed.")
    parser.add_argument("--save-review", action="store_true", help="Save interactive HTML review artifact.")
    return parser.parse_args()


def print_result(result: dict[str, Any], json_path: Path) -> None:
    print(f"status: {result.get('status')}")
    if result.get("status") == "ok":
        print(f"t_camera_grasp: {result.get('t_camera_grasp')}")
        quality = result.get("registration_quality") or {}
        print(f"fitness: {quality.get('fitness')}")
        print(f"inlier_rmse: {quality.get('inlier_rmse')}")
        semantic = result.get("semantic_points_camera") or {}
        print(f"grasp_center_camera_m: {semantic.get('grasp_center_camera_m')}")
        print(f"tail_center_camera_m: {semantic.get('tail_center_camera_m')}")
        print(f"head_center_camera_m: {semantic.get('head_center_camera_m')}")
    else:
        print(f"reason: {result.get('reason')}")
        warnings = result.get("warnings") or []
        if warnings:
            print(f"warnings: {warnings}")
    artifacts = result.get("artifacts") or {}
    for key, value in artifacts.items():
        print(f"{key}: {value}")
    print(f"json: {json_path}")


def main() -> None:
    args = parse_args()
    result, json_path = run(args)
    print_result(result, json_path)
    if result.get("status") not in {"ok", "ambiguous"}:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
