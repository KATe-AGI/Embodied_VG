#!/usr/bin/env python3
"""Generate camera-frame registration candidates for the reviewed unstable plugs."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ultralytics"))

from infer_6d_single import _save_rgb_projection, choose_segmentation, run_segmentation  # noqa: E402
from plug_vg.config import DEFAULT_SEG_WEIGHTS, load_camera  # noqa: E402
from plug_vg.geometry import polygon_mask  # noqa: E402
from plug_vg.grasp_model import load_grasp_model, semantic_points_camera  # noqa: E402
from plug_vg.io import read_depth_raw  # noqa: E402
from plug_vg.registration_review import write_registration_comparison_ply  # noqa: E402
from plug_vg.symmetric_registration import (  # noqa: E402
    RenderContext,
    _render_candidate_cost,
    canonical_rotation_from_axis,
    optimize_profile_candidates,
)
from plug_vg.visible_points import erode_mask, extract_visible_points_from_mask  # noqa: E402
from ultralytics import YOLO  # noqa: E402


BENCHMARK = ROOT / "test_data/plug_6d_camera_benchmark_20261105"
OUTPUT = ROOT / "output/unstable_plug_camera_registrations_20261106"
PREVIOUS = ROOT / "output/manual_registration_20261030_145928"
PREVIOUS_ID = "20261030_145928_3915816_001_plug"
SEG_ARGS = SimpleNamespace(imgsz=640, conf=0.25, iou=0.7, device=None, max_det=10)


def score_pose(
    axis: np.ndarray,
    origin: np.ndarray,
    model: np.ndarray,
    observed: np.ndarray,
    context: RenderContext,
    camera: dict,
) -> dict[str, float]:
    axis = np.asarray(axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    origin = np.asarray(origin, dtype=np.float64)
    transformed = model @ canonical_rotation_from_axis(axis).T + origin
    distances = cKDTree(transformed).query(observed, workers=1)[0]
    render = _render_candidate_cost(
        model, {"axis_camera": axis, "origin_camera_m": origin}, context, camera, 0.008
    )
    return {
        "visible_iou": float(render["visible_iou"]),
        "depth_cost": float(render["depth_cost"]),
        "surface_median_mm": float(np.median(distances) * 1000),
        "surface_p75_mm": float(np.percentile(distances, 75) * 1000),
        "surface_p90_mm": float(np.percentile(distances, 90) * 1000),
        "surface_trimmed_mean_mm": float(np.mean(np.sort(distances)[:int(len(distances) * 0.75)]) * 1000),
        "surface_inlier_5mm": float(np.mean(distances <= 0.005)),
    }


def candidate_cost(item: dict) -> float:
    metrics = item["metrics"]
    return (
        metrics["surface_trimmed_mean_mm"]
        + metrics["depth_cost"]
        + 2.0 * (1.0 - metrics["visible_iou"])
        + 0.25 * item["profile_loss"]
    )


def review_flags(previous: dict, selected: dict) -> list[str]:
    flags = []
    distance_gain = previous["surface_trimmed_mean_mm"] - selected["surface_trimmed_mean_mm"]
    depth_gain = previous["depth_cost"] - selected["depth_cost"]
    if distance_gain < 0.1 and depth_gain < 0.05:
        flags.append("limited_metric_gain")
    if selected["visible_iou"] < 0.7:
        flags.append("low_visible_iou")
    if previous["visible_iou"] - selected["visible_iou"] > 0.08:
        flags.append("silhouette_drop_gt_0.08")
    return flags


def render_context(mask: np.ndarray, depth: np.ndarray, camera: dict) -> RenderContext:
    ys, xs = np.nonzero(mask)
    return RenderContext(
        target_mask=mask,
        scene_depth_m=np.asarray(depth, dtype=np.float64) * float(camera["depth_scale"]),
        image_height=mask.shape[0],
        image_width=mask.shape[1],
        target_x0=int(xs.min()),
        target_x1=int(xs.max()) + 1,
        target_y0=int(ys.min()),
        target_y1=int(ys.max()) + 1,
    )


def process(row: dict[str, str], segmenter: YOLO, model: np.ndarray, model_config: dict) -> dict:
    sample_id = row["sample_id"]
    destination = OUTPUT / sample_id
    destination.mkdir(parents=True, exist_ok=True)
    rgb_path = ROOT / row["rgb"]
    depth_path = ROOT / row["depth"]
    camera_path = ROOT / row["camera_config"]
    image = cv2.imread(str(rgb_path))
    depth = read_depth_raw(depth_path)
    if image is None or depth is None:
        raise ValueError(f"Unreadable input: {sample_id}")
    camera = load_camera(camera_path)
    seg = choose_segmentation(run_segmentation(image, segmenter, SEG_ARGS))
    if seg is None:
        raise ValueError(f"Segmentation missing: {sample_id}")
    polygon = seg.get("polygon_xy") or seg.get("polygon")
    mask = polygon_mask(polygon, depth.shape[:2]).astype(bool)
    extraction = extract_visible_points_from_mask(
        image, depth, camera, polygon, min_depth_m=0.1, max_depth_m=1.0,
        voxel_size_m=0.004, min_points=200, mask_erosion_px=5, mad_z_threshold=0.0,
    )
    if extraction.status != "ok":
        raise ValueError(f"Point extraction failed: {sample_id}: {extraction.reason}")
    observed = extraction.visible_points_camera_m
    context = render_context(mask, depth, camera)
    candidates = optimize_profile_candidates(
        model, observed, direction_count=64, axial_anchor_count=5,
        max_scene_points=len(observed), iterations=100,
    )
    # Retain two starts per direction so nearby local minima do not hide an
    # alternative head/tail orientation in the scored shortlist.
    selected = []
    counts: dict[int, int] = {}
    for candidate in candidates:
        seed = int(candidate["axis_seed_index"])
        if counts.get(seed, 0) < 2:
            selected.append(candidate)
            counts[seed] = counts.get(seed, 0) + 1
    selected = selected[:100]
    evaluated = []
    for candidate in selected:
        metrics = score_pose(
            candidate["axis_camera"], candidate["origin_camera_m"], model,
            observed, context, camera,
        )
        evaluated.append({
            "axis_camera": np.asarray(candidate["axis_camera"]).tolist(),
            "origin_camera_m": np.asarray(candidate["origin_camera_m"]).tolist(),
            "profile_loss": float(candidate["profile_loss"]),
            "axis_seed_index": int(candidate["axis_seed_index"]),
            "metrics": metrics,
        })
    profile_best = min(evaluated, key=lambda item: item["profile_loss"])
    best = min(evaluated, key=candidate_cost)
    axis = np.asarray(best["axis_camera"])
    origin = np.asarray(best["origin_camera_m"])
    transform = np.eye(4)
    transform[:3, :3] = canonical_rotation_from_axis(axis)
    transform[:3, 3] = origin
    baseline = json.loads((ROOT / row["result_json"]).read_text(encoding="utf-8"))
    baseline_transform = np.asarray(baseline["t_camera_grasp"], dtype=np.float64)
    baseline_metrics = score_pose(
        baseline_transform[:3, 0], baseline_transform[:3, 3],
        model, observed, context, camera,
    )
    alternatives = {
        "best_profile_loss": profile_best,
        "best_visible_iou": max(evaluated, key=lambda item: item["metrics"]["visible_iou"]),
        "best_depth_cost": min(evaluated, key=lambda item: item["metrics"]["depth_cost"]),
        "best_surface_distance": min(evaluated, key=lambda item: item["metrics"]["surface_trimmed_mean_mm"]),
        "best_reverse_axis": min(
            (item for item in evaluated if np.dot(item["axis_camera"], axis) < 0),
            key=lambda item: item["profile_loss"], default=None,
        ),
    }
    semantic = semantic_points_camera(model_config, transform)
    result = {
        "sample_id": sample_id,
        "status": "candidate_manual_review_pending",
        "frame": "camera_rgb",
        "camera_config": row["camera_config"],
        "input": {"rgb": row["rgb"], "depth": row["depth"]},
        "model": "plug_model/plugCAD_grasp.ply",
        "method": "full_observation_profile_multistart",
        "selection": "lowest_combined_surface_depth_silhouette_profile_cost",
        "combined_cost": candidate_cost(best),
        "t_camera_grasp": transform.tolist(),
        "axis_camera": axis.tolist(),
        "origin_camera_m": origin.tolist(),
        "semantic_points_camera": semantic,
        "observed_point_count": len(observed),
        "quality_comparison": {
            "previous": baseline_metrics,
            "profile_initial": profile_best["metrics"],
            "selected": best["metrics"],
        },
        "profile_loss": best["profile_loss"],
        "alternatives": alternatives,
        "limitations": [
            "Roll about the plug axis is assigned by canonical_rotation_from_axis; not independently estimated.",
            "A low profile loss does not by itself certify head/tail identity in occluded scenes.",
        ],
    }
    (destination / "registration.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_registration_comparison_ply(
        destination / "point_cloud_comparison.ply", observed, model, transform, semantic
    )
    _save_rgb_projection(
        image, extraction.mask, semantic, camera, destination / "rgb_rgb_projection.jpg"
    )
    return result


def save_existing_projections(manifest: dict[str, dict[str, str]], segmenter: YOLO) -> None:
    """Render the stored poses without rerunning registration."""
    summary_path = OUTPUT / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    for item in summary:
        sample_id = item["sample_id"]
        destination = OUTPUT / sample_id
        result_path = destination / "registration.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        row = manifest[sample_id]
        image = cv2.imread(str(ROOT / row["rgb"]))
        if image is None:
            raise ValueError(f"Unreadable RGB: {sample_id}")
        camera = load_camera(ROOT / result["camera_config"])
        seg = choose_segmentation(run_segmentation(image, segmenter, SEG_ARGS))
        if seg is None:
            raise ValueError(f"Segmentation missing: {sample_id}")
        polygon = seg.get("polygon_xy") or seg.get("polygon")
        mask = erode_mask(polygon_mask(polygon, image.shape[:2]), 5)
        projection_path = destination / "rgb_rgb_projection.jpg"
        _save_rgb_projection(image, mask, result["semantic_points_camera"], camera, projection_path)
        if cv2.imread(str(projection_path)) is None:
            raise ValueError(f"Projection image not written: {sample_id}")
        relative_path = str(projection_path.relative_to(ROOT))
        result.setdefault("artifacts", {})["rgb_projection"] = relative_path
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        item["rgb_projection"] = relative_path
        print(f"Saved {sample_id}/rgb_rgb_projection.jpg", flush=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projections-only", action="store_true", help="Render RGB projections from existing saved poses.")
    args = parser.parse_args()
    selection = json.loads((BENCHMARK / "review_selection.json").read_text(encoding="utf-8"))
    sample_ids = selection["unstable_sample_ids"]
    if len(sample_ids) != 20 or len(set(sample_ids)) != 20:
        raise ValueError("Expected exactly 20 distinct unstable sample IDs")
    with (BENCHMARK / "candidates.csv").open(newline="", encoding="utf-8") as stream:
        manifest = {row["sample_id"]: row for row in csv.DictReader(stream)}
    segmenter = YOLO(str(DEFAULT_SEG_WEIGHTS))
    if args.projections_only:
        save_existing_projections(manifest, segmenter)
        return
    model = load_grasp_model()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary = []
    for index, sample_id in enumerate(sample_ids, 1):
        print(f"[{index:02d}/20] {sample_id}", flush=True)
        if sample_id == PREVIOUS_ID:
            destination = OUTPUT / sample_id
            destination.mkdir(parents=True, exist_ok=True)
            for filename in ("registration.json", "point_cloud_comparison.ply"):
                shutil.copy2(PREVIOUS / filename, destination / filename)
            shutil.copy2(PREVIOUS / "rgb_projection.jpg", destination / "rgb_rgb_projection.jpg")
            result = json.loads((destination / "registration.json").read_text(encoding="utf-8"))
            quality = result["quality_comparison"]["profile_initial"]
            old = result["quality_comparison"]["previous"]
        else:
            result = process(manifest[sample_id], segmenter, model.points_grasp_m, model.config)
            quality = result["quality_comparison"]["selected"]
            old = result["quality_comparison"]["previous"]
        result["sample_id"] = sample_id
        result["quality_comparison"]["selected"] = quality
        result["review_flags"] = review_flags(old, quality)
        result.setdefault("artifacts", {})["rgb_projection"] = str(
            (OUTPUT / sample_id / "rgb_rgb_projection.jpg").relative_to(ROOT)
        )
        (OUTPUT / sample_id / "registration.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        summary.append({
            "sample_id": sample_id,
            "status": result["status"],
            "review_flags": result["review_flags"],
            "selected_surface_trimmed_mean_mm": quality["surface_trimmed_mean_mm"],
            "previous_surface_trimmed_mean_mm": old["surface_trimmed_mean_mm"],
            "selected_visible_iou": quality["visible_iou"],
            "previous_visible_iou": old["visible_iou"],
            "selected_depth_cost": quality["depth_cost"],
            "previous_depth_cost": old["depth_cost"],
            "registration_json": str((OUTPUT / sample_id / "registration.json").relative_to(ROOT)),
            "point_cloud_comparison_ply": str((OUTPUT / sample_id / "point_cloud_comparison.ply").relative_to(ROOT)),
            "rgb_projection": result["artifacts"]["rgb_projection"],
        })
        (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(
            f"    distance {old['surface_trimmed_mean_mm']:.2f} -> {quality['surface_trimmed_mean_mm']:.2f} mm; "
            f"IoU {old['visible_iou']:.3f} -> {quality['visible_iou']:.3f}; "
            f"depth {old['depth_cost']:.3f} -> {quality['depth_cost']:.3f}", flush=True,
        )


if __name__ == "__main__":
    main()
