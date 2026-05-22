#!/usr/bin/env python3
"""Estimate plug grasp 6D poses in the RGB camera frame from stage1 JSON and D2RGB depth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from plug_vg.config import (
    DEFAULT_CAMERA,
    DEFAULT_MANIFEST,
    GRASP_REGION_LENGTH_M,
    GRASP_REGION_THICKNESS_M,
    GRASP_REGION_WIDTH_M,
    HEAD_TAIL_DISTANCE_M,
    RGBD_TEST,
    load_camera,
)
from plug_vg.geometry import (
    build_rotation,
    depth_to_points,
    derive_grasp_target_2d,
    draw_overlay,
    fit_plane_normal,
    grasp_center_from_surface,
    keypoint_3d,
    normalize,
    pca_axes,
    polygon_mask,
    project_point,
    robust_depth_filter,
    robust_extent,
    rotation_to_quaternion_xyzw,
    save_ply,
)
from plug_vg.grasp_pose import estimate_record, make_failure, quality_score, resolve_d2rgb_path
from plug_vg.io import collect_stage1_records, load_manifest, output_stem, raw_id_from_image, write_json


DATASET = RGBD_TEST
DEFAULT_STAGE1 = Path(__file__).resolve().parent / "ultralytics" / "runs" / "plug_stage1_infer" / "jsons"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "ultralytics" / "runs" / "plug_grasp_3d"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1", type=Path, default=DEFAULT_STAGE1, help="Stage1 JSON file, summary JSON, or JSON directory.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output directory for 3D JSON and visualizations.")
    parser.add_argument("--dataset", type=Path, default=DATASET, help="Dataset root containing color/, D2RGB/, and meta files.")
    parser.add_argument("--camera-config", type=Path, default=DEFAULT_CAMERA, help="RGB-D camera intrinsics YAML.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="Frame manifest CSV.")
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
    parser.add_argument("--no-overlay", action="store_true", help="Disable 2D overlay output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    camera = load_camera(args.camera_config)
    manifest = load_manifest(args.manifest, args.dataset)
    records = collect_stage1_records(args.stage1)

    json_dir = args.output / "jsons"
    overlay_dir = args.output / "overlays"
    ply_dir = args.output / "ply"
    json_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for index, record in enumerate(records):
        stem = output_stem(record, index)
        result, mask, points, rotation, _head_xy, _tail_xy = estimate_record(record, camera, manifest, args)
        write_json(json_dir / f"{stem}_3d.json", result)
        summary.append(result)

        if result.get("status") == "ok" and mask is not None and rotation is not None:
            center = np.asarray(result["grasp_pose_camera"]["translation_m"], dtype=np.float32)
            if not args.no_overlay:
                draw_overlay(record, mask, center, rotation, camera, overlay_dir / f"{stem}_grasp3d.jpg", args.axis_scale, args.axis_thickness)
            if args.save_ply and points is not None:
                save_ply(points, ply_dir / f"{stem}_points.ply")

    with (json_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    ok_count = sum(1 for item in summary if item.get("status") == "ok")
    print(f"Processed {len(summary)} stage1 record(s): {ok_count} ok, {len(summary) - ok_count} failed. Results saved to {args.output}")


if __name__ == "__main__":
    main()
