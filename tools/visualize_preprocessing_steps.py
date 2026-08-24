#!/usr/bin/env python3
"""Visualize every point-cloud preprocessing stage for one RGB-D frame."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ULTRALYTICS_DIR = ROOT / "ultralytics"
for path in (ROOT, ULTRALYTICS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ultralytics import YOLO  # noqa: E402

from plug_vg.config import DEFAULT_CAMERA, DEFAULT_SEG_WEIGHTS, load_camera  # noqa: E402
from plug_vg.geometry import polygon_mask  # noqa: E402
from plug_vg.grasp_model import DEFAULT_GRASP_MODEL_CONFIG, load_grasp_model  # noqa: E402
from plug_vg.io import read_depth_raw  # noqa: E402
from plug_vg.visible_points import _depth_to_camera_points, erode_mask, global_mad_depth_keep, voxel_downsample  # noqa: E402
from plug_vg.vision import serialize_seg  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb", type=Path, required=True)
    parser.add_argument("--d2rgb", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seg-weights", type=Path, default=DEFAULT_SEG_WEIGHTS)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_GRASP_MODEL_CONFIG)
    parser.add_argument("--camera-config", type=Path, default=DEFAULT_CAMERA)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=10)
    parser.add_argument("--min-depth", type=float, default=0.1)
    parser.add_argument("--max-depth", type=float, default=1.0)
    parser.add_argument("--mask-erosion-px", type=int, default=5)
    parser.add_argument("--mad-z-threshold", type=float, default=3.5)
    parser.add_argument("--voxel-size", type=float, default=0.004)
    parser.add_argument("--max-model-points", type=int, default=12000)
    parser.add_argument("--max-scene-points", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def save_rgb(path: Path, image_bgr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image_bgr)


def mask_overlay(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    canvas = image_bgr.copy()
    layer = canvas.copy()
    layer[mask > 0] = (40, 180, 60)
    canvas = cv2.addWeighted(layer, 0.38, canvas, 0.62, 0)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, contours, -1, (20, 220, 80), 3)
    return canvas


def save_image_pair(path: Path, before: np.ndarray, after: np.ndarray, before_title: str, after_title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    for axis, image, title in zip(axes, (before, after), (before_title, after_title)):
        axis.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axis.set_title(title)
        axis.axis("off")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def depth_rgba(depth_m: np.ndarray, include: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    normalized = np.clip((depth_m - vmin) / max(vmax - vmin, 1e-9), 0.0, 1.0)
    rgba = plt.get_cmap("turbo")(normalized)
    rgba[~include, :3] = 0.08
    rgba[~include, 3] = 1.0
    return rgba


def save_depth_pair(
    path: Path,
    depth_m: np.ndarray,
    before_keep: np.ndarray,
    after_keep: np.ndarray,
    before_title: str,
    after_title: str,
    vmin: float,
    vmax: float,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    for axis, keep, title in zip(axes, (before_keep, after_keep), (before_title, after_title)):
        axis.imshow(depth_rgba(depth_m, keep, vmin, vmax))
        axis.set_title(title)
        axis.axis("off")
    scalar = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax), cmap="turbo")
    fig.colorbar(scalar, ax=axes, shrink=0.8, label="Depth Z (m)")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def limits_for(*clouds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    nonempty = [np.asarray(points) for points in clouds if len(points)]
    joined = np.concatenate(nonempty, axis=0)
    low = np.min(joined, axis=0)
    high = np.max(joined, axis=0)
    center = (low + high) * 0.5
    radius = max(float(np.max(high - low)) * 0.58, 0.005)
    return center - radius, center + radius


def plot_cloud(axis: Any, points: np.ndarray, title: str, low: np.ndarray, high: np.ndarray) -> None:
    points = np.asarray(points, dtype=np.float64)
    if len(points):
        if len(points) > 30000:
            indices = np.linspace(0, len(points) - 1, 30000, dtype=np.int64)
            shown = points[indices]
        else:
            shown = points
        axis.scatter(shown[:, 0], shown[:, 2], -shown[:, 1], c=shown[:, 2], cmap="turbo", s=4, alpha=0.8)
    axis.set_xlim(low[0], high[0])
    axis.set_ylim(low[2], high[2])
    axis.set_zlim(-high[1], -low[1])
    axis.set_xlabel("Camera/CAD X (m)")
    axis.set_ylabel("Z / forward (m)")
    axis.set_zlabel("-Y / up (m)")
    axis.set_title(f"{title}\nN={len(points):,}")
    axis.view_init(elev=24, azim=-68)
    axis.grid(True, alpha=0.25)


def save_cloud_pair(path: Path, before: np.ndarray, after: np.ndarray, before_title: str, after_title: str) -> None:
    low, high = limits_for(before, after)
    fig = plt.figure(figsize=(16, 7), constrained_layout=True)
    before_axis = fig.add_subplot(1, 2, 1, projection="3d")
    after_axis = fig.add_subplot(1, 2, 2, projection="3d")
    plot_cloud(before_axis, before, before_title, low, high)
    plot_cloud(after_axis, after, after_title, low, high)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def save_ply(path: Path, points: np.ndarray) -> None:
    points = np.asarray(points, dtype=np.float64)
    with path.open("w", encoding="ascii") as stream:
        stream.write("ply\nformat ascii 1.0\n")
        stream.write(f"element vertex {len(points)}\n")
        stream.write("property float x\nproperty float y\nproperty float z\nend_header\n")
        for x, y, z in points:
            stream.write(f"{x:.8f} {y:.8f} {z:.8f}\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    image = cv2.imread(str(args.rgb))
    depth_raw = read_depth_raw(args.d2rgb)
    if image is None or depth_raw is None:
        raise RuntimeError("RGB or depth could not be read")
    if depth_raw.ndim == 3:
        depth_raw = depth_raw[:, :, 0]
    if image.shape[:2] != depth_raw.shape[:2]:
        raise ValueError(f"RGB/depth shape mismatch: {image.shape[:2]} vs {depth_raw.shape[:2]}")
    camera = load_camera(args.camera_config)

    model = YOLO(str(args.seg_weights))
    prediction = model.predict(
        source=image,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        max_det=args.max_det,
        verbose=False,
    )[0]
    detections = serialize_seg(prediction)
    if not detections:
        raise RuntimeError("No segmentation was detected")
    detection = detections[0]
    polygon = detection.get("polygon_xy") or detection.get("polygon")
    raw_mask = polygon_mask(polygon, depth_raw.shape[:2]).astype(np.uint8)
    depth_mask = erode_mask(raw_mask, args.mask_erosion_px)

    save_rgb(args.output_dir / "00_input_rgb.png", image)
    overlay = mask_overlay(image, raw_mask)
    save_rgb(args.output_dir / "01_segmentation_overlay.png", overlay)
    save_image_pair(
        args.output_dir / "01_segmentation_before_after.png",
        image,
        overlay,
        "Before: input RGB",
        f"After: selected YOLO mask (confidence={detection.get('confidence')})",
    )

    eroded_overlay = mask_overlay(image, depth_mask)
    save_image_pair(
        args.output_dir / "02_mask_erosion_before_after.png",
        overlay,
        eroded_overlay,
        f"Before: raw YOLO mask (N={np.count_nonzero(raw_mask):,} px)",
        f"After: {args.mask_erosion_px} px inward erosion (N={np.count_nonzero(depth_mask):,} px)",
    )

    depth_m = np.asarray(depth_raw, dtype=np.float64) * float(camera["depth_scale"])
    finite_positive = (depth_mask > 0) & np.isfinite(depth_m) & (depth_m > 0.0)
    range_valid = finite_positive & (depth_m >= args.min_depth) & (depth_m <= args.max_depth)
    masked_values = depth_m[finite_positive]
    display_min = max(args.min_depth, float(np.percentile(masked_values, 1))) if len(masked_values) else args.min_depth
    display_max = min(args.max_depth, float(np.percentile(masked_values, 99))) if len(masked_values) else args.max_depth
    save_depth_pair(
        args.output_dir / "03_depth_range_filter_before_after.png",
        depth_m,
        finite_positive,
        range_valid,
        f"Before: finite positive mask depth (N={np.count_nonzero(finite_positive):,})",
        f"After: {args.min_depth:.3f} <= Z <= {args.max_depth:.3f} m (N={np.count_nonzero(range_valid):,})",
        display_min,
        display_max,
    )

    valid_points, _valid_pixels, valid_z = _depth_to_camera_points(
        depth_mask, depth_raw, camera, args.min_depth, args.max_depth
    )
    mad_keep, mad_stats = global_mad_depth_keep(valid_z, args.mad_z_threshold)
    mad_points = valid_points[mad_keep]
    save_cloud_pair(
        args.output_dir / "04_global_mad_before_after.png",
        valid_points,
        mad_points,
        "Before: pass-through-filtered points",
        f"After: global Z MAD (threshold={args.mad_z_threshold:.1f})",
    )

    scene_voxel = voxel_downsample(mad_points, args.voxel_size)
    save_cloud_pair(
        args.output_dir / "05_scene_centroid_voxel_before_after.png",
        mad_points,
        scene_voxel,
        "Before: global-MAD-filtered points",
        f"After: centroid voxel downsample ({args.voxel_size * 1000:.1f} mm, once)",
    )

    rng = np.random.default_rng(args.seed)
    scene_capped = scene_voxel
    if len(scene_capped) > args.max_scene_points:
        scene_capped = scene_capped[np.sort(rng.choice(len(scene_capped), args.max_scene_points, replace=False))]
    save_cloud_pair(
        args.output_dir / "06_scene_point_cap_before_after.png",
        scene_voxel,
        scene_capped,
        "Before: extracted visible cloud",
        f"After: deterministic random cap (max={args.max_scene_points:,})",
    )

    scene_final = scene_capped

    grasp_model = load_grasp_model(args.model_config)
    model_raw = grasp_model.points_grasp_m
    model_capped = model_raw
    if len(model_capped) > args.max_model_points:
        model_capped = model_capped[np.sort(rng.choice(len(model_capped), args.max_model_points, replace=False))]
    save_cloud_pair(
        args.output_dir / "08_cad_point_cap_before_after.png",
        model_raw,
        model_capped,
        f"Before: raw {grasp_model.config.get('model_id')} grasp-frame samples",
        f"After: {grasp_model.config.get('model_id')} random cap (max={args.max_model_points:,})",
    )

    model_final = voxel_downsample(model_capped, args.voxel_size)
    save_cloud_pair(
        args.output_dir / "09_cad_voxel_before_after.png",
        model_capped,
        model_final,
        "Before: capped CAD cloud",
        f"After: CAD centroid voxel downsample ({args.voxel_size * 1000:.1f} mm)",
    )

    save_ply(args.output_dir / "scene_final_preprocessed.ply", scene_final)
    save_ply(args.output_dir / "cad_final_preprocessed.ply", model_final)

    median = float(np.median(valid_z)) if len(valid_z) else None
    summary = {
        "sample": args.rgb.stem.removesuffix("_color"),
        "inputs": {"rgb": str(args.rgb), "d2rgb": str(args.d2rgb)},
        "segmentation": {
            "confidence": detection.get("confidence"),
            "bbox_xyxy": detection.get("bbox_xyxy"),
            "polygon_points": len(polygon),
            "raw_mask_pixels": int(np.count_nonzero(raw_mask)),
            "eroded_mask_pixels": int(np.count_nonzero(depth_mask)),
        },
        "parameters": {
            "depth_range_m": [args.min_depth, args.max_depth],
            "mask_erosion_px": args.mask_erosion_px,
            "depth_filter_strategy": "pass_through_then_global_mad",
            "mad_z_threshold": args.mad_z_threshold,
            "voxel_size_m": args.voxel_size,
            "max_scene_points": args.max_scene_points,
            "max_model_points": args.max_model_points,
            "seed": args.seed,
        },
        "scene_counts": {
            "finite_positive_mask_depth": int(np.count_nonzero(finite_positive)),
            "range_valid": int(len(valid_points)),
            "global_mad_filtered": int(len(mad_points)),
            "global_mad_rejected": int(len(valid_points) - len(mad_points)),
            "centroid_voxel": int(len(scene_voxel)),
            "point_capped": int(len(scene_capped)),
        },
        "mad_statistics": mad_stats,
        "depth_statistics_m": {
            "valid_min": float(np.min(valid_z)) if len(valid_z) else None,
            "valid_median": median,
            "valid_max": float(np.max(valid_z)) if len(valid_z) else None,
        },
        "cad_counts": {
            "raw": int(len(model_raw)),
            "point_capped": int(len(model_capped)),
            "voxel": int(len(model_final)),
        },
        "artifacts": sorted(path.name for path in args.output_dir.iterdir()),
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
        stream.write("\n")

    readme = f"""# Preprocessing visualization: {summary['sample']}

The numbered PNG files show the input and output of every preprocessing stage using the runtime defaults.

1. `01_segmentation_before_after.png`: RGB before/after YOLO mask selection.
2. `02_mask_erosion_before_after.png`: raw mask before/after inward erosion.
3. `03_depth_range_filter_before_after.png`: eroded-mask depth before/after absolute depth gating.
4. `04_global_mad_before_after.png`: pass-through-filtered points before/after global camera-Z MAD filtering.
5. `05_scene_centroid_voxel_before_after.png`: MAD-filtered points before/after the single centroid-voxel filter.
6. `06_scene_point_cap_before_after.png`: scene points before/after the registration point cap. No statistical outlier filter is applied.
8. `08_cad_point_cap_before_after.png`: CAD samples before/after the registration point cap.
9. `09_cad_voxel_before_after.png`: CAD samples before/after voxel filtering.

`summary.json` contains exact counts and parameters. The two final PLY files contain the arrays passed to coarse registration.
"""
    (args.output_dir / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
