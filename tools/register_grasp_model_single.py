#!/usr/bin/env python3
"""Register one scene point cloud to a grasp-frame CAD model."""

from __future__ import annotations

import argparse
import base64
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "plug_models" / "plugCAD.yaml"
DEFAULT_OUTPUT_DIR = ROOT / "ultralytics" / "runs" / "grasp_model_registration_single"


@dataclass(frozen=True)
class PointCloud:
    points: np.ndarray
    colors: np.ndarray | None = None


@dataclass(frozen=True)
class PlyHeader:
    format_name: str
    vertex_count: int
    vertex_properties: list[tuple[str, str]]
    header_bytes: int


PLY_PROPERTY_TYPES = {
    "float": "<f4",
    "float32": "<f4",
    "double": "<f8",
    "float64": "<f8",
    "uchar": "u1",
    "uint8": "u1",
    "char": "i1",
    "int8": "i1",
    "ushort": "<u2",
    "uint16": "<u2",
    "short": "<i2",
    "int16": "<i2",
    "uint": "<u4",
    "uint32": "<u4",
    "int": "<i4",
    "int32": "<i4",
}


def round_list(values: Any, digits: int = 8) -> list[Any]:
    rounded = np.round(np.asarray(values, dtype=np.float64), digits)
    rounded[np.isclose(rounded, 0.0, atol=10.0 ** -digits)] = 0.0
    return rounded.tolist()


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def resolve_path(config_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    root_path = ROOT / path
    if root_path.exists() or not (config_path.parent / path).exists():
        return root_path
    return config_path.parent / path


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return data


def parse_ply_header(path: Path) -> PlyHeader:
    with path.open("rb") as f:
        lines: list[bytes] = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"PLY header missing end_header: {path}")
            lines.append(line)
            if line.strip() == b"end_header":
                break
    header_text = b"".join(lines).decode("ascii", errors="strict")
    if not header_text.startswith("ply"):
        raise ValueError(f"Not a PLY file: {path}")
    header_lines = header_text.splitlines()
    format_name: str | None = None
    for line in header_lines:
        parts = line.split()
        if len(parts) == 3 and parts[0] == "format" and parts[2] == "1.0":
            format_name = parts[1]
            break
    if format_name is None:
        raise ValueError(f"PLY format line missing or unsupported: {path}")
    if format_name != "binary_little_endian":
        raise ValueError(f"Only binary_little_endian PLY is supported for scene clouds, got {format_name}: {path}")
    vertex_count: int | None = None
    for line in header_lines:
        parts = line.split()
        if len(parts) == 3 and parts[0] == "element" and parts[1] == "vertex":
            vertex_count = int(parts[2])
            break
    if vertex_count is None:
        raise ValueError(f"PLY vertex element missing: {path}")

    vertex_properties: list[tuple[str, str]] = []
    in_vertex = False
    for raw_line in header_lines:
        parts = raw_line.split()
        if len(parts) >= 3 and parts[0] == "element":
            in_vertex = parts[1] == "vertex"
            continue
        if not in_vertex or not parts or parts[0] != "property":
            continue
        if len(parts) >= 3 and parts[1] != "list":
            vertex_properties.append((parts[2], parts[1]))
    return PlyHeader(
        format_name=format_name,
        vertex_count=vertex_count,
        vertex_properties=vertex_properties,
        header_bytes=sum(len(line) for line in lines),
    )


def read_realsense_binary_ply(path: Path) -> PointCloud:
    header = parse_ply_header(path)
    dtype_fields: list[tuple[str, str]] = []
    for name, type_name in header.vertex_properties:
        if type_name not in PLY_PROPERTY_TYPES:
            raise ValueError(f"Unsupported PLY vertex property type {type_name!r} for {name!r}: {path}")
        dtype_fields.append((name, PLY_PROPERTY_TYPES[type_name]))
    dtype = np.dtype(dtype_fields)
    with path.open("rb") as f:
        f.seek(header.header_bytes)
        data = np.fromfile(f, dtype=dtype, count=header.vertex_count)
    required = {"x", "y", "z"}
    if not required.issubset(data.dtype.names or ()):
        raise ValueError(f"PLY vertex properties must contain x/y/z: {path}")
    points = np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float64, copy=False)
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    colors = None
    if {"red", "green", "blue"}.issubset(data.dtype.names or ()):
        colors = np.column_stack([data["red"], data["green"], data["blue"]]).astype(np.float64, copy=False) / 255.0
        colors = colors[finite]
    return PointCloud(points=points, colors=colors)


def read_ascii_ply_points(path: Path) -> np.ndarray:
    with path.open("r", encoding="ascii", errors="ignore") as f:
        vertex_count: int | None = None
        for line in f:
            if line.startswith("element vertex"):
                vertex_count = int(line.split()[2])
            if line.strip() == "end_header":
                break
        if vertex_count is None:
            raise ValueError(f"PLY vertex element missing: {path}")
        points = []
        for index, line in enumerate(f):
            if index >= vertex_count:
                break
            parts = line.split()
            if len(parts) >= 3:
                points.append([float(parts[0]), float(parts[1]), float(parts[2])])
    if not points:
        raise ValueError(f"PLY contains no points: {path}")
    return np.asarray(points, dtype=np.float64)


def crop_aabb(points: np.ndarray, crop_min: np.ndarray, crop_max: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lo = np.minimum(crop_min, crop_max)
    hi = np.maximum(crop_min, crop_max)
    keep = np.all((points >= lo) & (points <= hi), axis=1)
    return points[keep], keep


def color_filter_mask(colors: np.ndarray | None, mode: str) -> np.ndarray:
    if mode == "none":
        if colors is None:
            return np.ones(0, dtype=bool)
        return np.ones(len(colors), dtype=bool)
    if colors is None:
        raise ValueError("color_filter_requested_but_scene_has_no_rgb")

    rgb = np.asarray(colors, dtype=np.float64)
    if rgb.size == 0:
        return np.ones(0, dtype=bool)
    if float(np.nanmax(rgb)) <= 1.0:
        rgb = rgb * 255.0
    red = rgb[:, 0]
    green = rgb[:, 1]
    blue = rgb[:, 2]

    if mode == "not-yellow":
        yellow = (red > 105.0) & (green > 85.0) & (blue < 95.0) & ((red - blue) > 45.0) & ((green - blue) > 35.0)
        return ~yellow
    if mode == "red-or-gray":
        red_body = (red > 90.0) & (red > green + 20.0) & (red > blue + 25.0) & (green < 125.0) & (blue < 125.0)
        spread = np.maximum.reduce([red, green, blue]) - np.minimum.reduce([red, green, blue])
        mean = np.mean(rgb, axis=1)
        gray_body = (spread < 48.0) & (mean > 35.0) & (mean < 190.0)
        return red_body | gray_body
    raise ValueError(f"unsupported_color_filter:{mode}")


def random_sample_indices(count: int, max_count: int, seed: int) -> np.ndarray:
    if count <= max_count:
        return np.arange(count)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(count, size=max_count, replace=False))


def pca_basis(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = points - np.mean(points, axis=0)
    cov = np.cov(centered.T)
    values, vectors = np.linalg.eigh(cov)
    order = np.argsort(values)[::-1]
    return values[order], vectors[:, order]


def candidate_transforms(source_points: np.ndarray, target_points: np.ndarray) -> list[dict[str, Any]]:
    values, vectors = pca_basis(target_points)
    source_centroid = np.mean(source_points, axis=0)
    target_centroid = np.mean(target_points, axis=0)
    candidates: list[dict[str, Any]] = []
    for x_sign in (1.0, -1.0):
        for y_sign in (1.0, -1.0):
            x_axis = vectors[:, 0] * x_sign
            y_seed = vectors[:, 1] * y_sign
            y_axis = y_seed - float(np.dot(y_seed, x_axis)) * x_axis
            y_norm = float(np.linalg.norm(y_axis))
            if y_norm < 1e-9:
                continue
            y_axis = y_axis / y_norm
            z_axis = np.cross(x_axis, y_axis)
            z_norm = float(np.linalg.norm(z_axis))
            if z_norm < 1e-9:
                continue
            z_axis = z_axis / z_norm
            rotation = np.column_stack([x_axis, y_axis, z_axis])
            transform = np.eye(4, dtype=np.float64)
            transform[:3, :3] = rotation
            transform[:3, 3] = target_centroid - rotation @ source_centroid
            candidates.append(
                {
                    "name": f"pca_x_{int(x_sign):+d}_y_{int(y_sign):+d}",
                    "transform": transform,
                    "target_pca_eigenvalues": round_list(values, digits=10),
                }
            )
    return candidates


def semantic_points_camera(config: dict[str, Any], t_camera_grasp: np.ndarray) -> dict[str, list[float]]:
    raw_points = config.get("semantic_points_grasp_m") or {}
    result: dict[str, list[float]] = {}
    for key in ("grasp_center", "tail_center", "head_center"):
        point = np.asarray(raw_points[key], dtype=np.float64)
        homo = np.ones(4, dtype=np.float64)
        homo[:3] = point
        result[f"{key}_camera_m"] = round_list((t_camera_grasp @ homo)[:3])
    return result


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def set_focus_limits(axis: Any, points: np.ndarray, margin_ratio: float = 0.12) -> None:
    if len(points) == 0:
        return
    lo = np.percentile(points, 1.0, axis=0)
    hi = np.percentile(points, 99.0, axis=0)
    center = (lo + hi) * 0.5
    span = np.maximum(hi - lo, 0.02)
    max_span = float(np.max(span)) * (1.0 + margin_ratio)
    axis.set_xlim(center[0] - max_span * 0.5, center[0] + max_span * 0.5)
    axis.set_ylim(center[1] - max_span * 0.5, center[1] + max_span * 0.5)
    if hasattr(axis, "set_zlim"):
        axis.set_zlim(center[2] - max_span * 0.5, center[2] + max_span * 0.5)


def run_registration(
    model_points: np.ndarray,
    cropped_scene_points: np.ndarray,
    voxel_size: float,
    outlier_nb_neighbors: int,
    outlier_std_ratio: float,
    max_model_points: int,
    max_scene_points: int,
    icp_threshold: float,
    icp_iterations: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any], np.ndarray, np.ndarray]:
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError("open3d is required for registration. Install it with `pip install open3d`.") from exc

    model_idx = random_sample_indices(len(model_points), max_model_points, seed)
    scene_idx = random_sample_indices(len(cropped_scene_points), max_scene_points, seed + 1)
    model_sample = model_points[model_idx]
    scene_sample = cropped_scene_points[scene_idx]

    def make_pcd(points: np.ndarray, remove_outliers: bool = False):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if voxel_size > 0:
            pcd = pcd.voxel_down_sample(voxel_size)
        if remove_outliers and outlier_nb_neighbors > 0 and len(pcd.points) > outlier_nb_neighbors:
            pcd, _indices = pcd.remove_statistical_outlier(
                nb_neighbors=int(outlier_nb_neighbors),
                std_ratio=float(outlier_std_ratio),
            )
        return pcd

    model_pcd = make_pcd(model_sample)
    scene_pcd = make_pcd(scene_sample, remove_outliers=True)
    if len(model_pcd.points) < 20 or len(scene_pcd.points) < 20:
        raise ValueError("not_enough_points_after_downsampling")

    model_down = np.asarray(model_pcd.points)
    scene_down = np.asarray(scene_pcd.points)
    candidates = candidate_transforms(model_down, scene_down)
    if not candidates:
        raise ValueError("initial_transform_generation_failed")

    model_tree = o3d.geometry.KDTreeFlann(model_pcd)
    scored: list[dict[str, Any]] = []
    for item in candidates:
        t_camera_grasp_init = item["transform"]
        t_grasp_camera_init = np.linalg.inv(t_camera_grasp_init)
        transformed = transform_points(scene_down, t_grasp_camera_init)
        distances: list[float] = []
        for point in transformed:
            k, _idx, d2 = model_tree.search_knn_vector_3d(point, 1)
            if k:
                distances.append(float(math.sqrt(d2[0])))
        if distances:
            inlier = np.asarray(distances) <= icp_threshold
            fitness = float(np.mean(inlier))
            rmse = float(np.sqrt(np.mean(np.square(np.asarray(distances)[inlier])))) if np.any(inlier) else float("inf")
        else:
            fitness = 0.0
            rmse = float("inf")
        scored.append({**item, "coarse_fitness": fitness, "coarse_rmse": rmse})
    scored.sort(key=lambda item: (-item["coarse_fitness"], item["coarse_rmse"]))

    estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=int(icp_iterations))
    best_result: dict[str, Any] | None = None
    best_transform_camera_grasp: np.ndarray | None = None
    candidate_summaries: list[dict[str, Any]] = []
    for item in scored:
        reg = o3d.pipelines.registration.registration_icp(
            scene_pcd,
            model_pcd,
            float(icp_threshold),
            np.linalg.inv(item["transform"]),
            estimation,
            criteria,
        )
        summary = {
            "name": item["name"],
            "coarse_fitness": round(float(item["coarse_fitness"]), 6),
            "coarse_rmse": None if not np.isfinite(item["coarse_rmse"]) else round(float(item["coarse_rmse"]), 8),
            "icp_fitness": round(float(reg.fitness), 6),
            "icp_inlier_rmse": round(float(reg.inlier_rmse), 8),
            "target_pca_eigenvalues": item["target_pca_eigenvalues"],
        }
        candidate_summaries.append(summary)
        score_key = (float(reg.fitness), -float(reg.inlier_rmse))
        if best_result is None or score_key > (best_result["fitness"], -best_result["inlier_rmse"]):
            t_camera_grasp = np.linalg.inv(np.asarray(reg.transformation, dtype=np.float64))
            best_result = {
                "candidate": item["name"],
                "fitness": float(reg.fitness),
                "inlier_rmse": float(reg.inlier_rmse),
                "model_points_used": int(len(model_pcd.points)),
                "scene_points_used": int(len(scene_pcd.points)),
                "registration_direction": "scene_to_model_then_invert",
            }
            best_transform_camera_grasp = t_camera_grasp

    if best_transform_camera_grasp is None or best_result is None:
        raise ValueError("registration_failed_no_candidate")
    quality = {
        **best_result,
        "fitness": round(float(best_result["fitness"]), 6),
        "inlier_rmse": round(float(best_result["inlier_rmse"]), 8),
        "icp_threshold_m": float(icp_threshold),
        "voxel_size_m": float(voxel_size),
        "outlier_nb_neighbors": int(outlier_nb_neighbors),
        "outlier_std_ratio": float(outlier_std_ratio),
        "candidates": candidate_summaries,
    }
    return best_transform_camera_grasp, quality, model_down, scene_down


def draw_review(
    output_png: Path,
    output_html: Path,
    scene_points: np.ndarray,
    cropped_points: np.ndarray,
    model_points: np.ndarray,
    t_camera_grasp: np.ndarray | None,
    semantic_camera: dict[str, list[float]] | None,
    seed: int,
) -> None:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    full_idx = random_sample_indices(len(scene_points), min(25000, len(scene_points)), seed)
    crop_idx = random_sample_indices(len(cropped_points), min(20000, len(cropped_points)), seed + 1)
    model_idx = random_sample_indices(len(model_points), min(20000, len(model_points)), seed + 2)
    full = scene_points[full_idx]
    crop = cropped_points[crop_idx]
    model_transformed = transform_points(model_points[model_idx], t_camera_grasp) if t_camera_grasp is not None else None

    fig = plt.figure(figsize=(16, 11))
    ax = fig.add_subplot(2, 2, 1, projection="3d")
    ax.scatter(full[:, 0], full[:, 1], full[:, 2], s=0.3, c="#98a2b3", alpha=0.12, depthshade=False, label="scene sample")
    ax.scatter(crop[:, 0], crop[:, 1], crop[:, 2], s=1.2, c="#079455", alpha=0.38, depthshade=False, label="cropped scene")
    if model_transformed is not None:
        ax.scatter(model_transformed[:, 0], model_transformed[:, 1], model_transformed[:, 2], s=1.0, c="#d92d20", alpha=0.45, depthshade=False, label="registered model")
        focus_points = np.vstack([crop, model_transformed])
    else:
        focus_points = crop
    if t_camera_grasp is not None and semantic_camera is not None:
        origin = np.asarray(semantic_camera["grasp_center_camera_m"], dtype=np.float64)
        axis_scale = 0.06
        axes = [("+X", 0, "#d92d20"), ("+Y", 1, "#079455"), ("+Z", 2, "#1570ef")]
        for label, col, color in axes:
            direction = t_camera_grasp[:3, col]
            ax.quiver(origin[0], origin[1], origin[2], direction[0], direction[1], direction[2], length=axis_scale, color=color, linewidth=2)
            end = origin + direction * axis_scale
            ax.text(end[0], end[1], end[2], label, color=color, fontsize=9)
        markers = {
            "grasp_center_camera_m": ("o", "#ffffff"),
            "tail_center_camera_m": ("^", "#255e9e"),
            "head_center_camera_m": ("s", "#d92d20"),
        }
        for name, (marker, color) in markers.items():
            point = np.asarray(semantic_camera[name], dtype=np.float64)
            ax.scatter(point[0], point[1], point[2], marker=marker, s=65, c=color, edgecolors="#111827", linewidths=1.1)
            ax.text(point[0], point[1], point[2], name.replace("_camera_m", ""), fontsize=8)
    ax.set_title("Scene/Crop/Registered Model")
    ax.set_xlabel("camera X (m)")
    ax.set_ylabel("camera Y (m)")
    ax.set_zlabel("camera Z (m)")
    set_focus_limits(ax, focus_points)
    ax.legend(loc="upper left")

    def projection(axis, dims: tuple[int, int], title: str) -> None:
        a, b = dims
        axis.scatter(crop[:, a], crop[:, b], s=0.8, c="#079455", alpha=0.25, label="crop")
        if model_transformed is not None:
            axis.scatter(model_transformed[:, a], model_transformed[:, b], s=0.8, c="#d92d20", alpha=0.25, label="model")
        axis.set_title(title)
        axis.set_xlabel(["camera X (m)", "camera Y (m)", "camera Z (m)"][a])
        axis.set_ylabel(["camera X (m)", "camera Y (m)", "camera Z (m)"][b])
        axis.set_aspect("equal", adjustable="box")
        set_focus_limits(axis, focus_points[:, [a, b, b]])
        axis.grid(True, alpha=0.25)
        axis.legend(loc="best")

    projection(fig.add_subplot(2, 2, 2), (0, 1), "XY Projection")
    projection(fig.add_subplot(2, 2, 3), (0, 2), "XZ Projection")
    projection(fig.add_subplot(2, 2, 4), (1, 2), "YZ Projection")
    fig.tight_layout()
    fig.savefig(output_png, dpi=200)
    plt.close(fig)

    encoded = base64.b64encode(output_png.read_bytes()).decode("ascii")
    html = f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>grasp model registration review</title>
<style>body{{font-family:Arial,sans-serif;margin:24px;color:#111827}}img{{max-width:100%;border:1px solid #d0d5dd}}</style></head>
<body>
<h1>Grasp Model Registration Review</h1>
<p><img src="data:image/png;base64,{encoded}" alt="registration review"></p>
</body>
</html>
"""
    output_html.write_text(html, encoding="utf-8")


def make_failure(args: argparse.Namespace, reason: str, warnings: list[str] | None = None) -> dict[str, Any]:
    return {
        "status": "failed",
        "reason": reason,
        "warnings": list(warnings or []),
        "input": {
            "config": str(args.config),
            "scene_ply": str(args.scene_ply),
            "crop_min": None if args.crop_min is None else [float(v) for v in args.crop_min],
            "crop_max": None if args.crop_max is None else [float(v) for v in args.crop_max],
            "color_filter": getattr(args, "color_filter", "none"),
        },
    }


def run(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    start = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.scene_ply.stem.replace("_pointcloud", "")
    json_path = args.output_dir / f"{stem}_grasp_model_registration.json"
    png_path = args.output_dir / f"{stem}_grasp_model_registration.png"
    html_path = args.output_dir / f"{stem}_grasp_model_registration.html"

    warnings: list[str] = []
    try:
        config = load_yaml(args.config)
        model_ply = resolve_path(args.config, config["assets"]["pointcloud_ply"])
        if config.get("frame") != "grasp":
            warnings.append("config_frame_not_grasp")
        if config.get("asset_unit") != "meter":
            warnings.append("config_asset_unit_not_meter")
        scene = read_realsense_binary_ply(args.scene_ply)
        model_points = read_ascii_ply_points(model_ply)

        if args.crop_min is None or args.crop_max is None:
            result = make_failure(args, "crop_min_and_crop_max_required", warnings)
            write_json(json_path, result)
            return result, json_path
        crop_min = np.asarray(args.crop_min, dtype=np.float64)
        crop_max = np.asarray(args.crop_max, dtype=np.float64)
        cropped_aabb, keep = crop_aabb(scene.points, crop_min, crop_max)
        cropped_colors = scene.colors[keep] if scene.colors is not None else None
        if args.color_filter == "none":
            color_keep = np.ones(len(cropped_aabb), dtype=bool)
        else:
            color_keep = color_filter_mask(cropped_colors, args.color_filter)
        cropped = cropped_aabb[color_keep]
        if len(cropped) < args.min_crop_points:
            result = make_failure(args, "insufficient_cropped_scene_points", warnings)
            result["quality"] = {
                "scene_points": int(len(scene.points)),
                "aabb_cropped_scene_points": int(len(cropped_aabb)),
                "cropped_scene_points": int(len(cropped)),
                "min_crop_points": int(args.min_crop_points),
            }
            write_json(json_path, result)
            if len(cropped) > 0 and args.save_review:
                draw_review(png_path, html_path, scene.points, cropped, model_points, None, None, args.seed)
                result["artifacts"] = {"review_png": str(png_path), "review_html": str(html_path)}
                write_json(json_path, result)
            return result, json_path

        extents = np.ptp(scene.points, axis=0)
        if np.max(extents) > 20.0:
            warnings.append("scene_extent_large_check_units")
        t_camera_grasp, quality, model_down, scene_down = run_registration(
            model_points,
            cropped,
            args.voxel_size,
            args.outlier_nb_neighbors,
            args.outlier_std_ratio,
            args.max_model_points,
            args.max_scene_points,
            args.icp_threshold,
            args.icp_iterations,
            args.seed,
        )
        semantic = semantic_points_camera(config, t_camera_grasp)
        rotation = t_camera_grasp[:3, :3]
        translation = t_camera_grasp[:3, 3]
        result = {
            "status": "ok",
            "warnings": warnings,
            "input": {
                "config": str(args.config),
                "scene_ply": str(args.scene_ply),
                "model_ply": str(model_ply),
                "crop_min": round_list(crop_min),
                "crop_max": round_list(crop_max),
                "color_filter": args.color_filter,
            },
            "t_camera_grasp": round_list(t_camera_grasp),
            "grasp_pose_camera": {
                "translation_m": round_list(translation),
                "rotation_matrix": round_list(rotation),
            },
            **semantic,
            "quality": {
                **quality,
                "scene_points": int(len(scene.points)),
                "aabb_cropped_scene_points": int(len(cropped_aabb)),
                "cropped_scene_points": int(len(cropped)),
                "model_points": int(len(model_points)),
            },
        }
        if args.save_review:
            draw_review(png_path, html_path, scene.points, cropped, model_points, t_camera_grasp, semantic, args.seed)
            result["artifacts"] = {"review_png": str(png_path), "review_html": str(html_path)}
        elapsed = time.perf_counter() - start
        result["timing"] = {"registration_single_s": round(float(elapsed), 6)}
        write_json(json_path, result)
        return result, json_path
    except Exception as exc:
        result = make_failure(args, type(exc).__name__, [*warnings, str(exc)])
        elapsed = time.perf_counter() - start
        result["timing"] = {"registration_single_s": round(float(elapsed), 6)}
        write_json(json_path, result)
        return result, json_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Grasp model YAML config.")
    parser.add_argument("--scene-ply", type=Path, required=True, help="Input RealSense binary scene point cloud PLY.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory for JSON and review artifacts.")
    parser.add_argument("--crop-min", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"), help="AABB crop minimum in camera meters.")
    parser.add_argument("--crop-max", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"), help="AABB crop maximum in camera meters.")
    parser.add_argument(
        "--color-filter",
        choices=("none", "not-yellow", "red-or-gray"),
        default="none",
        help="Optional RGB cleanup after AABB crop. Use none for geometry-only validation.",
    )
    parser.add_argument("--min-crop-points", type=int, default=100, help="Minimum cropped scene points required.")
    parser.add_argument("--voxel-size", type=float, default=0.004, help="Voxel size for registration downsampling in meters.")
    parser.add_argument("--outlier-nb-neighbors", type=int, default=0, help="Enable statistical outlier filtering for cropped scene points when > 0.")
    parser.add_argument("--outlier-std-ratio", type=float, default=2.0, help="Standard deviation ratio for statistical outlier filtering.")
    parser.add_argument("--icp-threshold", type=float, default=0.025, help="ICP correspondence threshold in meters.")
    parser.add_argument("--icp-iterations", type=int, default=80, help="ICP iteration count per candidate.")
    parser.add_argument("--max-model-points", type=int, default=30000, help="Maximum model points used for registration.")
    parser.add_argument("--max-scene-points", type=int, default=50000, help="Maximum cropped scene points used for registration.")
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed.")
    parser.add_argument("--save-review", action="store_true", help="Save headless PNG/HTML review.")
    return parser.parse_args()


def print_result(result: dict[str, Any], json_path: Path) -> None:
    print(f"status: {result.get('status')}")
    if result.get("status") == "ok":
        print(f"t_camera_grasp: {result.get('t_camera_grasp')}")
        quality = result.get("quality") or {}
        print(f"fitness: {quality.get('fitness')}")
        print(f"inlier_rmse: {quality.get('inlier_rmse')}")
        print(f"grasp_center_camera_m: {result.get('grasp_center_camera_m')}")
        print(f"tail_center_camera_m: {result.get('tail_center_camera_m')}")
        print(f"head_center_camera_m: {result.get('head_center_camera_m')}")
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
    if result.get("status") != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
