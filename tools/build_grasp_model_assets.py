#!/usr/bin/env python3
"""Build grasp-frame mesh, point cloud, and metadata assets from a STEP model."""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STEP = ROOT / "plug_model" / "2175B.stp"
DEFAULT_OBJ = ROOT / "plug_model" / "2175B_grasp.obj"
DEFAULT_PLY = ROOT / "plug_model" / "2175B_grasp.ply"
DEFAULT_CONFIG = ROOT / "configs" / "plug_models" / "2175B.yaml"
DEFAULT_REPORT = ROOT / "plug_model" / "2175B_grasp_report.md"

MM_TO_M = 0.001


@dataclass(frozen=True)
class MeshData:
    vertices_mm: np.ndarray
    faces: np.ndarray


@dataclass(frozen=True)
class GraspFrameAssets:
    vertices_m: np.ndarray
    faces: np.ndarray
    origin_raw_mm: np.ndarray
    raw_basis_from_grasp: np.ndarray
    semantic_points_grasp_m: dict[str, list[float]]
    diagnostics: dict[str, Any]


def round_vector(values: Any, digits: int = 8) -> list[float]:
    rounded = np.round(np.asarray(values, dtype=np.float64), digits)
    rounded[np.isclose(rounded, 0.0, atol=10.0 ** -digits)] = 0.0
    return [float(v) for v in rounded.tolist()]


def parse_step_length_unit(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="ignore")
    if re.search(r"SI_UNIT\s*\(\s*\.MILLI\.\s*,\s*\.METRE\.\s*\)", text, re.IGNORECASE):
        return "millimeter"
    if re.search(r"SI_UNIT\s*\(\s*\$\s*,\s*\.METRE\.\s*\)", text, re.IGNORECASE):
        return "meter"
    raise ValueError(f"Unsupported or unknown STEP length unit: {path}")


def mesh_step_with_gmsh(step_path: Path, mesh_size_mm: float) -> MeshData:
    try:
        import gmsh  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "STEP meshing requires the gmsh Python package. Install it with `pip install gmsh` "
            "or run this tool in an environment that provides gmsh."
        ) from exc

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Mesh.MeshSizeMax", float(mesh_size_mm))
        gmsh.option.setNumber("Mesh.MeshSizeMin", float(mesh_size_mm) * 0.25)
        gmsh.open(str(step_path))
        gmsh.model.mesh.generate(2)

        node_tags, coords, _ = gmsh.model.mesh.getNodes()
        vertices = np.asarray(coords, dtype=np.float64).reshape(-1, 3)
        tag_to_index = {int(tag): index for index, tag in enumerate(node_tags.tolist())}

        faces: list[list[int]] = []
        element_types, _, element_node_tags = gmsh.model.mesh.getElements(2)
        for element_type, flat_tags in zip(element_types, element_node_tags):
            _name, dim, _order, node_count, _local_coords, _primary_count = gmsh.model.mesh.getElementProperties(
                int(element_type)
            )
            if int(dim) != 2:
                continue
            tags = np.asarray(flat_tags, dtype=np.int64).reshape(-1, int(node_count))
            if int(node_count) == 3:
                for tri in tags:
                    faces.append([tag_to_index[int(tag)] for tag in tri[:3]])
            elif int(node_count) == 4:
                for quad in tags:
                    a, b, c, d = [tag_to_index[int(tag)] for tag in quad[:4]]
                    faces.append([a, b, c])
                    faces.append([a, c, d])

        if len(vertices) == 0 or len(faces) == 0:
            raise RuntimeError(f"gmsh produced an empty mesh for {step_path}")
        return MeshData(vertices_mm=vertices, faces=np.asarray(faces, dtype=np.int64))
    finally:
        gmsh.finalize()


def bin_axis_profile(vertices_mm: np.ndarray, bins: int = 160, min_points: int = 15) -> list[dict[str, Any]]:
    z = vertices_mm[:, 2]
    edges = np.linspace(float(np.min(z)), float(np.max(z)), int(bins) + 1)
    profile: list[dict[str, Any]] = []
    for index in range(int(bins)):
        lo, hi = float(edges[index]), float(edges[index + 1])
        keep = (z >= lo) & (z < hi if index < int(bins) - 1 else z <= hi)
        subset = vertices_mm[keep]
        row: dict[str, Any] = {
            "index": index,
            "z_min_mm": lo,
            "z_max_mm": hi,
            "z_center_mm": (lo + hi) * 0.5,
            "count": int(len(subset)),
        }
        if len(subset) >= int(min_points):
            x_lo, x_hi = np.percentile(subset[:, 0], [2, 98])
            y_lo, y_hi = np.percentile(subset[:, 1], [2, 98])
            width_x = float(x_hi - x_lo)
            width_y = float(y_hi - y_lo)
            radius = np.hypot(subset[:, 0], subset[:, 1])
            row.update(
                {
                    "width_x_mm": width_x,
                    "width_y_mm": width_y,
                    "bbox_area_mm2": width_x * width_y,
                    "r95_mm": float(np.percentile(radius, 95)),
                    "r50_mm": float(np.percentile(radius, 50)),
                    "circularity_xy": min(width_x, width_y) / max(width_x, width_y, 1e-9),
                }
            )
        profile.append(row)
    return profile


def contiguous_true_regions(values: np.ndarray) -> list[tuple[int, int]]:
    regions: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(values.tolist()):
        if bool(value) and start is None:
            start = index
        if start is not None and (not bool(value) or index == len(values) - 1):
            end = index - 1 if not bool(value) else index
            regions.append((start, end))
            start = None
    return regions


def estimate_grasp_origin(vertices_mm: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    profile = bin_axis_profile(vertices_mm)
    z_min = float(np.min(vertices_mm[:, 2]))
    z_max = float(np.max(vertices_mm[:, 2]))
    axial_mid = (z_min + z_max) * 0.5

    areas = np.asarray([row.get("bbox_area_mm2", np.nan) for row in profile], dtype=np.float64)
    counts = np.asarray([row["count"] for row in profile], dtype=np.int64)
    circularity = np.asarray([row.get("circularity_xy", np.nan) for row in profile], dtype=np.float64)
    z_centers = np.asarray([row["z_center_mm"] for row in profile], dtype=np.float64)

    finite_areas = areas[np.isfinite(areas)]
    if len(finite_areas) == 0:
        axis_center_xy = np.mean(vertices_mm[:, :2], axis=0)
        origin = np.asarray([axis_center_xy[0], axis_center_xy[1], axial_mid], dtype=np.float64)
        return origin, {
            "method": "axis_bbox_midpoint_fallback",
            "confidence": "low",
            "reason": "no_valid_axis_profile_bins",
        }

    max_area = float(np.nanmax(areas))
    min_area = max(1200.0, float(np.nanpercentile(finite_areas, 30)))
    max_cylinder_area = max_area * 0.55
    z_margin = (z_max - z_min) * 0.05
    candidate = (
        (counts >= 15)
        & np.isfinite(areas)
        & (areas >= min_area)
        & (areas <= max_cylinder_area)
        & np.isfinite(circularity)
        & (circularity >= 0.45)
        & (z_centers >= z_min + z_margin)
        & (z_centers <= z_max - z_margin)
    )
    regions = contiguous_true_regions(candidate)
    if not regions:
        z_center = axial_mid
        selected_region: tuple[int, int] | None = None
        confidence = "low"
        reason = "no_stable_cylindrical_region_found"
    else:
        def region_score(region: tuple[int, int]) -> float:
            start, end = region
            center = float(np.mean(z_centers[start : end + 1]))
            count_term = min(25.0, float(np.sum(counts[start : end + 1])) / 100.0)
            return abs(center - axial_mid) - count_term

        selected_region = min(regions, key=region_score)
        start, end = selected_region
        z_center = float(np.average(z_centers[start : end + 1], weights=np.maximum(counts[start : end + 1], 1)))
        confidence = "medium"
        reason = "selected_stable_cylindrical_region_closest_to_axis_midpoint"

    axis_center_xy = np.asarray(
        [
            (float(np.min(vertices_mm[:, 0])) + float(np.max(vertices_mm[:, 0]))) * 0.5,
            (float(np.min(vertices_mm[:, 1])) + float(np.max(vertices_mm[:, 1]))) * 0.5,
        ],
        dtype=np.float64,
    )
    origin = np.asarray([axis_center_xy[0], axis_center_xy[1], z_center], dtype=np.float64)

    if selected_region is None:
        selected: dict[str, Any] = {}
    else:
        start, end = selected_region
        selected = {
            "bin_start": int(start),
            "bin_end": int(end),
            "z_min_mm": round(float(profile[start]["z_min_mm"]), 6),
            "z_max_mm": round(float(profile[end]["z_max_mm"]), 6),
            "z_center_mm": round(float(z_center), 6),
            "mean_bbox_area_mm2": round(float(np.nanmean(areas[start : end + 1])), 6),
            "mean_circularity_xy": round(float(np.nanmean(circularity[start : end + 1])), 6),
            "point_count": int(np.sum(counts[start : end + 1])),
        }

    diagnostics = {
        "method": "axis_profile_stable_cylinder_closest_to_axis_midpoint",
        "confidence": confidence,
        "reason": reason,
        "axis": "raw_cad_z",
        "axis_z_min_mm": round(float(z_min), 6),
        "axis_z_max_mm": round(float(z_max), 6),
        "axis_midpoint_z_mm": round(float(axial_mid), 6),
        "head_tail_direction_assumption": "raw_cad_+Z_is_tail_to_head",
        "area_threshold_min_mm2": round(float(min_area), 6),
        "area_threshold_max_mm2": round(float(max_cylinder_area), 6),
        "selected_region": selected,
        "grasp_center_raw_mm": round_vector(origin, digits=6),
    }
    return origin, diagnostics


def build_grasp_frame_assets(mesh: MeshData) -> GraspFrameAssets:
    vertices_mm = np.asarray(mesh.vertices_mm, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    origin_raw_mm, grasp_origin = estimate_grasp_origin(vertices_mm)

    x_axis_raw = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    y_axis_raw = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    y_axis_raw = y_axis_raw - float(np.dot(y_axis_raw, x_axis_raw)) * x_axis_raw
    y_axis_raw = y_axis_raw / np.linalg.norm(y_axis_raw)
    z_axis_raw = np.cross(x_axis_raw, y_axis_raw)
    z_axis_raw = z_axis_raw / np.linalg.norm(z_axis_raw)
    raw_basis_from_grasp = np.column_stack([x_axis_raw, y_axis_raw, z_axis_raw])

    centered_raw = vertices_mm - origin_raw_mm
    vertices_grasp_mm = centered_raw @ raw_basis_from_grasp
    vertices_grasp_m = vertices_grasp_mm * MM_TO_M

    z_min = float(np.min(vertices_mm[:, 2]))
    z_max = float(np.max(vertices_mm[:, 2]))
    tail_raw = np.asarray([origin_raw_mm[0], origin_raw_mm[1], z_min], dtype=np.float64)
    head_raw = np.asarray([origin_raw_mm[0], origin_raw_mm[1], z_max], dtype=np.float64)
    tail_grasp_m = ((tail_raw - origin_raw_mm) @ raw_basis_from_grasp) * MM_TO_M
    head_grasp_m = ((head_raw - origin_raw_mm) @ raw_basis_from_grasp) * MM_TO_M

    bbox_min_raw = np.min(vertices_mm, axis=0)
    bbox_max_raw = np.max(vertices_mm, axis=0)
    diagnostics = {
        "source_bbox_raw_mm": {
            "min": round_vector(bbox_min_raw, digits=6),
            "max": round_vector(bbox_max_raw, digits=6),
            "extent": round_vector(bbox_max_raw - bbox_min_raw, digits=6),
        },
        "raw_basis_from_grasp_columns": {
            "x_tail_to_head_raw": round_vector(x_axis_raw),
            "y_closing_raw": round_vector(y_axis_raw),
            "z_approach_raw": round_vector(z_axis_raw),
        },
        "grasp_origin_estimation": grasp_origin,
    }

    return GraspFrameAssets(
        vertices_m=vertices_grasp_m,
        faces=faces,
        origin_raw_mm=origin_raw_mm,
        raw_basis_from_grasp=raw_basis_from_grasp,
        semantic_points_grasp_m={
            "grasp_center": [0.0, 0.0, 0.0],
            "tail_center": round_vector(tail_grasp_m),
            "head_center": round_vector(head_grasp_m),
        },
        diagnostics=diagnostics,
    )


def write_obj(path: Path, vertices_m: np.ndarray, faces: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as f:
        f.write("# grasp-frame mesh, units: meter\n")
        for vertex in vertices_m:
            f.write(f"v {vertex[0]:.9f} {vertex[1]:.9f} {vertex[2]:.9f}\n")
        for face in faces:
            a, b, c = (int(index) + 1 for index in face[:3])
            f.write(f"f {a} {b} {c}\n")


def sample_mesh_points(vertices_m: np.ndarray, faces: np.ndarray, count: int, seed: int) -> np.ndarray:
    triangles = vertices_m[faces[:, :3]]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    areas = np.linalg.norm(cross, axis=1) * 0.5
    valid = areas > 1e-16
    if not np.any(valid):
        raise ValueError("mesh has no non-degenerate triangle faces")
    triangles = triangles[valid]
    areas = areas[valid]
    probabilities = areas / float(np.sum(areas))

    rng = np.random.default_rng(seed)
    ids = rng.choice(len(triangles), size=int(count), replace=True, p=probabilities)
    chosen = triangles[ids]
    r1 = np.sqrt(rng.random(int(count)))
    r2 = rng.random(int(count))
    points = (1.0 - r1)[:, None] * chosen[:, 0] + (r1 * (1.0 - r2))[:, None] * chosen[:, 1] + (r1 * r2)[:, None] * chosen[:, 2]
    return points.astype(np.float64)


def write_ply_points(path: Path, points_m: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("comment grasp-frame point cloud, units: meter\n")
        f.write(f"element vertex {len(points_m)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for point in points_m:
            f.write(f"{point[0]:.9f} {point[1]:.9f} {point[2]:.9f}\n")


def write_config(
    path: Path,
    model_id: str,
    step_path: Path,
    obj_path: Path,
    ply_path: Path,
    source_unit: str,
    assets: GraspFrameAssets,
    report_path: Path,
    sample_points: int,
    mesh_size_mm: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tail = np.asarray(assets.semantic_points_grasp_m["tail_center"], dtype=np.float64)
    head = np.asarray(assets.semantic_points_grasp_m["head_center"], dtype=np.float64)
    axis_length = float(np.linalg.norm(head - tail))
    data = {
        "model_id": model_id,
        "source_step": str(step_path.relative_to(ROOT) if step_path.is_relative_to(ROOT) else step_path),
        "source_step_unit": source_unit,
        "asset_unit": "meter",
        "frame": "grasp",
        "downstream_transform_name": "t_camera_grasp",
        "requires_manual_review": True,
        "assets": {
            "mesh_obj": str(obj_path.relative_to(ROOT) if obj_path.is_relative_to(ROOT) else obj_path),
            "pointcloud_ply": str(ply_path.relative_to(ROOT) if ply_path.is_relative_to(ROOT) else ply_path),
            "review_report": str(report_path.relative_to(ROOT) if report_path.is_relative_to(ROOT) else report_path),
        },
        "semantic_points_grasp_m": assets.semantic_points_grasp_m,
        "dimensions_m": {
            "head_tail_axis_length": round(float(axis_length), 8),
        },
        "coordinate_system": {
            "origin": "estimated_grasp_grasp_origin_center",
            "x_axis": "tail_to_head",
            "y_axis": "gripper_closing_direction_from_projected_raw_cad_+Y",
            "z_axis": "approach_direction_right_handed_x_cross_y",
            "raw_cad_to_grasp": {
                "x_grasp_from_raw": "+Z",
                "y_grasp_from_raw": "+Y_projected_perpendicular_to_+Z",
                "z_grasp_from_raw": "-X",
                "origin_raw_mm": round_vector(assets.origin_raw_mm, digits=6),
                "raw_basis_from_grasp_columns": assets.diagnostics["raw_basis_from_grasp_columns"],
            },
        },
        "build": {
            "tool": "tools/build_grasp_model_assets.py",
            "mesh_size_mm": float(mesh_size_mm),
            "sample_points": int(sample_points),
        },
        "diagnostics": assets.diagnostics,
    }
    import yaml

    class NoAliasDumper(yaml.SafeDumper):
        def ignore_aliases(self, data: Any) -> bool:
            return True

    with path.open("w", encoding="utf-8") as f:
        yaml.dump(data, f, Dumper=NoAliasDumper, allow_unicode=True, sort_keys=False)


def write_review_report(
    path: Path,
    model_id: str,
    step_path: Path,
    obj_path: Path,
    ply_path: Path,
    config_path: Path,
    assets: GraspFrameAssets,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    def display_path(item: Path) -> str:
        return str(item.relative_to(ROOT) if item.is_relative_to(ROOT) else item)

    grasp_origin = assets.diagnostics["grasp_origin_estimation"]
    bbox = assets.diagnostics["source_bbox_raw_mm"]
    tail = assets.semantic_points_grasp_m["tail_center"]
    head = assets.semantic_points_grasp_m["head_center"]
    axis_length = float(np.linalg.norm(np.asarray(head, dtype=np.float64) - np.asarray(tail, dtype=np.float64)))
    lines = [
        f"# {model_id} Grasp Asset Review",
        "",
        "## Files",
        f"- STEP source: `{display_path(step_path)}`",
        f"- Grasp OBJ: `{display_path(obj_path)}`",
        f"- Grasp PLY: `{display_path(ply_path)}`",
        f"- Config: `{display_path(config_path)}`",
        "",
        "## Coordinate System",
        "- Frame: `grasp`",
        "- Origin: estimated grasp grasp_origin center",
        "- `+X`: tail to head, mapped from raw CAD `+Z`",
        "- `+Y`: gripper closing direction, mapped from projected raw CAD `+Y`",
        "- `+Z`: approach direction, right-handed `X x Y`",
        "",
        "## Semantic Points",
        f"- `grasp_center_m`: `{assets.semantic_points_grasp_m['grasp_center']}`",
        f"- `tail_center_m`: `{tail}`",
        f"- `head_center_m`: `{head}`",
        f"- `head_tail_axis_length_m`: `{axis_length:.8f}`",
        "",
        "## Raw CAD Diagnostics",
        f"- Raw bbox min mm: `{bbox['min']}`",
        f"- Raw bbox max mm: `{bbox['max']}`",
        f"- Raw bbox extent mm: `{bbox['extent']}`",
        f"- Grasp center raw mm: `{grasp_origin.get('grasp_center_raw_mm')}`",
        f"- Grasp-origin confidence: `{grasp_origin.get('confidence')}`",
        f"- Grasp-origin reason: `{grasp_origin.get('reason')}`",
        f"- Selected region: `{grasp_origin.get('selected_region')}`",
        "",
        "## Manual Review Items",
        "- Confirm raw CAD `+Z` is the intended `tail -> head` direction.",
        "- Confirm the selected grasp-origin region is the physical graspable middle section.",
        "- Confirm projected raw CAD `+Y` is acceptable as the gripper closing direction.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="2175B", help="Model identifier written into YAML metadata.")
    parser.add_argument("--step", type=Path, default=DEFAULT_STEP, help="Input STEP model path.")
    parser.add_argument("--obj", type=Path, default=DEFAULT_OBJ, help="Output grasp-frame OBJ mesh path.")
    parser.add_argument("--ply", type=Path, default=DEFAULT_PLY, help="Output grasp-frame sampled PLY point cloud path.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Output grasp model YAML path.")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT, help="Output manual review report path.")
    parser.add_argument("--mesh-size-mm", type=float, default=2.0, help="Target STEP meshing size in source millimeters.")
    parser.add_argument("--sample-points", type=int, default=30000, help="Number of surface points sampled into the PLY.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for mesh surface sampling.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mesh_size_mm <= 0:
        raise SystemExit("--mesh-size-mm must be positive")
    if args.sample_points <= 0:
        raise SystemExit("--sample-points must be positive")
    if not args.step.is_file():
        raise SystemExit(f"STEP file not found: {args.step}")

    source_unit = parse_step_length_unit(args.step)
    if source_unit != "millimeter":
        raise SystemExit(f"Only millimeter STEP models are currently supported, got {source_unit!r}: {args.step}")

    mesh = mesh_step_with_gmsh(args.step, args.mesh_size_mm)
    assets = build_grasp_frame_assets(mesh)
    points_m = sample_mesh_points(assets.vertices_m, assets.faces, args.sample_points, args.seed)

    write_obj(args.obj, assets.vertices_m, assets.faces)
    write_ply_points(args.ply, points_m)
    write_review_report(args.report, args.model_id, args.step, args.obj, args.ply, args.config, assets)
    write_config(
        args.config,
        args.model_id,
        args.step,
        args.obj,
        args.ply,
        source_unit,
        assets,
        args.report,
        args.sample_points,
        args.mesh_size_mm,
    )

    print(f"obj: {args.obj}")
    print(f"ply: {args.ply}")
    print(f"config: {args.config}")
    print(f"report: {args.report}")
    print(f"grasp_center_raw_mm: {assets.diagnostics['grasp_origin_estimation']['grasp_center_raw_mm']}")
    print(f"head_center_grasp_m: {assets.semantic_points_grasp_m['head_center']}")
    print(f"tail_center_grasp_m: {assets.semantic_points_grasp_m['tail_center']}")


if __name__ == "__main__":
    main()
