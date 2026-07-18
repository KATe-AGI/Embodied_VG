#!/usr/bin/env python3
"""Build the measured plug CAD by changing only the original model's long axis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.build_grasp_model_assets import sample_mesh_points, write_obj, write_ply_points  # noqa: E402


ORIGINAL_STEP = ROOT / "plug_model" / "2175B.stp"
ORIGINAL_OBJ = ROOT / "plug_model" / "2175B_grasp.obj"
ACTUAL_STEP = ROOT / "plug_model" / "plugCAD.stp"
ACTUAL_OBJ = ROOT / "plug_model" / "plugCAD_grasp.obj"
ACTUAL_PLY = ROOT / "plug_model" / "plugCAD_grasp.ply"
ACTUAL_CONFIG = ROOT / "configs" / "plug_models" / "plugCAD.yaml"
ACTUAL_REPORT = ROOT / "plug_model" / "plugCAD_grasp_report.md"
COMPARISON_IMAGE = ROOT / "plug_model" / "2175B_vs_plugCAD_dimensions.png"
COMPARISON_SVG = ROOT / "plug_model" / "2175B_vs_plugCAD_dimensions.svg"

# Coordinates are distances from the tail end along the tail->head axis.
# Original landmarks were identified from the original CAD axial profile.
ORIGINAL_LANDMARKS_MM = np.asarray([0.0, 30.0, 127.0, 149.0, 180.3], dtype=np.float64)
ACTUAL_LANDMARKS_MM = np.asarray([0.0, 32.0, 118.0, 139.0, 172.0], dtype=np.float64)


def axial_map_mm(values: np.ndarray | float) -> np.ndarray:
    """Piecewise-linearly map original tail-referenced coordinates to measured coordinates."""

    raw = np.asarray(values, dtype=np.float64)
    mapped = np.interp(raw, ORIGINAL_LANDMARKS_MM, ACTUAL_LANDMARKS_MM)
    if np.any(raw < ORIGINAL_LANDMARKS_MM[0]) or np.any(raw > ORIGINAL_LANDMARKS_MM[-1]):
        raise ValueError("axial coordinate lies outside the original CAD bounds")
    return mapped


def read_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    with path.open("r", encoding="ascii", errors="ignore") as stream:
        for line in stream:
            if line.startswith("v "):
                vertices.append([float(value) for value in line.split()[1:4]])
            elif line.startswith("f "):
                faces.append([int(value.split("/")[0]) - 1 for value in line.split()[1:4]])
    if not vertices or not faces:
        raise ValueError(f"OBJ mesh is incomplete: {path}")
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def deform_grasp_mesh(vertices_m: np.ndarray) -> tuple[np.ndarray, float]:
    """Apply the axial landmark mapping while preserving both transverse coordinates."""

    vertices_mm = np.asarray(vertices_m, dtype=np.float64) * 1000.0
    original_tail_x_mm = float(np.min(vertices_mm[:, 0]))
    original_u_mm = vertices_mm[:, 0] - original_tail_x_mm
    original_grasp_u_mm = -original_tail_x_mm
    actual_grasp_u_mm = float(axial_map_mm(original_grasp_u_mm))

    actual = vertices_mm.copy()
    actual[:, 0] = axial_map_mm(original_u_mm) - actual_grasp_u_mm
    return actual * 0.001, actual_grasp_u_mm


def build_actual_step(source_path: Path, output_path: Path) -> None:
    """Split, axially scale, and rejoin the original exact CAD solid."""

    import cadquery as cq

    source = cq.importers.importStep(str(source_path)).val()
    bbox = source.BoundingBox()
    total = float(ORIGINAL_LANDMARKS_MM[-1])
    xy_size = max(bbox.xlen, bbox.ylen) + 20.0
    parts = []
    for source_lo_u, source_hi_u, target_lo_u, target_hi_u in zip(
        ORIGINAL_LANDMARKS_MM[:-1],
        ORIGINAL_LANDMARKS_MM[1:],
        ACTUAL_LANDMARKS_MM[:-1],
        ACTUAL_LANDMARKS_MM[1:],
    ):
        source_lo_z = float(source_lo_u - total)
        source_hi_z = float(source_hi_u - total)
        box = (
            cq.Workplane("XY")
            .box(xy_size, xy_size, source_hi_z - source_lo_z)
            .translate((0.0, 0.0, (source_lo_z + source_hi_z) * 0.5))
            .val()
        )
        clipped = source.intersect(box)
        scale = float((target_hi_u - target_lo_u) / (source_hi_u - source_lo_u))
        target_lo_z = float(target_lo_u - ACTUAL_LANDMARKS_MM[-1])
        offset = target_lo_z - scale * source_lo_z
        matrix = cq.Matrix(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, scale, offset],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        parts.append(clipped.transformGeometry(matrix))

    result = parts[0].fuse(*parts[1:]).clean()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cq.exporters.export(result, str(output_path), exportType="STEP")

    check = cq.importers.importStep(str(output_path)).val()
    check_bbox = check.BoundingBox()
    if len(cq.Workplane(obj=check).solids().vals()) != 1:
        raise RuntimeError("generated plugCAD STEP is not one solid")
    if not np.isclose(check_bbox.zlen, ACTUAL_LANDMARKS_MM[-1], atol=1e-4):
        raise RuntimeError(f"generated STEP length is {check_bbox.zlen}, expected 172 mm")


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def write_actual_config(actual_grasp_u_mm: float) -> None:
    tail_m = -actual_grasp_u_mm * 0.001
    head_m = (float(ACTUAL_LANDMARKS_MM[-1]) - actual_grasp_u_mm) * 0.001
    data: dict[str, Any] = {
        "model_id": "plugCAD",
        "source_step": _relative(ACTUAL_STEP),
        "source_step_unit": "millimeter",
        "asset_unit": "meter",
        "frame": "grasp",
        "downstream_transform_name": "t_camera_grasp",
        "requires_manual_review": True,
        "assets": {
            "mesh_obj": _relative(ACTUAL_OBJ),
            "pointcloud_ply": _relative(ACTUAL_PLY),
            "review_report": _relative(ACTUAL_REPORT),
            "dimension_comparison": _relative(COMPARISON_IMAGE),
        },
        "semantic_points_grasp_m": {
            "grasp_center": [0.0, 0.0, 0.0],
            "tail_center": [round(tail_m, 8), 0.0, 0.0],
            "head_center": [round(head_m, 8), 0.0, 0.0],
        },
        "dimensions_m": {
            "head_tail_axis_length": 0.172,
            "tail_section_length": 0.032,
            "tail_end_to_large_ring_near_edge": 0.118,
            "large_ring_length": 0.021,
            "large_ring_far_edge_to_head": 0.033,
        },
        "coordinate_system": {
            "origin": "mapped_original_2175B_grasp_origin",
            "x_axis": "tail_to_head",
            "y_axis": "unchanged_from_2175B_gripper_closing_direction",
            "z_axis": "unchanged_from_2175B_approach_direction",
        },
        "axial_adaptation": {
            "method": "piecewise_linear_tail_to_head_only",
            "transverse_coordinates_unchanged": True,
            "original_model": _relative(ORIGINAL_STEP),
            "landmark_names": ["tail_end", "tail_section_end", "large_ring_near_edge", "large_ring_far_edge", "head_end"],
            "original_landmarks_from_tail_mm": ORIGINAL_LANDMARKS_MM.tolist(),
            "actual_landmarks_from_tail_mm": ACTUAL_LANDMARKS_MM.tolist(),
            "interpretation": "tail center means the center of the tail end face",
        },
        "build": {
            "tool": "tools/build_actual_plug_cad.py",
            "sample_points": 30000,
            "seed": 0,
        },
    }
    ACTUAL_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    with ACTUAL_CONFIG.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(data, stream, allow_unicode=True, sort_keys=False)


def write_report(actual_grasp_u_mm: float) -> None:
    scales = np.diff(ACTUAL_LANDMARKS_MM) / np.diff(ORIGINAL_LANDMARKS_MM)
    lines = [
        "# plugCAD Actual-Dimension Grasp Asset Review",
        "",
        "## Files",
        f"- Original STEP: `{_relative(ORIGINAL_STEP)}`",
        f"- Actual STEP: `{_relative(ACTUAL_STEP)}`",
        f"- Actual grasp OBJ: `{_relative(ACTUAL_OBJ)}`",
        f"- Actual grasp PLY: `{_relative(ACTUAL_PLY)}`",
        f"- Runtime config: `{_relative(ACTUAL_CONFIG)}`",
        f"- Dimension comparison: `{_relative(COMPARISON_IMAGE)}`",
        "",
        "## Measurement Interpretation",
        "- All dimensions are projections onto the plug center axis.",
        "- `tail center` is interpreted as the center of the tail end face.",
        "- Tail section length: `32.0 mm`.",
        "- Tail end center to large-ring near edge: `118.0 mm`.",
        "- Large-ring axial length: `21.0 mm`.",
        "- Overall length: `172.0 mm`.",
        "- Implied large-ring far edge to head: `33.0 mm`.",
        "",
        "## Axial Mapping",
        f"- Original landmarks from tail, mm: `{ORIGINAL_LANDMARKS_MM.tolist()}`",
        f"- Actual landmarks from tail, mm: `{ACTUAL_LANDMARKS_MM.tolist()}`",
        f"- Segment scale factors: `{np.round(scales, 8).tolist()}`",
        "- Only the tail-to-head coordinate changes; every transverse coordinate is copied unchanged.",
        f"- Mapped grasp origin from tail: `{actual_grasp_u_mm:.6f} mm`.",
        "",
        "## Manual Review",
        "- Confirm that the 32 mm tail boundary is the first major shoulder shown in the comparison image.",
        "- Confirm that tail center refers to the tail end-face center, as interpreted above.",
        "- Confirm that the large ring is bounded by the annotated 118 mm and 139 mm landmarks.",
        "",
    ]
    ACTUAL_REPORT.write_text("\n".join(lines), encoding="utf-8")


def draw_dimension(ax, start: float, end: float, y: float, text: str, color: str) -> None:
    ax.annotate("", xy=(end, y), xytext=(start, y), arrowprops={"arrowstyle": "<->", "color": color, "lw": 1.5})
    ax.text((start + end) * 0.5, y - 2.0, text, color=color, ha="center", va="top", fontsize=9)
    ax.plot([start, start], [y + 2.0, y - 1.5], color=color, lw=0.8)
    ax.plot([end, end], [y + 2.0, y - 1.5], color=color, lw=0.8)


def write_comparison_image(original_m: np.ndarray, actual_m: np.ndarray) -> None:
    original_mm = original_m * 1000.0
    actual_mm = actual_m * 1000.0
    original_u = original_mm[:, 0] - np.min(original_mm[:, 0])
    actual_u = actual_mm[:, 0] - np.min(actual_mm[:, 0])

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.scatter(original_u, original_mm[:, 1] + 115.0, s=0.35, color="#667085", alpha=0.22, rasterized=True)
    ax.scatter(actual_u, actual_mm[:, 1], s=0.35, color="#1570ef", alpha=0.25, rasterized=True)
    ax.text(0.0, 169.0, "2175B original CAD (180.3 mm)", fontsize=13, weight="bold", color="#344054")
    ax.text(0.0, 58.0, "plugCAD measured model (172.0 mm)", fontsize=13, weight="bold", color="#175cd3")

    for value in ACTUAL_LANDMARKS_MM:
        ax.axvline(value, ymin=0.04, ymax=0.48, color="#84adff", lw=0.8, ls="--")

    draw_dimension(ax, 0.0, 32.0, -55.0, "tail section 32 mm", "#067647")
    draw_dimension(ax, 0.0, 118.0, -66.0, "tail end center to ring near edge 118 mm", "#b54708")
    draw_dimension(ax, 118.0, 139.0, -77.0, "large ring 21 mm", "#b42318")
    draw_dimension(ax, 139.0, 172.0, -88.0, "head remainder 33 mm", "#6941c6")
    draw_dimension(ax, 0.0, 172.0, -99.0, "overall 172 mm", "#175cd3")

    ax.set_xlim(-6.0, 187.0)
    ax.set_ylim(-108.0, 174.0)
    ax.set_xlabel("Distance from tail end along center axis (mm)")
    ax.set_ylabel("Side projection; transverse dimensions unchanged (mm)")
    ax.set_title("2175B original CAD vs plugCAD measured axial adaptation", fontsize=15)
    ax.grid(True, color="#eaecf0", lw=0.7)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(COMPARISON_IMAGE, dpi=220)
    fig.savefig(COMPARISON_SVG)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-step", action="store_true", help="Do not rebuild plugCAD.stp.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    original_vertices, faces = read_obj(ORIGINAL_OBJ)
    actual_vertices, actual_grasp_u_mm = deform_grasp_mesh(original_vertices)

    if not args.skip_step:
        build_actual_step(ORIGINAL_STEP, ACTUAL_STEP)
    write_obj(ACTUAL_OBJ, actual_vertices, faces)
    points = sample_mesh_points(actual_vertices, faces, count=30000, seed=0)
    write_ply_points(ACTUAL_PLY, points)
    write_actual_config(actual_grasp_u_mm)
    write_report(actual_grasp_u_mm)
    write_comparison_image(original_vertices, actual_vertices)

    print(f"original_step: {ORIGINAL_STEP}")
    print(f"actual_step: {ACTUAL_STEP}")
    print(f"actual_obj: {ACTUAL_OBJ}")
    print(f"actual_ply: {ACTUAL_PLY}")
    print(f"config: {ACTUAL_CONFIG}")
    print(f"comparison: {COMPARISON_IMAGE}")
    print(f"mapped_grasp_origin_from_tail_mm: {actual_grasp_u_mm:.6f}")


if __name__ == "__main__":
    main()
