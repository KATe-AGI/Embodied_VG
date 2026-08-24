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

from tools.build_grasp_model_assets import sample_mesh_points_with_normals, write_obj, write_ply_points  # noqa: E402


ORIGINAL_STEP = ROOT / "plug_model" / "2175B.stp"
ORIGINAL_OBJ = ROOT / "plug_model" / "2175B_grasp.obj"
ACTUAL_STEP = ROOT / "plug_model" / "plugCAD.stp"
ACTUAL_OBJ = ROOT / "plug_model" / "plugCAD_grasp.obj"
ACTUAL_PLY = ROOT / "plug_model" / "plugCAD_grasp.ply"
ACTUAL_CONFIG = ROOT / "configs" / "plug_models" / "plugCAD.yaml"
ACTUAL_REPORT = ROOT / "plug_model" / "plugCAD_grasp_report.md"
COMPARISON_IMAGE = ROOT / "plug_model" / "2175B_vs_plugCAD_dimensions.png"
COMPARISON_SVG = ROOT / "plug_model" / "2175B_vs_plugCAD_dimensions.svg"

# Coordinates are distances from the tail end along the tail->head axis.  S1-S8
# are the major external axial sections identified from the original STEP.
ORIGINAL_SEGMENTS_MM = np.asarray([29.3, 18.0, 8.5, 50.0, 21.7, 21.0, 22.3, 9.5], dtype=np.float64)
ACTUAL_SEGMENTS_MM = np.asarray([29.3, 1.0, 8.5, 50.0, 21.7, 23.0, 22.3, 12.0], dtype=np.float64)
ORIGINAL_LANDMARKS_MM = np.concatenate(([0.0], np.cumsum(ORIGINAL_SEGMENTS_MM)))
ACTUAL_LANDMARKS_MM = np.concatenate(([0.0], np.cumsum(ACTUAL_SEGMENTS_MM)))


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
        raise RuntimeError(
            f"generated STEP length is {check_bbox.zlen}, expected {ACTUAL_LANDMARKS_MM[-1]:.6f} mm"
        )


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
            "head_tail_axis_length": float(ACTUAL_LANDMARKS_MM[-1] * 0.001),
            "tail_section_length": float(ACTUAL_SEGMENTS_MM[0] * 0.001),
            "tail_end_to_large_ring_near_edge": float(ACTUAL_LANDMARKS_MM[5] * 0.001),
            "large_ring_length": float(ACTUAL_SEGMENTS_MM[5] * 0.001),
            "large_ring_far_edge_to_head": float(np.sum(ACTUAL_SEGMENTS_MM[6:]) * 0.001),
            "axial_segments_s1_to_s8": [float(value * 0.001) for value in ACTUAL_SEGMENTS_MM],
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
            "landmark_names": ["tail_end", *[f"s{index}_end" for index in range(1, 9)]],
            "original_landmarks_from_tail_mm": ORIGINAL_LANDMARKS_MM.tolist(),
            "actual_landmarks_from_tail_mm": ACTUAL_LANDMARKS_MM.tolist(),
            "original_segment_lengths_mm": ORIGINAL_SEGMENTS_MM.tolist(),
            "actual_segment_lengths_mm": ACTUAL_SEGMENTS_MM.tolist(),
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
        f"- S1-S8 lengths, mm: `{ACTUAL_SEGMENTS_MM.tolist()}`.",
        f"- Tail section (S1) length: `{ACTUAL_SEGMENTS_MM[0]:.1f} mm`.",
        f"- Tail end to large-ring near edge (S1-S5): `{ACTUAL_LANDMARKS_MM[5]:.1f} mm`.",
        f"- Large-ring axial length (S6): `{ACTUAL_SEGMENTS_MM[5]:.1f} mm`.",
        f"- Overall length: `{ACTUAL_LANDMARKS_MM[-1]:.1f} mm`.",
        f"- Large-ring far edge to head (S7+S8): `{np.sum(ACTUAL_SEGMENTS_MM[6:]):.1f} mm`.",
        "",
        "## Axial Mapping",
        f"- Original landmarks from tail, mm: `{ORIGINAL_LANDMARKS_MM.tolist()}`",
        f"- Actual landmarks from tail, mm: `{ACTUAL_LANDMARKS_MM.tolist()}`",
        f"- Segment scale factors: `{np.round(scales, 8).tolist()}`",
        "- Only the tail-to-head coordinate changes; every transverse coordinate is copied unchanged.",
        f"- Mapped grasp origin from tail: `{actual_grasp_u_mm:.6f} mm`.",
        "",
        "## Manual Review",
        "- Confirm that S2 is intentionally axially compressed from 18.0 mm to 1.0 mm.",
        "- Confirm that tail center refers to the tail end-face center, as interpreted above.",
        f"- Confirm that the large ring (S6) is bounded by {ACTUAL_LANDMARKS_MM[5]:.1f} mm and {ACTUAL_LANDMARKS_MM[6]:.1f} mm.",
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

    colors = ["#067647", "#b54708", "#b42318", "#6941c6", "#175cd3", "#c11574", "#026aa2", "#475467"]

    def draw_side_panel(
        ax,
        u_mm: np.ndarray,
        transverse_mm: np.ndarray,
        landmarks_mm: np.ndarray,
        segments_mm: np.ndarray,
        title: str,
        point_color: str,
        overall_color: str,
    ) -> None:
        ax.scatter(u_mm, transverse_mm, s=0.32, color=point_color, alpha=0.28, rasterized=True)
        for value in landmarks_mm:
            ax.axvline(value, ymin=0.04, ymax=0.84, color=overall_color, lw=0.75, ls="--", alpha=0.55)
        for index, (start, end, length, color) in enumerate(
            zip(landmarks_mm[:-1], landmarks_mm[1:], segments_mm, colors)
        ):
            y = -50.0 - 8.0 * (index % 4)
            draw_dimension(ax, float(start), float(end), y, f"S{index + 1} {length:g} mm", color)
        draw_dimension(
            ax,
            0.0,
            float(landmarks_mm[-1]),
            53.0,
            f"overall {landmarks_mm[-1]:.1f} mm",
            overall_color,
        )
        ax.set_xlim(-6.0, 187.0)
        ax.set_ylim(-82.0, 61.0)
        ax.set_ylabel("Transverse (mm)")
        ax.set_title(title, fontsize=13, weight="bold", color=overall_color, loc="left")
        ax.grid(True, color="#eaecf0", lw=0.65)
        ax.set_aspect("equal", adjustable="box")
        ax.set_anchor("C")

    def draw_end_panel(ax, vertices_mm: np.ndarray, title: str, color: str) -> None:
        y_mm = vertices_mm[:, 1]
        z_mm = vertices_mm[:, 2]
        ax.scatter(y_mm, z_mm, s=0.32, color=color, alpha=0.28, rasterized=True)
        draw_dimension(ax, (-47.0), 47.0, -53.0, "94.0 mm", color)
        ax.annotate("", xy=(53.0, 47.0), xytext=(53.0, -47.0), arrowprops={"arrowstyle": "<->", "color": color, "lw": 1.5})
        ax.text(55.0, 0.0, "94.0 mm", color=color, ha="left", va="center", rotation=90, fontsize=9)
        ax.plot([-47.0, -47.0], [-50.0, -55.0], color=color, lw=0.8)
        ax.plot([47.0, 47.0], [-50.0, -55.0], color=color, lw=0.8)
        ax.plot([50.0, 55.0], [-47.0, -47.0], color=color, lw=0.8)
        ax.plot([50.0, 55.0], [47.0, 47.0], color=color, lw=0.8)
        ax.set_xlim(-60.0, 63.0)
        ax.set_ylim(-60.0, 60.0)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title, fontsize=12, weight="bold", color=color)
        ax.set_xlabel("Y (mm)")
        ax.set_ylabel("Z (mm)")
        ax.grid(True, color="#eaecf0", lw=0.65)
        ax.set_anchor("C")

    fig = plt.figure(figsize=(16, 13))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.42, 1.0], hspace=0.30, wspace=0.22)
    original_side = fig.add_subplot(grid[0, 0])
    actual_side = fig.add_subplot(grid[1, 0])
    original_end = fig.add_subplot(grid[0, 1])
    actual_end = fig.add_subplot(grid[1, 1])

    draw_side_panel(
        original_side,
        original_u,
        original_mm[:, 1],
        ORIGINAL_LANDMARKS_MM,
        ORIGINAL_SEGMENTS_MM,
        "2175B original CAD | axial S1-S8",
        "#667085",
        "#344054",
    )
    draw_side_panel(
        actual_side,
        actual_u,
        actual_mm[:, 1],
        ACTUAL_LANDMARKS_MM,
        ACTUAL_SEGMENTS_MM,
        "plugCAD corrected CAD | axial S1-S8",
        "#1570ef",
        "#175cd3",
    )
    original_side.set_xlabel("Distance from tail end along center axis (mm)")
    actual_side.set_xlabel("Distance from tail end along center axis (mm)")
    draw_end_panel(original_end, original_mm, "2175B end-view envelope", "#667085")
    draw_end_panel(actual_end, actual_mm, "plugCAD end-view envelope", "#1570ef")

    fig.suptitle("2175B original CAD vs plugCAD corrected dimensions", fontsize=18, weight="bold", y=0.965)
    fig.text(
        0.5,
        0.025,
        "End-view envelopes remain identical because only the axial coordinate is adapted.",
        ha="center",
        fontsize=10,
        color="#475467",
    )
    fig.subplots_adjust(top=0.91, bottom=0.07, left=0.06, right=0.955)
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
    points, normals = sample_mesh_points_with_normals(actual_vertices, faces, count=30000, seed=0)
    write_ply_points(ACTUAL_PLY, points, normals)
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
