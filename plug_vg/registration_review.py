"""Interactive HTML review for visible point cloud to grasp-model registration."""

from __future__ import annotations

import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .grasp_model import transform_points


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


_SEMANTIC_MARKER_COLORS = {
    "grasp_center_camera_m": (255, 215, 0),
    "tail_center_camera_m": (0, 170, 255),
    "head_center_camera_m": (200, 0, 255),
}
_SEMANTIC_MARKER_RADIUS_M = 0.006
_SEMANTIC_MARKER_SPACING_M = 0.0015


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
    return PlyHeader(format_name, vertex_count, vertex_properties, sum(len(line) for line in lines))


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
    if not {"x", "y", "z"}.issubset(data.dtype.names or ()):
        raise ValueError(f"PLY vertex properties must contain x/y/z: {path}")
    points = np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float64, copy=False)
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    colors = None
    if {"red", "green", "blue"}.issubset(data.dtype.names or ()):
        colors = np.column_stack([data["red"], data["green"], data["blue"]]).astype(np.float64, copy=False) / 255.0
        colors = colors[finite]
    return PointCloud(points=points, colors=colors)


def _sample_points(points: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if len(points) <= max_points:
        return points
    rng = np.random.default_rng(seed)
    return points[np.sort(rng.choice(len(points), size=max_points, replace=False))]


def _sample_colors(colors: np.ndarray | None, keep_count: int, original_count: int, seed: int) -> list[str] | None:
    if colors is None or original_count == 0:
        return None
    colors = np.asarray(colors, dtype=np.float64)
    if len(colors) != original_count:
        return None
    if original_count <= keep_count:
        sampled = colors
    else:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(original_count, size=keep_count, replace=False))
        sampled = colors[idx]
    if float(np.nanmax(sampled)) <= 1.0:
        sampled = sampled * 255.0
    sampled = np.clip(np.round(sampled), 0, 255).astype(int)
    return [f"rgb({r},{g},{b})" for r, g, b in sampled.tolist()]


def _points_json(points: np.ndarray, digits: int = 5) -> list[list[float]]:
    rounded = np.round(np.asarray(points, dtype=np.float64), digits)
    rounded[np.isclose(rounded, 0.0, atol=10.0 ** -digits)] = 0.0
    return rounded.tolist()


def _cad_bbox_corners(model_points: np.ndarray, t_camera_grasp: np.ndarray | None) -> np.ndarray:
    """Return the eight grasp-frame CAD AABB corners transformed into the camera frame."""

    points = np.asarray(model_points, dtype=np.float64)
    if t_camera_grasp is None or not len(points):
        return np.empty((0, 3), dtype=np.float64)
    lo = np.min(points, axis=0)
    hi = np.max(points, axis=0)
    corners_grasp = np.asarray(
        [
            [lo[0], lo[1], lo[2]],
            [hi[0], lo[1], lo[2]],
            [hi[0], hi[1], lo[2]],
            [lo[0], hi[1], lo[2]],
            [lo[0], lo[1], hi[2]],
            [hi[0], lo[1], hi[2]],
            [hi[0], hi[1], hi[2]],
            [lo[0], hi[1], hi[2]],
        ],
        dtype=np.float64,
    )
    return transform_points(corners_grasp, t_camera_grasp)


_QUALITY_KEYS = (
    "candidate",
    "fitness",
    "inlier_rmse",
    "coarse_candidate",
    "coarse_fitness",
    "coarse_rmse",
    "roll_equivalent_candidates",
    "directed_axis_groups",
    "opposite_axis_candidate",
    "opposite_axis_fitness",
    "opposite_axis_inlier_rmse",
    "opposite_axis_fitness_gap",
    "opposite_axis_rmse_gap_m",
    "reason",
)


def _registration_group(
    visible_points: np.ndarray,
    model_points: np.ndarray,
    t_camera_grasp: np.ndarray | None,
    semantic_camera: dict[str, list[float]] | None,
    status: str,
    reason: str | None,
    quality: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    visible_sample = _sample_points(visible_points, 12000, seed + 1)
    model_sample = _sample_points(model_points, 16000, seed + 2)
    model_registered = (
        transform_points(model_sample, t_camera_grasp)
        if t_camera_grasp is not None
        else np.empty((0, 3), dtype=np.float64)
    )
    coarse_raw = quality.get("t_grasp_camera_coarse")
    coarse_grasp_camera = None if coarse_raw is None else np.asarray(coarse_raw, dtype=np.float64)
    if coarse_grasp_camera is not None and coarse_grasp_camera.shape != (4, 4):
        coarse_grasp_camera = None
    coarse_transform = None if coarse_grasp_camera is None else np.linalg.inv(coarse_grasp_camera)
    model_coarse = (
        transform_points(model_sample, coarse_transform)
        if coarse_transform is not None
        else np.empty((0, 3), dtype=np.float64)
    )
    axes = None
    if t_camera_grasp is not None:
        axes = {
            "x": _points_json([t_camera_grasp[:3, 0]], digits=6)[0],
            "y": _points_json([t_camera_grasp[:3, 1]], digits=6)[0],
            "z": _points_json([t_camera_grasp[:3, 2]], digits=6)[0],
        }
    return {
        "status": status,
        "reason": reason,
        "quality": {key: quality.get(key) for key in _QUALITY_KEYS if key in quality},
        "visible": _points_json(visible_sample),
        "model": _points_json(model_registered),
        "cad_bbox": _points_json(_cad_bbox_corners(model_points, t_camera_grasp), digits=6),
        "coarse_model": _points_json(model_coarse),
        "coarse_cad_bbox": _points_json(_cad_bbox_corners(model_points, coarse_transform), digits=6),
        "semantic": semantic_camera or {},
        "axes": axes,
    }


def write_registration_comparison_ply(
    output_path: Path,
    visible_points: np.ndarray,
    model_points: np.ndarray,
    t_camera_grasp: np.ndarray | None,
    semantic_camera: dict[str, list[float]] | None = None,
) -> None:
    """Write observed/model points and colored semantic markers in one camera-frame PLY."""

    visible = np.asarray(visible_points, dtype=np.float64)
    visible = visible[np.isfinite(visible).all(axis=1)]
    model = np.asarray(model_points, dtype=np.float64)
    model = model[np.isfinite(model).all(axis=1)]
    registered = (
        transform_points(model, np.asarray(t_camera_grasp, dtype=np.float64))
        if t_camera_grasp is not None
        else np.empty((0, 3), dtype=np.float64)
    )
    marker_groups: list[tuple[np.ndarray, tuple[int, int, int]]] = []
    if semantic_camera:
        offsets = np.arange(
            -_SEMANTIC_MARKER_RADIUS_M,
            _SEMANTIC_MARKER_RADIUS_M + _SEMANTIC_MARKER_SPACING_M * 0.5,
            _SEMANTIC_MARKER_SPACING_M,
            dtype=np.float64,
        )
        xx, yy, zz = np.meshgrid(offsets, offsets, offsets, indexing="ij")
        ball_offsets = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
        ball_offsets[np.abs(ball_offsets) < 1e-12] = 0.0
        ball_offsets = ball_offsets[
            np.einsum("ij,ij->i", ball_offsets, ball_offsets)
            <= _SEMANTIC_MARKER_RADIUS_M**2 + 1e-15
        ]
        for key, color in _SEMANTIC_MARKER_COLORS.items():
            raw_center = semantic_camera.get(key)
            if raw_center is None:
                continue
            center = np.asarray(raw_center, dtype=np.float64)
            if center.shape != (3,) or not np.isfinite(center).all():
                continue
            marker_groups.append((center + ball_offsets, color))

    marker_count = sum(len(points) for points, _ in marker_groups)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="ascii") as stream:
        stream.write("ply\nformat ascii 1.0\n")
        stream.write("comment frame: camera_rgb; units: meter\n")
        stream.write("comment green: D2RGB visible points; red: registered CAD points\n")
        stream.write("comment semantic markers: yellow=grasp center; cyan=tail center; magenta=head center\n")
        stream.write(f"element vertex {len(visible) + len(registered) + marker_count}\n")
        stream.write("property float x\nproperty float y\nproperty float z\n")
        stream.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        stream.write("end_header\n")
        for x, y, z in visible:
            stream.write(f"{x:.9f} {y:.9f} {z:.9f} 18 183 106\n")
        for x, y, z in registered:
            stream.write(f"{x:.9f} {y:.9f} {z:.9f} 240 68 56\n")
        for marker_points, (red, green, blue) in marker_groups:
            for x, y, z in marker_points:
                stream.write(f"{x:.9f} {y:.9f} {z:.9f} {red} {green} {blue}\n")


def write_interactive_review_html(
    output_path: Path,
    scene_points: np.ndarray,
    scene_colors: np.ndarray | None,
    visible_points: np.ndarray,
    model_points: np.ndarray,
    t_camera_grasp: np.ndarray | None,
    semantic_camera: dict[str, list[float]] | None,
    status: str,
    reason: str | None,
    quality: dict[str, Any],
    seed: int,
    camera: dict[str, Any] | None = None,
) -> None:
    scene_sample = _sample_points(scene_points, 30000, seed)
    scene_color_sample = _sample_colors(scene_colors, len(scene_sample), len(scene_points), seed)
    registration = _registration_group(
        visible_points, model_points, t_camera_grasp, semantic_camera, status, reason, quality, seed
    )
    camera_view = None
    if camera is not None and len(visible_points):
        frustum_source = np.asarray(scene_points if len(scene_points) else visible_points, dtype=np.float64)
        finite_z = frustum_source[np.isfinite(frustum_source).all(axis=1), 2]
        if len(finite_z):
            plane_z = float(np.percentile(finite_z, 99.5))
            plane_z_source = "scene_depth_p99_5_m"
        else:
            plane_z = float(np.max(np.asarray(visible_points, dtype=np.float64)[:, 2]))
            plane_z_source = "visible_depth_max_m"
        width = float(camera["image_width"])
        height = float(camera["image_height"])
        fx = float(camera["fx"])
        fy = float(camera["fy"])
        cx = float(camera["cx"])
        cy = float(camera["cy"])

        def pixel_to_camera(u: float, v: float) -> list[float]:
            return [
                (u - cx) * plane_z / fx,
                (v - cy) * plane_z / fy,
                plane_z,
            ]

        camera_view = {
            "origin": [0.0, 0.0, 0.0],
            "plane_z_m": round(float(plane_z), 6),
            "plane_z_source": plane_z_source,
            "rgb_plane": [
                pixel_to_camera(0.0, 0.0),
                pixel_to_camera(width - 1.0, 0.0),
                pixel_to_camera(width - 1.0, height - 1.0),
                pixel_to_camera(0.0, height - 1.0),
            ],
        }
    view_data = {
        "scene": _points_json(scene_sample),
        "scene_colors": scene_color_sample,
        "camera": camera_view,
        "registration": registration,
    }
    data_json = json.dumps(view_data, ensure_ascii=False, separators=(",", ":"))
    title = "Visible Grasp Registration 3D Review"
    escaped_title = html.escape(title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escaped_title}</title>
<style>
  body {{ margin: 0; overflow: hidden; font-family: Arial, sans-serif; color: #111827; }}
  #scene {{ width: 100vw; height: 100vh; display: block; background: #f8fafc; cursor: grab; }}
  #scene:active {{ cursor: grabbing; }}
  #panel {{
    position: fixed; top: 14px; right: 14px; width: 330px; max-height: calc(100vh - 28px);
    overflow: auto; background: rgba(255,255,255,.94); border: 1px solid #d0d5dd;
    border-radius: 8px; padding: 12px 14px; box-shadow: 0 12px 32px rgba(15,23,42,.16);
    font-size: 13px;
  }}
  h1 {{ font-size: 16px; margin: 0 0 10px; }}
  .row {{ display: grid; grid-template-columns: 118px minmax(0,1fr); gap: 6px; margin: 5px 0; }}
  .key {{ color: #667085; }}
  .value {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; overflow-wrap: anywhere; }}
  .legend {{ display: grid; gap: 6px; margin: 10px 0; }}
  .views {{ display: grid; grid-template-columns: 1fr 1fr; gap: 6px; margin: 10px 0; }}
  .controls {{ display: grid; gap: 8px; margin: 10px 0; }}
  .control-row {{ display: grid; grid-template-columns: 112px minmax(0,1fr); gap: 8px; align-items: center; }}
  .checks {{ display: grid; grid-template-columns: 1fr 1fr; gap: 6px 10px; }}
  label {{ font-size: 12px; color: #344054; }}
  input[type="range"] {{ width: 100%; }}
  select {{ width: 100%; border: 1px solid #cbd5e1; border-radius: 6px; padding: 6px; background: #ffffff; }}
  button {{
    border: 1px solid #cbd5e1; border-radius: 6px; background: #ffffff; color: #111827;
    padding: 7px 8px; font-size: 12px; cursor: pointer;
  }}
  button:hover {{ background: #f1f5f9; }}
  button.active {{ background: #155eef; border-color: #155eef; color: #ffffff; }}
  .swatch {{ display: inline-block; width: 10px; height: 10px; margin-right: 7px; border-radius: 2px; }}
  .note {{ color: #667085; line-height: 1.42; }}
</style>
</head>
<body>
<canvas id="scene"></canvas>
<div id="panel">
  <h1>{escaped_title}</h1>
  <div class="legend">
    <div><span class="swatch" style="background:#94a3b8"></span>Full scene point cloud</div>
    <div><span class="swatch" style="background:#12b76a"></span>Visible plug points</div>
    <div><span class="swatch" style="background:#f04438"></span>Registered CAD model points</div>
    <div><span class="swatch" style="background:#7f56d9"></span>CAD oriented bounding box</div>
    <div><span class="swatch" style="background:#06aed4"></span>Coarse PCA CAD model / box</div>
    <div><span class="swatch" style="background:#fdb022"></span>Camera frustum / RGB image plane</div>
    <div><span class="swatch" style="background:#d92d20"></span>+X tail->head</div>
    <div><span class="swatch" style="background:#079455"></span>+Y closing</div>
    <div><span class="swatch" style="background:#1570ef"></span>+Z approach</div>
  </div>
  <div class="views">
    <button type="button" onclick="setView('front')">Camera Front</button>
    <button type="button" onclick="setView('top')">Camera Top</button>
    <button type="button" onclick="setView('side')">Camera Side</button>
    <button type="button" onclick="setView('iso')">Iso View</button>
    <button type="button" onclick="resetViewer()">Reset View</button>
    <button type="button" onclick="setFocus('visible')">Focus Visible</button>
    <button type="button" onclick="setFocus('cad')">Focus CAD</button>
  </div>
  <div class="controls">
    <div class="control-row">
      <label for="pointSize">Point size</label>
      <input id="pointSize" type="range" min="0.5" max="4" step="0.25" value="1">
    </div>
    <div class="control-row">
      <label for="colorMode">Scene color</label>
      <select id="colorMode">
        <option value="rgb">RGB if available</option>
        <option value="depth">Depth color</option>
        <option value="solid">Solid gray</option>
      </select>
    </div>
    <div class="checks">
      <label><input id="layerScene" type="checkbox" checked> Scene</label>
      <label><input id="layerVisible" type="checkbox" checked> Visible</label>
      <label><input id="layerModel" type="checkbox" checked> CAD</label>
      <label><input id="layerCadBox" type="checkbox" checked> CAD box</label>
      <label><input id="layerCoarseCad" type="checkbox" checked> Coarse CAD</label>
      <label><input id="layerCamera" type="checkbox" checked> Camera</label>
      <label><input id="layerAxes" type="checkbox" checked> Axes</label>
      <label><input id="projectionMode" type="checkbox"> Perspective</label>
    </div>
  </div>
  <div id="info"></div>
  <p class="note">Left-drag rotates, right-drag pans, and the wheel zooms.</p>
</div>
<script>
const data = {data_json};
const canvas = document.getElementById("scene");
const ctx = canvas.getContext("2d");
const info = document.getElementById("info");
const ui = {{
  pointSize: document.getElementById("pointSize"),
  colorMode: document.getElementById("colorMode"),
  layerScene: document.getElementById("layerScene"),
  layerVisible: document.getElementById("layerVisible"),
  layerModel: document.getElementById("layerModel"),
  layerCadBox: document.getElementById("layerCadBox"),
  layerCoarseCad: document.getElementById("layerCoarseCad"),
  layerCamera: document.getElementById("layerCamera"),
  layerAxes: document.getElementById("layerAxes"),
  projectionMode: document.getElementById("projectionMode")
}};
// Start in the RGB optical view.  The camera frame follows the image convention:
// +X points right, +Y points down and +Z points forward into the scene.
const state = {{ yaw: 0.0, pitch: 0.0, zoom: 1, panX: 0, panY: 0, focusMode: "all", dragging: false, dragMode: "rotate", lastX: 0, lastY: 0 }};
const viewPresets = {{
  front: {{ yaw: 0.0, pitch: 0.0, zoom: 1.0 }},
  top: {{ yaw: 0.0, pitch: -1.5708, zoom: 1.0 }},
  side: {{ yaw: 1.5708, pitch: 0.0, zoom: 1.0 }},
  iso: {{ yaw: -0.65, pitch: 0.55, zoom: 1.0 }}
}};
function setView(name) {{
  const preset = viewPresets[name] || viewPresets.iso;
  state.yaw = preset.yaw; state.pitch = preset.pitch; state.zoom = preset.zoom; state.panX = 0; state.panY = 0;
  draw();
}}
function activeGroup() {{ return data.registration; }}
function setFocus(mode) {{ state.focusMode = mode; state.panX = 0; state.panY = 0; state.zoom = 1; draw(); }}
function resetViewer() {{ state.focusMode = "all"; state.panX = 0; state.panY = 0; state.zoom = 1; setView("front"); ui.pointSize.value = "1"; ui.colorMode.value = "rgb"; for (const key of ["layerScene","layerVisible","layerModel","layerCadBox","layerCoarseCad","layerCamera","layerAxes"]) ui[key].checked = true; ui.projectionMode.checked = false; fillInfo(); draw(); }}
function row(k, v) {{ return `<div class="row"><div class="key">${{k}}</div><div class="value">${{v ?? "n/a"}}</div></div>`; }}
function fmt(v, d=6) {{ return v === null || v === undefined || Number.isNaN(Number(v)) ? "n/a" : Number(v).toFixed(d); }}
function fillInfo() {{
  const group = activeGroup(), q = group.quality || {{}};
  info.innerHTML = [
    row("status", group.status),
    row("reason", group.reason || q.reason || "n/a"),
    row("candidate", q.candidate),
    row("fitness", fmt(q.fitness, 5)),
    row("rmse m", fmt(q.inlier_rmse, 6)),
    row("coarse candidate", q.coarse_candidate),
    row("coarse fitness", fmt(q.coarse_fitness, 5)),
    row("coarse rmse m", fmt(q.coarse_rmse, 6)),
    row("roll-equivalent", q.roll_equivalent_candidates),
    row("axis groups", q.directed_axis_groups),
    row("opposite axis", q.opposite_axis_candidate),
    row("opposite fitness", fmt(q.opposite_axis_fitness, 5)),
    row("axis fitness gap", fmt(q.opposite_axis_fitness_gap, 6)),
    row("axis rmse gap", fmt(q.opposite_axis_rmse_gap_m, 6)),
    row("scene pts", data.scene.length),
    row("visible pts", group.visible.length),
    row("model pts", group.model.length)
  ].join("");
}}
function allPoints() {{ const group = activeGroup(); return [...group.visible, ...group.model, ...data.scene.slice(0, Math.min(data.scene.length, 5000))]; }}
function focusPoints() {{
  const group = activeGroup();
  if (state.focusMode === "visible" && group.visible.length) return group.visible;
  if (state.focusMode === "cad" && group.model.length) return group.model;
  return allPoints();
}}
function add(a,b) {{ return [a[0]+b[0], a[1]+b[1], a[2]+b[2]]; }}
function sub(a,b) {{ return [a[0]-b[0], a[1]-b[1], a[2]-b[2]]; }}
function mul(a,s) {{ return [a[0]*s, a[1]*s, a[2]*s]; }}
function unit(v) {{ const n = Math.hypot(v[0],v[1],v[2]) || 1; return [v[0]/n,v[1]/n,v[2]/n]; }}
function center() {{
  const pts = focusPoints();
  const lo = [Infinity,Infinity,Infinity], hi = [-Infinity,-Infinity,-Infinity];
  for (const p of pts) for (let i=0;i<3;i++) {{ lo[i]=Math.min(lo[i],p[i]); hi[i]=Math.max(hi[i],p[i]); }}
  return mul(add(lo, hi), 0.5);
}}
function view(p, c) {{
  const q = sub(p, c);
  const cy = Math.cos(state.yaw), sy = Math.sin(state.yaw), cp = Math.cos(state.pitch), sp = Math.sin(state.pitch);
  const x1 = cy*q[0] + sy*q[1], y1 = -sy*q[0] + cy*q[1], z1 = q[2];
  return [x1, cp*y1 - sp*z1, sp*y1 + cp*z1];
}}
function scale(c) {{
  const pts = focusPoints().map(p => view(p, c));
  let span = 0.02;
  for (let i=0;i<3;i++) {{
    const vals = pts.map(p => p[i]);
    span = Math.max(span, Math.max(...vals) - Math.min(...vals));
  }}
  return Math.min(canvas.width, canvas.height) * 0.70 / span * state.zoom;
}}
function project(p, c, s) {{
  const v = view(p, c);
  let factor = 1.0;
  if (ui.projectionMode.checked) factor = 1.0 / Math.max(0.18, 1.0 + v[2] * 1.6);
  // Canvas Y and RGB pixel Y both grow downwards.  Keeping the positive sign
  // makes Camera Front match the source RGB image instead of mirroring it.
  return {{ x: canvas.width/2 + state.panX + v[0]*s*factor, y: canvas.height/2 + state.panY + v[1]*s*factor, z: v[2] }};
}}
function depthColor(z, lo, hi) {{
  const t = Math.max(0, Math.min(1, (z - lo) / Math.max(1e-6, hi - lo)));
  const r = Math.round(40 + 210 * t), g = Math.round(180 - 120 * Math.abs(t - 0.5)), b = Math.round(230 - 190 * t);
  return `rgb(${{r}},${{g}},${{b}})`;
}}
function depthRange(points) {{
  if (!points.length) return [0,1];
  let lo = Infinity, hi = -Infinity;
  for (const p of points) {{ lo = Math.min(lo, p[2]); hi = Math.max(hi, p[2]); }}
  return [lo, hi];
}}
function drawPoints(points, c, s, color, radius, alpha, colors=null, colorMode="solid") {{
  const pointScale = Number(ui.pointSize.value || 1);
  const [zLo, zHi] = depthRange(points);
  ctx.globalAlpha = alpha;
  for (let i=0; i<points.length; i++) {{
    const p = project(points[i], c, s);
    if (colorMode === "depth") ctx.fillStyle = depthColor(points[i][2], zLo, zHi);
    else if (colorMode === "rgb" && colors) ctx.fillStyle = colors[i];
    else ctx.fillStyle = color;
    const r = radius * pointScale;
    ctx.fillRect(p.x - r, p.y - r, r*2, r*2);
  }}
  ctx.globalAlpha = 1;
}}
function drawLine(a,b,c,s,color,label,width=3) {{
  const pa = project(a,c,s), pb = project(b,c,s);
  ctx.strokeStyle = color; ctx.lineWidth = width;
  ctx.beginPath(); ctx.moveTo(pa.x,pa.y); ctx.lineTo(pb.x,pb.y); ctx.stroke();
  ctx.fillStyle = color; ctx.font = "13px ui-monospace, monospace"; ctx.fillText(label, pb.x+7, pb.y-7);
}}
function drawPolygon(points, c, s, fill, stroke, label=null) {{
  if (!points || points.length < 3) return;
  const projected = points.map(p => project(p, c, s));
  ctx.save();
  ctx.globalAlpha = 0.025;
  ctx.fillStyle = fill;
  ctx.beginPath();
  ctx.moveTo(projected[0].x, projected[0].y);
  for (const p of projected.slice(1)) ctx.lineTo(p.x, p.y);
  ctx.closePath();
  ctx.fill();
  ctx.globalAlpha = 0.55;
  ctx.strokeStyle = stroke;
  ctx.lineWidth = 2;
  ctx.stroke();
  if (label) {{
    ctx.fillStyle = stroke;
    ctx.font = "13px ui-monospace, monospace";
    ctx.fillText(label, projected[0].x + 8, projected[0].y + 18);
  }}
  ctx.restore();
}}
function drawCameraFrustum(c, s) {{
  if (!data.camera || !data.camera.rgb_plane) return;
  const origin = data.camera.origin || [0,0,0];
  const plane = data.camera.rgb_plane;
  drawPolygon(plane, c, s, "#fdb022", "#9a6700", "RGB image plane");
  for (const corner of plane) drawLine(origin, corner, c, s, "#9a6700", "", 1.0);
  const center = mul(add(add(plane[0], plane[1]), add(plane[2], plane[3])), 0.25);
  drawLine(origin, center, c, s, "#f79009", "+Z camera view", 3);
}}
function drawCadBoundingBox(c, s) {{
  const p = activeGroup().cad_bbox;
  if (!p || p.length !== 8) return;
  const edges = [
    [0,1],[1,2],[2,3],[3,0],
    [4,5],[5,6],[6,7],[7,4],
    [0,4],[1,5],[2,6],[3,7]
  ];
  for (const [a,b] of edges) drawLine(p[a], p[b], c, s, "#7f56d9", "", 2.2);
  const anchor = project(p[6], c, s);
  ctx.fillStyle = "#6941c6";
  ctx.font = "13px ui-monospace, monospace";
  ctx.fillText("CAD bbox", anchor.x + 7, anchor.y - 7);
}}
function drawCoarseCad(c, s) {{
  const group = activeGroup();
  if (group.coarse_model && group.coarse_model.length) {{
    drawPoints(group.coarse_model, c, s, "#06aed4", 1.0, 0.62, null, "solid");
  }}
  const p = group.coarse_cad_bbox;
  if (!p || p.length !== 8) return;
  const edges = [
    [0,1],[1,2],[2,3],[3,0],
    [4,5],[5,6],[6,7],[7,4],
    [0,4],[1,5],[2,6],[3,7]
  ];
  for (const [a,b] of edges) drawLine(p[a], p[b], c, s, "#067a9c", "", 1.6);
  const anchor = project(p[6], c, s);
  ctx.fillStyle = "#067a9c";
  ctx.font = "13px ui-monospace, monospace";
  ctx.fillText("Coarse CAD", anchor.x + 7, anchor.y + 16);
}}
function drawScreenLine(a,b,color,label,width=3) {{
  ctx.strokeStyle = color; ctx.lineWidth = width;
  ctx.beginPath(); ctx.moveTo(a.x,a.y); ctx.lineTo(b.x,b.y); ctx.stroke();
  ctx.fillStyle = color; ctx.font = "12px ui-monospace, monospace"; ctx.fillText(label, b.x+6, b.y-6);
}}
function drawCameraGizmo() {{
  const origin = {{ x: 58, y: canvas.height - 58 }};
  const length = 42;
  const axes = [
    {{ label: "+X right", vector: [1,0,0], color: "#d92d20" }},
    {{ label: "+Y down", vector: [0,1,0], color: "#079455" }},
    {{ label: "+Z forward", vector: [0,0,1], color: "#1570ef" }}
  ];
  ctx.save();
  ctx.globalAlpha = 0.96;
  ctx.fillStyle = "rgba(255,255,255,.86)";
  ctx.strokeStyle = "#cbd5e1";
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.roundRect(12, canvas.height - 126, 185, 104, 8); ctx.fill(); ctx.stroke();
  ctx.fillStyle = "#475467"; ctx.font = "12px Arial, sans-serif"; ctx.fillText("Camera frame", 24, canvas.height - 102);
  for (const axis of axes) {{
    const v = view(axis.vector, [0,0,0]);
    const n = Math.hypot(v[0], v[1]) || 1;
    const end = {{ x: origin.x + v[0] / n * length, y: origin.y + v[1] / n * length }};
    drawScreenLine(origin, end, axis.color, axis.label, 3);
  }}
  ctx.fillStyle = "#111827"; ctx.beginPath(); ctx.arc(origin.x, origin.y, 3, 0, Math.PI*2); ctx.fill();
  ctx.restore();
}}
function semanticPoint(name) {{ const semantic = activeGroup().semantic; return semantic ? semantic[name + "_camera_m"] : null; }}
function drawAxes(c, s) {{
  const group = activeGroup();
  const o = semanticPoint("grasp_center");
  if (!o || !group.axes) return;
  const length = 0.07;
  const tail = semanticPoint("tail_center"), head = semanticPoint("head_center");
  if (tail && head) drawLine(tail, head, c, s, "#d92d20", "tail->head", 4);
  drawLine(o, add(o, mul(unit(group.axes.x), length)), c, s, "#d92d20", "+X", 3);
  drawLine(o, add(o, mul(unit(group.axes.y), length)), c, s, "#079455", "+Y", 3);
  drawLine(o, add(o, mul(unit(group.axes.z), length)), c, s, "#1570ef", "+Z", 3);
}}
function draw() {{
  const r = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, Math.round(r.width)); canvas.height = Math.max(1, Math.round(r.height));
  ctx.clearRect(0,0,canvas.width,canvas.height);
  const group = activeGroup();
  const c = center(), s = scale(c);
  if (ui.layerScene.checked) drawPoints(data.scene, c, s, "#94a3b8", 0.7, 0.18, data.scene_colors, ui.colorMode.value);
  if (ui.layerCamera.checked) drawCameraFrustum(c, s);
  if (ui.layerVisible.checked) drawPoints(group.visible, c, s, "#12b76a", 1.4, 0.82, null, "solid");
  if (ui.layerCoarseCad.checked) drawCoarseCad(c, s);
  if (ui.layerModel.checked) drawPoints(group.model, c, s, "#f04438", 1.2, 0.78, null, "solid");
  if (ui.layerCadBox.checked) drawCadBoundingBox(c, s);
  if (ui.layerAxes.checked) drawAxes(c, s);
  drawCameraGizmo();
}}
canvas.addEventListener("contextmenu", e => e.preventDefault());
canvas.addEventListener("mousedown", e => {{ state.dragging=true; state.dragMode = e.button === 2 ? "pan" : "rotate"; state.lastX=e.clientX; state.lastY=e.clientY; }});
window.addEventListener("mouseup", () => state.dragging=false);
window.addEventListener("mousemove", e => {{
  if (!state.dragging) return;
  if (state.dragMode === "pan") {{
    state.panX += e.clientX - state.lastX; state.panY += e.clientY - state.lastY;
  }} else {{
    state.yaw += (e.clientX-state.lastX)*0.008; state.pitch = Math.max(-1.45, Math.min(1.45, state.pitch+(e.clientY-state.lastY)*0.008));
  }}
  state.lastX=e.clientX; state.lastY=e.clientY; draw();
}});
canvas.addEventListener("wheel", e => {{ e.preventDefault(); state.zoom *= Math.exp(-e.deltaY*0.001); state.zoom=Math.max(.2,Math.min(10,state.zoom)); draw(); }}, {{passive:false}});
window.addEventListener("resize", draw);
for (const control of Object.values(ui)) control.addEventListener("input", draw);
fillInfo(); draw();
</script>
</body>
</html>
""",
        encoding="utf-8",
    )


__all__ = [
    "PointCloud",
    "PlyHeader",
    "parse_ply_header",
    "read_realsense_binary_ply",
    "write_interactive_review_html",
    "write_registration_comparison_ply",
]
