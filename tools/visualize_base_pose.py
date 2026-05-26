#!/usr/bin/env python3
"""Generate an interactive base-frame 3D view for a saved 6D grasp pose."""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from typing import Any


DEFAULT_AXIS_LENGTH_M = 0.1
DEFAULT_MODEL_LENGTH_M = 0.085
DEFAULT_MODEL_WIDTH_M = 0.055
DEFAULT_MODEL_THICKNESS_M = 0.055


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, required=True, help="Path to an OK *_6d_base.json result.")
    parser.add_argument("--output", type=Path, default=None, help="Output HTML path. Defaults to *_base_pose_view.html.")
    parser.add_argument("--axis-length", type=float, default=DEFAULT_AXIS_LENGTH_M, help="Axis length in meters.")
    parser.add_argument("--model-length", type=float, default=DEFAULT_MODEL_LENGTH_M, help="Simplified target length along +X, meters.")
    parser.add_argument("--model-width", type=float, default=DEFAULT_MODEL_WIDTH_M, help="Simplified target width along +Y, meters.")
    parser.add_argument("--model-thickness", type=float, default=DEFAULT_MODEL_THICKNESS_M, help="Simplified target thickness along +Z, meters.")
    return parser.parse_args()


def default_output_path(json_path: Path) -> Path:
    stem = json_path.stem
    for suffix in ("_6d_base", "_3d"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return json_path.with_name(f"{stem}_base_pose_view.html")


def load_result(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def as_float_vector(value: Any, length: int, name: str) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{name} must be a list of {length} numbers")
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain only numbers") from exc


def as_rotation_matrix(value: Any, name: str) -> list[list[float]]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{name} must be a 3x3 matrix")
    return [as_float_vector(row, 3, f"{name}[{index}]") for index, row in enumerate(value)]


def optional_float_vector(value: Any, length: int, name: str) -> list[float] | None:
    if value is None:
        return None
    return as_float_vector(value, length, name)


def build_window_view_data(result: dict[str, Any]) -> dict[str, Any] | None:
    window = result.get("window_geometry_base")
    if not isinstance(window, dict):
        return None

    corners = window.get("corners_base_m") or {}
    full_corners = []
    if isinstance(corners, dict):
        for index in range(1, 5):
            full_corners.append(as_float_vector(corners.get(f"W{index}"), 3, f"window_geometry_base.corners_base_m.W{index}"))
    else:
        return None

    center = optional_float_vector(window.get("center_base_m"), 3, "window_geometry_base.center_base_m")
    x_axis = optional_float_vector(window.get("x_window_base"), 3, "window_geometry_base.x_window_base")
    y_axis = optional_float_vector(window.get("y_window_base"), 3, "window_geometry_base.y_window_base")
    effective_width = window.get("effective_width_m")
    effective_height = window.get("effective_height_m")
    effective_corners = None
    if center is not None and x_axis is not None and y_axis is not None and effective_width is not None and effective_height is not None:
        ex = float(effective_width) * 0.5
        ey = float(effective_height) * 0.5
        effective_corners = [
            [center[i] - x_axis[i] * ex - y_axis[i] * ey for i in range(3)],
            [center[i] + x_axis[i] * ex - y_axis[i] * ey for i in range(3)],
            [center[i] + x_axis[i] * ex + y_axis[i] * ey for i in range(3)],
            [center[i] - x_axis[i] * ex + y_axis[i] * ey for i in range(3)],
        ]

    candidates = []
    raw_candidates = result.get("window_constrained_grasp_candidates") or []
    if isinstance(raw_candidates, list):
        for item in raw_candidates:
            if not isinstance(item, dict):
                continue
            point = optional_float_vector(item.get("window_point_base"), 3, "candidate.window_point_base")
            z_axis = optional_float_vector(item.get("z_approach_base"), 3, "candidate.z_approach_base")
            if point is None or z_axis is None:
                continue
            candidates.append(
                {
                    "index": item.get("index"),
                    "window_point": point,
                    "z_approach": z_axis,
                    "score": item.get("score_visual_geometry"),
                }
            )

    return {
        "source": window.get("source"),
        "full_corners": full_corners,
        "effective_corners": effective_corners,
        "center": center,
        "normal": optional_float_vector(window.get("normal_base"), 3, "window_geometry_base.normal_base"),
        "width": window.get("width_m"),
        "height": window.get("height_m"),
        "margin": window.get("margin_m"),
        "effective_width": effective_width,
        "effective_height": effective_height,
        "candidates": candidates,
        "candidate_stats": result.get("window_candidate_stats") or {},
    }


def pose_view_from_dict(pose: dict[str, Any], prefix: str, pose_rad_key: str, pose_deg_key: str) -> dict[str, Any]:
    pose_rad = pose.get(pose_rad_key)
    pose_deg = pose.get(pose_deg_key)
    return {
        "center": as_float_vector(pose.get("translation_m"), 3, f"{prefix}.translation_m"),
        "rotation": as_rotation_matrix(pose.get("rotation_matrix"), f"{prefix}.rotation_matrix"),
        "quaternion_xyzw": pose.get("quaternion_xyzw"),
        "pose_rad": None if pose_rad is None else as_float_vector(pose_rad, 6, f"{prefix}.{pose_rad_key}"),
        "pose_deg": None if pose_deg is None else as_float_vector(pose_deg, 6, f"{prefix}.{pose_deg_key}"),
    }


def build_view_data(result: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    status = result.get("status")
    if status != "ok":
        reason = result.get("reason") or "unknown_reason"
        raise ValueError(f"Result status is {status!r}, expected 'ok' (reason: {reason})")

    base_pose = result.get("grasp_pose_base")
    if not isinstance(base_pose, dict):
        raise ValueError("OK result missing grasp_pose_base")

    reference_pose = pose_view_from_dict(base_pose, "grasp_pose_base", "robot_pose_xyzrpy_m_rad", "robot_pose_xyzrpy_m_deg")
    best_pose = result.get("best_grasp_pose_base")
    if isinstance(best_pose, dict):
        primary_pose = pose_view_from_dict(best_pose, "best_grasp_pose_base", "xyzrpy_m_rad", "xyzrpy_m_deg")
        pose_role = "best_grasp_pose_base"
        candidate_index = best_pose.get("index")
        candidate_score = best_pose.get("score_visual_geometry")
    else:
        primary_pose = reference_pose
        pose_role = "grasp_pose_base"
        candidate_index = None
        candidate_score = None

    return {
        "source_json": str(args.json),
        "input": result.get("input") or {},
        "center": primary_pose["center"],
        "rotation": primary_pose["rotation"],
        "quaternion_xyzw": primary_pose["quaternion_xyzw"],
        "pose_rad": primary_pose["pose_rad"],
        "pose_deg": primary_pose["pose_deg"],
        "pose_role": pose_role,
        "candidate_index": candidate_index,
        "candidate_score": candidate_score,
        "reference_pose": {
            **reference_pose,
            "role": result.get("grasp_pose_base_role", "surface_normal_reference"),
        },
        "quality_score": (result.get("quality") or {}).get("quality_score"),
        "warnings": result.get("warnings") or [],
        "window": build_window_view_data(result),
        "axis_length": float(args.axis_length),
        "model": {
            "length": float(args.model_length),
            "width": float(args.model_width),
            "thickness": float(args.model_thickness),
        },
        "convention": "fixed-axis XYZ RPY: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)",
    }


def render_html(view_data: dict[str, Any]) -> str:
    title = "Base Frame 3D Pose View"
    data_json = json.dumps(view_data, ensure_ascii=False, indent=2)
    escaped_title = html.escape(title)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escaped_title}</title>
<style>
  :root {{
    color-scheme: light;
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    background: #f6f8fb;
    color: #172033;
  }}
  body {{
    margin: 0;
    min-height: 100vh;
    display: grid;
    grid-template-columns: minmax(0, 1fr) 360px;
    overflow: hidden;
  }}
  #scene {{
    width: 100%;
    height: 100vh;
    display: block;
    background: linear-gradient(#fbfcff, #eef3f8);
    cursor: grab;
  }}
  #scene:active {{
    cursor: grabbing;
  }}
  aside {{
    box-sizing: border-box;
    height: 100vh;
    overflow: auto;
    padding: 18px 18px 24px;
    border-left: 1px solid #d8e0ea;
    background: #ffffff;
  }}
  h1 {{
    margin: 0 0 14px;
    font-size: 18px;
    line-height: 1.25;
  }}
  h2 {{
    margin: 18px 0 8px;
    font-size: 13px;
    letter-spacing: .04em;
    text-transform: uppercase;
    color: #5f6b7c;
  }}
  .row {{
    display: grid;
    grid-template-columns: 92px minmax(0, 1fr);
    gap: 8px;
    margin: 6px 0;
    font-size: 13px;
  }}
  .key {{
    color: #697586;
  }}
  .value {{
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    overflow-wrap: anywhere;
  }}
  .legend {{
    display: grid;
    gap: 7px;
    font-size: 13px;
  }}
  .swatch {{
    display: inline-block;
    width: 10px;
    height: 10px;
    margin-right: 7px;
    border-radius: 2px;
  }}
  .note {{
    color: #5f6b7c;
    font-size: 12px;
    line-height: 1.45;
  }}
  @media (max-width: 900px) {{
    body {{
      grid-template-columns: 1fr;
      grid-template-rows: minmax(0, 1fr) 300px;
    }}
    #scene {{
      height: calc(100vh - 300px);
    }}
    aside {{
      height: 300px;
      border-left: 0;
      border-top: 1px solid #d8e0ea;
    }}
  }}
</style>
</head>
<body>
<canvas id="scene" aria-label="Base-frame 3D pose viewer"></canvas>
<aside>
  <h1>Base Frame 3D Pose View</h1>
  <div class="legend">
    <div><span class="swatch" style="background:#d92d20"></span>Base/grasp +X</div>
    <div><span class="swatch" style="background:#079455"></span>Base/grasp +Y</div>
    <div><span class="swatch" style="background:#1570ef"></span>Base/grasp +Z</div>
    <div><span class="swatch" style="background:#f59e0b"></span>Simplified grasp body</div>
    <div><span class="swatch" style="background:#94a3b8"></span>Surface-normal reference</div>
    <div><span class="swatch" style="background:#0ea5e9"></span>Window</div>
    <div><span class="swatch" style="background:#22d3ee"></span>Approach cone</div>
  </div>
  <h2>Pose</h2>
  <div id="pose"></div>
  <h2>Window</h2>
  <div id="window"></div>
  <h2>Quality</h2>
  <div id="quality"></div>
  <h2>Controls</h2>
  <p class="note">Drag to rotate. Use the mouse wheel or touchpad scroll to zoom. The cuboid is centered at the estimated target pose in the robot base frame.</p>
</aside>
<script>
const data = {data_json};

const canvas = document.getElementById("scene");
const ctx = canvas.getContext("2d");
const poseEl = document.getElementById("pose");
const windowEl = document.getElementById("window");
const qualityEl = document.getElementById("quality");
const state = {{
  yaw: -0.8,
  pitch: 0.65,
  zoom: 1,
  dragging: false,
  lastX: 0,
  lastY: 0
}};

function fmt(value, digits = 6) {{
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "n/a";
  return Number(value).toFixed(digits);
}}

function row(key, value) {{
  return `<div class="row"><div class="key">${{key}}</div><div class="value">${{value}}</div></div>`;
}}

function vec(values, digits = 6) {{
  if (!Array.isArray(values)) return "n/a";
  return "[" + values.map(v => fmt(v, digits)).join(", ") + "]";
}}

function fillPanel() {{
  poseEl.innerHTML = [
    row("source", data.source_json),
    row("visualized", data.pose_role || "n/a"),
    row("candidate", data.candidate_index === null || data.candidate_index === undefined ? "n/a" : data.candidate_index),
    row("score", data.candidate_score === null || data.candidate_score === undefined ? "n/a" : fmt(data.candidate_score, 4)),
    row("xyz m", vec(data.center)),
    row("rpy rad", vec(data.pose_rad)),
    row("rpy deg", vec(data.pose_deg)),
    row("quat xyzw", vec(data.quaternion_xyzw)),
    row("convention", data.convention)
  ].join("");
  if (data.window) {{
    const stats = data.window.candidate_stats || {{}};
    const candidates = data.window.candidates || [];
    const best = candidates.length ? candidates[0] : null;
    windowEl.innerHTML = [
      row("source", data.window.source || "n/a"),
      row("size m", `${{fmt(data.window.width, 4)}} x ${{fmt(data.window.height, 4)}}`),
      row("effective", `${{fmt(data.window.effective_width, 4)}} x ${{fmt(data.window.effective_height, 4)}}`),
      row("margin", fmt(data.window.margin, 4)),
      row("candidates", `${{stats.kept_count ?? candidates.length}} / ${{stats.sampled_count ?? "n/a"}}`),
      row("best score", best ? fmt(best.score, 4) : "n/a")
    ].join("");
  }} else {{
    windowEl.innerHTML = row("window", "not available");
  }}
  qualityEl.innerHTML = [
    row("score", fmt(data.quality_score, 4)),
    row("warnings", (data.warnings || []).length ? data.warnings.join("<br>") : "none"),
    row("model m", `${{fmt(data.model.length, 3)}} x ${{fmt(data.model.width, 3)}} x ${{fmt(data.model.thickness, 3)}}`)
  ].join("");
}}

function add(a, b) {{ return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]; }}
function sub(a, b) {{ return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }}
function mul(a, s) {{ return [a[0] * s, a[1] * s, a[2] * s]; }}
function matVec(m, v) {{
  return [
    m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
    m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
    m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2]
  ];
}}

function poseAxisVector(pose, column) {{
  return [pose.rotation[0][column], pose.rotation[1][column], pose.rotation[2][column]];
}}

function posePoint(local, pose = data) {{
  return add(pose.center, matVec(pose.rotation, local));
}}

function buildCuboid(pose = data) {{
  const hx = data.model.length / 2;
  const hy = data.model.width / 2;
  const hz = data.model.thickness / 2;
  const local = [
    [-hx, -hy, -hz], [ hx, -hy, -hz], [ hx,  hy, -hz], [-hx,  hy, -hz],
    [-hx, -hy,  hz], [ hx, -hy,  hz], [ hx,  hy,  hz], [-hx,  hy,  hz]
  ];
  const v = local.map(p => posePoint(p, pose));
  const faces = [
    [0, 1, 2, 3], [4, 7, 6, 5], [0, 4, 5, 1],
    [1, 5, 6, 2], [2, 6, 7, 3], [3, 7, 4, 0]
  ];
  return {{ v, faces }};
}}

function axisEnd(column, length, pose = data) {{
  return add(pose.center, mul(poseAxisVector(pose, column), length));
}}

function allScenePoints() {{
  const body = buildCuboid();
  const l = data.axis_length;
  const points = [
    [0, 0, 0], [l, 0, 0], [0, l, 0], [0, 0, l],
    data.center, axisEnd(0, l), axisEnd(1, l), axisEnd(2, l),
    ...body.v
  ];
  if (data.reference_pose) {{
    points.push(data.reference_pose.center);
    points.push(axisEnd(0, l, data.reference_pose));
    points.push(axisEnd(1, l, data.reference_pose));
    points.push(axisEnd(2, l, data.reference_pose));
  }}
  if (data.window) {{
    if (Array.isArray(data.window.full_corners)) points.push(...data.window.full_corners);
    if (Array.isArray(data.window.effective_corners)) points.push(...data.window.effective_corners);
    for (const candidate of data.window.candidates || []) {{
      points.push(candidate.window_point);
    }}
  }}
  return points;
}}

function sceneFocus() {{
  const pts = allScenePoints();
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  for (const p of pts) {{
    for (let i = 0; i < 3; i++) {{
      min[i] = Math.min(min[i], p[i]);
      max[i] = Math.max(max[i], p[i]);
    }}
  }}
  return mul(add(min, max), 0.5);
}}

function viewPoint(p, focus) {{
  const q = sub(p, focus);
  const cy = Math.cos(state.yaw), sy = Math.sin(state.yaw);
  const cp = Math.cos(state.pitch), sp = Math.sin(state.pitch);
  const x1 = cy * q[0] + sy * q[1];
  const y1 = -sy * q[0] + cy * q[1];
  const z1 = q[2];
  return [x1, cp * y1 - sp * z1, sp * y1 + cp * z1];
}}

function projectedScale(focus) {{
  const pts = allScenePoints().map(p => viewPoint(p, focus));
  let maxSpan = 0.01;
  for (let i = 0; i < 3; i++) {{
    const vals = pts.map(p => p[i]);
    maxSpan = Math.max(maxSpan, Math.max(...vals) - Math.min(...vals));
  }}
  return Math.min(canvas.width, canvas.height) * 0.62 / maxSpan * state.zoom;
}}

function project(p, focus, scale) {{
  const v = viewPoint(p, focus);
  return {{
    x: canvas.width / 2 + v[0] * scale,
    y: canvas.height / 2 - v[1] * scale,
    z: v[2]
  }};
}}

function drawLine(a, b, color, label, width = 3) {{
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.beginPath();
  ctx.moveTo(a.x, a.y);
  ctx.lineTo(b.x, b.y);
  ctx.stroke();
  const angle = Math.atan2(b.y - a.y, b.x - a.x);
  ctx.beginPath();
  ctx.moveTo(b.x, b.y);
  ctx.lineTo(b.x - 12 * Math.cos(angle - 0.45), b.y - 12 * Math.sin(angle - 0.45));
  ctx.lineTo(b.x - 12 * Math.cos(angle + 0.45), b.y - 12 * Math.sin(angle + 0.45));
  ctx.closePath();
  ctx.fillStyle = color;
  ctx.fill();
  ctx.font = "13px ui-monospace, monospace";
  ctx.fillText(label, b.x + 7, b.y - 7);
}}

function drawGrid(focus, scale) {{
  const gridCenter = data.center;
  const extent = Math.max(0.2, Math.abs(gridCenter[0]) + 0.12, Math.abs(gridCenter[1]) + 0.12);
  const step = 0.05;
  ctx.strokeStyle = "#d9e2ec";
  ctx.lineWidth = 1;
  for (let v = -extent; v <= extent + 1e-9; v += step) {{
    const a = project([-extent, v, 0], focus, scale);
    const b = project([ extent, v, 0], focus, scale);
    const c = project([v, -extent, 0], focus, scale);
    const d = project([v,  extent, 0], focus, scale);
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.moveTo(c.x, c.y);
    ctx.lineTo(d.x, d.y);
    ctx.stroke();
  }}
}}

function drawCuboid(focus, scale) {{
  const body = buildCuboid();
  const projected = body.v.map(p => project(p, focus, scale));
  const faces = body.faces.map(ids => {{
    const depth = ids.reduce((sum, id) => sum + projected[id].z, 0) / ids.length;
    return {{ ids, depth }};
  }}).sort((a, b) => a.depth - b.depth);

  for (const face of faces) {{
    ctx.beginPath();
    for (let i = 0; i < face.ids.length; i++) {{
      const p = projected[face.ids[i]];
      if (i === 0) ctx.moveTo(p.x, p.y); else ctx.lineTo(p.x, p.y);
    }}
    ctx.closePath();
    ctx.fillStyle = "rgba(245, 158, 11, 0.34)";
    ctx.strokeStyle = "rgba(146, 64, 14, 0.92)";
    ctx.lineWidth = 1.5;
    ctx.fill();
    ctx.stroke();
  }}
}}

function drawPolygon(points, focus, scale, fillStyle, strokeStyle, lineWidth = 1.5) {{
  if (!Array.isArray(points) || points.length < 3) return;
  const projected = points.map(p => project(p, focus, scale));
  ctx.beginPath();
  for (let i = 0; i < projected.length; i++) {{
    const p = projected[i];
    if (i === 0) ctx.moveTo(p.x, p.y); else ctx.lineTo(p.x, p.y);
  }}
  ctx.closePath();
  if (fillStyle) {{
    ctx.fillStyle = fillStyle;
    ctx.fill();
  }}
  ctx.strokeStyle = strokeStyle;
  ctx.lineWidth = lineWidth;
  ctx.stroke();
}}

function drawPlainLine(a, b, color, width = 1.5) {{
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.beginPath();
  ctx.moveTo(a.x, a.y);
  ctx.lineTo(b.x, b.y);
  ctx.stroke();
}}

function drawDashedLine(a, b, color, label, width = 2) {{
  ctx.save();
  ctx.setLineDash([7, 5]);
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.beginPath();
  ctx.moveTo(a.x, a.y);
  ctx.lineTo(b.x, b.y);
  ctx.stroke();
  ctx.restore();
  ctx.font = "12px ui-monospace, monospace";
  ctx.fillStyle = color;
  ctx.fillText(label, b.x + 6, b.y + 12);
}}

function faceDepth(points, focus) {{
  return points.reduce((sum, point) => sum + viewPoint(point, focus)[2], 0) / points.length;
}}

function drawWindowAndCone(focus, scale) {{
  if (!data.window) return;
  const full = data.window.full_corners || [];
  const effective = data.window.effective_corners || [];
  const center = data.center;
  const faces = [];

  if (effective.length === 4) {{
    faces.push({{ points: effective, fill: "rgba(14, 165, 233, 0.16)", stroke: "rgba(2, 132, 199, 0.95)", width: 2.4 }});
    for (let i = 0; i < 4; i++) {{
      faces.push({{
        points: [center, effective[i], effective[(i + 1) % 4]],
        fill: "rgba(34, 211, 238, 0.18)",
        stroke: "rgba(8, 145, 178, 0.42)",
        width: 1.1
      }});
    }}
  }}
  if (full.length === 4) {{
    faces.push({{ points: full, fill: "rgba(100, 116, 139, 0.10)", stroke: "rgba(71, 85, 105, 0.9)", width: 1.8 }});
  }}

  faces.sort((a, b) => faceDepth(a.points, focus) - faceDepth(b.points, focus));
  for (const face of faces) {{
    drawPolygon(face.points, focus, scale, face.fill, face.stroke, face.width);
  }}

  const centerPx = project(center, focus, scale);
  const candidates = data.window.candidates || [];
  for (let i = 0; i < candidates.length; i++) {{
    const candidate = candidates[i];
    const p = project(candidate.window_point, focus, scale);
    const isBest = i === 0;
    drawPlainLine(p, centerPx, isBest ? "rgba(14, 116, 144, 0.95)" : "rgba(71, 85, 105, 0.38)", isBest ? 3.2 : 1.3);
    ctx.fillStyle = isBest ? "#0e7490" : "#64748b";
    ctx.beginPath();
    ctx.arc(p.x, p.y, isBest ? 4.2 : 2.8, 0, Math.PI * 2);
    ctx.fill();
  }}
}}

function draw() {{
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, Math.round(rect.width));
  canvas.height = Math.max(1, Math.round(rect.height));

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const focus = sceneFocus();
  const scale = projectedScale(focus);
  drawGrid(focus, scale);
  drawWindowAndCone(focus, scale);
  drawCuboid(focus, scale);

  const origin = project([0, 0, 0], focus, scale);
  const l = data.axis_length;
  drawLine(origin, project([l, 0, 0], focus, scale), "#d92d20", "base +X", 3);
  drawLine(origin, project([0, l, 0], focus, scale), "#079455", "base +Y", 3);
  drawLine(origin, project([0, 0, l], focus, scale), "#1570ef", "base +Z", 3);

  if (data.reference_pose && data.pose_role === "best_grasp_pose_base") {{
    const refCenter = project(data.reference_pose.center, focus, scale);
    drawDashedLine(refCenter, project(axisEnd(0, l, data.reference_pose), focus, scale), "rgba(185, 28, 28, 0.62)", "ref +X", 2);
    drawDashedLine(refCenter, project(axisEnd(1, l, data.reference_pose), focus, scale), "rgba(4, 120, 87, 0.62)", "ref +Y", 2);
    drawDashedLine(refCenter, project(axisEnd(2, l, data.reference_pose), focus, scale), "rgba(29, 78, 216, 0.62)", "ref +Z", 2);
  }}

  const center = project(data.center, focus, scale);
  ctx.fillStyle = "#111827";
  ctx.beginPath();
  ctx.arc(center.x, center.y, 4.5, 0, Math.PI * 2);
  ctx.fill();
  drawLine(center, project(axisEnd(0, l), focus, scale), "#d92d20", "grasp +X", 4);
  drawLine(center, project(axisEnd(1, l), focus, scale), "#079455", "grasp +Y", 4);
  drawLine(center, project(axisEnd(2, l), focus, scale), "#1570ef", "grasp +Z", 4);
}}

canvas.addEventListener("mousedown", event => {{
  state.dragging = true;
  state.lastX = event.clientX;
  state.lastY = event.clientY;
}});
window.addEventListener("mouseup", () => state.dragging = false);
window.addEventListener("mousemove", event => {{
  if (!state.dragging) return;
  const dx = event.clientX - state.lastX;
  const dy = event.clientY - state.lastY;
  state.lastX = event.clientX;
  state.lastY = event.clientY;
  state.yaw += dx * 0.008;
  state.pitch = Math.max(-1.45, Math.min(1.45, state.pitch + dy * 0.008));
  draw();
}});
canvas.addEventListener("wheel", event => {{
  event.preventDefault();
  state.zoom *= Math.exp(-event.deltaY * 0.001);
  state.zoom = Math.max(0.25, Math.min(8, state.zoom));
  draw();
}}, {{ passive: false }});
window.addEventListener("resize", draw);

fillPanel();
draw();
</script>
</body>
</html>
"""


def write_html(output_path: Path, html_text: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    try:
        if not args.json.is_file():
            raise FileNotFoundError(f"JSON file not found: {args.json}")
        output_path = args.output or default_output_path(args.json)
        view_data = build_view_data(load_result(args.json), args)
        write_html(output_path, render_html(view_data))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
