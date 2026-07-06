#!/usr/bin/env python3
"""Render headless review images for grasp-frame model assets."""

from __future__ import annotations

import argparse
import base64
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "plug_models" / "2175B.yaml"
DEFAULT_OUTPUT_DIR = ROOT / "plug_model" / "review"


def resolve_path(config_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    root_candidate = ROOT / path
    if root_candidate.exists() or not (config_path.parent / path).exists():
        return root_candidate
    return config_path.parent / path


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return data


def load_obj_vertices(path: Path) -> np.ndarray:
    vertices: list[list[float]] = []
    with path.open("r", encoding="ascii", errors="ignore") as f:
        for line in f:
            if not line.startswith("v "):
                continue
            parts = line.split()
            if len(parts) >= 4:
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if not vertices:
        raise ValueError(f"OBJ contains no vertices: {path}")
    return np.asarray(vertices, dtype=np.float64)


def equalize_3d_axes(ax, points: np.ndarray) -> None:
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    centers = (mins + maxs) * 0.5
    radius = float(np.max(maxs - mins) * 0.58)
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


def draw_axes_3d(ax, origin: np.ndarray, scale: float) -> None:
    axes = [
        ("+X tail->head", np.asarray([1.0, 0.0, 0.0]), "#d92d20"),
        ("+Y closing", np.asarray([0.0, 1.0, 0.0]), "#079455"),
        ("+Z approach", np.asarray([0.0, 0.0, 1.0]), "#1570ef"),
    ]
    for label, direction, color in axes:
        end = origin + direction * scale
        ax.quiver(origin[0], origin[1], origin[2], direction[0], direction[1], direction[2], length=scale, color=color, linewidth=2.2)
        ax.text(end[0], end[1], end[2], label, color=color, fontsize=9)


def draw_projection(ax, points: np.ndarray, dims: tuple[int, int], semantic: dict[str, np.ndarray], title: str) -> None:
    a, b = dims
    ax.scatter(points[:, a], points[:, b], s=0.5, c="#5b6472", alpha=0.25, linewidths=0)
    styles = {
        "grasp_center": ("o", "#ffffff", "#111827", 70),
        "tail_center": ("^", "#255e9e", "#102a43", 70),
        "head_center": ("s", "#d92d20", "#7a271a", 70),
    }
    for name, point in semantic.items():
        marker, face, edge, size = styles[name]
        ax.scatter(point[a], point[b], marker=marker, s=size, c=face, edgecolors=edge, linewidths=1.2, zorder=5)
        ax.text(point[a], point[b], f" {name}", fontsize=8, color=edge, zorder=6)
    ax.axhline(0.0, color="#d0d5dd", linewidth=0.8)
    ax.axvline(0.0, color="#d0d5dd", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel(["X tail->head (m)", "Y closing (m)", "Z approach (m)"][a])
    ax.set_ylabel(["X tail->head (m)", "Y closing (m)", "Z approach (m)"][b])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)


def render_review(config_path: Path, output_dir: Path, max_points: int, seed: int) -> tuple[Path, Path]:
    config = load_config(config_path)
    obj_path = resolve_path(config_path, config["assets"]["mesh_obj"])
    points = load_obj_vertices(obj_path)
    if len(points) > max_points:
        rng = np.random.default_rng(seed)
        points = points[rng.choice(len(points), size=max_points, replace=False)]

    semantic_raw = config["semantic_points_grasp_m"]
    semantic = {
        "grasp_center": np.asarray(semantic_raw["grasp_center"], dtype=np.float64),
        "tail_center": np.asarray(semantic_raw["tail_center"], dtype=np.float64),
        "head_center": np.asarray(semantic_raw["head_center"], dtype=np.float64),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / f"{config.get('model_id', 'model')}_grasp_frame_review.png"
    html_path = output_dir / f"{config.get('model_id', 'model')}_grasp_frame_review.html"

    fig = plt.figure(figsize=(16, 11))
    ax3d = fig.add_subplot(2, 2, 1, projection="3d")
    ax3d.scatter(points[:, 0], points[:, 1], points[:, 2], s=0.4, c="#475467", alpha=0.22, depthshade=False)
    ax3d.scatter(*semantic["grasp_center"], marker="o", s=70, c="#ffffff", edgecolors="#111827", linewidths=1.4, label="grasp_center")
    ax3d.scatter(*semantic["tail_center"], marker="^", s=80, c="#255e9e", edgecolors="#102a43", linewidths=1.2, label="tail_center")
    ax3d.scatter(*semantic["head_center"], marker="s", s=80, c="#d92d20", edgecolors="#7a271a", linewidths=1.2, label="head_center")
    axis_scale = max(0.03, float(config["dimensions_m"]["head_tail_axis_length"]) * 0.28)
    draw_axes_3d(ax3d, semantic["grasp_center"], axis_scale)
    ax3d.set_title("Grasp Frame 3D Review")
    ax3d.set_xlabel("X tail->head (m)")
    ax3d.set_ylabel("Y closing (m)")
    ax3d.set_zlabel("Z approach (m)")
    ax3d.view_init(elev=22, azim=-55)
    equalize_3d_axes(ax3d, points)
    ax3d.legend(loc="upper left")

    draw_projection(fig.add_subplot(2, 2, 2), points, (0, 1), semantic, "XY Projection")
    draw_projection(fig.add_subplot(2, 2, 3), points, (0, 2), semantic, "XZ Projection")
    draw_projection(fig.add_subplot(2, 2, 4), points, (1, 2), semantic, "YZ Projection")

    fig.suptitle(
        f"{config.get('model_id', 'model')} grasp-frame assets | unit={config.get('asset_unit')} | review_required={config.get('requires_manual_review')}",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(image_path, dpi=220)
    plt.close(fig)

    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    report = config.get("assets", {}).get("review_report")
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{config.get('model_id', 'model')} grasp-frame review</title>
  <style>
    body {{ margin: 24px; font-family: Arial, sans-serif; color: #111827; }}
    img {{ max-width: 100%; border: 1px solid #d0d5dd; }}
    code {{ background: #f2f4f7; padding: 2px 4px; border-radius: 3px; }}
    .meta {{ line-height: 1.5; }}
  </style>
</head>
<body>
  <h1>{config.get('model_id', 'model')} Grasp Frame Review</h1>
  <div class="meta">
    <div>Frame: <code>{config.get('frame')}</code></div>
    <div>Transform name: <code>{config.get('downstream_transform_name')}</code></div>
    <div>Head center: <code>{semantic_raw['head_center']}</code></div>
    <div>Tail center: <code>{semantic_raw['tail_center']}</code></div>
    <div>Axis length: <code>{config['dimensions_m']['head_tail_axis_length']} m</code></div>
    <div>Report: <code>{report}</code></div>
  </div>
  <p><img src="data:image/png;base64,{encoded}" alt="grasp frame review"></p>
</body>
</html>
"""
    html_path.write_text(html, encoding="utf-8")
    return image_path, html_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Grasp model YAML config.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for review PNG/HTML.")
    parser.add_argument("--max-points", type=int, default=20000, help="Maximum OBJ vertices to render.")
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed when decimating vertices.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path, html_path = render_review(args.config, args.output_dir, args.max_points, args.seed)
    print(f"png: {image_path}")
    print(f"html: {html_path}")


if __name__ == "__main__":
    main()
