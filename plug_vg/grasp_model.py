"""Grasp-frame CAD model asset loading and semantic point transforms."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import ROOT
from .robot_transform import round_list


DEFAULT_GRASP_MODEL_CONFIG = ROOT / "configs" / "plug_models" / "plugCAD.yaml"


@dataclass(frozen=True)
class GraspModel:
    config_path: Path
    config: dict[str, Any]
    pointcloud_path: Path
    points_grasp_m: np.ndarray


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


def load_grasp_model(config_path: Path = DEFAULT_GRASP_MODEL_CONFIG) -> GraspModel:
    config = load_yaml(config_path)
    if config.get("frame") != "grasp":
        raise ValueError(f"Grasp model config frame must be 'grasp', got {config.get('frame')!r}: {config_path}")
    if config.get("asset_unit") != "meter":
        raise ValueError(f"Grasp model asset_unit must be 'meter', got {config.get('asset_unit')!r}: {config_path}")
    assets = config.get("assets")
    if not isinstance(assets, dict) or "pointcloud_ply" not in assets:
        raise KeyError(f"Grasp model config missing assets.pointcloud_ply: {config_path}")
    pointcloud_path = resolve_path(config_path, str(assets["pointcloud_ply"]))
    return GraspModel(
        config_path=config_path,
        config=config,
        pointcloud_path=pointcloud_path,
        points_grasp_m=read_ascii_ply_points(pointcloud_path),
    )


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return np.asarray(points, dtype=np.float64) @ transform[:3, :3].T + transform[:3, 3]


def semantic_points_camera(config: dict[str, Any], t_camera_grasp: np.ndarray) -> dict[str, list[float]]:
    raw_points = config.get("semantic_points_grasp_m") or {}
    result: dict[str, list[float]] = {}
    for key in ("grasp_center", "tail_center", "head_center"):
        point = np.asarray(raw_points[key], dtype=np.float64)
        homo = np.ones(4, dtype=np.float64)
        homo[:3] = point
        result[f"{key}_camera_m"] = round_list((t_camera_grasp @ homo)[:3])
    return result


def transform_semantic_points_to_base(semantic_camera: dict[str, list[float]], t_base_camera: np.ndarray) -> dict[str, list[float]]:
    result: dict[str, list[float]] = {}
    for key in ("grasp_center", "tail_center", "head_center"):
        camera_key = f"{key}_camera_m"
        point = np.asarray(semantic_camera[camera_key], dtype=np.float64)
        homo = np.ones(4, dtype=np.float64)
        homo[:3] = point
        result[f"{key}_base_m"] = round_list((t_base_camera @ homo)[:3])
    return result


def axis_from_semantic_points(
    semantic_points: dict[str, list[float]],
    tail_key: str,
    head_key: str,
    source: str,
) -> dict[str, Any]:
    tail = np.asarray(semantic_points[tail_key], dtype=np.float64)
    head = np.asarray(semantic_points[head_key], dtype=np.float64)
    direction = head - tail
    norm = float(np.linalg.norm(direction))
    if norm < 1e-9:
        raise ValueError("semantic head/tail axis has near-zero length")
    return {
        "tail_point_m": round_list(tail),
        "head_point_m": round_list(head),
        "direction_unit": round_list(direction / norm),
        "length_m": round(norm, 8),
        "source": source,
    }


__all__ = [
    "DEFAULT_GRASP_MODEL_CONFIG",
    "GraspModel",
    "axis_from_semantic_points",
    "load_grasp_model",
    "load_yaml",
    "read_ascii_ply_points",
    "resolve_path",
    "semantic_points_camera",
    "transform_points",
    "transform_semantic_points_to_base",
]
