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
    normals_grasp: np.ndarray | None


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


def read_ascii_ply_points_and_normals(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    with path.open("r", encoding="ascii", errors="ignore") as f:
        vertex_count: int | None = None
        vertex_properties: list[str] = []
        in_vertex_element = False
        for line in f:
            if line.startswith("element vertex"):
                vertex_count = int(line.split()[2])
                in_vertex_element = True
            elif line.startswith("element "):
                in_vertex_element = False
            elif in_vertex_element and line.startswith("property "):
                vertex_properties.append(line.split()[-1])
            if line.strip() == "end_header":
                break
        if vertex_count is None:
            raise ValueError(f"PLY vertex element missing: {path}")
        required = ("x", "y", "z")
        if any(name not in vertex_properties for name in required):
            raise ValueError(f"PLY vertex XYZ properties missing: {path}")
        xyz_indices = [vertex_properties.index(name) for name in required]
        normal_names = ("nx", "ny", "nz")
        has_normals = all(name in vertex_properties for name in normal_names)
        if any(name in vertex_properties for name in normal_names) and not has_normals:
            raise ValueError(f"PLY must contain either all or none of nx/ny/nz: {path}")
        normal_indices = [vertex_properties.index(name) for name in normal_names] if has_normals else []
        points = []
        normals = []
        for index, line in enumerate(f):
            if index >= vertex_count:
                break
            parts = line.split()
            if len(parts) >= len(vertex_properties):
                points.append([float(parts[item]) for item in xyz_indices])
                if has_normals:
                    normals.append([float(parts[item]) for item in normal_indices])
    if not points:
        raise ValueError(f"PLY contains no points: {path}")
    point_array = np.asarray(points, dtype=np.float64)
    if not has_normals:
        return point_array, None
    normal_array = np.asarray(normals, dtype=np.float64)
    lengths = np.linalg.norm(normal_array, axis=1)
    if np.any(~np.isfinite(normal_array)) or np.any(lengths <= 1e-12):
        raise ValueError(f"PLY contains invalid normals: {path}")
    return point_array, normal_array / lengths[:, None]


def read_ascii_ply_points(path: Path) -> np.ndarray:
    points, _normals = read_ascii_ply_points_and_normals(path)
    return points


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
    points, normals = read_ascii_ply_points_and_normals(pointcloud_path)
    return GraspModel(
        config_path=config_path,
        config=config,
        pointcloud_path=pointcloud_path,
        points_grasp_m=points,
        normals_grasp=normals,
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
    "read_ascii_ply_points_and_normals",
    "resolve_path",
    "semantic_points_camera",
    "transform_points",
    "transform_semantic_points_to_base",
]
