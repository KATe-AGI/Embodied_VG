"""Window-constrained base-frame grasp candidate generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import HEAD_TAIL_DISTANCE_M
from .geometry import rotation_to_quaternion_xyzw
from .robot_transform import rotation_to_rpy_xyz, round_list


CORNER_ORDER = "left_top_right_top_right_bottom_left_bottom"
DEFAULT_MARGIN_M = 0.02
DEFAULT_GRID_SIZE = 3
MIN_EDGE_M = 1e-4
PLANAR_TOLERANCE_M = 0.005
MIN_AXIS_CROSS_NORM = 0.1
MIN_WINDOW_NORMAL_ALIGNMENT = 0.25
MIN_AXIS_PROJECTION_NORM = 0.15


class WindowGraspError(ValueError):
    """Raised when window geometry or candidate generation is invalid."""

    def __init__(self, reason: str, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.reason = reason
        self.details = details or {}


@dataclass(frozen=True)
class WindowInputs:
    corners: np.ndarray
    margin_m: float
    source: str


def normalize(vector: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm < 1e-9:
        return None
    return vector / norm


def _as_corners(value: Any, source: str) -> np.ndarray:
    if isinstance(value, dict):
        try:
            raw = [value[f"W{index}"] for index in range(1, 5)]
        except KeyError as exc:
            raise WindowGraspError(
                "window_corners_missing",
                f"Window corners from {source} must contain W1, W2, W3, and W4.",
            ) from exc
    else:
        raw = value

    corners = np.asarray(raw, dtype=np.float64)
    if corners.shape != (4, 3):
        raise WindowGraspError(
            "window_corners_missing",
            f"Window corners from {source} must have shape 4x3, got {corners.shape}.",
        )
    if not np.all(np.isfinite(corners)):
        raise WindowGraspError("window_geometry_invalid", f"Window corners from {source} contain non-finite values.")
    return corners


def _load_window_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise WindowGraspError("window_corners_missing", f"Window config must be a YAML mapping: {path}")
    return data


def resolve_window_inputs(
    config_path: Path | None,
    corners_override: list[float] | tuple[float, ...] | None,
    margin_override: float | None,
) -> WindowInputs:
    """Resolve window corners and margin from YAML plus CLI overrides."""

    config: dict[str, Any] = {}
    source = "cli"
    if config_path is not None:
        config = _load_window_yaml(config_path)
        frame = config.get("frame", "base")
        if frame != "base":
            raise WindowGraspError("window_geometry_invalid", f"Window config frame must be 'base', got {frame!r}.")
        corner_order = config.get("corner_order", CORNER_ORDER)
        if corner_order != CORNER_ORDER:
            raise WindowGraspError(
                "window_geometry_invalid",
                f"Window config corner_order must be {CORNER_ORDER!r}, got {corner_order!r}.",
            )
        source = str(config_path)

    if corners_override is not None:
        if len(corners_override) != 12:
            raise WindowGraspError(
                "window_corners_missing",
                f"--window-corners-base requires 12 numbers, got {len(corners_override)}.",
            )
        corners = _as_corners(np.asarray(corners_override, dtype=np.float64).reshape(4, 3), "cli")
        source = "cli"
    elif "corners_base_m" in config:
        corners = _as_corners(config.get("corners_base_m"), source)
    else:
        raise WindowGraspError(
            "window_corners_missing",
            "Window corners are required. Provide --window-config or --window-corners-base with W1 W2 W3 W4.",
        )

    margin_value = margin_override if margin_override is not None else config.get("margin_m", DEFAULT_MARGIN_M)
    try:
        margin_m = float(margin_value)
    except (TypeError, ValueError) as exc:
        raise WindowGraspError("window_geometry_invalid", f"Window margin_m must be numeric, got {margin_value!r}.") from exc
    if not np.isfinite(margin_m) or margin_m < 0.0:
        raise WindowGraspError("window_geometry_invalid", f"Window margin_m must be finite and non-negative, got {margin_m!r}.")

    return WindowInputs(corners=corners, margin_m=margin_m, source=source)


def build_window_geometry(corners: np.ndarray, margin_m: float, grid_size: int = DEFAULT_GRID_SIZE) -> dict[str, Any]:
    corners = _as_corners(corners, "geometry")
    w1, w2, w3, w4 = corners
    edge_top = w2 - w1
    edge_bottom = w3 - w4
    edge_left = w4 - w1
    edge_right = w3 - w2
    width_top = float(np.linalg.norm(edge_top))
    width_bottom = float(np.linalg.norm(edge_bottom))
    height_left = float(np.linalg.norm(edge_left))
    height_right = float(np.linalg.norm(edge_right))
    if min(width_top, width_bottom, height_left, height_right) < MIN_EDGE_M:
        raise WindowGraspError("window_geometry_invalid", "Window edge length is too small.")

    x_axis = normalize(edge_top)
    y_axis = normalize(edge_left)
    if x_axis is None or y_axis is None:
        raise WindowGraspError("window_geometry_invalid", "Window axes are degenerate.")
    cross = np.cross(x_axis, y_axis)
    cross_norm = float(np.linalg.norm(cross))
    if cross_norm < MIN_AXIS_CROSS_NORM:
        raise WindowGraspError("window_geometry_invalid", "Window horizontal and vertical axes are nearly parallel.")
    normal = cross / cross_norm

    center = np.mean(corners, axis=0)
    plane_distances = np.abs((corners - w1) @ normal)
    max_plane_error = float(np.max(plane_distances))
    if max_plane_error > PLANAR_TOLERANCE_M:
        raise WindowGraspError(
            "window_geometry_invalid",
            f"Window corners are not coplanar within {PLANAR_TOLERANCE_M:.4f} m.",
            {"max_plane_error_m": round(max_plane_error, 8)},
        )

    width = float((width_top + width_bottom) * 0.5)
    height = float((height_left + height_right) * 0.5)
    effective_width = width - 2.0 * float(margin_m)
    effective_height = height - 2.0 * float(margin_m)
    if effective_width <= 0.0 or effective_height <= 0.0:
        raise WindowGraspError(
            "window_geometry_invalid",
            "Window margin leaves no valid sampling area.",
            {"effective_width_m": round(effective_width, 8), "effective_height_m": round(effective_height, 8)},
        )

    return {
        "frame": "base",
        "corner_order": CORNER_ORDER,
        "corners_base_m": {
            f"W{index}": round_list(corner)
            for index, corner in enumerate(corners, start=1)
        },
        "center_base_m": round_list(center),
        "x_window_base": round_list(x_axis),
        "y_window_base": round_list(y_axis),
        "normal_base": round_list(normal),
        "width_m": round(float(width), 8),
        "height_m": round(float(height), 8),
        "margin_m": round(float(margin_m), 8),
        "effective_width_m": round(float(effective_width), 8),
        "effective_height_m": round(float(effective_height), 8),
        "sampling": {
            "mode": f"{int(grid_size)}x{int(grid_size)}_grid",
            "grid_size": int(grid_size),
        },
        "_numeric": {
            "center": center,
            "x_axis": x_axis,
            "y_axis": y_axis,
            "normal": normal,
            "effective_width": effective_width,
            "effective_height": effective_height,
        },
    }


def public_window_geometry(geometry: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in geometry.items() if key != "_numeric"}


def _sample_window_points(geometry: dict[str, Any]) -> list[tuple[int, int, np.ndarray, float]]:
    numeric = geometry["_numeric"]
    grid_size = int(geometry["sampling"]["grid_size"])
    center = numeric["center"]
    x_axis = numeric["x_axis"]
    y_axis = numeric["y_axis"]
    effective_width = float(numeric["effective_width"])
    effective_height = float(numeric["effective_height"])
    if grid_size <= 1:
        offsets = [0.0]
    else:
        offsets = np.linspace(-0.5, 0.5, grid_size)

    points: list[tuple[int, int, np.ndarray, float]] = []
    half_diagonal = float(np.hypot(effective_width * 0.5, effective_height * 0.5))
    for row, y_fraction in enumerate(offsets):
        for col, x_fraction in enumerate(offsets):
            dx = float(x_fraction) * effective_width
            dy = float(y_fraction) * effective_height
            point = center + x_axis * dx + y_axis * dy
            distance_norm = 0.0 if half_diagonal < 1e-9 else min(1.0, float(np.hypot(dx, dy)) / half_diagonal)
            points.append((row, col, point, distance_norm))
    return points


def _candidate_pose(rotation: np.ndarray, translation: np.ndarray, tail_to_head_axis: np.ndarray) -> dict[str, Any]:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    rpy = rotation_to_rpy_xyz(rotation)
    rpy_deg = np.degrees(np.asarray(rpy, dtype=np.float64))
    x_axis = np.asarray(tail_to_head_axis, dtype=np.float64)
    half_axis_length = float(HEAD_TAIL_DISTANCE_M) * 0.5
    tail_point = translation - x_axis * half_axis_length
    head_point = translation + x_axis * half_axis_length
    return {
        "translation_m": round_list(translation),
        "grasp_point_base_m": round_list(translation),
        "rotation_matrix": round_list(rotation),
        "quaternion_xyzw": rotation_to_quaternion_xyzw(rotation),
        "xyzrpy_m_rad": round_list([*translation.tolist(), *rpy]),
        "xyzrpy_m_deg": round_list([*translation.tolist(), *rpy_deg.tolist()]),
        "T_base_grasp": round_list(transform),
        "tail_to_head_axis_base": {
            "tail_point_m": round_list(tail_point),
            "head_point_m": round_list(head_point),
            "direction_unit": round_list(x_axis),
            "length_m": round(float(HEAD_TAIL_DISTANCE_M), 8),
            "passes_through": "grasp_point_base_m",
            "source": "surface_normal_reference_x_axis",
        },
    }


def generate_window_constrained_candidates(
    grasp_pose_base: dict[str, Any],
    window_geometry: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Generate sorted grasp pose candidates constrained by a window."""

    translation = np.asarray(grasp_pose_base.get("translation_m"), dtype=np.float64)
    rotation_reference = np.asarray(grasp_pose_base.get("rotation_matrix"), dtype=np.float64)
    if translation.shape != (3,) or rotation_reference.shape != (3, 3):
        raise WindowGraspError("window_candidate_generation_failed", "grasp_pose_base must contain translation_m and rotation_matrix.")

    a_base = normalize(rotation_reference[:, 0])
    if a_base is None:
        raise WindowGraspError("window_candidate_generation_failed", "Reference tail-to-head axis is invalid.")

    window_normal = np.asarray(window_geometry["_numeric"]["normal"], dtype=np.float64)
    candidates: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for index, (row, col, point, distance_norm) in enumerate(_sample_window_points(window_geometry)):
        approach = normalize(translation - point)
        filter_info: dict[str, Any] = {"grid_row": row, "grid_col": col, "status": "kept"}
        if approach is None:
            filter_info.update({"status": "rejected", "reason": "window_point_equals_grasp_center"})
            rejected.append({"index": index, "filter_info": filter_info})
            continue

        normal_alignment = abs(float(np.dot(approach, window_normal)))
        axis_projected = a_base - float(np.dot(a_base, approach)) * approach
        axis_projection_norm = float(np.linalg.norm(axis_projected))
        filter_info.update(
            {
                "window_normal_alignment_abs": round(normal_alignment, 8),
                "axis_projection_norm": round(axis_projection_norm, 8),
            }
        )
        if normal_alignment < MIN_WINDOW_NORMAL_ALIGNMENT:
            filter_info.update({"status": "rejected", "reason": "approach_parallel_to_window_plane"})
            rejected.append({"index": index, "window_point_base": round_list(point), "filter_info": filter_info})
            continue
        if axis_projection_norm < MIN_AXIS_PROJECTION_NORM:
            filter_info.update({"status": "rejected", "reason": "tail_head_axis_parallel_to_approach"})
            rejected.append({"index": index, "window_point_base": round_list(point), "filter_info": filter_info})
            continue

        x_axis = axis_projected / axis_projection_norm
        y_axis = normalize(np.cross(approach, x_axis))
        if y_axis is None:
            filter_info.update({"status": "rejected", "reason": "y_axis_degenerate"})
            rejected.append({"index": index, "window_point_base": round_list(point), "filter_info": filter_info})
            continue
        z_axis = normalize(np.cross(x_axis, y_axis))
        if z_axis is None:
            filter_info.update({"status": "rejected", "reason": "z_axis_degenerate"})
            rejected.append({"index": index, "window_point_base": round_list(point), "filter_info": filter_info})
            continue

        rotation = np.column_stack([x_axis, y_axis, z_axis]).astype(np.float64)
        center_score = max(0.0, 1.0 - float(distance_norm))
        axis_stability_score = min(1.0, max(0.0, axis_projection_norm))
        score = 0.7 * center_score + 0.3 * axis_stability_score
        candidate = {
            "index": int(index),
            "window_point_base": round_list(point),
            "z_approach_base": round_list(z_axis),
            "x_grasp_base": round_list(x_axis),
            "y_grasp_base": round_list(y_axis),
            "score_visual_geometry": round(float(score), 6),
            "filter_info": {
                **filter_info,
                "center_distance_normalized": round(float(distance_norm), 8),
                "center_score": round(float(center_score), 8),
                "axis_stability_score": round(float(axis_stability_score), 8),
            },
            **_candidate_pose(rotation, translation, a_base),
        }
        candidates.append(candidate)

    candidates.sort(key=lambda item: (-float(item["score_visual_geometry"]), int(item["index"])))
    stats = {
        "sampled_count": int(len(candidates) + len(rejected)),
        "kept_count": int(len(candidates)),
        "rejected_count": int(len(rejected)),
        "rejected": rejected,
        "filters": {
            "min_window_normal_alignment_abs": MIN_WINDOW_NORMAL_ALIGNMENT,
            "min_axis_projection_norm": MIN_AXIS_PROJECTION_NORM,
        },
    }
    return candidates, stats


def add_window_candidates(
    result: dict[str, Any],
    config_path: Path | None,
    corners_override: list[float] | tuple[float, ...] | None,
    margin_override: float | None,
) -> dict[str, Any]:
    """Attach window geometry and constrained candidates to an OK base-frame result."""

    inputs = resolve_window_inputs(config_path, corners_override, margin_override)
    geometry = build_window_geometry(inputs.corners, inputs.margin_m)
    candidates, stats = generate_window_constrained_candidates(result["grasp_pose_base"], geometry)
    result["grasp_pose_base_role"] = "surface_normal_reference"
    result["window_geometry_base"] = public_window_geometry(geometry)
    result["window_geometry_base"]["source"] = inputs.source
    result["window_candidate_stats"] = stats
    result["window_constrained_grasp_candidates"] = candidates
    if not candidates:
        result.pop("best_grasp_pose_base", None)
        result.pop("grasp_point_base_m", None)
        result.pop("tail_to_head_axis_base", None)
        result["status"] = "failed"
        result["reason"] = "window_candidate_generation_failed"
        warnings = result.setdefault("warnings", [])
        warnings.append("window_candidate_generation_failed_no_candidates_after_filtering")
    else:
        result["best_grasp_pose_base"] = dict(candidates[0])
        result["grasp_point_base_m"] = result["best_grasp_pose_base"]["grasp_point_base_m"]
        result["tail_to_head_axis_base"] = result["best_grasp_pose_base"]["tail_to_head_axis_base"]
    return result


__all__ = [
    "DEFAULT_MARGIN_M",
    "WindowGraspError",
    "add_window_candidates",
    "build_window_geometry",
    "generate_window_constrained_candidates",
    "public_window_geometry",
    "resolve_window_inputs",
]
