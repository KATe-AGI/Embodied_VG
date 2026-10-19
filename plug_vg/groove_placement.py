"""Allowed groove axis segment and metric placement decision in robot base frame."""

from typing import Any

import numpy as np


def groove_axis_segment(center_base: Any, normal_base: Any, config: dict) -> dict:
    """Construct the groove centerline independently of approach preset points."""
    center = np.asarray(center_base, dtype=float)
    normal = np.asarray(normal_base, dtype=float)
    if (
        center.shape != (3,)
        or normal.shape != (3,)
        or not np.isfinite([center, normal]).all()
    ):
        raise ValueError("Reference center and normal must be finite XYZ vectors")
    if not np.isclose(np.linalg.norm(normal), 1.0):
        raise ValueError("Reference normal must be a unit vector")
    normal = normal / np.linalg.norm(normal)
    up = np.array([0.0, 0.0, 1.0])
    direction = up - np.dot(up, normal) * normal
    if np.linalg.norm(direction) < 1e-6:
        raise ValueError("Groove up direction is undefined: normal parallel to base Z")
    direction /= np.linalg.norm(direction)
    reference = config["reference"]
    origin = center + float(reference["groove_plane_normal_offset_m"]) * normal
    return {
        "origin_base_m": origin.tolist(),
        "direction_base": direction.tolist(),
        "tail_limit_base_m": (
            origin + reference["tail_base_z_offset_m"] * direction
        ).tolist(),
        "head_limit_base_m": (
            origin + reference["head_base_z_offset_m"] * direction
        ).tolist(),
    }


def evaluate_groove_placement(
    tail_base: Any, head_base: Any, segment: dict, tolerance_m: float
) -> dict:
    """Test whether both CAD end centers lie inside the allowed segment capsule."""
    points = np.asarray([tail_base, head_base], dtype=float)
    limits = np.asarray(
        [segment["tail_limit_base_m"], segment["head_limit_base_m"]], dtype=float
    )
    if (
        points.shape != (2, 3)
        or limits.shape != (2, 3)
        or not np.isfinite([points, limits]).all()
    ):
        raise ValueError(
            "Predicted endpoints and groove limits must be finite XYZ vectors"
        )
    if not np.isfinite(tolerance_m) or tolerance_m <= 0:
        raise ValueError("Placement tolerance must be finite and positive")
    delta = limits[1] - limits[0]
    length_squared = float(np.dot(delta, delta))
    if length_squared < 1e-12 or np.linalg.norm(points[1] - points[0]) < 1e-6:
        raise ValueError("Predicted axis and allowed segment must have nonzero length")
    if points[1, 2] <= points[0, 2]:
        raise ValueError("Registration contradicts the head-up prior")
    fraction = np.clip((points - limits[0]) @ delta / length_squared, 0.0, 1.0)
    closest = limits[0] + fraction[:, None] * delta
    distances = np.linalg.norm(points - closest, axis=1)
    maximum = float(np.max(distances))
    return {
        "is_seated": bool(maximum <= tolerance_m),
        "tail_distance_mm": float(distances[0] * 1000),
        "head_distance_mm": float(distances[1] * 1000),
        "max_distance_mm": maximum * 1000,
        "tolerance_mm": float(tolerance_m * 1000),
        "closest_tail_base_m": closest[0].tolist(),
        "closest_head_base_m": closest[1].tolist(),
    }
