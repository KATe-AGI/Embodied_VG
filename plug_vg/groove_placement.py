"""Groove centerline and finite-axis containment in a circular frustum."""

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
    tail_base: Any, head_base: Any, segment: dict, config: dict
) -> dict:
    """Measure the exact fraction of a finite plug axis inside a capped frustum.

    The smaller top disk is centered at the head limit; the larger bottom disk
    is centered at the tail limit. Boundary points are inside. Intersecting a
    parameterized line segment with the two caps and quadratic side surface
    avoids dependence on an arbitrary number of sampled axis points.
    """
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
    settings = validate_groove_check(config)
    top_radius, bottom_radius, minimum = (
        settings[key]
        for key in ("top_radius_m", "bottom_radius_m", "min_inside_fraction")
    )
    delta = limits[1] - limits[0]
    length_squared = float(np.dot(delta, delta))
    if length_squared < 1e-12 or np.linalg.norm(points[1] - points[0]) < 1e-6:
        raise ValueError("Predicted axis and allowed segment must have nonzero length")
    if points[1, 2] <= points[0, 2]:
        raise ValueError("Registration contradicts the head-up prior")
    length = float(np.sqrt(length_squared))
    axis = delta / length
    relative = points[0] - limits[0]
    travel = points[1] - points[0]
    height0, height_delta = float(relative @ axis), float(travel @ axis)
    radial0 = relative - height0 * axis
    radial_delta = travel - height_delta * axis
    radius_slope = (top_radius - bottom_radius) / length
    radius0 = bottom_radius + radius_slope * height0
    radius_delta = radius_slope * height_delta
    # f(t) <= 0 describes the side surface, subject to 0 <= height(t) <= L.
    coefficients = np.array([
        radial_delta @ radial_delta - radius_delta**2,
        2 * (radial0 @ radial_delta - radius0 * radius_delta),
        radial0 @ radial0 - radius0**2,
    ], dtype=float)
    scale = max(float(np.max(np.abs(coefficients))), np.finfo(float).tiny)
    quadratic, linear, constant = coefficients / scale
    breaks = [0.0, 1.0]
    if abs(height_delta) > 1e-14:
        breaks.extend([-height0 / height_delta, (length - height0) / height_delta])
    if abs(quadratic) < 1e-12:
        if abs(linear) >= 1e-12:
            breaks.append(-constant / linear)
    else:
        discriminant = linear**2 - 4 * quadratic * constant
        if discriminant >= 0:
            root = float(np.sqrt(discriminant))
            # Stable quadratic roots even when one root is close to zero.
            q = -0.5 * (linear + np.copysign(root, linear))
            if q != 0:
                breaks.extend([q / quadratic, constant / q])
            else:
                breaks.append(-linear / (2 * quadratic))
    breaks = sorted(set(float(t) for t in breaks if 0 <= t <= 1))
    intervals = []
    for start, end in zip(breaks, breaks[1:]):
        middle = (start + end) / 2
        height = height0 + middle * height_delta
        radius = bottom_radius + radius_slope * height
        radial = radial0 + middle * radial_delta
        # A tiny metric tolerance includes segments exactly on a cap or side.
        if -1e-12 <= height <= length + 1e-12 and float(radial @ radial) <= radius**2 + 1e-15:
            if intervals and np.isclose(intervals[-1][1], start, atol=1e-12, rtol=0):
                intervals[-1][1] = end
            else:
                intervals.append([start, end])
    inside_fraction = float(np.clip(sum(end - start for start, end in intervals), 0, 1))
    plug_length = float(np.linalg.norm(travel))
    fraction = np.clip((points - limits[0]) @ delta / length_squared, 0.0, 1.0)
    closest = limits[0] + fraction[:, None] * delta
    distances = np.linalg.norm(points - closest, axis=1)
    maximum = float(np.max(distances))
    return {
        "criterion": "axis_length_inside_circular_frustum",
        "is_seated": bool(inside_fraction >= minimum - 1e-12),
        "inside_fraction": inside_fraction,
        "outside_fraction": 1 - inside_fraction,
        "min_inside_fraction": minimum,
        "axis_length_mm": plug_length * 1000,
        "inside_length_mm": inside_fraction * plug_length * 1000,
        "top_radius_mm": top_radius * 1000,
        "bottom_radius_mm": bottom_radius * 1000,
        "inside_intervals_t": intervals,
        "tail_distance_mm": float(distances[0] * 1000),
        "head_distance_mm": float(distances[1] * 1000),
        "max_distance_mm": maximum * 1000,
        "closest_tail_base_m": closest[0].tolist(),
        "closest_head_base_m": closest[1].tolist(),
    }


def validate_groove_check(config: dict) -> dict[str, float]:
    """Validate the frustum settings used by inference and review rendering."""
    values = {key: float(config[key]) for key in (
        "top_radius_m", "bottom_radius_m", "min_inside_fraction"
    )}
    if not np.isfinite(list(values.values())).all():
        raise ValueError("Groove settings must be finite")
    if not 0 < values["top_radius_m"] <= values["bottom_radius_m"]:
        raise ValueError("Groove radii must satisfy 0 < top_radius_m <= bottom_radius_m")
    if not 0 < values["min_inside_fraction"] <= 1:
        raise ValueError("min_inside_fraction must be in (0,1]")
    return values
