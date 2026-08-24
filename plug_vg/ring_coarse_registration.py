"""Non-invasive large-ring hypotheses for PCA coarse registration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ModelRing:
    center_grasp_m: np.ndarray
    radius_m: float
    length_m: float


@dataclass(frozen=True)
class SceneRing:
    center_camera_m: np.ndarray
    radius_m: float
    fit_residual_m: float
    axial_center_m: float
    point_count: int


def _orthogonal_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    seed = np.asarray([0.0, 0.0, 1.0]) if abs(float(axis[2])) < 0.9 else np.asarray([0.0, 1.0, 0.0])
    first = np.cross(axis, seed)
    first = first / np.linalg.norm(first)
    return first, np.cross(axis, first)


def detect_model_ring(model_points: np.ndarray, bin_size_m: float) -> ModelRing | None:
    """Find the dominant high-radius axial band in a grasp-X CAD cloud."""

    points = np.asarray(model_points, dtype=np.float64)
    if len(points) < 20 or bin_size_m <= 0.0:
        return None
    x = points[:, 0]
    radii = np.linalg.norm(points[:, 1:3], axis=1)
    origin = float(np.min(x))
    bins = np.floor((x - origin) / float(bin_size_m)).astype(np.int64)
    profile: dict[int, float] = {}
    for index in np.unique(bins):
        values = radii[bins == index]
        if len(values) >= 5:
            profile[int(index)] = float(np.quantile(values, 0.9))
    if len(profile) < 3:
        return None
    peak_index = max(profile, key=profile.get)
    threshold = 0.5 * (float(np.median(list(profile.values()))) + float(profile[peak_index]))
    selected = {index for index, radius in profile.items() if radius >= threshold}
    group = {peak_index}
    for direction in (-1, 1):
        index = peak_index + direction
        while index in selected:
            group.add(index)
            index += direction
    first = min(group)
    last = max(group)
    center_x = origin + (float(first + last + 1) * 0.5) * float(bin_size_m)
    length = float(last - first + 1) * float(bin_size_m)
    radius = float(np.median([profile[index] for index in group]))
    if length <= 0.0 or radius <= 0.0:
        return None
    return ModelRing(
        center_grasp_m=np.asarray([center_x, 0.0, 0.0], dtype=np.float64),
        radius_m=radius,
        length_m=length,
    )


def _fit_circle(points_2d: np.ndarray) -> tuple[np.ndarray, float, float] | None:
    points = np.asarray(points_2d, dtype=np.float64)
    if len(points) < 3:
        return None
    matrix = np.column_stack([2.0 * points[:, 0], 2.0 * points[:, 1], np.ones(len(points))])
    target = np.sum(np.square(points), axis=1)
    solution, _residuals, rank, _singular = np.linalg.lstsq(matrix, target, rcond=None)
    if rank < 3:
        return None
    center = solution[:2]
    radius_squared = float(solution[2] + np.dot(center, center))
    if radius_squared <= 0.0:
        return None
    radius = float(np.sqrt(radius_squared))
    residual = np.abs(np.linalg.norm(points - center, axis=1) - radius)
    return center, radius, float(np.median(residual))


def detect_scene_ring_hypotheses(
    scene_points: np.ndarray,
    axis_camera: np.ndarray,
    model_ring: ModelRing,
    bin_size_m: float,
    max_hypotheses: int = 2,
) -> list[SceneRing]:
    """Fit large-ring centers in windows along the scene PCA axis."""

    points = np.asarray(scene_points, dtype=np.float64)
    if len(points) < 20 or bin_size_m <= 0.0:
        return []
    axis = np.asarray(axis_camera, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    first, second = _orthogonal_basis(axis)
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    axial = centered @ axis
    transverse = np.column_stack([centered @ first, centered @ second])
    half_length = 0.5 * float(model_ring.length_m)
    start = float(np.min(axial)) + half_length
    stop = float(np.max(axial)) - half_length
    if start > stop:
        return []

    fits: list[tuple[float, SceneRing]] = []
    for axial_center in np.arange(start, stop + 0.5 * bin_size_m, 0.5 * bin_size_m):
        keep = np.abs(axial - axial_center) <= half_length
        if int(np.count_nonzero(keep)) < 30:
            continue
        fit = _fit_circle(transverse[keep])
        if fit is None:
            continue
        center_2d, radius, residual = fit
        center_3d = centroid + axis * axial_center + first * center_2d[0] + second * center_2d[1]
        hypothesis = SceneRing(
            center_camera_m=center_3d,
            radius_m=radius,
            fit_residual_m=residual,
            axial_center_m=float(axial_center),
            point_count=int(np.count_nonzero(keep)),
        )
        score = abs(radius - model_ring.radius_m) + residual
        fits.append((float(score), hypothesis))

    selected: list[SceneRing] = []
    for _score, hypothesis in sorted(fits, key=lambda item: item[0]):
        if any(
            abs(hypothesis.axial_center_m - current.axial_center_m) < model_ring.length_m
            for current in selected
        ):
            continue
        selected.append(hypothesis)
        if len(selected) >= int(max_hypotheses):
            break
    return selected


def ring_aligned_candidates(
    model_points: np.ndarray,
    scene_points: np.ndarray,
    scene_axis_camera: np.ndarray,
    base_candidates: list[dict[str, Any]],
    bin_size_m: float,
) -> list[dict[str, Any]]:
    """Clone scene-to-model PCA rotations and align detected ring centers."""

    model_ring = detect_model_ring(model_points, bin_size_m)
    if model_ring is None:
        return []
    hypotheses = detect_scene_ring_hypotheses(
        scene_points,
        scene_axis_camera,
        model_ring,
        bin_size_m,
    )
    candidates: list[dict[str, Any]] = []
    for hypothesis_index, hypothesis in enumerate(hypotheses, start=1):
        for base in base_candidates:
            t_grasp_camera = np.asarray(base["t_grasp_camera"], dtype=np.float64).copy()
            t_grasp_camera[:3, 3] = (
                model_ring.center_grasp_m
                - t_grasp_camera[:3, :3] @ hypothesis.center_camera_m
            )
            candidates.append(
                {
                    **base,
                    "name": f"ring_{hypothesis_index}_{base['name']}",
                    "t_grasp_camera": t_grasp_camera,
                    "initializer": "large_ring_center",
                    "ring_hypothesis": int(hypothesis_index),
                    "model_ring_center_grasp_m": model_ring.center_grasp_m.tolist(),
                    "scene_ring_center_camera_m": hypothesis.center_camera_m.tolist(),
                    "model_ring_radius_m": float(model_ring.radius_m),
                    "scene_ring_radius_m": float(hypothesis.radius_m),
                    "ring_fit_residual_m": float(hypothesis.fit_residual_m),
                    "ring_point_count": int(hypothesis.point_count),
                }
            )
    return candidates


__all__ = [
    "ModelRing",
    "SceneRing",
    "detect_model_ring",
    "detect_scene_ring_hypotheses",
    "ring_aligned_candidates",
]
