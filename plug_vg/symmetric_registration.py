"""Long-axis-symmetry-aware registration for partial RGB-D observations."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import os
from typing import Any

import cv2
import numpy as np

from .robot_transform import round_list


@dataclass(frozen=True)
class ProfileGrid:
    distances_m: np.ndarray
    axial_min_m: float
    axial_max_m: float
    radial_min_m: float
    radial_max_m: float


@dataclass(frozen=True)
class RenderContext:
    target_mask: np.ndarray
    scene_depth_m: np.ndarray
    image_height: int
    image_width: int
    target_x0: int
    target_x1: int
    target_y0: int
    target_y1: int


def canonical_rotation_from_axis(axis_camera: np.ndarray) -> np.ndarray:
    """Return a deterministic right-handed rotation whose X column is ``axis_camera``."""

    x_axis = np.asarray(axis_camera, dtype=np.float64)
    x_axis /= np.linalg.norm(x_axis)
    seed = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(seed, x_axis))) > 0.9:
        seed = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    y_axis = seed - float(np.dot(seed, x_axis)) * x_axis
    y_axis /= np.linalg.norm(y_axis)
    z_axis = np.cross(x_axis, y_axis)
    z_axis /= np.linalg.norm(z_axis)
    return np.column_stack([x_axis, y_axis, z_axis])


def build_profile_grid(model_points: np.ndarray, resolution_m: float = 0.0005) -> ProfileGrid:
    """Build an unsigned axial/radial surface-distance grid from a grasp-X CAD cloud."""

    from scipy.ndimage import binary_dilation, distance_transform_edt

    model_points = np.asarray(model_points, dtype=np.float64)
    axial_radial = np.column_stack(
        [model_points[:, 0], np.linalg.norm(model_points[:, 1:3], axis=1)]
    )
    axial_min = float(np.min(axial_radial[:, 0]) - 0.01)
    axial_max = float(np.max(axial_radial[:, 0]) + 0.01)
    radial_min = 0.0
    radial_max = float(np.max(axial_radial[:, 1]) + 0.01)
    width = int(np.ceil((axial_max - axial_min) / resolution_m)) + 1
    height = int(np.ceil((radial_max - radial_min) / resolution_m)) + 1
    occupied = np.zeros((height, width), dtype=bool)
    axial_index = np.clip(
        np.round((axial_radial[:, 0] - axial_min) / resolution_m).astype(np.int64),
        0,
        width - 1,
    )
    radial_index = np.clip(
        np.round((axial_radial[:, 1] - radial_min) / resolution_m).astype(np.int64),
        0,
        height - 1,
    )
    occupied[radial_index, axial_index] = True
    # Join sub-grid gaps caused by finite CAD sampling without smoothing the
    # axial profile or introducing a task-specific landmark.
    occupied = binary_dilation(occupied, iterations=1)
    distances = distance_transform_edt(~occupied) * float(resolution_m)
    return ProfileGrid(
        distances_m=np.asarray(distances, dtype=np.float32),
        axial_min_m=axial_min,
        axial_max_m=axial_max,
        radial_min_m=radial_min,
        radial_max_m=radial_max,
    )


def _fibonacci_directions(count: int) -> np.ndarray:
    index = np.arange(int(count), dtype=np.float64)
    golden_ratio = (1.0 + np.sqrt(5.0)) * 0.5
    z = 1.0 - 2.0 * (index + 0.5) / float(count)
    radius = np.sqrt(np.maximum(0.0, 1.0 - np.square(z)))
    angle = 2.0 * np.pi * index / golden_ratio
    return np.column_stack([radius * np.cos(angle), radius * np.sin(angle), z])


def _axis_seeds(scene_points: np.ndarray, direction_count: int) -> np.ndarray:
    centered = scene_points - np.mean(scene_points, axis=0)
    _values, vectors = np.linalg.eigh(np.cov(centered.T))
    extras = np.vstack(
        [
            vectors.T,
            -vectors.T,
            np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]),
        ]
    )
    return np.vstack([_fibonacci_directions(direction_count), extras])


def _torch_device(device: Any):
    import torch

    value = "" if device is None else str(device).strip().lower()
    if value == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    if value.startswith("cuda:"):
        return torch.device(value)
    if value and value.split(",", 1)[0].isdigit():
        return torch.device(f"cuda:{value.split(',', 1)[0]}")
    return torch.device("cuda:0")


def optimize_profile_candidates(
    model_points: np.ndarray,
    scene_points: np.ndarray,
    device: Any = None,
    direction_count: int = 48,
    axial_anchor_count: int = 5,
    max_scene_points: int = 600,
    iterations: int = 60,
) -> list[dict[str, Any]]:
    """Jointly optimize directed long axes and grasp origins against the CAD profile."""

    import torch
    import torch.nn.functional as functional

    scene_points = np.asarray(scene_points, dtype=np.float64)
    if len(scene_points) > int(max_scene_points):
        indices = np.linspace(0, len(scene_points) - 1, int(max_scene_points), dtype=np.int64)
        sampled_scene = scene_points[indices]
    else:
        sampled_scene = scene_points
    profile = build_profile_grid(model_points)
    torch_device = _torch_device(device)
    profile_tensor = torch.as_tensor(
        profile.distances_m,
        dtype=torch.float32,
        device=torch_device,
    )[None, None]
    points_tensor = torch.as_tensor(sampled_scene, dtype=torch.float32, device=torch_device)

    directions = _axis_seeds(sampled_scene, direction_count)
    model_axial_min = float(np.min(np.asarray(model_points)[:, 0]))
    model_axial_max = float(np.max(np.asarray(model_points)[:, 0]))
    anchors = np.linspace(model_axial_min, model_axial_max, int(axial_anchor_count))
    initial_axes = np.repeat(directions, len(anchors), axis=0)
    repeated_anchors = np.tile(anchors, len(directions))
    scene_center = np.median(sampled_scene, axis=0)
    initial_origins = scene_center - initial_axes * repeated_anchors[:, None]

    axes = torch.tensor(initial_axes, dtype=torch.float32, device=torch_device, requires_grad=True)
    origins = torch.tensor(initial_origins, dtype=torch.float32, device=torch_device, requires_grad=True)
    optimizer = torch.optim.Adam([axes, origins], lr=0.006)

    def calculate():
        unit_axes = axes / torch.linalg.norm(axes, dim=1, keepdim=True).clamp_min(1e-9)
        delta = points_tensor[None, :, :] - origins[:, None, :]
        axial = torch.sum(delta * unit_axes[:, None, :], dim=2)
        radial_squared = torch.sum(delta * delta, dim=2) - axial * axial
        radial = torch.sqrt(torch.clamp(radial_squared, min=1e-10))
        grid_x = 2.0 * (axial - profile.axial_min_m) / (
            profile.axial_max_m - profile.axial_min_m
        ) - 1.0
        grid_y = 2.0 * (radial - profile.radial_min_m) / (
            profile.radial_max_m - profile.radial_min_m
        ) - 1.0
        grid = torch.stack([grid_x, grid_y], dim=-1)[:, None, :, :]
        distances = functional.grid_sample(
            profile_tensor.expand(len(axes), -1, -1, -1),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )[:, 0, 0, :]
        outside = (
            torch.relu(profile.axial_min_m - axial)
            + torch.relu(axial - profile.axial_max_m)
            + torch.relu(profile.radial_min_m - radial)
            + torch.relu(radial - profile.radial_max_m)
        )
        distances = distances + outside
        robust_scale_m = 0.004
        losses = torch.log1p(torch.square(distances / robust_scale_m)).mean(dim=1)
        return unit_axes, losses

    for _iteration in range(int(iterations)):
        optimizer.zero_grad()
        _unit_axes, losses = calculate()
        losses.mean().backward()
        optimizer.step()

    with torch.no_grad():
        unit_axes, losses = calculate()
        axes_numpy = unit_axes.cpu().numpy().astype(np.float64)
        origins_numpy = origins.cpu().numpy().astype(np.float64)
        losses_numpy = losses.cpu().numpy().astype(np.float64)
    order = np.argsort(losses_numpy)
    return [
        {
            "axis_camera": axes_numpy[index],
            "origin_camera_m": origins_numpy[index],
            "profile_loss": float(losses_numpy[index]),
            "profile_rank": int(rank),
            "axis_seed_index": int(index // len(anchors)),
            "axial_anchor_index": int(index % len(anchors)),
        }
        for rank, index in enumerate(order)
    ]


def _render_candidate_cost(
    model_points: np.ndarray,
    candidate: dict[str, Any],
    context: RenderContext,
    camera: dict[str, Any],
    occlusion_tolerance_m: float,
) -> dict[str, float]:
    """Score one pose using visible rendered CAD depth and target-mask agreement."""

    rotation = canonical_rotation_from_axis(candidate["axis_camera"])
    model_camera = np.asarray(model_points, dtype=np.float64) @ rotation.T
    model_camera += np.asarray(candidate["origin_camera_m"], dtype=np.float64)
    z = model_camera[:, 2]
    valid = z > 1e-6
    model_camera = model_camera[valid]
    z = z[valid]
    u = np.rint(float(camera["fx"]) * model_camera[:, 0] / z + float(camera["cx"])).astype(np.int64)
    v = np.rint(float(camera["fy"]) * model_camera[:, 1] / z + float(camera["cy"])).astype(np.int64)
    image_height = context.image_height
    image_width = context.image_width
    valid = (u >= 0) & (u < image_width) & (v >= 0) & (v < image_height)
    if not np.any(valid):
        return {"geometry_cost": float("inf"), "visible_iou": 0.0, "depth_cost": 1.0}
    u = u[valid]
    v = v[valid]
    z = z[valid]

    margin = 12
    x0 = max(0, min(context.target_x0, int(np.min(u))) - margin)
    x1 = min(image_width, max(context.target_x1 - 1, int(np.max(u))) + margin + 1)
    y0 = max(0, min(context.target_y0, int(np.min(v))) - margin)
    y1 = min(image_height, max(context.target_y1 - 1, int(np.max(v))) + margin + 1)
    roi_width = x1 - x0
    roi_height = y1 - y0
    rendered_depth = np.full((roi_height, roi_width), np.inf, dtype=np.float64)
    np.minimum.at(rendered_depth, (v - y0, u - x0), z)

    target_roi = context.target_mask[y0:y1, x0:x1]
    scene_roi = context.scene_depth_m[y0:y1, x0:x1]
    rendered = np.isfinite(rendered_depth)
    observed_valid = scene_roi > 0.0
    occluded = rendered & observed_valid & (scene_roi < rendered_depth - occlusion_tolerance_m)
    visible_samples = rendered & ~occluded
    # The CAD asset has finite surface sampling.  Expand only enough to join
    # adjacent projected samples; this affects rendering density, not geometry.
    visible_mask = cv2.dilate(
        visible_samples.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        iterations=1,
    ).astype(bool)
    intersection = int(np.count_nonzero(visible_mask & target_roi))
    union = int(np.count_nonzero(visible_mask | target_roi))
    visible_iou = float(intersection / union) if union else 0.0

    depth_pixels = visible_samples & target_roi & observed_valid
    if np.any(depth_pixels):
        depth_residual = np.abs(scene_roi[depth_pixels] - rendered_depth[depth_pixels])
        depth_cost = float(np.mean(np.minimum(depth_residual / occlusion_tolerance_m, 1.0)))
    else:
        depth_cost = 1.0
    return {
        # Sparse CAD surfels make point-sampled depth residuals sensitive to
        # roll and sampling phase.  Silhouette agreement is the stable coarse
        # selector; metric depth remains a reported diagnostic.
        "geometry_cost": float(1.0 - visible_iou),
        "visible_iou": visible_iou,
        "depth_cost": depth_cost,
    }


def _refine_axial_shift_by_render(
    model_points: np.ndarray,
    candidate: dict[str, Any],
    context: RenderContext,
    camera: dict[str, Any],
    occlusion_tolerance_m: float,
) -> dict[str, Any]:
    """Resolve partial-surface axial sliding with full rendered silhouette agreement."""

    axis = np.asarray(candidate["axis_camera"], dtype=np.float64)
    origin = np.asarray(candidate["origin_camera_m"], dtype=np.float64)

    def shifted(shift_m: float) -> dict[str, Any]:
        item = dict(candidate)
        item["origin_camera_m"] = origin + axis * float(shift_m)
        item["axial_render_shift_m"] = float(shift_m)
        item.update(
            _render_candidate_cost(
                model_points,
                item,
                context,
                camera,
                occlusion_tolerance_m,
            )
        )
        return item

    coarse = [shifted(float(value)) for value in np.arange(-0.06, 0.0601, 0.005)]
    coarse_best = min(coarse, key=lambda item: float(item["geometry_cost"]))
    center = float(coarse_best["axial_render_shift_m"])
    fine = [shifted(float(value)) for value in np.arange(center - 0.005, center + 0.0051, 0.001)]
    return min([coarse_best, *fine], key=lambda item: float(item["geometry_cost"]))


def register_symmetric_points(
    model_points: np.ndarray,
    visible_points: np.ndarray,
    target_mask: np.ndarray,
    depth_raw: np.ndarray,
    camera: dict[str, Any],
    device: Any = None,
    render_workers: int = 4,
) -> tuple[str, np.ndarray | None, dict[str, Any]]:
    """Estimate task-equivalent pose using profile proposals and visible CAD scoring."""

    visible_points = np.asarray(visible_points, dtype=np.float64)
    if len(visible_points) < 20:
        return "failed", None, {"reason": "not_enough_points_for_symmetric_registration"}
    if depth_raw.ndim == 3:
        depth_raw = depth_raw[:, :, 0]
    scene_depth_m = np.asarray(depth_raw, dtype=np.float64) * float(camera["depth_scale"])
    target_mask = np.asarray(target_mask, dtype=bool)
    mask_y, mask_x = np.nonzero(target_mask)
    if not len(mask_x):
        return "failed", None, {"reason": "symmetric_registration_empty_target_mask"}
    render_context = RenderContext(
        target_mask=target_mask,
        scene_depth_m=scene_depth_m,
        image_height=int(target_mask.shape[0]),
        image_width=int(target_mask.shape[1]),
        target_x0=int(np.min(mask_x)),
        target_x1=int(np.max(mask_x)) + 1,
        target_y0=int(np.min(mask_y)),
        target_y1=int(np.max(mask_y)) + 1,
    )
    occlusion_tolerance_m = 0.008

    render_model_points = np.asarray(model_points, dtype=np.float64)

    profile_candidates = optimize_profile_candidates(model_points, visible_points, device=device)
    # Preserve directional coverage: select the best axial initialization for
    # every uniform/PCA axis seed instead of allowing one wrong local mode to
    # occupy the entire render shortlist.
    best_by_axis_seed: dict[int, list[dict[str, Any]]] = {}
    for candidate in profile_candidates:
        seed_index = int(candidate["axis_seed_index"])
        bucket = best_by_axis_seed.setdefault(seed_index, [])
        if len(bucket) < 2:
            bucket.append(candidate)
    candidates = sorted(
        [candidate for bucket in best_by_axis_seed.values() for candidate in bucket],
        key=lambda item: float(item["profile_loss"]),
    )

    def score_render_candidate(candidate: dict[str, Any]) -> None:
        candidate.update(
            _render_candidate_cost(
                render_model_points,
                candidate,
                render_context,
                camera,
                occlusion_tolerance_m,
            )
        )

    worker_count = min(max(1, int(render_workers)), len(candidates), os.cpu_count() or 1)
    if worker_count == 1:
        for candidate in candidates:
            score_render_candidate(candidate)
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            list(executor.map(score_render_candidate, candidates))
    candidates.sort(key=lambda item: (float(item["geometry_cost"]), float(item["profile_loss"])))

    axial_shift_candidates = candidates[:2]

    def refine_axial(candidate: dict[str, Any]) -> dict[str, Any]:
        return _refine_axial_shift_by_render(
            render_model_points,
            candidate,
            render_context,
            camera,
            occlusion_tolerance_m,
        )

    if len(axial_shift_candidates) > 1 and worker_count > 1:
        with ThreadPoolExecutor(max_workers=len(axial_shift_candidates)) as executor:
            shifted_candidates = list(executor.map(refine_axial, axial_shift_candidates))
    else:
        shifted_candidates = [refine_axial(candidate) for candidate in axial_shift_candidates]
    candidates = sorted(
        [*candidates, *shifted_candidates],
        key=lambda item: (float(item["geometry_cost"]), float(item["profile_loss"])),
    )

    ranked = candidates
    if not ranked or not np.isfinite(float(ranked[0]["geometry_cost"])):
        return "failed", None, {"reason": "symmetric_registration_no_finite_candidate"}

    best = ranked[0]
    t_camera_grasp = np.eye(4, dtype=np.float64)
    t_camera_grasp[:3, :3] = canonical_rotation_from_axis(best["axis_camera"])
    t_camera_grasp[:3, 3] = best["origin_camera_m"]
    second_cost = float(ranked[1]["geometry_cost"]) if len(ranked) > 1 else None
    quality = {
        "registration_strategy": "robust_symmetric_profile_then_occlusion_aware_render",
        "roll_about_grasp_x_ignored": True,
        "profile_candidate_count": int(len(profile_candidates)),
        "rendered_candidate_count": int(len(candidates)),
        "render_model_point_count": int(len(render_model_points)),
        "render_worker_count": int(worker_count),
        "axial_shift_candidate_count": int(len(axial_shift_candidates)),
        "geometry_cost": round(float(best["geometry_cost"]), 8),
        "visible_iou": round(float(best["visible_iou"]), 8),
        "depth_cost": round(float(best["depth_cost"]), 8),
        "profile_loss": round(float(best["profile_loss"]), 8),
        "profile_rank": int(best["profile_rank"]),
        "candidate_cost_margin": None
        if second_cost is None
        else round(second_cost - float(best["geometry_cost"]), 8),
        "axis_camera": round_list(best["axis_camera"]),
        "origin_camera_m": round_list(best["origin_camera_m"]),
    }
    if "axial_render_shift_m" in best:
        quality["axial_render_shift_m"] = round(float(best["axial_render_shift_m"]), 8)
    return "ok", t_camera_grasp, quality


__all__ = [
    "ProfileGrid",
    "RenderContext",
    "build_profile_grid",
    "canonical_rotation_from_axis",
    "optimize_profile_candidates",
    "register_symmetric_points",
]
