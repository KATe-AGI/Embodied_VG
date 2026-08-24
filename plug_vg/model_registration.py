"""Partial visible point registration against a grasp-frame CAD point cloud."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .grasp_model import transform_points
from .ring_coarse_registration import ring_aligned_candidates
from .robot_transform import round_list
from .visible_points import voxel_downsample


def pca_basis(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = np.asarray(points, dtype=np.float64) - np.mean(points, axis=0)
    cov = np.cov(centered.T)
    values, vectors = np.linalg.eigh(cov)
    order = np.argsort(values)[::-1]
    return values[order], vectors[:, order]


def _rotation_about_x(angle_rad: float) -> np.ndarray:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.asarray([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def candidate_transforms_with_rolls(
    source_points: np.ndarray,
    target_points: np.ndarray,
    roll_angles_deg: tuple[float, ...],
) -> list[dict[str, Any]]:
    values, vectors = pca_basis(target_points)
    source_centroid_grasp = np.mean(source_points, axis=0)
    target_centroid_camera = np.mean(target_points, axis=0)
    candidates: list[dict[str, Any]] = []
    for x_sign in (1.0, -1.0):
        x_axis = vectors[:, 0] * x_sign
        y_seed = vectors[:, 1]
        y_axis = y_seed - float(np.dot(y_seed, x_axis)) * x_axis
        y_norm = float(np.linalg.norm(y_axis))
        if y_norm < 1e-9:
            continue
        y_axis = y_axis / y_norm
        z_axis = np.cross(x_axis, y_axis)
        z_norm = float(np.linalg.norm(z_axis))
        if z_norm < 1e-9:
            continue
        base_rotation_camera_grasp = np.column_stack([x_axis, y_axis, z_axis / z_norm])
        for roll_deg in roll_angles_deg:
            rotation_camera_grasp = base_rotation_camera_grasp @ _rotation_about_x(
                math.radians(float(roll_deg))
            )
            rotation_grasp_camera = rotation_camera_grasp.T
            t_grasp_camera = np.eye(4, dtype=np.float64)
            t_grasp_camera[:3, :3] = rotation_grasp_camera
            t_grasp_camera[:3, 3] = source_centroid_grasp - rotation_grasp_camera @ target_centroid_camera
            candidates.append(
                {
                    "name": f"pca_x_{int(x_sign):+d}_roll_{int(roll_deg):+d}",
                    "t_grasp_camera": t_grasp_camera,
                    "target_pca_eigenvalues": round_list(values, digits=10),
                }
            )
    return candidates


def select_registration_candidate(
    ranked: list[dict[str, Any]],
    min_registration_fitness: float,
    max_inlier_rmse_m: float,
) -> tuple[str, dict[str, Any] | None, dict[str, Any]]:
    if not ranked:
        return "failed", None, {"reason": "registration_failed_no_candidate"}
    ranked = sorted(ranked, key=lambda item: (-float(item["fitness"]), float(item["rmse"])))
    best = ranked[0]
    quality: dict[str, Any] = {
        "candidate": best["summary"]["name"],
        "fitness": round(float(best["fitness"]), 6),
        "inlier_rmse": round(float(best["rmse"]), 8),
        "min_registration_fitness": float(min_registration_fitness),
        "max_inlier_rmse_m": float(max_inlier_rmse_m),
    }
    if best["fitness"] < min_registration_fitness:
        quality["reason"] = "registration_low_fitness"
        return "failed", None, quality
    if best["rmse"] > max_inlier_rmse_m:
        quality["reason"] = "registration_high_rmse"
        return "failed", None, quality
    # Roll about the plug's grasp-frame X axis does not change its collinear
    # grasp/tail/head semantic points.  Treat all candidates with the same
    # directed X axis as one grasp solution and only compare against the best
    # head-tail-reversed solution.
    # Internally every candidate maps camera/scene points into the grasp/model
    # frame.  Row 0 of R_grasp_camera is the directed grasp X axis expressed
    # in camera coordinates.
    best_axis = np.asarray(best["t_grasp_camera"], dtype=np.float64)[0, :3]
    same_axis = []
    opposite_axis = []
    for item in ranked:
        axis = np.asarray(item["t_grasp_camera"], dtype=np.float64)[0, :3]
        (same_axis if float(np.dot(best_axis, axis)) >= 0.0 else opposite_axis).append(item)
    quality["roll_equivalent_candidates"] = int(len(same_axis))
    quality["directed_axis_groups"] = 1 + int(bool(opposite_axis))
    quality["roll_about_grasp_x_ignored"] = True
    if opposite_axis:
        opposite = opposite_axis[0]
        fitness_gap = float(best["fitness"] - opposite["fitness"])
        rmse_gap = float(opposite["rmse"] - best["rmse"])
        quality["opposite_axis_candidate"] = opposite["summary"]["name"]
        quality["opposite_axis_fitness"] = round(float(opposite["fitness"]), 6)
        quality["opposite_axis_inlier_rmse"] = round(float(opposite["rmse"]), 8)
        quality["opposite_axis_fitness_gap"] = round(fitness_gap, 8)
        quality["opposite_axis_rmse_gap_m"] = round(rmse_gap, 8)
        if abs(fitness_gap) < 1e-6 and abs(rmse_gap) < 1e-6:
            quality["reason"] = "registration_ambiguous_head_tail"
            return "ambiguous", best, quality
    return "ok", best, quality


def _pcd(points: np.ndarray):
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    return pcd


def _nearest_neighbor_score(model_points: np.ndarray, scene_points: np.ndarray, t_grasp_camera: np.ndarray, threshold_m: float) -> tuple[float, float]:
    import open3d as o3d

    model_pcd = _pcd(model_points)
    tree = o3d.geometry.KDTreeFlann(model_pcd)
    scene_in_grasp = transform_points(scene_points, t_grasp_camera)
    distances: list[float] = []
    for point in scene_in_grasp:
        k, _idx, d2 = tree.search_knn_vector_3d(point, 1)
        if k:
            distances.append(float(math.sqrt(d2[0])))
    if not distances:
        return 0.0, float("inf")
    d = np.asarray(distances, dtype=np.float64)
    inlier = d <= float(threshold_m)
    if not np.any(inlier):
        return 0.0, float("inf")
    return float(np.mean(inlier)), float(np.sqrt(np.mean(np.square(d[inlier]))))


def register_visible_points(
    model_points: np.ndarray,
    visible_points: np.ndarray,
    voxel_size_m: float,
    icp_threshold_m: float,
    icp_iterations: int,
    max_model_points: int,
    max_scene_points: int,
    min_registration_fitness: float,
    max_inlier_rmse_m: float,
    seed: int,
) -> tuple[str, np.ndarray | None, dict[str, Any]]:
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError("open3d is required for registration. Install it with `pip install open3d`.") from exc

    rng = np.random.default_rng(seed)
    model_points = np.asarray(model_points, dtype=np.float64)
    visible_points = np.asarray(visible_points, dtype=np.float64)
    if len(model_points) > max_model_points:
        model_points = model_points[np.sort(rng.choice(len(model_points), max_model_points, replace=False))]
    if len(visible_points) > max_scene_points:
        visible_points = visible_points[np.sort(rng.choice(len(visible_points), max_scene_points, replace=False))]
    model_down = voxel_downsample(model_points, voxel_size_m)
    # The visible cloud is already voxelized during mask/depth extraction.
    # Repeating the same grid operation here is an exact no-op and obscures
    # which stage owns scene preprocessing.
    scene_down = visible_points
    if len(model_down) < 20 or len(scene_down) < 20:
        return "failed", None, {
            "reason": "not_enough_points_after_preprocessing",
            "model_points_used": int(len(model_down)),
            "scene_points_used": int(len(scene_down)),
        }

    pca_candidates = candidate_transforms_with_rolls(
        model_down,
        scene_down,
        (0.0, 90.0, 180.0, 270.0),
    )
    scene_axis = pca_basis(scene_down)[1][:, 0]
    candidates = ring_aligned_candidates(
        model_down,
        scene_down,
        scene_axis,
        pca_candidates,
        voxel_size_m,
    )
    if not candidates:
        return "failed", None, {"reason": "large_ring_detection_failed"}

    scored: list[dict[str, Any]] = []
    for candidate in candidates:
        fitness, rmse = _nearest_neighbor_score(
            model_down, scene_down, candidate["t_grasp_camera"], icp_threshold_m
        )
        scored.append({**candidate, "coarse_fitness": fitness, "coarse_rmse": rmse})
    scored.sort(key=lambda item: (-float(item["coarse_fitness"]), float(item["coarse_rmse"])))

    model_pcd = _pcd(model_down)
    scene_pcd = _pcd(scene_down)
    estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=int(icp_iterations))
    candidate_summaries: list[dict[str, Any]] = []
    ranked: list[dict[str, Any]] = []
    for item in scored:
        reg = o3d.pipelines.registration.registration_icp(
            scene_pcd,
            model_pcd,
            float(icp_threshold_m),
            item["t_grasp_camera"],
            estimation,
            criteria,
        )
        t_grasp_camera = np.asarray(reg.transformation, dtype=np.float64)
        summary = {
            "name": item["name"],
            "coarse_fitness": round(float(item["coarse_fitness"]), 6),
            "coarse_rmse": None if not np.isfinite(item["coarse_rmse"]) else round(float(item["coarse_rmse"]), 8),
            "icp_fitness": round(float(reg.fitness), 6),
            "icp_inlier_rmse": round(float(reg.inlier_rmse), 8),
            "target_pca_eigenvalues": item["target_pca_eigenvalues"],
        }
        if item.get("initializer") == "large_ring_center":
            summary.update(
                {
                    "initializer": item["initializer"],
                    "ring_hypothesis": item["ring_hypothesis"],
                    "model_ring_center_grasp_m": round_list(item["model_ring_center_grasp_m"]),
                    "scene_ring_center_camera_m": round_list(item["scene_ring_center_camera_m"]),
                    "model_ring_radius_m": round(float(item["model_ring_radius_m"]), 8),
                    "scene_ring_radius_m": round(float(item["scene_ring_radius_m"]), 8),
                    "ring_fit_residual_m": round(float(item["ring_fit_residual_m"]), 8),
                    "ring_point_count": int(item["ring_point_count"]),
                }
            )
        candidate_summaries.append(summary)
        ranked.append(
            {
                "summary": summary,
                "t_grasp_camera": t_grasp_camera,
                "fitness": float(reg.fitness),
                "rmse": float(reg.inlier_rmse),
                "initializer": item.get("initializer", "pca_centroid"),
            }
        )

    status, best, gate_quality = select_registration_candidate(
        ranked,
        min_registration_fitness,
        max_inlier_rmse_m,
    )
    quality: dict[str, Any] = {
        **gate_quality,
        "coarse_candidate": scored[0]["name"],
        "coarse_fitness": round(float(scored[0]["coarse_fitness"]), 6),
        "coarse_rmse": None
        if not np.isfinite(scored[0]["coarse_rmse"])
        else round(float(scored[0]["coarse_rmse"]), 8),
        "t_grasp_camera_coarse": round_list(scored[0]["t_grasp_camera"]),
        "icp_threshold_m": float(icp_threshold_m),
        "voxel_size_m": float(voxel_size_m),
        "model_points_used": int(len(model_down)),
        "scene_points_used": int(len(scene_down)),
        "scene_voxelized_before_registration": True,
        "scene_filter_strategy": "pass_through_then_global_mad_then_centroid_voxel",
        "registration_direction": "scene_camera_to_model_grasp",
        "registration_strategy": "large_ring_coarse_then_icp",
        "candidates": candidate_summaries,
    }
    if status == "failed" or best is None:
        return status, None, quality
    # Public/downstream convention is model/grasp -> camera.  Registration
    # stays camera/scene -> grasp/model internally and is inverted only here.
    t_camera_grasp = np.linalg.inv(np.asarray(best["t_grasp_camera"], dtype=np.float64))
    return status, t_camera_grasp, quality


__all__ = [
    "candidate_transforms_with_rolls",
    "pca_basis",
    "register_visible_points",
    "select_registration_candidate",
]
