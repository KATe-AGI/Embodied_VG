"""Partial visible point registration against a grasp-frame CAD point cloud."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .grasp_model import transform_points
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


def candidate_transforms_with_rolls(source_points: np.ndarray, target_points: np.ndarray, roll_angles_deg: tuple[float, ...]) -> list[dict[str, Any]]:
    values, vectors = pca_basis(target_points)
    source_centroid = np.mean(source_points, axis=0)
    target_centroid = np.mean(target_points, axis=0)
    candidates: list[dict[str, Any]] = []
    for x_sign in (1.0, -1.0):
        x_axis = vectors[:, 0] * x_sign
        for y_sign in (1.0, -1.0):
            y_seed = vectors[:, 1] * y_sign
            y_axis = y_seed - float(np.dot(y_seed, x_axis)) * x_axis
            y_norm = float(np.linalg.norm(y_axis))
            if y_norm < 1e-9:
                continue
            y_axis = y_axis / y_norm
            z_axis = np.cross(x_axis, y_axis)
            z_norm = float(np.linalg.norm(z_axis))
            if z_norm < 1e-9:
                continue
            base_rotation = np.column_stack([x_axis, y_axis, z_axis / z_norm])
            for roll_deg in roll_angles_deg:
                rotation = base_rotation @ _rotation_about_x(math.radians(float(roll_deg)))
                transform = np.eye(4, dtype=np.float64)
                transform[:3, :3] = rotation
                transform[:3, 3] = target_centroid - rotation @ source_centroid
                candidates.append(
                    {
                        "name": f"pca_x_{int(x_sign):+d}_y_{int(y_sign):+d}_roll_{int(roll_deg):+d}",
                        "transform": transform,
                        "target_pca_eigenvalues": round_list(values, digits=10),
                    }
                )
    return candidates


def select_registration_candidate(
    ranked: list[dict[str, Any]],
    min_registration_fitness: float,
    max_inlier_rmse_m: float,
    ambiguity_fitness_margin: float,
    ambiguity_rmse_margin_m: float,
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
    if len(ranked) > 1:
        second = ranked[1]
        fitness_gap = float(best["fitness"] - second["fitness"])
        rmse_gap = float(second["rmse"] - best["rmse"])
        quality["second_candidate"] = second["summary"]["name"]
        quality["candidate_fitness_gap"] = round(fitness_gap, 8)
        quality["candidate_rmse_gap_m"] = round(rmse_gap, 8)
        if fitness_gap <= ambiguity_fitness_margin and abs(rmse_gap) <= ambiguity_rmse_margin_m:
            quality["reason"] = "registration_ambiguous_pose"
            return "ambiguous", best, quality
    return "ok", best, quality


def _pcd(points: np.ndarray):
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    return pcd


def _statistical_outlier_filter(points: np.ndarray, nb_neighbors: int, std_ratio: float) -> tuple[np.ndarray, int]:
    if nb_neighbors <= 0 or len(points) <= nb_neighbors:
        return points, 0
    pcd = _pcd(points)
    filtered, indices = pcd.remove_statistical_outlier(nb_neighbors=int(nb_neighbors), std_ratio=float(std_ratio))
    return np.asarray(filtered.points), int(len(points) - len(indices))


def _nearest_neighbor_score(model_points: np.ndarray, scene_points: np.ndarray, t_camera_grasp: np.ndarray, threshold_m: float) -> tuple[float, float]:
    import open3d as o3d

    model_pcd = _pcd(model_points)
    tree = o3d.geometry.KDTreeFlann(model_pcd)
    t_grasp_camera = np.linalg.inv(t_camera_grasp)
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
    outlier_nb_neighbors: int,
    outlier_std_ratio: float,
    icp_threshold_m: float,
    icp_iterations: int,
    max_model_points: int,
    max_scene_points: int,
    min_registration_fitness: float,
    max_inlier_rmse_m: float,
    ambiguity_fitness_margin: float,
    ambiguity_rmse_margin_m: float,
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
    scene_down = voxel_downsample(visible_points, voxel_size_m)
    scene_down, outlier_rejected = _statistical_outlier_filter(scene_down, outlier_nb_neighbors, outlier_std_ratio)
    if len(model_down) < 20 or len(scene_down) < 20:
        return "failed", None, {
            "reason": "not_enough_points_after_preprocessing",
            "model_points_used": int(len(model_down)),
            "scene_points_used": int(len(scene_down)),
            "scene_outlier_rejected": int(outlier_rejected),
        }

    candidates = candidate_transforms_with_rolls(model_down, scene_down, (0.0, 90.0, 180.0, 270.0))
    if not candidates:
        return "failed", None, {"reason": "initial_transform_generation_failed"}

    scored: list[dict[str, Any]] = []
    for candidate in candidates:
        fitness, rmse = _nearest_neighbor_score(model_down, scene_down, candidate["transform"], icp_threshold_m)
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
            np.linalg.inv(item["transform"]),
            estimation,
            criteria,
        )
        t_camera_grasp = np.linalg.inv(np.asarray(reg.transformation, dtype=np.float64))
        summary = {
            "name": item["name"],
            "coarse_fitness": round(float(item["coarse_fitness"]), 6),
            "coarse_rmse": None if not np.isfinite(item["coarse_rmse"]) else round(float(item["coarse_rmse"]), 8),
            "icp_fitness": round(float(reg.fitness), 6),
            "icp_inlier_rmse": round(float(reg.inlier_rmse), 8),
            "target_pca_eigenvalues": item["target_pca_eigenvalues"],
        }
        candidate_summaries.append(summary)
        ranked.append({"summary": summary, "transform": t_camera_grasp, "fitness": float(reg.fitness), "rmse": float(reg.inlier_rmse)})

    status, best, gate_quality = select_registration_candidate(
        ranked,
        min_registration_fitness,
        max_inlier_rmse_m,
        ambiguity_fitness_margin,
        ambiguity_rmse_margin_m,
    )
    quality: dict[str, Any] = {
        **gate_quality,
        "icp_threshold_m": float(icp_threshold_m),
        "voxel_size_m": float(voxel_size_m),
        "model_points_used": int(len(model_down)),
        "scene_points_used": int(len(scene_down)),
        "scene_outlier_rejected": int(outlier_rejected),
        "registration_direction": "scene_to_model_then_invert",
        "candidates": candidate_summaries,
    }
    if status == "failed" or best is None:
        return status, None, quality
    return status, np.asarray(best["transform"], dtype=np.float64), quality


__all__ = [
    "candidate_transforms_with_rolls",
    "pca_basis",
    "register_visible_points",
    "select_registration_candidate",
]
