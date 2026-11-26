"""Standard statistical outlier removal, preserving observed coordinates and order.

The filter is opt-in: its pose accuracy must be validated for a new capture domain.
"""
from __future__ import annotations

import numpy as np


class PreprocessingError(RuntimeError):
    """A numerical or dependency failure that must not reach registration."""


def add_point_filter_arguments(parser) -> None:
    """Share the same opt-in preprocessing controls across inference entrypoints."""
    import argparse

    def neighbors(value):
        try:
            number = int(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("must be an integer >= 2") from exc
        if number < 2:
            raise argparse.ArgumentTypeError("must be an integer >= 2")
        return number

    def ratio(value):
        try:
            number = float(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("must be finite and > 0") from exc
        if not np.isfinite(number) or number <= 0:
            raise argparse.ArgumentTypeError("must be finite and > 0")
        return number

    parser.add_argument("--point-filter", choices=("none", "sor", "bilateral", "edge_clean", "depth_consistency"), default="none",
                        help="Point preprocessing defaults to the original baseline (none); alternatives are explicit experiments.")
    parser.add_argument("--sor-neighbors", type=neighbors, default=20,
                        help="Open3D SOR neighborhood size (>=2).")
    parser.add_argument("--sor-std-ratio", type=ratio, default=6.0,
                        help="Open3D SOR standard-deviation multiplier; larger values remove fewer points.")
    def angle(value):
        number = ratio(value)
        if number > 90:
            raise argparse.ArgumentTypeError("must be in (0, 90]")
        return number
    parser.add_argument("--bilateral-neighbors", type=neighbors, default=20,
                        help="CGAL PCA and bilateral smoothing KNN size.")
    parser.add_argument("--bilateral-sharpness-angle", type=angle, default=15.,
                        help="CGAL sharpness angle in degrees, (0,90]; larger values smooth more.")


def statistical_inlier_mask(
    points: np.ndarray, nb_neighbors: int = 20, std_ratio: float = 6.0,
) -> np.ndarray:
    """Return Open3D SOR inliers; never smooth, reorder or refill observations.

    Tiny and coincident clouds have no usable neighborhood-distance statistics
    and pass through. The extraction pipeline still enforces its minimum count.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("points must be a finite Nx3 array")
    if isinstance(nb_neighbors, bool) or int(nb_neighbors) != nb_neighbors or nb_neighbors < 2:
        raise ValueError("sor_neighbors must be an integer >= 2")
    if not np.isfinite(std_ratio) or std_ratio <= 0:
        raise ValueError("sor_std_ratio must be finite and > 0")
    count = len(points)
    if count < 3 or np.all(points == points[0]):
        return np.ones(count, dtype=bool)

    import open3d as o3d

    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    _, indices = cloud.remove_statistical_outlier(
        nb_neighbors=min(int(nb_neighbors), count), std_ratio=float(std_ratio),
    )
    keep = np.zeros(count, dtype=bool)
    keep[np.asarray(indices, dtype=np.int64)] = True
    return keep


def bilateral_smooth_points(points: np.ndarray, nb_neighbors: int = 20,
                            sharpness_angle: float = 15.) -> np.ndarray:
    """Run one CGAL bilateral iteration with KNN support and PCA normals.

    A fixed radius is deliberately not used: isolated points can have empty
    support and produce NaNs in CGAL 6.0.1. Invalid results fail explicitly.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("points must be a finite Nx3 array")
    if isinstance(nb_neighbors, bool) or int(nb_neighbors) != nb_neighbors or nb_neighbors < 2:
        raise ValueError("bilateral_neighbors must be an integer >= 2")
    if not np.isfinite(sharpness_angle) or not 0 < sharpness_angle <= 90:
        raise ValueError("bilateral_sharpness_angle must be in (0, 90]")
    if len(points) < 3 or np.all(points == points[0]):
        return points.copy()
    try:
        from CGAL.CGAL_Kernel import Point_3
        from CGAL.CGAL_Point_set_3 import Point_set_3
        from CGAL.CGAL_Point_set_processing_3 import bilateral_smooth_point_set, pca_estimate_normals
    except ImportError as exc:
        raise PreprocessingError("bilateral requires CGAL==6.0.1.post202410241521") from exc

    cloud = Point_set_3()
    for point in points:
        cloud.insert(Point_3(*point))
    cloud.add_normal_map()
    neighbors = min(int(nb_neighbors), len(points) - 1)
    pca_estimate_normals(cloud, neighbors)
    bilateral_smooth_point_set(cloud, neighbors, 0., float(sharpness_angle))
    smoothed = np.asarray([[p.x(), p.y(), p.z()] for p in cloud.points()], dtype=np.float64)
    if smoothed.shape != points.shape or not np.isfinite(smoothed).all():
        raise PreprocessingError("CGAL bilateral returned invalid coordinates")
    return smoothed
