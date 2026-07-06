"""Small geometry helpers shared by the CAD-registration pipeline."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np


def polygon_mask(polygon_xy: list[list[float]], shape: tuple[int, int]) -> np.ndarray:
    import cv2

    mask = np.zeros(shape, dtype=np.uint8)
    if not polygon_xy:
        return mask
    pts = np.asarray(polygon_xy, dtype=np.float32)
    pts[:, 0] = np.clip(pts[:, 0], 0, shape[1] - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, shape[0] - 1)
    cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 1)
    return mask


def rotation_to_quaternion_xyzw(rotation: np.ndarray) -> list[float]:
    m = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(m))
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        qw = (m[2, 1] - m[1, 2]) / s
        qx = 0.25 * s
        qy = (m[0, 1] + m[1, 0]) / s
        qz = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        qw = (m[0, 2] - m[2, 0]) / s
        qx = (m[0, 1] + m[1, 0]) / s
        qy = 0.25 * s
        qz = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        qw = (m[1, 0] - m[0, 1]) / s
        qx = (m[0, 2] + m[2, 0]) / s
        qy = (m[1, 2] + m[2, 1]) / s
        qz = 0.25 * s
    quat = np.asarray([qx, qy, qz, qw], dtype=np.float64)
    quat /= np.linalg.norm(quat)
    return [round(float(v), 8) for v in quat.tolist()]


def project_point(point: np.ndarray, camera: dict) -> tuple[int, int] | None:
    if point[2] <= 1e-6:
        return None
    u = float(camera["fx"]) * point[0] / point[2] + float(camera["cx"])
    v = float(camera["fy"]) * point[1] / point[2] + float(camera["cy"])
    if not np.isfinite(u) or not np.isfinite(v):
        return None
    return int(round(u)), int(round(v))


def project_point_float(point: np.ndarray, camera: dict) -> np.ndarray | None:
    if point[2] <= 1e-6:
        return None
    u = float(camera["fx"]) * point[0] / point[2] + float(camera["cx"])
    v = float(camera["fy"]) * point[1] / point[2] + float(camera["cy"])
    if not np.isfinite(u) or not np.isfinite(v):
        return None
    return np.asarray([u, v], dtype=np.float64)


def save_ply(points: np.ndarray, output_path: Path, max_points: int = 20000) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=np.float64)
    if len(points) > max_points:
        rng = np.random.default_rng(0)
        points = points[rng.choice(len(points), size=max_points, replace=False)]
    with output_path.open("w", encoding="ascii") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("end_header\n")
        for point in points:
            f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")
