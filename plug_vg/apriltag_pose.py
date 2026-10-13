"""Official AprilRobotics pose estimation and camera-facing center/normal geometry."""

from __future__ import annotations

import ctypes
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import shlex
import subprocess
import tempfile

import numpy as np

SOURCE = Path(__file__).resolve().parents[1] / "third_party/apriltag_pose"


@lru_cache(maxsize=1)
def _pose_library() -> ctypes.CDLL:
    """Build only the vendored pose/math subset; reuse its content-addressed cache."""
    digest = hashlib.sha256()
    for path in sorted(SOURCE.rglob("*")):
        if path.suffix in (".c", ".h"):
            digest.update(path.relative_to(SOURCE).as_posix().encode())
            digest.update(path.read_bytes())
    cache = Path(tempfile.gettempdir()) / f"embodiedvg-apriltag-pose-{os.getuid()}"
    cache.mkdir(mode=0o700, exist_ok=True)
    library_path = cache / f"pose-{digest.hexdigest()[:20]}.so"
    if not library_path.is_file():
        compiler = shlex.split(os.environ.get("CC", "cc"))
        sources = ["pose_bridge.c", "apriltag_pose.c", "common/homography.c", "common/matd.c", "common/svd22.c"]
        with tempfile.TemporaryDirectory(dir=cache) as build_dir:
            output = Path(build_dir) / "pose.so"
            command = [*compiler, "-std=c99", "-D_DEFAULT_SOURCE", "-O2", "-fPIC", "-shared", "-pthread",
                       "-I", str(SOURCE), *[str(SOURCE / name) for name in sources], "-lm", "-o", str(output)]
            try:
                subprocess.run(command, check=True, capture_output=True, text=True)
            except FileNotFoundError as exc:
                raise RuntimeError("Official AprilTag pose code needs a C compiler (Ubuntu: install build-essential)") from exc
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(f"Could not compile official AprilTag pose code:\n{exc.stderr}") from exc
            os.replace(output, library_path)
    library = ctypes.CDLL(str(library_path))
    pointer = ctypes.POINTER(ctypes.c_double)
    library.embodiedvg_estimate_tag_pose.argtypes = [pointer, pointer, *[ctypes.c_double] * 5, pointer, pointer, pointer]
    library.embodiedvg_estimate_tag_pose.restype = ctypes.c_int
    return library


def estimate_tag_pose(tag: object, camera: dict, tagsize: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Run upstream estimate_tag_pose() with the detector's original corner order/H.

    tagsize is the measured blue 5x5 pose square side in meters, passed directly
    to the official estimator. Outputs are R_camera_tag and t_camera_tag.
    """
    calibration = np.array([tagsize, *[camera[key] for key in ("fx", "fy", "cx", "cy")]], float)
    if not np.isfinite(calibration).all() or min(tagsize, camera["fx"], camera["fy"]) <= 0:
        raise ValueError("Tag size and camera focal lengths must be finite and positive")
    homography = np.ascontiguousarray(tag.homography, dtype=np.float64).reshape(3, 3)
    corners = np.ascontiguousarray(tag.corners, dtype=np.float64).reshape(4, 2)
    if not np.isfinite(homography).all() or not np.isfinite(corners).all() or abs(np.linalg.det(homography)) < 1e-12:
        raise ValueError("Tag has invalid corners or homography")
    rotation = np.zeros((3, 3), dtype=np.float64)
    translation = np.zeros(3, dtype=np.float64)
    error = ctypes.c_double()
    pointer = ctypes.POINTER(ctypes.c_double)
    status = _pose_library().embodiedvg_estimate_tag_pose(
        homography.ctypes.data_as(pointer), corners.ctypes.data_as(pointer), tagsize,
        *[float(camera[key]) for key in ("fx", "fy", "cx", "cy")],
        rotation.ctypes.data_as(pointer), translation.ctypes.data_as(pointer), ctypes.byref(error),
    )
    if status or not np.isfinite(rotation).all() or not np.isfinite(translation).all() or translation[2] <= 0:
        raise ValueError("Official AprilTag estimator did not return a finite pose in front of camera")
    return rotation, translation, error.value


def get_tag_center_normal(rotation: np.ndarray, translation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return tag center and unit normal pointing into the camera's half-space."""
    rotation = np.asarray(rotation, dtype=float).reshape(3, 3)
    center = np.asarray(translation, dtype=float).reshape(3).copy()
    if not np.isfinite(rotation).all() or not np.isfinite(center).all():
        raise ValueError("Pose must be finite")
    normal = rotation[:, 2].copy()
    length = float(np.linalg.norm(normal))
    if length < 1e-12:
        raise ValueError("Pose has a zero normal")
    normal /= length
    if np.dot(normal, -center) < 0:
        normal = -normal
    return center, normal


def center_normal_to_base(center_camera: np.ndarray, normal_camera: np.ndarray,
                          t_base_camera: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Transform a point with R,t and a direction with R only."""
    transform = np.asarray(t_base_camera, dtype=float)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("T_base_camera must be a finite rigid 4x4 transform")
    rotation = transform[:3, :3]
    if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6)
            or not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-8)):
        raise ValueError("T_base_camera must be a rigid transform with a proper rotation")
    center = rotation @ np.asarray(center_camera) + transform[:3, 3]
    normal = rotation @ np.asarray(normal_camera)
    normal /= np.linalg.norm(normal)
    return center, normal
