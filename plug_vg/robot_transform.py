"""Robot-frame transforms for converting camera 6D poses to base-frame poses."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .geometry import rotation_to_quaternion_xyzw


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return data


def load_hand_eye_matrix(path: Path) -> np.ndarray:
    data = load_yaml(path)
    transform_name = data.get("transform_name")
    if transform_name != "T_end_camera":
        raise ValueError(f"Hand-eye config must contain transform_name=T_end_camera, got {transform_name!r}: {path}")
    matrix = data.get("matrix_4x4")
    if matrix is None:
        raise KeyError(f"Hand-eye config missing matrix_4x4: {path}")
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"Hand-eye matrix_4x4 must be 4x4, got {matrix.shape}: {path}")
    return matrix


def rpy_xyz_to_rotation(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Return rotation for fixed-axis XYZ RPY: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    rx = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, math.cos(roll), -math.sin(roll)], [0.0, math.sin(roll), math.cos(roll)]],
        dtype=np.float64,
    )
    ry = np.asarray(
        [[math.cos(pitch), 0.0, math.sin(pitch)], [0.0, 1.0, 0.0], [-math.sin(pitch), 0.0, math.cos(pitch)]],
        dtype=np.float64,
    )
    rz = np.asarray(
        [[math.cos(yaw), -math.sin(yaw), 0.0], [math.sin(yaw), math.cos(yaw), 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return rz @ ry @ rx


def rotation_to_rpy_xyz(rotation: np.ndarray) -> list[float]:
    """Extract fixed-axis XYZ RPY matching R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    r = np.asarray(rotation, dtype=np.float64)
    if r.shape != (3, 3):
        raise ValueError(f"rotation must be 3x3, got {r.shape}")
    sy = math.hypot(float(r[0, 0]), float(r[1, 0]))
    singular = sy < 1e-9
    if not singular:
        roll = math.atan2(float(r[2, 1]), float(r[2, 2]))
        pitch = math.atan2(float(-r[2, 0]), sy)
        yaw = math.atan2(float(r[1, 0]), float(r[0, 0]))
    else:
        roll = math.atan2(float(-r[1, 2]), float(r[1, 1]))
        pitch = math.atan2(float(-r[2, 0]), sy)
        yaw = 0.0
    return [float(roll), float(pitch), float(yaw)]


def robot_pose_to_matrix(pose_xyzrpy_m_rad: list[float] | tuple[float, ...]) -> np.ndarray:
    if len(pose_xyzrpy_m_rad) != 6:
        raise ValueError(f"Robot pose must have 6 values [x,y,z,roll,pitch,yaw], got {len(pose_xyzrpy_m_rad)}")
    x, y, z, roll, pitch, yaw = [float(value) for value in pose_xyzrpy_m_rad]
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rpy_xyz_to_rotation(roll, pitch, yaw)
    matrix[:3, 3] = [x, y, z]
    return matrix


def matrix_to_robot_pose(matrix: np.ndarray) -> list[float]:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"matrix must be 4x4, got {matrix.shape}")
    x, y, z = matrix[:3, 3].tolist()
    roll, pitch, yaw = rotation_to_rpy_xyz(matrix[:3, :3])
    return [float(x), float(y), float(z), roll, pitch, yaw]


def pose_dict_to_matrix(pose: dict[str, Any]) -> np.ndarray:
    translation = np.asarray(pose.get("translation_m"), dtype=np.float64)
    rotation = np.asarray(pose.get("rotation_matrix"), dtype=np.float64)
    if translation.shape != (3,):
        raise ValueError(f"translation_m must have shape (3,), got {translation.shape}")
    if rotation.shape != (3, 3):
        raise ValueError(f"rotation_matrix must have shape (3,3), got {rotation.shape}")
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    return matrix


def round_list(values: Any, digits: int = 8) -> list[Any]:
    rounded = np.round(np.asarray(values, dtype=np.float64), digits)
    rounded[np.isclose(rounded, 0.0, atol=10.0 ** -digits)] = 0.0
    return rounded.tolist()


def convert_camera_grasp_to_base(
    result: dict[str, Any],
    t_base_end: np.ndarray,
    t_end_camera: np.ndarray,
    hand_eye_path: Path,
    robot_config_path: Path | None = None,
) -> dict[str, Any]:
    if result.get("status") != "ok":
        return result
    camera_pose = result.get("grasp_pose_camera")
    if not isinstance(camera_pose, dict):
        raise KeyError("OK result missing grasp_pose_camera")

    t_camera_grasp = pose_dict_to_matrix(camera_pose)
    t_base_grasp = t_base_end @ t_end_camera @ t_camera_grasp
    rotation = t_base_grasp[:3, :3]
    translation = t_base_grasp[:3, 3]
    rpy = rotation_to_rpy_xyz(rotation)
    rpy_deg = np.degrees(np.asarray(rpy, dtype=np.float64))
    robot_pose = [float(translation[0]), float(translation[1]), float(translation[2]), *rpy]
    robot_pose_deg = [float(translation[0]), float(translation[1]), float(translation[2]), *rpy_deg.tolist()]

    result["grasp_pose_base"] = {
        "translation_m": round_list(translation),
        "rotation_matrix": round_list(rotation),
        "quaternion_xyzw": rotation_to_quaternion_xyzw(rotation),
        "rpy_xyz_rad": round_list(rpy),
        "rpy_xyz_deg": round_list(rpy_deg),
        "robot_pose_xyzrpy_m_rad": round_list(robot_pose),
        "robot_pose_xyzrpy_m_deg": round_list(robot_pose_deg),
    }
    result["frame_transform"] = {
        "chain": "T_base_grasp = T_base_end_current @ T_end_camera @ T_camera_grasp",
        "hand_eye_config": str(hand_eye_path),
        "robot_config": None if robot_config_path is None else str(robot_config_path),
        "T_base_end_current": round_list(t_base_end),
        "T_end_camera": round_list(t_end_camera),
        "T_camera_grasp": round_list(t_camera_grasp),
    }
    return result


class RobotPoseProvider:
    """Placeholder for future realtime robot-pose integration."""

    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.config = load_yaml(config_path)

    def get_current_pose(self) -> list[float]:
        raise NotImplementedError(
            "Realtime robot pose provider is not implemented yet. "
            "Use --robot-pose x y z roll pitch yaw to validate base-frame conversion."
        )
