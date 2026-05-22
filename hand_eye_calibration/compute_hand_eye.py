#!/usr/bin/env python3
"""Compute eye-in-hand or eye-to-hand calibration from manually collected data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "config.yaml"
DEFAULT_DATA_ROOT = ROOT / "eye_hand_data"
POSE_FILE = "poses.txt"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
MIN_VALID_IMAGES = 5

METHODS = {
    "TSAI": cv2.CALIB_HAND_EYE_TSAI,
    "PARK": cv2.CALIB_HAND_EYE_PARK,
    "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
    "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
    "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("in-hand", "to-hand"), required=True, help="Hand-eye calibration mode.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Calibration batch directory containing numbered images and poses.txt.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to the selected calibration batch directory.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Checkerboard config YAML.")
    parser.add_argument("--method", choices=tuple(METHODS), default="TSAI", help="OpenCV hand-eye method.")
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    return (ROOT / path).resolve()


def resolve_data_dir(data_dir: Path | None) -> Path:
    if data_dir is not None:
        path = resolve_path(data_dir)
        if not path.is_dir():
            raise NotADirectoryError(f"Calibration data directory not found: {path}")
        return path

    if not DEFAULT_DATA_ROOT.is_dir():
        raise NotADirectoryError(f"Default calibration data root not found: {DEFAULT_DATA_ROOT}")

    candidates = sorted(path for path in DEFAULT_DATA_ROOT.iterdir() if path.is_dir() and (path / POSE_FILE).is_file())
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"No calibration batch with {POSE_FILE} found under {DEFAULT_DATA_ROOT}")
    names = "\n  ".join(str(path.relative_to(ROOT)) for path in candidates)
    raise ValueError(f"Multiple calibration batches found. Please specify --data-dir explicitly:\n  {names}")


def load_checkerboard_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    checkerboard = data.get("checkerboard_args") or {}
    required = ("XX", "YY", "L")
    missing = [name for name in required if name not in checkerboard]
    if missing:
        raise KeyError(f"Checkerboard config missing required key(s): {', '.join(missing)}")
    return {
        "XX": int(checkerboard["XX"]),
        "YY": int(checkerboard["YY"]),
        "L": float(checkerboard["L"]),
    }


def collect_numbered_images(data_dir: Path) -> tuple[list[tuple[int, Path]], list[dict[str, str]]]:
    by_index: dict[int, list[Path]] = {}
    ignored: list[dict[str, str]] = []

    for path in sorted(data_dir.iterdir()):
        if not path.is_file() or path.name == POSE_FILE:
            continue
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if not path.stem.isdigit():
            ignored.append({"file": path.name, "reason": "non_numeric_stem"})
            continue
        by_index.setdefault(int(path.stem), []).append(path)

    images: list[tuple[int, Path]] = []
    for index, paths in sorted(by_index.items()):
        if len(paths) > 1:
            names = ", ".join(path.name for path in paths)
            raise ValueError(f"Duplicate image index {index}: {names}")
        images.append((index, paths[0]))
    if not images:
        raise FileNotFoundError(f"No numbered calibration images found in {data_dir}")
    return images, ignored


def load_poses(path: Path) -> list[list[float]]:
    poses: list[list[float]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            values = [float(item.strip()) for item in line.split(",")]
            if len(values) != 6:
                raise ValueError(f"{path}:{line_no} must contain 6 comma-separated values, got {len(values)}")
            poses.append(values)
    if not poses:
        raise ValueError(f"No robot poses found in {path}")
    return poses


def euler_angles_to_rotation_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    rx_matrix = np.array(
        [[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]],
        dtype=np.float64,
    )
    ry_matrix = np.array(
        [[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]],
        dtype=np.float64,
    )
    rz_matrix = np.array(
        [[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]],
        dtype=np.float64,
    )
    return rz_matrix @ ry_matrix @ rx_matrix


def pose_to_matrix(pose: list[float]) -> np.ndarray:
    x, y, z, rx, ry, rz = pose
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = euler_angles_to_rotation_matrix(rx, ry, rz)
    matrix[:3, 3] = [x, y, z]
    return matrix


def inverse_matrix(matrix: np.ndarray) -> np.ndarray:
    inverse = np.eye(4, dtype=np.float64)
    rotation = matrix[:3, :3]
    translation = matrix[:3, 3]
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -(rotation.T @ translation)
    return inverse


def object_points(checkerboard: dict[str, Any]) -> np.ndarray:
    xx = int(checkerboard["XX"])
    yy = int(checkerboard["YY"])
    square_size = float(checkerboard["L"])
    points = np.zeros((xx * yy, 3), np.float32)
    points[:, :2] = np.mgrid[0:xx, 0:yy].T.reshape(-1, 2)
    return square_size * points


def calibrate_camera(
    images: list[tuple[int, Path]],
    poses: list[list[float]],
    checkerboard: dict[str, Any],
) -> tuple[
    float,
    np.ndarray,
    np.ndarray,
    list[np.ndarray],
    list[np.ndarray],
    list[dict[str, Any]],
    list[dict[str, Any]],
    tuple[int, int],
]:
    criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)
    template_points = object_points(checkerboard)
    object_points_list: list[np.ndarray] = []
    image_points_list: list[np.ndarray] = []
    used_images: list[dict[str, Any]] = []
    failed_images: list[dict[str, Any]] = []
    image_size: tuple[int, int] | None = None

    for index, image_path in images:
        if index < 1 or index > len(poses):
            raise IndexError(f"Image {image_path.name} maps to pose line {index}, but {POSE_FILE} has {len(poses)} rows")

        image = cv2.imread(str(image_path))
        if image is None:
            failed_images.append({"index": index, "file": image_path.name, "reason": "unreadable"})
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_size = gray.shape[::-1]
        ok, corners = cv2.findChessboardCorners(gray, (checkerboard["XX"], checkerboard["YY"]), None)
        if not ok:
            failed_images.append({"index": index, "file": image_path.name, "reason": "corners_not_found"})
            continue

        refined_corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
        object_points_list.append(template_points)
        image_points_list.append(refined_corners)
        used_images.append({"index": index, "file": image_path.name})

    if image_size is None:
        raise RuntimeError("No readable calibration image found")
    if len(image_points_list) < MIN_VALID_IMAGES:
        raise RuntimeError(f"Need at least {MIN_VALID_IMAGES} valid calibration images, got {len(image_points_list)}")

    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points_list, image_points_list, image_size, None, None
    )

    for used, objp, imgp, rvec, tvec in zip(used_images, object_points_list, image_points_list, rvecs, tvecs):
        projected, _ = cv2.projectPoints(objp, rvec, tvec, camera_matrix, dist_coeffs)
        error = cv2.norm(imgp, projected, cv2.NORM_L2) / len(projected)
        used["reprojection_error_px"] = round(float(error), 8)

    return rms, camera_matrix, dist_coeffs, rvecs, tvecs, used_images, failed_images, image_size


def robot_inputs_for_mode(mode: str, used_images: list[dict[str, Any]], poses: list[list[float]]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    rotations: list[np.ndarray] = []
    translations: list[np.ndarray] = []
    for item in used_images:
        pose = poses[int(item["index"]) - 1]
        matrix = pose_to_matrix(pose)
        if mode == "to-hand":
            matrix = inverse_matrix(matrix)
        rotations.append(matrix[:3, :3])
        translations.append(matrix[:3, 3].reshape(3, 1))
    return rotations, translations


def target_to_camera_inputs(rvecs: list[np.ndarray], tvecs: list[np.ndarray]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    rotations: list[np.ndarray] = []
    translations: list[np.ndarray] = []
    for rvec, tvec in zip(rvecs, tvecs):
        rotation, _ = cv2.Rodrigues(rvec)
        rotations.append(rotation.astype(np.float64))
        translations.append(tvec.astype(np.float64))
    return rotations, translations


def matrix_from_rt(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation.reshape(3)
    return matrix


def rounded_list(array: np.ndarray, digits: int = 8) -> list[Any]:
    return np.round(array.astype(np.float64), digits).tolist()


def build_result(
    mode: str,
    method: str,
    data_dir: Path,
    config_path: Path,
    checkerboard: dict[str, Any],
    image_size: tuple[int, int],
    camera_rms: float,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    hand_eye_matrix: np.ndarray,
    used_images: list[dict[str, Any]],
    failed_images: list[dict[str, Any]],
    ignored_images: list[dict[str, str]],
) -> dict[str, Any]:
    rotation = hand_eye_matrix[:3, :3]
    translation = hand_eye_matrix[:3, 3]
    scipy_rotation = Rotation.from_matrix(rotation)
    mean_error = float(np.mean([item["reprojection_error_px"] for item in used_images]))
    transform_name = "T_end_camera" if mode == "in-hand" else "T_base_camera"
    return {
        "mode": mode,
        "method": method,
        "transform_name": transform_name,
        "data_dir": str(data_dir),
        "config": str(config_path),
        "checkerboard": checkerboard,
        "image_size": {"width": int(image_size[0]), "height": int(image_size[1])},
        "matrix_4x4": rounded_list(hand_eye_matrix),
        "rotation_matrix": rounded_list(rotation),
        "translation_m": rounded_list(translation),
        "quaternion_xyzw": rounded_list(scipy_rotation.as_quat()),
        "euler_xyz_deg": rounded_list(scipy_rotation.as_euler("xyz", degrees=True)),
        "camera_calibration_rms": round(float(camera_rms), 8),
        "camera_matrix": rounded_list(camera_matrix),
        "dist_coeffs": rounded_list(dist_coeffs.reshape(-1)),
        "reprojection_error_mean_px": round(mean_error, 8),
        "used_images": used_images,
        "failed_images": failed_images,
        "ignored_images": ignored_images,
        "pose_convention": {
            "row_format": "x,y,z,rx,ry,rz",
            "translation_unit": "meter",
            "rotation_unit": "radian",
            "robot_pose": "T_base_end",
            "euler_order": "R = Rz @ Ry @ Rx",
        },
    }


def write_result(result: dict[str, Any], output_dir: Path, mode: str) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = output_dir / f"hand_eye_result_{mode}.yaml"
    json_path = output_dir / f"hand_eye_result_{mode}.json"
    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(result, f, allow_unicode=True, sort_keys=False)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return yaml_path, json_path


def run(args: argparse.Namespace) -> dict[str, Any]:
    data_dir = resolve_data_dir(args.data_dir)
    output_dir = data_dir if args.output_dir is None else resolve_path(args.output_dir)
    config_path = resolve_path(args.config)
    checkerboard = load_checkerboard_config(config_path)
    poses = load_poses(data_dir / POSE_FILE)
    images, ignored_images = collect_numbered_images(data_dir)

    camera_rms, camera_matrix, dist_coeffs, rvecs, tvecs, used_images, failed_images, image_size = calibrate_camera(
        images, poses, checkerboard
    )
    robot_rotations, robot_translations = robot_inputs_for_mode(args.mode, used_images, poses)
    target_rotations, target_translations = target_to_camera_inputs(rvecs, tvecs)
    rotation, translation = cv2.calibrateHandEye(
        robot_rotations,
        robot_translations,
        target_rotations,
        target_translations,
        method=METHODS[args.method],
    )
    hand_eye_matrix = matrix_from_rt(rotation, translation)

    result = build_result(
        args.mode,
        args.method,
        data_dir,
        config_path,
        checkerboard,
        image_size,
        camera_rms,
        camera_matrix,
        dist_coeffs,
        hand_eye_matrix,
        used_images,
        failed_images,
        ignored_images,
    )
    yaml_path, json_path = write_result(result, output_dir, args.mode)
    result["_output_yaml"] = str(yaml_path)
    result["_output_json"] = str(json_path)
    return result


def main() -> None:
    np.set_printoptions(precision=8, suppress=True)
    result = run(parse_args())
    print(f"mode: {result['mode']}")
    print(f"method: {result['method']}")
    print(f"transform: {result['transform_name']}")
    print("matrix_4x4:")
    print(np.asarray(result["matrix_4x4"], dtype=np.float64))
    print(f"translation_m: {result['translation_m']}")
    print(f"quaternion_xyzw: {result['quaternion_xyzw']}")
    print(f"camera_calibration_rms: {result['camera_calibration_rms']}")
    print(f"reprojection_error_mean_px: {result['reprojection_error_mean_px']}")
    print(f"used_images: {len(result['used_images'])}")
    print(f"failed_images: {len(result['failed_images'])}")
    print(f"ignored_images: {len(result['ignored_images'])}")
    print(f"yaml: {result['_output_yaml']}")
    print(f"json: {result['_output_json']}")


if __name__ == "__main__":
    main()
