"""Shared placement offsets and endpoint quality settings."""

from pathlib import Path

import numpy as np
import yaml

from .config import ROOT

DEFAULT_PLACEMENT = ROOT / "configs/placement/plug_placement.yaml"


def load_placement(path: Path = DEFAULT_PLACEMENT) -> dict:
    with path.open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict) or data.get("position_unit") != "meter":
        raise ValueError("Placement config position_unit must be meter")
    reference = data["reference"]
    for key in ("axis_offset_m", "surface_offset_m", "head_base_z_offset_m", "tail_base_z_offset_m"):
        if not np.isfinite(reference[key]):
            raise ValueError(f"Invalid placement offset: {key}")
    if reference["axis_offset_m"] < 0 or reference["surface_offset_m"] < 0:
        raise ValueError("Normal offsets must be nonnegative")
    if reference["head_base_z_offset_m"] <= 0 or reference["tail_base_z_offset_m"] >= 0:
        raise ValueError("Head offset must be +Z and tail offset must be -Z")
    endpoint = data["endpoint"]
    for key, value in endpoint.items():
        if not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"Endpoint setting must be finite and positive: {key}")
    for key in ("mask_erosion_px", "patch_radius_px", "min_valid_points"):
        if endpoint[key] != int(endpoint[key]):
            raise ValueError(f"Endpoint setting must be an integer: {key}")
    if not 0 < endpoint["min_inlier_ratio"] <= 1:
        raise ValueError("min_inlier_ratio must be in (0,1]")
    if not 0 < endpoint["max_axis_angle_deg"] < 90:
        raise ValueError("max_axis_angle_deg must be in (0,90)")
    if endpoint["min_depth_m"] >= endpoint["max_depth_m"]:
        raise ValueError("Invalid endpoint depth range")
    if endpoint["min_valid_points"] < 12:
        raise ValueError("At least 12 depth samples are required")
    return data


def preset_points(center: np.ndarray, normal: np.ndarray, config: dict) -> dict[str, np.ndarray]:
    center = np.asarray(center, float).reshape(3)
    normal = np.asarray(normal, float).reshape(3)
    if not np.isfinite(np.r_[center, normal]).all() or not np.isclose(np.linalg.norm(normal), 1):
        raise ValueError("Reference center must be finite and normal must be unit length")
    cfg = config["reference"]
    pre = center + (cfg["axis_offset_m"] + cfg["surface_offset_m"]) * normal
    return {"reference": center, "preset": pre,
            "head_preset": pre + [0, 0, cfg["head_base_z_offset_m"]],
            "tail_preset": pre + [0, 0, cfg["tail_base_z_offset_m"]]}
