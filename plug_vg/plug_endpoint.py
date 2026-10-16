"""Single-frame visible end-edge midpoint and mask-contained local depth fitting.

The projected +base-Z prior selects head/tail direction. This is a visible side
surface point, not an end-face center or the plug's axis center.
"""
from __future__ import annotations

import cv2
import numpy as np

from .visible_points import erode_mask


def choose_end_instance(result: object) -> dict:
    """Keep model class identity; reject ambiguous single-plug observations."""
    if result.masks is None or not len(result.boxes):
        raise ValueError("No plug_head or plug_tail segmentation detected")
    matches = []
    for i, box in enumerate(result.boxes):
        class_id = int(box.cls.item())
        name = result.names[class_id]
        if name in ("plug_head", "plug_tail"):
            matches.append((i, class_id, name, float(box.conf.item())))
    if len(matches) != 1:
        raise ValueError(f"Expected one visible head/tail instance, detected {len(matches)}")
    i, class_id, name, confidence = matches[0]
    mask = result.masks.data[i].detach().cpu().numpy() > 0.5
    if mask.shape != tuple(result.orig_shape):
        raise ValueError("Segmentation mask resolution mismatch; enable retina_masks")
    # Tiny disconnected mask islands must not move the endpoint.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if count < 2:
        raise ValueError("Empty segmentation mask")
    label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    largest = labels == label
    if largest.sum() < 0.9 * mask.sum():
        raise ValueError("Segmentation has substantial disconnected components")
    return {"class_id": class_id, "class_name": name, "confidence": confidence,
            "mask": largest.astype(np.uint8), "removed_island_pixels": int(mask.sum() - largest.sum())}


def projected_base_axis(mask: np.ndarray, depth_raw: np.ndarray, camera: dict,
                        t_base_camera: np.ndarray, cfg: dict) -> np.ndarray:
    depth = np.asarray(depth_raw, float) * camera["depth_scale"]
    valid = (mask > 0) & np.isfinite(depth) & (depth >= cfg["min_depth_m"]) & (depth <= cfg["max_depth_m"])
    if valid.sum() < cfg["min_valid_points"]:
        raise ValueError("Not enough valid object depth to project the base +Z axis")
    ys, xs = np.nonzero(mask)
    uv = np.array([np.median(xs), np.median(ys)])
    z = float(np.median(depth[valid]))
    point = np.array([(uv[0] - camera["cx"]) * z / camera["fx"],
                      (uv[1] - camera["cy"]) * z / camera["fy"], z])
    axis_camera = np.asarray(t_base_camera)[:3, :3].T @ [0, 0, 1]
    tip = point + 0.01 * axis_camera
    if tip[2] <= 0:
        raise ValueError("Base-axis projection behind camera")
    projected = tip[:2] / tip[2] * [camera["fx"], camera["fy"]] + [camera["cx"], camera["cy"]]
    direction = projected - uv
    if np.linalg.norm(direction) < 2:
        raise ValueError("Plug axis nearly points at camera; side edge is not reliably observable")
    return direction / np.linalg.norm(direction)


def locate_edge_point(mask: np.ndarray, base_z_image: np.ndarray, class_name: str, cfg: dict) -> dict:
    if class_name not in ("plug_head", "plug_tail"):
        raise ValueError("Unsupported plug class")
    direction = np.asarray(base_z_image, float)
    if direction.shape != (2,) or not np.isfinite(direction).all() or np.linalg.norm(direction) < 1e-8:
        raise ValueError("Invalid projected plug axis")
    direction = direction / np.linalg.norm(direction) * (1 if class_name == "plug_head" else -1)
    mask = (np.asarray(mask) > 0).astype(np.uint8)
    ys, xs = np.nonzero(mask)
    xy = np.column_stack([xs, ys]).astype(float)
    if len(xy) < 100:
        raise ValueError("Plug mask too small for end-edge localization")
    if xs.min() == 0 or ys.min() == 0 or xs.max() == mask.shape[1]-1 or ys.max() == mask.shape[0]-1:
        raise ValueError("Plug silhouette clipped by image boundary")
    covariance = np.cov(xy.T)
    values, axes = np.linalg.eigh(covariance)
    alignment = abs(float(axes[:, -1] @ direction))
    # A short tail or a large ring can make mask PCA transverse to the true axis.
    # Only use PCA as a veto when the silhouette has a strong longitudinal axis.
    elongated = values[-1] >= 2.5 * values[0]
    if elongated and alignment < np.cos(np.deg2rad(cfg["max_axis_angle_deg"])):
        raise ValueError("Plug silhouette inconsistent with the base +Z long-axis prior")
    transverse = np.array([-direction[1], direction[0]])
    axial, lateral = xy @ direction, xy @ transverse
    low, high = np.quantile(axial, [0.005, 0.995])
    length = high - low
    if length < 20:
        raise ValueError("End region has insufficient pixel extent")
    # Include enough of the terminal section to see both sides of a slanted rim.
    # A tiny band at the extreme only sees one corner and biases the midpoint.
    band = (axial >= high - max(4, length * 0.25)) & (axial <= high)
    qlow, qhigh = np.quantile(lateral[band], [0.02, 0.98])
    width = qhigh - qlow
    if width < 6:
        raise ValueError("End edge too narrow")
    mid = (qlow + qhigh) / 2
    # Intersect the transverse midpoint line with the ORIGINAL outer contour.
    # Quantiles of interior pixels stop short of the true segmentation boundary.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    crossings = []
    for contour in contours:
        vertices = contour[:, 0, :].astype(float)
        a, b = vertices, np.roll(vertices, -1, axis=0)
        la, lb = a @ transverse, b @ transverse
        crossing = ((la <= mid) & (lb > mid)) | ((lb <= mid) & (la > mid))
        fraction = (mid - la[crossing]) / (lb[crossing] - la[crossing])
        points = a[crossing] + fraction[:, None] * (b[crossing] - a[crossing])
        crossings.extend(points @ direction)
    if not crossings:
        raise ValueError("No end contour intersection at transverse midpoint")
    tip_axial = float(max(crossings))
    edge = direction * tip_axial + transverse * mid
    safe = erode_mask(mask, int(cfg["mask_erosion_px"]))
    return {"edge_pixel": edge, "target_pixel": edge.copy(), "direction_image": direction,
            "inset_actual_px": 0.0, "terminal_width_px": float(width),
            "terminal_axial_range": [float(high - max(4, length * 0.25)), float(high)],
            "terminal_lateral_range": [float(qlow), float(qhigh)],
            "midpoint_lateral": float(mid),
            "axis_check": "elongated_silhouette_checked" if elongated else "base_Z_prior_only_short_or_wide_mask",
            "safe_mask": safe}


def _design(xy: np.ndarray, degree: int) -> np.ndarray:
    x, y = xy.T
    linear = np.column_stack([np.ones(len(x)), x, y])
    return linear if degree == 1 else np.column_stack([linear, x*x, x*y, y*y])


def _robust_model(xy: np.ndarray, depth: np.ndarray, degree: int, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """RANSAC inverse-depth surface; plane is degree 1, local curvature degree 2."""
    a = _design(xy, degree)
    size = a.shape[1]
    rng = np.random.default_rng(7 + degree)
    best = None
    best_score = (-1, -np.inf)
    for _ in range(160):
        indices = rng.choice(len(depth), size, replace=False)
        sample = a[indices]
        if np.linalg.cond(sample) > 1e5:
            continue
        beta = np.linalg.lstsq(sample, 1 / depth[indices], rcond=None)[0]
        inverse = a @ beta
        predicted = np.divide(1, inverse, out=np.full_like(inverse, np.inf), where=inverse > 0)
        residual = np.abs(predicted - depth)
        keep = residual <= cfg["ransac_threshold_m"]
        score = (int(keep.sum()), -float(np.median(residual[keep])) if keep.any() else -np.inf)
        if score > best_score:
            best, best_score = keep, score
    if best is None or best.sum() < cfg["min_valid_points"] or best.mean() < cfg["min_inlier_ratio"]:
        raise ValueError("Insufficient consistent depth for local surface fit")
    for _ in range(5):
        if np.linalg.cond(a[best]) > 1e5:
            raise ValueError("Local fit has degenerate spatial support")
        beta = np.linalg.lstsq(a[best], 1 / depth[best], rcond=None)[0]
        inverse = a @ beta
        predicted = np.divide(1, inverse, out=np.full_like(inverse, np.inf), where=inverse > 0)
        best = np.abs(predicted - depth) <= cfg["ransac_threshold_m"]
        if best.sum() < cfg["min_valid_points"] or best.mean() < cfg["min_inlier_ratio"]:
            raise ValueError("Local fit lost depth support")
    beta = np.linalg.lstsq(a[best], 1 / depth[best], rcond=None)[0]
    return beta, best


def fit_local_depth(depth_raw: np.ndarray, location: dict, camera: dict, cfg: dict) -> dict:
    if depth_raw.ndim != 2 or depth_raw.shape != location["safe_mask"].shape:
        raise ValueError("D2RGB must be a 2D depth array at the RGB resolution")
    depth = depth_raw.astype(float) * camera["depth_scale"]
    radius = int(cfg["patch_radius_px"])
    target = location["target_pixel"]
    u, v = target
    x0, x1 = max(0, int(np.floor(u-radius))), min(depth.shape[1], int(np.ceil(u+radius))+1)
    y0, y1 = max(0, int(np.floor(v-radius))), min(depth.shape[0], int(np.ceil(v+radius))+1)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    selected = ((xx-u)**2 + (yy-v)**2 <= radius**2) & (location["safe_mask"][y0:y1, x0:x1] > 0)
    z = depth[y0:y1, x0:x1]
    selected &= np.isfinite(z) & (z >= cfg["min_depth_m"]) & (z <= cfg["max_depth_m"])
    pixels = np.column_stack([xx[selected], yy[selected]])
    z = z[selected]
    if len(z) < cfg["min_valid_points"] * 1.3:
        raise ValueError("Insufficient mask-contained local depth samples")
    xy = (pixels - target) / radius
    # Interleaved spatial holdout: validation samples are never used to fit the candidate.
    validation = ((pixels[:, 0] + 2*pixels[:, 1]) % 5) == 0
    if validation.sum() < 6 or (~validation).sum() < cfg["min_valid_points"]:
        raise ValueError("Insufficient local depth for independent fit validation")
    candidates = []
    for degree in (1, 2):
        try:
            beta, _ = _robust_model(xy[~validation], z[~validation], degree, cfg)
            inverse = _design(xy[validation], degree) @ beta
            if np.any(inverse <= 0):
                continue
            residual = 1/inverse - z[validation]
            hold_keep = np.abs(residual) <= cfg["ransac_threshold_m"]
            if hold_keep.mean() < cfg["min_inlier_ratio"] or hold_keep.sum() < 6:
                continue
            score = float(np.sqrt(np.mean(residual[hold_keep]**2)))
            candidates.append((degree, score))
        except ValueError:
            continue
    if not candidates:
        raise ValueError("Local depth models failed spatial holdout validation")
    degree, score = candidates[0]
    # Prefer a plane unless curvature materially improves held-out depth prediction.
    for other_degree, other_score in candidates[1:]:
        if other_score < score * 0.75:
            degree, score = other_degree, other_score
    if score > cfg["max_validation_rmse_m"]:
        raise ValueError("Local depth validation RMSE exceeds limit")
    beta, keep = _robust_model(xy, z, degree, cfg)
    hull = cv2.convexHull(pixels[keep].astype(np.float32))
    signed_distance = cv2.pointPolygonTest(hull, tuple(map(float, target)), True)
    extrapolation = max(0.0, -signed_distance)
    if extrapolation > cfg["max_depth_extrapolation_px"]:
        raise ValueError("Edge depth extrapolation exceeds maximum supported distance")
    direction = location["direction_image"]
    transverse = np.array([-direction[1], direction[0]])
    local_pixels = pixels[keep] - target
    if np.ptp(local_pixels @ transverse) < 6 or np.ptp(local_pixels @ direction) < 4:
        raise ValueError("Insufficient spatial depth support at edge")
    prediction = 1 / (_design(xy[keep], degree) @ beta)
    rmse = float(np.sqrt(np.mean((prediction-z[keep])**2)))
    if rmse > cfg["max_fit_rmse_m"]:
        raise ValueError("Local fit RMSE exceeds limit")
    fitted_z = float(1 / beta[0])
    if not np.isfinite(fitted_z) or not cfg["min_depth_m"] <= fitted_z <= cfg["max_depth_m"]:
        raise ValueError("Fitted target depth outside valid range")
    xyz = np.array([(u-camera["cx"])*fitted_z/camera["fx"],
                    (v-camera["cy"])*fitted_z/camera["fy"], fitted_z])
    sample_points = np.column_stack([(pixels[:, 0]-camera["cx"])*z/camera["fx"],
                                    (pixels[:, 1]-camera["cy"])*z/camera["fy"], z])
    return {"point_camera_m": xyz, "sample_pixels": pixels, "inlier_mask": keep,
            "sample_depth_m": z, "sample_predicted_depth_m": 1 / (_design(xy, degree) @ beta),
            "inlier_hull": hull,
            "sample_points_camera_m": sample_points,
            "quality": {"model": "inverse_depth_plane" if degree == 1 else "inverse_depth_quadratic",
                        "valid_samples": int(len(z)), "inlier_samples": int(keep.sum()),
                        "inlier_ratio": float(keep.mean()), "fit_rmse_m": rmse,
                        "validation_rmse_m": score, "target_inside_inlier_hull": signed_distance >= 0,
                        "depth_extrapolation_px": extrapolation,
                        "max_depth_extrapolation_px": cfg["max_depth_extrapolation_px"],
                        "candidate_validation_rmse_m": {str(d): s for d, s in candidates},
                        "patch_radius_px": radius, "coefficients": beta.tolist(),
                        "accuracy_goal_m": cfg["accuracy_goal_m"], "physical_accuracy_verified": False}}
