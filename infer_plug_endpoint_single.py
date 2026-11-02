#!/usr/bin/env python3
"""Locate one visible plug head/tail side-surface edge point from RGB + D2RGB."""
from __future__ import annotations

import argparse
import base64
import html
from pathlib import Path
import sys

import cv2
import numpy as np

from infer_apriltag_cylinder import DEFAULT_HAND_EYE, show_visualization
from plug_vg.config import DEFAULT_CAMERA, ROOT, load_camera
from plug_vg.io import read_depth_raw, write_json
from plug_vg.placement import DEFAULT_PLACEMENT, load_placement
from plug_vg.plug_endpoint import choose_end_instance, fit_local_depth, locate_edge_point, projected_base_axis
from plug_vg.robot_transform import load_hand_eye_matrix, robot_pose_to_matrix, round_list

DEFAULT_WEIGHTS = ROOT / "ultralytics/runs/segment/plug_yolo26n_seg_20260904/weights/best.pt"

r'''
conda activate embodiedvg
python infer_plug_endpoint_single.py \
  --rgb test_gt_20260902/20260902_133202_557_color.png \
  --d2rgb test_gt_20260902/20260902_133202_557_d2rgb.npy \
  --robot-pose -0.014293 0.460711 0.742759 2.167158 0.044541 -3.126827 \
  --output-dir output/plug_endpoint
'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb", type=Path, required=True)
    parser.add_argument("--d2rgb", type=Path, required=True, help="RGB-aligned raw depth PNG/NPY; scale from camera config.")
    parser.add_argument("--robot-pose", type=float, nargs=6, required=True,
                        metavar=("X", "Y", "Z", "ROLL", "PITCH", "YAW"),
                        help="Capture-time flange pose, meters and fixed-axis XYZ RPY radians.")
    parser.add_argument("--seg-weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--camera-config", type=Path, default=DEFAULT_CAMERA)
    parser.add_argument("--hand-eye-config", type=Path, default=DEFAULT_HAND_EYE)
    parser.add_argument("--placement-config", type=Path, default=DEFAULT_PLACEMENT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output/plug_endpoint")
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None, help="Ultralytics device, e.g. cpu or 0.")
    parser.add_argument("--show", action="store_true", help="Open the combined RGB/3D HTML when a desktop is available.")
    return parser.parse_args()


def draw_endpoint_overlay(image: np.ndarray, instance: dict | None, location: dict | None,
                          fit: dict | None, result: dict) -> np.ndarray:
    overlay = image.copy()
    if instance is not None:
        mask = instance["mask"] > 0
        overlay[mask] = (overlay[mask] * 0.7 + np.array([230, 150, 30]) * 0.3).astype(np.uint8)
        contours, _ = cv2.findContours(instance["mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (255, 180, 0), 2)
    if location is not None:
        safe_contours, _ = cv2.findContours(location["safe_mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, safe_contours, -1, (0, 180, 0), 1)
        edge = tuple(np.rint(location["edge_pixel"]).astype(int))
        target = tuple(location["target_pixel"].astype(int))
        cv2.circle(overlay, edge, 6, (0, 200, 255), 2)
        cv2.arrowedLine(overlay, edge, target, (0, 200, 255), 2)
        cv2.drawMarker(overlay, target, (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
        cv2.putText(overlay, "Inset surface target", (target[0]+12, target[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    if fit is not None:
        for pixel, keep in zip(fit["sample_pixels"], fit["inlier_mask"]):
            cv2.circle(overlay, tuple(pixel.astype(int)), 1, (0, 255, 0) if keep else (255, 0, 255), -1)
        cv2.drawMarker(overlay, tuple(location["target_pixel"].astype(int)), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
    texts = [f"{result['status']} / {result.get('class_name', 'no class')} / side surface point, not axis center",
             "Cyan: mask; green outline: eroded mask; yellow: edge; red: inset target",
             "Green dots: fitted depth inliers; magenta dots: rejected samples"]
    if result["status"] == "ok":
        fit_rmse = fit["quality"].get("fit_rmse_m")
        rmse_text = f"{fit_rmse*1000:.2f} mm" if fit_rmse is not None else "not fitted (median fallback)"
        texts += [f"Camera XYZ (m): {result['point_camera_m']}", f"Base XYZ (m): {result['point_base_m']}",
                  f"Depth fit RMSE: {rmse_text}; accuracy goal unverified"]
    else:
        texts += [f"Failed: {result.get('reason', '')}"]
    height = len(texts) * 26 + 12
    cv2.rectangle(overlay, (0, 0), (min(image.shape[1]-1, 1280), height), (25, 25, 25), -1)
    for i, line in enumerate(texts):
        cv2.putText(overlay, line, (10, 24 + i*26), cv2.FONT_HERSHEY_SIMPLEX, 0.53, (255, 255, 255), 1)
    return overlay


def draw_depth_fit_overlay(depth_raw: np.ndarray, instance: dict | None, location: dict | None,
                           fit: dict | None, result: dict, camera: dict, cfg: dict) -> np.ndarray:
    """Render the local depth candidate area and the evidence used for its point estimate."""
    depth_m = np.asarray(depth_raw, dtype=np.float32) * float(camera["depth_scale"])
    valid = np.isfinite(depth_m) & (depth_m >= cfg["min_depth_m"]) & (depth_m <= cfg["max_depth_m"])
    depth_view = np.zeros((*depth_m.shape, 3), dtype=np.uint8)
    if np.any(valid):
        if location is not None:
            u, v = location["target_pixel"].astype(int)
            radius = int(cfg["patch_radius_px"])
            yy, xx = np.ogrid[:depth_m.shape[0], :depth_m.shape[1]]
            circle = (xx-u)**2 + (yy-v)**2 <= radius**2
            scale_region = valid & circle
        else:
            scale_region = valid
        values = depth_m[scale_region] if np.any(scale_region) else depth_m[valid]
        lo, hi = np.percentile(values, [2, 98])
        normalized = np.zeros(depth_m.shape, dtype=np.uint8)
        normalized[valid] = np.clip((depth_m[valid]-lo) / max(float(hi-lo), 1e-6) * 255, 0, 255).astype(np.uint8)
        depth_view = cv2.applyColorMap(255-normalized, cv2.COLORMAP_TURBO)
        depth_view[~valid] = (18, 18, 18)

    panel = depth_view
    status = result.get("status", "failed")
    radius = int(cfg["patch_radius_px"])
    target = None
    candidate_pixels = valid_count = 0
    coverage = None
    if location is not None:
        target = tuple(location["target_pixel"].astype(int))
        u, v = target
        y0, y1 = max(0, v-radius), min(depth_m.shape[0], v+radius+1)
        x0, x1 = max(0, u-radius), min(depth_m.shape[1], u+radius+1)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        circle_local = (xx-u)**2 + (yy-v)**2 <= radius**2
        mask_local = location["safe_mask"][y0:y1, x0:x1] > 0
        selected = circle_local & mask_local
        valid_local = selected & valid[y0:y1, x0:x1]
        candidate_pixels = int(selected.sum())
        valid_count = int(valid_local.sum())
        coverage = valid_count / max(candidate_pixels, 1)

        # Show the eroded plug mask so the fit candidate's support is legible.
        safe = location["safe_mask"] > 0
        tint = np.zeros_like(panel)
        tint[safe] = (35, 35, 35)
        panel = cv2.addWeighted(panel, 0.86, tint, 0.14, 0)

        if fit is not None:
            pixels = fit["sample_pixels"].astype(int)
            keep = np.asarray(fit["inlier_mask"], dtype=bool)
            median_fallback = fit["quality"].get("model") == "local_median"
            for pixel, accepted in zip(pixels, keep):
                color = (255, 210, 40) if median_fallback else ((50, 245, 70) if accepted else (220, 40, 220))
                cv2.circle(panel, tuple(pixel), 1, color, -1)
        else:
            # No fitted inlier set exists yet; mark valid observations neutrally.
            py, px = np.nonzero(valid_local)
            for x, y in zip(px[::3] + x0, py[::3] + y0):
                cv2.circle(panel, (int(x), int(y)), 1, (235, 235, 235), -1)

        cv2.circle(panel, target, radius, (0, 215, 255), 3, cv2.LINE_AA)
        cv2.drawMarker(panel, target, (255, 255, 255), cv2.MARKER_CROSS, 24, 3, cv2.LINE_AA)
        cv2.putText(panel, f"target ({u}, {v})  |  fit radius {radius}px", (u+radius+10, v-12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 215, 255), 2, cv2.LINE_AA)
        contours, _ = cv2.findContours(location["safe_mask"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(panel, contours, -1, (255, 255, 255), 1, cv2.LINE_AA)

    quality = result.get("quality", {})
    if coverage is None:
        coverage = quality.get("valid_coverage")
        candidate_pixels = int(quality.get("candidate_surface_pixels", 0))
        valid_count = int(quality.get("valid_samples", 0))
    median = quality.get("depth_median_m")
    if median is None and location is not None and valid_count:
        roi = depth_m[y0:y1, x0:x1]
        median = float(np.median(roi[valid_local]))
    model = quality.get("model", "not fitted")
    fit_rmse = quality.get("fit_rmse_m")
    validation_rmse = quality.get("validation_rmse_m")
    lines = [
        f"Status: {status}    Model: {model}",
        f"Valid depth: {valid_count}/{candidate_pixels} px    Coverage: {coverage*100:.1f}%" if coverage is not None else "Valid depth coverage: unavailable",
        f"Depth median: {median*1000:.1f} mm" if median is not None else "Depth median: unavailable",
        f"Fit RMSE: {fit_rmse*1000:.2f} mm" if fit_rmse is not None else "Fit RMSE: N/A",
        f"Holdout RMSE: {validation_rmse*1000:.2f} mm" if validation_rmse is not None else "Holdout RMSE: N/A",
        ("White: valid observations   Blue: median support   Yellow: candidate range" if model == "local_median"
         else "White: valid observations   Green: fit inliers   Magenta: rejected   Yellow: candidate range"),
    ]
    if status != "ok" and result.get("reason"):
        lines.append(f"Reason: {result['reason']}")
    header_h = 32 + 25 * len(lines)
    header = np.zeros((header_h, panel.shape[1], 3), dtype=np.uint8)
    for index, line in enumerate(lines):
        color = (70, 220, 255) if index == 0 and status != "ok" else (245, 245, 245)
        cv2.putText(header, line[:150], (12, 24 + index*25), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 1, cv2.LINE_AA)
    return np.vstack([header, panel])


def save_review(result: dict, fit: dict | None, t_base_camera: np.ndarray) -> None:
    import plotly.graph_objects as go

    fig = go.Figure()
    if fit is not None:
        camera_samples = fit["sample_points_camera_m"]
        samples = camera_samples @ t_base_camera[:3, :3].T + t_base_camera[:3, 3]
        keep = fit["inlier_mask"]
        for accepted, color, label in [(True, "green", "Local depth inliers"), (False, "magenta", "Rejected depth")]:
            points = samples[keep == accepted]
            if len(points):
                fig.add_trace(go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2], mode="markers",
                                          name=label, marker=dict(size=3, color=color)))
        point = np.asarray(result["point_base_m"])
        fig.add_trace(go.Scatter3d(x=[point[0]], y=[point[1]], z=[point[2]], mode="markers+text",
                                  name="Inset side-surface point", text=[result["class_name"]],
                                  marker=dict(size=9, color="red")))
        # Keep depth outliers from making the local point invisible.
        bounds = np.vstack([samples[keep], point])
        middle = (bounds.min(axis=0) + bounds.max(axis=0))/2
        half = max(float(np.ptp(bounds, axis=0).max()) * 0.6, 0.008)
        fig.update_layout(scene=dict(aspectmode="cube", **{
            f"{axis}axis": dict(title=f"Base {axis.upper()} (m)", range=[middle[i]-half, middle[i]+half])
            for i, axis in enumerate("xyz")}))
    fig.update_layout(title="Plug endpoint local depth review / " + result["status"],
                      margin=dict(l=0, r=0, b=0, t=55))
    rgb = base64.b64encode(Path(result["artifacts"]["rgb_overlay"]).read_bytes()).decode("ascii")
    image_section = ('<h2>Segmentation, edge, inward target and depth support</h2>'
                     f'<img src="data:image/jpeg;base64,{rgb}" style="max-width:100%">')
    depth_path = result.get("artifacts", {}).get("depth_fit_overlay")
    if depth_path and Path(depth_path).is_file():
        depth_image = base64.b64encode(Path(depth_path).read_bytes()).decode("ascii")
        image_section += ('<h2>Depth fit range and quality</h2>'
                          f'<img src="data:image/jpeg;base64,{depth_image}" style="max-width:100%">')
    if "localization" in result:
        image = cv2.imread(result["artifacts"]["rgb_overlay"])
        u, v = map(int, result["localization"]["target_pixel"])
        crop = image[max(0, v-65):min(image.shape[0], v+66), max(0, u-100):min(image.shape[1], u+101)]
        success, encoded = cv2.imencode(".jpg", crop)
        if success:
            zoom = base64.b64encode(encoded).decode("ascii")
            image_section += ('<h2>Local edge and fitted depth samples</h2>'
                              f'<img src="data:image/jpeg;base64,{zoom}" style="width:700px;max-width:100%;image-rendering:pixelated">')
    if result["status"] != "ok":
        image_section += f'<p>Localization failed: {html.escape(result["reason"])}</p>'
    page = fig.to_html(include_plotlyjs=True, full_html=True)
    page = page.replace("<head>", '<head><meta charset="utf-8"><title>Plug endpoint review</title>', 1)
    page = page.replace("<body>", "<body>" + image_section, 1)
    Path(result["artifacts"]["review_html"]).write_text(page, encoding="utf-8")


def run(args: argparse.Namespace) -> tuple[dict, Path]:
    if not np.isfinite(args.robot_pose).all() or not 0 < args.conf <= 1 or args.imgsz <= 0:
        raise ValueError("Invalid robot pose or inference settings")
    camera = load_camera(args.camera_config)
    cfg = load_placement(args.placement_config)["endpoint"]
    image = cv2.imread(str(args.rgb))
    if image is None:
        raise ValueError(f"Cannot read RGB: {args.rgb}")
    expected = (camera["image_height"], camera["image_width"])
    if image.shape[:2] != expected:
        raise ValueError("RGB resolution differs from camera intrinsics")
    depth = read_depth_raw(args.d2rgb)
    if depth is None or depth.ndim != 2 or depth.shape != image.shape[:2]:
        raise ValueError("D2RGB must be a readable 2D raw depth array matching RGB resolution")
    if not np.isfinite([camera[k] for k in ("fx", "fy", "cx", "cy", "depth_scale")]).all() or min(camera["fx"],camera["fy"],camera["depth_scale"]) <= 0:
        raise ValueError("Invalid camera intrinsics or depth scale")
    t_base_end = robot_pose_to_matrix(args.robot_pose)
    t_end_camera = load_hand_eye_matrix(args.hand_eye_config)
    t_base_camera = t_base_end @ t_end_camera
    sys.path.insert(0, str(ROOT / "ultralytics"))
    from ultralytics import YOLO

    model = YOLO(str(args.seg_weights))
    prediction = model.predict(image, conf=args.conf, imgsz=args.imgsz, device=args.device,
                               retina_masks=True, verbose=False)[0]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_dir / args.rgb.stem
    path = Path(str(prefix) + "_plug_endpoint.json")
    result = {"status": "failed", "valid": False, "position_unit": "meter",
              "point_definition": "camera-visible side-surface end-edge midpoint, moved inward in image",
              "axis_prior": "tail to head approximately base +Z; fixed grasp orientation",
              "input": {"rgb": str(args.rgb), "d2rgb": str(args.d2rgb), "seg_weights": str(args.seg_weights),
                        "robot_pose_xyzrpy_m_rad": list(args.robot_pose), "pose_frame": "flange",
                        "camera_config": str(args.camera_config), "hand_eye_config": str(args.hand_eye_config),
                        "placement_config": str(args.placement_config)},
              "endpoint_settings": cfg,
              "frame_transform": {"T_base_end": round_list(t_base_end), "T_end_camera": round_list(t_end_camera),
                                  "T_base_camera": round_list(t_base_camera)},
              "artifacts": {"rgb_overlay": str(prefix)+"_endpoint_overlay.jpg",
                            "depth_fit_overlay": str(prefix)+"_depth_fit_overlay.jpg",
                            "review_html": str(prefix)+"_endpoint_review.html"}}
    instance = location = fit = None
    try:
        instance = choose_end_instance(prediction)
        result.update({key: value for key, value in instance.items() if key != "mask"})
        direction = projected_base_axis(instance["mask"], depth, camera, t_base_camera, cfg)
        location = locate_edge_point(instance["mask"], direction, instance["class_name"], cfg)
        result["localization"] = {key: round_list(value) if isinstance(value, np.ndarray) else value
                                  for key, value in location.items() if key != "safe_mask"}
        fit = fit_local_depth(depth, location, camera, cfg)
        point = fit["point_camera_m"]
        base_point = t_base_camera[:3, :3] @ point + t_base_camera[:3, 3]
        result.update(status="ok", valid=True, point_camera_m=round_list(point), point_base_m=round_list(base_point),
                      quality=fit["quality"])
        result["quality"]["localization_method"] = "segmentation_terminal_midpoint_with_base_Z_prior"
        result["quality"]["semantic_endpoint_verified"] = False
    except ValueError as exc:
        result["reason"] = str(exc)
        result["status"] = "invalid" if location is not None else "failed"
    overlay = draw_endpoint_overlay(image, instance, location, fit, result)
    if not cv2.imwrite(result["artifacts"]["rgb_overlay"], overlay):
        raise OSError("Could not save endpoint RGB overlay")
    depth_overlay = draw_depth_fit_overlay(depth, instance, location, fit, result, camera, cfg)
    if not cv2.imwrite(result["artifacts"]["depth_fit_overlay"], depth_overlay):
        raise OSError("Could not save endpoint depth fit overlay")
    save_review(result, fit, t_base_camera)
    write_json(path, result)
    if args.show:
        show_visualization(Path(result["artifacts"]["review_html"]))
    return result, path


def main() -> int:
    args = parse_args()
    path = args.output_dir / f"{args.rgb.stem}_plug_endpoint.json"
    try:
        result, path = run(args)
    except (ValueError, OSError, KeyError, ImportError, RuntimeError) as exc:
        result = {"status": "failed", "valid": False, "reason": str(exc)}
        write_json(path, result)
    print("# 定位状态：ok 表示通过当前定位检查；failed 表示定位失败")
    print(f"status: {result['status']}")
    if result["status"] != "ok":
        print(f"failed: {result['reason']}")
    else:
        print("\n# 端部类别：plug_head 为头部，plug_tail 为尾部")
        print(f"class: {result['class_name']}")
        print("\n# 定位点：机器人基坐标系下的 XYZ 坐标，单位为米")
        print(f"Base point XYZ (m): {result['point_base_m']}")
    print("\n# 输出文件：定位数据、图像叠加结果和交互式复核视图")
    print(f"JSON: {path}")
    for name, artifact in result.get("artifacts", {}).items():
        print(f"{name}: {artifact}")
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
