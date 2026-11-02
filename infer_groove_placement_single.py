"""Check one plug's groove placement from RGB-D and a saved AprilTag reference."""

from __future__ import annotations

import argparse
import base64
import html
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from infer_apriltag_cylinder import DEFAULT_HAND_EYE, show_visualization
from plug_vg.config import DEFAULT_CAMERA, ROOT, load_camera
from plug_vg.geometry import project_point_float
from plug_vg.grasp_model import (
    DEFAULT_GRASP_MODEL_CONFIG,
    load_grasp_model,
    semantic_points_camera,
    transform_points,
)
from plug_vg.groove_placement import (
    evaluate_groove_placement,
    groove_axis_segment,
    validate_groove_check,
)
from plug_vg.io import read_depth_raw, read_json, write_json
from plug_vg.placement import DEFAULT_PLACEMENT, load_placement
from plug_vg.registration_review import (
    write_interactive_review_html,
    write_registration_comparison_ply,
)
from plug_vg.robot_transform import (
    load_hand_eye_matrix,
    robot_pose_to_matrix,
    round_list,
)
from plug_vg.symmetric_registration import register_symmetric_points
from plug_vg.visible_points import (
    extract_visible_points_from_mask,
    save_visible_points_ply,
    synthesize_scene_point_cloud,
)

DEFAULT_WEIGHTS = (
    ROOT / "ultralytics/runs/segment/plug_yolo26n_seg_20261019/weights/best.pt"
)
LABELS = {"seated": "正确入槽", "not_seated": "未正确入槽", "unknown": "无法判断"}


r"""
conda activate embodiedvg

python infer_groove_placement_single.py \
  --rgb test_data/插头放置判断数据/true/20261023_154709_074_color.png \
  --d2rgb test_data/插头放置判断数据/true/20261023_154709_074_d2rgb.npy \
  --apriltag-json output/apriltag_pose/20261023_163823_957_color_apriltag_pose.json \
  --robot-pose -0.338532 -0.045871 0.243139 1.551650 -0.030055 -3.123110 \
  --output-dir output/groove_placement \
  --show

"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb", type=Path, required=True)
    parser.add_argument(
        "--d2rgb", type=Path, required=True, help="RGB-aligned raw depth PNG/NPY."
    )
    parser.add_argument(
        "--apriltag-json",
        type=Path,
        required=True,
        help="Saved successful AprilTag result in base frame.",
    )
    parser.add_argument(
        "--robot-pose",
        nargs=6,
        type=float,
        required=True,
        metavar=("X", "Y", "Z", "ROLL", "PITCH", "YAW"),
        help="Capture flange XYZ/RPY (m/rad); must be provided for this image.",
    )
    parser.add_argument("--seg-weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_GRASP_MODEL_CONFIG)
    parser.add_argument("--camera-config", type=Path, default=DEFAULT_CAMERA)
    parser.add_argument("--hand-eye-config", type=Path, default=DEFAULT_HAND_EYE)
    parser.add_argument("--placement-config", type=Path, default=DEFAULT_PLACEMENT)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "output/groove_placement"
    )
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open the offline RGB/3D review in a browser.",
    )
    return parser.parse_args(argv)


def save_registration_review(
    result: dict,
    image: np.ndarray,
    depth: np.ndarray,
    camera: dict,
    visible_points: np.ndarray,
    model_points: np.ndarray,
    transform: np.ndarray,
    semantic: dict,
) -> None:
    """Save the same camera-frame PLY/HTML registration artifacts as 6D inference."""
    prefix = Path(result["artifacts"]["review_html"]).parent / Path(result["input"]["rgb"]).stem
    comparison_path = Path(str(prefix) + "_point_cloud_comparison.ply")
    write_registration_comparison_ply(
        comparison_path, visible_points, model_points, transform, semantic
    )
    result["artifacts"]["point_cloud_comparison_ply"] = str(comparison_path)
    scene_points, scene_colors = synthesize_scene_point_cloud(image, depth, camera, 0.1, 1.5)
    review_path = Path(str(prefix) + "_registration_review.html")
    write_interactive_review_html(
        review_path, scene_points, scene_colors, visible_points, model_points,
        transform, semantic, "ok", None, result["registration_quality"], 0, camera,
    )
    result["artifacts"]["registration_review_html"] = str(review_path)


def save_review(
    result: dict,
    image: np.ndarray | None,
    camera: dict,
    t_base_camera: np.ndarray | None,
    mask: np.ndarray | None,
    points_base: np.ndarray | None,
    cad_base: np.ndarray | None,
) -> None:
    """Always save an offline review; add image and base-frame geometry when available."""
    import plotly.graph_objects as go

    figure = go.Figure()
    overlay = image.copy() if image is not None else None
    if overlay is not None and mask is not None:
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay, contours, -1, (255, 255, 0), 2)
    for points, name, color in [
        (points_base, "Observed depth", "green"),
        (cad_base, "Registered CAD", "red"),
    ]:
        if points is not None and len(points):
            subset = points[:: max(1, len(points) // 5000)]
            figure.add_trace(
                go.Scatter3d(
                    x=subset[:, 0],
                    y=subset[:, 1],
                    z=subset[:, 2],
                    mode="markers",
                    name=name,
                    marker={"size": 2, "color": color},
                )
            )
    segments = []
    if "allowed_frustum" in result:
        envelope = result["allowed_frustum"]
        centers = np.array([envelope["bottom_center_base_m"], envelope["top_center_base_m"]])
        direction = centers[1] - centers[0]
        direction /= np.linalg.norm(direction)
        helper = np.eye(3)[int(np.argmin(np.abs(direction)))]
        side = np.cross(direction, helper)
        side /= np.linalg.norm(side)
        second = np.cross(direction, side)
        theta = np.linspace(0, 2 * np.pi, 65)
        circle = np.cos(theta)[:, None] * side + np.sin(theta)[:, None] * second
        rings = np.array([
            centers[0] + envelope["bottom_radius_m"] * circle,
            centers[1] + envelope["top_radius_m"] * circle,
        ])
        # Close the two disks as well as the side surface.
        vertices = np.vstack([rings.reshape(-1, 3), centers])
        faces = []
        count = len(theta)
        for i in range(count - 1):
            faces.extend([(i, i + 1, count + i), (i + 1, count + i + 1, count + i),
                          (2 * count, i + 1, i), (2 * count + 1, count + i, count + i + 1)])
        faces = np.asarray(faces)
        figure.add_trace(go.Mesh3d(
            x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            color="royalblue", opacity=0.16, name="Allowed frustum", showlegend=True,
        ))
        # A wireframe projects the actual allowance onto the RGB review too.
        for index, ring in enumerate(rings):
            segments.append((ring, "Top rim" if index else "Bottom rim", "royalblue", (255, 100, 0)))
        for index in range(0, count - 1, 16):
            segments.append((rings[:, index], "Frustum side", "royalblue", (255, 100, 0)))
    if "allowed_axis_segment" in result:
        segment = result["allowed_axis_segment"]
        segments.append(
            (
                np.array([segment["tail_limit_base_m"], segment["head_limit_base_m"]]),
                "Allowed axis",
                "royalblue",
                (255, 100, 0),
            )
        )
    if "predicted_axis_base" in result:
        axis = result["predicted_axis_base"]
        segments.append(
            (
                np.array([axis["tail_center_base_m"], axis["head_center_base_m"]]),
                "Predicted axis",
                "orange",
                (0, 140, 255),
            )
        )
        if "placement_check" in result:
            endpoints = np.array([axis["tail_center_base_m"], axis["head_center_base_m"]])
            intervals = result["placement_check"]["inside_intervals_t"]
            breaks = sorted(set([0.0, 1.0, *[t for interval in intervals for t in interval]]))
            for start, end in zip(breaks, breaks[1:]):
                inside = any(a <= (start + end) / 2 <= b for a, b in intervals)
                line = endpoints[0] + np.array([start, end])[:, None] * (endpoints[1] - endpoints[0])
                segments.append((line, "Axis inside" if inside else "Axis outside",
                                 "limegreen" if inside else "crimson",
                                 (0, 210, 0) if inside else (0, 0, 220)))
    for line, name, color, bgr in segments:
        figure.add_trace(
            go.Scatter3d(
                x=line[:, 0],
                y=line[:, 1],
                z=line[:, 2],
                mode="lines+markers",
                name=name,
                line={"color": color, "width": 7},
            )
        )
        if overlay is not None and t_base_camera is not None:
            line_camera = transform_points(line, np.linalg.inv(t_base_camera))
            pixels = [project_point_float(point, camera) for point in line_camera]
            if all(pixel is not None for pixel in pixels):
                uv = np.clip(np.rint(np.asarray(pixels)), -1e6, 1e6).astype(int)
                for a, b in zip(uv, uv[1:]):
                    visible, start, end = cv2.clipLine(
                        (0, 0, overlay.shape[1], overlay.shape[0]), tuple(a), tuple(b)
                    )
                    if visible:
                        cv2.line(overlay, start, end, bgr, 2, cv2.LINE_AA)
    title = result["status"]
    if "placement_check" in result:
        check = result["placement_check"]
        title += f" | axis inside {check['inside_fraction']:.2%} / required {check['min_inside_fraction']:.0%}"
    image_html = ""
    if overlay is not None:
        cv2.putText(
            overlay,
            title,
            (24, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (0, 200, 255),
            2,
            cv2.LINE_AA,
        )
        overlay_path = Path(result["artifacts"]["review_html"]).with_name(
            Path(result["input"]["rgb"]).stem + "_groove_overlay.jpg"
        )
        if not cv2.imwrite(str(overlay_path), overlay):
            raise OSError("Could not save groove overlay")
        result["artifacts"]["rgb_overlay"] = str(overlay_path)
        encoded = base64.b64encode(overlay_path.read_bytes()).decode("ascii")
        image_html = f'<img style="width:100%;max-width:1280px" src="data:image/jpeg;base64,{encoded}">'
    figure.update_layout(
        title="Base frame (meters): blue=allowed frustum, green=axis inside, red=axis outside",
        scene={
            "aspectmode": "data",
            "xaxis_title": "Base X (m)",
            "yaxis_title": "Base Y (m)",
            "zaxis_title": "Base Z (m)",
        },
    )
    registration_links = " ".join(
        f'<a href="{html.escape(Path(result["artifacts"][key]).name, quote=True)}">{label}</a>'
        for key, label in (
            ("registration_review_html", "点云配准交互复核（相机系）"),
            ("point_cloud_comparison_ply", "配准对比点云 PLY（相机系）"),
        )
        if key in result["artifacts"]
    )
    document = (
        f'<!doctype html><html><head><meta charset="utf-8"><title>Groove placement</title></head>'
        f"<body><h2>{html.escape(LABELS[result['status']])}: {html.escape(title)}</h2>"
        f"<p>{html.escape(result.get('reason', ''))}</p>"
        f"<p>{registration_links}</p>{image_html}"
        "<p>Single observation; registration accuracy and physical contact are not verified.</p>"
        f"{figure.to_html(full_html=False, include_plotlyjs=True)}</body></html>"
    )
    Path(result["artifacts"]["review_html"]).write_text(document, encoding="utf-8")


def run(args: argparse.Namespace) -> tuple[dict, Path]:
    start = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_dir / args.rgb.stem
    path = Path(str(prefix) + "_groove_placement.json")
    result: dict[str, Any] = {
        "status": "unknown",
        "is_seated": None,
        "position_unit": "meter",
        "input": {
            key: str(getattr(args, key))
            for key in (
                "rgb",
                "d2rgb",
                "apriltag_json",
                "seg_weights",
                "model_config",
                "camera_config",
                "hand_eye_config",
                "placement_config",
            )
        },
        "registration_stability_checked": False,
        "physical_accuracy_verified": False,
        "artifacts": {"review_html": str(prefix) + "_groove_review.html"},
        "timing": {},
    }
    image: np.ndarray | None = None
    mask: np.ndarray | None = None
    points_base: np.ndarray | None = None
    cad_base: np.ndarray | None = None
    t_base_camera: np.ndarray | None = None
    camera: dict = {}
    try:
        camera = load_camera(args.camera_config)
        if (
            not np.isfinite(
                [camera[k] for k in ("fx", "fy", "cx", "cy", "depth_scale")]
            ).all()
            or min(camera["fx"], camera["fy"], camera["depth_scale"]) <= 0
        ):
            raise ValueError("Invalid camera intrinsics or depth scale")
        cfg = load_placement(args.placement_config)
        check_config = validate_groove_check(cfg["groove_check"])
        reference = read_json(args.apriltag_json)
        if (
            not isinstance(reference, dict)
            or reference.get("status") != "ok"
            or reference.get("position_unit") != "meter"
        ):
            raise ValueError(
                "AprilTag reference must have status=ok and position_unit=meter"
            )
        segment = groove_axis_segment(
            reference["tag_center_base_m"], reference["tag_normal_base"], cfg
        )
        result["allowed_axis_segment"] = segment
        result["allowed_frustum"] = {
            **check_config,
            "bottom_center_base_m": segment["tail_limit_base_m"],
            "top_center_base_m": segment["head_limit_base_m"],
        }
        result["reference_base"] = {
            "point_m": reference["tag_center_base_m"],
            "normal": reference["tag_normal_base"],
            "offsets": cfg["reference"],
            "groove_check": check_config,
        }
        pose = args.robot_pose
        if not np.isfinite(pose).all():
            raise ValueError("Capture flange pose must be finite")
        t_base_end = robot_pose_to_matrix(pose)
        t_end_camera = load_hand_eye_matrix(args.hand_eye_config)
        t_base_camera = t_base_end @ t_end_camera
        if (
            not np.isfinite(t_base_camera).all()
            or not np.allclose(t_base_camera[3], [0, 0, 0, 1])
            or not np.allclose(
                t_base_camera[:3, :3].T @ t_base_camera[:3, :3], np.eye(3), atol=1e-5
            )
            or not np.isclose(np.linalg.det(t_base_camera[:3, :3]), 1.0)
        ):
            raise ValueError("Hand-eye/capture transform must be finite and rigid")
        result["input"].update(
            robot_pose_xyzrpy_m_rad=list(pose),
            pose_frame="flange",
            pose_source="command_line",
        )
        result["frame_transform"] = {
            "T_base_end": round_list(t_base_end),
            "T_end_camera": round_list(t_end_camera),
            "T_base_camera": round_list(t_base_camera),
        }
        image = cv2.imread(str(args.rgb))
        if image is None:
            raise ValueError(f"Cannot read RGB: {args.rgb}")
        expected = (camera["image_height"], camera["image_width"])
        if image.shape[:2] != expected:
            raise ValueError("RGB resolution differs from camera intrinsics")
        depth = read_depth_raw(args.d2rgb)
        if depth is None or depth.ndim != 2 or depth.shape != image.shape[:2]:
            raise ValueError(
                "D2RGB must be a raw 2D depth array matching RGB resolution"
            )
        if not 0 < args.conf <= 1 or args.imgsz <= 0:
            raise ValueError("Invalid segmentation settings")
        sys.path.insert(0, str(ROOT / "ultralytics"))
        from ultralytics import YOLO

        stage = time.perf_counter()
        model = YOLO(str(args.seg_weights))
        prediction = model.predict(
            image,
            conf=args.conf,
            imgsz=args.imgsz,
            device=args.device,
            retina_masks=True,
            max_det=10,
            verbose=False,
        )[0]
        result["timing"]["segmentation_and_load_s"] = time.perf_counter() - stage
        count = len(prediction.boxes)
        result["segmentation"] = {"instance_count": count}
        if count != 1 or prediction.masks is None:
            raise ValueError(
                f"Expected one plug segmentation instance, detected {count}"
            )
        mask = prediction.masks.data[0].detach().cpu().numpy().astype(np.uint8)
        if mask.shape != image.shape[:2]:
            raise ValueError("Segmentation mask does not match RGB resolution")
        result["segmentation"].update(
            class_name=prediction.names[int(prediction.boxes.cls[0])],
            confidence=float(prediction.boxes.conf[0]),
        )
        extraction = extract_visible_points_from_mask(
            image,
            depth,
            camera,
            prediction.masks.xy[0].tolist(),
            0.1,
            1.5,
            voxel_size_m=0.004,
            min_points=200,
            mask_erosion_px=5,
            mad_z_threshold=0.0,
        )
        result["visible_point_cloud"] = {
            "status": extraction.status,
            "reason": extraction.reason,
            **extraction.quality,
        }
        result["warnings"] = extraction.warnings
        if extraction.status != "ok":
            raise ValueError(extraction.reason or "Visible point extraction failed")
        points_base = transform_points(
            extraction.visible_points_camera_m, t_base_camera
        )
        ply_path = Path(str(prefix) + "_visible_points.ply")
        save_visible_points_ply(extraction.visible_points_camera_m, ply_path)
        result["artifacts"]["visible_points_camera_ply"] = str(ply_path)
        grasp_model = load_grasp_model(args.model_config)
        stage = time.perf_counter()
        status, transform, quality = register_symmetric_points(
            grasp_model.points_grasp_m,
            extraction.visible_points_camera_m,
            mask,
            depth,
            camera,
            device=args.device,
            axis_up_camera=t_base_camera[:3, :3].T @ [0.0, 0.0, 1.0],
        )
        result["timing"]["registration_s"] = time.perf_counter() - stage
        result["registration_quality"] = quality
        if status != "ok" or transform is None or not np.isfinite(transform).all():
            raise ValueError(quality.get("reason", "No finite registration estimate"))
        result["frame_transform"]["T_camera_grasp"] = round_list(transform)
        t_base_grasp = t_base_camera @ transform
        result["frame_transform"]["T_base_grasp"] = round_list(t_base_grasp)
        semantic = grasp_model.config["semantic_points_grasp_m"]
        ends = transform_points(
            np.array([semantic["tail_center"], semantic["head_center"]]), t_base_grasp
        )
        result["predicted_axis_base"] = {
            "tail_center_base_m": ends[0].tolist(),
            "head_center_base_m": ends[1].tolist(),
            "direction_base": (
                (ends[1] - ends[0]) / np.linalg.norm(ends[1] - ends[0])
            ).tolist(),
        }
        cad_base = transform_points(grasp_model.points_grasp_m, t_base_grasp)
        check = evaluate_groove_placement(ends[0], ends[1], segment, check_config)
        result.update(
            status="seated" if check["is_seated"] else "not_seated",
            is_seated=check["is_seated"],
            placement_check=check,
        )
        try:
            save_registration_review(
                result, image, depth, camera, extraction.visible_points_camera_m,
                grasp_model.points_grasp_m, transform,
                semantic_points_camera(grasp_model.config, transform),
            )
        except (OSError, ValueError, ImportError) as exc:
            # Export errors do not change the completed geometric decision.
            result["artifact_error"] = str(exc)
    except (ValueError, OSError, KeyError, TypeError, ImportError, RuntimeError) as exc:
        result.update(status="unknown", is_seated=None, reason=str(exc))
    try:
        save_review(result, image, camera, t_base_camera, mask, points_base, cad_base)
    except (OSError, ValueError, ImportError) as exc:
        # Visualization failure must not discard the measured result JSON.
        result["artifact_error"] = "; ".join(filter(None, [result.get("artifact_error"), str(exc)]))
        result["artifacts"] = {
            key: value
            for key, value in result["artifacts"].items()
            if Path(value).is_file()
        }
    result["timing"]["total_s"] = time.perf_counter() - start
    write_json(path, result)
    if args.show and "review_html" in result["artifacts"]:
        show_visualization(Path(result["artifacts"]["review_html"]))
    return result, path


def main() -> int:
    result, path = run(parse_args())
    print(f"# 放置结论：{LABELS[result['status']]}")
    print(f"status: {result['status']}")
    if "placement_check" in result:
        check = result["placement_check"]
        print(
            f"Diagnostic head / tail / max axis distance (mm): {check['head_distance_mm']:.3f} / {check['tail_distance_mm']:.3f} / {check['max_distance_mm']:.3f}"
        )
        print(f"Axis inside frustum: {check['inside_fraction']:.2%}; required: {check['min_inside_fraction']:.0%}")
        print(f"Frustum top / bottom radius (mm): {check['top_radius_mm']:g} / {check['bottom_radius_mm']:g}")
    else:
        print(f"reason: {result.get('reason')}")
    print(f"JSON: {path}")
    for name, artifact in result["artifacts"].items():
        print(f"{name}: {artifact}")
    if "artifact_error" in result:
        print(f"artifact_error: {result['artifact_error']}")
    return {"seated": 0, "not_seated": 1, "unknown": 2}[result["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
