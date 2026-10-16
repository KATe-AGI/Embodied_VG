#!/usr/bin/env python3
"""RGB-only AprilTag center and camera-facing normal in camera and robot base frames."""

from __future__ import annotations

import argparse
import base64
import html
import os
from pathlib import Path
import sys
import webbrowser

import cv2
import numpy as np

from plug_vg.apriltag_pose import center_normal_to_base, estimate_tag_pose, get_tag_center_normal
from plug_vg.config import DEFAULT_CAMERA, ROOT, load_camera
from plug_vg.grasp_model import transform_points
from plug_vg.io import write_json
from plug_vg.placement import DEFAULT_PLACEMENT, load_placement, preset_points
from plug_vg.robot_transform import load_hand_eye_matrix, robot_pose_to_matrix, round_list


r'''
conda activate embodiedvg

python infer_apriltag_cylinder.py \
  --rgb camera_data/20261013_131742_642_color.png \
  --tagsize 0.018 \
  --robot-pose -0.30118 -0.49979 0.24279 1.65792 -0.08564 -3.14141 \
  --output-dir output/apriltag_pose \
  --show
'''

DEFAULT_HAND_EYE = ROOT / "hand_eye_calibration/eye_hand_data/calib_20260812_PARK/hand_eye_result_in-hand.yaml"
GROOVE_PLANE_NORMAL_OFFSET_M = 0.055


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb", type=Path, required=True)
    parser.add_argument("--robot-pose", type=float, nargs=6, required=True,
                        metavar=("X", "Y", "Z", "ROLL", "PITCH", "YAW"),
                        help="T_base_end at capture time, meters and fixed-axis XYZ RPY radians.")
    parser.add_argument("--camera-config", type=Path, default=DEFAULT_CAMERA, help="Use the existing D435IF intrinsics by default.")
    parser.add_argument("--hand-eye-config", type=Path, default=DEFAULT_HAND_EYE)
    parser.add_argument("--placement-config", type=Path, default=DEFAULT_PLACEMENT)
    parser.add_argument("--tagsize", type=float, default=0.018,
                        help="Measured blue 5x5 pose square side in meters (official tagsize); default 0.018.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output/apriltag_pose")
    parser.add_argument("--show", action="store_true",
                        help="Open the HTML with RGB overlay and interactive 3D view when a desktop is available.")
    return parser.parse_args()


def detect_tag(image: np.ndarray) -> object:
    """Decode ID 0 only; pose estimation is delegated to the vendored official code."""
    from pupil_apriltags import Detector

    detector = Detector(families="tagStandard41h12", nthreads=4, quad_decimate=1.0)
    detections = detector.detect(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), estimate_tag_pose=False)
    matches = [tag for tag in detections if tag.tag_id == 0 and tag.hamming == 0]
    if not matches:
        raise ValueError("No correctly decoded tagStandard41h12 ID 0 detected")
    if len(matches) != 1:
        raise ValueError("Multiple ID 0 tags detected; use an image with one target tag")
    return matches[0]


def tag_object_corners(size: float) -> np.ndarray:
    """Official estimate_tag_pose() corner order; do not reorder detector corners."""
    half = size / 2
    return np.array([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]])


def draw_overlay(image: np.ndarray, tag: object, camera: dict, center: np.ndarray,
                 normal: np.ndarray, arrow_length: float, result: dict) -> np.ndarray:
    overlay = image.copy()
    matrix = np.array([[camera["fx"], 0, camera["cx"]], [0, camera["fy"], camera["cy"]], [0, 0, 1]], float)
    cv2.polylines(overlay, [np.rint(tag.corners).astype(np.int32)], True, (0, 220, 0), 2, cv2.LINE_AA)
    endpoints = np.vstack([center, center + arrow_length * normal])
    pixels = (endpoints @ matrix.T)[:, :2] / endpoints[:, 2, None]
    points = np.clip(np.rint(pixels), -1e6, 1e6).astype(int)
    start, end = [tuple(point) for point in points]
    visible, a, b = cv2.clipLine((0, 0, image.shape[1], image.shape[0]), start, end)
    if visible:
        cv2.arrowedLine(overlay, a, b, (0, 165, 255), 3, cv2.LINE_AA, tipLength=0.2)
    cv2.drawMarker(overlay, start, (0, 230, 0), cv2.MARKER_CROSS, 22, 3)
    cv2.putText(overlay, "ID 0 center / normal toward camera", (start[0] + 12, start[1] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2, cv2.LINE_AA)
    transform = np.asarray(result["frame_transform"]["T_base_camera"])
    for key, color in [("preset", (220, 70, 160)), ("head_preset", (70, 70, 255)), ("tail_preset", (200, 180, 0))]:
        point_base = np.asarray(result["placement_points_base_m"][key])
        point_camera = transform[:3, :3].T @ (point_base - transform[:3, 3])
        if point_camera[2] <= 0:
            continue
        pixel = matrix @ point_camera
        u, v = np.rint(pixel[:2] / pixel[2]).astype(int)
        if 0 <= u < image.shape[1] and 0 <= v < image.shape[0]:
            cv2.drawMarker(overlay, (u, v), color, cv2.MARKER_CROSS, 18, 2)
            cv2.putText(overlay, key + " (virtual)", (u + 12, v + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    texts = ["standard41h12 ID 0 / official estimate_tag_pose / RGB only",
             f"Camera center (m): {np.array2string(center, precision=5)}",
             f"Camera unit normal: {np.array2string(normal, precision=5)}",
             f"Base center (m): {np.array2string(np.array(result['tag_center_base_m']), precision=5)}",
             f"Base unit normal: {np.array2string(np.array(result['tag_normal_base']), precision=5)}",
             f"Blue pose square (tagsize): {result['tag']['pose_square_size_m'] * 1000:.1f} mm; arrow length is visualization only"]
    cv2.rectangle(overlay, (10, 10), (min(image.shape[1] - 1, 1120), 195), (30, 30, 30), -1)
    for i, text in enumerate(texts):
        cv2.putText(overlay, text, (20, 35 + i * 29), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return overlay


def save_3d_views(center: np.ndarray, normal: np.ndarray, corners: np.ndarray,
                  camera_origin: np.ndarray, arrow_length: float, result: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import plotly.graph_objects as go

    arrow_end = center + arrow_length * normal
    outline = np.vstack([corners, corners[0]])
    fig = go.Figure()
    fig.add_trace(go.Mesh3d(x=corners[:, 0], y=corners[:, 1], z=corners[:, 2],
                           i=[0, 0], j=[1, 2], k=[2, 3], color="limegreen", opacity=0.25,
                           name="Tag plane", showlegend=True))
    fig.add_trace(go.Scatter3d(x=outline[:, 0], y=outline[:, 1], z=outline[:, 2], mode="lines",
                              name="Tag pose square", line=dict(color="green", width=4)))
    points = np.vstack([center, camera_origin])
    fig.add_trace(go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2], mode="markers+text",
                              text=["Tag center", "Camera optical center"], textposition="top center",
                              marker=dict(size=6, color=["green", "blue"]), name="Centers"))
    line = np.vstack([center, arrow_end])
    fig.add_trace(go.Scatter3d(x=line[:, 0], y=line[:, 1], z=line[:, 2], mode="lines",
                              line=dict(color="orange", width=7), name="Camera-facing unit normal"))
    fig.add_trace(go.Cone(x=[arrow_end[0]], y=[arrow_end[1]], z=[arrow_end[2]],
                         u=[normal[0]], v=[normal[1]], w=[normal[2]], anchor="tip", sizemode="absolute",
                         sizeref=arrow_length * 0.25, colorscale=[[0, "orange"], [1, "orange"]], showscale=False))
    sight = np.vstack([center, camera_origin])
    fig.add_trace(go.Scatter3d(x=sight[:, 0], y=sight[:, 1], z=sight[:, 2], mode="lines",
                              line=dict(color="blue", width=2, dash="dash"), name="Center to camera"))
    presets = result["placement_points_base_m"]
    colors = {"preset": "purple", "head_preset": "crimson", "tail_preset": "teal"}
    for key, color in colors.items():
        point = np.asarray(presets[key])
        fig.add_trace(go.Scatter3d(x=[point[0]], y=[point[1]], z=[point[2]], mode="markers+text",
                                  text=[key], name=key, marker=dict(size=7, color=color)))
        origin = center if key == "preset" else np.asarray(presets["preset"])
        link = np.vstack([origin, point])
        length_mm = np.linalg.norm(point - origin) * 1000
        fig.add_trace(go.Scatter3d(x=link[:, 0], y=link[:, 1], z=link[:, 2], mode="lines+text",
                                  text=["", f"{length_mm:.1f} mm"], name=f"Offset to {key}",
                                  line=dict(color=color, width=3, dash="dash")))
    for axis, color in enumerate(["red", "green", "blue"]):
        end = center.copy()
        end[axis] += arrow_length
        pair = np.vstack([center, end])
        fig.add_trace(go.Scatter3d(x=pair[:, 0], y=pair[:, 1], z=pair[:, 2], mode="lines+text",
                                  text=["", "+" + "XYZ"[axis]], name="Base " + "XYZ"[axis], line=dict(color=color, width=3)))
    bounds = np.vstack([corners, center, arrow_end, camera_origin, *presets.values()])
    middle = (bounds.max(axis=0) + bounds.min(axis=0)) / 2
    half = max(float(np.ptp(bounds, axis=0).max()) * 0.65, 0.10)
    title = f"Base center (m): {np.round(center, 5).tolist()} — unit normal: {np.round(normal, 5).tolist()}"
    fig.update_layout(title=title, margin=dict(l=0, r=0, b=0, t=65), scene=dict(aspectmode="cube",
                      **{f"{axis}axis": dict(title=f"Base {axis.upper()} (m)", range=[middle[i] - half, middle[i] + half])
                         for i, axis in enumerate("xyz")}))
    # Embed RGB and Plotly so a single offline HTML also works when copied from SSH.
    rgb_data = base64.b64encode(Path(result["artifacts"]["rgb_overlay"]).read_bytes()).decode("ascii")
    rgb_section = (
        '<h2>RGB: AprilTag center and camera-facing normal</h2>'
        f'<img alt="AprilTag RGB overlay" src="data:image/jpeg;base64,{rgb_data}" '
        'style="display:block;max-width:100%;height:auto;margin:auto">'
        '<h2>Robot base frame: center and directed normal</h2>'
    )
    page = fig.to_html(include_plotlyjs=True, full_html=True)
    page = page.replace("<head>", '<head><meta charset="utf-8">'
                        f'<title>{html.escape(title)}</title>'
                        '<style>body{margin:20px;font-family:sans-serif}h2{font-size:18px}</style>', 1)
    page = page.replace("<body>", "<body>" + rgb_section, 1)
    Path(result["artifacts"]["base_view_html"]).write_text(page, encoding="utf-8")

    static = plt.figure(figsize=(10, 8))
    ax = static.add_subplot(111, projection="3d")
    ax.add_collection3d(Poly3DCollection([corners], facecolor="green", alpha=0.2))
    ax.plot(*outline.T, color="green", label="Tag pose square")
    ax.scatter(*center, color="green", s=70, label="Tag center")
    ax.scatter(*camera_origin, color="blue", s=70, label="Camera optical center")
    ax.plot(*sight.T, "--", color="blue", alpha=0.4)
    ax.quiver(*center, *normal, length=arrow_length, color="orange", linewidth=3,
              label="Unit normal toward camera")
    for key, color in colors.items():
        point = np.asarray(presets[key])
        ax.scatter(*point, color=color, s=60, label=key)
        ax.text(*point, key, fontsize=8)
        origin = center if key == "preset" else np.asarray(presets["preset"])
        ax.plot(*np.vstack([origin, point]).T, "--", color=color)
    for i, axis in enumerate("xyz"):
        getattr(ax, f"set_{axis}lim")(middle[i] - half, middle[i] + half)
        getattr(ax, f"set_{axis}label")(f"Base {axis.upper()} (m)")
    ax.set_box_aspect((1, 1, 1))
    ax.set_title("AprilTag center and directed normal — robot base frame")
    ax.legend()
    static.tight_layout()
    static.savefig(result["artifacts"]["base_view_png"], dpi=160)
    plt.close(static)


def show_visualization(path: Path, *, quiet: bool = False) -> None:
    """Open the offline visualization without initializing OpenCV's Qt backend."""
    path = path.resolve()
    if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        if not quiet:
            print(f"No graphical display; open the saved HTML in a desktop browser: {path}")
        return
    try:
        opened = webbrowser.open(path.as_uri())
    except (webbrowser.Error, OSError) as exc:
        if not quiet:
            print(f"Could not open browser ({exc}); open the saved HTML manually: {path}")
        return
    if not opened and not quiet:
        print(f"No browser opened; open the saved HTML manually: {path}")


def run(args: argparse.Namespace) -> tuple[dict, Path]:
    if not np.isfinite([*args.robot_pose, args.tagsize]).all() or args.tagsize <= 0:
        raise ValueError("Robot pose must be finite and tag size must be finite and positive")
    image = cv2.imread(str(args.rgb))
    if image is None:
        raise ValueError(f"Cannot read RGB: {args.rgb}")
    camera = load_camera(args.camera_config)
    if image.shape[:2] != (camera["image_height"], camera["image_width"]):
        raise ValueError("RGB resolution differs from camera calibration; provide matching --camera-config")
    tag = detect_tag(image)
    rotation, translation, error = estimate_tag_pose(tag, camera, args.tagsize)
    center_camera, normal_camera = get_tag_center_normal(rotation, translation)
    t_base_end = robot_pose_to_matrix(args.robot_pose)
    t_end_camera = load_hand_eye_matrix(args.hand_eye_config)
    t_base_camera = t_base_end @ t_end_camera
    center_base, normal_base = center_normal_to_base(center_camera, normal_camera, t_base_camera)
    placement_config = load_placement(args.placement_config)
    placement_points = preset_points(center_base, normal_base, placement_config)
    groove_plane_point = placement_points["reference"] + GROOVE_PLANE_NORMAL_OFFSET_M * normal_base
    camera_origin_base = t_base_camera[:3, 3]
    pose_size = args.tagsize
    corners_camera = tag_object_corners(pose_size) @ rotation.T + center_camera
    corners_base = transform_points(corners_camera, t_base_camera)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_dir / args.rgb.stem
    json_path = Path(str(prefix) + "_apriltag_pose.json")
    result = {
        "status": "ok", "position_unit": "meter", "normal_convention": "unit vector toward camera half-space",
        "input": {"rgb": str(args.rgb), "robot_pose_xyzrpy_m_rad": args.robot_pose,
                  "pose_frame": "flange",
                  "camera_config": str(args.camera_config), "hand_eye_config": str(args.hand_eye_config),
                  "placement_config": str(args.placement_config)},
        "tag": {"family": "tagStandard41h12", "id": int(tag.tag_id), "hamming": int(tag.hamming),
                "decision_margin": float(tag.decision_margin), "center_px": round_list(tag.center),
                "corners_px": round_list(tag.corners), "full_pattern_size_m": pose_size * 9 / 5, "pose_square_size_m": pose_size},
        "pose_estimation": {"method": "AprilRobotics.estimate_tag_pose", "object_space_error_m2": error,
                            "rotation_camera_tag": round_list(rotation)},
        "tag_center_camera_m": round_list(center_camera), "tag_normal_camera": round_list(normal_camera),
        "tag_center_base_m": round_list(center_base), "tag_normal_base": round_list(normal_base),
        "placement_points_base_m": {key: round_list(point) for key, point in placement_points.items()},
        "groove_tangent_plane_base": {
            "normal_offset_m": GROOVE_PLANE_NORMAL_OFFSET_M,
            "point_base_m": round_list(groove_plane_point),
            "point_y_base_m": round(float(groove_plane_point[1]), 8),
            "normal_base": round_list(normal_base),
            "equation": "normal_base dot (P_base_m - point_base_m) = 0",
        },
        "placement_offsets": placement_config["reference"],
        "approach_direction_base": round_list(-normal_base),
        "center_normal_base": dict(zip(("center_x", "center_y", "center_z", "normal_x", "normal_y", "normal_z"),
                                       round_list(np.r_[center_base, normal_base]))),
        "camera_origin_base_m": round_list(camera_origin_base),
        "frame_transform": {"point": "P_base = R_base_camera @ P_camera + t_base_camera",
                            "normal": "n_base = R_base_camera @ n_camera",
                            "chain": "T_base_camera = T_base_end @ T_end_camera",
                            "T_base_end": round_list(t_base_end), "T_end_camera": round_list(t_end_camera),
                            "T_base_camera": round_list(t_base_camera)},
        "artifacts": {"rgb_overlay": str(prefix) + "_overlay.jpg", "base_view_html": str(prefix) + "_base_view.html",
                      "base_view_png": str(prefix) + "_base_view.png"},
    }
    arrow_length = min(0.08, float(center_camera[2]) * 0.25)
    overlay = draw_overlay(image, tag, camera, center_camera, normal_camera, arrow_length, result)
    if not cv2.imwrite(result["artifacts"]["rgb_overlay"], overlay):
        raise OSError("Could not save RGB overlay")
    save_3d_views(center_base, normal_base, corners_base, camera_origin_base, arrow_length, result)
    write_json(json_path, result)
    if args.show:
        show_visualization(Path(result["artifacts"]["base_view_html"]), quiet=True)
    return result, json_path


def main() -> int:
    args = parse_args()
    try:
        result, path = run(args)
    except (ValueError, OSError, KeyError, ImportError, RuntimeError) as exc:
        write_json(args.output_dir / f"{args.rgb.stem}_apriltag_pose.json", {"status": "failed", "reason": str(exc)})
        print(f"failed: {exc}")
        return 1
    print("# 定位点：机器人基坐标系下的 XYZ 坐标，单位为米")
    print("# reference：参考点；preset：预设点；head_preset：头部预设点；tail_preset：尾部预设点")
    for name, point in result["placement_points_base_m"].items():
        print(f"{name} base XYZ (m): {point}")
    print("\n# 法线：机器人基坐标系下朝向相机一侧的单位向量，无单位")
    print(f"tag_normal_base: {result['tag_normal_base']}")
    print("\n# 凹槽切平面：参考点沿法线正方向平移 5.5 厘米后，定位点的基坐标系 Y 值，单位为米")
    print("# 切平面存在倾斜，此 Y 值仅对应定位点")
    print(f"groove tangent plane point base Y (m): {result['groove_tangent_plane_base']['point_y_base_m']}")
    print("\n# 输出文件：位姿数据、图像叠加结果、交互式三维视图和静态三维视图")
    print(f"JSON: {path}")
    for name, artifact in result["artifacts"].items():
        print(f"{name}: {artifact}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
