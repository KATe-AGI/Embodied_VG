"""Batch stage-1 diagnostics; deliberately does not run CAD registration."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "ultralytics")]

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from PIL import Image, ImageOps
import plotly.graph_objects as go
from ultralytics import YOLO

from plug_vg.calibration_registration import (
    calibration_observation, calibration_priors, initial_endpoint_candidates,
)
from plug_vg.config import CALIBRATION_SEG_WEIGHTS, DEFAULT_CAMERA, load_camera
from plug_vg.robot_transform import load_hand_eye_matrix, robot_pose_to_matrix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "test_gt_20260902")
    parser.add_argument("--output", type=Path, default=ROOT / "output/calibration_endpoint_icp/stage1/revision_0")
    parser.add_argument("--clamp-center-end", type=float, nargs=3, default=[0., 0., .420])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    handeye = ROOT / "hand_eye_calibration/eye_hand_data/calib_20260812_PARK/hand_eye_result_in-hand.yaml"
    robot_pose = [-.014293, .460711, .742759, 2.167158, .044541, -3.126827]
    t_ec = load_hand_eye_matrix(handeye)
    center, up = calibration_priors(t_ec, robot_pose_to_matrix(robot_pose), args.clamp_center_end)
    assert np.allclose(t_ec @ np.r_[center, 1.], np.r_[args.clamp_center_end, 1.])
    camera = load_camera(DEFAULT_CAMERA)
    model = YOLO(str(CALIBRATION_SEG_WEIGHTS))
    records, pngs, hashes = [], [], {}
    colors = ["#0072b2", "#cc7900"]
    rgbs = [[0., .447, .698], [.8, .475, 0.]]
    for path in sorted(args.dataset.glob("*_color.png")):
        sample = path.stem.removesuffix("_color")
        depth_path = args.dataset / f"{sample}_d2rgb.npy"
        for input_path in (path, depth_path):
            hashes[input_path.name] = hashlib.sha256(input_path.read_bytes()).hexdigest()
        image = cv2.imread(str(path))
        prediction = model.predict(image, imgsz=640, conf=.25, iou=.7, max_det=10, verbose=False)[0]
        try:
            obs = calibration_observation(image, np.load(depth_path), camera, prediction)
            core, raw = obs["core"].visible_points_camera_m, obs["raw"].visible_points_camera_m
            candidates = initial_endpoint_candidates(core, raw, center, up, obs["class_name"])
        except ValueError as error:
            records.append(dict(sample_id=sample, status="failed", reason=str(error)))
            continue

        fig, axes = plt.subplots(1, 3, figsize=(17, 5), layout="constrained")
        axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        polygon = np.asarray(obs["polygon_xy"])
        axes[0].plot(*np.vstack([polygon, polygon[0]]).T, color="lime", linewidth=1)
        axes[0].set_title(f"YOLO: {obs['class_name']} ({obs['confidence']:.3f})")
        axes[0].axis("off")
        detail = np.all(np.abs(raw - center) <= .12, axis=1)
        interactive = go.Figure()
        ply_points, ply_colors = [raw, core, center[None]], [np.tile([.65, .65, .65], (len(raw), 1)), np.tile([.1, .65, .25], (len(core), 1)), [[1., 0., 0.]]]
        for points, color, name, size in [(raw, "#aaaaaa", "Raw mask depth", 1), (core, "#199d40", "Core points", 2), (center[None], "red", "Clamp center", 7)]:
            interactive.add_trace(go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2], mode="markers", marker=dict(color=color, size=size), name=name))
        for ax, dims, title in [(axes[1], (0, 1), "Camera XY"), (axes[2], (2, 1), "Camera ZY (side)")]:
            a, b = dims
            ax.scatter(raw[detail, a]*1000, raw[detail, b]*1000, s=1, color=".7", label="Raw mask depth")
            ax.scatter(core[:, a]*1000, core[:, b]*1000, s=5, color="#199d40", label="Core points")
            ax.scatter(center[a]*1000, center[b]*1000, s=140, c="red", marker="*", label="Clamp center")
            ax.set(xlim=((center[a]-.12)*1000, (center[a]+.12)*1000), ylim=((center[b]+.12)*1000, (center[b]-.12)*1000),
                   xlabel=f"Camera {'XYZ'[a]} (mm)", ylabel=f"Camera {'XYZ'[b]} (mm)", title=title)
            ax.set_aspect("equal"); ax.grid(alpha=.2)
        summaries = []
        for candidate, color, rgb in zip(candidates, colors, rgbs):
            axis, end = candidate["axis_camera"], candidate["endpoint_camera_m"]
            line = center + np.linspace(-.15, .15, 301)[:, None]*axis
            support = raw[candidate["support_mask"]]
            for ax, (a, b) in zip(axes[1:], [(0, 1), (2, 1)]):
                ax.plot(line[:, a]*1000, line[:, b]*1000, color=color, linewidth=1, label=candidate["name"])
                ax.scatter(end[a]*1000, end[b]*1000, color=color, marker="x", s=85)
                ax.scatter(support[:, a]*1000, support[:, b]*1000, color=color, s=3, alpha=.4)
            interactive.add_trace(go.Scatter3d(x=line[:, 0], y=line[:, 1], z=line[:, 2], mode="lines", line=dict(color=color), name=candidate["name"]+" axis"))
            interactive.add_trace(go.Scatter3d(x=[end[0]], y=[end[1]], z=[end[2]], mode="markers", marker=dict(color=color, size=6, symbol="diamond"), name=candidate["name"]+" endpoint"))
            interactive.add_trace(go.Scatter3d(x=support[:, 0], y=support[:, 1], z=support[:, 2], mode="markers", marker=dict(color=color, size=2), name=candidate["name"]+" boundary support (+/-3mm)"))
            ply_points.extend([line, end[None]])
            ply_colors.extend([np.tile(rgb, (len(line), 1)), [rgb]])
            summaries.append({k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in candidate.items() if k != "support_mask"})
            summaries[-1]["support_points"] = len(support)
        axes[1].legend(fontsize=7)
        fig.suptitle(f"{sample} | STAGE 1: unranked axis seeds, no CAD / ICP\nDetail +/-120 mm from clamp; HTML/PLY retain ALL points; crosses = axial boundary estimates")
        png = args.output / f"{sample}.png"
        fig.savefig(png, dpi=150); plt.close(fig); pngs.append(png)
        interactive.update_layout(title=f"{sample} | Stage 1 only | {obs['class_name']} | RGB camera (m)", scene=dict(aspectmode="data", xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)"))
        interactive.write_html(args.output / f"{sample}.html", include_plotlyjs="directory")
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.vstack(ply_points)))
        cloud.colors = o3d.utility.Vector3dVector(np.vstack(ply_colors))
        assert o3d.io.write_point_cloud(str(args.output / f"{sample}.ply"), cloud, write_ascii=True)
        np.savez_compressed(args.output / f"{sample}_observation.npz", raw=raw, core=core, raw_mask=obs["raw"].mask, core_mask=obs["core"].mask)
        records.append(dict(sample_id=sample, status="awaiting_visual_review", class_name=obs["class_name"], confidence=obs["confidence"], raw_points=len(raw), core_points=len(core), raw_points_outside_png_detail=int((~detail).sum()), candidates=summaries))
        print(sample, obs["class_name"], [(c["name"], round(c["endpoint_offset_m"]*1000, 2), c["support_points"]) for c in summaries], flush=True)

    if not records:
        raise RuntimeError("No RGB frames found")
    if pngs:
        overview = Image.new("RGB", (1700, 500*len(pngs)), "white")
        for i, png in enumerate(pngs):
            overview.paste(ImageOps.contain(Image.open(png), (1700, 500)), (0, 500*i))
        overview.save(args.output / "overview.png")
    report = dict(stage="1_observation_and_endpoints", revision=0, status="awaiting_visual_review", ground_truth_used=False,
                  clamp_center_end_m=args.clamp_center_end, clamp_center_camera_m=center.tolist(), nominal_up_camera=up.tolist(),
                  robot_pose_xyzrpy_m_rad=robot_pose, hand_eye=str(handeye), t_end_camera=t_ec.tolist(),
                  weights=str(CALIBRATION_SEG_WEIGHTS), camera=str(DEFAULT_CAMERA), input_sha256=hashes,
                  parameters=dict(core_erosion_px=5, core_voxel_m=.004, raw_erosion_px=0, raw_voxel_m=0, depth_range_m=[.1, 1.], global_mad=False, tail_quantile=.02, head_quantile=.98, support_band_m=.003), frames=records)
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    links = "".join(f'<li>{r["sample_id"]}: {r["status"]}'+(f' · <a href="{r["sample_id"]}.html">旋转点云</a> · <a href="{r["sample_id"]}.png">三视图</a> · <a href="{r["sample_id"]}.ply">PLY</a>' if r["status"] != "failed" else f' ({r["reason"]})')+'</li>' for r in records)
    (args.output / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>校准第一阶段</title><h1>观测、轴向初值与端部定位</h1><p>灰：原掩膜有效深度；绿：腐蚀掩膜核心点；红：夹爪中心；蓝：PCA 初值；橙：夹爪到观测初值。叉号/菱形表示对应轴线上的端部估计，同色点表示边界附近3mm的观测支持。</p><p>两种轴线尚未经过 CAD 拟合或排序，不能视为最终配准。PNG仅显示夹爪中心±120mm；HTML/PLY保留全部点。请同时查看侧视图与完整点云，检查远处异常深度是否影响轴向和端部。</p>'+f'<ul>{links}</ul><img src="overview.png" style="max-width:100%">')


if __name__ == "__main__":
    main()
