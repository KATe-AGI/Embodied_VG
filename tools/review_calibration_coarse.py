"""Stage-2 batch CAD review using the frozen stage-1 YOLO/depth observations."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from PIL import Image, ImageOps
import plotly.graph_objects as go

from plug_vg.calibration_registration import EndpointCADRegistration, filter_calibration_clouds
from plug_vg.grasp_model import load_grasp_model, resolve_path


def serialize(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, default=ROOT/'output/calibration_endpoint_icp/stage1/revision_0')
    parser.add_argument("--dataset", type=Path, default=ROOT/'test_gt_20260902')
    parser.add_argument("--output", type=Path, default=ROOT/'output/calibration_endpoint_icp/stage2/revision_0')
    parser.add_argument("--icp-from", type=Path, help="Refine frozen stage-2 poses; requires a separate --output directory")
    args = parser.parse_args()
    if args.icp_from and args.output.resolve() == args.icp_from.resolve():
        parser.error("ICP output must not overwrite the accepted coarse results")
    frozen = json.loads((args.observations/'report.json').read_text())
    for name, digest in frozen['input_sha256'].items():
        assert hashlib.sha256((args.dataset/name).read_bytes()).hexdigest() == digest, name
    for name in ['weights', 'camera', 'hand_eye']:
        path = Path(frozen[name])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == frozen['source_and_config_sha256'][str(path)], name
    args.output.mkdir(parents=True, exist_ok=True)
    model = load_grasp_model()
    mesh_path = resolve_path(model.config_path, model.config['assets']['mesh_obj'])
    semantic = model.config['semantic_points_grasp_m']
    names = ['tail_center', 'grasp_center', 'head_center']
    material = np.array([semantic[name] for name in names])
    start = time.perf_counter()
    solver = EndpointCADRegistration(o3d.io.read_triangle_mesh(str(mesh_path)), material[0, 0], material[2, 0])
    init_seconds = time.perf_counter()-start
    center, up = np.array(frozen['clamp_center_camera_m']), np.array(frozen['nominal_up_camera'])
    cad = model.points_grasp_m[::5]
    records, pngs, errors = [], [], []
    for frame in frozen['frames']:
        sample = frame['sample_id']
        data = np.load(args.observations/f'{sample}_observation.npz')
        start = time.perf_counter()
        keep_core, keep_raw = filter_calibration_clouds(data['core'], data['raw'])
        core, raw = data['core'][keep_core], data['raw'][keep_raw]
        rejected = data['raw'][~keep_raw]
        filter_seconds = time.perf_counter()-start
        start = time.perf_counter()
        coarse = None
        if args.icp_from:
            coarse = json.loads((args.icp_from/f'{sample}.json').read_text())
            result = solver.refine_icp(core, coarse['t_camera_grasp'], center, frame['class_name'])
            if result['status'] == 'failed':
                import shutil
                result.update(sample_id=sample, coarse_t_camera_grasp=coarse['t_camera_grasp'])
                (args.output/f'{sample}.json').write_text(json.dumps(result, default=serialize, indent=2))
                for suffix in ('png', 'ply'):
                    shutil.copyfile(args.icp_from/f'{sample}.{suffix}', args.output/f'{sample}.{suffix}')
                (args.output/f'{sample}.html').write_text(f'<!doctype html><meta charset="utf-8"><h1>ICP失败：{result["reason"]}</h1><p>下图及PLY仅为保留的粗配准证据，不是精配准结果。</p><img src="{sample}.png" style="max-width:100%">')
                records.append(result)
                continue
            result['coarse_t_camera_grasp'] = coarse['t_camera_grasp']
            result['coarse_surface_median_m'] = coarse['surface_median_m']
            result['coarse_surface_p95_m'] = coarse['surface_p95_m']
        else:
            result = solver.fit_coarse(core, raw, center, up, frame['class_name'])
        fit_seconds = time.perf_counter()-start
        t = result['t_camera_grasp']
        aligned, points = cad @ t[:3, :3].T + t[:3, 3], material @ t[:3, :3].T + t[:3, 3]
        axis = t[:3, 0]
        result.update(sample_id=sample, class_name=frame['class_name'], clamp_center_camera_m=center,
                      clamp_axis_distance_m=float(np.linalg.norm(np.cross(center-t[:3, 3], axis))),
                      semantic_points_camera=dict(zip(names, points)),
                      preprocessing_seconds=filter_seconds, **({'icp_seconds': fit_seconds} if coarse else {'coarse_seconds': fit_seconds}),
                      core_before=len(data['core']), core_after=len(core), raw_before=len(data['raw']), raw_after=len(raw))
        # GT is read only after automatic registration, solely for an optional review layer.
        gt_path = args.dataset/f'{sample}_color_6d_base/pose_gt.txt'
        gt_cad = None
        if gt_path.exists():
            gt = np.loadtxt(gt_path)
            gt_cad = cad @ gt[:3, :3].T + gt[:3, 3]
            gt_points = material @ gt[:3, :3].T + gt[:3, 3]
            error = np.linalg.norm(points-gt_points, axis=1)*1000
            errors.extend(error)
            gt_axis = gt[:3, 0]/np.linalg.norm(gt[:3, 0])
            result['gt_reference_only'] = dict(point_errors_mm=dict(zip(names, error)), directed_axis_error_deg=float(np.rad2deg(np.arccos(np.clip(axis @ gt_axis, -1, 1)))))

        plot = go.Figure()
        for cloud, color, name, visible, size in [
            (data['raw'][::4], '#aaaaaa', 'Original valid mask depth', 'legendonly', 1),
            (rejected[::2], '#a000b0', 'Rejected distant depth', 'legendonly', 2),
            (raw[::4], '#aaaaaa', 'Retained raw boundary depth', True, 1),
            (core, '#199d40', 'Retained core', True, 2),
            (aligned, '#e08020', 'ICP CAD' if coarse else 'Coarse CAD (no ICP)', True, 1),
            (center[None], 'red', 'Clamp center', True, 7)]:
            plot.add_trace(go.Scatter3d(x=cloud[:, 0], y=cloud[:, 1], z=cloud[:, 2], mode='markers', marker=dict(color=color, size=size), name=name, visible=visible))
        if gt_cad is not None:
            plot.add_trace(go.Scatter3d(x=gt_cad[:, 0], y=gt_cad[:, 1], z=gt_cad[:, 2], mode='markers', marker=dict(color='blue', size=1), name='GT reference only', visible='legendonly'))
        if coarse:
            tc = np.asarray(coarse['t_camera_grasp'])
            coarse_cad = cad @ tc[:3,:3].T + tc[:3,3]
            plot.add_trace(go.Scatter3d(x=coarse_cad[:,0], y=coarse_cad[:,1], z=coarse_cad[:,2], mode='markers', marker=dict(color='#0072b2',size=1), name='Accepted coarse CAD', visible='legendonly'))
        plot.add_trace(go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2], mode='markers+lines+text', text=names, marker=dict(color=['cyan', 'purple', 'gold'], size=5), name='Axis / semantic points'))
        stage_label = 'ICP refinement' if coarse else 'COARSE ONLY, no ICP'
        plot.update_layout(title=f'{sample}: {stage_label} | RGB camera, meters', scene=dict(aspectmode='data', xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Z (m)'))
        plot.write_html(args.output/f'{sample}.html', include_plotlyjs='directory')
        fig, grid = plt.subplots(2 if coarse else 1, 3, figsize=(16, 10 if coarse else 5), layout='constrained', squeeze=False)
        axes = grid[-1]
        for ax, (a, b) in zip(axes, [(0, 1), (2, 1), (0, 2)]):
            ax.scatter(raw[::4, a]*1000, raw[::4, b]*1000, s=1, color='.7', label='Retained boundary depth')
            ax.scatter(aligned[:, a]*1000, aligned[:, b]*1000, s=1, color='#e08020', alpha=.4, label=stage_label)
            ax.scatter(core[:, a]*1000, core[:, b]*1000, s=6, color='#199d40', label='Core')
            ax.plot(points[:, a]*1000, points[:, b]*1000, c='black', linewidth=.8)
            ax.scatter(points[:, a]*1000, points[:, b]*1000, c=['cyan', 'purple', 'gold'], s=50, edgecolors='black')
            ax.scatter(center[a]*1000, center[b]*1000, c='red', marker='*', s=130, label='Clamp')
            extent = np.vstack([aligned, core, center[None]])*1000
            midpoint = (extent.min(axis=0)+extent.max(axis=0))/2
            half = max(np.ptp(extent[:, a]), np.ptp(extent[:, b]))/2+10
            ax.set(xlim=(midpoint[a]-half, midpoint[a]+half), ylim=(midpoint[b]+half, midpoint[b]-half), xlabel=f'Camera {"XYZ"[a]} (mm)', ylabel=f'Camera {"XYZ"[b]} (mm)')
            ax.set_aspect('equal'); ax.grid(alpha=.2)
        if coarse:
            for ax, reference, (a,b) in zip(grid[0], axes, [(0,1),(2,1),(0,2)]):
                ax.scatter(coarse_cad[:,a]*1000,coarse_cad[:,b]*1000,s=1,color='#0072b2',alpha=.4,label='Accepted coarse CAD')
                ax.scatter(core[:,a]*1000,core[:,b]*1000,s=6,color='#199d40')
                ax.scatter(center[a]*1000,center[b]*1000,c='red',marker='*',s=130)
                ax.set(xlim=reference.get_xlim(),ylim=reference.get_ylim(),xlabel=reference.get_xlabel(),ylabel=reference.get_ylabel(),title='BEFORE: coarse')
                ax.set_aspect('equal');ax.grid(alpha=.2)
                reference.set_title('AFTER: ICP (same scale)')
        axes[0].legend(fontsize=7)
        fig.suptitle(f"{sample} | {stage_label} | surface median / P95: {result['surface_median_m']*1000:.2f} / {result['surface_p95_m']*1000:.2f} mm\nHead=gold, material grasp=purple, tail=cyan; red=clamp; black=axis")
        png = args.output/f'{sample}.png'
        fig.savefig(png, dpi=150); plt.close(fig); pngs.append(png)
        all_points = np.vstack([core, aligned, center[None], points])
        all_colors = np.vstack([np.tile([.1,.65,.25], (len(core),1)), np.tile([.9,.5,.1], (len(aligned),1)), [[1,0,0],[0,1,1],[.6,0,.6],[1,.8,0]]])
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(all_points)); cloud.colors = o3d.utility.Vector3dVector(all_colors)
        assert o3d.io.write_point_cloud(str(args.output/f'{sample}.ply'), cloud, write_ascii=True)
        np.savez_compressed(args.output/f'{sample}_preprocessing.npz', core_original=data['core'], raw_original=data['raw'], core_keep=keep_core, raw_keep=keep_raw)
        (args.output/f'{sample}.json').write_text(json.dumps(result, default=serialize, ensure_ascii=False, indent=2)+'\n')
        records.append(result)
        print(sample, 'surface median/P95 mm', np.round([result['surface_median_m']*1000, result['surface_p95_m']*1000],2), 'icp s' if coarse else 'coarse s', round(fit_seconds,3), flush=True)
    height = 1000 if args.icp_from else 500
    overview = Image.new('RGB', (1600, max(1,height*len(pngs))), 'white')
    for i, png in enumerate(pngs):
        overview.paste(ImageOps.contain(Image.open(png), (1600, height)), (0, height*i))
    overview.save(args.output/'overview.png')
    summary = dict(stage='3_icp' if args.icp_from else '2_coarse', revision=0, status='awaiting_visual_review', initialization_seconds=init_seconds,
                   input_observation_report=str(args.observations/'report.json'),
                   parameters=dict(dbscan_eps_m=.012, dbscan_min_points=5, raw_nearest_core_m=.008, cauchy_scale_m=.003, max_surface_points=512, candidate_count=8, max_evaluations_per_candidate=30),
                   gt_reference_only=dict(global_three_point_rmse_mm=float(np.sqrt(np.mean(np.square(errors)))), max_point_error_mm=float(max(errors))) if errors else None,
                   frames=records)
    if args.icp_from:
        summary['parameters'] = dict(max_iterations=20, correspondence_distance_m=.010, cauchy_scale_m=.003, clamp_scale_m=.005, endpoint_scale_m=.003, max_points=2000)
        summary['coarse_input_directory'] = str(args.icp_from)
    (args.output/'report.json').write_text(json.dumps(summary, default=serialize, ensure_ascii=False, indent=2)+'\n')
    links=''.join(f'<li>{f["sample_id"]} · <a href="{f["sample_id"]}.html">旋转点云 / 切换剔除点及GT</a> · <a href="{f["sample_id"]}.png">三视图</a> · <a href="{f["sample_id"]}.ply">PLY</a></li>' for f in records)
    description = '第三阶段：点到面 ICP。PNG 上排粗配准，下排精配准；交互图可切换粗配准。' if args.icp_from else '第二阶段：端部约束 CAD 粗配准，尚未运行 ICP。'
    (args.output/'index.html').write_text(f'<!doctype html><meta charset="utf-8"><h1>{description}</h1><p>绿：保留核心点；橙：当前 CAD；红星：夹爪中心；青/紫/金：尾端/材料抓取点/头端。原始深度、剔除点和GT可在交互图中打开。GT仅在求解完成后读取供参考。</p>'+f'<ul>{links}</ul><img src="overview.png" style="max-width:100%">')


if __name__ == '__main__':
    main()
