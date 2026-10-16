"""Render the actual intermediate endpoint results as an offline process figure."""
from pathlib import Path

import cv2
import numpy as np


def save_process_summary(image, depth_raw, camera, instance, location, fit, result, transform):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.patches import Circle

    font_path = Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    if font_path.is_file():
        font_manager.fontManager.addfont(str(font_path))
        family = font_manager.FontProperties(fname=str(font_path)).get_name()
    else:
        family = 'DejaVu Sans'
    with plt.rc_context({'font.family': family, 'axes.unicode_minus': False, 'font.size': 10}):
        fig = plt.figure(figsize=(19, 16), layout='constrained')
        axes = [fig.add_subplot(3, 3, i + 1, projection='3d' if i == 8 else None) for i in range(9)]
        def failure_note(ax):
            method = ax.text2D if hasattr(ax, 'text2D') else ax.text
            method(.05, .5, result.get('reason', '处理失败'), transform=ax.transAxes)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ax = axes[0]
        ax.imshow(rgb)
        ax.set_title('① 原始 RGB：先确认头/尾端在图中可见', loc='left', fontweight='bold')
        ax.set_xlabel('像素 u'); ax.set_ylabel('像素 v')
        if instance is None:
            for ax in axes[1:]:
                failure_note(ax)
            fig.savefig(result['artifacts']['process_summary_png'], dpi=160)
            plt.close(fig)
            return
        mask = instance['mask'] > 0
        yy, xx = np.nonzero(mask)
        x0, x1 = max(0, xx.min()-35), min(image.shape[1], xx.max()+36)
        y0, y1 = max(0, yy.min()-35), min(image.shape[0], yy.max()+36)
        extent = (x0-.5, x1-.5, y1-.5, y0-.5)
        crop = rgb[y0:y1, x0:x1]
        mask_crop = mask[y0:y1, x0:x1]

        def background(ax):
            ax.imshow(crop, extent=extent)
            ax.set_xlim(x0, x1); ax.set_ylim(y1, y0)
            ax.set_xlabel('原图像素 u'); ax.set_ylabel('原图像素 v')

        def contour(ax, values, color, label):
            contours, _ = cv2.findContours(values.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for item in contours:
                points = item[:, 0, :]
                closed = np.vstack([points, points[0]])
                ax.plot(closed[:, 0], closed[:, 1], color=color, lw=1.4)
            ax.plot([], [], color=color, label=label)

        ax = axes[1]; background(ax)
        overlay = np.zeros((*mask_crop.shape, 4)); overlay[mask_crop] = [0, .8, 1, .4]
        ax.imshow(overlay, extent=extent)
        contour(ax, mask, 'cyan', '原始掩膜')
        ax.set_title(f"② 分割 / {instance['class_name']} / 置信度 {instance['confidence']:.3f}", loc='left', fontweight='bold')
        ax.legend(loc='best')
        if location is None:
            for ax in axes[2:]:
                failure_note(ax)
            fig.savefig(result['artifacts']['process_summary_png'], dpi=160)
            plt.close(fig)
            return
        edge = np.asarray(location['edge_pixel'])
        direction = np.asarray(location['direction_image'])
        transverse = np.array([-direction[1], direction[0]])
        ax = axes[0]; ax.plot(*edge, 'r+', markersize=10, label='最终边缘像素'); ax.legend()
        ax = axes[2]; background(ax); contour(ax, mask, 'cyan', '掩膜')
        origin = np.array([np.median(xx), np.median(yy)])
        length = min(x1-x0, y1-y0)*.38
        ax.annotate('', xy=origin + length*direction, xytext=origin,
                    arrowprops=dict(arrowstyle='->', color='yellow', lw=3))
        ax.plot([], [], color='yellow', label='当前类别的端部方向')
        ax.plot(*edge, 'r+', markersize=12, label='边缘中点')
        ax.set_title('③ 基座 Z 轴投影：头沿 +Z，尾沿 −Z', loc='left', fontweight='bold')
        ax.text(.02, .02, f"方向检查：{location['axis_check']}", transform=ax.transAxes, fontsize=8,
                bbox=dict(facecolor='white', alpha=.85))
        ax.legend(loc='upper right')

        ax = axes[3]; background(ax)
        coords = np.column_stack([xx, yy]); axial = coords @ direction
        low, high = location['terminal_axial_range']
        terminal = (axial >= low) & (axial <= high)
        band = np.zeros((*mask_crop.shape, 4))
        band[yy[terminal]-y0, xx[terminal]-x0] = [1, .85, 0, .6]
        ax.imshow(band, extent=extent)
        lateral_low, lateral_high = location['terminal_lateral_range']
        for lateral, color, label in [(lateral_low, 'cyan', '端部两侧分位边界'),
                                       (lateral_high, 'cyan', None),
                                       (location['midpoint_lateral'], 'white', '横向中线')]:
            pts = np.array([direction*a+transverse*lateral for a in [low-15, high+15]])
            ax.plot(pts[:,0], pts[:,1], color=color, label=label)
        contour(ax, mask, 'lime', '原始轮廓')
        ax.plot(*edge, 'r+', markersize=14, markeredgewidth=2, label='中线与端部轮廓交点')
        ax.set_title('④ 末端 25% 区域 → 两侧中间 → 原轮廓交点', loc='left', fontweight='bold')
        ax.legend(loc='best', fontsize=8)

        ax = axes[4]; background(ax)
        safe = location['safe_mask'] > 0
        removed = mask[y0:y1,x0:x1] & ~safe[y0:y1,x0:x1]
        erosion = np.zeros((*mask_crop.shape,4))
        erosion[removed] = [1,.3,0,.75]; erosion[safe[y0:y1,x0:x1]] = [0,1,0,.2]
        ax.imshow(erosion, extent=extent)
        contour(ax, mask, 'cyan', '原始边界'); contour(ax, safe, 'lime', '腐蚀后采样边界')
        ax.plot(*edge, 'r+', markersize=14, label='目标保持原边缘：退让 0 px')
        ax.add_patch(Circle(edge, result['endpoint_settings']['patch_radius_px'], fill=False, color='white'))
        ax.set_title('⑤ 仅腐蚀采样掩膜；目标点不移动', loc='left', fontweight='bold')
        ax.legend(loc='best', fontsize=8)

        ax = axes[5]
        z = depth_raw.astype(float)*camera['depth_scale']*1000
        valid = safe & np.isfinite(z) & (z >= result['endpoint_settings']['min_depth_m']*1000) & (z <= result['endpoint_settings']['max_depth_m']*1000)
        depth_crop = np.ma.masked_where(~valid[y0:y1,x0:x1], z[y0:y1,x0:x1])
        cmap = plt.get_cmap('viridis').copy(); cmap.set_bad('#eeeeee')
        artist = ax.imshow(depth_crop, extent=extent, cmap=cmap)
        fig.colorbar(artist, ax=ax, shrink=.7, label='有效掩膜内深度 (mm)')
        ax.plot(*edge, 'r+', markersize=12)
        ax.add_patch(Circle(edge, result['endpoint_settings']['patch_radius_px'], fill=False, color='red'))
        ax.set_title('⑥ RGB 对齐深度：圆形邻域 ∩ 腐蚀掩膜', loc='left', fontweight='bold')
        ax.set_xlabel('原图像素 u'); ax.set_ylabel('原图像素 v')
        ax.set_xlim(x0,x1); ax.set_ylim(y1,y0)

        ax = axes[6]; background(ax); contour(ax, safe, 'cyan', '腐蚀边界')
        ax.set_title('⑦ 局部稳健拟合：有效样本与异常值', loc='left', fontweight='bold')
        ax = axes[7]; ax.set_title('⑧ 深度残差与边缘外推检查', loc='left', fontweight='bold')
        if fit is None:
            for ax in axes[6:]:
                failure_note(ax)
        else:
            pixels, keep = fit['sample_pixels'], fit['inlier_mask']
            ax = axes[6]
            for accepted, color, label in [(True,'lime','拟合内点'),(False,'magenta','剔除样本')]:
                pts = pixels[keep == accepted]
                ax.scatter(pts[:,0],pts[:,1],s=10,c=color,label=label)
            hull = fit['inlier_hull'][:,0,:]
            ax.plot(*np.vstack([hull,hull[0]]).T,color='yellow',label='内点凸包')
            ax.plot(*edge,'r+',markersize=14,markeredgewidth=2,label='输出目标')
            radius = result['endpoint_settings']['patch_radius_px']
            ax.set_xlim(edge[0]-radius-5,edge[0]+radius+5)
            ax.set_ylim(edge[1]+radius+5,edge[1]-radius-5)
            ax.legend(loc='best',fontsize=8)
            ax = axes[7]
            axial_offset = (pixels-edge)@direction
            residual = (fit['sample_predicted_depth_m']-fit['sample_depth_m'])*1000
            for accepted,color,label in [(True,'green','内点'),(False,'magenta','剔除')]:
                ax.scatter(axial_offset[keep==accepted],residual[keep==accepted],s=14,c=color,label=label)
            threshold = result['endpoint_settings']['ransac_threshold_m']*1000
            ax.axhline(0,color='black',lw=.7)
            for sign in [-1,1]: ax.axhline(sign*threshold,color='orange',ls='--')
            ax.axvline(0,color='red',ls='--',label='边缘目标轴向位置')
            q = fit['quality']
            ax.set_xlabel('样本相对目标的轴向偏移 (px)，负值在内部')
            ax.set_ylabel('拟合深度 − 实测深度 (mm)')
            ax.legend(loc='upper right',fontsize=8)
            ax.text(.02,.02,f"{q['model']}\n内点 {q['inlier_samples']}/{q['valid_samples']}\n"
                    f"RMSE {q['fit_rmse_m']*1000:.3f} mm；验证 {q['validation_rmse_m']*1000:.3f} mm\n"
                    f"边缘至内点凸包 {q['depth_extrapolation_px']:.2f} px / 上限 {q['max_depth_extrapolation_px']:.1f} px",
                    transform=ax.transAxes,bbox=dict(facecolor='white',alpha=.9),fontsize=9)
            ax = axes[8]
            pts = fit['sample_points_camera_m'][keep]@transform[:3,:3].T+transform[:3,3]
            point = np.asarray(result['point_base_m'])
            relative = (pts-point)*1000
            ax.scatter(*relative.T,s=5,c='green',label='内部深度样本')
            ax.scatter(0,0,0,s=65,c='red',label='边缘点')
            half = max(np.abs(relative).max()*1.1,5)
            ax.set_xlim(-half,half);ax.set_ylim(-half,half);ax.set_zlim(-half,half);ax.set_box_aspect((1,1,1))
            ax.set_xlabel('ΔX (mm)');ax.set_ylabel('ΔY (mm)');ax.set_zlabel('ΔZ (mm)')
            ax.set_title('⑨ 相机点 → 基座点：目标为局部原点',loc='left',fontweight='bold')
            ax.legend(fontsize=8)
            ax.text2D(.02,.90,f"基座 XYZ (m): {result['point_base_m']}\n"
                      f"目标像素: {np.round(edge,3).tolist()}；拟合 Z: {fit['point_camera_m'][2]*1000:.3f} mm",
                      transform=ax.transAxes,fontsize=8, bbox=dict(facecolor='white', alpha=.85))
        fig.suptitle('插头头/尾边缘中点定位 · 逐步处理汇总\n'
                     f"{Path(result['input']['rgb']).name} | {result['status']} | 红色目标始终位于原始分割边缘\n"
                     '像素边缘准确性取决于分割；内部深度向边缘有限外推；示例基座位姿待实际采集确认',fontsize=15)
        fig.savefig(result['artifacts']['process_summary_png'],dpi=160)
        plt.close(fig)
