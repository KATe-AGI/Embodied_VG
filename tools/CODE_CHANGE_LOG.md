# 代码修改日志

## 2026-05-27 - 抓取点支持 Mask/轴线融合锚点与轴向微调

- 抓取点表面锚点改为由 mask 主体中心和 tail→head 轴线投影中心融合得到，避免关键点偏移直接拖动中段区域。
- 新增 `--grasp-axis-offset-m`，可沿插头 `tail -> head` 轴手动微调最终抓取点。
- `quality.grasp_center_estimation` 增加融合锚点、轴向偏移和最终抓取点诊断字段。

## 2026-05-26 - 抓取点改为鲁棒中段几何中心

- `grasp_pose_camera.translation_m` 不再由单个表面锚点直接偏移得到。
- 新增鲁棒中段点云中心估计，先过滤 mask 边界、head-tail 中段外点和局部深度离群点，再沿 `+z_approach` 偏移半厚度得到抓取点。
- 输出新增 `quality.grasp_center_estimation`，记录候选点数量、过滤数量、离群点原因和表面中心。

## 2026-05-26 - 支持可选窗口约束

- `infer_6d_single.py` 默认可不传窗口几何，直接输出视觉 6D 抓取姿态在机器人基坐标系下的结果。
- 传入 `--window-config` 或 `--window-corners-base` 时启用窗口候选生成，并继续输出 `best_grasp_pose_base`。
- 新增 `grasp_solution_mode` 区分 `direct_visual` 和 `window_constrained`。
- `grasp_point_base_m` 和 `tail_to_head_axis_base` 两种模式都会输出；Tail→Head 轴始终使用视觉参考姿态的 `+X` 方向。

## 2026-05-26 - 统一为窗口约束单帧 6D 位姿方案

- 真实机器 6D 推理入口统一为 `infer_6d_single.py`。
- 窗口几何曾作为必需输入，必须通过 `--window-config` 或 `--window-corners-base` 提供。
- `grasp_pose_base` 保留为视觉估计在机器人基坐标系下的参考姿态，角色为 `surface_normal_reference`。
- 新增 `best_grasp_pose_base` 作为当前方案的最终抓取姿态；该字段等同于排序后的 `window_constrained_grasp_candidates[0]`。
- 删除旧批量 6D 推理入口和旧批量结果验证模块，避免继续维护两套最终姿态语义。

## 2026-05-23 - 纠正抓取帧 Z 轴接近方向

- 抓取帧 `+X` 保持 `tail -> head`。
- 抓取帧 `+Z` 改为从相机可见侧指向物体内部的接近方向。
- 抓取中心补偿沿 `+z_approach` 偏移半个抓取区域厚度。

## 2026-05-21 - 插头 6D 位姿几何参数更新

- 抓取区域长度更新为 `0.085 m`。
- 抓取区域宽度和厚度更新为 `0.055 m`。
- 头尾距离更新为 `0.165 m`。
- 输出保留表面锚点和中心补偿诊断字段，便于排查深度和几何误差。

## 2026-05-21 - 项目代码保守结构化

- 新增 `plug_vg/` 包承载配置、IO、视觉推理、几何计算、抓取位姿估计和机器人坐标转换等可复用代码。
- 顶层训练、验证、阶段一推理和单帧真实机器推理脚本保留为命令行入口。
