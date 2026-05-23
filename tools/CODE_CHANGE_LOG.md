# 代码修改日志

## 2026-05-23 - 移除旧版 grasp_3d.py 入口

### 背景

`grasp_3d.py` 原本用于读取已经保存好的 stage1 JSON，再结合 D2RGB 深度图输出相机坐标系下的 3D 抓取位姿。
当前核心几何逻辑已经下沉到 `plug_vg/grasp_pose.py`，端到端入口也已经由 `infer_6d.py` 和 `infer_6d_single.py` 覆盖。

### 修改

- 删除根目录旧入口 `grasp_3d.py`。
- 保留 `plug_vg/grasp_pose.py`、`plug_vg/geometry.py`、`plug_vg/io.py` 中的可复用实现。
- 后续批量 RGB-D 推理使用 `infer_6d.py`。
- 后续真实机器单帧验证使用 `infer_6d_single.py`。

## 2026-05-21 - 插头 6D 位姿几何参数更新

### 背景

更新 6D 抓取位姿估计流程中使用的插头物理模型。
当前将插头中段近似看作一个圆柱形抓取区域。

### 修改前

- 插头物理参数在 `grasp_3d.py` 中设置为：
  - 抓取区域长度：`0.070 m`
  - 抓取区域宽度：`0.040 m`
  - 头尾距离：`0.160 m`
  - 没有单独建模抓取区域厚度。

- 抓取关键点的计算方式：
  - 先在 mask 内沿 `plug_tail -> plug_head` 方向取中段区域。
  - 取该中段区域的二维中位点作为 `grasp_target_2d`。
  - 在 D2RGB 深度图中查询该点深度，并反投影得到 `grasp_target_3d`。
  - 最终 `grasp_pose_camera.translation_m` 直接使用这个 `grasp_target_3d`。

- 因为 D2RGB 深度来自相机看到的表面，所以修改前输出的抓取平移点本质上是“可见表面点”，不是插头中段的几何中心。

### 修改后

- 插头物理参数在 `grasp_3d.py` 中更新为：
  - 抓取区域长度：`0.085 m`
  - 抓取区域宽度：`0.055 m`
  - 抓取区域厚度：`0.055 m`
  - 头尾距离：`0.165 m`

- 抓取区域厚度的设置：
  - 由于插头中段近似为圆柱体，抓取区域厚度近似取为抓取区域宽度。
  - 即 `GRASP_REGION_THICKNESS_M = GRASP_REGION_WIDTH_M = 0.055 m`。

- 抓取关键点的计算方式：
  - 仍然先按原方法从 D2RGB 得到可见表面锚点。
  - 该表面点保存在 `quality.grasp_surface_anchor_camera_m`。
  - 然后沿 `-z_approach` 方向向物体内部偏移半个厚度。
  - 偏移距离为 `0.055 / 2 = 0.0275 m`。
  - 最终 `grasp_pose_camera.translation_m` 使用偏移后的圆柱中段中心点。

- 在 3D JSON 输出中新增诊断字段：
  - `quality.grasp_surface_anchor_camera_m`
  - `quality.grasp_center_adjustment`
  - `quality.grasp_region_expected_m.thickness_z_m`

- 同步更新 `verify_6d_outputs.py`，使验证逻辑与新的物理参数和中心点平移定义保持一致。

### 主要影响

- 修改前：输出位姿的平移点落在相机可见表面上，更像是“表面抓取锚点”。
- 修改后：输出位姿的平移点更接近插头中段圆柱体的中心，更适合作为平行夹爪的抓取中心/TCP 目标点。
- 旋转矩阵的计算逻辑没有改变，仍然使用：
  - X 轴：`tail -> head`
  - Z 轴：抓取区域表面法向，朝向相机侧
  - Y 轴：由叉乘补齐

### 说明

- `infer_6d.py` 仍然只作为端到端编排脚本使用。当前会调用 `plug_vg.grasp_pose.estimate_record()`，因此会自动使用更新后的物理模型。
- 没有在 `infer_6d.py` 中新增额外运行功能或额外配置输出。
- 现有旧推理 JSON 是基于旧的表面点平移定义生成的。若要按照新的中心点定义验证结果，需要先重新运行 `infer_6d.py`，再运行 `verify_6d_outputs.py`。

## 2026-05-21 - 项目代码保守结构化

### 背景

原项目主要由多个根目录脚本组成。`infer.py`、`grasp_3d.py`、`infer_6d.py`、`verify_6d_outputs.py` 之间存在脚本互相导入和几何逻辑重复的问题，后续维护时容易出现参数或实现不一致。

### 结构化前

- 可复用逻辑直接写在顶层脚本中：
  - `infer.py` 同时负责 YOLO 推理、结果序列化和 2D 可视化。
  - `grasp_3d.py` 同时负责相机配置、深度反投影、旋转估计、3D JSON 输出和 CLI。
  - `verify_6d_outputs.py` 复制了一份几何计算逻辑，用于独立验证 6D 输出。
  - `infer_6d.py` 通过导入 `infer.py` 和 `grasp_3d.py` 复用功能。

- 主要问题：
  - 物理参数和几何函数容易在多个脚本中漂移。
  - 顶层脚本既是命令行入口，又承担库函数职责。
  - 验证逻辑和推理逻辑存在重复实现。

### 结构化后

- 新增 `plug_vg/` 包，用于承载可复用代码：
  - `plug_vg/config.py`：项目路径、默认权重、相机配置加载、插头物理参数。
  - `plug_vg/io.py`：JSON 读写、source 收集、manifest 读取、stage1 record 收集、raw_id 解析。
  - `plug_vg/vision.py`：YOLO stage1 结果序列化、stage1 overlay、模型推理封装。
  - `plug_vg/geometry.py`：mask、深度反投影、点云过滤、PCA/RANSAC、旋转矩阵、四元数、投影、PLY 和 3D overlay。
  - `plug_vg/grasp_pose.py`：从 stage1 record 和 D2RGB 深度图估计 6D 抓取位姿。
  - `plug_vg/verification.py`：复核已有 6D 输出并生成报告。

- 当时顶层脚本保留原名称和原 CLI 参数：
  - `infer.py`
  - `grasp_3d.py`
  - `infer_6d.py`
  - `verify_6d_outputs.py`

- 顶层脚本主要作为轻量命令行入口使用，核心逻辑下沉到 `plug_vg/`。
- 2026-05-23 起，旧版 `grasp_3d.py` 入口已删除；历史的两阶段 stage1 JSON 流程可直接复用 `plug_vg.grasp_pose.estimate_record()`。

### 主要影响

- 结构化当时旧命令仍然可用，例如：
  - `python infer.py --help`
  - `python infer_6d.py --help`
  - `python verify_6d_outputs.py --help`

- 输出 JSON 结构保持不变。
- 数据集转换、合并、训练和验证脚本暂时没有纳入本轮结构化，避免一次改动范围过大。
- 后续新增手眼标定、机器人坐标转换或其他 6D 后处理时，建议优先放到 `plug_vg/` 中，而不是继续堆在根目录脚本里。
