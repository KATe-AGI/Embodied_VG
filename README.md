# EmbodiedVG

EmbodiedVG 是插头抓取的视觉侧工程。当前主线使用 YOLO 分割可见插头、在 D2RGB 深度上反投影得到可见点云，再与 `plugCAD` 抓取坐标系模板进行 PCA/ICP 配准，最终输出相机系和机器人基座系下的抓取点、头尾语义点与有向长轴。

## 当前主线

```text
RGB 图像 ──YOLO-seg──> 可见插头掩膜 ──边界腐蚀──> 深度掩膜
                            │
D2RGB 深度 ──范围门控/全局 MAD/反投影/质心体素──> 相机系可见点云
                            │
plugCAD 抓取系点云 ─PCA/ICP──> T_camera_grasp
                            │
机器人当前位姿 + 手眼标定 ──> T_base_grasp
```

配准只使用 YOLO 掩膜内的 D2RGB 点，不需要额外的场景 PLY。

## 任务语义与坐标系

本项目的核心任务不是恢复轴对称插头的唯一完整旋转，而是稳定估计：

- `grasp_center`：抓取点，同时是 CAD 抓取坐标系原点；
- `tail_center`：尾部端面中心；
- `head_center`：头部端面中心；
- `+X`：从 `tail_center` 指向 `head_center`；
- `+Y`：夹爪闭合方向；
- `+Z`：满足右手系的接近方向。

绕正确 X 轴的 roll 对当前抓取任务是等价自由度；仅 roll 不同不应判为配准失败。但长轴方向错误、头尾颠倒、沿轴错位或抓取中心偏离仍是真实失败。

基座系变换链为：

```text
T_base_grasp = T_base_end_current @ T_end_camera @ T_camera_grasp
```

`--robot-pose` 按 `[x, y, z, roll, pitch, yaw]` 传入，平移单位为米，角度单位为弧度，采用固定轴 XYZ RPY：`R = Rz(yaw) @ Ry(pitch) @ Rx(roll)`。

## 项目结构

```text
infer_6d_single.py                 单帧 6D 抓取主入口
infer.py                           YOLO 分割调试入口
train.py / val.py                  分割训练与验证
plug_vg/                           点云、配准、语义点和坐标变换模块
configs/camera/plug_rgbd.yaml      RGB/D2RGB 相机内参
configs/plug_models/plugCAD.yaml   当前默认抓取模板
configs/plug_models/2175B.yaml     原始供应商模型配置
configs/robot/cs_robot.yaml        机器人位姿约定；实时接口尚未实现
plug_model/                        本地私有 CAD 资产目录（不上传 GitHub）
hand_eye_calibration/              手眼标定工具与结果
tools/                             数据集、CAD 资产和复核工具
tests/                             单元测试
```

数据集转换与合并说明见 `tools/DATASET_TOOLS.md`，手眼标定说明见 `hand_eye_calibration/README.md`。

## 环境安装

推荐 Python 3.11，环境名固定为 `embodiedvg`：

```bash
conda create -n embodiedvg python=3.11 -y
conda activate embodiedvg
```

PyTorch 需要根据实际 CUDA/CPU 环境安装。例如 NVIDIA CUDA 12.8：

```bash
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
```

然后在项目根目录安装其余依赖：

```bash
pip install -r requirement.txt
```

`requirement.txt` 会以 editable 方式安装项目内的 `./ultralytics`，因此迁移项目时必须保留该目录。日常推理不需要 CadQuery；只有重建 `plugCAD.stp` 时才需要额外提供 `cadquery` Python 包。

### 安装验证

Ubuntu / bash：

```bash
conda activate embodiedvg
python -m py_compile infer_6d_single.py plug_vg/*.py
python infer_6d_single.py --help
python -m unittest discover -s tests
```

Windows PowerShell：

```powershell
conda activate embodiedvg
python -m py_compile infer_6d_single.py plug_vg\grasp_model.py plug_vg\model_registration.py plug_vg\visible_points.py
python .\infer_6d_single.py --help
python -m unittest discover -s tests
```

## 默认运行资产

| 用途 | 默认文件 |
|---|---|
| YOLO 分割权重 | `ultralytics/runs/segment/plug_yolo26s_seg_20260529-2/weights/best.pt` |
| 插头模型 | `configs/plug_models/plugCAD.yaml` |
| 相机内参 | `configs/camera/plug_rgbd.yaml` |
| 手眼标定 | `hand_eye_calibration/eye_hand_data/calib_20260618/hand_eye_result_in-hand.yaml` |
| 机器人约定 | `configs/robot/cs_robot.yaml` |

当前 D2RGB 配置为 `1920 x 1080`，深度单位缩放为 `0.001 m`；输入支持项目已使用的 D2RGB PNG/NPY 格式。

## 单帧 6D 推理

### Ubuntu / bash

```bash
conda activate embodiedvg
python infer_6d_single.py \
  --rgb test_20260701/20260701_155018_359_color.png \
  --d2rgb test_20260701/20260701_155018_359_d2rgb.npy \
  --robot-pose -0.712547 0.000064 0.581025 -2.279 0.216 1.488 \
  --output-dir output/plug_6d_single \
  --model-config configs/plug_models/plugCAD.yaml \
  --save-ply \
  --save-review
```

### Windows PowerShell

```powershell
conda activate embodiedvg
python .\infer_6d_single.py `
  --rgb .\test_20260701\20260701_155018_359_color.png `
  --d2rgb .\test_20260701\20260701_155018_359_d2rgb.npy `
  --robot-pose -0.712547 0.000064 0.581025 -2.279 0.216 1.488 `
  --output-dir .\output\plug_6d_single `
  --model-config .\configs\plug_models\plugCAD.yaml `
  --save-ply `
  --save-review
```

PowerShell 的换行符是反引号 `` ` ``，反引号后不要再放空格。

常用参数：

- `--device 0` 指定 CUDA GPU，`--device cpu` 强制 CPU；
- `--mask-erosion-px` 控制深度反投影前的掩膜内缩半径，默认 `5` 像素；
- `--mad-z-threshold` 控制直通滤波后的全局相机 Z 轴 MAD 阈值，默认 `3.5`；
- `--save-overlay` 保存可见掩膜叠加图；
- `--save-ply` 保存可见插头点云；
- `--save-review` 保存掩膜、可见点云、RGB 语义点投影、交互 HTML 和点云对比 PLY；
- `--min-registration-fitness` 和 `--max-inlier-rmse` 控制当前 ICP 质量门。

当前观测点预处理采用“绝对深度直通滤波 + 全局相机 Z 轴 MAD”：掩膜默认向内腐蚀 5 像素，先保留绝对深度范围内的点，再以 `median(Z) ± 3.5 × 1.4826 × MAD(Z)` 剔除全局深度异常点，随后只进行一次 4 mm 体素质心降采样。配准入口不再执行 Open3D 统计离群过滤，也不重复体素化场景点云。

## 输出产物

对于输入 `<sample>_color.png`，主要产物为：

| 文件 | 生成条件 | 含义 |
|---|---|---|
| `<sample>_color_6d_base.json` | 始终 | 状态、质量、语义点、位姿、变换链与耗时 |
| `<sample>_color_visible_mask.jpg` | `--save-overlay` 或 `--save-review` | YOLO 可见掩膜 |
| `<sample>_color_visible_points.ply` | `--save-ply` 或 `--save-review` | RGB 相机系下的目标可见点云（仅 XYZ） |
| `<sample>_color_rgb_projection.jpg` | `--save-review` 且存在候选变换 | 掩膜与 tail/grasp/head 的 RGB 投影 |
| `<sample>_color_registration_review.html` | `--save-review` | 全场景、可见点、CAD、包围盒、相机视锥与抓取轴的交互复核 |
| `<sample>_color_point_cloud_comparison.ply` | `--save-review` | RGB 相机系下的对比点云：绿色为 D2RGB 观测，红色为已配准 CAD；黄色、青色、紫色点球分别标记抓取中心、尾部中心、头部中心 |

PLY 只保存点和颜色，不能控制 IDE 插件的拖拽、坐标轴、背景或小窗口。使用 `kleinicke.ply-visualizer` 时，拖拽时出现的坐标轴原点是插件的当前旋转中心，不是 CAD 抓取原点；按 `W` 可将旋转中心设为 PLY 世界原点，按 `A` 切换坐标轴常驻显示。

JSON 的 `status` 可为：

- `ok`：通过当前 fitness/RMSE 门控，并输出相机系与基座系位姿；
- `failed`：分割、深度、点数或配准质量不合格；
- `ambiguous`：存在无法区分的头尾反向候选。

只有 `status=ok` 时的 `grasp_pose_base` 才是可供下游使用的候选。主要 JSON 字段包括 `t_camera_grasp`、`grasp_pose_camera`、`semantic_points_camera`、`semantic_points_base`、`grasp_pose_base`、`tail_to_head_axis_base`、`registration_quality`、`artifacts` 和 `timing`。

## plugCAD 实际尺寸模板

`plug_model/` 已加入 `.gitignore`，其中的 STEP、OBJ、PLY、尺寸对比图和审核报告只保留在本地，不随 GitHub 仓库分发。新环境需要从受控存储中单独取得该目录，并保持下列文件路径不变，否则默认 `plugCAD` 配置无法完成配准。

- `plug_model/2175B.stp`：原始供应商数模，总长 `180.3 mm`，保持不变；
- `plug_model/plugCAD.stp`：当前修正尺寸数模，轴向 S1–S8 为 `29.3/1/8.5/50/21.7/23/22.3/12 mm`，总长 `167.8 mm`；
- 尾部 S1 为 `29.3 mm`，尾端到大圆环近侧（S1–S5）为 `110.5 mm`，大圆环 S6 为 `23 mm`；
- 仅沿尾到头的长轴做分段线性适配，两个横向坐标保持不变；
- 新抓取原点是原 `2175B` 稳定可抓取圆柱截面经同一轴向映射后的位置，距尾部 `67.011979 mm`；
- 抓取系语义点为 `tail=[-0.06701198,0,0] m`、`grasp=[0,0,0] m`、`head=[0.10078802,0,0] m`。

详细尺寸和映射见 `plug_model/plugCAD_grasp_report.md` 与 `plug_model/2175B_vs_plugCAD_dimensions.png`。

如需重建 STEP、OBJ、PLY、YAML 和尺寸对比图：

```bash
conda activate embodiedvg
python tools/build_actual_plug_cad.py
```

该命令会覆盖当前 `plugCAD` 资产，只应在确认尺寸映射后执行。

## 分割调试、训练与验证

分割调试：

```bash
python infer.py --source test_20260703 --output ultralytics/runs/plug_stage1_seg_debug
```

训练：

```bash
python train.py --device 0
```

验证：

```bash
python val.py --device 0
```

可通过 `--model`、`--data`、`--weights`、`--batch`、`--imgsz` 和 `--device` 等参数覆盖默认训练/验证配置。

## 测试

```bash
conda activate embodiedvg
python -m unittest discover -s tests
python -m py_compile infer_6d_single.py plug_vg/*.py tools/*.py
```

测试覆盖实际尺寸 CAD 映射、抓取坐标系语义、深度反投影、可见点云、PCA/ICP 候选选择、质量门控和复核产物。

## 已知限制

- `configs/robot/cs_robot.yaml` 中的实时机器人位姿提供器尚未实现，当前必须显式传入 `--robot-pose`。
- 单目视角的局部可见点云可能让 PCA 长轴退化；单向局部 ICP 也可能对错误姿态给出高 fitness。
- `status=ok` 只表示通过当前数值门控，不代表拥有 3D 真值验证；现阶段仍需同时检查 RGB 投影、交互 HTML 和红绿 PLY。
- 面对相机的近圆形端面或严重遮挡场景尤其容易出现长轴误判。
- 没有逐帧真实机器人位姿时，重用同一 `--robot-pose` 只能用于相机系配准复核，不能用于声称基座系绝对精度。
