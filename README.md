# EmbodiedVG

EmbodiedVG 是插头抓取的视觉侧工程。当前主线使用 YOLO 分割可见插头、在原生 `1920×1080` D2RGB 深度上反投影得到可见点云，再以长轴对称性降维的 CAD profile 和遮挡感知渲染完成配准，最终输出相机系和机器人基座系下的抓取点、头尾语义点与有向长轴。

项目还提供夹爪抓取状态检测：通过 YOLO 检测模型判断夹爪是否抓住插头，支持单张 RGB 图像、离线视频和实时视频流。

## 当前主线

```text
RGB 图像 ──YOLO-seg──> 原始掩膜 ───────────────┐
                         └─5 px 内缩核心掩膜    │
D2RGB ─范围门控/反投影/4 mm 质心体素─> 局部点云 │
                                               │
plugCAD ─轴向/径向距离场─> 鲁棒 5DoF 候选       │
                         └─遮挡感知可见轮廓评分/轴向搜索─> T_camera_grasp
                                                               │
机器人当前位姿 + 手眼标定 ─────────────────────────────────────> T_base_grasp
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
infer_6d_batch.py                  批量 6D 抓取入口
infer_gripper_plug_single.py       图像/视频/实时流夹爪抓取状态检测
infer.py                           YOLO 分割调试入口
train.py                          YOLO 分割/检测训练
val.py                            分割验证
augmentation.yaml                 按任务选择的数据增强策略
plug_vg/                           点云、配准、语义点和坐标变换模块
evaluation/                        相机系真值评估与留一模块比较
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

本项目随仓库保存的 Ultralytics 版本为 `8.4.138`。数据集、模型权重、CAD 资产和训练/推理结果保存在本地，不随 GitHub 仓库分发；在新环境运行前需准备相应资产，或通过命令行参数指定自己的路径。

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
| 抓取 YOLO 分割权重 | `ultralytics/runs/segment/plug_yolo26n_seg_20260814/weights/best.pt` |
| 夹爪状态 YOLO 检测权重 | `ultralytics/runs/detect/plug_yolo26n_det_20261009/weights/best.pt` |
| 插头模型 | `configs/plug_models/plugCAD.yaml` |
| 相机内参 | `configs/camera/plug_rgbd.yaml` |
| 手眼标定 | `hand_eye_calibration/eye_hand_data/calib_20260812_PARK/hand_eye_result_in-hand.yaml` |
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
- `--registration-method symmetric` 使用当前默认方法；`legacy` 保留旧 PCA/环特征/ICP 作为回归对照；
- `--mask-erosion-px` 控制深度反投影前的掩膜内缩半径，默认 `5` 像素；
- `--mad-z-threshold` 只影响 `legacy`；对称主线不使用全局相机 Z 轴 MAD；
- `--save-overlay` 保存可见掩膜叠加图；
- `--save-ply` 保存可见插头点云；
- `--save-review` 保存掩膜、可见点云、RGB 语义点投影、交互 HTML 和点云对比 PLY；
- `--min-registration-fitness` 和 `--max-inlier-rmse` 只控制 `legacy` 的 ICP 质量门。

当前对称主线采用双掩膜：5 px 内缩掩膜提供高置信度 proposal 点，原始掩膜和完整 D2RGB 用于最终可见性评分。观测点只做绝对深度范围过滤和一次 4 mm 体素质心降采样，不使用会误删近光轴有效几何的全局 Z-MAD，也不执行 SOR。配准把绕长轴 roll 作为任务等价自由度，在轴向/径向 CAD 距离场上以鲁棒损失生成有向长轴与抓取中心候选，再用遮挡感知 CAD 可见轮廓和一维轴向搜索统一排序。运行时没有 fitness/RMSE 拒绝门；只要输入有效且存在有限解，就输出最佳几何估计。

## 夹爪抓取状态判断

`infer_gripper_plug_single.py` 使用 YOLO 检测模型判断 RGB 图像或视频中的夹爪状态，不需要深度图、机器人位姿或 CAD 配准。

```bash
conda activate embodiedvg
python infer_gripper_plug_single.py \
  --rgb '检测标注（已标注）_yolo/images/val/v01_s000015_h1_t000015600_f0000468.png' \
  --output-dir output/gripper_plug_single \
  --save-visualization \
  --device 0
```

类别 `0` 输出 `gripper_with_plug`，类别 `1` 输出 `gripper_without_plug`；未检测到夹爪时输出 `no_detection`。终端打印英文状态、类别、置信度、输出目录和模型加载/YOLO/端到端耗时；单图结果 JSON 保存到 `<output-dir>/<图像名>_gripper_plug.json`。指定 `--save-visualization`（或 `--save-overlay`）时，单图额外保存 JPG，视频额外保存带检测结果的 MP4。默认输出目录为 `ultralytics/runs/gripper_plug_single`，默认权重为 `ultralytics/runs/detect/plug_yolo26n_det_20261009/weights/best.pt`；可通过 `--weights` 和 `--conf` 覆盖。

视频输入会复用同一个 YOLO 模型逐帧推理，结果 JSON 中包含每个已处理帧的状态、类别、置信度和时间戳。`--frame-stride 2` 可每隔一帧处理一次，减少推理量，同时保持可视化视频的原始帧率：

```bash
python infer_gripper_plug_single.py \
  --video input/gripper.mp4 \
  --output-dir output/gripper_plug_video \
  --frame-stride 1 \
  --save-overlay \
  --device 0
```

处理离线视频时，终端会显示 `tqdm` 帧进度。实时输入使用 `--camera`（或别名 `--stream`），参数可以是摄像头编号、设备路径或 RTSP/HTTP 地址。指定 `--show` 可打开实时检测窗口，按 `q` 退出；不打开窗口时可用 `Ctrl+C` 退出：

```bash
python infer_gripper_plug_single.py \
  --camera 0 \
  --output-dir output/gripper_plug_live \
  --show \
  --device 0
```

实时模式会定期在终端刷新当前帧的状态、类别和置信度，并在退出后保存汇总 JSON。需要限制测试时长可使用 `--max-frames N`；指定 `--save-overlay` 会把实时结果保存为 `live_gripper_plug.mp4`。

网络摄像头可使用 `--stream 'rtsp://camera-ip/stream'`。实时 JSON 保存为 `live_gripper_plug.json`，包含类别统计和最后一帧的结果；离线 JSON 保存全部已处理帧的结果。模型在每次运行中只加载一次，首次预测包含初始化/预热开销，终端的 YOLO 耗时包含预测调用的预处理与后处理。

## 检测数据集与训练

检测标注使用 LabelMe 矩形，标签 `0` 为 `gripper_with_plug`、`1` 为 `gripper_without_plug`。将原始数据放到 `检测标注（已标注）` 的各子目录后执行：

```bash
python tools/convert_gripper_detection.py \
  --source '检测标注（已标注）' \
  --output '检测标注（已标注）_yolo'
```

转换器仅纳入有对应 JSON 的图像；空标注作为背景样本。按类别分层进行 8:2 划分，默认随机种子为 `42`，输出包含 `images/train`、`images/val`、`labels/train`、`labels/val`、`data.yaml` 和划分清单。工具详情见 `tools/DATASET_TOOLS.md`。

使用 YOLO26n 检测模型训练：

```bash
python train.py \
  --task detect \
  --model yolo26n.pt \
  --data '检测标注（已标注）_yolo/data.yaml' \
  --epochs 150 \
  --project ultralytics/runs/detect \
  --name plug_yolo26n_det_20261009
```

`train.py` 根据 `--task` 从同目录的 `augmentation.yaml` 加载策略。检测策略采用 YOLO 默认增强并增加上下翻转（`flipud=0.5`）。默认 `imgsz=640`、`batch=32`、`workers=8`、`patience=10`；150 为 epoch 上限，连续 10 个 epoch 无提升时提前结束。需要完整运行 150 个 epoch 可设置 `--patience 0`。脚本默认任务仍为分割，检测训练需显式指定上面的模型、数据和输出目录。

本地本次数据集包含 402 张训练图像、100 张验证图像，训练在第 49 个 epoch 早停，最佳权重来自第 39 个 epoch。最佳模型在此验证集上 P/R/mAP50/mAP50–95 为 `0.954/1.000/0.995/0.969`；这些指标代表本次划分的结果。

## 输出产物

对于输入 `<sample>_color.png`，主要产物为：

| 文件 | 生成条件 | 含义 |
|---|---|---|
| `<sample>_color_6d_base.json` | 始终 | 状态、质量、语义点、位姿、变换链与耗时 |
| `<sample>_color_visible_mask.jpg` | `--save-overlay` 或 `--save-review` | YOLO 可见掩膜 |
| `<sample>_color_visible_points.ply` | `--save-ply` 或 `--save-review` | RGB 相机系下的目标可见点云 |
| `<sample>_color_rgb_projection.jpg` | `--save-review` 且存在候选变换 | 掩膜与 tail/grasp/head 的 RGB 投影 |
| `<sample>_color_registration_review.html` | `--save-review` | 全场景、可见点、CAD、包围盒、相机视锥与抓取轴的交互复核 |
| `<sample>_color_point_cloud_comparison.ply` | `--save-review` | RGB 相机系下的对比点云：绿色为 D2RGB 观测，红色为已配准 CAD；黄色、青色、紫色点球分别标记抓取中心、尾部中心、头部中心 |

PLY 只保存点和颜色，不能控制 IDE 插件的拖拽、坐标轴、背景或小窗口。使用 `kleinicke.ply-visualizer` 时，拖拽时出现的坐标轴原点是插件的当前旋转中心，不是 CAD 抓取原点；按 `W` 可将旋转中心设为 PLY 世界原点，按 `A` 切换坐标轴常驻显示。

JSON 的 `status` 可为：

- `ok`：得到有限最佳估计，并输出相机系与基座系位姿；
- `failed`：输入缺失、分割/深度/点数无效，或没有有限配准解；
- `ambiguous`：仅旧 `legacy` 方法可能返回的兼容状态。

只有 `status=ok` 时的 `grasp_pose_base` 才是可供下游使用的候选。主要 JSON 字段包括 `t_camera_grasp`、`grasp_pose_camera`、`semantic_points_camera`、`semantic_points_base`、`grasp_pose_base`、`tail_to_head_axis_base`、`registration_quality`、`artifacts` 和 `timing`。

## plugCAD 实际尺寸模板

`plug_model/` 已加入 `.gitignore`，其中的 STEP、OBJ、PLY、尺寸对比图和审核报告只保留在本地，不随 GitHub 仓库分发。新环境需要从受控存储中单独取得该目录，并保持下列文件路径不变，否则默认 `plugCAD` 配置无法完成配准。

- `plug_model/2175B.stp`：原始供应商数模，总长 `180.3 mm`，保持不变；
- `plug_model/plugCAD.stp`：当前修正尺寸数模，轴向 S1–S8 为 `29.3/1/8.5/50/21.7/23/22.3/12 mm`，总长 `167.8 mm`；
- 尾部 S1 为 `29.3 mm`，尾端到大圆环近侧（S1–S5）为 `110.5 mm`，大圆环 S6 为 `23 mm`；
- 仅沿尾到头的长轴做分段线性适配，两个横向坐标保持不变；
- 新抓取原点为距头部端点 `90 mm` 的任务指定位置，距尾部 `77.8 mm`；
- 抓取系语义点为 `tail=[-0.0778,0,0] m`、`grasp=[0,0,0] m`、`head=[0.09,0,0] m`。

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

相机系真值评估及与旧模块的逐帧留一比较：

```bash
python evaluation/evaluate_camera_pose.py \
  --gt-dir test_20260814 \
  --reference-dir output/test_0824 \
  --prediction-dir output/test_symmetric_final_batch \
  --output-json output/test_symmetric_final_batch/evaluation_with_loo.json \
  --output-csv output/test_symmetric_final_batch/evaluation.csv
```

主指标是相机系 `grasp/tail/head` 三点的全局 RMSE；绕长轴 roll 不计入任务误差。测试覆盖实际尺寸 CAD 映射、抓取坐标系语义、深度反投影、对称 profile、相机系评估、PCA/ICP 回归对照和复核产物。

## 已知限制

- `configs/robot/cs_robot.yaml` 中的实时机器人位姿提供器尚未实现，当前必须显式传入 `--robot-pose`。
- 单固定视角、近圆形端面和严重遮挡仍可能使有向长轴欠约束；`status=ok` 表示最佳有限估计，不代表低误差保证。
- 当前 12 帧相机系人工真值只覆盖 `test_20260814`，上线前仍需扩大独立工况测试集，并继续检查 RGB 投影、交互 HTML 和红绿 PLY。
- 没有逐帧真实机器人位姿时，重用同一 `--robot-pose` 只能用于相机系配准复核，不能用于声称基座系绝对精度。
