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
infer_apriltag_cylinder.py         RGB AprilTag 中心与朝向相机的法向量，自动转换基座系
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

## AprilTag 参考点、预设点与有向法向量

采集图像、手眼标定结果、STEP 模型、模型权重及推理输出仅保存在本地，不随代码提交。新环境需自行准备对应文件，或通过脚本参数指定本地路径；下方示例使用的图片及默认手眼标定路径不是仓库内附带的数据。

`infer_apriltag_cylinder.py` 保留原入口文件名，现改为纯 RGB 的中心与法向量估计。检测 `tagStandard41h12 / ID 0` 后，将原始四角点及单应矩阵传入迁入的官方 AprilRobotics `estimate_tag_pose()`，得到 `R_camera_tag` 和 `t_camera_tag`。中心为 `P_camera=t`；法向量为归一化的 `R[:,2]`，当 `dot(n,-P)<0` 时翻转，始终指向相机光心所在半空间。输出使用标记几何中心，而不是四个边角之一。

```bash
conda activate embodiedvg
pip install -r requirement.txt
# 首次运行需 C 编译器；Ubuntu 未安装时运行 sudo apt install build-essential
python infer_apriltag_cylinder.py \
  --rgb camera_data/20261010_144623_467_color.png \
  --tagsize 0.018 \
  --robot-pose -0.014293 0.460711 0.742759 2.167158 0.044541 -3.126827 \
  --output-dir output/apriltag_pose \
  --show
```

相机内参和手眼标定默认读取已有配置，无需特意传参。上述机器人位姿用于示例流程；实际定位请替换为拍摄时的末端位姿（米、弧度）。当前机器人位姿实时接口尚未实现，所以基座系转换仍需 `--robot-pose`。计算自动完成：

```text
T_base_camera = T_base_end @ T_end_camera
P_base = R_base_camera @ P_camera + t_base_camera
n_base = R_base_camera @ n_camera
```

法向量是方向，只乘旋转、不加平移，并保持单位长度；它在基座系满足 `dot(n_base, camera_origin_base-P_base)>=0`。JSON 同时输出两种坐标系的中心与法向量，`center_normal_base` 明确提供最终六个数 `center_x/y/z`（米）及 `normal_x/y/z`（无量纲单位向量）。

固定偏移在 [configs/placement/plug_placement.yaml](configs/placement/plug_placement.yaml) 统一管理，使用米。可用 `--placement-config` 指定其他配置。脚本同时输出 `placement_points_base_m`：

```text
reference   = P_base
preset      = reference + (0.200 + 0.020) * n_base
head_preset = preset + [0, 0,  0.080]
tail_preset = preset + [0, 0, -0.087]
```

200 mm 是沿法向的轴线偏移，额外 20 mm 是轴线到侧表面的偏移；两者保留为独立配置项。头尾偏移沿固定基座 Z 轴，不沿标记局部坐标。`approach_direction_base=-n_base` 给出接近方向。JSON 中 `placement_offsets` 记录使用的数值；HTML/PNG 标注四个点、220/80/87 mm 偏移，RGB 叠加可见的虚拟预设点。预设点只表示几何目标，不输出机械臂运动或夹爪动作。

`--tagsize` 可显式设置实际尺寸，默认 `0.018 m`（实测 18 mm），表示检测器四角所围成的蓝框：**5×5 白色定位环的外边界**。此尺寸直接传入官方 `estimate_tag_pose()`，无需乘 `5/9`。完整 9×9 编码图案的推导边长为 `0.018×9/5=0.0324 m`；灰色打印参考框不参与位姿尺寸定义。旧参数 `--tag-size` 已移除。纯 RGB 距离直接依赖实际打印尺寸，应按实物测量设置；测量蓝框不代表可以沿蓝框裁剪，完整图案及周围留白仍需保留。输入应为当前无畸变 RGB 或已校正图像，分辨率需与内参匹配。

旧 D2RGB 定位、深度回退、基座 −Y 偏移和 CAD 圆柱包络已删除。检测依旧使用 `pupil-apriltags`，仅取 ID/四角/单应矩阵，不调用其位姿接口。官方位姿与数学依赖的源码子集放在 `third_party/apriltag_pose/`，没有克隆整个官方仓库，算法源码保持原样并保留许可证与固定提交号；首次运行编译共享库，之后复用缓存。来源及构建说明见 [third_party/apriltag_pose/README.md](third_party/apriltag_pose/README.md)。

每张图像始终生成四个文件：

- `*_apriltag_pose.json`：检测信息、官方位姿误差、相机系/基座系的中心与单位法向量、变换矩阵。
- `*_overlay.jpg`：RGB 上的中心（绿色）、朝向相机的法向量箭头（橙色），两种坐标系的数值。
- `*_base_view.html`：RGB 标注图与标记平面、参考/预设/头尾预设点、相机光心、有向法向量及基座 XYZ 的交互三维视图。图片和 Plotly 均嵌入 HTML，离线可打开，也可复制到其他电脑查看。
- `*_base_view.png`：基座系三维静态预览。

箭头长度仅用于可视化，输出法向量始终为单位向量；正对相机时箭头在 RGB 上可能投影为很短的线，三维视图仍可见其方向。`--show` 在桌面会话中尝试用浏览器打开包含 RGB 标注与三维视图的 HTML，不再调用 OpenCV/Qt 窗口。SSH 或无桌面 Linux 会话（没有 `DISPLAY`、`WAYLAND_DISPLAY`）仍正常完成计算与文件保存，并输出 HTML 路径；可在有浏览器的电脑上打开该文件。浏览器启动失败也不影响已计算的结果。未检测到目标 ID、存在多个同 ID 标记、位姿无效或分辨率不匹配时，保存 `status=failed` JSON 并以非零状态退出。

## 单帧插头头尾侧表面端部点定位

`infer_plug_endpoint_single.py` 独立处理夹持插头的照片，不读取 AprilTag 结果，也不计算运动修正量。默认使用 `ultralytics/runs/segment/plug_yolo26n_seg_20260904/weights/best.pt` 的 `plug_head` / `plug_tail` 类别，只定位当前可见端。输入接口与 AprilTag 脚本对齐，并增加对齐到 RGB 的原始深度：

```bash
python infer_plug_endpoint_single.py \
  --rgb test_gt_20260902/20260902_133202_557_color.png \
  --d2rgb test_gt_20260902/20260902_133202_557_d2rgb.npy \
  --robot-pose -0.014293 0.460711 0.742759 2.167158 0.044541 -3.126827 \
  --output-dir output/plug_endpoint \
  --show
```

示例法兰位姿仅用于跑通流程，不能作为这些历史照片的真实拍摄位姿。RGB 与 D2RGB 必须是同一拍摄时刻、相同像素网格，默认 1920×1080。深度 PNG/NPY 保留传感器原始数值，由相机配置中的 `depth_scale` 转为米；不能传入深度彩色可视化。实际使用时将 `--robot-pose` 替换为曝光时刻的法兰位姿（米、固定轴 XYZ RPY 弧度），不是含工具偏移的 TCP 位姿。相机内参、手眼标定和偏移/定位配置均有默认值，分别可用 `--camera-config`、`--hand-eye-config`、`--placement-config` 覆盖。`--seg-weights`、`--conf`、`--imgsz`、`--device` 可调整分割推理设置。

点位定义为**靠相机侧表面的头端或尾端边缘中点，允许向插头表面内部退让**，不是端面圆心、插头轴心或 TCP 中心。处理步骤：

1. 保留原始分割类别和全分辨率掩膜；要求唯一目标实例，拒绝多实例歧义。
2. 使用“尾到头近似基座 +Z”的先验，通过拍摄时相机姿态投影确定头尾方向。只在该方向的端部区域估计两侧中间位置，避免大圆环拉偏整个掩膜质心。明显细长且方向冲突的掩膜会被拒绝；短尾或宽圆环区域仅依赖 +Z 先验，JSON 中 `localization.axis_check` 明确记录。
3. 将目标沿图像中的轴向向内退让，默认 3 px，最多 12 px；同时要求距原掩膜边界至少 3 px。`localization.edge_pixel`、`target_pixel`、`inset_actual_px` 记录边缘估计和实际退让。
4. 采样仅来自目标附近圆形邻域与向内腐蚀掩膜的交集。默认半径 14 px、掩膜腐蚀 2 px；排除无效深度，并用 RANSAC 剔除不符合局部表面的深度。限制在掩膜内部降低背景污染，但分割错误或深度传感器边界混合仍需实物验证。
5. 比较逆深度平面与二次曲面模型，用留出的空间采样点验证预测误差；曲面明显改善验证误差时才采用。目标必须处于有效内点的凸包中，避免在深度不足时外推。拟合深度后，沿目标像素射线反投影 XYZ，再通过法兰位姿和手眼变换得到基座 XYZ。

输出 `*_plug_endpoint.json` 中的 `class_name`、`point_camera_m`、`point_base_m`、`localization`、`quality` 和变换矩阵。JSON 的 `status=ok / valid=true` 表示通过当前分割和数值拟合检查，**不表示已经实测达到 3 mm 精度或已验证端部语义点**；相应字段为 `physical_accuracy_verified=false`、`semantic_endpoint_verified=false`。局部拟合残差不包含分割边缘误差、退让偏差、手眼标定和法兰位姿误差。

同时生成 `*_endpoint_overlay.jpg` 和可离线查看的 `*_endpoint_review.html`，显示全图、端部局部放大和基座系局部点云。青色为分割边界，绿色线为腐蚀边界，黄色标记为估计边缘中点，红色十字为实际退让后的目标，绿色点为深度内点，紫色点为剔除样本。`--show` 使用浏览器，无桌面时只打印路径。分割、端部定位或拟合失败时保存失败 JSON；已读入图像且完成分割推理的失败也保留可视化，退出码为 1。

这一版只处理单帧，不进行实时跟踪、机械臂控制或松夹爪。3 mm 为待验证的物理误差目标。退让参数改变目标点的物理位置，验证其与头尾预设点的对应关系时应固定点位定义；不自动把退让后的点补偿回严格边缘。

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
