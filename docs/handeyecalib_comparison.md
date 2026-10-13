# HandEyeCalib 标记、坐标与采样流程核验

后续变更：本报告记录重构前的 AprilTag 脚本。该脚本现已改为迁入官方 `estimate_tag_pose()` 的纯 RGB 中心与朝向相机法向量估计；深度定位与圆柱偏移分支已移除。尺寸参数现为 `--tagsize`，直接表示蓝框（5×5 定位框）边长，默认实测 18 mm，无需乘 `5/9`。第 6–8 节的旧实现及尺寸描述保留为调查时的记录，现行行为见项目 README“AprilTag 中心与有向法向量”。HandEyeCalib 使用 ChArUco、其外参接口可供本项目使用的结论仍适用。

核验日期：2026-10-11。来源为 [KATe-AGI/HandEyeCalib](https://github.com/KATe-AGI/HandEyeCalib)，固定提交 [`4a9c8e302e3ee9bd23e83964d4baec3fe4aa03c3`](https://github.com/KATe-AGI/HandEyeCalib/tree/4a9c8e302e3ee9bd23e83964d4baec3fe4aa03c3)。以下结论来自该提交的 README、依赖、配置、实际源码及同图检测对照；未连接机器人或相机、未执行机器人运动或完整手眼标定，也未验证实物打印尺寸与标定精度。文档中的标记配置是该仓库的配置事实，不表示实物已经核验。

## 1. 标记家族与尺寸

**主配置采用 ChArUco 板，内部 marker 使用 ArUco `DICT_5X5_100`，与当前项目的 AprilTag `tagStandard41h12` 不同。** 标定入口只支持 `checkerboard` 和 `charuco`，工厂分别建立棋盘格检测器和 ChArUco 检测器。当前源码没有对应当前项目 AprilTag 的检测实现；不能因为配置接受 OpenCV `DICT_*` 字符串便判定可直接检测当前 AprilTag。[配置][config]；[检测器工厂][factory]；[配置校验][board-config]；[本项目检测器](/home/stoor/桌面/LY/proj/Embodied_VG/infer_apriltag_cylinder.py:56)

| 项目 | 普通棋盘格主配置 | ChArUco 主配置 |
| --- | --- | --- |
| 方格数 | 12 × 9（由内角点数推得） | 14 × 9 |
| 内角点数 | 11 × 8 = 88 | 13 × 8 = 104 |
| 方格边长 | 0.030 m = 30 mm | 0.020 m = 20 mm |
| marker 边长 | 无 | 0.015 m = 15 mm |
| marker 字典 | 无 | `DICT_5X5_100` |
| marker ID | 无 | 0–62，共 63 个，按板布局位置顺序传入 |
| 图案区域尺寸 | 360 × 270 mm（不含外白边） | 280 × 180 mm（不含外白边） |
| 图案兼容开关 | 无 | `legacy_pattern: false` |

以上数值取自 [config.yaml:1–24][config]；方格数/内角点数的含义和 ChArUco 实物布局要求见 [README:150–165][readme-board]。源码用 `marker_ids` 构造整块 `cv2.aruco.CharucoBoard`，要求 ID 数量为方格总数的一半、互不重复、位于字典范围内；ID 的列表顺序有几何含义。[构板源码:10–18][charuco]；[校验源码:73–103][board-config]

仓库的数字测试板另用 **12 × 9 方格、15 mm 格边、11.25 mm marker、ID 0–53**，不能与主配置混用。[README:178][readme-example]

`requirements.txt` 固定 NumPy 2.2.6、`opencv-python==4.13.0.92`、PyYAML 6.0.3、RealSense SDK 2.58.1.10581；Pillow 范围为 `>=11.3,<13`，Codroid SDK 固定到另一仓库提交 `fa5c329…`。依赖列表没有独立 `apriltag` / `pupil-apriltags` 包。[requirements.txt:1–6][requirements]

## 2. 实际检测与单帧位姿链

ChArUco 流程为：RGB → 灰度 → `cv2.aruco.CharucoDetector.detectBoard` → 按 ChArUco **角点 ID** 匹配整板 3D 坐标 → PnP。构造 `CharucoParameters` 时保留 `minMarkers=2`，有内参时同时传入内参与畸变系数。输出同时保存 marker ID 和 ChArUco 角点 ID，两者不是同一种编号。[charuco.py:24–46][charuco]；[自动采集调用:144][auto-session]

主配置允许至少 **6 个非共线 ChArUco 角点**，无需整板 104 角点全部可见；源码检查 ID 唯一性与范围，调用 `checkCharucoCornersCollinear`，再用 `board.matchImagePoints(corners, ids)` 获取对应的整板坐标。不同帧可见不同角点子集，但目标坐标系保持同一个整板坐标系。[config.yaml:22–24][config]；[charuco.py:48–80][charuco]

共同 PnP 使用 `cv2.solvePnP(..., flags=cv2.SOLVEPNP_ITERATIVE)`，至少 4 个有限点，拒绝共线点、求解失败、非有限解、任一点落在相机后方，以及超过配置上限的重投影 RMS。RMS 是每点二维像素残差平方和的均值再开方；ChArUco 主配置上限为 **2.0 px**。[pose.py:10–37][pose]；[config.yaml:24][config]

普通棋盘格走 `cv2.findChessboardCorners`，必须检出全部 88 个内角点，再用 `cornerSubPix` 精修；物点按行列网格乘 0.03 m 建立。它没有使用 ChArUco/marker ID 来消除局部板匹配问题。[checkerboard.py:10–55][checkerboard]

## 3. 坐标与手眼求解

该工具实现 **Eye-in-Hand**：相机刚性固定在末端，板固定在工作空间；结果是 **相机到末端** `T_end_camera`。[README:3–5][readme-intro]；[solver.py:178–220][solver-output]

`T_A_B` 把 B 系点变换到 A 系，所用关系为：

```text
p_end  = T_end_camera @ p_camera
p_base = T_base_end @ T_end_camera @ p_camera
T_base_target = T_base_end @ T_end_camera @ T_camera_target
```

`T_base_end` 是机器人输入，PnP 给出 `T_camera_target`；一致性计算实际就是上述三矩阵相乘。[README:247–259][readme-coordinates]；[geometry.py:92–113][geometry]

| 坐标项 | 约定与处理 | 源码证据 |
| --- | --- | --- |
| 机器人平移 | CRI `tcp_pose` 的毫米除以 1000，落盘为米 | [codroid.py:35–48][codroid-si] |
| 机器人旋转 | CRI 角度转弧度；欧拉组合 `R = Rz @ Ry @ Rx` | [codroid.py:35–48][codroid-si]；[geometry.py:12–19][geometry-euler] |
| 机器人末端 | 把 `tcp_pose` 当法兰，因此必须为零 TCP 偏移；元数据明确 `tcp_offset_verified: false`，代码没有自动验证 | [codroid.py:165–174][codroid-sample]；[runner.py:174][hardware-runner] |
| 相机系 | RealSense RGB optical frame：X 右、Y 下、Z 前 | [README:259][readme-coordinates]；[solver.py:217][solver-output] |
| ChArUco 目标系 | OpenCV 整板外图案左上角原点，X 右、Y 下、Z 为 X×Y，长度为米 | [solver.py:218–220][solver-output]；[charuco.py:75–79][charuco] |
| 普通棋盘格目标系 | 第一个内角点原点，X 沿列、Y 沿行，长度为米 | [checkerboard.py:10–13][checkerboard]；[solver.py:220][solver-output] |
| 输出 | 4×4、旋转矩阵、米制平移、`quaternion_xyzw`、角度制 `euler_xyz_deg` | [solver.py:188–199][solver-output] |

手眼求解将机器人旋转/平移和 PnP 旋转/平移直接交给 `cv2.calibrateHandEye`。`AUTO` 比较 Tsai、Park、Daniilidis，以固定板在基座系的一致性平移 RMS 最小为优先，完全相同时再比旋转 RMS；显式选择还可用 Horaud、Andreff。最终要求至少 **5 张有效检测且有效 PnP 的图像**。[solver.py:18–58][solver-methods]；[solver.py:138–175][solver-data]

算法选择使用本批次同一组图像，并未使用独立验证集；一致性误差不等于绝对空间精度。[solver.py:182–186][solver-output]；[README:239–245][readme-result]

## 4. 机器人采样流程：人工与真机自动

人工模式默认仅读取 CRI、不发运动命令；每帧检测，操作者按 Space 才保存。采集拒绝离线、运动、报警、位姿不可用、检测失败及近重复位姿；默认近重复门槛是平移同时小于 10 mm、旋转小于 3°，完成还需平移跨度至少 50 mm、旋转跨度至少 30°。坐标、关节和状态取自同一 CRI 快照，但不代表与相机有硬件同步。[README:27–29、63][readme-manual]；[capture.py:47–127][manual-capture]；[pose_quality.py:47–65][pose-quality]；[codroid.py:152–174][codroid-sample]

**当前源码已实现 `--mode auto --backend hardware`。** 它调用 `CodroidMotionRobot` 与 `RealSenseCamera`，实际会发送 CRI 关节命令；不能按 README 的旧描述把此组合当成必然拒绝运行。[runner.py:43–46、158–183][hardware-dispatch]

真机自动流程按当前源码为：

1. 读取现场关节作为 `rail_safe_joints`，检查机器人静止和运动可用；按最近的 360° 等效关节角移到配置的 overhead 起始姿态。主 `motion.hardware.yaml` 指定起始六轴角为 `[240.786, 5.914, 97.703, -165.088, -83.391, -27.666]` 度，速度 8°/s。[runner.py:187–219][hardware-runner]；[motion.hardware.yaml:6–17][hardware-motion]；[codroid.py:22–32][joint-wrap]
2. 起始帧做真实检测/PnP；RMS ≤2 px，ChArUco 还需至少 **20 个角点**，比一般采样 6 角点更严格。[runner.py:139–155][confirm-board]
3. 若指定 `targets_deg`，直接使用该关节列表；否则依赖既有 `eye_hand_data/calib_20260914/hand_eye_result_in-hand.yaml` 先验外参，并通过控制器 FK/IK 生成朝向板中心的候选关节。先验用于瞄准和推算板中心，最终求解仍重新执行检测与手眼标定。[runner.py:19–21、223–233][hardware-prior]；[physical.py:171–196][physical-plan]；[codroid.py:291–309][robot-fk-ik]；[runner.py:307–310][hardware-finish]
4. 候选为起始极点 1 个、25° 倾角六方位环在 **0.80× 和 1.20×** 初始距离各 6 个、35° 环 6 个、35° 环两个方位分别 ±20° roll 共 4 个，即过滤前 **23 个**。这些角度是相对初始观察方向，不是强制相对实物板法线。FK 复核拒绝瞄准误差 >8°、距离越出初始半径 0.75–1.25×、实际距离偏离目标 >8%、近重复关节目标，随后按最大关节变化最近优先排序。[physical.py:20–34、90–111][physical-directions]；[physical.py:145–208][physical-plan]
5. 每个目标经 CRI 关节流运动，目标角度选现场读数最近的 360° 分支；最大关节变化拆成 ≤12° 分段，段内采用 `10u³−15u⁴+6u⁵` 静止到静止插值，默认流周期 4 ms，通过 `CriRealtimeDispatcher.SendCommand(..., TrajectorySpace.JOINT)` 发送。[runner.py:177–183][hardware-runner]；[codroid.py:202–240][robot-interpolation]；[codroid.py:247–259、359–443][robot-motion]
6. 到位需静止且关节误差 ≤0.5°，连续稳定 0.4 s、关节变化 ≤0.08° 后取新帧。机器人状态和到达的相机帧要求 300 ms 新鲜度，采集时移动则失败；连续至少 2 帧通过真实检测、PnP ≤2 px 和位姿质量检查后保存，单目标检测等待上限 5 s。[motion.hardware.yaml:7–15][hardware-motion]；[session.py:63–78、109–172][auto-session]
7. 正常完成要求 **至少 15 样本**、机器人位姿覆盖达标，且默认规划时 PnP 测得的相机到板中心距离跨度 **≥120 mm**；不是固定半径恰好 15 样本。轨迹耗尽但已有 ≥5 样本也会正式求解并报告覆盖不足警告。异常/取消时会请求停止运动；外层 `finally` 还会在无报警时尝试返回原 `rail_safe_joints`。[session.py:166–222][auto-session]；[runner.py:270–276、294–310][hardware-finish]；[physical.py:33][physical-directions]

真机规划源码只复核 FK/IK、距离、瞄准角和关节重复；本次所见真机 RealSense 适配器没有提供模拟器的 `collision_quality` / `observation_quality` 接口，因此不能把模型仿真的路径碰撞检测或投影面积门槛当成真机自动模式已经执行的检查。[physical.py:185–205][physical-plan]；[session.py:79–87、148–160][auto-session]；[realsense.py][realsense]

RealSense 实际开启 RGB 1920×1080/30 FPS、depth 1280×720/30 FPS，预热 20 帧并将深度对齐至 RGB；采集返回 RGB 与帧元数据，PnP 使用 RGB 内参，不使用深度点求板位姿。[realsense.py:13–19、104–126、195–241][realsense]；[pose.py:21][pose]；[README:146][readme-camera]

## 5. README 与源码的差异

| README 描述 | 固定提交实际源码 |
| --- | --- |
| “`auto + hardware` 会在连接前报错”“仅交付模拟闭环” | `run_auto` 已分发到 `run_auto_hardware`，存在真实 CRI 关节发送。[README:120][readme-auto]；[runner.py:43–46][hardware-dispatch]；[codroid.py:439][robot-motion] |
| 自动模式固定 15 样本、所有点同一半径 | 真机为至少 15，变距离 0.8/1.2×且要求 ≥120 mm 跨度；README 此段所说的是 model-sim，不能概括真机。[README:86–88][readme-sim]；[runner.py:171–176][hardware-runner]；[session.py:211–222][auto-session] |
| ChArUco 6 角点可以 READY | 一般 ChArUco 门槛确为 6；真机自动起始帧另要求 20。[config.yaml:23][config]；[runner.py:153–154][confirm-board] |

核验时 `git ls-tree -r HEAD eye_hand_data` 没有列出已跟踪的先验标定文件；默认真机规划需要操作者已有该先验文件，否则明确报错。也可给出 `targets_deg` 绕过先验规划。[runner.py:223–233][hardware-prior]

## 6. 当前项目 AprilTag 圆柱定位实现对照

本节直接核对当前工作区的 [infer_apriltag_cylinder.py](/home/stoor/桌面/LY/proj/Embodied_VG/infer_apriltag_cylinder.py:1) 和 [plug_vg/robot_transform.py](/home/stoor/桌面/LY/proj/Embodied_VG/plug_vg/robot_transform.py:24)，描述其现有行为。

| 项目 | HandEyeCalib | 当前项目 `infer_apriltag_cylinder.py` |
| --- | --- | --- |
| 任务 | 多姿态求解 `T_end_camera` | 使用已有 `T_end_camera` 定位 tag 中心并画圆柱 |
| 检测目标 | 整块 ChArUco，ArUco `DICT_5X5_100`，marker ID 0–62 | 单 AprilTag `tagStandard41h12`，默认 ID 0，必须 `hamming==0` |
| 检测器 | `cv2.aruco.CharucoDetector` | `pupil_apriltags.Detector`，4 线程、`quad_decimate=1.0` |
| 尺寸输入 | 格边 20 mm、marker 边 15 mm | 完整 9×9 编码图案默认 40 mm，不含外白边；pose 输入使用内部 5×5 检测方形 `40×5/9=22.222… mm` |
| 单帧几何 | 多 ChArUco 角点与整板物点匹配后 PnP | 每次均请求 tag pose；定位中心时有深度优先用深度，无深度用 `tag.pose_t` |
| 目标原点 | 整板外图案左上角 | tag 中心 |
| 输出点到基座 | 用于比较板位姿一致性 | `T_base_end @ T_end_camera` 变换 tag 中心，再沿基座 −Y 偏移 |

HandEyeCalib 一列证据：[构板与检测][charuco]、[配置][config]、[求解输出][solver-output]。当前项目一列证据：[检测与尺寸换算:56–72](/home/stoor/桌面/LY/proj/Embodied_VG/infer_apriltag_cylinder.py:56)、[默认参数:36–44](/home/stoor/桌面/LY/proj/Embodied_VG/infer_apriltag_cylinder.py:36)、[中心与坐标变换:254–275](/home/stoor/桌面/LY/proj/Embodied_VG/infer_apriltag_cylinder.py:254)。

当前项目的 D2RGB 分支以 tag 中心为圆形邻域，半径为最短检测边长的 15%、截断到 1–6 px；只保留角点多边形内部、有限且深度范围合格的像素，至少 5 个样本。深度原值乘相机 `depth_scale` 后取中位数作为 **Z**，然后按中心像素反投影：`X=(u-cx)Z/fx`、`Y=(v-cy)Z/fy`。这是光轴深度反投影，不是用到相机距离的欧氏模长。[depth_center:75–99](/home/stoor/桌面/LY/proj/Embodied_VG/infer_apriltag_cylinder.py:75)

未指定 `--d2rgb` 时脚本自动寻找同级 `*_d2rgb.npy/.png/.tif/.tiff`；找到深度后若格式、分辨率或有效样本不合格会报错，不会自动降级到 RGB pose。仅当没有深度路径时才用 `tag.pose_t`，并记录 `pose_err`。[run:255–268](/home/stoor/桌面/LY/proj/Embodied_VG/infer_apriltag_cylinder.py:255)

脚本输入机器人位姿为米与弧度，组合与 HandEyeCalib 一致：`Rz @ Ry @ Rx`。先算 `T_base_camera = T_base_end @ T_end_camera`，再变换中心，最后加 `[0,-offset,0]`，默认偏移 0.10 m。该偏移是**基座 −Y**，不跟随 tag 朝向；圆柱轴固定平行基座 Y。tag 的 `pose_R` 仅用于把四条角点射线与通过中心的 tag 平面相交并画检测方形。[机器人变换:38–80](/home/stoor/桌面/LY/proj/Embodied_VG/plug_vg/robot_transform.py:38)、[run:271–284](/home/stoor/桌面/LY/proj/Embodied_VG/infer_apriltag_cylinder.py:271)、[结果字段:301–305](/home/stoor/桌面/LY/proj/Embodied_VG/infer_apriltag_cylinder.py:301)

## 7. 同图无硬件检测对照

2026-10-11 在 `embodiedvg` 环境，对现有 [camera_data/20261010_144623_467_color.png](/home/stoor/桌面/LY/proj/Embodied_VG/camera_data/20261010_144623_467_color.png) 分别执行两套真实检测入口。通过 `PYTHONPATH` 导入目标仓库的 `create_detector`，使用其主 `config.yaml` 的 `charuco` 配置；另一分支调用本项目 `detect_tag`。结果如下：

| 检测入口 | 结果 |
| --- | --- |
| HandEyeCalib 主配置 ChArUco | `found=False`，`reason=charuco_insufficient_corners`，`corner_count=0` |
| 本项目 AprilTag | `family=tagStandard41h12`，`tag_id=0`，`hamming=0`，`pose_square_size_m=0.0222222222…` |

此结果验证当前 HandEyeCalib 主配置不能直接识别本图的 AprilTag。测试只执行静态图像检测，不包含机器人运动或完整标定。入口源码证据：[HandEyeCalib 工厂][factory]、[ChArUco 检测][charuco]、[本项目 detect_tag:56–72](/home/stoor/桌面/LY/proj/Embodied_VG/infer_apriltag_cylinder.py:56)。

## 8. 接口兼容性结论

**外参文件与坐标链兼容；标记检测流程不能直接互换。** 当前项目的外参加载器明确要求 `transform_name: T_end_camera` 和 `matrix_4x4`；HandEyeCalib 结果正好提供这两个字段，平移同为米，机器人 RPY 顺序也一致。因此在相机光学坐标、末端参考系及安装关系相同的前提下，其输出可以供当前脚本使用。校准时使用 ChArUco，并不要求后续定位也使用 ChArUco：`T_end_camera` 描述的是相机与末端的刚性关系，而不是标记图案的编码家族。[当前加载器:24–35](/home/stoor/桌面/LY/proj/Embodied_VG/plug_vg/robot_transform.py:24)、[当前应用外参:271–274](/home/stoor/桌面/LY/proj/Embodied_VG/infer_apriltag_cylinder.py:271)、[HandEyeCalib 输出][solver-output]、[坐标链][geometry]

以下是对接建议，不是已经实施的修改：

- 保留当前 `tagStandard41h12`、ID 和 `5/9` 尺寸换算，用 HandEyeCalib 只生成/更新外参即可；不能将它的 15 mm ArUco marker 或整板角点 ID 套到当前 tag 上。[本项目检测](/home/stoor/桌面/LY/proj/Embodied_VG/infer_apriltag_cylinder.py:56)、[HandEyeCalib 构板][charuco]
- 若要改成“使用当前单 AprilTag 做手眼标定”，则需要新增检测适配、按 tag 角点顺序建立米制物点、质量检查及跨姿态采样。可复用的是“二维点 + 三维物点 → PnP → `T_camera_target` → 手眼求解”的接口；现成 ChArUco 检测器不能完成该工作。[检测器工厂][factory]、[PnP 接口][pose]、[手眼求解][solver-methods]
- 直接比较单帧平移时须区分整板左上角与 tag 中心。只有想把两者解释为同一实物目标系，才需要板系到 tag 系固定变换；单纯将各自相机系点经同一外参映射到基座，不要求两种目标原点相同。[HandEyeCalib 目标约定][solver-output]、[当前 tag 中心变换](/home/stoor/桌面/LY/proj/Embodied_VG/infer_apriltag_cylinder.py:274)
- 可以参考其近重复剔除、稳定后新帧、PnP 质量、变距离采样和多算法一致性比较；复用自动运动还要匹配 Codroid FK/IK、TCP 与 CRI 接口。其模型仿真检查不能当作真机已检查。[位姿质量][pose-quality]、[自动状态机][auto-session]、[真机规划][physical-plan]、[控制器 FK/IK][robot-fk-ik]

[config]: https://github.com/KATe-AGI/HandEyeCalib/blob/4a9c8e302e3ee9bd23e83964d4baec3fe4aa03c3/config.yaml#L1-L30
[factory]: https://github.com/KATe-AGI/HandEyeCalib/blob/4a9c8e302e3ee9bd23e83964d4baec3fe4aa03c3/src/hand_eye_calib/boards/detector.py#L16-L20
[board-config]: https://github.com/KATe-AGI/HandEyeCalib/blob/4a9c8e302e3ee9bd23e83964d4baec3fe4aa03c3/src/hand_eye_calib/boards/config.py#L73-L106
[charuco]: https://github.com/KATe-AGI/HandEyeCalib/blob/4a9c8e302e3ee9bd23e83964d4baec3fe4aa03c3/src/hand_eye_calib/boards/charuco.py#L10-L80
[checkerboard]: https://github.com/KATe-AGI/HandEyeCalib/blob/4a9c8e302e3ee9bd23e83964d4baec3fe4aa03c3/src/hand_eye_calib/boards/checkerboard.py#L10-L55
[requirements]: https://github.com/KATe-AGI/HandEyeCalib/blob/4a9c8e302e3ee9bd23e83964d4baec3fe4aa03c3/requirements.txt#L1-L6
[pose]: https://github.com/KATe-AGI/HandEyeCalib/blob/4a9c8e302e3ee9bd23e83964d4baec3fe4aa03c3/src/hand_eye_calib/calibration/pose.py#L10-L37
[geometry]: https://github.com/KATe-AGI/HandEyeCalib/blob/4a9c8e302e3ee9bd23e83964d4baec3fe4aa03c3/src/hand_eye_calib/calibration/geometry.py#L92-L133
[geometry-euler]: https://github.com/KATe-AGI/HandEyeCalib/blob/4a9c8e302e3ee9bd23e83964d4baec3fe4aa03c3/src/hand_eye_calib/calibration/geometry.py#L12-L19
[solver-methods]: https://github.com/KATe-AGI/HandEyeCalib/blob/4a9c8e302e3ee9bd23e83964d4baec3fe4aa03c3/src/hand_eye_calib/calibration/solver.py#L18-L58
[solver-data]: https://github.com/KATe-AGI/HandEyeCalib/blob/4a9c8e302e3ee9bd23e83964d4baec3fe4aa03c3/src/hand_eye_calib/calibration/solver.py#L138-L175
[solver-output]: https://github.com/KATe-AGI/HandEyeCalib/blob/4a9c8e302e3ee9bd23e83964d4baec3fe4aa03c3/src/hand_eye_calib/calibration/solver.py#L178-L222
[codroid-si]: https://github.com/KATe-AGI/HandEyeCalib/blob/4a9c8e302e3ee9bd23e83964d4baec3fe4aa03c3/src/hand_eye_calib/hardware/codroid.py#L35-L48
[codroid-sample]: https://github.com/KATe-AGI/HandEyeCalib/blob/4a9c8e302e3ee9bd23e83964d4baec3fe4aa03c3/src/hand_eye_calib/hardware/codroid.py#L152-L174
[joint-wrap]: https://github.com/KATe-AGI/HandEyeCalib/blob/4a9c8e302e3ee9bd23e83964d4baec3fe4aa03c3/src/hand_eye_calib/hardware/codroid.py#L22-L32
[robot-interpolation]: https://github.com/KATe-AGI/HandEyeCalib/blob/4a9c8e302e3ee9bd23e83964d4baec3fe4aa03c3/src/hand_eye_calib/hardware/codroid.py#L202-L240
[robot-motion]: https://github.com/KATe-AGI/HandEyeCalib/blob/4a9c8e302e3ee9bd23e83964d4baec3fe4aa03c3/src/hand_eye_calib/hardware/codroid.py#L247-L443
[robot-fk-ik]: https://github.com/KATe-AGI/HandEyeCalib/blob/4a9c8e302e3ee9bd23e83964d4baec3fe4aa03c3/src/hand_eye_calib/hardware/codroid.py#L291-L309
[hardware-dispatch]: https://github.com/KATe-AGI/HandEyeCalib/blob/4a9c8e302e3ee9bd23e83964d4baec3fe4aa03c3/src/hand_eye_calib/automatic/runner.py#L43-L183
[hardware-runner]: https://github.com/KATe-AGI/HandEyeCalib/blob/4a9c8e302e3ee9bd23e83964d4baec3fe4aa03c3/src/hand_eye_calib/automatic/runner.py#L158-L269
[hardware-prior]: https://github.com/KATe-AGI/HandEyeCalib/blob/4a9c8e302e3ee9bd23e83964d4baec3fe4aa03c3/src/hand_eye_calib/automatic/runner.py#L19-L233
[confirm-board]: https://github.com/KATe-AGI/HandEyeCalib/blob/4a9c8e302e3ee9bd23e83964d4baec3fe4aa03c3/src/hand_eye_calib/automatic/runner.py#L139-L155
[hardware-finish]: https://github.com/KATe-AGI/HandEyeCalib/blob/4a9c8e302e3ee9bd23e83964d4baec3fe4aa03c3/src/hand_eye_calib/automatic/runner.py#L270-L317
[hardware-motion]: https://github.com/KATe-AGI/HandEyeCalib/blob/4a9c8e302e3ee9bd23e83964d4baec3fe4aa03c3/motion.hardware.yaml#L1-L17
[physical-directions]: https://github.com/KATe-AGI/HandEyeCalib/blob/4a9c8e302e3ee9bd23e83964d4baec3fe4aa03c3/src/hand_eye_calib/automatic/physical.py#L20-L111
[physical-plan]: https://github.com/KATe-AGI/HandEyeCalib/blob/4a9c8e302e3ee9bd23e83964d4baec3fe4aa03c3/src/hand_eye_calib/automatic/physical.py#L145-L209
[auto-session]: https://github.com/KATe-AGI/HandEyeCalib/blob/4a9c8e302e3ee9bd23e83964d4baec3fe4aa03c3/src/hand_eye_calib/automatic/session.py#L49-L222
[manual-capture]: https://github.com/KATe-AGI/HandEyeCalib/blob/4a9c8e302e3ee9bd23e83964d4baec3fe4aa03c3/src/hand_eye_calib/capture.py#L47-L127
[pose-quality]: https://github.com/KATe-AGI/HandEyeCalib/blob/4a9c8e302e3ee9bd23e83964d4baec3fe4aa03c3/src/hand_eye_calib/pose_quality.py#L11-L65
[realsense]: https://github.com/KATe-AGI/HandEyeCalib/blob/4a9c8e302e3ee9bd23e83964d4baec3fe4aa03c3/src/hand_eye_calib/hardware/realsense.py#L13-L241
[readme-intro]: https://github.com/KATe-AGI/HandEyeCalib/blob/4a9c8e302e3ee9bd23e83964d4baec3fe4aa03c3/README.md#L3-L5
[readme-manual]: https://github.com/KATe-AGI/HandEyeCalib/blob/4a9c8e302e3ee9bd23e83964d4baec3fe4aa03c3/README.md#L27-L63
[readme-board]: https://github.com/KATe-AGI/HandEyeCalib/blob/4a9c8e302e3ee9bd23e83964d4baec3fe4aa03c3/README.md#L150-L165
[readme-example]: https://github.com/KATe-AGI/HandEyeCalib/blob/4a9c8e302e3ee9bd23e83964d4baec3fe4aa03c3/README.md#L178
[readme-result]: https://github.com/KATe-AGI/HandEyeCalib/blob/4a9c8e302e3ee9bd23e83964d4baec3fe4aa03c3/README.md#L239-L245
[readme-coordinates]: https://github.com/KATe-AGI/HandEyeCalib/blob/4a9c8e302e3ee9bd23e83964d4baec3fe4aa03c3/README.md#L247-L259
[readme-auto]: https://github.com/KATe-AGI/HandEyeCalib/blob/4a9c8e302e3ee9bd23e83964d4baec3fe4aa03c3/README.md#L120
[readme-sim]: https://github.com/KATe-AGI/HandEyeCalib/blob/4a9c8e302e3ee9bd23e83964d4baec3fe4aa03c3/README.md#L86-L88
[readme-camera]: https://github.com/KATe-AGI/HandEyeCalib/blob/4a9c8e302e3ee9bd23e83964d4baec3fe4aa03c3/README.md#L146
