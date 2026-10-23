# RealSense 官方深度后处理与离线试验说明

核验日期：2026-10-22。试验 SDK：`pyrealsense2 2.58.3.10794`。本文核验官方 `librealsense v2.58.3` 的文档与源码，固定提交为 [`dfd6aa91250f5c31521d72d627865417989bb4e7`](https://github.com/realsenseai/librealsense/tree/dfd6aa91250f5c31521d72d627865417989bb4e7)，避免后续 `master` 更新改变参数含义。

## 官方推荐流程

官方文档给出的 D4xx 双目深度相机处理顺序为：

```text
Depth → Decimation → Depth to Disparity → Spatial → Temporal
      → Disparity to Depth → Hole Filling → Filtered Depth
```

各模块可以单独启用，SDK 不强制固定顺序；这是官方工具和演示采用的推荐组合。文档同时说明整数深度域可用于 D400，但更推荐在视差域完成空间与时间滤波。[官方后处理说明](https://github.com/realsenseai/librealsense/blob/dfd6aa91250f5c31521d72d627865417989bb4e7/doc/post-processing-filters.md#using-filters-in-application-code)

| 模块 | 处理方法 | 默认参数与可用范围 |
| --- | --- | --- |
| Decimation，降采样 | 对有效的非零深度聚合；2×2、3×3 核取中值，4×4～8×8 核取均值；同时降低空间分辨率 | `filter_magnitude=2`。文档描述有效降采样倍率 2～8；v2.58.3 源码实际允许 1～8，1 表示不缩小 |
| Depth to Disparity，深度转视差 | 利用焦距、双目基线、深度单位，把深度转换为与 `1/Z` 成正比的浮点视差 | `disparity_transform(True)`；SDK 内部使用 5 位小数对应的视差单位，即 1/32 像素 |
| Spatial，保边空间滤波 | 通过水平方向和垂直方向的多轮一维滤波平滑局部深度，同时以阈值保留突变边界 | 迭代次数 `filter_magnitude=2`，范围 1～5；`filter_smooth_alpha=0.5`，范围 0.25～1；`filter_smooth_delta=20`，范围 1～50；内部 `holes_fill=0`，范围 0～5 |
| Temporal，时间滤波 | 使用同一像素的历史帧做指数平滑；按历史有效性规则，用历史值补当前缺失值 | `filter_smooth_alpha=0.4`，范围 0～1；`filter_smooth_delta=20`，范围 1～100；持久性 `holes_fill=3`，范围 0～8，默认规则为最近 4 帧至少 2 帧有效 |
| Disparity to Depth，视差转深度 | 将滤波后的视差恢复为深度 | `disparity_transform(False)` |
| Hole Filling，独立补洞 | 用邻近像素的深度值填充零值像素 | `holes_fill=1`；0 为从左侧补，1 为取邻域较远值，2 为取邻域较近值 |

参数与算法来源：[官方后处理文档](https://github.com/realsenseai/librealsense/blob/dfd6aa91250f5c31521d72d627865417989bb4e7/doc/post-processing-filters.md)、[Decimation 源码](https://github.com/realsenseai/librealsense/blob/dfd6aa91250f5c31521d72d627865417989bb4e7/src/proc/decimation-filter.cpp)、[Spatial 源码](https://github.com/realsenseai/librealsense/blob/dfd6aa91250f5c31521d72d627865417989bb4e7/src/proc/spatial-filter.cpp)、[Temporal 源码](https://github.com/realsenseai/librealsense/blob/dfd6aa91250f5c31521d72d627865417989bb4e7/src/proc/temporal-filter.cpp)、[Hole Filling 源码](https://github.com/realsenseai/librealsense/blob/dfd6aa91250f5c31521d72d627865417989bb4e7/src/proc/hole-filling-filter.cpp)、[深度／视差换算源码](https://github.com/realsenseai/librealsense/blob/dfd6aa91250f5c31521d72d627865417989bb4e7/src/proc/disparity-transform.h)。

需要区分三个容易混淆的参数：

- `alpha` 是当前样本的权重；越接近 1，平滑越弱，1 表示不平滑。
- `delta` 直接作用于当前数据表示的数值差。在 Z16 深度域，20 表示 20 个原始深度单位；本批数据的单位约 1 mm，因此约为 20 mm。在 SDK 的视差表示下，它表示视差数值阈值，不能继续解释成 20 mm。[Spatial 阈值代码](https://github.com/realsenseai/librealsense/blob/dfd6aa91250f5c31521d72d627865417989bb4e7/src/proc/spatial-filter.cpp)、[Temporal 数值比较代码](https://github.com/realsenseai/librealsense/blob/dfd6aa91250f5c31521d72d627865417989bb4e7/src/proc/temporal-filter.h)
- Spatial 的 `holes_fill` 是水平扫描内的补洞半径模式，0～5 对应关闭、2、4、8、16、无限像素；Temporal 的同名选项是时间持久性规则；独立 Hole Filling 的同名选项是补洞取值方式。它们不是同一项功能。[官方参数表](https://github.com/realsenseai/librealsense/blob/dfd6aa91250f5c31521d72d627865417989bb4e7/doc/post-processing-filters.md)

## 本批测试数据能验证什么

测试目录为 `test_depth_261022/`，包含 5 组 RGB、D2RGB NPY、元数据：

| 样本 ID | 输入深度 |
| --- | --- |
| `20260828_203408_702` | `20260828_203408_702_d2rgb.npy` |
| `20260828_203754_038` | `20260828_203754_038_d2rgb.npy` |
| `20260902_133046_461` | `20260902_133046_461_d2rgb.npy` |
| `20260902_133202_557` | `20260902_133202_557_d2rgb.npy` |
| `20260902_133445_381` | `20260902_133445_381_d2rgb.npy` |

5 份元数据均记录相机为 `Intel RealSense D435IF`，深度已对齐到 1920×1080 RGB 网格，`depth_scale=0.0010000000474974513 m`，采集时 `enable_post_processing=false`。保存的是未经 SDK 后处理的对齐深度，而不是原始双目深度传感器网格。元数据只有 RGB／对齐深度内参，没有原生深度内参、深度到 RGB 外参和双目基线。

SDK 换算系数为 `baseline_m × fx × 32 / depth_units`，并要求帧关联的传感器提供双目深度能力。因此，只有这些 NPY 和元数据时，不能声称已经按真实标定完整复现双目视差处理链；本次不填入假定基线。[官方视差换算实现](https://github.com/realsenseai/librealsense/blob/dfd6aa91250f5c31521d72d627865417989bb4e7/src/proc/disparity-transform.h)

本次离线试验直接调用官方 SDK 的以下深度域处理组合，各图像独立处理：

| 对照组 | 作用 |
| --- | --- |
| Raw | 原始 D2RGB |
| Spatial | 默认空间滤波，内部补洞关闭，观察有效深度的平滑 |
| Spatial → Hole Filling | 默认空间滤波后，再使用默认独立补洞模式 1 |
| Decimation ×2 | 观察降采样和有效像素聚合的作用 |
| Decimation ×2 → Spatial → Hole Filling | 观察空间处理组合的效果 |
| Temporal，首帧检查 | 每张图使用全新时间滤波器，仅处理一次；用于确认没有历史帧时不会改善该帧 |

这是“官方滤波模块在已对齐单帧深度上的离线对照”，与“原生双目深度上的完整推荐流程”需要分别理解。官方文档明确允许模块独立使用，同时指出 D400 更推荐视差域，因此本次结果不能代替未来完整视差流程的验证。[官方模块与流水线说明](https://github.com/realsenseai/librealsense/blob/dfd6aa91250f5c31521d72d627865417989bb4e7/doc/post-processing-filters.md#design-and-implementation)

## 单帧、对齐与补洞的限制

时间滤波的改善依赖真实连续历史。源码在初始配置时清零历史，首帧只建立历史；没有历史值时既不能平滑，也不能利用历史补洞。因此不把 5 个跨时刻样本串成一段时间序列，也不重复同一静帧来宣称获得时间降噪效果。之后应通过同一固定相机的连续帧或 `.bag` 录像测试，动态场景还需检查拖影；官方同样指出 Temporal 更适合静态场景。[历史初始化](https://github.com/realsenseai/librealsense/blob/dfd6aa91250f5c31521d72d627865417989bb4e7/src/proc/temporal-filter.cpp)、[首帧和历史处理逻辑](https://github.com/realsenseai/librealsense/blob/dfd6aa91250f5c31521d72d627865417989bb4e7/src/proc/temporal-filter.h)、[官方 Temporal 说明](https://github.com/realsenseai/librealsense/blob/dfd6aa91250f5c31521d72d627865417989bb4e7/doc/post-processing-filters.md#temporal-filter)

对齐会把原生深度像素投影到 RGB 网格，改变像素邻域和空洞分布。官方 `align` 实现使用深度与目标相机的内外参，多个深度点覆盖同一目标像素时保留较近值。因此，在已对齐图上做空间滤波，与先处理原生深度再对齐，不是同一运算。本项目后续若要评估完整采集流程，建议保留原生深度与标定，先在原生网格进行推荐后处理，再对齐到 RGB；这是根据投影与滤波的差别提出的试验建议。[官方对齐源码](https://github.com/realsenseai/librealsense/blob/dfd6aa91250f5c31521d72d627865417989bb4e7/src/proc/align.cpp)

Decimation 会重算输出内参，并将输出宽高补齐到 4 的倍数，多出的边缘补零。对于本次 1920×1080、倍率 2，实际输出为 960×540，`fx/fy/ppx/ppy` 均除以 2。放大到原图尺寸只是为了并排显示，不恢复被降采样去掉的结构，也不能配合原始内参直接用于几何计算。[官方 Decimation 文档](https://github.com/realsenseai/librealsense/blob/dfd6aa91250f5c31521d72d627865417989bb4e7/doc/post-processing-filters.md#decimation-filter)、[内参和尺寸重算实现](https://github.com/realsenseai/librealsense/blob/dfd6aa91250f5c31521d72d627865417989bb4e7/src/proc/decimation-filter.cpp)

独立 Hole Filling 是邻域深度复制，不是利用物体形状重建真实深度。v2.58.3 的较远／较近模式按扫描次序就地修改图像，已补出的值可以参与后续像素补洞；默认模式 1 可能把远处背景延伸到缺失区域。因此空洞减少可证明输出更完整，但不能证明这些新增深度正确。特别应检查插头轮廓、槽口和前景遮挡边界。[官方补洞算法源码](https://github.com/realsenseai/librealsense/blob/dfd6aa91250f5c31521d72d627865417989bb4e7/src/proc/hole-filling-filter.h)

## Z16 空间滤波的整数舍入

v2.58.3 的 Spatial 对整数深度和浮点视差使用不同实现。Z16 分支每次平滑后写入 `uint16_t(filtered + 0.5f)`：正值按最近整数舍入，恰好半个单位时向上取整。默认 `alpha=0.5` 时，两个整数的和为奇数就会出现这种半整数结果。每轮迭代包含左右和上下四个方向的扫描，默认两轮共八次方向扫描；反复取整可能积累向较大深度方向的偏差。均匀的整数常量平面本身不会因上述取整而改变。[整数平滑、舍入与迭代代码](https://github.com/realsenseai/librealsense/blob/dfd6aa91250f5c31521d72d627865417989bb4e7/src/proc/spatial-filter.h#L29)

浮点分支直接存储 `filtered`，没有逐次写回整数时的 `+0.5`；它的边缘判断与整数分支也有差别。因此不能把浮点／视差分支简单理解为“整数分支关闭取整”，也不能用本次 Z16 结果代替完整视差流程的结论。[浮点平滑实现](https://github.com/realsenseai/librealsense/blob/dfd6aa91250f5c31521d72d627865417989bb4e7/src/proc/spatial-filter.cpp#L241)

这项实现细节意味着：比较原始与空间滤波结果时，应同时记录有符号差值 `filtered - raw` 的均值／中位数，以及绝对差值；仅凭绝对差值无法判断是否存在方向偏移。上述代码解释了产生向上量化偏差的可能机制，并不单独证明某张实拍图的变化量完全来自舍入；实际变化还包含平滑和边缘判断的影响。

## 对照图与数值的解读

对照图应使用相同的物理深度色标，零值空洞单独显示；否则每张图分别拉伸色阶会掩盖或夸大差异。降采样图需要标出真实分辨率。变化图应区分“原来有效的深度发生变化”与“原来的空洞被补出”，并保持输入和输出的像素坐标一致。

在没有逐像素深度真值的情况下，可比较有效像素率、补洞像素数、原有效点的深度改变量、固定 ROI 的局部平滑程度。这些量衡量完整性、变化和表面平滑程度，不能直接给出绝对深度精度是否提高。均匀表面更平滑也可能伴随边缘变钝，补洞更完整也可能伴随背景污染。要评估毫米级机器人定位效果，还需加入已知平面／已知距离或独立深度真值，并复测插头几何定位误差。

本次工作仅增加独立试验与说明，不修改入槽、抓取、端部定位的主推理流程。
