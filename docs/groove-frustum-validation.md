# 圆台入槽判据与四帧验证

日期：2026-10-23。

当前示例 `20261023_154709_074` 已由 `not_seated` 改为 `seated`。旧判据的误判复现于独立几何接口：已保存的预测头尾轴输入 `evaluate_groove_placement()`，最大端点距离为 6.347 mm，超过原先 5 mm。基准高度处轴线距离只有 1.451 mm，相对倾角 3.382°，尾端轴向越界仅 0.088 mm。旧接受域将头部和可摆动的尾部同等限制，不能表达用户确认的放置要求。

## 本次确认的判据

用户说明成功放置时头部固定，尾部允许轻微摆动，并明确选择：

- 允许区域为圆形截面的圆台，所有横向方向均向下放宽。
- 上端半径 **5 mm**，下端半径 **20 mm**，半径沿轴向高度线性变化。
- 有限预测长轴的长度**至少 80% 在圆台内**时通过；超出 20% 则失败。
- 继续使用默认相机、手眼标定及第一帧 `output/apriltag_pose/20261023_163823_957_color_apriltag_pose.json` 为固定参考。

圆台的两端面中心沿用当前允许轴段的上下界，沿入槽轴坐标为 −90～+82 mm。上下端面外的长度计为在外，圆台边界包含在内。分母为预测头尾端面中心之间的完整三维轴长，不先裁剪到目标高度，不使用无限延长线。没有额外新增头部硬位置门；判定按上述长度比例执行。

## 实现

`configs/placement/plug_placement.yaml` 的参数为：

```yaml
groove_check:
  top_radius_m: 0.005
  bottom_radius_m: 0.020
  min_inside_fraction: 0.80
```

`plug_vg/groove_placement.py` 对参数化有限轴线和圆台的二次侧面、两个端面求交，将交点之间位于内部的参数区间长度相加。因此长度比例不依赖离散采样密度。保留头尾点到有限参考轴段的距离作诊断，不再用该距离决定入槽结论。

JSON 新增 `allowed_frustum`，记录上下端面中心、半径和比例阈值；`placement_check` 新增 `inside_fraction`、`outside_fraction`、`axis_length_mm`、`inside_length_mm`、`inside_intervals_t` 等。`inside_intervals_t` 按 `tail + t*(head-tail)` 表示在内部分。`evaluate_groove_placement()` 第四个参数已由旧的距离容差改为圆台配置映射，旧自定义配置需迁移至上述三个参数。

RGB 叠加图显示圆台投影线框；HTML 显示蓝色半透明圆台、绿色在内轴段、红色在外轴段及对应比例。配准、观测点提取、相机、手眼变换、参考位置和原有轴向上下界不变。每帧仍只执行一次配准。

## 完整推理结果

在 `embodiedvg` 环境中，对 `test_data/插头放置判断数据` 中四帧分别使用该帧 meta 的法兰位姿完整运行脚本。全部使用同一参考和默认设置。

| 样例 | 标签 | 轴线在圆台内比例 | 新结论 | 旧结论 |
|---|---|---:|---|---|
| 20261023_154709_074 | true | 99.9475% | seated | not_seated |
| 20261023_154831_018 | true | 96.9537% | seated | not_seated |
| 20261023_154744_708 | false | 24.4414% | not_seated | not_seated |
| 20261023_155046_091 | false | 11.3727% | not_seated | not_seated |

两个正样例退出码为 0，两个负样例退出码为 1，均完成可视化文件保存，没有 `artifact_error`。新结果位于 `output/groove_placement/`，包含每帧 JSON、RGB 叠加图、HTML 和可见点云；当前示例原有 HTML 已刷新。汇总和四帧叠加对照图位于：

- `output/groove_geometry_discussion_261023/frustum_validation.json`
- `output/groove_geometry_discussion_261023/frustum_overlays.png`

历史几何诊断与旧结果快照位于同目录的 `discussion.md`、`axis_decomposition.json` 和 `before_frustum/`。这些数据和运行产物只保存在本地。

## 验证范围

- `python -m unittest discover -s tests`：**64 项通过**。
- 其中 9 项入槽测试覆盖 55 mm 偏移与投影向上、轴向活动余量、80% 边界、上下端面裁剪、圆柱/圆台侧面、轴线两端在外但中间穿越、相切、刚体变换、用户样例回归及无效配置拒绝。
- 对 40 条固定随机种子的轴线，以 20,000 个均匀长度中点的独立数值测量核对解析比例。
- 修改后的 Python 文件语法编译通过。当前环境未安装 Ruff，未运行 Ruff 检查。

这四帧参与了判据讨论，属于调试验证集，没有将其作为独立泛化测试。新判据检查的是轴线相对允许圆台的几何摆放，尚未验证绝对定位精度或物理接触。历史六帧报告对应旧参考和旧 5 mm 端点判据，不与本次结果混为一组。
