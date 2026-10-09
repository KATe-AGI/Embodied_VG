# Ultralytics YOLO-Depth 核验

核验日期：2026-09-02

## 结论

- Ultralytics 官方已经提供单目深度估计任务，正式名称为 **YOLO26-Depth**，模型名为 `yolo26{n,s,m,l,x}-depth.pt`。
- 该任务首次随 `ultralytics 8.4.104` 发布（2026-07-21）。当前 PyPI 最新版为 `8.4.138`（2026-09-01）。
- 核验时本项目环境不能使用该任务：editable 安装元数据显示 `8.4.38`，vendored 源码运行时报告 `8.4.48`；两者都早于 `8.4.104`。2026-09-02 已将 vendored 源码和 editable 安装统一升级到 `8.4.138`，当前环境已经包含 `depth` task、`ultralytics.models.yolo.depth` 和 `yolo26-depth.yaml`。
- YOLO26-Depth 是从单幅 RGB 预测整幅逐像素米制深度图的单目模型。它不是本项目现有的“YOLO-seg 掩膜 + RealSense D2RGB 实测深度”管线，也不是 YOLO 网络配置里 `depth_multiple` 的网络深度参数。

## 本地核验

环境：`/home/stoor/miniconda3/envs/embodiedvg`

```text
pip metadata version: 8.4.138
editable source: /home/stoor/桌面/LY/proj/Embodied_VG/ultralytics
runtime source version: 8.4.138
ultralytics.models.yolo.depth spec: present
```

升级前的版本不一致是 editable 安装后的源码曾被更新、但 `*.dist-info` 元数据没有重新生成造成的。本次重新执行 editable 安装后，两处版本已统一为 `8.4.138`。

## 官方依据

- [Ultralytics v8.4.104 发布说明](https://github.com/ultralytics/ultralytics/releases/tag/v8.4.104)：首次加入完整的 YOLO26 单目深度估计任务，以及 n/s/m/l/x 五种模型。
- [Ultralytics 官方 Depth task 文档](https://docs.ultralytics.com/tasks/depth)：说明训练、验证、推理、输出 `result.depth`、标定和数据格式。
- [YOLO26 Depth 官方模型配置](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/26/yolo26-depth.yaml)：官方 DPT 风格多尺度 depth head 配置。
- [PyPI ultralytics](https://pypi.org/project/ultralytics/)：截至核验日最新发行版为 `8.4.138`，上传于 2026-09-01。

## 对本项目的意义

本项目已有 RealSense D2RGB 传感器深度，配准阶段需要可信的绝对尺度和细小插头几何。YOLO26-Depth 可以作为 RGB-only 降级方案、深度空洞补全候选或交叉检查信号，但不应未经本相机/工况标定和独立精度评估就替换 D2RGB。本次升级前已用现有分割权重完成 12 帧端到端 A/B 回归：新旧版本均为 12/12 成功，全局三语义点 RMSE 从 `11.989 mm` 变为 `11.936 mm`，未发现实质退化。
