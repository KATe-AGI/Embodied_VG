#!/usr/bin/env python3
"""Write a readable summary from a completed camera baseline run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def summarize(output: Path) -> None:
    summary = json.loads((output / "baseline_summary.json").read_text())
    first = json.loads((output / "metrics.json").read_text())
    metadata = json.loads((output / "run_metadata.json").read_text())
    aggregate = first["aggregate"]
    timing = summary["pooled_all_runs"]["timing"]
    stability = summary["repeatability_summary"]
    lines = [
        "# 当前相机系配准基线（2026-11-06）", "",
        f"评测集：`{metadata['dataset']}`。共 {aggregate['total']} 条用户接受的算法配准参考。",
        f"方法：`infer_6d_single.py`，`registration_method={metadata['registration_method']}`，使用当前默认算法参数。",
        f"硬件：{', '.join(metadata['gpu'])}。Torch {metadata['packages']['torch']}；CPU 线程 {metadata['torch_threads']}。",
        f"预热 {metadata['warmup_frames']} 帧，执行 {metadata['repeat_count']} 轮（seed={metadata['seeds']}）。",
        "准确率采用第一轮；时间采用全部预热后轮次。所有推理从 RGB/深度重新计算。", "",
        "## 联合成功率", "",
        "| 子集 | 有效输出 | 2 mm / 2° | 5 mm / 5°（主指标） | 10 mm / 10° |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, group in [("全部", aggregate), *first["by_subset"].items()]:
        cells = [name, f"{group['valid_estimates']}/{group['total']}"]
        for threshold in ("2mm_2deg", "5mm_5deg", "10mm_10deg"):
            success = group["joint_success"][threshold]
            cells.append(f"{success['passed']}/{success['total']}（{success['rate']:.2%}）")
        lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "", "失败、缺失、拒绝输出或退化长轴均计入成功率分母。长轴角度区分头尾，忽略 roll。", "",
        "## 位姿误差", "",
        "| 子集 | 指标 | Median | P90 | P95 | RMSE | Max |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    names = {
        "grasp_error_mm": "抓取中心 / mm", "directed_axis_error_deg": "有向长轴 / °",
        "grasp_axial_error_mm": "轴向位置 / mm", "grasp_lateral_error_mm": "横向位置 / mm",
        "head_error_mm": "头中心 / mm", "tail_error_mm": "尾中心 / mm",
        "semantic_rmse_mm": "三语义点 RMSE / mm",
    }
    for group_name, group in [("全部", aggregate), *first["by_subset"].items()]:
        for key, label in names.items():
            stats = group["errors"][key]
            lines.append("| " + " | ".join([group_name, label, *[f"{stats[field]:.4f}" for field in ("p50", "p90", "p95", "rmse", "max")]]) + " |")
    lines += [
        "", f"严重方向错误（有向轴误差 >90°）：{aggregate['severe_direction_errors_gt90deg']} 条。",
        "原 156 条参考来自当前方法，第一轮的语义点输出与参考完全一致，因此全体中位误差为 0。",
        "原子集的结果主要说明复现一致性；应结合重新配准 20 条的成绩判断改进效果。",
        "这些参考未经独立物理测量，因此本报告不等同于绝对物理精度认证。", "",
        "## 推理时间", "",
        f"共 {timing['warm_online_inference_ms']['count']} 次预热后推理，单位 ms。GPU 在计时边界同步。",
        "分割模型和 CAD 驻留；算法本体计时包含分割、点云提取、配准和相机姿态打包。",
        "算法本体不包括输入读取、模型加载和写盘；完整调用另记输入读取与预测 JSON 保存。", "",
        "| 阶段 | Mean | Median | P95 |",
        "|---|---:|---:|---:|",
    ]
    for key, name in (
        ("segmentation_ms", "分割"), ("visible_points_ms", "点云提取"),
        ("registration_ms", "配准"), ("coordinate_transform_ms", "相机姿态打包"),
        ("warm_online_inference_ms", "算法本体合计"), ("benchmark_call_wall_ms", "完整调用（驻留模型）"),
    ):
        values = timing[key]
        lines.append(f"| {name} | {values['mean']:.3f} | {values['p50']:.3f} | {values['p95']:.3f} |")
    load = metadata["model_loading_s"]
    lines += [
        "", f"模型构造耗时单独记录：分割 {load['segmentation'] * 1000:.3f} ms，CAD {load['cad'] * 1000:.3f} ms。",
        "这两个值不包含首帧推理初始化；首帧初始化已通过预热排除。", "",
        "## 本机重复性", "",
        f"运行轮数：{stability['run_count']}。同帧两两比较共 {stability['pairwise_grasp_delta_mm']['count']} 对。",
        f"最大抓取中心差：{stability['pairwise_grasp_delta_mm']['max']:.9f} mm；最大轴角差：{stability['pairwise_directed_axis_delta_deg']['max']:.9f}°（浮点舍入量级）。",
        f"主成功率标准差：{stability['success_rate_std']:.9f}。",
        "这是本机运行结果，尚未测量另一台服务器的重复性。", "",
        "## 5 mm / 5° 未通过样本", "",
        "| 样本 | 抓取中心误差 / mm | 长轴角误差 / ° |",
        "|---|---:|---:|",
    ]
    for frame in first["frames"]:
        if not frame["joint_success"]["5mm_5deg"]:
            position = frame.get("grasp_error_mm")
            angle = frame.get("directed_axis_error_deg")
            lines.append(f"| {frame['sample_id']} | {position:.3f} | {angle:.3f} |")
    lines += [
        "", "## 产物", "",
        "- `metrics.json`、`per_sample.csv`：第一轮逐帧指标、分组汇总和时间。",
        "- `baseline_summary.json`：首轮基线、所有轮次汇总和重复性摘要。",
        "- `repeatability.json`：逐帧重复性。",
        "- `evaluations/run_XX/`：每轮 JSON 和 CSV。",
        "- `predictions/run_XX/<sample_id>/rgb_6d_base.json`：每轮真实推理输出。",
        "- `run_metadata.json`、`code_snapshot/`：配置、硬件、版本、数据/模型哈希和实际运行代码。", "",
        "复现命令见 `evaluation/CAMERA_EVALUATION_METRICS.md`。比较未来方法时使用同一冻结评测集、",
        "内参、阈值和计时协议；运行输出目录须使用新名称。", "",
    ]
    (output / "BASELINE_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    summarize(args.run_dir)
    print(args.run_dir / "BASELINE_REPORT.md")


if __name__ == "__main__":
    main()
