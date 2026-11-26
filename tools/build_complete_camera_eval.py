#!/usr/bin/env python3
"""Freeze the 156 earlier references and 20 newly accepted poses in one dataset."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "test_data/plug_6d_camera_benchmark_20261105"
CORRECTIONS = ROOT / "output/unstable_plug_camera_registrations_20261106"
DEFAULT_OUTPUT = ROOT / "test_data/plug_camera_eval_20261106"


def read_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_pose(record: dict, semantic_model: dict) -> None:
    transform = np.asarray(record["t_camera_grasp"], dtype=float)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("Invalid camera-frame transform")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise ValueError("Rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1, atol=1e-6):
        raise ValueError("Rotation is not right handed")
    if not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-8):
        raise ValueError("Invalid homogeneous transform")
    for name in ("grasp_center", "tail_center", "head_center"):
        expected = rotation @ np.asarray(semantic_model[name]) + transform[:3, 3]
        actual = np.asarray(record["semantic_points_camera"][f"{name}_camera_m"], dtype=float)
        if actual.shape != (3,) or not np.allclose(actual, expected, atol=1e-7, rtol=0):
            raise ValueError(f"Pose and semantic point disagree: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite frozen dataset: {output}")

    previous_gt = read_json(BENCHMARK / "ground_truth_camera.json")
    previous = {item["sample_id"]: item for item in previous_gt["references"]}
    correction_ids = set(read_json(BENCHMARK / "review_selection.json")["unstable_sample_ids"])
    correction_summary = read_json(CORRECTIONS / "summary.json")
    if len(correction_summary) != 20 or {item["sample_id"] for item in correction_summary} != correction_ids:
        raise ValueError("Corrected output does not contain exactly the 20 reviewed samples")
    with (BENCHMARK / "candidates.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    all_ids = {row["sample_id"] for row in rows}
    if len(previous) != 156 or len(rows) != 176 or len(all_ids) != len(rows):
        raise ValueError("Expected 156 previous references and 176 unique total samples")
    if set(previous) & correction_ids or set(previous) | correction_ids != all_ids:
        raise ValueError("Earlier and newly accepted sets must exactly partition the dataset")

    model_config_path = ROOT / "configs/plug_models/plugCAD.yaml"
    model_config = yaml.safe_load(model_config_path.read_text(encoding="utf-8"))
    semantic_model = model_config["semantic_points_grasp_m"]
    plans = []
    # Validate all sources before creating the output dataset.
    for row in rows:
        sample_id = row["sample_id"]
        corrected = sample_id in correction_ids
        source_result = (
            CORRECTIONS / sample_id / "registration.json"
            if corrected else ROOT / row["result_json"]
        )
        source_record = read_json(source_result)
        truth = source_record if corrected else previous[sample_id]
        validate_pose(truth, semantic_model)
        if not corrected:
            if sha256(source_result) != truth["source_result_sha256"]:
                raise ValueError(f"Earlier reference source changed: {sample_id}")
            for key in ("t_camera_grasp", "semantic_points_camera"):
                if source_record[key] != truth[key]:
                    raise ValueError(f"Earlier reference pose changed: {sample_id}")
        for kind in ("rgb", "depth"):
            expected_hash = row[f"{kind}_sha256"]
            if sha256(ROOT / row[kind]) != expected_hash:
                raise ValueError(f"Input changed: {sample_id}: {kind}")
            if not corrected and truth[f"{kind}_sha256"] != expected_hash:
                raise ValueError(f"Earlier reference input disagrees: {sample_id}: {kind}")
        if corrected:
            source_ply = CORRECTIONS / sample_id / "point_cloud_comparison.ply"
            source_projection = CORRECTIONS / sample_id / "rgb_rgb_projection.jpg"
        else:
            source_ply = ROOT / source_record["artifacts"]["point_cloud_comparison_ply"]
            source_projection = ROOT / source_record["artifacts"]["rgb_projection"]
        for path in (source_ply, source_projection, ROOT / row["camera_config"]):
            if not path.is_file():
                raise FileNotFoundError(path)
        plans.append((row, truth, source_record, source_result, source_ply, source_projection))

    output.mkdir(parents=True)
    cameras = output / "cameras"
    cameras.mkdir()
    calibration_names = {
        "configs/camera/plug_rgbd.yaml": "legacy_d435if.yaml",
        "test_data/plug_6d_camera_benchmark_20261105/camera_261822072212.yaml": "261822072212.yaml",
    }
    for source, name in calibration_names.items():
        shutil.copy2(ROOT / source, cameras / name)
    model_dir = output / "model"
    model_dir.mkdir()
    for name in ("plugCAD_grasp.ply", "plugCAD_grasp.obj", "plugCAD_grasp_report.md", "2175B_vs_plugCAD_dimensions.png"):
        shutil.copy2(ROOT / "plug_model" / name, model_dir / name)
    # Asset paths resolve relative to this YAML after moving the dataset.
    for key, original in model_config["assets"].items():
        model_config["assets"][key] = Path(original).name
    (model_dir / "plugCAD.yaml").write_text(yaml.safe_dump(model_config, sort_keys=False), encoding="utf-8")

    references = []
    manifest = []
    for index, (row, truth, source_record, source_result, source_ply, source_projection) in enumerate(plans, 1):
        sample_id = row["sample_id"]
        corrected = sample_id in correction_ids
        sample_dir = output / "samples" / sample_id
        input_dir = sample_dir / "input"
        input_dir.mkdir(parents=True)
        files = {
            "rgb": (ROOT / row["rgb"], input_dir / "rgb.png"),
            "depth": (ROOT / row["depth"], input_dir / "d2rgb.npy"),
            "source_registration": (source_result, sample_dir / "source_registration.json"),
            "point_cloud_comparison_ply": (source_ply, sample_dir / "point_cloud_comparison.ply"),
            "rgb_projection": (source_projection, sample_dir / "rgb_rgb_projection.jpg"),
        }
        for source, destination in files.values():
            shutil.copy2(source, destination)
        paths = {name: str(destination.relative_to(output)) for name, (_, destination) in files.items()}
        camera_path = f"cameras/{calibration_names[row['camera_config']]}"
        semantic = truth["semantic_points_camera"]
        tail = np.asarray(semantic["tail_center_camera_m"], dtype=float)
        head = np.asarray(semantic["head_center_camera_m"], dtype=float)
        axis = head - tail
        length = float(np.linalg.norm(axis))
        axis /= length
        review_subset = "reregistered_20" if corrected else "original_156"
        camera_note = (
            "legacy_intrinsics; capture_serial_unconfirmed"
            if row["group"] in ("20261020", "20261022")
            else "261822072212_intrinsics; capture_serial_assignment_provisional"
            if row["group"] == "20261023"
            else "legacy_intrinsics; documented_serial_327122072321"
            if row["group"] == "test_20260923"
            else "261822072212_intrinsics"
        )
        record = {
            "sample_id": sample_id,
            "group": row["group"],
            "subset": review_subset,
            "status": "accepted",
            "review_decision": "accepted_as_camera_frame_ground_truth",
            "review_date": "2026-11-06" if corrected else "2026-11-05",
            "review_source": "explicit_user_acceptance",
            "coordinate_frame": "camera_rgb",
            "reference_type": "user_accepted_algorithm_registration",
            "length_unit": "meter",
            "pose_scope": "grasp_center_and_directed_long_axis; roll_is_task_equivalent",
            "camera_config": camera_path,
            "camera_assignment": camera_note,
            "model_config": "model/plugCAD.yaml",
            "rgb": paths["rgb"],
            "depth": paths["depth"],
            "rgb_sha256": row["rgb_sha256"],
            "depth_sha256": row["depth_sha256"],
            "t_camera_grasp": truth["t_camera_grasp"],
            "semantic_points_camera": semantic,
            "tail_to_head_axis_camera": {
                "tail_point_m": tail.tolist(),
                "head_point_m": head.tolist(),
                "direction_unit": axis.tolist(),
                "length_m": length,
            },
            "artifacts": {key: paths[key] for key in ("source_registration", "point_cloud_comparison_ply", "rgb_projection")},
            "registration_diagnostics": (
                source_record.get("quality_comparison", {}).get("selected", {})
                if corrected else source_record.get("registration_quality", {})
            ),
            "historical_review_flags": source_record.get("review_flags", []),
            "provenance": {
                "source_result_json": str(source_result.relative_to(ROOT)),
                "source_result_sha256": sha256(source_result),
                "source_status": source_record["status"],
                "source_rgb": row["rgb"],
                "source_depth": row["depth"],
                "source_camera_config": row["camera_config"],
                "source_camera_sha256": sha256(ROOT / row["camera_config"]),
            },
        }
        gt_path = sample_dir / "ground_truth.json"
        write_json(gt_path, record)
        references.append({**record, "ground_truth_json": str(gt_path.relative_to(output))})
        manifest.append({
            "sample_id": sample_id, "group": row["group"], "subset": review_subset,
            "rgb": paths["rgb"], "depth": paths["depth"],
            "camera_config": camera_path, "ground_truth_json": str(gt_path.relative_to(output)),
            "point_cloud_comparison_ply": paths["point_cloud_comparison_ply"],
            "rgb_projection": paths["rgb_projection"],
        })
        if index % 25 == 0 or index == len(plans):
            print(f"Copied and indexed {index}/{len(plans)}", flush=True)

    write_json(output / "ground_truth_camera.json", {
        "schema_version": 1,
        "coordinate_frame": "camera_rgb",
        "length_unit": "meter",
        "path_base": "dataset_root",
        "reference_type": "user_accepted_algorithm_registrations",
        "review_source": "explicit_user_acceptance_on_2026-11-05_and_2026-11-06",
        "references": references,
    })
    with (output / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    write_json(output / "metadata.json", {
        "dataset_id": "plug_camera_eval_20261106",
        "created_date": "2026-11-06",
        "timezone": "Asia/Shanghai",
        "coordinate_frame": "camera_rgb",
        "path_base": "dataset_root",
        "total": len(references),
        "accepted": len(references),
        "subsets": dict(Counter(item["subset"] for item in references)),
        "groups": dict(Counter(item["group"] for item in references)),
        "cameras": dict(Counter(item["camera_config"] for item in references)),
        "review_authorization": "User accepted all 20 re-registered results on 2026-11-06 and requested their merger with the 156 prior accepted references.",
        "annotation_source": "algorithm_registrations_accepted_by_user; no independent metrology",
        "ground_truth_frozen": True,
        "splits": "not_assigned; preserve capture groups when choosing development and held-out evaluation sets",
        "input_policy": "Byte-identical copies of original RGB and aligned depth; per-sample original calibration preserved.",
        "symmetry": "Ignore roll about grasp X; preserve tail-to-head direction.",
        "build_script_sha256": sha256(Path(__file__)),
        "original_reference_index_sha256": sha256(BENCHMARK / "ground_truth_camera.json"),
        "correction_summary_sha256": sha256(CORRECTIONS / "summary.json"),
    })
    shutil.copy2(ROOT / "evaluation/CAMERA_EVALUATION_METRICS.md", output / "METRICS_PROPOSAL.md")
    (output / "README.md").write_text("""# 插头相机系完整评测集（2026-11-06）

共 **176 条用户已接受的配准真值**：原先 156 条 + 重新配准 20 条。
20 条重新配准结果的接受依据是用户在 2026-11-06 的明确确认。
真值冻结为当前版本，全部只涉及 RGB 相机系。

## 文件结构

```text
manifest.csv                    # 176 条样本的输入、内参、真值及可视化相对路径
ground_truth_camera.json         # 176 条完整参考位姿和语义点汇总
metadata.json                    # 数量、分组、来源、相机说明
integrity.json                   # 除自身之外所有文件的 SHA-256
cameras/                        # 两套原配准使用的相机内参
model/                          # CAD 点云、网格、模型配置和尺寸说明
samples/<sample_id>/
  input/rgb.png                 # 原始 RGB，字节完全一致
  input/d2rgb.npy               # 原始对齐深度，字节完全一致，原始单位 mm
  ground_truth.json             # 正式已接受真值，位姿单位 m
  source_registration.json      # 原始算法结果存档，保留原文件状态和诊断
  point_cloud_comparison.ply    # 观测点云与配准 CAD，方便人工查看
  rgb_rgb_projection.jpg        # 分割轮廓及头/抓取/尾投影，方便人工查看
```

所有正式索引路径均相对于**本评测集根目录**，复制整个文件夹即可保留输入、
内参、真值和可视化的可用性。`provenance` 内的历史仓库路径仅用于追溯，
不参与读取评测输入。`source_registration.json` 是逐字节存档，内部仍可能引用
原仓库路径或保留 `candidate_manual_review_pending`；当前接受状态以
`ground_truth.json` 和根目录 `ground_truth_camera.json` 为准。

`t_camera_grasp` 将抓取坐标系点转换到 RGB 相机系：
`p_camera = R_camera_grasp @ p_grasp + t_camera_grasp`。
相机坐标为 X 向右、Y 向下、Z 向前，长度单位为米。
抓取原点距离头中心 90 mm，距离尾中心 77.8 mm。
绕插头长轴的 roll 是任务等价自由度；头尾方向仍须区分。

## 相机内参

- `test_20260923`、`20261020`、`20261022`：`cameras/legacy_d435if.yaml`，
  fx=1359.2、fy=1358.4、cx=984.7、cy=552.1。
- `20261023` 及以后：`cameras/261822072212.yaml`，
  fx=1365.9、fy=1365.6、cx=976.9、cy=551.1。
- 两套输入均为 1920×1080、RGB 对齐深度、depth_scale=0.001。
- September 的相机记录为 327122072321；October 20–22 的序列号未在输入中确认，
  October 23 的内参归属也是历史暂定。每条真值的 `camera_assignment` 记录此信息。
  本版保留原真值使用的内参，不因当前相机更换而重算既有参考。

## 评测与复核

两个来源子集命名为 `original_156` 和 `reregistered_20`，两者均已被用户接受为真值。
历史质量标记保留于 `historical_review_flags`，只供查看。
本集是用户接受的算法配准，未经过独立物理测量。
176 条目前尚未分配开发/留出划分，应结合采集序列和场景分组后固定。
指标建议见 [METRICS_PROPOSAL.md](METRICS_PROPOSAL.md)，阈值尚待讨论确定。
当前旧版评测命令的文件布局与本集不同，尚未改写其目录入口。

PLY 颜色：绿色为观测点云，红色为 CAD；黄色为抓取中心，青色为尾中心，品红为头中心。
RGB 图的头尾箭头、文字及掩膜仅用于人工检查。

构建脚本：`tools/build_complete_camera_eval.py`。为防止意外覆盖冻结数据，
脚本要求输出目录不存在。原 156 条来自上次冻结参考，20 条来自新配准输出；
构建过程中验证输入及历史结果哈希，并逐字节复制原文件。
""", encoding="utf-8")
    write_json(output / "integrity.json", {
        "algorithm": "sha256",
        "path_base": "dataset_root",
        "files": {
            str(path.relative_to(output)): sha256(path)
            for path in sorted(output.rglob("*")) if path.is_file()
        },
    })
    print(f"Created {len(references)} accepted camera-frame references at {output}")


if __name__ == "__main__":
    main()
