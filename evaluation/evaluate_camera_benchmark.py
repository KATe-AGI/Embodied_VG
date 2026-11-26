#!/usr/bin/env python3
"""Evaluate frozen manifest datasets using directed axes, joint success, and time."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.evaluate_camera_pose import evaluate_pair  # noqa: E402


THRESHOLDS = ((2.0, 2.0), (5.0, 5.0), (10.0, 10.0))
ERROR_NAMES = (
    "grasp_error_mm", "directed_axis_error_deg", "grasp_axial_error_mm",
    "grasp_lateral_error_mm", "head_error_mm", "tail_error_mm", "semantic_rmse_mm",
)
TIMING_NAMES = (
    "warm_online_inference_s", "segmentation_s", "visible_points_s",
    "registration_s", "coordinate_transform_s", "benchmark_call_wall_s",
)


def distribution(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    if not len(array):
        return {"count": 0, **dict.fromkeys(("mean", "p50", "p90", "p95", "rmse", "max"))}
    return {
        "count": len(array), "mean": float(np.mean(array)),
        "p50": float(np.percentile(array, 50)), "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)), "rmse": float(np.sqrt(np.mean(array ** 2))),
        "max": float(np.max(array)),
    }


def threshold_name(position_mm: float, angle_deg: float) -> str:
    return f"{position_mm:g}mm_{angle_deg:g}deg"


def summarize(frames: list[dict]) -> dict:
    valid = [frame for frame in frames if frame["has_estimate"]]
    total = len(frames)
    success = {}
    for position, angle in THRESHOLDS:
        name = threshold_name(position, angle)
        count = sum(frame["joint_success"][name] for frame in frames)
        success[name] = {"passed": count, "total": total, "rate": count / total if total else None}
    timing = {}
    for name in TIMING_NAMES:
        values = [frame["timing"][name] * 1000 for frame in frames if name in frame["timing"]]
        timing[name.removesuffix("_s") + "_ms"] = distribution(values)
    return {
        "total": total, "valid_estimates": len(valid),
        "missing_or_failed_or_invalid": total - len(valid),
        "estimate_rate": len(valid) / total if total else None,
        "errors": {name: distribution([frame[name] for frame in valid]) for name in ERROR_NAMES},
        "global_semantic_rmse_mm": (
            math.sqrt(float(np.mean([frame["semantic_rmse_mm"] ** 2 for frame in valid])))
            if valid else None
        ),
        "joint_success": success,
        "severe_direction_errors_gt90deg": sum(frame["directed_axis_error_deg"] > 90 for frame in valid),
        "timing": timing,
    }


def evaluate_dataset(dataset: Path, predictions: Path) -> dict:
    with (dataset / "manifest.csv").open(newline="", encoding="utf-8") as stream:
        manifest = list(csv.DictReader(stream))
    if not manifest or len({row["sample_id"] for row in manifest}) != len(manifest):
        raise ValueError("Manifest must contain distinct sample IDs")
    frames = []
    for row in manifest:
        path = predictions / row["sample_id"] / "rgb_6d_base.json"
        prediction = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"status": "missing"}
        truth = json.loads((dataset / row["ground_truth_json"]).read_text(encoding="utf-8"))
        metrics = evaluate_pair(truth, prediction)
        # An explicitly failed/rejected estimate never counts as successful.
        metrics["has_estimate"] = bool(metrics["has_estimate"] and prediction.get("status") == "ok")
        timing = {}
        for key in TIMING_NAMES:
            value = (prediction.get("timing") or {}).get(key)
            if isinstance(value, (int, float)) and np.isfinite(value) and value >= 0:
                timing[key] = float(value)
        frame = {
            "sample_id": row["sample_id"], "group": row["group"], "subset": row["subset"],
            "camera_config": row["camera_config"], "prediction_json": str(path),
            **metrics, "timing": timing,
        }
        frame["joint_success"] = {
            threshold_name(position, angle): bool(
                metrics["has_estimate"]
                and metrics["grasp_error_mm"] <= position
                and metrics["directed_axis_error_deg"] <= angle
            )
            for position, angle in THRESHOLDS
        }
        if metrics["has_estimate"]:
            semantic = prediction["semantic_points_camera"]
            head = np.asarray(semantic["head_center_camera_m"], dtype=float)
            tail = np.asarray(semantic["tail_center_camera_m"], dtype=float)
            direction = head - tail
            frame["axis_camera"] = (direction / np.linalg.norm(direction)).tolist()
            frame["grasp_center_camera_m"] = semantic["grasp_center_camera_m"]
        frames.append(frame)
    grouped = {}
    for key in ("subset", "group", "camera_config"):
        groups = defaultdict(list)
        for frame in frames:
            groups[frame[key]].append(frame)
        grouped[key] = {name: summarize(group) for name, group in sorted(groups.items())}
    return {
        "schema_version": 1, "dataset": str(dataset), "predictions": str(predictions),
        "coordinate_frame": "camera_rgb", "roll_ignored": True,
        "primary_metric": "joint_success_5mm_5deg",
        "success_denominator": "all_manifest_samples_including_missing_failed_and_invalid",
        "error_statistics_population": "finite_estimates_with_status_ok",
        "aggregate": summarize(frames), "by_subset": grouped["subset"],
        "by_group": grouped["group"], "by_camera": grouped["camera_config"], "frames": frames,
    }


def repeatability(reports: list[dict]) -> dict:
    if len(reports) < 2:
        return {"measured": False, "reason": "requires_at_least_two_runs"}
    runs = [{frame["sample_id"]: frame for frame in report["frames"]} for report in reports]
    ids = set(runs[0])
    if any(set(run) != ids for run in runs):
        raise ValueError("Repeat runs must cover the same manifest")
    positions, angles, per_sample = [], [], []
    for sample_id in sorted(ids):
        sample_positions, sample_angles = [], []
        for left in range(len(runs)):
            for right in range(left + 1, len(runs)):
                a, b = runs[left][sample_id], runs[right][sample_id]
                if not (a["has_estimate"] and b["has_estimate"]):
                    continue
                delta = np.asarray(a["grasp_center_camera_m"]) - np.asarray(b["grasp_center_camera_m"])
                position = float(np.linalg.norm(delta) * 1000)
                angle = math.degrees(math.acos(float(np.clip(np.dot(a["axis_camera"], b["axis_camera"]), -1, 1))))
                sample_positions.append(position)
                sample_angles.append(angle)
        positions.extend(sample_positions)
        angles.extend(sample_angles)
        per_sample.append({
            "sample_id": sample_id, "valid_pair_count": len(sample_positions),
            "max_grasp_delta_mm": max(sample_positions, default=None),
            "max_axis_delta_deg": max(sample_angles, default=None),
            "successful_runs_5mm_5deg": sum(run[sample_id]["joint_success"]["5mm_5deg"] for run in runs),
        })
    rates = [report["aggregate"]["joint_success"]["5mm_5deg"]["rate"] for report in reports]
    return {
        "measured": True, "machine_scope": "same_machine", "run_count": len(reports),
        "expected_pair_count": len(ids) * len(runs) * (len(runs) - 1) // 2,
        "pairwise_grasp_delta_mm": distribution(positions),
        "pairwise_directed_axis_delta_deg": distribution(angles),
        "success_5mm_5deg_rate_by_run": rates,
        "success_rate_std": float(np.std(rates)), "per_sample": per_sample,
    }


def write_report(report: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    with (output / "per_sample.csv").open("w", newline="", encoding="utf-8") as stream:
        columns = ["sample_id", "group", "subset", "prediction_status", "has_estimate", *ERROR_NAMES]
        columns += ["success_" + threshold_name(*pair) for pair in THRESHOLDS]
        columns += [name.removesuffix("_s") + "_ms" for name in TIMING_NAMES]
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for frame in report["frames"]:
            writer.writerow({
                **frame,
                **{"success_" + name: value for name, value in frame["joint_success"].items()},
                **{name.removesuffix("_s") + "_ms": value * 1000 for name, value in frame["timing"].items()},
            })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate_dataset(args.dataset.resolve(), args.prediction_dir.resolve())
    write_report(report, args.output_dir)
    print(json.dumps(report["aggregate"], indent=2))


if __name__ == "__main__":
    main()
