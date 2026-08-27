#!/usr/bin/env python3
"""Evaluate camera-frame grasp/tail/head predictions against manual ground truth."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


SEMANTIC_NAMES = ("grasp_center", "tail_center", "head_center")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def _semantic_array(record: dict[str, Any]) -> np.ndarray | None:
    semantic = record.get("semantic_points_camera")
    if not isinstance(semantic, dict):
        return None
    points = []
    for name in SEMANTIC_NAMES:
        value = np.asarray(semantic.get(f"{name}_camera_m"), dtype=np.float64)
        if value.shape != (3,) or not np.all(np.isfinite(value)):
            return None
        points.append(value)
    return np.asarray(points, dtype=np.float64)


def evaluate_pair(ground_truth: dict[str, Any], prediction: dict[str, Any]) -> dict[str, Any]:
    gt_points = _semantic_array(ground_truth)
    if gt_points is None:
        raise ValueError("Ground truth is missing finite camera-frame semantic points")
    pred_points = _semantic_array(prediction)
    if pred_points is None:
        return {
            "prediction_status": prediction.get("status"),
            "has_estimate": False,
            "semantic_rmse_mm": None,
        }

    errors_mm = np.linalg.norm(pred_points - gt_points, axis=1) * 1000.0
    gt_axis = gt_points[2] - gt_points[1]
    pred_axis = pred_points[2] - pred_points[1]
    gt_axis /= np.linalg.norm(gt_axis)
    pred_axis /= np.linalg.norm(pred_axis)
    directed_axis_error_deg = math.degrees(math.acos(float(np.clip(np.dot(gt_axis, pred_axis), -1.0, 1.0))))

    grasp_delta_mm = (pred_points[0] - gt_points[0]) * 1000.0
    grasp_axial_error_mm = abs(float(np.dot(grasp_delta_mm, gt_axis)))
    grasp_lateral_error_mm = float(
        np.linalg.norm(grasp_delta_mm - np.dot(grasp_delta_mm, gt_axis) * gt_axis)
    )
    return {
        "prediction_status": prediction.get("status"),
        "has_estimate": True,
        "grasp_error_mm": float(errors_mm[0]),
        "tail_error_mm": float(errors_mm[1]),
        "head_error_mm": float(errors_mm[2]),
        "semantic_rmse_mm": float(np.sqrt(np.mean(np.square(errors_mm)))),
        "directed_axis_error_deg": directed_axis_error_deg,
        "grasp_axial_error_mm": grasp_axial_error_mm,
        "grasp_lateral_error_mm": grasp_lateral_error_mm,
    }


def evaluate_directory(
    gt_dir: Path,
    prediction_dir: Path,
    timing_warmup_frames: int = 1,
) -> dict[str, Any]:
    gt_paths = sorted(gt_dir.glob("*_grasp_gt.json"))
    if not gt_paths:
        raise FileNotFoundError(f"No *_grasp_gt.json files found in {gt_dir}")

    frames: list[dict[str, Any]] = []
    all_point_errors_mm: list[float] = []
    warm_online_times_s: list[float] = []
    missing_or_invalid = 0
    for gt_path in gt_paths:
        sample_id = gt_path.name.removesuffix("_grasp_gt.json")
        prediction_path = prediction_dir / f"{sample_id}_color_6d_base.json"
        if prediction_path.is_file():
            prediction = _read_json(prediction_path)
            metrics = evaluate_pair(_read_json(gt_path), prediction)
        else:
            prediction = {}
            metrics = {
                "prediction_status": "missing",
                "has_estimate": False,
                "semantic_rmse_mm": None,
            }
        frame = {
            "sample_id": sample_id,
            "ground_truth": str(gt_path),
            "prediction": str(prediction_path),
            **metrics,
        }
        frames.append(frame)
        if metrics["has_estimate"]:
            all_point_errors_mm.extend(
                [
                    float(metrics["grasp_error_mm"]),
                    float(metrics["tail_error_mm"]),
                    float(metrics["head_error_mm"]),
                ]
            )
            timing = prediction.get("timing") or {}
            warm_time = timing.get("warm_online_inference_s")
            if warm_time is not None and np.isfinite(float(warm_time)):
                warm_online_times_s.append(float(warm_time))
        else:
            missing_or_invalid += 1

    errors = np.asarray(all_point_errors_mm, dtype=np.float64)
    all_times = np.asarray(warm_online_times_s, dtype=np.float64)
    skip = min(max(0, int(timing_warmup_frames)), len(all_times))
    times = all_times[skip:]
    aggregate = {
        "annotated_frames": len(gt_paths),
        "frames_with_estimate": len(gt_paths) - missing_or_invalid,
        "missing_or_invalid_estimates": missing_or_invalid,
        "global_semantic_rmse_mm": None
        if missing_or_invalid or not len(errors)
        else float(np.sqrt(np.mean(np.square(errors)))),
        "point_error_mean_mm": None if not len(errors) else float(np.mean(errors)),
        "point_error_median_mm": None if not len(errors) else float(np.median(errors)),
        "point_error_max_mm": None if not len(errors) else float(np.max(errors)),
        "timing_warmup_frames_excluded": skip,
        "warm_online_p50_s": None if not len(times) else float(np.percentile(times, 50)),
        "warm_online_p95_s": None if not len(times) else float(np.percentile(times, 95)),
    }
    return {
        "metric": "camera_frame_grasp_tail_head_global_rmse",
        "unit": "millimeter",
        "ground_truth_dir": str(gt_dir),
        "prediction_dir": str(prediction_dir),
        "aggregate": aggregate,
        "frames": frames,
    }


def compare_evaluations(
    reference: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    """Compare two complete pipelines with frame-level leave-one-out selection."""

    reference_frames = reference.get("frames") or []
    candidate_frames = candidate.get("frames") or []
    reference_by_id = {str(frame["sample_id"]): frame for frame in reference_frames}
    candidate_by_id = {str(frame["sample_id"]): frame for frame in candidate_frames}
    if reference_by_id.keys() != candidate_by_id.keys():
        raise ValueError("Reference and candidate evaluations must contain the same sample IDs")
    sample_ids = sorted(reference_by_id)
    if len(sample_ids) < 2:
        raise ValueError("Leave-one-out comparison requires at least two annotated frames")

    reference_rmse = np.asarray(
        [reference_by_id[sample_id].get("semantic_rmse_mm") for sample_id in sample_ids],
        dtype=np.float64,
    )
    candidate_rmse = np.asarray(
        [candidate_by_id[sample_id].get("semantic_rmse_mm") for sample_id in sample_ids],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(reference_rmse)) or not np.all(np.isfinite(candidate_rmse)):
        raise ValueError("Both evaluations need finite estimates for every annotated frame")

    def aggregate(values: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.square(values))))

    folds = []
    candidate_selected_folds = 0
    held_out_candidate_better_folds = 0
    for held_out_index, sample_id in enumerate(sample_ids):
        train_mask = np.arange(len(sample_ids)) != held_out_index
        reference_train = aggregate(reference_rmse[train_mask])
        candidate_train = aggregate(candidate_rmse[train_mask])
        selected = "candidate" if candidate_train < reference_train else "reference"
        candidate_selected_folds += int(selected == "candidate")
        held_out_candidate_better_folds += int(
            candidate_rmse[held_out_index] < reference_rmse[held_out_index]
        )
        folds.append(
            {
                "held_out_sample_id": sample_id,
                "reference_train_rmse_mm": reference_train,
                "candidate_train_rmse_mm": candidate_train,
                "selected_pipeline": selected,
                "reference_held_out_rmse_mm": float(reference_rmse[held_out_index]),
                "candidate_held_out_rmse_mm": float(candidate_rmse[held_out_index]),
            }
        )

    reference_global = aggregate(reference_rmse)
    candidate_global = aggregate(candidate_rmse)
    return {
        "selection_protocol": "leave_one_frame_out_select_lower_train_global_semantic_rmse",
        "fold_count": len(folds),
        "reference_global_semantic_rmse_mm": reference_global,
        "candidate_global_semantic_rmse_mm": candidate_global,
        "candidate_delta_mm": candidate_global - reference_global,
        "candidate_selected_folds": candidate_selected_folds,
        "held_out_candidate_better_folds": held_out_candidate_better_folds,
        "module_accepted": bool(
            candidate_global < reference_global and candidate_selected_folds == len(folds)
        ),
        "folds": folds,
    }


def _write_csv(path: Path, frames: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "prediction_status",
        "has_estimate",
        "grasp_error_mm",
        "tail_error_mm",
        "head_error_mm",
        "semantic_rmse_mm",
        "directed_axis_error_deg",
        "grasp_axial_error_mm",
        "grasp_lateral_error_mm",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(frames)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-dir", type=Path, default=Path("test_20260814"))
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument(
        "--reference-dir",
        type=Path,
        help="Optional previous-module predictions for leave-one-out comparison.",
    )
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--timing-warmup-frames", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = evaluate_directory(
        args.gt_dir,
        args.prediction_dir,
        timing_warmup_frames=args.timing_warmup_frames,
    )
    if args.reference_dir is not None:
        reference = evaluate_directory(
            args.gt_dir,
            args.reference_dir,
            timing_warmup_frames=args.timing_warmup_frames,
        )
        report["comparison"] = compare_evaluations(reference, report)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    if args.output_csv is not None:
        _write_csv(args.output_csv, report["frames"])
    summary = {"aggregate": report["aggregate"]}
    if "comparison" in report:
        summary["comparison"] = {
            key: value
            for key, value in report["comparison"].items()
            if key != "folds"
        }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
