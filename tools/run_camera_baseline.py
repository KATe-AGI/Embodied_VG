#!/usr/bin/env python3
"""Run the current single-frame pipeline on a frozen camera evaluation manifest."""

from __future__ import annotations

import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
import copy
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import infer_6d_single as single  # noqa: E402
from evaluation.evaluate_camera_benchmark import (  # noqa: E402
    evaluate_dataset, repeatability, summarize, write_report,
)
from plug_vg.grasp_model import load_grasp_model  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def seed_runtime(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "test_data/plug_camera_eval_20261106")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmup-frames", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--registration-method", choices=("symmetric", "legacy"), default="symmetric")
    single.add_point_filter_arguments(parser)
    parser.add_argument("--voxel-size", type=float, default=0.004)
    parser.add_argument("--save-ply", action="store_true", help="Export full cleaned and registration clouds for each frame.")
    parser.add_argument("--save-review", action="store_true", help="Export interactive registration review and comparison PLY.")
    args = parser.parse_args()
    if args.repeats < 1 or args.warmup_frames < 1:
        parser.error("At least one run and one warmup frame are required")
    dataset, output = args.dataset.resolve(), args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite a baseline run: {output}")
    with (dataset / "manifest.csv").open(newline="", encoding="utf-8") as stream:
        manifest = list(csv.DictReader(stream))
    if not manifest or len({row["sample_id"] for row in manifest}) != len(manifest):
        raise ValueError("Evaluation manifest must contain distinct sample IDs")
    integrity = json.loads((dataset / "integrity.json").read_text())["files"]
    for name, digest in integrity.items():
        if sha256(dataset / name) != digest:
            raise ValueError(f"Frozen dataset integrity failed: {name}")

    output.mkdir(parents=True)
    logical_devices = single._cuda_device_indices(args.device, torch)

    def synchronized_timer() -> float:
        for index in logical_devices:
            torch.cuda.synchronize(index)
        return time.perf_counter()

    seed_runtime(args.seed)
    first = manifest[0]
    defaults = single.parse_args([
        "--rgb", str(dataset / first["rgb"]),
        "--d2rgb", str(dataset / first["depth"]),
        "--camera-config", str(dataset / first["camera_config"]),
        "--model-config", str(dataset / "model/plugCAD.yaml"),
        "--output-dir", str(output / "warmup"), "--camera-only",
        "--device", args.device, "--registration-method", args.registration_method,
        "--seed", str(args.seed),
        "--point-filter", args.point_filter,
        "--sor-neighbors", str(args.sor_neighbors),
        "--sor-std-ratio", str(args.sor_std_ratio),
        "--voxel-size", str(args.voxel_size),
        "--bilateral-neighbors", str(args.bilateral_neighbors),
        "--bilateral-sharpness-angle", str(args.bilateral_sharpness_angle),
    ])

    defaults.save_ply = args.save_ply
    defaults.save_review = args.save_review

    def frame_args(row: dict, destination: Path, seed: int) -> argparse.Namespace:
        values = copy.copy(defaults)
        values.rgb = dataset / row["rgb"]
        values.d2rgb = dataset / row["depth"]
        values.camera_config = dataset / row["camera_config"]
        values.output_dir = destination
        values.seed = seed
        return values

    start = synchronized_timer()
    segmenter = single.YOLO(str(defaults.seg_weights))
    segmentation_load_s = synchronized_timer() - start
    start = synchronized_timer()
    grasp_model = load_grasp_model(defaults.model_config)
    cad_load_s = synchronized_timer() - start
    snapshot = output / "code_snapshot"
    code_files = [
        "infer_6d_single.py", "evaluation/evaluate_camera_pose.py",
        "evaluation/evaluate_camera_benchmark.py", "tools/run_camera_baseline.py",
        *[str(path.relative_to(ROOT)) for path in sorted((ROOT / "plug_vg").glob("*.py"))],
    ]
    for relative in code_files:
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    metadata = {
        "date": datetime.now(ZoneInfo("Asia/Shanghai")).date().isoformat(), "timezone": "Asia/Shanghai",
        "dataset": str(dataset), "dataset_integrity_sha256": sha256(dataset / "integrity.json"),
        "sample_count": len(manifest), "repeat_count": args.repeats,
        "seeds": [args.seed + index for index in range(args.repeats)],
        "warmup_frames": args.warmup_frames,
        "registration_method": args.registration_method,
        "inference_defaults": {key: str(value) if isinstance(value, Path) else value for key, value in vars(defaults).items()},
        "segmentation_weights_sha256": sha256(defaults.seg_weights),
        "cad_points_sha256": sha256(grasp_model.pointcloud_path),
        "code_sha256": {relative: sha256(ROOT / relative) for relative in code_files},
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
        "platform": platform.platform(), "python": sys.version,
        "cpu_count": __import__("os").cpu_count(),
        "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
        "packages": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "torch", "opencv-python", "PyYAML")},
        "torch_cuda": torch.version.cuda, "cudnn_version": torch.backends.cudnn.version(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "gpu": [torch.cuda.get_device_name(index) for index in logical_devices],
        "model_loading_s": {"segmentation": segmentation_load_s, "cad": cad_load_s},
        "timing_protocol": {
            "clock": "perf_counter_with_cuda_synchronize_at_stage_boundaries",
            "batch_size": 1, "resident_segmentation_model": True, "resident_cad": True,
            "warm_online_inference_s": "segmentation + point extraction + registration + camera pose packaging; excludes RGB/depth read, asset loading, review and file writes",
            "benchmark_call_wall_s": "whole run() call including input reads, prediction JSON writing and requested artifact exports; resident assets",
            "warmup_excluded_from_both_accuracy_and_timing": True,
        },
        "repeatability_scope": "one_machine; global RNG seed changes each run; symmetric solver has no explicit seed argument and may use nondeterministic CUDA backward",
        "reference_scope": "user_accepted_algorithm_poses; original_156 overlap the baseline method; no independent physical accuracy claim",
    }
    try:
        metadata["nvidia_smi"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        metadata["nvidia_smi"] = None
    metadata["packages"]["open3d"] = importlib.metadata.version("open3d")
    if args.point_filter == "bilateral":
        metadata["packages"]["CGAL"] = importlib.metadata.version("CGAL")
    write_json(output / "run_metadata.json", metadata)

    for index in range(args.warmup_frames):
        row = manifest[index % len(manifest)]
        call_args = frame_args(row, output / "warmup" / f"frame_{index + 1:02d}", args.seed)
        result, _ = single.run(call_args, seg_model=segmenter, grasp_model=grasp_model, timer=synchronized_timer)
        if result.get("status") != "ok":
            raise RuntimeError(f"Warmup failed: {row['sample_id']}: {result.get('reason')}")
        print(f"Warmup {index + 1}/{args.warmup_frames}", flush=True)

    reports = []
    session_start = synchronized_timer()
    for repeat in range(args.repeats):
        run_name = f"run_{repeat + 1:02d}"
        seed = args.seed + repeat
        seed_runtime(seed)
        predictions = output / "predictions" / run_name
        for index, row in enumerate(manifest, 1):
            call_args = frame_args(row, predictions / row["sample_id"], seed)
            start = synchronized_timer()
            try:
                result, result_path = single.run(
                    call_args, seg_model=segmenter, grasp_model=grasp_model, timer=synchronized_timer
                )
            except Exception as exc:
                result = {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
                result_path = single.output_json_path(call_args.output_dir, call_args.rgb)
                result_path.parent.mkdir(parents=True, exist_ok=True)
                with (output / "exceptions.log").open("a", encoding="utf-8") as stream:
                    stream.write(f"{run_name} {row['sample_id']}\n{traceback.format_exc()}\n")
            elapsed = synchronized_timer() - start
            result.setdefault("timing", {})["benchmark_call_wall_s"] = elapsed
            result["benchmark"] = {"sample_id": row["sample_id"], "run": run_name, "seed": seed}
            write_json(result_path, result)
            if index % 25 == 0 or index == len(manifest):
                print(f"{run_name} {index}/{len(manifest)} elapsed={synchronized_timer() - session_start:.1f}s", flush=True)
        # Ground truth is first read by the evaluator, after predictions are saved.
        report = evaluate_dataset(dataset, predictions)
        write_report(report, output / "evaluations" / run_name)
        reports.append(report)
        joint = report["aggregate"]["joint_success"]["5mm_5deg"]
        print(f"{run_name}: primary success {joint['passed']}/{joint['total']} ({joint['rate']:.2%})", flush=True)

    write_report(reports[0], output)
    stability = repeatability(reports)
    write_json(output / "repeatability.json", stability)
    pooled = [frame for report in reports for frame in report["frames"]]
    summary = {
        "primary_baseline": reports[0]["aggregate"],
        "by_subset_first_run": reports[0]["by_subset"],
        "pooled_all_runs": summarize(pooled),
        "repeat_run_summaries": [report["aggregate"] for report in reports],
        "repeatability_summary": {key: value for key, value in stability.items() if key != "per_sample"},
        "total_measured_session_s": synchronized_timer() - session_start,
    }
    write_json(output / "baseline_summary.json", summary)
    print(json.dumps({"output": str(output), "baseline": summary["primary_baseline"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
