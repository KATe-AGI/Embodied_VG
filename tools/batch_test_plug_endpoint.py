#!/usr/bin/env python3
"""Run endpoint inference for every local gripper-calibration capture."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_pose(path: Path) -> list[float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    values = data.get("robot_pose_xyzrpy_m_rad")
    if values is None:
        values = data.get("robot_pose")
    if not isinstance(values, list) or len(values) != 6:
        raise ValueError(f"invalid robot pose in {path}")
    return [float(value) for value in values]


def load_hand_eye(output_dir: Path, temp_dir: Path) -> Path:
    response = output_dir / "response.json"
    if not response.is_file():
        raise FileNotFoundError(f"missing response.json: {response}")
    data = json.loads(response.read_text(encoding="utf-8"))
    matrix = data.get("embodied_result", {}).get("frame_transform", {}).get("T_end_camera")
    if matrix is None:
        raise ValueError(f"missing T_end_camera in {response}")
    path = temp_dir / f"{output_dir.name}_hand_eye.yaml"
    path.write_text(
        "transform_name: T_end_camera\n"
        f"matrix_4x4: {json.dumps(matrix)}\n",
        encoding="utf-8",
    )
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=ROOT / "output/batch_endpoint_20261031")
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.5)
    args = parser.parse_args()
    inputs = sorted(
        path for path in (ROOT / "test_data").glob("*/input/*_gripper_calibration")
        if path.is_dir()
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    with tempfile.TemporaryDirectory(prefix="endpoint_hand_eye_") as temp:
        temp_dir = Path(temp)
        for index, input_dir in enumerate(inputs, 1):
            case = input_dir.name
            output_dir = args.output_root / case
            output_dir.mkdir(parents=True, exist_ok=True)
            row = {"case": case, "date": input_dir.parent.parent.name, "input": str(input_dir), "status": "setup_failed"}
            try:
                rgb = input_dir / "rgb.png"
                depth = input_dir / "d2rgb.npy"
                if not depth.is_file():
                    depth = input_dir / "depth.npy"
                pose_path = input_dir / "flange_pose.json"
                if not rgb.is_file() or not depth.is_file() or not pose_path.is_file():
                    raise FileNotFoundError("missing rgb.png, d2rgb.npy, or flange_pose.json")
                pose = load_pose(pose_path)
                hand_eye = load_hand_eye(ROOT / "test_data" / input_dir.parent.parent.name / "output" / case, temp_dir)
                command = [
                    "python", "infer_plug_endpoint_single.py", "--rgb", str(rgb), "--d2rgb", str(depth),
                    "--robot-pose", *[str(value) for value in pose], "--hand-eye-config", str(hand_eye),
                    "--output-dir", str(output_dir), "--device", args.device, "--conf", str(args.conf),
                ]
                completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
                row["returncode"] = completed.returncode
                json_path = output_dir / f"{rgb.stem}_plug_endpoint.json"
                if json_path.is_file():
                    result = json.loads(json_path.read_text(encoding="utf-8"))
                    row.update({"status": result.get("status"), "reason": result.get("reason", ""),
                                "result_json": str(json_path)})
                    quality = result.get("quality", {})
                    row.update({key: quality.get(key) for key in (
                        "model", "valid_coverage", "valid_samples", "candidate_surface_pixels",
                        "depth_median_m", "depth_scaled_mad_m", "fit_rmse_m", "validation_rmse_m",
                    )})
                else:
                    row["reason"] = (completed.stderr or completed.stdout)[-1000:]
            except Exception as exc:
                row["reason"] = str(exc)
            rows.append(row)
            print(f"[{index}/{len(inputs)}] {case}: {row['status']} {row.get('reason', '')}", flush=True)
    summary_path = args.output_root / "summary.json"
    summary_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    counts = Counter(row["status"] for row in rows)
    (args.output_root / "summary.txt").write_text(
        f"cases={len(rows)}\nstatus_counts={dict(counts)}\nsummary_json={summary_path}\n",
        encoding="utf-8",
    )
    print(json.dumps({"cases": len(rows), "status_counts": dict(counts), "summary": str(summary_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
