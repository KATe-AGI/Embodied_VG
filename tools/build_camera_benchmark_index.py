#!/usr/bin/env python3
"""Index camera-frame batch predictions for manual ground-truth selection."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "ultralytics/runs/segment/plug_yolo26n_seg_20260814/weights/best.pt"
CAD = ROOT / "plug_model/plugCAD_grasp.ply"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark_dir", type=Path)
    args = parser.parse_args()
    root = args.benchmark_dir
    index_path = root / "candidates.csv"
    selection_path = root / "review_selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8")) if selection_path.is_file() else None
    unstable_list = selection.get("unstable_sample_ids", []) if selection else []
    if len(unstable_list) != len(set(unstable_list)):
        raise ValueError("review_selection.json contains duplicate sample IDs")
    unstable_ids = set(unstable_list)
    reviewed = {}
    if index_path.is_file():
        with index_path.open(newline="", encoding="utf-8") as stream:
            reviewed = {
                (row["group"], row["sample_id"]): row
                for row in csv.DictReader(stream)
            }
    rows = []
    groups = []
    references = []
    for summary_path in sorted(root.glob("*/summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        samples = [sample for sample in summary["samples"] if sample["status"] == "ok"]
        groups.append({
            "group": summary_path.parent.name,
            "total": len(samples),
            "ok": len(samples),
            "failed": 0,
            "camera_config": summary["camera_config"],
        })
        for sample in samples:
            result_path = Path(sample["result_json"])
            if result_path.is_file():
                result = json.loads(result_path.read_text(encoding="utf-8"))
            elif sample["status"] != "ok":
                result = {}
            else:
                raise FileNotFoundError(f"Successful sample result is missing: {result_path}")
            quality = result.get("registration_quality") or {}
            points = result.get("visible_point_cloud") or {}
            semantic = result.get("semantic_points_camera") or {}
            rgb_hash = sha256(Path(sample["rgb"]))
            depth_hash = sha256(Path(sample["depth"]))
            existing = reviewed.get((summary_path.parent.name, sample["sample_id"]), {})
            decision = "unstable" if sample["sample_id"] in unstable_ids else existing.get("review_decision", "")
            if not decision and selection:
                decision = "accepted"
            if decision == "accepted":
                if not result.get("t_camera_grasp") or not semantic or not result.get("tail_to_head_axis_camera"):
                    raise ValueError(f"Accepted sample lacks camera-frame pose: {sample['sample_id']}")
                references.append({
                    "sample_id": sample["sample_id"],
                    "group": summary_path.parent.name,
                    "rgb": sample["rgb"],
                    "depth": sample["depth"],
                    "rgb_sha256": rgb_hash,
                    "depth_sha256": depth_hash,
                    "camera_config": sample["camera_config"],
                    "source_result_json": sample["result_json"],
                    "source_result_sha256": sha256(Path(sample["result_json"])),
                    "t_camera_grasp": result["t_camera_grasp"],
                    "semantic_points_camera": semantic,
                    "tail_to_head_axis_camera": result["tail_to_head_axis_camera"],
                })
            rows.append({
                "group": summary_path.parent.name,
                "sample_id": sample["sample_id"],
                "status": sample["status"],
                "reason": sample["reason"] or "",
                "camera_config": sample["camera_config"],
                "rgb": sample["rgb"],
                "depth": sample["depth"],
                "rgb_sha256": rgb_hash,
                "depth_sha256": depth_hash,
                "result_json": sample["result_json"],
                "review_html": sample["registration_review_html"] or "",
                "rgb_projection": sample["rgb_projection"] or "",
                "visible_iou": quality.get("visible_iou", ""),
                "depth_cost": quality.get("depth_cost", ""),
                "candidate_cost_margin": quality.get("candidate_cost_margin", ""),
                "visible_points": points.get("visible_downsampled_points", ""),
                "grasp_center_camera_m": json.dumps(semantic.get("grasp_center_camera_m"), ensure_ascii=False),
                "head_center_camera_m": json.dumps(semantic.get("head_center_camera_m"), ensure_ascii=False),
                "tail_center_camera_m": json.dumps(semantic.get("tail_center_camera_m"), ensure_ascii=False),
                "review_decision": decision,
                "review_notes": existing.get("review_notes", "") or ("registration_unstable" if decision == "unstable" else ""),
            })

    if not rows:
        raise SystemExit(f"No batch summaries found in {root}")
    known_ids = {row["sample_id"] for row in rows}
    unknown_ids = unstable_ids - known_ids
    if unknown_ids:
        raise ValueError(f"Unknown unstable sample IDs: {sorted(unknown_ids)}")
    failed_unstable = unstable_ids - {row["sample_id"] for row in rows if row["status"] == "ok"}
    if failed_unstable:
        raise ValueError(f"Unstable samples must have successful predictions: {sorted(failed_unstable)}")
    with index_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    decisions = {name: sum(row["review_decision"] == name for row in rows) for name in ("accepted", "unstable", "unavailable")}
    decisions["unreviewed"] = len(rows) - sum(decisions.values())
    metadata = {
        "coordinate_frame": "camera_rgb",
        "ground_truth_status": "reviewed_pseudo_labels" if decisions["unreviewed"] == 0 else "pending_manual_review",
        "review_source": selection.get("source") if selection else None,
        "review_selection_sha256": sha256(selection_path) if selection else None,
        "review_counts": decisions,
        "code_commit": commit,
        "code_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
        "code_sha256": {
            str(path): sha256(ROOT / path)
            for path in (
                "infer_6d_single.py", "infer_6d_batch.py", "plug_vg/symmetric_registration.py",
                "plug_vg/visible_points.py", "plug_vg/vision.py", "plug_vg/config.py",
                "configs/camera/plug_rgbd.yaml",
            )
        },
        "benchmark_camera_sha256": {
            path.name: sha256(path) for path in sorted(root.glob("camera_*.yaml"))
        },
        "seg_weights_sha256": sha256(WEIGHTS),
        "cad_pointcloud_sha256": sha256(CAD),
        "groups": groups,
        "total": len(rows),
        "ok": sum(row["status"] == "ok" for row in rows),
        "failed": sum(row["status"] != "ok" for row in rows),
    }
    (root / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    (root / "ground_truth_camera.json").write_text(json.dumps({
        "coordinate_frame": "camera_rgb",
        "reference_type": "manually_accepted_algorithm_predictions",
        "review_source": metadata["review_source"],
        "references": references,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Indexed {metadata['total']} samples: {decisions}")


if __name__ == "__main__":
    main()
