from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from evaluation.evaluate_camera_benchmark import evaluate_dataset, repeatability
from evaluation.evaluate_camera_pose import evaluate_pair


def record(grasp=(0.0, 0.0, 0.8), axis=(1.0, 0.0, 0.0), status="ok"):
    grasp = np.asarray(grasp)
    axis = np.asarray(axis)
    return {
        "status": status,
        "semantic_points_camera": {
            "grasp_center_camera_m": grasp.tolist(),
            "tail_center_camera_m": (grasp - axis * 0.0778).tolist(),
            "head_center_camera_m": (grasp + axis * 0.09).tolist(),
        },
    }


class CameraBenchmarkEvaluationTests(unittest.TestCase):
    def test_head_tail_reversal_remains_a_180_degree_error(self):
        metrics = evaluate_pair(record(), record(axis=(-1.0, 0.0, 0.0)))
        self.assertAlmostEqual(metrics["directed_axis_error_deg"], 180)
        self.assertAlmostEqual(metrics["grasp_error_mm"], 0)
        self.assertAlmostEqual(metrics["head_error_mm"], 180)

    def test_axial_and_lateral_displacements_are_separate(self):
        metrics = evaluate_pair(record(), record(grasp=(0.003, 0.004, 0.8)))
        self.assertAlmostEqual(metrics["grasp_error_mm"], 5)
        self.assertAlmostEqual(metrics["grasp_axial_error_mm"], 3)
        self.assertAlmostEqual(metrics["grasp_lateral_error_mm"], 4)

    def test_roll_does_not_change_semantic_metrics(self):
        truth = record()
        prediction = record()
        truth["t_camera_grasp"] = np.eye(4).tolist()
        angle = 1.2
        rotation = np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])
        prediction["rotation_matrix"] = rotation.tolist()
        metrics = evaluate_pair(truth, prediction)
        self.assertAlmostEqual(metrics["directed_axis_error_deg"], 0)
        self.assertAlmostEqual(metrics["semantic_rmse_mm"], 0)

    def test_collapsed_or_non_numeric_prediction_is_invalid(self):
        self.assertFalse(evaluate_pair(record(), record(axis=(0, 0, 0)))["has_estimate"])
        invalid = record()
        invalid["semantic_points_camera"]["head_center_camera_m"] = ["bad", 0, 0]
        self.assertFalse(evaluate_pair(record(), invalid)["has_estimate"])

    def test_failures_stay_in_denominator_and_timing_is_not_filtered_by_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, predictions = root / "dataset", root / "predictions"
            dataset.mkdir()
            ids = ("valid", "failed", "degenerate", "missing", "shifted")
            rows = []
            for sample_id in ids:
                truth = dataset / f"{sample_id}.json"
                truth.write_text(json.dumps(record()))
                rows.append({"sample_id": sample_id, "group": "group", "subset": "subset", "camera_config": "camera.yaml", "ground_truth_json": truth.name})
            with (dataset / "manifest.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            cases = {
                "valid": record(), "failed": record(status="failed"),
                "degenerate": record(axis=(0, 0, 0)), "shifted": record(grasp=(0.006, 0, 0.8)),
            }
            cases["valid"]["timing"] = {"warm_online_inference_s": 0.1}
            cases["failed"]["timing"] = {"warm_online_inference_s": 0.3}
            for sample_id, value in cases.items():
                folder = predictions / sample_id
                folder.mkdir(parents=True)
                (folder / "rgb_6d_base.json").write_text(json.dumps(value))
            report = evaluate_dataset(dataset, predictions)
            aggregate = report["aggregate"]
            self.assertEqual(aggregate["total"], 5)
            self.assertEqual(aggregate["valid_estimates"], 2)
            self.assertEqual(aggregate["joint_success"]["5mm_5deg"]["passed"], 1)
            self.assertAlmostEqual(aggregate["joint_success"]["5mm_5deg"]["rate"], 0.2)
            self.assertEqual(aggregate["joint_success"]["10mm_10deg"]["passed"], 2)
            self.assertAlmostEqual(aggregate["timing"]["warm_online_inference_ms"]["p50"], 200)
            self.assertAlmostEqual(aggregate["errors"]["grasp_error_mm"]["p50"], 3)
            self.assertEqual(report["by_subset"]["subset"]["total"], 5)

    def test_repeatability_uses_directed_pairwise_axes(self):
        reports = []
        for grasp, axis, passed in (((0, 0, 0.8), (1, 0, 0), True), ((0, 0.003, 0.8), (-1, 0, 0), False)):
            reports.append({
                "frames": [{"sample_id": "a", "has_estimate": True, "grasp_center_camera_m": list(grasp), "axis_camera": list(axis), "joint_success": {"5mm_5deg": passed}}],
                "aggregate": {"joint_success": {"5mm_5deg": {"rate": float(passed)}}},
            })
        result = repeatability(reports)
        self.assertEqual(result["expected_pair_count"], 1)
        self.assertAlmostEqual(result["pairwise_grasp_delta_mm"]["max"], 3)
        self.assertAlmostEqual(result["pairwise_directed_axis_delta_deg"]["max"], 180)
        self.assertEqual(result["per_sample"][0]["successful_runs_5mm_5deg"], 1)


if __name__ == "__main__":
    unittest.main()
