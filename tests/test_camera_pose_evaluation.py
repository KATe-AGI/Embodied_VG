from __future__ import annotations

import unittest

import numpy as np

from evaluation.evaluate_camera_pose import compare_evaluations, evaluate_pair


class CameraPoseEvaluationTests(unittest.TestCase):
    @staticmethod
    def record(grasp, tail, head, status="ok"):
        return {
            "status": status,
            "semantic_points_camera": {
                "grasp_center_camera_m": grasp,
                "tail_center_camera_m": tail,
                "head_center_camera_m": head,
            },
        }

    def test_three_point_rmse_and_directed_axis(self) -> None:
        ground_truth = self.record([0, 0, 1], [-0.1, 0, 1], [0.1, 0, 1])
        prediction = self.record([0, 0.003, 1], [-0.1, 0.004, 1], [0.1, 0, 1])

        result = evaluate_pair(ground_truth, prediction)

        self.assertTrue(result["has_estimate"])
        self.assertAlmostEqual(result["grasp_error_mm"], 3.0)
        self.assertAlmostEqual(result["tail_error_mm"], 4.0)
        self.assertAlmostEqual(result["head_error_mm"], 0.0)
        self.assertAlmostEqual(result["semantic_rmse_mm"], np.sqrt(25.0 / 3.0))
        self.assertGreater(result["directed_axis_error_deg"], 1.0)

    def test_missing_prediction_is_not_assigned_zero_error(self) -> None:
        ground_truth = self.record([0, 0, 1], [-0.1, 0, 1], [0.1, 0, 1])

        result = evaluate_pair(ground_truth, {"status": "failed"})

        self.assertFalse(result["has_estimate"])
        self.assertIsNone(result["semantic_rmse_mm"])

    def test_compare_evaluations_uses_leave_one_out_training_error(self) -> None:
        reference = {
            "frames": [
                {"sample_id": "a", "semantic_rmse_mm": 4.0},
                {"sample_id": "b", "semantic_rmse_mm": 5.0},
                {"sample_id": "c", "semantic_rmse_mm": 6.0},
            ]
        }
        candidate = {
            "frames": [
                {"sample_id": "a", "semantic_rmse_mm": 2.0},
                {"sample_id": "b", "semantic_rmse_mm": 3.0},
                {"sample_id": "c", "semantic_rmse_mm": 4.0},
            ]
        }

        comparison = compare_evaluations(reference, candidate)

        self.assertEqual(comparison["fold_count"], 3)
        self.assertEqual(comparison["candidate_selected_folds"], 3)
        self.assertEqual(comparison["held_out_candidate_better_folds"], 3)
        self.assertTrue(comparison["module_accepted"])


if __name__ == "__main__":
    unittest.main()
