from __future__ import annotations

import unittest

import numpy as np

from plug_vg.grasp_model import axis_from_semantic_points
from plug_vg.model_registration import select_registration_candidate


class RegisterVisibleGraspModelSingleTests(unittest.TestCase):
    def ranked(self, name: str, fitness: float, rmse: float) -> dict:
        return {
            "summary": {"name": name},
            "fitness": fitness,
            "rmse": rmse,
            "transform": np.eye(4, dtype=np.float64),
        }

    def test_selects_highest_fitness_then_lowest_rmse(self) -> None:
        status, best, quality = select_registration_candidate(
            [
                self.ranked("weaker", 0.7, 0.004),
                self.ranked("best", 0.9, 0.006),
                self.ranked("noisy", 0.9, 0.009),
            ],
            min_registration_fitness=0.35,
            max_inlier_rmse_m=0.012,
            ambiguity_fitness_margin=0.01,
            ambiguity_rmse_margin_m=0.001,
        )

        self.assertEqual(status, "ok")
        self.assertIsNotNone(best)
        assert best is not None
        self.assertEqual(best["summary"]["name"], "best")
        self.assertEqual(quality["candidate"], "best")

    def test_ambiguous_when_top_candidates_are_close(self) -> None:
        status, best, quality = select_registration_candidate(
            [self.ranked("a", 0.9, 0.006), self.ranked("b", 0.87, 0.007)],
            min_registration_fitness=0.35,
            max_inlier_rmse_m=0.012,
            ambiguity_fitness_margin=0.05,
            ambiguity_rmse_margin_m=0.003,
        )

        self.assertEqual(status, "ambiguous")
        self.assertIsNotNone(best)
        assert best is not None
        self.assertEqual(best["summary"]["name"], "a")
        self.assertEqual(quality["reason"], "registration_ambiguous_pose")

    def test_quality_gates_reject_low_fitness_and_high_rmse(self) -> None:
        low_status, _best, low_quality = select_registration_candidate(
            [self.ranked("low", 0.2, 0.004)],
            min_registration_fitness=0.35,
            max_inlier_rmse_m=0.012,
            ambiguity_fitness_margin=0.05,
            ambiguity_rmse_margin_m=0.003,
        )
        high_status, _best, high_quality = select_registration_candidate(
            [self.ranked("high", 0.8, 0.02)],
            min_registration_fitness=0.35,
            max_inlier_rmse_m=0.012,
            ambiguity_fitness_margin=0.05,
            ambiguity_rmse_margin_m=0.003,
        )

        self.assertEqual(low_status, "failed")
        self.assertEqual(low_quality["reason"], "registration_low_fitness")
        self.assertEqual(high_status, "failed")
        self.assertEqual(high_quality["reason"], "registration_high_rmse")

    def test_tail_to_head_axis_camera_from_semantic_points(self) -> None:
        axis = axis_from_semantic_points(
            {
                "tail_center_camera_m": [1.0, 2.0, 3.0],
                "head_center_camera_m": [1.0, 2.2, 3.0],
            },
            "tail_center_camera_m",
            "head_center_camera_m",
            "unit_test",
        )

        np.testing.assert_allclose(axis["direction_unit"], [0.0, 1.0, 0.0])
        self.assertAlmostEqual(axis["length_m"], 0.2)


if __name__ == "__main__":
    unittest.main()
