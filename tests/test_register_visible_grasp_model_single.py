from __future__ import annotations

import unittest

import numpy as np

from plug_vg.grasp_model import axis_from_semantic_points
from plug_vg.model_registration import candidate_transforms_with_rolls, select_registration_candidate


class RegisterVisibleGraspModelSingleTests(unittest.TestCase):
    def ranked(self, name: str, fitness: float, rmse: float, x_sign: float = 1.0) -> dict:
        transform = np.eye(4, dtype=np.float64)
        transform[0, 0] = x_sign
        transform[2, 2] = x_sign
        return {
            "summary": {"name": name},
            "fitness": fitness,
            "rmse": rmse,
            "t_grasp_camera": transform,
        }

    def test_generates_eight_unique_pca_rotations(self) -> None:
        model = np.asarray(
            [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.25]],
            dtype=np.float64,
        )
        scene = model @ np.asarray(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ).T + np.asarray([0.2, -0.1, 0.7])

        candidates = candidate_transforms_with_rolls(model, scene, (0.0, 90.0, 180.0, 270.0))
        rotations = {
            np.round(item["t_grasp_camera"][:3, :3], decimals=12).tobytes()
            for item in candidates
        }

        self.assertEqual(len(candidates), 8)
        self.assertEqual(len(rotations), 8)
        self.assertFalse(any("_y_" in item["name"] for item in candidates))
        for item in candidates:
            mapped_centroid = (
                item["t_grasp_camera"][:3, :3] @ np.mean(scene, axis=0)
                + item["t_grasp_camera"][:3, 3]
            )
            np.testing.assert_allclose(mapped_centroid, np.mean(model, axis=0), atol=1e-12)

    def test_selects_highest_fitness_then_lowest_rmse(self) -> None:
        status, best, quality = select_registration_candidate(
            [
                self.ranked("weaker", 0.7, 0.004),
                self.ranked("best", 0.9, 0.006),
                self.ranked("noisy", 0.9, 0.009),
            ],
            min_registration_fitness=0.35,
            max_inlier_rmse_m=0.012,
        )

        self.assertEqual(status, "ok")
        self.assertIsNotNone(best)
        assert best is not None
        self.assertEqual(best["summary"]["name"], "best")
        self.assertEqual(quality["candidate"], "best")

    def test_roll_equivalent_candidates_do_not_make_pose_ambiguous(self) -> None:
        status, best, quality = select_registration_candidate(
            [self.ranked("roll_0", 0.9, 0.006), self.ranked("roll_90", 0.9, 0.0061)],
            min_registration_fitness=0.35,
            max_inlier_rmse_m=0.012,
        )

        self.assertEqual(status, "ok")
        self.assertIsNotNone(best)
        self.assertEqual(quality["roll_equivalent_candidates"], 2)
        self.assertTrue(quality["roll_about_grasp_x_ignored"])

    def test_only_equal_opposite_axes_are_ambiguous(self) -> None:
        status, _best, quality = select_registration_candidate(
            [self.ranked("forward", 0.9, 0.006), self.ranked("reverse", 0.9, 0.006, -1.0)],
            min_registration_fitness=0.35,
            max_inlier_rmse_m=0.012,
        )

        self.assertEqual(status, "ambiguous")
        self.assertEqual(quality["reason"], "registration_ambiguous_head_tail")

    def test_better_directed_axis_is_accepted(self) -> None:
        status, best, quality = select_registration_candidate(
            [self.ranked("forward", 0.9, 0.006), self.ranked("reverse", 0.87, 0.007, -1.0)],
            min_registration_fitness=0.35,
            max_inlier_rmse_m=0.012,
        )

        self.assertEqual(status, "ok")
        assert best is not None
        self.assertEqual(best["summary"]["name"], "forward")
        self.assertEqual(quality["opposite_axis_candidate"], "reverse")

    def test_quality_gates_reject_low_fitness_and_high_rmse(self) -> None:
        low_status, _best, low_quality = select_registration_candidate(
            [self.ranked("low", 0.2, 0.004)],
            min_registration_fitness=0.35,
            max_inlier_rmse_m=0.012,
        )
        high_status, _best, high_quality = select_registration_candidate(
            [self.ranked("high", 0.8, 0.02)],
            min_registration_fitness=0.35,
            max_inlier_rmse_m=0.012,
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
