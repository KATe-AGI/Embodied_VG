from __future__ import annotations

import unittest

import numpy as np

from plug_vg.config import HEAD_TAIL_DISTANCE_M
from plug_vg.window_grasp import (
    WindowGraspError,
    add_window_candidates,
    attach_direct_visual_grasp,
    build_grasp_axis_base,
    build_window_geometry,
    generate_window_constrained_candidates,
    resolve_window_inputs,
)


class WindowGraspTests(unittest.TestCase):
    def setUp(self) -> None:
        self.corners = np.asarray(
            [
                [-0.3, 0.3, 0.0],
                [0.3, 0.3, 0.0],
                [0.3, -0.3, 0.0],
                [-0.3, -0.3, 0.0],
            ],
            dtype=np.float64,
        )
        self.reference_pose = {
            "translation_m": [0.0, 0.0, 1.0],
            "rotation_matrix": [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
        }

    def test_window_geometry_from_rectangle(self) -> None:
        geometry = build_window_geometry(self.corners, margin_m=0.1)

        self.assertEqual(geometry["center_base_m"], [0.0, 0.0, 0.0])
        self.assertAlmostEqual(geometry["width_m"], 0.6)
        self.assertAlmostEqual(geometry["height_m"], 0.6)
        np.testing.assert_allclose(geometry["x_window_base"], [1.0, 0.0, 0.0])
        np.testing.assert_allclose(geometry["y_window_base"], [0.0, -1.0, 0.0])
        np.testing.assert_allclose(geometry["normal_base"], [0.0, 0.0, -1.0])

    def test_invalid_non_coplanar_window(self) -> None:
        corners = self.corners.copy()
        corners[2, 2] = 0.02

        with self.assertRaises(WindowGraspError) as raised:
            build_window_geometry(corners, margin_m=0.01)

        self.assertEqual(raised.exception.reason, "window_geometry_invalid")

    def test_invalid_repeated_corner_window(self) -> None:
        corners = self.corners.copy()
        corners[1] = corners[0]

        with self.assertRaises(WindowGraspError) as raised:
            build_window_geometry(corners, margin_m=0.01)

        self.assertEqual(raised.exception.reason, "window_geometry_invalid")

    def test_generate_3x3_candidates_sorted_and_orthonormal(self) -> None:
        geometry = build_window_geometry(self.corners, margin_m=0.1)
        candidates, stats = generate_window_constrained_candidates(self.reference_pose, geometry)

        self.assertEqual(stats["sampled_count"], 9)
        self.assertEqual(len(candidates), 9)
        self.assertEqual(candidates[0]["window_point_base"], [0.0, 0.0, 0.0])
        self.assertEqual(candidates[0]["score_visual_geometry"], 1.0)
        scores = [candidate["score_visual_geometry"] for candidate in candidates]
        self.assertEqual(scores, sorted(scores, reverse=True))

        for candidate in candidates:
            rotation = np.asarray(candidate["rotation_matrix"], dtype=np.float64)
            x_axis = rotation[:, 0]
            y_axis = rotation[:, 1]
            z_axis = rotation[:, 2]
            np.testing.assert_allclose(np.linalg.norm(x_axis), 1.0, atol=1e-7)
            np.testing.assert_allclose(np.linalg.norm(y_axis), 1.0, atol=1e-7)
            np.testing.assert_allclose(np.linalg.norm(z_axis), 1.0, atol=1e-7)
            np.testing.assert_allclose(x_axis @ y_axis, 0.0, atol=1e-7)
            np.testing.assert_allclose(x_axis @ z_axis, 0.0, atol=1e-7)
            np.testing.assert_allclose(y_axis @ z_axis, 0.0, atol=1e-7)
            np.testing.assert_allclose(candidate["x_grasp_base"], x_axis, atol=1e-7)
            np.testing.assert_allclose(candidate["y_grasp_base"], y_axis, atol=1e-7)
            np.testing.assert_allclose(candidate["z_approach_base"], z_axis, atol=1e-7)

    def test_cli_corners_override_yaml_corners(self) -> None:
        override = [float(value) for value in self.corners.reshape(-1)]
        inputs = resolve_window_inputs(config_path=None, corners_override=override, margin_override=0.03)

        np.testing.assert_allclose(inputs.corners, self.corners)
        self.assertEqual(inputs.margin_m, 0.03)
        self.assertEqual(inputs.source, "cli")

    def test_build_grasp_axis_base_uses_virtual_axis_through_grasp_point(self) -> None:
        point = np.asarray([0.3, -0.2, 0.8], dtype=np.float64)
        axis_fields = build_grasp_axis_base(point, [0.0, 2.0, 0.0], "unit_test_axis")

        self.assertEqual(axis_fields["grasp_point_base_m"], [0.3, -0.2, 0.8])
        axis = axis_fields["tail_to_head_axis_base"]
        tail = np.asarray(axis["tail_point_m"], dtype=np.float64)
        head = np.asarray(axis["head_point_m"], dtype=np.float64)
        direction = np.asarray(axis["direction_unit"], dtype=np.float64)
        np.testing.assert_allclose(direction, [0.0, 1.0, 0.0], atol=1e-8)
        np.testing.assert_allclose((tail + head) * 0.5, point, atol=1e-8)
        np.testing.assert_allclose(head - tail, direction * HEAD_TAIL_DISTANCE_M, atol=1e-8)
        self.assertAlmostEqual(axis["length_m"], HEAD_TAIL_DISTANCE_M)
        self.assertEqual(axis["source"], "unit_test_axis")

    def test_missing_window_corners(self) -> None:
        with self.assertRaises(WindowGraspError) as raised:
            resolve_window_inputs(config_path=None, corners_override=None, margin_override=None)

        self.assertEqual(raised.exception.reason, "window_corners_missing")

    def test_attach_direct_visual_grasp_sets_final_pose_without_window_fields(self) -> None:
        result = {
            "status": "ok",
            "warnings": [],
            "grasp_pose_base": dict(self.reference_pose),
            "best_grasp_pose_base": {"stale": True},
            "window_geometry_base": {"stale": True},
            "window_candidate_stats": {"stale": True},
            "window_constrained_grasp_candidates": [{"stale": True}],
        }

        updated = attach_direct_visual_grasp(result)

        self.assertEqual(updated["grasp_solution_mode"], "direct_visual")
        self.assertEqual(updated["grasp_pose_base_role"], "final_grasp_pose")
        self.assertNotIn("best_grasp_pose_base", updated)
        self.assertNotIn("window_geometry_base", updated)
        self.assertNotIn("window_candidate_stats", updated)
        self.assertNotIn("window_constrained_grasp_candidates", updated)
        self.assertEqual(updated["grasp_point_base_m"], self.reference_pose["translation_m"])
        axis = updated["tail_to_head_axis_base"]
        tail = np.asarray(axis["tail_point_m"], dtype=np.float64)
        head = np.asarray(axis["head_point_m"], dtype=np.float64)
        point = np.asarray(updated["grasp_point_base_m"], dtype=np.float64)
        direction = np.asarray(axis["direction_unit"], dtype=np.float64)
        reference_x = np.asarray(self.reference_pose["rotation_matrix"], dtype=np.float64)[:, 0]
        np.testing.assert_allclose((tail + head) * 0.5, point, atol=1e-8)
        np.testing.assert_allclose(head - tail, direction * HEAD_TAIL_DISTANCE_M, atol=1e-8)
        np.testing.assert_allclose(direction, reference_x, atol=1e-8)
        self.assertEqual(axis["source"], "direct_visual_grasp_x_axis")

    def test_add_window_candidates_sets_best_pose(self) -> None:
        override = [float(value) for value in self.corners.reshape(-1)]
        result = {
            "status": "ok",
            "warnings": [],
            "grasp_pose_base": self.reference_pose,
        }

        updated = add_window_candidates(result, config_path=None, corners_override=override, margin_override=0.1)

        self.assertEqual(updated["status"], "ok")
        self.assertEqual(updated["grasp_solution_mode"], "window_constrained")
        self.assertEqual(updated["grasp_pose_base_role"], "surface_normal_reference")
        self.assertIn("best_grasp_pose_base", updated)
        self.assertEqual(updated["best_grasp_pose_base"], updated["window_constrained_grasp_candidates"][0])
        best = updated["best_grasp_pose_base"]
        self.assertEqual(best["grasp_point_base_m"], best["translation_m"])
        self.assertEqual(updated["grasp_point_base_m"], best["grasp_point_base_m"])
        self.assertEqual(updated["tail_to_head_axis_base"], best["tail_to_head_axis_base"])
        axis = best["tail_to_head_axis_base"]
        tail = np.asarray(axis["tail_point_m"], dtype=np.float64)
        head = np.asarray(axis["head_point_m"], dtype=np.float64)
        point = np.asarray(best["grasp_point_base_m"], dtype=np.float64)
        direction = np.asarray(axis["direction_unit"], dtype=np.float64)
        reference_x = np.asarray(self.reference_pose["rotation_matrix"], dtype=np.float64)[:, 0]
        np.testing.assert_allclose((tail + head) * 0.5, point, atol=1e-8)
        np.testing.assert_allclose(head - tail, direction * HEAD_TAIL_DISTANCE_M, atol=1e-8)
        np.testing.assert_allclose(direction, reference_x, atol=1e-8)
        self.assertAlmostEqual(axis["length_m"], HEAD_TAIL_DISTANCE_M)

    def test_add_window_candidates_fails_without_candidates(self) -> None:
        override = [float(value) for value in self.corners.reshape(-1)]
        in_plane_pose = {
            "translation_m": [1.0, 1.0, 0.0],
            "rotation_matrix": self.reference_pose["rotation_matrix"],
        }
        result = {
            "status": "ok",
            "warnings": [],
            "grasp_pose_base": in_plane_pose,
        }

        updated = add_window_candidates(result, config_path=None, corners_override=override, margin_override=0.1)

        self.assertEqual(updated["status"], "failed")
        self.assertEqual(updated["reason"], "window_candidate_generation_failed")
        self.assertNotIn("best_grasp_pose_base", updated)
        self.assertNotIn("grasp_point_base_m", updated)
        self.assertNotIn("tail_to_head_axis_base", updated)


if __name__ == "__main__":
    unittest.main()
