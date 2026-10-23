"""Placement checks through the public geometry and single-frame interfaces."""

import argparse
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from plug_vg.groove_placement import evaluate_groove_placement, groove_axis_segment
from plug_vg.placement import load_placement


class GroovePlacementTests(unittest.TestCase):
    def test_groove_axis_uses_55_mm_offset_and_projected_up(self):
        # n=(0,.8,.6) gives projected up=(0,-.6,.8).
        segment = groove_axis_segment([0.1, 0.2, 0.3], [0, 0.8, 0.6], load_placement())
        np.testing.assert_allclose(segment["origin_base_m"], [0.1, 0.244, 0.333])
        np.testing.assert_allclose(segment["direction_base"], [0, -0.6, 0.8])
        np.testing.assert_allclose(segment["tail_limit_base_m"], [0.1, 0.298, 0.261])
        np.testing.assert_allclose(segment["head_limit_base_m"], [0.1, 0.1948, 0.3986])

    def test_allowed_axial_sliding_and_frustum_boundary(self):
        segment = groove_axis_segment([0, 0, 0], [0, 1, 0], load_placement())
        config = load_placement()["groove_check"]
        # 167.8 mm plug against 172 mm limits: both extreme translations fit.
        for tail_z in [-0.090, -0.0858]:
            result = evaluate_groove_placement(
                [0.005, 0.055, tail_z],
                [0.005, 0.055, tail_z + 0.1678],
                segment,
                config,
            )
            self.assertTrue(result["is_seated"])
            self.assertAlmostEqual(result["inside_fraction"], 1.0)
        # A 10 mm parallel offset occupies exactly 2/3 of a 20 -> 5 mm frustum.
        partial = evaluate_groove_placement(
            [0.010, 0.055, -0.09], [0.010, 0.055, 0.082], segment, config
        )
        self.assertFalse(partial["is_seated"])
        self.assertAlmostEqual(partial["inside_fraction"], 2 / 3)

    def test_missing_reference_writes_unknown_json_and_review(self):
        from infer_groove_placement_single import parse_args, run

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = parse_args(
                [
                    "--rgb",
                    str(root / "missing.png"),
                    "--d2rgb",
                    str(root / "missing.npy"),
                    "--apriltag-json",
                    str(root / "missing.json"),
                    "--robot-pose",
                    "0",
                    "0",
                    "0",
                    "0",
                    "0",
                    "0",
                    "--output-dir",
                    str(root),
                ]
            )
            self.assertIsInstance(args, argparse.Namespace)
            result, path = run(args)
            saved = json.loads(path.read_text())
            self.assertEqual(saved["status"], "unknown")
            self.assertIsNone(saved["is_seated"])
            self.assertEqual(result["status"], "unknown")
            self.assertTrue(Path(saved["artifacts"]["review_html"]).is_file())

    def test_exact_80_percent_boundary_and_axial_caps(self):
        segment = groove_axis_segment([0, 0, 0], [0, 1, 0], load_placement())
        config = load_placement()["groove_check"]
        # 172 / 215 = 80%; measure the full finite axis, including out-of-cap parts.
        boundary = evaluate_groove_placement(
            [0, 0.055, -0.133], [0, 0.055, 0.082], segment, config
        )
        self.assertTrue(boundary["is_seated"])
        self.assertAlmostEqual(boundary["inside_fraction"], 0.8)
        outside = evaluate_groove_placement(
            [0, 0.055, -0.1331], [0, 0.055, 0.082], segment, config
        )
        self.assertFalse(outside["is_seated"])
        no_overlap = evaluate_groove_placement(
            [0, 0.055, 0.1], [0, 0.055, 0.2], segment, config
        )
        self.assertEqual(no_overlap["inside_fraction"], 0)

    def test_line_can_cross_frustum_with_both_endpoints_outside(self):
        segment = groove_axis_segment([0, 0, 0], [0, 1, 0], load_placement())
        config = {"top_radius_m": 0.005, "bottom_radius_m": 0.005, "min_inside_fraction": 0.8}
        crossing = evaluate_groove_placement(
            [-0.01, 0.055, -0.08], [0.01, 0.055, 0.08], segment, config
        )
        self.assertAlmostEqual(crossing["inside_fraction"], 0.5)
        self.assertFalse(crossing["is_seated"])
        # A whole line on the cylinder boundary is included, not discarded.
        on_surface = evaluate_groove_placement(
            [0.005, 0.055, -0.08], [0.005, 0.055, 0.08], segment, config
        )
        self.assertEqual(on_surface["inside_fraction"], 1)
        tangent = evaluate_groove_placement(
            [-0.01, 0.06, -0.08], [0.01, 0.06, 0.08], segment, config
        )
        self.assertAlmostEqual(tangent["inside_fraction"], 0)

    def test_user_confirmed_seated_example_passes_with_tail_sway(self):
        # Frozen observed axis and reference from 20261023_154709_074.
        segment = groove_axis_segment(
            [-0.3384035, 0.73287078, 0.1332143],
            [0.0018114, -0.99910679, 0.04221767], load_placement()
        )
        result = evaluate_groove_placement(
            [-0.33846926838680275, 0.6804551263265713, 0.04526073286561112],
            [-0.3364719802818888, 0.6778415752772249, 0.21302849022891668],
            segment, load_placement()["groove_check"]
        )
        self.assertTrue(result["is_seated"])
        self.assertGreater(result["max_distance_mm"], 5)
        self.assertGreater(result["inside_fraction"], 0.99)

    def test_containment_is_invariant_under_rigid_frame_change(self):
        from plug_vg.robot_transform import rpy_xyz_to_rotation
        segment = groove_axis_segment([0, 0, 0], [0, 1, 0], load_placement())
        config = load_placement()["groove_check"]
        tail, head = np.array([0.01, 0.055, -0.08]), np.array([0.002, 0.055, 0.08])
        original = evaluate_groove_placement(tail, head, segment, config)
        rotation = rpy_xyz_to_rotation(.1, -.2, .3)
        def transform(p):
            return rotation @ p + [0.3, -0.4, 0.5]
        changed = {key: transform(np.array(segment[key])) for key in ("tail_limit_base_m", "head_limit_base_m")}
        result = evaluate_groove_placement(transform(tail), transform(head), changed, config)
        self.assertAlmostEqual(result["inside_fraction"], original["inside_fraction"])

    def test_analytic_fraction_matches_independent_uniform_length_measurement(self):
        segment = groove_axis_segment([0, 0, 0], [0, 1, 0], load_placement())
        config = load_placement()["groove_check"]
        rng = np.random.default_rng(261023)
        t = (np.arange(20000) + .5) / 20000
        for _ in range(40):
            tail = rng.uniform([-0.04, 0.015, -0.13], [0.04, 0.095, -0.02])
            head = rng.uniform([-0.02, 0.035, 0.02], [0.02, 0.075, 0.12])
            result = evaluate_groove_placement(tail, head, segment, config)
            points = tail + t[:, None] * (head - tail)
            height = points[:, 2] + .09
            radius = .020 - height / .172 * .015
            inside = (height >= 0) & (height <= .172) & (np.hypot(points[:, 0], points[:, 1] - .055) <= radius)
            self.assertAlmostEqual(result["inside_fraction"], float(inside.mean()), delta=2 / len(t))

    def test_invalid_reference_and_axis_cannot_pass(self):
        with self.assertRaisesRegex(ValueError, "undefined"):
            groove_axis_segment([0, 0, 0], [0, 0, 1], load_placement())
        with self.assertRaisesRegex(ValueError, "finite"):
            groove_axis_segment([float("nan"), 0, 0], [0, 1, 0], load_placement())
        segment = groove_axis_segment([0, 0, 0], [0, 1, 0], load_placement())
        with self.assertRaisesRegex(ValueError, "finite"):
            evaluate_groove_placement(
                [0, 0.055, -0.08], [0, float("nan"), 0.08], segment, load_placement()["groove_check"]
            )
        with self.assertRaisesRegex(ValueError, "head-up"):
            evaluate_groove_placement(
                [0, 0.055, 0.08], [0, 0.055, -0.08], segment, load_placement()["groove_check"]
            )
        for changes in ({"top_radius_m": 0}, {"bottom_radius_m": .001},
                        {"min_inside_fraction": 0}, {"min_inside_fraction": 1.1},
                        {"bottom_radius_m": float("nan")}):
            with self.assertRaises(ValueError):
                evaluate_groove_placement([0, .055, -.08], [0, .055, .08], segment,
                                         {**load_placement()["groove_check"], **changes})


if __name__ == "__main__":
    unittest.main()
