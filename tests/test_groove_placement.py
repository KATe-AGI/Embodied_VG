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

    def test_allowed_axial_sliding_and_5_mm_boundary(self):
        segment = groove_axis_segment([0, 0, 0], [0, 1, 0], load_placement())
        # 167.8 mm plug against 172 mm limits: both extreme translations fit.
        for tail_z in [-0.090, -0.0858]:
            result = evaluate_groove_placement(
                [0.005, 0.055, tail_z],
                [0.005, 0.055, tail_z + 0.1678],
                segment,
                0.005,
            )
            self.assertTrue(result["is_seated"])
            self.assertAlmostEqual(result["max_distance_mm"], 5.0)
        outside = evaluate_groove_placement(
            [0.0051, 0.055, -0.09], [0.0051, 0.055, 0.0778], segment, 0.005
        )
        self.assertFalse(outside["is_seated"])

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

    def test_axial_overrun_and_tilt_use_the_same_metric(self):
        segment = groove_axis_segment([0, 0, 0], [0, 1, 0], load_placement())
        axial = evaluate_groove_placement(
            [0, 0.055, -0.096], [0, 0.055, 0.0718], segment, 0.005
        )
        self.assertFalse(axial["is_seated"])
        self.assertAlmostEqual(axial["max_distance_mm"], 6.0)
        tilted = evaluate_groove_placement(
            [0, 0.055, -0.08], [0.006, 0.055, 0.08], segment, 0.005
        )
        self.assertFalse(tilted["is_seated"])
        self.assertAlmostEqual(tilted["max_distance_mm"], 6.0)

    def test_invalid_reference_and_axis_cannot_pass(self):
        with self.assertRaisesRegex(ValueError, "undefined"):
            groove_axis_segment([0, 0, 0], [0, 0, 1], load_placement())
        with self.assertRaisesRegex(ValueError, "finite"):
            groove_axis_segment([float("nan"), 0, 0], [0, 1, 0], load_placement())
        segment = groove_axis_segment([0, 0, 0], [0, 1, 0], load_placement())
        with self.assertRaisesRegex(ValueError, "finite"):
            evaluate_groove_placement(
                [0, 0.055, -0.08], [0, float("nan"), 0.08], segment, 0.005
            )
        with self.assertRaisesRegex(ValueError, "head-up"):
            evaluate_groove_placement(
                [0, 0.055, 0.08], [0, 0.055, -0.08], segment, 0.005
            )


if __name__ == "__main__":
    unittest.main()
