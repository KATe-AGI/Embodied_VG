from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from plug_vg.registration_review import _cad_bbox_corners, write_interactive_review_html, write_registration_comparison_ply


class RegistrationReviewTests(unittest.TestCase):
    def test_cad_bbox_uses_model_frame_bounds_and_pose(self) -> None:
        model = np.asarray([[-1.0, -2.0, -3.0], [1.0, 2.0, 3.0]], dtype=np.float64)
        transform = np.eye(4, dtype=np.float64)
        transform[:3, 3] = [10.0, 20.0, 30.0]

        corners = _cad_bbox_corners(model, transform)

        self.assertEqual(corners.shape, (8, 3))
        np.testing.assert_allclose(np.min(corners, axis=0), [9.0, 18.0, 27.0])
        np.testing.assert_allclose(np.max(corners, axis=0), [11.0, 22.0, 33.0])

    def test_html_defaults_to_rgb_camera_orientation(self) -> None:
        points = np.asarray([[0.0, 0.0, 1.0], [0.1, 0.1, 1.0]], dtype=np.float64)
        camera = {
            "fx": 100.0,
            "fy": 100.0,
            "cx": 1.0,
            "cy": 1.0,
            "image_width": 3,
            "image_height": 3,
        }
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "review.html"
            write_interactive_review_html(
                output,
                points,
                None,
                points,
                np.empty((0, 3), dtype=np.float64),
                None,
                None,
                "failed",
                "test",
                {},
                7,
                camera,
            )
            content = output.read_text(encoding="utf-8")

        self.assertIn('const state = { yaw: 0.0, pitch: 0.0', content)
        self.assertIn('setView("front")', content)
        self.assertIn("state.panY + v[1]*s*factor", content)
        self.assertIn("origin.y + v[1] / n * length", content)
        self.assertIn('id="layerCadBox"', content)
        self.assertIn("function drawCadBoundingBox", content)
        self.assertIn('"cad_bbox":[]', content)
        self.assertIn('id="layerCoarseCad"', content)
        self.assertIn("function drawCoarseCad", content)
        self.assertIn('"coarse_model":[]', content)

    def test_point_cloud_comparison_ply_contains_colored_observation_and_model(self) -> None:
        visible = np.asarray([[0.0, 0.0, 0.5], [0.02, 0.01, 0.51], [-0.02, -0.01, 0.49]])
        model = np.asarray([[-0.03, 0.0, 0.0], [0.03, 0.0, 0.0], [0.0, 0.02, 0.0]])
        transform = np.eye(4, dtype=np.float64)
        transform[:3, 3] = [0.0, 0.0, 0.5]
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "comparison.ply"
            write_registration_comparison_ply(output, visible, model, transform)
            content = output.read_text(encoding="ascii")

        self.assertIn("element vertex 6", content)
        self.assertIn("property uchar red", content)
        self.assertEqual(content.count(" 18 183 106\n"), 3)
        self.assertEqual(content.count(" 240 68 56\n"), 3)
        self.assertIn("-0.030000000 0.000000000 0.500000000 240 68 56", content)

    def test_point_cloud_comparison_without_transform_contains_only_observations(self) -> None:
        visible = np.asarray([[0.0, 0.0, 0.5], [0.01, 0.0, 0.5]])
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "failed.ply"
            write_registration_comparison_ply(output, visible, np.asarray([[0.0, 0.0, 0.0]]), None)
            content = output.read_text(encoding="ascii")

        self.assertIn("element vertex 2", content)
        self.assertEqual(content.count(" 18 183 106\n"), 2)
        self.assertNotIn(" 240 68 56\n", content)

    def test_point_cloud_comparison_marks_three_semantic_points_with_colored_balls(self) -> None:
        semantic = {
            "grasp_center_camera_m": [0.0, 0.0, 0.5],
            "tail_center_camera_m": [0.05, 0.0, 0.5],
            "head_center_camera_m": [-0.05, 0.0, 0.5],
        }
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "semantic.ply"
            write_registration_comparison_ply(
                output,
                np.empty((0, 3), dtype=np.float64),
                np.empty((0, 3), dtype=np.float64),
                None,
                semantic,
            )
            content = output.read_text(encoding="ascii")

        lines = content.splitlines()
        vertex_count = int(next(line for line in lines if line.startswith("element vertex ")).split()[2])
        marker_lines = lines[lines.index("end_header") + 1 :]
        self.assertEqual(vertex_count, len(marker_lines))
        self.assertGreater(vertex_count, 3)
        self.assertIn("comment semantic markers: yellow=grasp center; cyan=tail center; magenta=head center", content)
        self.assertIn("0.000000000 0.000000000 0.500000000 255 215 0", content)
        self.assertIn("0.050000000 0.000000000 0.500000000 0 170 255", content)
        self.assertIn("-0.050000000 0.000000000 0.500000000 200 0 255", content)

if __name__ == "__main__":
    unittest.main()
