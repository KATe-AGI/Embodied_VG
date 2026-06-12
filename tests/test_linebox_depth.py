from __future__ import annotations

import unittest

import numpy as np

from plug_vg.linebox_depth import (
    choose_farthest_stable_depth_peak,
    clamp_roi,
    depth_pixels_to_camera_points,
    robust_axis_stats,
    roi_valid_depth_pixels,
    select_depth_band,
    transform_points,
)


class LineboxDepthTests(unittest.TestCase):
    def test_farthest_stable_peak_ignores_far_noise(self) -> None:
        depth_values = np.concatenate(
            [
                np.full(90, 0.55, dtype=np.float64),
                np.full(160, 1.20, dtype=np.float64),
                np.full(8, 1.82, dtype=np.float64),
            ]
        )

        peak, histogram = choose_farthest_stable_depth_peak(
            depth_values,
            min_depth=0.1,
            max_depth=2.0,
            bin_size_m=0.01,
            min_peak_points=20,
            min_peak_fraction=0.05,
        )

        self.assertIsNotNone(peak)
        assert peak is not None
        self.assertAlmostEqual(float(peak["center_m"]), 1.205, places=6)
        self.assertEqual(histogram["stable_peak_threshold_count"], 20)

    def test_no_stable_peak_when_only_sparse_depth_noise(self) -> None:
        depth_values = np.asarray([0.52, 0.88, 1.26, 1.84], dtype=np.float64)

        peak, histogram = choose_farthest_stable_depth_peak(
            depth_values,
            min_depth=0.1,
            max_depth=2.0,
            bin_size_m=0.01,
            min_peak_points=20,
            min_peak_fraction=0.05,
        )

        self.assertIsNone(peak)
        self.assertEqual(histogram["reason"], "no_stable_depth_peak")

    def test_no_valid_depth_reports_reason(self) -> None:
        peak, histogram = choose_farthest_stable_depth_peak(
            np.asarray([0.0, np.nan, 9.0], dtype=np.float64),
            min_depth=0.1,
            max_depth=2.0,
            bin_size_m=0.01,
            min_peak_points=20,
            min_peak_fraction=0.05,
        )

        self.assertIsNone(peak)
        self.assertEqual(histogram["reason"], "no_valid_roi_depth")

    def test_roi_depth_selection_uses_clamped_absolute_pixels(self) -> None:
        depth = np.zeros((6, 8), dtype=np.float64)
        depth[1:5, 2:7] = 1.2
        roi = clamp_roi([-3, 1, 7, 10], width=8, height=6)

        xs, ys, z = roi_valid_depth_pixels(depth, roi, min_depth=0.1, max_depth=2.0)
        selected_xs, selected_ys, selected_z, keep = select_depth_band(xs, ys, z, 1.205, 0.01)

        self.assertEqual(roi, (0, 1, 7, 6))
        self.assertEqual(len(z), 20)
        self.assertEqual(len(selected_z), 20)
        self.assertEqual(int(np.count_nonzero(keep)), 20)
        self.assertGreaterEqual(float(np.min(selected_xs)), 2.0)
        self.assertLessEqual(float(np.max(selected_ys)), 4.0)

    def test_back_projection_transform_and_robust_base_x_stats(self) -> None:
        camera = {"fx": 100.0, "fy": 100.0, "cx": 10.0, "cy": 20.0}
        xs = np.asarray([10.0, 20.0, 30.0], dtype=np.float64)
        ys = np.asarray([20.0, 20.0, 20.0], dtype=np.float64)
        z = np.asarray([1.0, 1.0, 1.0], dtype=np.float64)
        points_camera = depth_pixels_to_camera_points(xs, ys, z, camera)
        transform = np.eye(4, dtype=np.float64)
        transform[:3, 3] = [-0.9, 0.0, 0.0]

        points_base = transform_points(points_camera, transform)
        stats = robust_axis_stats(np.concatenate([points_base[:, 0], np.asarray([8.0])]))

        np.testing.assert_allclose(points_camera[:, 0], [0.0, 0.1, 0.2], atol=1e-8)
        np.testing.assert_allclose(points_base[:, 0], [-0.9, -0.8, -0.7], atol=1e-8)
        self.assertEqual(stats["count_raw"], 4)
        self.assertEqual(stats["count_filtered"], 3)
        self.assertAlmostEqual(stats["median_m"], -0.8)


if __name__ == "__main__":
    unittest.main()
