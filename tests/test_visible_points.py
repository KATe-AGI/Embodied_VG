from __future__ import annotations

import unittest

import numpy as np

from plug_vg.visible_points import extract_visible_points_from_mask, voxel_downsample


class VisiblePointsTests(unittest.TestCase):
    def camera(self) -> dict[str, float]:
        return {
            "fx": 100.0,
            "fy": 100.0,
            "cx": 2.0,
            "cy": 2.0,
            "depth_scale": 0.001,
            "image_width": 5,
            "image_height": 5,
        }

    def test_mask_depth_back_projection(self) -> None:
        image = np.zeros((5, 5, 3), dtype=np.uint8)
        depth = np.full((5, 5), 1000, dtype=np.uint16)
        polygon = [[1.0, 1.0], [3.0, 1.0], [3.0, 3.0], [1.0, 3.0]]

        result = extract_visible_points_from_mask(image, depth, self.camera(), polygon, 0.1, 2.0, 0.0, min_points=1)

        self.assertEqual(result.status, "ok")
        self.assertGreaterEqual(len(result.visible_points_camera_m), 4)
        self.assertEqual(result.quality["visible_raw_points"], result.quality["visible_filtered_points"])
        self.assertAlmostEqual(result.quality["depth_median_m"], 1.0)
        self.assertTrue(np.any(np.all(np.isclose(result.visible_points_camera_m, [0.0, 0.0, 1.0]), axis=1)))

    def test_empty_mask_fails(self) -> None:
        image = np.zeros((5, 5, 3), dtype=np.uint8)
        depth = np.full((5, 5), 1000, dtype=np.uint16)

        result = extract_visible_points_from_mask(image, depth, self.camera(), [], 0.1, 2.0, 0.0)

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.reason, "segmentation_missing")

    def test_invalid_depth_and_min_points_fail(self) -> None:
        image = np.zeros((5, 5, 3), dtype=np.uint8)
        depth = np.zeros((5, 5), dtype=np.uint16)
        polygon = [[1.0, 1.0], [3.0, 1.0], [3.0, 3.0], [1.0, 3.0]]

        result = extract_visible_points_from_mask(image, depth, self.camera(), polygon, 0.1, 2.0, 0.0, min_points=1)

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.reason, "no_valid_depth_points")

    def test_voxel_downsample_keeps_one_point_per_voxel(self) -> None:
        points = np.asarray(
            [
                [0.001, 0.001, 1.0],
                [0.002, 0.002, 1.0],
                [0.020, 0.020, 1.0],
            ],
            dtype=np.float64,
        )

        down = voxel_downsample(points, 0.01)

        self.assertEqual(down.shape, (2, 3))


if __name__ == "__main__":
    unittest.main()
