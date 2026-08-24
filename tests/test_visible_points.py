from __future__ import annotations

import unittest

import numpy as np

from plug_vg.visible_points import (
    erode_mask,
    extract_visible_points_from_mask,
    global_mad_depth_keep,
    synthesize_scene_point_cloud,
    voxel_downsample,
)


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

        result = extract_visible_points_from_mask(
            image, depth, self.camera(), polygon, 0.1, 2.0, 0.0, min_points=1, mask_erosion_px=0
        )

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

        result = extract_visible_points_from_mask(
            image, depth, self.camera(), polygon, 0.1, 2.0, 0.0, min_points=1, mask_erosion_px=0
        )

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.reason, "no_valid_depth_points")

    def test_voxel_downsample_uses_voxel_centroids(self) -> None:
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
        np.testing.assert_allclose(down[0], [0.0015, 0.0015, 1.0])
        np.testing.assert_allclose(down[1], [0.020, 0.020, 1.0])

    def test_global_mad_depth_filter_rejects_far_outlier_without_iqr_fallback(self) -> None:
        z = np.asarray([1.0, 1.0, 1.01, 0.99, 2.0], dtype=np.float64)

        keep, stats = global_mad_depth_keep(z, 3.5)

        np.testing.assert_array_equal(keep, [True, True, True, True, False])
        self.assertAlmostEqual(float(stats["median_m"]), 1.0)
        self.assertAlmostEqual(float(stats["mad_m"]), 0.01)

        tied_keep, tied_stats = global_mad_depth_keep(np.asarray([1.0, 1.0, 2.0]), 3.5)
        np.testing.assert_array_equal(tied_keep, [True, True, True])
        self.assertEqual(tied_stats["mad_m"], 0.0)

    def test_erode_mask_uses_pixel_radius(self) -> None:
        mask = np.ones((21, 21), dtype=np.uint8)

        eroded = erode_mask(mask, 5)

        self.assertEqual(int(np.count_nonzero(eroded)), 21 * 21)

        padded = np.zeros((31, 31), dtype=np.uint8)
        padded[5:26, 5:26] = 1
        eroded = erode_mask(padded, 5)
        self.assertTrue(np.all(eroded[:10] == 0))
        self.assertTrue(np.all(eroded[:, :10] == 0))
        self.assertEqual(int(eroded[15, 15]), 1)

    def test_synthesize_colored_scene_point_cloud(self) -> None:
        image = np.zeros((5, 5, 3), dtype=np.uint8)
        image[2, 2] = [10, 20, 30]  # BGR
        depth = np.zeros((5, 5), dtype=np.uint16)
        depth[2, 2] = 1000

        points, colors = synthesize_scene_point_cloud(image, depth, self.camera(), 0.1, 2.0)

        np.testing.assert_allclose(points, [[0.0, 0.0, 1.0]])
        np.testing.assert_allclose(colors, [[30 / 255.0, 20 / 255.0, 10 / 255.0]])

    def test_scene_synthesis_rejects_unaligned_shapes(self) -> None:
        image = np.zeros((5, 5, 3), dtype=np.uint8)
        depth = np.zeros((4, 5), dtype=np.uint16)

        with self.assertRaisesRegex(ValueError, "shapes must match"):
            synthesize_scene_point_cloud(image, depth, self.camera(), 0.1, 2.0)


if __name__ == "__main__":
    unittest.main()
