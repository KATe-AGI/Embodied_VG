from __future__ import annotations

import unittest

import numpy as np

from plug_vg.config import GRASP_REGION_THICKNESS_M
from plug_vg.geometry import robust_midsection_center


class RobustMidsectionCenterTests(unittest.TestCase):
    def make_scene(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float], list[float], np.ndarray]:
        mask = np.zeros((80, 120), dtype=np.uint8)
        mask[20:60, 10:110] = 1
        ys, xs = np.nonzero(mask)
        pixels = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
        points = np.column_stack(
            [
                (pixels[:, 0] - 60.0) / 1000.0,
                (pixels[:, 1] - 40.0) / 1000.0,
                np.ones(len(pixels), dtype=np.float64),
            ]
        )
        head_xy = [110.0, 40.0]
        tail_xy = [10.0, 40.0]
        rotation = np.eye(3, dtype=np.float64)
        return points, pixels, mask, head_xy, tail_xy, rotation

    def assert_center_near_expected(self, center: np.ndarray) -> None:
        expected = np.asarray([0.0, -0.0005, 1.0 + GRASP_REGION_THICKNESS_M * 0.5], dtype=np.float64)
        np.testing.assert_allclose(center, expected, atol=0.0015)

    def test_clean_midsection_center(self) -> None:
        points, pixels, mask, head_xy, tail_xy, rotation = self.make_scene()
        warnings: list[str] = []

        center, info = robust_midsection_center(points, pixels, mask, head_xy, tail_xy, rotation, warnings)

        self.assert_center_near_expected(center)
        self.assertEqual(info["mode"], "robust_midsection_center")
        self.assertEqual(info["source"], "midsection")
        self.assertEqual(info["keypoints_used_for_midsection"], False)
        self.assertGreaterEqual(info["filtered_count"], 30)
        self.assertEqual(warnings, [])

    def test_center_rejects_locator_pin_depth_outliers(self) -> None:
        points, pixels, mask, head_xy, tail_xy, rotation = self.make_scene()
        pin = (
            (pixels[:, 0] >= 55.0)
            & (pixels[:, 0] <= 65.0)
            & (pixels[:, 1] >= 35.0)
            & (pixels[:, 1] <= 45.0)
        )
        points[pin, 2] = 0.92
        warnings: list[str] = []

        center, info = robust_midsection_center(points, pixels, mask, head_xy, tail_xy, rotation, warnings)

        self.assert_center_near_expected(center)
        rejected_depth = (
            info["rejected_reason_counts"]["local_z_outlier"]
            + info["rejected_reason_counts"]["fused_anchor_region"]
        )
        self.assertGreater(rejected_depth, 0)
        self.assertEqual(info["source"], "midsection")

    def test_center_rejects_background_depth_outliers(self) -> None:
        points, pixels, mask, head_xy, tail_xy, rotation = self.make_scene()
        background = (
            (pixels[:, 0] >= 52.0)
            & (pixels[:, 0] <= 68.0)
            & (pixels[:, 1] >= 22.0)
            & (pixels[:, 1] <= 23.0)
        )
        points[background, 2] = 1.35
        warnings: list[str] = []

        center, info = robust_midsection_center(points, pixels, mask, head_xy, tail_xy, rotation, warnings)

        self.assert_center_near_expected(center)
        rejected_depth = (
            info["rejected_reason_counts"]["local_z_outlier"]
            + info["rejected_reason_counts"]["fused_anchor_region"]
        )
        self.assertGreater(rejected_depth, 0)
        self.assertEqual(info["source"], "midsection")

    def test_boundary_background_is_ignored_by_mask_interior_filter(self) -> None:
        points, pixels, mask, head_xy, tail_xy, rotation = self.make_scene()
        boundary_background = (
            (pixels[:, 0] <= 11.0)
            | (pixels[:, 0] >= 108.0)
            | (pixels[:, 1] <= 21.0)
            | (pixels[:, 1] >= 58.0)
        )
        points[boundary_background, 2] = 1.4
        warnings: list[str] = []

        center, info = robust_midsection_center(points, pixels, mask, head_xy, tail_xy, rotation, warnings)

        self.assert_center_near_expected(center)
        self.assertGreater(info["rejected_reason_counts"]["mask_boundary"], 0)

    def test_fallback_to_full_mask_when_local_anchor_has_too_few_points(self) -> None:
        points, pixels, mask, head_xy, tail_xy, rotation = self.make_scene()
        warnings: list[str] = []

        center, info = robust_midsection_center(
            points,
            pixels,
            mask,
            head_xy,
            tail_xy,
            rotation,
            warnings,
            min_points=len(points) + 1,
        )

        self.assert_center_near_expected(center)
        self.assertEqual(info["source"], "fallback_full_mask")
        self.assertIn("robust_midsection_center_fallback_full_mask", warnings)

    def test_keypoint_shift_does_not_move_mask_axis_anchor(self) -> None:
        points, pixels, mask, _head_xy, _tail_xy, rotation = self.make_scene()
        shifted_head_xy = [135.0, 50.0]
        shifted_tail_xy = [35.0, 50.0]
        warnings: list[str] = []

        center, info = robust_midsection_center(points, pixels, mask, shifted_head_xy, shifted_tail_xy, rotation, warnings)

        self.assert_center_near_expected(center)
        self.assertEqual(info["keypoints_used_for_midsection"], False)
        self.assertEqual(info["source"], "midsection")

    def test_axis_offset_moves_final_center_along_tail_to_head_x(self) -> None:
        points, pixels, mask, head_xy, tail_xy, rotation = self.make_scene()
        warnings: list[str] = []

        center, info = robust_midsection_center(
            points,
            pixels,
            mask,
            head_xy,
            tail_xy,
            rotation,
            warnings,
            axis_offset_m=0.01,
        )

        expected = np.asarray([0.01, -0.0005, 1.0 + GRASP_REGION_THICKNESS_M * 0.5], dtype=np.float64)
        np.testing.assert_allclose(center, expected, atol=0.0015)
        np.testing.assert_allclose(info["axis_offset_vector_camera_m"], [0.01, 0.0, 0.0], atol=1e-8)
        self.assertEqual(info["axis_offset_direction"], "tail_to_head")

    def test_negative_axis_offset_moves_toward_tail(self) -> None:
        points, pixels, mask, head_xy, tail_xy, rotation = self.make_scene()
        warnings: list[str] = []

        center, info = robust_midsection_center(
            points,
            pixels,
            mask,
            head_xy,
            tail_xy,
            rotation,
            warnings,
            axis_offset_m=-0.01,
        )

        expected = np.asarray([-0.01, -0.0005, 1.0 + GRASP_REGION_THICKNESS_M * 0.5], dtype=np.float64)
        np.testing.assert_allclose(center, expected, atol=0.0015)
        np.testing.assert_allclose(info["axis_offset_vector_camera_m"], [-0.01, 0.0, 0.0], atol=1e-8)


if __name__ == "__main__":
    unittest.main()
