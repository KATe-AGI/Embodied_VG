from __future__ import annotations

import unittest
import numpy as np
from plug_vg.point_preprocessing import PreprocessingError, bilateral_smooth_points, statistical_inlier_mask
from plug_vg.visible_points import extract_visible_points_from_mask


class StatisticalPreprocessingTests(unittest.TestCase):
    def test_bilateral_reduces_plane_jitter_and_preserves_count(self):
        rng = np.random.default_rng(17)
        x, y = np.meshgrid(np.arange(30)*.001, np.arange(30)*.001)
        points = np.c_[x.ravel(), y.ravel(), .7+rng.normal(0, .0003, x.size)]
        original = points.copy()
        result = bilateral_smooth_points(points)
        self.assertEqual(result.shape, points.shape)
        self.assertTrue(np.isfinite(result).all())
        self.assertLess(np.std(result[:, 2]), np.std(points[:, 2]))
        self.assertLess(abs(result[:,2].mean()-.7), .0001)
        np.testing.assert_array_equal(points, original)

    def test_bilateral_does_not_bridge_separated_depth_layers(self):
        x, y = np.meshgrid(np.arange(20)*.001, np.arange(20)*.001)
        low = np.c_[x.ravel(), y.ravel(), np.full(x.size, .7)]
        high = low + [0, 0, .04]
        result = bilateral_smooth_points(np.vstack([low, high]))
        np.testing.assert_allclose(result[:len(low),2], .7, atol=1e-8)
        np.testing.assert_allclose(result[len(low):,2], .74, atol=1e-8)

    def test_numerical_failure_is_reported_before_registration(self):
        from unittest.mock import patch
        depth = np.full((5,5), 700, np.uint16)
        image = np.zeros((5,5,3), np.uint8)
        cam = dict(fx=100.,fy=100.,cx=2.,cy=2.,depth_scale=.001,image_width=5,image_height=5)
        with patch('plug_vg.visible_points.bilateral_smooth_points', side_effect=PreprocessingError('bad coordinates')):
            result = extract_visible_points_from_mask(image,depth,cam,[[0,0],[4,0],[4,4],[0,4]],.1,1.,min_points=1,mask_erosion_px=0,mad_z_threshold=0.,point_filter='bilateral')
        self.assertEqual(result.status,'failed')
        self.assertEqual(result.reason,'point_preprocessing_failed')
        self.assertEqual(len(result.visible_points_camera_m),0)

    def test_removes_isolated_points_and_preserves_surface_coordinates(self):
        xx, yy = np.meshgrid(np.linspace(-.03, .03, 25), np.linspace(-.03, .03, 25))
        surface = np.c_[xx.ravel(), yy.ravel(), np.full(xx.size, .7)]
        points = np.vstack([surface, [[.4, .4, .8], [-.4, -.4, .5]]])
        keep = statistical_inlier_mask(points, 20, 6.)
        self.assertFalse(keep[-2:].any())
        self.assertGreater(keep[:-2].mean(), .98)
        np.testing.assert_array_equal(points[keep], surface[keep[:-2]])

    def test_rigid_transform_preserves_filter_decision(self):
        rng = np.random.default_rng(5)
        points = rng.normal(0, .01, (500, 3))
        points[-1] = [1, 1, 1]
        rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        np.testing.assert_array_equal(statistical_inlier_mask(points), statistical_inlier_mask(points @ rotation.T + [1, 2, 3]))

    def test_empty_tiny_and_coincident_clouds_are_well_defined(self):
        for points in [np.empty((0, 3)), np.array([[0., 0., 1.]]), np.ones((30, 3))]:
            np.testing.assert_array_equal(statistical_inlier_mask(points), np.ones(len(points), bool))

    def test_rejects_invalid_parameters_and_nonfinite_input(self):
        points = np.zeros((10, 3))
        for k, ratio in [(1, 2.), (2.5, 2.), (True, 2.), (20, 0), (20, float('nan'))]:
            with self.assertRaises(ValueError): statistical_inlier_mask(points, k, ratio)
        with self.assertRaises(ValueError): statistical_inlier_mask(np.array([[0., np.nan, 1.]]))

    def test_cli_filter_defaults_preserve_baseline_and_reject_invalid_scale(self):
        import argparse
        import contextlib
        import io
        from plug_vg.point_preprocessing import add_point_filter_arguments
        parser = argparse.ArgumentParser()
        add_point_filter_arguments(parser)
        self.assertEqual(parser.parse_args([]).point_filter, 'none')
        self.assertEqual(parser.parse_args(['--point-filter', 'sor']).sor_std_ratio, 6.)
        for flags in [['--sor-neighbors','1'], ['--sor-std-ratio','nan'], ['--sor-std-ratio','-1']]:
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                parser.parse_args(flags)

    def test_extraction_filters_before_voxelization_and_reports_counts(self):
        depth = np.full((25, 25), 700, dtype=np.uint16)
        depth[12, 12] = 950
        image = np.zeros((25, 25, 3), np.uint8)
        camera = dict(fx=500., fy=500., cx=12., cy=12., depth_scale=.001, image_width=25, image_height=25)
        polygon = [[0, 0], [24, 0], [24, 24], [0, 24]]
        kwargs = dict(min_depth_m=.1,max_depth_m=1.,voxel_size_m=.004,min_points=1,mask_erosion_px=0,mad_z_threshold=0.)
        raw = extract_visible_points_from_mask(image, depth, camera, polygon, point_filter="none", **kwargs)
        clean = extract_visible_points_from_mask(image, depth, camera, polygon, point_filter='sor', sor_neighbors=20, sor_std_ratio=6., **kwargs)
        self.assertEqual(clean.status, 'ok')
        self.assertGreater(raw.visible_points_camera_m[:,2].max(), .9)
        self.assertLess(clean.visible_points_camera_m[:,2].max(), .71)
        self.assertGreater(clean.quality['point_filter']['removed_points'], 0)
        self.assertEqual(clean.quality['mad_rejected_points'], 0)
        self.assertEqual(clean.quality['visible_filtered_points'], clean.quality['point_filter']['output_points'])
        np.testing.assert_array_equal(raw.mask, clean.mask)


if __name__ == '__main__': unittest.main()
