from __future__ import annotations

import unittest

import numpy as np

from plug_vg.symmetric_registration import build_profile_grid, canonical_rotation_from_axis


class SymmetricRegistrationTests(unittest.TestCase):
    def test_canonical_rotation_preserves_directed_axis(self) -> None:
        axis = np.asarray([0.2, -0.3, 0.9], dtype=np.float64)
        axis /= np.linalg.norm(axis)

        rotation = canonical_rotation_from_axis(axis)

        np.testing.assert_allclose(rotation[:, 0], axis, atol=1e-12)
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0)

    def test_profile_grid_is_invariant_to_roll_about_x(self) -> None:
        x = np.linspace(-0.05, 0.08, 20)
        angle = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
        points = np.asarray(
            [[item, 0.03 * np.cos(theta), 0.03 * np.sin(theta)] for item in x for theta in angle]
        )
        rolled = points.copy()
        rolled[:, 1] = points[:, 2]
        rolled[:, 2] = -points[:, 1]

        first = build_profile_grid(points)
        second = build_profile_grid(rolled)

        np.testing.assert_array_equal(first.distances_m, second.distances_m)
        self.assertEqual(first.axial_min_m, second.axial_min_m)
        self.assertEqual(first.radial_max_m, second.radial_max_m)


if __name__ == "__main__":
    unittest.main()
