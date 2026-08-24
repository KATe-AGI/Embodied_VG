from __future__ import annotations

import unittest

import numpy as np

from plug_vg.ring_coarse_registration import detect_model_ring, ring_aligned_candidates


class RingCoarseRegistrationTests(unittest.TestCase):
    @staticmethod
    def synthetic_plug() -> np.ndarray:
        points = []
        for x in np.arange(-0.08, 0.092, 0.002):
            radius = 0.047 if 0.038 <= x <= 0.059 else 0.026
            for angle in np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False):
                points.append([x, radius * np.cos(angle), radius * np.sin(angle)])
        return np.asarray(points, dtype=np.float64)

    def test_detects_model_large_ring(self) -> None:
        ring = detect_model_ring(self.synthetic_plug(), 0.004)

        self.assertIsNotNone(ring)
        assert ring is not None
        self.assertAlmostEqual(ring.center_grasp_m[0], 0.049, delta=0.004)
        self.assertAlmostEqual(ring.radius_m, 0.047, delta=0.002)
        self.assertAlmostEqual(ring.length_m, 0.020, delta=0.004)

    def test_adds_ring_aligned_candidates_without_removing_base_candidates(self) -> None:
        model = self.synthetic_plug()
        rotation = np.asarray(
            [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float64,
        )
        translation = np.asarray([0.1, -0.2, 0.5], dtype=np.float64)
        scene = model @ rotation.T + translation
        base_transform = np.eye(4, dtype=np.float64)
        base_transform[:3, :3] = rotation.T
        base = [{"name": "base", "t_grasp_camera": base_transform, "target_pca_eigenvalues": [1.0, 0.1, 0.1]}]

        candidates = ring_aligned_candidates(model, scene, rotation[:, 0], base, 0.004)

        self.assertTrue(candidates)
        best = min(candidates, key=lambda item: abs(item["scene_ring_radius_m"] - item["model_ring_radius_m"]))
        scene_ring_grasp = (
            best["t_grasp_camera"][:3, :3] @ np.asarray(best["scene_ring_center_camera_m"])
            + best["t_grasp_camera"][:3, 3]
        )
        np.testing.assert_allclose(scene_ring_grasp, best["model_ring_center_grasp_m"], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
