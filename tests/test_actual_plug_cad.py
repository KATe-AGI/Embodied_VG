from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import yaml

from plug_vg.grasp_model import DEFAULT_GRASP_MODEL_CONFIG, load_grasp_model
from tools.build_actual_plug_cad import (
    ACTUAL_LANDMARKS_MM,
    ACTUAL_SEGMENTS_MM,
    ORIGINAL_LANDMARKS_MM,
    TARGET_GRASP_TO_HEAD_MM,
    axial_map_mm,
    read_obj,
)


ROOT = Path(__file__).resolve().parents[1]


class ActualPlugCadTests(unittest.TestCase):
    def test_measured_landmarks_are_exact(self) -> None:
        np.testing.assert_allclose(axial_map_mm(ORIGINAL_LANDMARKS_MM), ACTUAL_LANDMARKS_MM, atol=1e-12)

    def test_actual_mesh_changes_only_long_axis(self) -> None:
        original, original_faces = read_obj(ROOT / "plug_model" / "2175B_grasp.obj")
        actual, actual_faces = read_obj(ROOT / "plug_model" / "plugCAD_grasp.obj")

        np.testing.assert_array_equal(actual_faces, original_faces)
        np.testing.assert_allclose(actual[:, 1:], original[:, 1:], atol=1e-12)
        self.assertAlmostEqual(float(np.ptp(original[:, 0])), 0.1803, places=7)
        self.assertAlmostEqual(float(np.ptp(actual[:, 0])), 0.1678, places=7)

    def test_default_runtime_model_is_measured_plug(self) -> None:
        self.assertEqual(DEFAULT_GRASP_MODEL_CONFIG.name, "plugCAD.yaml")
        model = load_grasp_model()
        self.assertEqual(model.config["model_id"], "plugCAD")
        self.assertAlmostEqual(model.config["dimensions_m"]["head_tail_axis_length"], 0.1678, places=8)
        self.assertIsNotNone(model.normals_grasp)
        assert model.normals_grasp is not None
        self.assertEqual(model.normals_grasp.shape, model.points_grasp_m.shape)
        np.testing.assert_allclose(np.linalg.norm(model.normals_grasp, axis=1), 1.0, atol=1e-7)

        with DEFAULT_GRASP_MODEL_CONFIG.open("r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        self.assertEqual(config["dimensions_m"]["tail_section_length"], 0.0293)
        self.assertEqual(config["dimensions_m"]["tail_end_to_large_ring_near_edge"], 0.1105)
        self.assertEqual(config["dimensions_m"]["large_ring_length"], 0.023)
        np.testing.assert_allclose(config["dimensions_m"]["axial_segments_s1_to_s8"], ACTUAL_SEGMENTS_MM * 0.001)
        semantic = config["semantic_points_grasp_m"]
        self.assertAlmostEqual(semantic["head_center"][0], TARGET_GRASP_TO_HEAD_MM * 0.001, places=8)
        self.assertAlmostEqual(semantic["tail_center"][0], -0.0778, places=8)
        self.assertAlmostEqual(
            semantic["head_center"][0] - semantic["tail_center"][0],
            config["dimensions_m"]["head_tail_axis_length"],
            places=8,
        )


if __name__ == "__main__":
    unittest.main()
