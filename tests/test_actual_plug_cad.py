from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import yaml

from plug_vg.grasp_model import DEFAULT_GRASP_MODEL_CONFIG, load_grasp_model
from tools.build_actual_plug_cad import ACTUAL_LANDMARKS_MM, ORIGINAL_LANDMARKS_MM, axial_map_mm, read_obj


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
        self.assertAlmostEqual(float(np.ptp(actual[:, 0])), 0.172, places=7)

    def test_default_runtime_model_is_measured_plug(self) -> None:
        self.assertEqual(DEFAULT_GRASP_MODEL_CONFIG.name, "plugCAD.yaml")
        model = load_grasp_model()
        self.assertEqual(model.config["model_id"], "plugCAD")
        self.assertAlmostEqual(model.config["dimensions_m"]["head_tail_axis_length"], 0.172, places=8)

        with DEFAULT_GRASP_MODEL_CONFIG.open("r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        self.assertEqual(config["dimensions_m"]["tail_section_length"], 0.032)
        self.assertEqual(config["dimensions_m"]["tail_end_to_large_ring_near_edge"], 0.118)
        self.assertEqual(config["dimensions_m"]["large_ring_length"], 0.021)


if __name__ == "__main__":
    unittest.main()
