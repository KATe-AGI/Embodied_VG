from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.build_grasp_model_assets import (
    MeshData,
    build_grasp_frame_assets,
    parse_step_length_unit,
    sample_mesh_points,
    sample_mesh_points_with_normals,
    write_ply_points,
)
from plug_vg.grasp_model import read_ascii_ply_points_and_normals


class TestGraspModelAssets(unittest.TestCase):
    def synthetic_mesh(self) -> MeshData:
        vertices = np.asarray(
            [
                [-1.0, -2.0, -10.0],
                [1.0, -2.0, -10.0],
                [1.0, 2.0, -10.0],
                [-1.0, 2.0, -10.0],
                [-1.0, -2.0, 10.0],
                [1.0, -2.0, 10.0],
                [1.0, 2.0, 10.0],
                [-1.0, 2.0, 10.0],
            ],
            dtype=np.float64,
        )
        faces = np.asarray(
            [
                [0, 1, 2],
                [0, 2, 3],
                [4, 6, 5],
                [4, 7, 6],
                [0, 4, 5],
                [0, 5, 1],
                [1, 5, 6],
                [1, 6, 2],
                [2, 6, 7],
                [2, 7, 3],
                [3, 7, 4],
                [3, 4, 0],
            ],
            dtype=np.int64,
        )
        return MeshData(vertices_mm=vertices, faces=faces)

    def test_grasp_frame_maps_raw_z_to_positive_x(self) -> None:
        assets = build_grasp_frame_assets(self.synthetic_mesh())
        tail = np.asarray(assets.semantic_points_grasp_m["tail_center"])
        head = np.asarray(assets.semantic_points_grasp_m["head_center"])

        np.testing.assert_allclose(assets.raw_basis_from_grasp.T @ assets.raw_basis_from_grasp, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(np.cross(assets.raw_basis_from_grasp[:, 0], assets.raw_basis_from_grasp[:, 1]), assets.raw_basis_from_grasp[:, 2])
        self.assertLess(tail[0], 0.0)
        self.assertGreater(head[0], 0.0)
        np.testing.assert_allclose(head - tail, [0.02, 0.0, 0.0], atol=1e-12)

    def test_sample_mesh_points_returns_meter_points(self) -> None:
        assets = build_grasp_frame_assets(self.synthetic_mesh())
        points = sample_mesh_points(assets.vertices_m, assets.faces, count=128, seed=42)
        self.assertEqual(points.shape, (128, 3))
        self.assertTrue(np.all(np.isfinite(points)))
        self.assertLessEqual(float(np.max(np.abs(points[:, 0]))), 0.0100001)

    def test_sampled_mesh_normals_are_written_and_loaded_offline(self) -> None:
        assets = build_grasp_frame_assets(self.synthetic_mesh())
        points, normals = sample_mesh_points_with_normals(
            assets.vertices_m, assets.faces, count=128, seed=42
        )

        self.assertEqual(normals.shape, points.shape)
        np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-12)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model_with_normals.ply"
            write_ply_points(path, points, normals)
            loaded_points, loaded_normals = read_ascii_ply_points_and_normals(path)

        np.testing.assert_allclose(loaded_points, points, atol=1e-9)
        self.assertIsNotNone(loaded_normals)
        assert loaded_normals is not None
        np.testing.assert_allclose(loaded_normals, normals, atol=1e-8)

    def test_parse_step_millimeter_unit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "unit.stp"
            path.write_text(
                "ISO-10303-21;\n#1=(LENGTH_UNIT()NAMED_UNIT(*)SI_UNIT(.MILLI.,.METRE.));\n",
                encoding="utf-8",
            )
            self.assertEqual(parse_step_length_unit(path), "millimeter")


if __name__ == "__main__":
    unittest.main()
