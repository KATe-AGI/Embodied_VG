from __future__ import annotations

import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.register_grasp_model_single import (
    color_filter_mask,
    crop_aabb,
    parse_ply_header,
    read_realsense_binary_ply,
    semantic_points_camera,
)


class TestRegisterGraspModelSingle(unittest.TestCase):
    def write_binary_ply(self, path: Path) -> None:
        header = (
            "ply\r\n"
            "format binary_little_endian 1.0\r\n"
            "element vertex 3\r\n"
            "property float32 x\r\n"
            "property float32 y\r\n"
            "property float32 z\r\n"
            "property uchar red\r\n"
            "property uchar green\r\n"
            "property uchar blue\r\n"
            "element face 0\r\n"
            "property list uchar int vertex_indices\r\n"
            "end_header\r\n"
        ).encode("ascii")
        rows = [
            (1.0, 2.0, -3.0, 255, 0, 0),
            (4.0, 5.0, -6.0, 0, 255, 0),
            (float("nan"), 0.0, 0.0, 0, 0, 255),
        ]
        with path.open("wb") as f:
            f.write(header)
            for row in rows:
                f.write(struct.pack("<fffBBB", *row))

    def test_parse_and_read_realsense_binary_ply(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scene.ply"
            self.write_binary_ply(path)
            header = parse_ply_header(path)
            self.assertEqual(header.format_name, "binary_little_endian")
            self.assertEqual(header.vertex_count, 3)
            self.assertEqual(header.vertex_properties[:3], [("x", "float32"), ("y", "float32"), ("z", "float32")])

            cloud = read_realsense_binary_ply(path)
            self.assertEqual(cloud.points.shape, (2, 3))
            np.testing.assert_allclose(cloud.points[0], [1.0, 2.0, -3.0])
            self.assertIsNotNone(cloud.colors)
            np.testing.assert_allclose(cloud.colors[0], [1.0, 0.0, 0.0])

    def test_crop_aabb_includes_boundaries(self) -> None:
        points = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
                [3.0, 3.0, 3.0],
            ],
            dtype=np.float64,
        )
        cropped, keep = crop_aabb(points, np.asarray([1.0, 1.0, 1.0]), np.asarray([2.0, 2.0, 2.0]))
        self.assertEqual(cropped.shape, (2, 3))
        self.assertEqual(keep.tolist(), [False, True, True, False])

    def test_color_filters_remove_yellow_and_keep_body_colors(self) -> None:
        colors = np.asarray(
            [
                [160, 140, 35],
                [130, 55, 45],
                [80, 82, 78],
                [20, 30, 220],
            ],
            dtype=np.float64,
        ) / 255.0

        self.assertEqual(color_filter_mask(colors, "not-yellow").tolist(), [False, True, True, True])
        self.assertEqual(color_filter_mask(colors, "red-or-gray").tolist(), [False, True, True, False])

    def test_semantic_points_camera_follow_grasp_x_axis(self) -> None:
        config = {
            "semantic_points_grasp_m": {
                "grasp_center": [0.0, 0.0, 0.0],
                "tail_center": [-0.08, 0.0, 0.0],
                "head_center": [0.10, 0.0, 0.0],
            }
        }
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = np.asarray(
            [
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        transform[:3, 3] = [1.0, 2.0, 3.0]
        semantic = semantic_points_camera(config, transform)
        tail = np.asarray(semantic["tail_center_camera_m"])
        head = np.asarray(semantic["head_center_camera_m"])
        axis = head - tail
        np.testing.assert_allclose(axis / np.linalg.norm(axis), transform[:3, 0])
        np.testing.assert_allclose(semantic["grasp_center_camera_m"], [1.0, 2.0, 3.0])


if __name__ == "__main__":
    unittest.main()
