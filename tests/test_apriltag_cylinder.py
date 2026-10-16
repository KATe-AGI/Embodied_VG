from __future__ import annotations

from types import SimpleNamespace
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import cv2
import numpy as np

from infer_apriltag_cylinder import detect_tag, tag_object_corners
from plug_vg.apriltag_pose import center_normal_to_base, estimate_tag_pose, get_tag_center_normal
from plug_vg.robot_transform import robot_pose_to_matrix


class AprilTagPoseTests(unittest.TestCase):
    def setUp(self) -> None:
        # AprilRobotics/apriltag-imgs: tagStandard41h12/tag41_12_00000.png.
        # The full 9x9 pattern is 180 pixels; the pose square is 100 pixels.
        bits = ["110111100", "000000000", "101111100", "001111101", "001001100",
                "001010101", "101111100", "000000000", "110100100"]
        marker = np.array([[int(bit) * 255 for bit in row] for row in bits], np.uint8)
        image = np.full((480, 640), 255, np.uint8)
        image[150:330, 230:410] = cv2.resize(marker, (180, 180), interpolation=cv2.INTER_NEAREST)
        self.image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        self.camera = {"fx": 1800.0, "fy": 1800.0, "cx": 320.0, "cy": 240.0}

    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux display regression")
    def test_show_without_display_exits_successfully_and_saves_visualization(self) -> None:
        import resource

        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            rgb = directory / "tag.png"
            cv2.imwrite(str(rgb), self.image)
            config = directory / "camera.yaml"
            config.write_text(json.dumps({**self.camera, "image_width": 640, "image_height": 480,
                                          "depth_scale": 0.001}))
            env = os.environ.copy()
            for key in ("DISPLAY", "WAYLAND_DISPLAY"):
                env.pop(key, None)
            env["BROWSER"] = "/bin/false"
            process = subprocess.run(
                [sys.executable, str(root / "infer_apriltag_cylinder.py"), "--rgb", str(rgb),
                 "--camera-config", str(config), "--robot-pose", "0", "0", "0", "0", "0", "0",
                 "--output-dir", str(directory), "--show"],
                cwd=root, env=env, capture_output=True, text=True, timeout=20,
                preexec_fn=lambda: resource.setrlimit(resource.RLIMIT_CORE, (0, 0)),
            )
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertNotIn("qt.qpa", process.stderr)
            result = json.loads((directory / "tag_apriltag_pose.json").read_text())
            self.assertEqual(result["status"], "ok")
            placement = result["placement_points_base_m"]
            np.testing.assert_allclose(np.array(placement["preset"])-placement["reference"],
                                       .22*np.array(result["tag_normal_base"]), atol=1e-8)
            np.testing.assert_allclose(np.array(placement["head_preset"])-placement["preset"], [0,0,.08], atol=1e-8)
            np.testing.assert_allclose(np.array(placement["tail_preset"])-placement["preset"], [0,0,-.087], atol=1e-8)
            html = Path(result["artifacts"]["base_view_html"]).read_text()
            self.assertIn("data:image/jpeg;base64,", html)
            self.assertIn("Plotly.newPlot", html)
            self.assertIn("No graphical display", process.stdout)

    def test_official_estimator_metric_scale_and_size_override(self) -> None:
        tag = detect_tag(self.image)
        rotation, translation, error = estimate_tag_pose(tag, self.camera, 0.018)
        # Blue pose square is 100 px: 1800 px * 0.018 m / 100 px = 0.324 m.
        np.testing.assert_allclose(translation, [0, 0, 0.324], atol=0.002)
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-8)
        self.assertTrue(np.isfinite(error))
        _, twice, _ = estimate_tag_pose(tag, self.camera, 0.036)
        np.testing.assert_allclose(twice, 2 * translation, atol=1e-8)

    def test_known_tilted_pose_recovers_center_and_camera_facing_normal(self) -> None:
        expected_rotation = robot_pose_to_matrix([0, 0, 0, 0.35, -0.25, 0.6])[:3, :3]
        expected_center = np.array([0.07, -0.04, 0.6])
        local = tag_object_corners(0.018)
        points = local @ expected_rotation.T + expected_center
        pixels = points[:, :2] / points[:, 2, None] * 1800 + [320, 240]
        # Homography maps the official normalized square to observed pixels.
        source = (local[:, :2] / (0.018 / 2)).astype(np.float32)
        tag = SimpleNamespace(corners=pixels, homography=cv2.getPerspectiveTransform(source, pixels.astype(np.float32)))
        rotation, translation, _ = estimate_tag_pose(tag, self.camera, 0.018)
        center, normal = get_tag_center_normal(rotation, translation)
        np.testing.assert_allclose(center, expected_center, atol=1e-6)
        np.testing.assert_allclose(normal, -expected_rotation[:, 2], atol=1e-5)
        self.assertGreater(np.dot(normal, -center), 0)

    def test_normal_flip_and_roll_equivalence(self) -> None:
        center = np.array([0.1, 0.2, 0.7])
        for roll in [0, 0.7, -1.9]:
            rotation = robot_pose_to_matrix([0, 0, 0, 0, 0, roll])[:3, :3]
            _, normal = get_tag_center_normal(rotation, center)
            np.testing.assert_allclose(normal, [0, 0, -1], atol=1e-12)
        _, already_facing = get_tag_center_normal(np.diag([1, -1, -1]), center)
        np.testing.assert_allclose(already_facing, [0, 0, -1], atol=1e-12)

    def test_base_point_uses_translation_normal_uses_rotation_only(self) -> None:
        transform = robot_pose_to_matrix([0.2, -0.3, 0.6, 0.7, -0.4, 1.2])
        camera_center = np.array([0.1, 0.2, 0.7])
        camera_normal = np.array([0, 0, -1])
        center, normal = center_normal_to_base(camera_center, camera_normal, transform)
        expected_point = (transform @ np.r_[camera_center, 1])[:3]
        expected_direction = (transform @ np.r_[camera_normal, 0])[:3]
        np.testing.assert_allclose(center, expected_point, atol=1e-12)
        np.testing.assert_allclose(normal, expected_direction, atol=1e-12)
        self.assertAlmostEqual(float(np.linalg.norm(normal)), 1.0)
        self.assertGreater(np.dot(normal, transform[:3, 3] - center), 0)
        transform[:3, 3] += [10, 20, 30]
        _, moved_normal = center_normal_to_base(camera_center, camera_normal, transform)
        np.testing.assert_allclose(moved_normal, normal, atol=1e-12)

    def test_blank_and_duplicate_id_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "No correctly decoded"):
            detect_tag(np.full_like(self.image, 255))
        with self.assertRaisesRegex(ValueError, "Multiple ID 0"):
            detect_tag(np.hstack([self.image, self.image]))

    def test_invalid_size_and_nonrigid_transform_are_rejected(self) -> None:
        tag = detect_tag(self.image)
        for size in [0, -0.04, float("nan")]:
            with self.assertRaises(ValueError):
                estimate_tag_pose(tag, self.camera, size)
        transform = np.eye(4)
        transform[0, 0] = 2
        with self.assertRaisesRegex(ValueError, "rigid transform"):
            center_normal_to_base(np.array([0, 0, 0.6]), np.array([0, 0, -1]), transform)


if __name__ == "__main__":
    unittest.main()
