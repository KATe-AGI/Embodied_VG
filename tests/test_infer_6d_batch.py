from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

from infer_6d_batch import discover_pairs, frame_args


class BatchDiscoveryTests(unittest.TestCase):
    def test_flat_pairs_and_capture_layout_use_distinct_output_directories(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            flat = root / "flat"
            flat.mkdir()
            (flat / "frame_color.png").touch()
            (flat / "frame_d2rgb.npy").touch()

            capture = root / "capture" / "input"
            plug = capture / "20261104_001_plug"
            gripper = capture / "20261104_002_gripper"
            plug.mkdir(parents=True)
            gripper.mkdir()
            (plug / "rgb.png").touch()
            (plug / "depth.npy").touch()
            (gripper / "rgb.png").touch()

            self.assertEqual(discover_pairs(flat), [(flat / "frame_color.png", flat / "frame_d2rgb.npy")])
            self.assertEqual(discover_pairs(root / "capture"), [(plug / "rgb.png", plug / "depth.npy")])

            args = argparse.Namespace(input_dir=root / "capture", output_dir=root / "results", camera_only=True)
            frame = frame_args(args, plug / "rgb.png", plug / "depth.npy")
            self.assertEqual(frame.output_dir, root / "results" / plug.name)
            self.assertFalse(hasattr(frame, "input_dir"))


if __name__ == "__main__":
    unittest.main()
