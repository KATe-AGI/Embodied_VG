#!/usr/bin/env python3
"""Run 6D plug-grasp inference for every RGB/D2RGB pair in a directory."""

from __future__ import annotations

import argparse
from pathlib import Path

from infer_6d_single import (
    DEFAULT_CAMERA,
    DEFAULT_GRASP_MODEL_CONFIG,
    DEFAULT_HAND_EYE,
    DEFAULT_ROBOT_CONFIG,
    DEFAULT_SEG_WEIGHTS,
    print_result,
    run,
)


r"""
# conda activate embodiedvg

python infer_6d_batch.py \
  --input-dir test_20260705 \
  --robot-pose -0.014293 0.460711 0.742759 2.167158 0.044541 -3.126827 \
  --output-dir output/test_0825 \
  --save-ply \
  --save-review
"""


RGB_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEPTH_SUFFIXES = (".npy", ".png", ".tif", ".tiff")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing matching <sample>_color images and <sample>_d2rgb depth files.",
    )
    parser.add_argument(
        "--robot-pose",
        type=float,
        nargs=6,
        required=True,
        metavar=("X", "Y", "Z", "ROLL", "PITCH", "YAW"),
        help="Robot end-effector pose shared by all frames, in meters/radians.",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for per-frame outputs.")
    parser.add_argument("--seg-weights", type=Path, default=DEFAULT_SEG_WEIGHTS)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_GRASP_MODEL_CONFIG)
    parser.add_argument("--camera-config", type=Path, default=DEFAULT_CAMERA)
    parser.add_argument("--hand-eye-config", type=Path, default=DEFAULT_HAND_EYE)
    parser.add_argument("--robot-config", type=Path, default=DEFAULT_ROBOT_CONFIG)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-det", type=int, default=10)
    parser.add_argument("--min-depth", type=float, default=0.1)
    parser.add_argument("--max-depth", type=float, default=1.0)
    parser.add_argument("--mask-erosion-px", type=int, default=5)
    parser.add_argument("--min-visible-points", type=int, default=200)
    parser.add_argument("--voxel-size", type=float, default=0.004)
    parser.add_argument("--mad-z-threshold", type=float, default=3.5)
    parser.add_argument("--icp-threshold", type=float, default=0.015)
    parser.add_argument("--icp-iterations", type=int, default=100)
    parser.add_argument("--registration-method", choices=("legacy", "symmetric"), default="symmetric")
    parser.add_argument("--max-model-points", type=int, default=12000)
    parser.add_argument("--max-scene-points", type=int, default=12000)
    parser.add_argument("--min-registration-fitness", type=float, default=0.35)
    parser.add_argument("--max-inlier-rmse", type=float, default=0.012)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--save-overlay", action="store_true")
    parser.add_argument("--save-ply", action="store_true")
    parser.add_argument("--save-review", action="store_true")
    return parser.parse_args()


def discover_pairs(input_dir: Path) -> list[tuple[Path, Path]]:
    """Find immediate-child ``*_color`` images and their D2RGB counterparts."""

    rgb_paths = sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in RGB_SUFFIXES
        and path.stem.lower().endswith("_color")
    )
    pairs: list[tuple[Path, Path]] = []
    for rgb_path in rgb_paths:
        sample_stem = rgb_path.stem[: -len("_color")]
        candidates = [input_dir / f"{sample_stem}_d2rgb{suffix}" for suffix in DEPTH_SUFFIXES]
        depth_path = next((path for path in candidates if path.is_file()), candidates[0])
        pairs.append((rgb_path, depth_path))
    return pairs


def frame_args(batch_args: argparse.Namespace, rgb_path: Path, depth_path: Path) -> argparse.Namespace:
    values = vars(batch_args).copy()
    values.pop("input_dir")
    values["rgb"] = rgb_path
    values["d2rgb"] = depth_path
    return argparse.Namespace(**values)


def main() -> int:
    args = parse_args()
    if not args.input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {args.input_dir}")

    pairs = discover_pairs(args.input_dir)
    if not pairs:
        raise SystemExit(f"No *_color images found in input directory: {args.input_dir}")

    ok_count = 0
    for index, (rgb_path, depth_path) in enumerate(pairs, start=1):
        print(f"\n=== frame {index}/{len(pairs)}: {rgb_path.name} ===")
        result, json_path = run(frame_args(args, rgb_path, depth_path))
        print_result(result, json_path)
        ok_count += int(result.get("status") == "ok")

    failed_count = len(pairs) - ok_count
    print(f"\nBatch summary: total={len(pairs)}, ok={ok_count}, failed={failed_count}")
    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
