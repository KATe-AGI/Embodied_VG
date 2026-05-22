#!/usr/bin/env python3
"""Run single-plug grasp-region segmentation and head/tail pose inference on images or videos."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

from plug_vg.config import DATASET, DEFAULT_POSE_WEIGHTS, DEFAULT_SEG_WEIGHTS, RGBD_TEST, ULTRALYTICS_DIR, YOLO_DATASET
from plug_vg.io import collect_sources
from plug_vg.vision import draw_overlay, run_models, serialize_pose, serialize_seg

ROOT = Path(__file__).resolve().parent
if str(ULTRALYTICS_DIR) not in sys.path:
    sys.path.insert(0, str(ULTRALYTICS_DIR))

from ultralytics import YOLO  # noqa: E402


SOURCE_PRESETS = {
    "rgbd-test": RGBD_TEST / "color",
    "seg-val": YOLO_DATASET / "seg" / "images" / "val",
    "pose-val": YOLO_DATASET / "pose" / "images" / "val",
}
DEFAULT_SOURCE_PRESET = "rgbd-test"
DEFAULT_OUTPUT = ULTRALYTICS_DIR / "runs" / "plug_stage1_infer"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Image/video file, directory, or glob pattern. Overrides --source-preset.",
    )
    parser.add_argument(
        "--source-preset",
        choices=tuple(SOURCE_PRESETS),
        default=DEFAULT_SOURCE_PRESET,
        help="Named RGB source used when --source is omitted. Inference is source-agnostic; seg-val and pose-val are just validation image subsets.",
    )
    parser.add_argument("--seg-weights", type=Path, default=DEFAULT_SEG_WEIGHTS, help="Segmentation weights.")
    parser.add_argument("--pose-weights", type=Path, default=DEFAULT_POSE_WEIGHTS, help="Pose weights.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output directory.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.7, help="IoU threshold.")
    parser.add_argument("--device", default=None, help="CUDA device, e.g. 0, or cpu.")
    parser.add_argument("--max-det", type=int, default=10, help="Maximum detections per model.")
    parser.add_argument("--frame-stride", type=int, default=1, help="Process every Nth video frame.")
    parser.add_argument("--no-save-video", action="store_true", help="Do not save annotated video outputs.")
    return parser.parse_args()


def resolve_source(args: argparse.Namespace) -> Path:
    return args.source if args.source is not None else SOURCE_PRESETS[args.source_preset]


def infer_image(
    image_path: Path,
    seg_model: YOLO,
    pose_model: YOLO,
    args: argparse.Namespace,
    json_dir: Path,
    overlay_dir: Path,
) -> dict | None:
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Skipping unreadable image: {image_path}")
        return None

    seg_items, pose_items = run_models(image, seg_model, pose_model, args)
    record = {
        "type": "image",
        "image": str(image_path),
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
        "segmentation": seg_items,
        "pose": pose_items,
    }
    with (json_dir / f"{image_path.stem}.json").open("w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    overlay = draw_overlay(image, seg_items, pose_items)
    cv2.imwrite(str(overlay_dir / f"{image_path.stem}.jpg"), overlay)
    return record


def infer_video(
    video_path: Path,
    seg_model: YOLO,
    pose_model: YOLO,
    args: argparse.Namespace,
    json_dir: Path,
    video_dir: Path,
) -> dict | None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Skipping unreadable video: {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stride = max(1, int(args.frame_stride))

    writer = None
    video_output = None
    if not args.no_save_video:
        video_dir.mkdir(parents=True, exist_ok=True)
        video_output = video_dir / f"{video_path.stem}_overlay.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_output), fourcc, fps / stride, (width, height))

    frames = []
    frame_index = 0
    processed = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index % stride != 0:
            frame_index += 1
            continue

        seg_items, pose_items = run_models(frame, seg_model, pose_model, args)
        frames.append(
            {
                "frame_index": frame_index,
                "time_sec": None if fps <= 0 else round(float(frame_index / fps), 6),
                "width": width,
                "height": height,
                "segmentation": seg_items,
                "pose": pose_items,
            }
        )
        if writer is not None:
            writer.write(draw_overlay(frame, seg_items, pose_items))
        processed += 1
        frame_index += 1

    cap.release()
    if writer is not None:
        writer.release()

    record = {
        "type": "video",
        "video": str(video_path),
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "frame_stride": stride,
        "processed_frames": processed,
        "overlay_video": None if video_output is None else str(video_output),
        "frames": frames,
    }
    with (json_dir / f"{video_path.stem}.json").open("w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    return record


def main() -> None:
    args = parse_args()
    source = resolve_source(args)
    images, videos = collect_sources(source)
    args.output.mkdir(parents=True, exist_ok=True)
    json_dir = args.output / "jsons"
    json_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = args.output / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    video_dir = args.output / "videos"

    seg_model = YOLO(str(args.seg_weights))
    pose_model = YOLO(str(args.pose_weights))
    summary = []

    for image_path in images:
        record = infer_image(image_path, seg_model, pose_model, args, json_dir, overlay_dir)
        if record is not None:
            summary.append(record)

    for video_path in videos:
        record = infer_video(video_path, seg_model, pose_model, args, json_dir, video_dir)
        if record is not None:
            summary.append(record)

    with (json_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Processed {len(images)} image source(s) and {len(videos)} video source(s). Results saved to {args.output}")


if __name__ == "__main__":
    main()
