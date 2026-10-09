#!/usr/bin/env python3
"""Determine whether the gripper is holding a plug from an RGB image or video."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import cv2
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
ULTRALYTICS_DIR = ROOT / "ultralytics"
if str(ULTRALYTICS_DIR) not in sys.path:
    sys.path.insert(0, str(ULTRALYTICS_DIR))

from ultralytics import YOLO  # noqa: E402

r'''
conda activate embodiedvg

# 推理单张图片
python infer_gripper_plug_single.py \
  --rgb '检测标注（已标注）_yolo/images/val/v01_s000015_h1_t000015600_f0000468.png' \
  --output-dir output/gripper_plug_single \
  --save-overlay

# 在线推理视频
python infer_gripper_plug_single.py \
  --camera 0 \
  --output-dir output/gripper_plug_live \
  --show

# 离线推理视频
python infer_gripper_plug_single.py \
  --video 成功1.mp4 \
  --output-dir output/gripper_plug_video \
  --save-overlay
'''

DEFAULT_WEIGHTS = ROOT / "ultralytics" / "runs" / "detect" / "plug_yolo26n_det_20261009" / "weights" / "best.pt"
DEFAULT_OUTPUT = ROOT / "ultralytics" / "runs" / "gripper_plug_single"
CLASS_NAMES = {0: "gripper_with_plug", 1: "gripper_without_plug"}
CLASS_MESSAGES = {
    "gripper_with_plug": "已抓住插头",
    "gripper_without_plug": "未抓住插头",
}
VISUAL_MESSAGES = {
    "已抓住插头": "GRIPPER_WITH_PLUG",
    "未抓住插头": "GRIPPER_WITHOUT_PLUG",
    "未检测到夹爪": "NO_GRIPPER",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--rgb", type=Path, help="RGB image path.")
    source.add_argument("--video", type=Path, help="Video file path.")
    source.add_argument(
        "--camera",
        "--stream",
        dest="camera",
        help="Live camera index, device path, or RTSP/HTTP stream URL.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT, help="Directory for the JSON result.")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help="YOLO detection weights.")
    parser.add_argument(
        "--save-visualization",
        "--save-overlay",
        dest="save_visualization",
        action="store_true",
        help="Save an annotated JPG (image input) or MP4 (video/stream input) in --output-dir.",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Detection confidence threshold.",
    )
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold.")
    parser.add_argument("--device", default=None, help="CUDA device, e.g. 0, or cpu.")
    parser.add_argument("--max-det", type=int, default=10, help="Maximum detections.")
    parser.add_argument("--frame-stride", type=int, default=1, help="Process every Nth video frame.")
    parser.add_argument(
        "--show",
        "--display",
        dest="show",
        action="store_true",
        help="Show an annotated live window for --camera/--stream input; press q to stop.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop a live stream after N frames; 0 means until q or Ctrl+C.",
    )
    parser.add_argument(
        "--status-interval",
        type=float,
        default=1.0,
        help="Seconds between live status updates in the terminal.",
    )
    return parser.parse_args()


def synchronize(device: str | None) -> None:
    """Synchronize CUDA before timing GPU work when CUDA is available."""
    try:
        import torch

        if torch.cuda.is_available() and (device is None or str(device).startswith("cuda") or str(device).isdigit()):
            torch.cuda.synchronize()
    except Exception:
        pass


def choose_detection(result: Any) -> dict[str, Any] | None:
    if result.boxes is None or len(result.boxes) == 0:
        return None
    best_index = int(result.boxes.conf.argmax().item())
    class_id = int(result.boxes.cls[best_index].item())
    confidence = float(result.boxes.conf[best_index].item())
    bbox = [round(float(value), 2) for value in result.boxes.xyxy[best_index].tolist()]
    model_names = result.names
    if isinstance(model_names, dict):
        model_class_name = model_names.get(class_id, class_id)
    else:
        model_class_name = model_names[class_id] if 0 <= class_id < len(model_names) else class_id
    class_name = CLASS_NAMES.get(class_id, str(model_class_name))
    return {
        "class_id": class_id,
        "class_name": class_name,
        "confidence": confidence,
        "bbox_xyxy": bbox,
    }


def load_model(args: argparse.Namespace) -> tuple[YOLO, float]:
    if not args.weights.is_file():
        raise FileNotFoundError(f"YOLO weights not found: {args.weights}")
    model_start = time.perf_counter()
    model = YOLO(str(args.weights))
    synchronize(args.device)
    return model, time.perf_counter() - model_start


def predict_frame(
    image: Any,
    model: YOLO,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], float]:
    inference_start = time.perf_counter()
    result = model.predict(
        source=image,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        max_det=args.max_det,
        verbose=False,
    )[0]
    synchronize(args.device)
    yolo_inference_s = time.perf_counter() - inference_start

    detection = choose_detection(result)
    if detection is None:
        status = "no_detection"
        message = "未检测到夹爪"
    else:
        status = detection["class_name"]
        message = CLASS_MESSAGES.get(status, f"检测到未知类别: {status}")
    return {
        "status": status,
        "message": message,
        "detection": detection,
        "image_size": [int(image.shape[1]), int(image.shape[0])],
    }, yolo_inference_s


def infer(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, float]]:
    start = time.perf_counter()
    if not args.rgb.is_file():
        raise FileNotFoundError(f"RGB image not found: {args.rgb}")
    image = cv2.imread(str(args.rgb))
    if image is None:
        raise ValueError(f"RGB image cannot be read: {args.rgb}")

    model, model_load_s = load_model(args)
    result, yolo_inference_s = predict_frame(image, model, args)
    result["image"] = str(args.rgb)
    return result, {
        "model_load_s": model_load_s,
        "yolo_inference_s": yolo_inference_s,
        "end_to_end_s": time.perf_counter() - start,
    }


def print_result(result: dict[str, Any], timing: dict[str, float], output_dir: Path) -> None:
    print(f"Status: {result['status']}")
    detection = result.get("detection")
    if detection is not None:
        print(f"Class: {detection['class_id']} ({detection['class_name']})")
        print(f"Confidence: {detection['confidence']:.4f}")
    else:
        print("Class: N/A")
        print("Confidence: N/A")
    print(f"Output directory: {output_dir}")
    print(
        "Timing: "
        f"model load={timing['model_load_s']:.4f} s, "
        f"YOLO inference={timing['yolo_inference_s']:.4f} s, "
        f"end-to-end={timing['end_to_end_s']:.4f} s"
    )


def save_visualization(
    image_path: Path,
    detection: dict[str, Any] | None,
    message: str,
    output_path: Path,
) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"RGB image cannot be read for visualization: {image_path}")
    image = draw_visualization(image, detection, message)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise OSError(f"Failed to save visualization: {output_path}")


def draw_visualization(
    image: Any,
    detection: dict[str, Any] | None,
    message: str,
) -> Any:
    """Return an annotated copy of one BGR frame."""
    annotated = image.copy()
    if detection is not None:
        x1, y1, x2, y2 = [int(round(value)) for value in detection["bbox_xyxy"]]
        color = (50, 190, 50) if detection["class_id"] == 0 else (0, 165, 255)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
        label = f"{detection['class_name']} {detection['confidence']:.3f}"
        cv2.putText(
            annotated,
            label,
            (max(0, x1), max(30, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            2,
        )
    cv2.putText(
        annotated,
        VISUAL_MESSAGES.get(message, message),
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 0, 255),
        3,
    )
    return annotated


def process_video(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, float]]:
    """Run inference on selected frames of a video with one model instance."""
    start = time.perf_counter()
    if not args.video.is_file():
        raise FileNotFoundError(f"Video not found: {args.video}")
    if args.frame_stride < 1:
        raise ValueError(f"--frame-stride must be at least 1, got {args.frame_stride}")

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"Video cannot be opened: {args.video}")

    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(source_fps) or source_fps <= 0:
        source_fps = 30.0
    metadata_frame_count_value = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    metadata_frame_count = (
        int(metadata_frame_count_value)
        if math.isfinite(metadata_frame_count_value) and metadata_frame_count_value >= 0
        else 0
    )
    metadata_width_value = float(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    metadata_height_value = float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    metadata_width = int(metadata_width_value) if math.isfinite(metadata_width_value) else 0
    metadata_height = int(metadata_height_value) if math.isfinite(metadata_height_value) else 0
    visualization_path = args.output_dir / f"{args.video.stem}_gripper_plug.mp4"
    writer = None
    frames: list[dict[str, Any]] = []
    class_counts = Counter(
        {
            "gripper_with_plug": 0,
            "gripper_without_plug": 0,
            "no_detection": 0,
        }
    )
    yolo_inference_total_s = 0.0
    frame_index = 0
    frames_read = 0
    processed_frames = 0
    progress = None
    try:
        model, model_load_s = load_model(args)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        progress = tqdm(
            total=metadata_frame_count or None,
            desc="Processing video",
            unit="frame",
            dynamic_ncols=True,
        )
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames_read += 1
            progress.update(1)

            if writer is None and args.save_visualization:
                frame_height, frame_width = frame.shape[:2]
                width = frame_width if frame_width > 0 else metadata_width
                height = frame_height if frame_height > 0 else metadata_height
                if width <= 0 or height <= 0:
                    raise ValueError(f"Video frame has invalid dimensions: {frame.shape}")
                writer = cv2.VideoWriter(
                    str(visualization_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    source_fps,
                    (width, height),
                )
                if not writer.isOpened():
                    writer.release()
                    writer = None
                    raise OSError(f"Failed to create visualization video: {visualization_path}")

            if frame_index % args.frame_stride == 0:
                frame_result, inference_s = predict_frame(frame, model, args)
                yolo_inference_total_s += inference_s
                frame_result.update(
                    {
                        "frame_index": frame_index,
                        "time_sec": round(frame_index / source_fps, 6),
                    }
                )
                frames.append(frame_result)
                class_counts[frame_result["status"]] += 1
                processed_frames += 1
                if writer is not None:
                    writer.write(
                        draw_visualization(frame, frame_result["detection"], frame_result["message"])
                    )
            elif writer is not None:
                # Keep the output video at the original frame rate. Frames skipped
                # by --frame-stride are copied without a new prediction overlay.
                writer.write(frame)

            frame_index += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if progress is not None:
            progress.close()

    if frames_read == 0:
        if args.save_visualization:
            visualization_path.unlink(missing_ok=True)
        raise ValueError(f"Video contains no readable frames: {args.video}")

    end_to_end_s = time.perf_counter() - start
    # OpenCV metadata can be missing for some codecs; actual dimensions from
    # decoded frames are more useful in the saved result when that happens.
    if frames:
        actual_height, actual_width = frames[0]["image_size"][1], frames[0]["image_size"][0]
    else:
        actual_width, actual_height = metadata_width, metadata_height
    result: dict[str, Any] = {
        "video": str(args.video),
        "video_info": {
            "fps": source_fps,
            "width": actual_width,
            "height": actual_height,
            "metadata_frame_count": metadata_frame_count,
            "frames_read": frames_read,
            "frames_processed": processed_frames,
            "frame_stride": args.frame_stride,
        },
        "class_counts": dict(class_counts),
        "frames": frames,
    }
    if args.save_visualization:
        result["visualization"] = str(visualization_path)
    timing = {
        "model_load_s": model_load_s,
        "yolo_inference_s": yolo_inference_total_s,
        "end_to_end_s": end_to_end_s,
        "average_per_processed_frame_s": yolo_inference_total_s / processed_frames,
    }
    return result, timing


def parse_camera_source(camera: str) -> int | str:
    """Convert a numeric camera argument to an OpenCV device index."""
    value = camera.strip()
    if value.lstrip("-").isdigit():
        return int(value)
    return camera


def print_live_update(frame_result: dict[str, Any], frame_index: int) -> None:
    detection = frame_result.get("detection")
    if detection is None:
        class_text = "Class: N/A | Confidence: N/A"
    else:
        class_text = (
            f"Class: {detection['class_id']} ({detection['class_name']}) | "
            f"Confidence: {detection['confidence']:.4f}"
        )
    print(
        f"\rFrame: {frame_index} | Status: {frame_result['status']} | {class_text}",
        end="",
        flush=True,
    )


def process_live_stream(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, float]]:
    """Run inference on a live camera or network stream until it stops."""
    start = time.perf_counter()
    if args.frame_stride < 1:
        raise ValueError(f"--frame-stride must be at least 1, got {args.frame_stride}")
    if args.max_frames < 0:
        raise ValueError(f"--max-frames must be non-negative, got {args.max_frames}")
    if args.status_interval < 0:
        raise ValueError(f"--status-interval must be non-negative, got {args.status_interval}")

    source = parse_camera_source(args.camera)
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"Live source cannot be opened: {args.camera}")

    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(source_fps) or source_fps <= 0:
        source_fps = 30.0
    metadata_width_value = float(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    metadata_height_value = float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    metadata_width = int(metadata_width_value) if math.isfinite(metadata_width_value) else 0
    metadata_height = int(metadata_height_value) if math.isfinite(metadata_height_value) else 0

    visualization_path = args.output_dir / "live_gripper_plug.mp4"
    writer = None
    model_load_s = 0.0
    frames_read = 0
    processed_frames = 0
    frame_index = 0
    yolo_inference_total_s = 0.0
    last_result: dict[str, Any] | None = None
    class_counts = Counter(
        {
            "gripper_with_plug": 0,
            "gripper_without_plug": 0,
            "no_detection": 0,
        }
    )
    interrupted = False
    status_printed = False
    last_status_time = -float("inf")
    stream_start = start

    try:
        model, model_load_s = load_model(args)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        stream_start = time.perf_counter()
        print("Live stream started. Press Ctrl+C to stop.")
        if args.show:
            print("Press q in the display window to stop.")

        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames_read += 1
            frame_to_show = frame

            if writer is None and args.save_visualization:
                frame_height, frame_width = frame.shape[:2]
                width = frame_width if frame_width > 0 else metadata_width
                height = frame_height if frame_height > 0 else metadata_height
                if width <= 0 or height <= 0:
                    raise ValueError(f"Live frame has invalid dimensions: {frame.shape}")
                writer = cv2.VideoWriter(
                    str(visualization_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    source_fps,
                    (width, height),
                )
                if not writer.isOpened():
                    writer.release()
                    writer = None
                    raise OSError(f"Failed to create live visualization video: {visualization_path}")

            if frame_index % args.frame_stride == 0:
                frame_result, inference_s = predict_frame(frame, model, args)
                yolo_inference_total_s += inference_s
                frame_result.update(
                    {
                        "frame_index": frame_index,
                        "time_sec": round(time.perf_counter() - stream_start, 6),
                    }
                )
                last_result = frame_result
                class_counts[frame_result["status"]] += 1
                processed_frames += 1
                frame_to_show = draw_visualization(
                    frame,
                    frame_result["detection"],
                    frame_result["message"],
                )

                now = time.perf_counter()
                if not status_printed or now - last_status_time >= args.status_interval:
                    print_live_update(frame_result, frame_index)
                    status_printed = True
                    last_status_time = now

            if writer is not None:
                writer.write(frame_to_show)

            if args.show:
                try:
                    cv2.imshow("Gripper plug detection", frame_to_show)
                    key = cv2.waitKey(1) & 0xFF
                except cv2.error as error:
                    raise ValueError(
                        "The live display cannot be opened; omit --show in a headless session."
                    ) from error
                if key == ord("q"):
                    interrupted = True
                    break

            frame_index += 1
            if args.max_frames and frames_read >= args.max_frames:
                interrupted = True
                break
    except KeyboardInterrupt:
        interrupted = True
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if args.show:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass
        if status_printed:
            print()

    if frames_read == 0:
        if args.save_visualization:
            visualization_path.unlink(missing_ok=True)
        raise ValueError(f"Live source produced no readable frames: {args.camera}")

    end_to_end_s = time.perf_counter() - start
    if last_result is not None:
        actual_height, actual_width = last_result["image_size"][1], last_result["image_size"][0]
    else:
        actual_width, actual_height = metadata_width, metadata_height
    result: dict[str, Any] = {
        "source": str(args.camera),
        "mode": "live",
        "video_info": {
            "fps": source_fps,
            "width": actual_width,
            "height": actual_height,
            "frames_read": frames_read,
            "frames_processed": processed_frames,
            "frame_stride": args.frame_stride,
        },
        "class_counts": dict(class_counts),
        "last_result": last_result,
        "stopped_by_user": interrupted,
    }
    if args.save_visualization:
        result["visualization"] = str(visualization_path)
    timing = {
        "model_load_s": model_load_s,
        "yolo_inference_s": yolo_inference_total_s,
        "end_to_end_s": end_to_end_s,
        "average_per_processed_frame_s": (
            yolo_inference_total_s / processed_frames if processed_frames else 0.0
        ),
    }
    return result, timing


def print_video_result(result: dict[str, Any], timing: dict[str, float], output_dir: Path) -> None:
    video_info = result["video_info"]
    counts = result["class_counts"]
    print("Status: video_processed")
    print(f"Frames: {video_info['frames_processed']}/{video_info['frames_read']}")
    print(
        "Class counts: "
        f"gripper_with_plug={counts['gripper_with_plug']}, "
        f"gripper_without_plug={counts['gripper_without_plug']}, "
        f"no_detection={counts['no_detection']}"
    )
    print(f"Output directory: {output_dir}")
    print(
        "Timing: "
        f"model load={timing['model_load_s']:.4f} s, "
        f"YOLO inference total={timing['yolo_inference_s']:.4f} s, "
        f"end-to-end={timing['end_to_end_s']:.4f} s, "
        f"average per processed frame={timing['average_per_processed_frame_s']:.4f} s"
    )


def print_live_result(result: dict[str, Any], timing: dict[str, float], output_dir: Path) -> None:
    video_info = result["video_info"]
    counts = result["class_counts"]
    print("Status: live_stream_stopped")
    print(f"Frames: {video_info['frames_processed']}/{video_info['frames_read']}")
    print(
        "Class counts: "
        f"gripper_with_plug={counts['gripper_with_plug']}, "
        f"gripper_without_plug={counts['gripper_without_plug']}, "
        f"no_detection={counts['no_detection']}"
    )
    print(f"Output directory: {output_dir}")
    print(
        "Timing: "
        f"model load={timing['model_load_s']:.4f} s, "
        f"YOLO inference total={timing['yolo_inference_s']:.4f} s, "
        f"end-to-end={timing['end_to_end_s']:.4f} s, "
        f"average per processed frame={timing['average_per_processed_frame_s']:.4f} s"
    )


def main() -> int:
    args = parse_args()
    try:
        if args.camera is not None:
            result, timing = process_live_stream(args)
            output_path = args.output_dir / "live_gripper_plug.json"
        elif args.video is not None:
            result, timing = process_video(args)
            output_path = args.output_dir / f"{args.video.stem}_gripper_plug.json"
        else:
            result, timing = infer(args)
            output_path = args.output_dir / f"{args.rgb.stem}_gripper_plug.json"
    except (FileNotFoundError, OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.camera is None and args.video is None and args.save_visualization:
        visualization_path = args.output_dir / f"{args.rgb.stem}_gripper_plug.jpg"
        save_visualization(args.rgb, result.get("detection"), result["message"], visualization_path)
        result["visualization"] = str(visualization_path)
    output_path.write_text(
        json.dumps({"result": result, "timing": timing}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.camera is not None:
        print_live_result(result, timing, args.output_dir)
    elif args.video is not None:
        print_video_result(result, timing, args.output_dir)
    else:
        print_result(result, timing, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
