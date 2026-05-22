"""YOLO stage1 serialization and visualization helpers."""

from __future__ import annotations

import cv2
import numpy as np


def xyxy_list(boxes, idx: int) -> list[float] | None:
    if boxes is None or len(boxes) <= idx:
        return None
    return [round(float(v), 3) for v in boxes.xyxy[idx].cpu().numpy().tolist()]


def conf_value(boxes, idx: int) -> float | None:
    if boxes is None or boxes.conf is None or len(boxes) <= idx:
        return None
    return round(float(boxes.conf[idx].cpu().item()), 6)


def serialize_seg(result) -> list[dict]:
    detections = []
    polygons = [] if result.masks is None else result.masks.xy
    for i, poly in enumerate(polygons):
        points = [[round(float(x), 3), round(float(y), 3)] for x, y in np.asarray(poly).tolist()]
        detections.append(
            {
                "label": "plug_grasp_region",
                "confidence": conf_value(result.boxes, i),
                "bbox_xyxy": xyxy_list(result.boxes, i),
                "polygon_xy": points,
            }
        )
    if not detections:
        return []
    return [max(detections, key=lambda item: -1.0 if item.get("confidence") is None else float(item["confidence"]))]


def serialize_pose(result) -> list[dict]:
    detections = []
    if result.keypoints is None or result.keypoints.xy is None:
        return detections

    xy = result.keypoints.xy.cpu().numpy()
    conf = None if result.keypoints.conf is None else result.keypoints.conf.cpu().numpy()
    for i, keypoints in enumerate(xy):
        item = {
            "label": "plug",
            "confidence": conf_value(result.boxes, i),
            "bbox_xyxy": xyxy_list(result.boxes, i),
            "keypoints": {
                "plug_head": {
                    "xy": [round(float(keypoints[0][0]), 3), round(float(keypoints[0][1]), 3)],
                    "confidence": None if conf is None else round(float(conf[i][0]), 6),
                },
                "plug_tail": {
                    "xy": [round(float(keypoints[1][0]), 3), round(float(keypoints[1][1]), 3)],
                    "confidence": None if conf is None else round(float(conf[i][1]), 6),
                },
            },
        }
        detections.append(item)
    if not detections:
        return []
    return [max(detections, key=lambda item: -1.0 if item.get("confidence") is None else float(item["confidence"]))]


def draw_overlay(image_bgr: np.ndarray, seg_items: list[dict], pose_items: list[dict]) -> np.ndarray:
    canvas = image_bgr.copy()
    mask_layer = canvas.copy()

    for item in seg_items:
        pts = np.asarray(item["polygon_xy"], dtype=np.int32)
        if pts.size:
            cv2.fillPoly(mask_layer, [pts], color=(40, 180, 60))
            cv2.polylines(canvas, [pts], isClosed=True, color=(20, 220, 80), thickness=2)
    canvas = cv2.addWeighted(mask_layer, 0.28, canvas, 0.72, 0)

    for item in pose_items:
        head = item["keypoints"]["plug_head"]["xy"]
        tail = item["keypoints"]["plug_tail"]["xy"]
        head_pt = (int(round(head[0])), int(round(head[1])))
        tail_pt = (int(round(tail[0])), int(round(tail[1])))
        cv2.line(canvas, tail_pt, head_pt, color=(0, 220, 255), thickness=2)
        cv2.circle(canvas, head_pt, 8, color=(0, 0, 255), thickness=-1)
        cv2.circle(canvas, tail_pt, 8, color=(255, 0, 0), thickness=-1)
        cv2.putText(canvas, "head", (head_pt[0] + 10, head_pt[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 0, 255), 3)
        cv2.putText(canvas, "tail", (tail_pt[0] + 10, tail_pt[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 0, 0), 3)
    return canvas


def run_models(image_bgr: np.ndarray, seg_model, pose_model, args) -> tuple[list[dict], list[dict]]:
    seg_result = seg_model.predict(
        source=image_bgr,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        max_det=args.max_det,
        verbose=False,
    )[0]
    pose_result = pose_model.predict(
        source=image_bgr,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        max_det=args.max_det,
        verbose=False,
    )[0]
    return serialize_seg(seg_result), serialize_pose(pose_result)

