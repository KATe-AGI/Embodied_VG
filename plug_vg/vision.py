"""YOLO segmentation serialization and visualization helpers."""

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
                "label": "visible_plug",
                "confidence": conf_value(result.boxes, i),
                "bbox_xyxy": xyxy_list(result.boxes, i),
                "polygon_xy": points,
                "polygon": points,
            }
        )
    if not detections:
        return []
    return [max(detections, key=lambda item: -1.0 if item.get("confidence") is None else float(item["confidence"]))]

def draw_overlay(image_bgr: np.ndarray, seg_items: list[dict]) -> np.ndarray:
    canvas = image_bgr.copy()
    mask_layer = canvas.copy()

    for item in seg_items:
        pts = np.asarray(item.get("polygon_xy") or item.get("polygon") or [], dtype=np.int32)
        if pts.size:
            cv2.fillPoly(mask_layer, [pts], color=(40, 180, 60))
            cv2.polylines(canvas, [pts], isClosed=True, color=(20, 220, 80), thickness=2)
    canvas = cv2.addWeighted(mask_layer, 0.28, canvas, 0.72, 0)
    return canvas


def run_segmentation(image_bgr: np.ndarray, seg_model, args) -> list[dict]:
    seg_result = seg_model.predict(
        source=image_bgr,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        max_det=args.max_det,
        verbose=False,
    )[0]
    return serialize_seg(seg_result)
