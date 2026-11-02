#!/usr/bin/env python3
"""Create an RGB/depth two-column review image for one endpoint result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb", type=Path, required=True)
    parser.add_argument("--depth", type=Path, required=True)
    parser.add_argument("--endpoint-overlay", type=Path, required=True)
    parser.add_argument("--endpoint-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--depth-scale", type=float, default=0.001)
    return parser.parse_args()


def fit_depth_colormap(depth_raw: np.ndarray, scale: float) -> tuple[np.ndarray, np.ndarray]:
    depth_m = depth_raw.astype(np.float32) * scale
    valid = np.isfinite(depth_m) & (depth_m > 0.1) & (depth_m < 1.5)
    if not valid.any():
        return np.zeros((*depth_m.shape, 3), dtype=np.uint8), valid
    lo, hi = np.percentile(depth_m[valid], [2, 98])
    normalized = np.zeros_like(depth_m, dtype=np.uint8)
    normalized[valid] = np.clip((depth_m[valid] - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
    # Nearer points are warmer and farther points cooler.
    return cv2.applyColorMap(255 - normalized, cv2.COLORMAP_TURBO), valid


def add_title(image: np.ndarray, title: str) -> np.ndarray:
    bar = np.zeros((44, image.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, title, (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (240, 240, 240), 2, cv2.LINE_AA)
    return np.vstack([bar, image])


def main() -> None:
    args = parse_args()
    rgb = cv2.imread(str(args.rgb), cv2.IMREAD_COLOR)
    endpoint_overlay = cv2.imread(str(args.endpoint_overlay), cv2.IMREAD_COLOR)
    depth_raw = np.load(args.depth)
    if rgb is None or endpoint_overlay is None:
        raise ValueError("RGB or endpoint overlay cannot be read")
    if depth_raw.ndim != 2 or depth_raw.shape != rgb.shape[:2]:
        raise ValueError(f"Depth shape {depth_raw.shape} does not match RGB {rgb.shape[:2]}")
    result = json.loads(args.endpoint_json.read_text(encoding="utf-8"))
    target = result.get("localization", {}).get("target_pixel")
    if target is None:
        raise ValueError("Endpoint JSON has no localization.target_pixel")
    u, v = (int(round(float(target[0]))), int(round(float(target[1]))))
    radius = int(result.get("endpoint_settings", {}).get("patch_radius_px", 48))

    depth_view, valid = fit_depth_colormap(depth_raw, args.depth_scale)
    blended = cv2.addWeighted(depth_view, 0.58, endpoint_overlay, 0.42, 0)
    yy, xx = np.ogrid[:depth_raw.shape[0], :depth_raw.shape[1]]
    circle = (xx - u) ** 2 + (yy - v) ** 2 <= radius ** 2
    valid_count = int((circle & valid).sum())
    cv2.circle(blended, (u, v), radius, (0, 215, 255), 3, cv2.LINE_AA)
    cv2.drawMarker(blended, (u, v), (255, 255, 255), cv2.MARKER_CROSS, 26, 3, cv2.LINE_AA)
    cv2.putText(blended, f"fit candidate: r={radius}px", (u + radius + 12, v - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 215, 255), 2, cv2.LINE_AA)
    cv2.putText(blended, f"valid depth samples: {valid_count}", (18, blended.shape[0] - 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)
    status = result.get("status", "unknown")
    reason = result.get("reason", "")
    message = f"endpoint fit: {status}"
    if reason:
        message += f" - {reason}"
    cv2.putText(blended, message[:110], (18, blended.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (40, 220, 255) if status != "ok" else (80, 255, 80), 2, cv2.LINE_AA)

    combined = np.hstack([add_title(rgb, "RGB original"), add_title(blended, "Depth + endpoint overlay + fit candidate range")])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), combined):
        raise OSError(f"Could not write {args.output}")
    print(json.dumps({"output": str(args.output), "target_pixel": [u, v], "radius_px": radius,
                      "valid_depth_samples": valid_count, "status": status, "reason": reason}, ensure_ascii=False))


if __name__ == "__main__":
    main()
