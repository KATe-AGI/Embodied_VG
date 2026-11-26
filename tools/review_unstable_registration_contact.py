#!/usr/bin/env python3
"""Create a temporary RGB contact sheet to inspect saved registration candidates."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "test_data/plug_6d_camera_benchmark_20261105"
OUTPUT = ROOT / "output/unstable_plug_camera_registrations_20261106"


def project(point: np.ndarray, camera: dict) -> tuple[int, int]:
    return (
        round(camera["fx"] * point[0] / point[2] + camera["cx"]),
        round(camera["fy"] * point[1] / point[2] + camera["cy"]),
    )


def panel(image: np.ndarray, pose: np.ndarray, camera: dict, bbox: list, label: str) -> np.ndarray:
    x0, y0, x1, y1 = [int(round(value)) for value in bbox]
    x0 = max(0, x0 - 30)
    y0 = max(0, y0 - 30)
    x1 = min(image.shape[1], x1 + 30)
    y1 = min(image.shape[0], y1 + 30)
    crop = image[y0:y1, x0:x1].copy()
    points = {}
    for name, model_x in (("tail", -0.0778), ("grasp", 0.0), ("head", 0.09)):
        world = pose[:3, :3] @ np.array([model_x, 0.0, 0.0]) + pose[:3, 3]
        pixel = project(world, camera)
        points[name] = (pixel[0] - x0, pixel[1] - y0)
    cv2.arrowedLine(crop, points["tail"], points["head"], (0, 0, 255), 3, tipLength=0.1)
    for name, color in (("tail", (255, 255, 0)), ("grasp", (255, 255, 255)), ("head", (0, 0, 255))):
        cv2.circle(crop, points[name], 6, color, -1)
    crop = cv2.resize(crop, (290, 220), interpolation=cv2.INTER_AREA)
    canvas = np.full((250, 300, 3), 245, dtype=np.uint8)
    canvas[30:250, 5:295] = crop
    cv2.putText(canvas, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    return canvas


def main() -> None:
    summary = json.loads((OUTPUT / "summary.json").read_text())
    with (BENCHMARK / "candidates.csv").open(newline="", encoding="utf-8") as stream:
        manifest = {row["sample_id"]: row for row in csv.DictReader(stream)}
    tiles = []
    for item in summary:
        sample_id = item["sample_id"]
        row = manifest[sample_id]
        old = json.loads((ROOT / row["result_json"]).read_text())
        new = json.loads((OUTPUT / sample_id / "registration.json").read_text())
        camera = __import__("yaml").safe_load((ROOT / row["camera_config"]).read_text())
        image = cv2.imread(str(ROOT / row["rgb"]))
        bbox = old["stage1_segmentation"]["bbox_xyxy"]
        left = panel(image, np.asarray(old["t_camera_grasp"]), camera, bbox, "Previous")
        right = panel(image, np.asarray(new["t_camera_grasp"]), camera, bbox, "Candidate")
        pair = np.concatenate([left, right], axis=1)
        cv2.putText(pair, sample_id, (5, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1)
        tiles.append(pair)
    sheet = np.concatenate(
        [np.concatenate(tiles[i:i + 4], axis=1) for i in range(0, len(tiles), 4)],
        axis=0,
    )
    path = Path("/tmp/unstable_plug_registration_contact.png")
    cv2.imwrite(str(path), sheet)
    print(path)


if __name__ == "__main__":
    main()
