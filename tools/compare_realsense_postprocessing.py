"""Compare official RealSense filters on saved aligned Z16 depth, without a camera.

Filtering needs pyrealsense2, NumPy and OpenCV. Plotting additionally needs
Matplotlib; --skip-plots / --render-only allow using separate environments.
This tool does not import or change the project's inference pipeline.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import cv2
import numpy as np


VARIANTS = (
    "raw", "spatial", "spatial_hole_fill", "decimation2",
    "decimation2_spatial_hole_fill",
)
LABELS = {
    "raw": "Raw aligned depth",
    "spatial": "Spatial (no hole filling)",
    "spatial_hole_fill": "Spatial + hole fill (farthest)",
    "decimation2": "Decimation x2",
    "decimation2_spatial_hole_fill": "Decimation x2 + spatial + hole fill",
}


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def profile_info(frame) -> dict:
    intr = frame.profile.as_video_stream_profile().intrinsics
    return {
        "width": intr.width, "height": intr.height,
        "fx": intr.fx, "fy": intr.fy, "cx": intr.ppx, "cy": intr.ppy,
        "distortion": str(intr.model), "coeffs": list(intr.coeffs),
        "depth_scale_m": frame.as_depth_frame().get_units(),
    }


def software_depth_frame(rs, depth: np.ndarray, meta: dict):
    """Inject actual uint16 samples with recorded intrinsics and depth units."""
    streams = meta["streams"]
    calibration = streams.get("depth_aligned_to_color_raw", streams["depth_aligned_to_color"])
    values = calibration["intrinsics"]
    intr = rs.intrinsics()
    intr.width, intr.height = values["width"], values["height"]
    intr.fx, intr.fy = values["fx"], values["fy"]
    intr.ppx, intr.ppy = values["ppx"], values["ppy"]
    if depth.shape != (intr.height, intr.width) or depth.dtype != np.uint16:
        raise ValueError("Expected uint16 Z16 depth matching the recorded intrinsics")
    distortion = values["model"].split(".")[-1]
    intr.model = getattr(rs.distortion, distortion)
    intr.coeffs = values["coeffs"]
    device = rs.software_device()
    sensor = device.add_sensor("Offline aligned depth")
    sensor.add_read_only_option(rs.option.depth_units, float(meta["depth_scale"]))
    stream = rs.video_stream()
    stream.type, stream.fmt = rs.stream.depth, rs.format.z16
    stream.index, stream.uid, stream.bpp = 0, 0, 2
    stream.width, stream.height = intr.width, intr.height
    stream.fps, stream.intrinsics = calibration["fps"], intr
    profile = sensor.add_video_stream(stream)
    queue = rs.frame_queue(2)
    sensor.open(profile)
    sensor.start(queue)
    video = rs.software_video_frame()
    video.pixels, video.stride, video.bpp = depth, depth.strides[0], 2
    video.timestamp, video.frame_number = 0.0, 1
    video.domain = rs.timestamp_domain.hardware_clock
    video.profile = profile.as_video_stream_profile()
    video.depth_units = float(meta["depth_scale"])
    sensor.on_video_frame(video)
    frame = queue.wait_for_frame(5000).as_depth_frame()
    if not np.array_equal(depth, np.asanyarray(frame.get_data())):
        raise RuntimeError("Software sensor changed the injected raw depth")
    return frame, (device, sensor, queue, video, depth)


def spatial_filter(rs):
    filt = rs.spatial_filter()
    for option, value in (
        (rs.option.filter_magnitude, 2),
        (rs.option.filter_smooth_alpha, 0.5),
        (rs.option.filter_smooth_delta, 20),
        (rs.option.holes_fill, 0),
    ):
        filt.set_option(option, value)
    return filt


def hole_filter(rs):
    filt = rs.hole_filling_filter()
    filt.set_option(rs.option.holes_fill, 1)  # farthest_from_around
    return filt


def fixed_smoothness_mask(raw: np.ndarray) -> np.ndarray:
    """Fixed 3x3 support: raw pixels all valid, raw local span <= 20 increments."""
    valid = cv2.erode((raw > 0).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    raw_float = raw.astype(np.float32)
    span = cv2.dilate(raw_float, None) - cv2.erode(raw_float, None)
    valid &= span <= 20
    valid[[0, -1], :] = False
    valid[:, [0, -1]] = False
    return valid


def metrics(array: np.ndarray, baseline: np.ndarray, scale: float, roi: list[int]) -> dict:
    valid = array > 0
    x0, y0, x1, y1 = roi
    result = {
        "shape_hw": list(array.shape), "valid_percent": float(100 * valid.mean()),
        "valid_pixels": int(valid.sum()), "invalid_pixels": int((~valid).sum()),
    }
    # Native resolution outputs only: never resample decimation for statistics.
    if array.shape == baseline.shape:
        raw_valid = baseline > 0
        common = raw_valid & valid
        signed_difference = (array.astype(float) - baseline.astype(float))[common] * scale * 1000
        difference = np.abs(signed_difference)
        newly_filled = ~raw_valid & valid
        filled_depths = array[newly_filled].astype(float) * scale
        filled_roi = newly_filled[y0:y1, x0:x1]
        filled_roi_depths = array[y0:y1, x0:x1][filled_roi].astype(float) * scale
        support = fixed_smoothness_mask(baseline)
        residual = (array.astype(float) - cv2.medianBlur(array, 3).astype(float))[support] * scale * 1000
        result.update(
            roi_valid_percent=float(100 * valid[y0:y1, x0:x1].mean()),
            newly_filled_pixels=int(newly_filled.sum()),
            newly_invalid_pixels=int((raw_valid & ~valid).sum()),
            changed_original_valid_pixels=int((common & (array != baseline)).sum()),
            original_valid_abs_change_median_mm=float(np.median(difference)) if difference.size else None,
            original_valid_abs_change_p95_mm=float(np.percentile(difference, 95)) if difference.size else None,
            original_valid_signed_change_median_mm=float(np.median(signed_difference)) if difference.size else None,
            original_valid_signed_change_mean_mm=float(np.mean(signed_difference)) if difference.size else None,
            newly_filled_depth_p95_m=float(np.percentile(filled_depths, 95)) if filled_depths.size else None,
            newly_filled_roi_depth_p95_m=float(np.percentile(filled_roi_depths, 95)) if filled_roi_depths.size else None,
            fixed_support_median_residual_rms_mm=float(np.sqrt(np.mean(residual**2))) if residual.size else None,
            fixed_support_pixels=int(support.sum()),
        )
    return result


def process_one(rs, meta_path: Path, root: Path, roi: list[int] | None) -> dict:
    meta = json.loads(meta_path.read_text(encoding="utf-8-sig"))
    sample = meta["sample_id"]
    folder = root / sample
    folder.mkdir(parents=True, exist_ok=True)
    source = meta_path.parent / meta["files"]["d2rgb"]
    rgb = meta_path.parent / meta["files"]["color"]
    original_hash = file_hash(source)
    raw = np.ascontiguousarray(np.load(source, allow_pickle=False))
    height, width = raw.shape
    roi = roi or [width // 3, height // 4, width * 2 // 3, height * 3 // 4]
    if not (0 <= roi[0] < roi[2] <= width and 0 <= roi[1] < roi[3] <= height):
        raise ValueError(f"Invalid crop for {sample}: {roi}")
    frame, resources = software_depth_frame(rs, raw, meta)
    try:
        stages = {"raw": frame}
        stage_start = time.perf_counter()
        stages["spatial"] = spatial_filter(rs).process(frame)
        stages["spatial_hole_fill"] = hole_filter(rs).process(stages["spatial"])
        decimation = rs.decimation_filter()
        decimation.set_option(rs.option.filter_magnitude, 2)
        stages["decimation2"] = decimation.process(frame)
        reduced_spatial = spatial_filter(rs).process(stages["decimation2"])
        stages["decimation2_spatial_hole_fill"] = hole_filter(rs).process(reduced_spatial)
        # Fresh state and exactly one injection: these unrelated photos are not a video.
        temporal = rs.temporal_filter()
        first_temporal = temporal.process(stages["spatial"])
        temporal_array = np.asanyarray(first_temporal.get_data())
        temporal_changes = int(np.count_nonzero(temporal_array != np.asanyarray(stages["spatial"].get_data())))
        processing_s = time.perf_counter() - stage_start
        outputs = {}
        scale = float(meta["depth_scale"])
        for name, output in stages.items():
            array = np.asanyarray(output.get_data()).copy()
            target = folder / f"{name}.npy"
            np.save(target, array, allow_pickle=False)
            outputs[name] = {
                "file": str(target.resolve()), "intrinsics": profile_info(output),
                "metrics": metrics(array, raw, scale, roi),
            }
    finally:
        resources[1].stop()
        resources[1].close()
    if file_hash(source) != original_hash:
        raise RuntimeError("Source depth hash changed")
    result = {
        "sample_id": sample, "meta_file": str(meta_path.resolve()),
        "rgb_file": str(rgb.resolve()), "depth_file": str(source.resolve()),
        "source_sha256": original_hash, "source_unchanged": True,
        "depth_scale_m": scale, "roi_xyxy": roi, "roi_is_object_mask": False,
        "sdk_filter_time_s": processing_s,
        "temporal_first_frame_changed_pixels": temporal_changes,
        "outputs": outputs,
    }
    write_json(folder / "result.json", result)
    return result


def run_filters(args) -> dict:
    import pyrealsense2 as rs

    paths = sorted(args.input_dir.glob("*_meta.json"))
    if not paths:
        raise ValueError(f"No *_meta.json in {args.input_dir}")
    rois = json.loads(args.roi_config.read_text()) if args.roi_config else {}
    summary = {
        "sdk_version": importlib.metadata.version("pyrealsense2"),
        "domain": "already_color_aligned_Z16_depth",
        "disparity_conversion_used": False,
        "disparity_skip_reason": "No recorded stereo baseline or native depth calibration",
        "temporal_history_available": False,
        "coverage_definition": "valid/nonzero means raw Z16 >0 only; it does not establish correct depth or enforce a distance range",
        "parameters": {
            "spatial": {"iterations": 2, "alpha": 0.5, "delta_raw_depth_increments": 20, "holes_fill": 0},
            "decimation": {"magnitude": 2},
            "hole_filling": {"mode": 1, "name": "farthest_from_around"},
            "temporal_cold_start": {"alpha": 0.4, "delta": 20, "persistence": 3, "frame_count": 1},
        },
        "smoothness_metric": "RMS residual to 3x3 median on a fixed raw-derived support: all 3x3 pixels >0 and local span<=20 raw units; a smoothness proxy, NOT accuracy or sensor noise ground truth",
        "samples": [],
    }
    for path in paths:
        sample_id = path.name.removesuffix("_meta.json")
        result = process_one(rs, path, args.output_dir, rois.get(sample_id))
        summary["samples"].append(result)
        print(f"Filtered {sample_id}: " + ", ".join(f"{key}={value['metrics']['valid_percent']:.2f}%" for key, value in result["outputs"].items()))
    write_json(args.output_dir / "summary.json", summary)
    rows = []
    for sample in summary["samples"]:
        for variant, data in sample["outputs"].items():
            rows.append({"sample_id": sample["sample_id"], "variant": variant, **data["metrics"]})
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (args.output_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return summary


def render(summary: dict, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Rectangle

    depth_map = plt.get_cmap("turbo").copy()
    depth_map.set_bad("#161b22")
    difference_map = plt.get_cmap("coolwarm").copy()
    difference_map.set_bad("#161b22")
    full_limits = (0.25, 2.5)  # Identical metric scale in every full-frame panel.
    overview, overview_axes = plt.subplots(len(summary["samples"]), 4, figsize=(18, 3.0 * len(summary["samples"])), constrained_layout=True)
    for row, sample in enumerate(summary["samples"]):
        folder = output / sample["sample_id"]
        arrays = {name: np.load(item["file"]) for name, item in sample["outputs"].items()}
        rgb = cv2.cvtColor(cv2.imread(sample["rgb_file"]), cv2.COLOR_BGR2RGB)
        raw = arrays["raw"]
        scale = sample["depth_scale_m"]
        height, width = raw.shape
        x0, y0, x1, y1 = sample["roi_xyxy"]
        crop_valid = raw[y0:y1, x0:x1]
        crop_valid = crop_valid[crop_valid > 0] * scale
        crop_limits = tuple(np.percentile(crop_valid, [2, 98])) if crop_valid.size else full_limits

        def depth_panel(axis, name, crop=False):
            array = arrays[name]
            if crop:
                # Only display is resized. Saved data, intrinsics and metrics stay native.
                display = cv2.resize(array, (width, height), interpolation=cv2.INTER_NEAREST)
                display = display[y0:y1, x0:x1]
            else:
                display = array
            image = axis.imshow(np.ma.masked_where(display == 0, display * scale), cmap=depth_map,
                                vmin=(crop_limits if crop else full_limits)[0],
                                vmax=(crop_limits if crop else full_limits)[1], interpolation="nearest")
            valid = sample["outputs"][name]["metrics"]["valid_percent"]
            axis.set_title(f"{LABELS[name]}\n{array.shape[1]} x {array.shape[0]} | nonzero {valid:.2f}%", fontsize=10)
            axis.set_axis_off()
            return image

        for crop, filename in ((False, "comparison_full.png"), (True, "comparison_roi.png")):
            fig, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
            axes[0, 0].imshow(rgb[y0:y1, x0:x1] if crop else rgb)
            axes[0, 0].set_title("RGB reference / manual context ROI" if crop else "RGB reference / yellow: context ROI")
            axes[0, 0].set_axis_off()
            if not crop:
                axes[0, 0].add_patch(Rectangle((x0, y0), x1-x0, y1-y0, fill=False, edgecolor="yellow", linewidth=2))
            for axis, name in zip(axes.flat[1:], VARIANTS):
                depth_image = depth_panel(axis, name, crop)
            fig.colorbar(depth_image, ax=list(axes.flat[1:]), shrink=0.75, extend="both", label="Depth Z (m); black = invalid; extremes clipped")
            qualifier = "ROI: same color scale; x2 data enlarged for display only" if crop else "Same depth scale across all samples; invalid depth is black"
            fig.suptitle(f"{sample['sample_id']} | RealSense SDK {summary['sdk_version']}\n{qualifier}", fontsize=13)
            fig.savefig(folder / filename, dpi=160)
            plt.close(fig)

        spatial = arrays["spatial"]
        filled = arrays["spatial_hole_fill"]
        common = (raw > 0) & (spatial > 0)
        difference = (spatial.astype(float) - raw.astype(float)) * scale * 1000
        added = (raw == 0) & (filled > 0)
        categories = np.zeros_like(raw, dtype=np.uint8)
        categories[raw > 0] = 1
        categories[added] = 2
        fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
        axes[0, 0].imshow(rgb[y0:y1, x0:x1])
        profile_y = (y0+y1)//2
        axes[0, 0].axhline(profile_y-y0, color="yellow", linewidth=1)
        axes[0, 0].set_title(f"RGB context ROI; profile at original y={profile_y}")
        diff_image = axes[0, 1].imshow(np.ma.masked_where(~common[y0:y1, x0:x1], difference[y0:y1, x0:x1]),
                                      cmap=difference_map, vmin=-10, vmax=10, interpolation="nearest")
        axes[0, 1].set_title("Spatial - raw on originally valid pixels (clipped +/-10 mm)")
        fig.colorbar(diff_image, ax=axes[0, 1], label="Depth change (mm)", shrink=0.8)
        axes[1, 0].imshow(categories[y0:y1, x0:x1], cmap=ListedColormap(["#161b22", "#9ca3af", "#ef9f27"]), vmin=0, vmax=2, interpolation="nearest")
        axes[1, 0].set_title("Gray: raw valid | Orange: filled | Black: still invalid")
        for axis in [axes[0, 0], axes[0, 1], axes[1, 0]]:
            axis.set_axis_off()
        for name, color in [("raw", "#64748b"), ("spatial", "#1670b8"), ("spatial_hole_fill", "#df861d")]:
            z = arrays[name][profile_y, x0:x1].astype(float) * scale
            z[z == 0] = np.nan
            axes[1, 1].plot(np.arange(x0, x1), z, label=LABELS[name], color=color, linewidth=1, alpha=0.9)
        axes[1, 1].set(xlabel="Original RGB pixel x", ylabel="Depth Z (m)", title="Single-row depth profile; gaps = invalid")
        axes[1, 1].legend(fontsize=8)
        axes[1, 1].grid(alpha=0.2)
        fig.suptitle(f"{sample['sample_id']} | Smoothness and filled pixels do not establish accuracy", fontsize=13)
        fig.savefig(folder / "changes_roi.png", dpi=160)
        plt.close(fig)
        overview_axes[row, 0].imshow(rgb)
        overview_axes[row, 0].set_title(sample["sample_id"], fontsize=10)
        overview_axes[row, 0].set_axis_off()
        for col, name in enumerate(("raw", "spatial", "spatial_hole_fill"), 1):
            overview_depth = depth_panel(overview_axes[row, col], name)
    overview.colorbar(overview_depth, ax=overview_axes, shrink=0.65, extend="both", label="Depth Z (m); black = invalid; extremes clipped")
    overview.suptitle("Official RealSense SDK depth-domain filters | five independent aligned frames", fontsize=15)
    overview.savefig(output / "overview_all.png", dpi=150)
    plt.close(overview)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--roi-config", type=Path, help="Optional sample_id -> [x0,y0,x1,y1] context crops")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--skip-plots", action="store_true")
    mode.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.render_only:
        summary = json.loads((args.output_dir / "summary.json").read_text())
    else:
        if args.input_dir is None:
            parser.error("--input-dir is required unless --render-only")
        summary = run_filters(args)
    if not args.skip_plots:
        render(summary, args.output_dir)


if __name__ == "__main__":
    main()
