# Hand-Eye Calibration

This module computes hand-eye calibration from manually collected checkerboard images and robot poses.
It supports both eye-in-hand and eye-to-hand calibration.

## Data Layout

Put each calibration batch under `eye_hand_data/<batch_name>`:

```text
eye_hand_data/
  calib_20260522/
    1.jpg
    2.png
    3.jpg
    ...
    poses.txt
```

Images must use numeric filenames. Image `N.jpg`, `N.jpeg`, or `N.png` is matched with line `N` in `poses.txt`.
The same numeric index must not appear with multiple image suffixes.

`poses.txt` uses one robot end-effector pose per line:

```text
x,y,z,rx,ry,rz
```

Conventions:

- `x,y,z` are in meters.
- `rx,ry,rz` are in radians.
- The pose is `T_base_end`, the end-effector pose in the robot base frame.
- Euler conversion follows the original implementation: `R = Rz @ Ry @ Rx`.

Checkerboard settings are read from `config.yaml`:

```yaml
checkerboard_args:
  XX: 11
  YY: 8
  L: 0.03
```

`XX` and `YY` are inner-corner counts. `L` is square size in meters.

## Modes

### Eye-In-Hand

Use this mode when the camera is mounted on the robot end-effector and the checkerboard is fixed in the workspace.
The output transform is:

```text
T_end_camera
```

Run:

```bash
python compute_hand_eye.py --mode in-hand --data-dir eye_hand_data/calib_20260522
```

### Eye-To-Hand

Use this mode when the camera is fixed outside the robot and the checkerboard is mounted on the end-effector.
The output transform is:

```text
T_base_camera
```

Run:

```bash
python compute_hand_eye.py --mode to-hand --data-dir eye_hand_data/<batch>
```

## CLI

```bash
python compute_hand_eye.py \
  --mode in-hand \
  --data-dir eye_hand_data/calib_20260522 \
  --method TSAI
```

Arguments:

- `--mode`: `in-hand` or `to-hand`.
- `--data-dir`: calibration batch directory. If omitted, the script uses the only valid batch under `eye_hand_data`; if multiple batches exist, it asks you to specify one.
- `--output-dir`: output directory. Defaults to the selected data batch directory.
- `--method`: OpenCV hand-eye method, one of `TSAI`, `PARK`, `HORAUD`, `ANDREFF`, `DANIILIDIS`. Default is `TSAI`.
- `--config`: checkerboard config YAML. Defaults to `config.yaml`.

## Outputs

The script writes both YAML and JSON:

```text
hand_eye_result_<mode>.yaml
hand_eye_result_<mode>.json
```

Each result contains:

- `mode`
- `method`
- `transform_name`
- `matrix_4x4`
- `rotation_matrix`
- `translation_m`
- `quaternion_xyzw`
- `euler_xyz_deg`
- `camera_calibration_rms`
- per-image reprojection error
- used, failed, and ignored images
- checkerboard and pose conventions
