# AprilRobotics pose subset

Source: [AprilRobotics/apriltag](https://github.com/AprilRobotics/apriltag/tree/b7c0ebe9aa20f82ec7a828579004f9e706bfecd9), commit recorded in `UPSTREAM_COMMIT`.

Only `apriltag_pose.c/.h`, `apriltag.h`, the math sources `common/homography.c`, `common/matd.c`, `common/svd22.c`, their transitive headers, and upstream `LICENSE.md` were downloaded. These 19 upstream files are unchanged. No detector implementation, examples, tests, other tag families, or repository checkout was imported.

`pose_bridge.c` is the project's small C interface. It copies the detector's original homography and four ordered corners into an `apriltag_detection_t`, calls the official `estimate_tag_pose()` unchanged, copies its R/t/object-space error, and frees the official matrices. The estimator performs homography initialization, 50 orthogonal-iteration steps and planar ambiguity comparison.

`plug_vg/apriltag_pose.py` compiles this subset into a shared library on first use using `cc` (or `CC`), and caches it under the current user's temporary directory using a source-content hash. On Ubuntu/Linux this requires a C compiler, e.g. `build-essential`. The vendored algorithm runs offline; inference performs no source downloads. Shared binaries are generated outside the repository.

Detection uses the existing `pupil-apriltags` dependency with `estimate_tag_pose=False`; its bundled estimator is not called. The returned homography and corner order are passed directly to the vendored official estimator.

For `tagStandard41h12`, upstream [`tagStandard41h12.c`](https://github.com/AprilRobotics/apriltag/blob/b7c0ebe9aa20f82ec7a828579004f9e706bfecd9/tagStandard41h12.c) defines `width_at_border=5`, `total_width=9`. This project's public `--tagsize` is the measured blue 5x5 pose square side in meters, passed directly to the official estimator. It defaults to 0.018 m (18 mm), corresponding to a derived full 9x9 pattern side of 32.4 mm. The metric translation depends directly on this physical size.

The upstream BSD 2-Clause license is retained in `LICENSE.md` and in the source headers where present.
