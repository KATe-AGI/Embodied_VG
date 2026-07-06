# 2175B Grasp Asset Review

## Files
- STEP source: `plug_model/2175B.stp`
- Grasp OBJ: `plug_model/2175B_grasp.obj`
- Grasp PLY: `plug_model/2175B_grasp.ply`
- Config: `configs/plug_models/2175B.yaml`

## Coordinate System
- Frame: `grasp`
- Origin: estimated grasp grasp_origin center
- `+X`: tail to head, mapped from raw CAD `+Z`
- `+Y`: gripper closing direction, mapped from projected raw CAD `+Y`
- `+Z`: approach direction, right-handed `X x Y`

## Semantic Points
- `grasp_center_m`: `[0.0, 0.0, 0.0]`
- `tail_center_m`: `[-0.08401198, 0.0, 0.0]`
- `head_center_m`: `[0.09628802, 0.0, 0.0]`
- `head_tail_axis_length_m`: `0.18030000`

## Raw CAD Diagnostics
- Raw bbox min mm: `[-47.000021, -47.0, -180.3]`
- Raw bbox max mm: `[47.000021, 47.0, 0.0]`
- Raw bbox extent mm: `[94.000042, 94.0, 180.3]`
- Grasp center raw mm: `[0.0, 0.0, -96.288021]`
- Grasp-origin confidence: `medium`
- Grasp-origin reason: `selected_stable_cylindrical_region_closest_to_axis_midpoint`
- Selected region: `{'bin_start': 59, 'bin_end': 88, 'z_min_mm': -113.814375, 'z_max_mm': -80.008125, 'z_center_mm': -96.288021, 'mean_bbox_area_mm2': 3107.946406, 'mean_circularity_xy': 0.942791, 'point_count': 2582}`

## Manual Review Items
- Confirm raw CAD `+Z` is the intended `tail -> head` direction.
- Confirm the selected grasp-origin region is the physical graspable middle section.
- Confirm projected raw CAD `+Y` is acceptable as the gripper closing direction.
