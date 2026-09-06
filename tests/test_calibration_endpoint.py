"""Stage-1 geometric and observation tests, independent of YOLO/GPU."""
from types import SimpleNamespace

import numpy as np
import pytest

from plug_vg.calibration_registration import (
    calibration_observation, calibration_priors, endpoint_on_axis,
    initial_endpoint_candidates, filter_calibration_clouds, EndpointCADRegistration,
)
from plug_vg.robot_transform import robot_pose_to_matrix


def test_clamp_transform_and_base_up_chain():
    t_ec = robot_pose_to_matrix([.03, -.05, .05, .1, -.2, 3.1])
    t_be = robot_pose_to_matrix([-.014, .461, .743, 2.167, .0445, -3.127])
    center, up = calibration_priors(t_ec, t_be)
    np.testing.assert_allclose(t_ec @ np.r_[center, 1.], [0., 0., .42, 1.], atol=1e-12)
    np.testing.assert_allclose((t_be @ t_ec)[:3, :3] @ up, [0., 0., 1.], atol=1e-12)
    np.testing.assert_allclose(t_be @ t_ec @ np.r_[center, 1.], t_be @ [0., 0., .42, 1.], atol=1e-12)


@pytest.mark.parametrize("label,expected", [("plug_tail", -.0778), ("plug_head", .09)])
def test_endpoint_with_tilt_side_offset_holes_and_sparse_outliers(label, expected):
    axis = np.array([.2, -.9, -.3]); axis /= np.linalg.norm(axis)
    side = np.cross(axis, [0., 0., 1.]); side /= np.linalg.norm(side)
    center = np.array([.03, -.07, .36])
    axial = np.linspace(expected, -.01, 1001) if label == "plug_tail" else np.linspace(.01, expected, 1001)
    points = center + axial[:, None]*axis + .025*side
    points = np.delete(points, np.arange(300, 500), axis=0)  # depth hole away from the physical end
    points = np.vstack([points, center - .8*axis, center + .8*axis])
    result = endpoint_on_axis(points, center, axis, label)
    assert abs(result["endpoint_offset_m"] - expected) < .002
    assert result["support_mask"].sum() > 10
    assert np.linalg.norm(np.cross(result["endpoint_camera_m"]-center, axis)) < 1e-12


def test_axis_seed_sign_and_boundary_class():
    center = np.array([0., 0., .4])
    points = center + np.column_stack([np.full(201, .02), np.linspace(.01, .08, 201), np.zeros(201)])
    candidates = initial_endpoint_candidates(points, points, center, [0., -1., 0.], "plug_tail")
    assert len(candidates) == 2
    for candidate in candidates:
        assert candidate["axis_camera"][1] < 0
        assert candidate["endpoint_offset_m"] < 0


def test_actual_class_and_uneroded_depth_are_preserved():
    image = np.zeros((80, 80, 3), np.uint8)
    depth = np.full((80, 80), 400, dtype=float)
    depth[35:40, 35:40] = 0
    depth[40, 40] = np.nan
    camera = dict(image_height=80, image_width=80, fx=100., fy=100., cx=40., cy=40., depth_scale=.001)
    polygon = np.array([[5, 5], [75, 5], [75, 75], [5, 75]])
    prediction = SimpleNamespace(masks=SimpleNamespace(xy=[polygon, polygon]),
                                 boxes=Boxes(), names={0: "plug_head", 1: "plug_tail"})
    result = calibration_observation(image, depth, camera, prediction)
    assert result["class_name"] == "plug_tail"
    assert result["class_id"] == 1
    assert result["raw"].mask.sum() > result["core"].mask.sum()
    assert len(result["raw"].visible_points_camera_m) > len(result["core"].visible_points_camera_m)
    assert np.isfinite(result["raw"].visible_points_camera_m).all()
    with pytest.raises(ValueError, match="no_valid_depth_points"):
        calibration_observation(image, np.zeros_like(depth), camera, prediction)


class Boxes:
    conf = np.array([.4, .9])
    cls = np.array([0, 1])

    def __len__(self):
        return 2


def test_invalid_endpoint_inputs_fail_explicitly():
    with pytest.raises(ValueError):
        endpoint_on_axis(np.empty((0, 3)), np.zeros(3), [1., 0., 0.], "plug_tail")
    with pytest.raises(ValueError):
        endpoint_on_axis(np.ones((3, 3)), np.zeros(3), [0., 0., 0.], "plug_head")
    with pytest.raises(ValueError):
        endpoint_on_axis(np.ones((3, 3)), np.zeros(3), [1., 0., 0.], "visible_plug")


def test_spatial_preprocessing_removes_distant_cloud_but_keeps_boundary():
    x, y = np.meshgrid(np.arange(20)*.004, np.arange(10)*.004)
    core = np.column_stack([x.ravel(), y.ravel(), np.full(x.size, .4)])
    distant = core[:30] + [0., 0., .3]
    raw_boundary = core[:20] + [0., -.003, 0.]
    all_core, raw = np.vstack([core, distant]), np.vstack([core, raw_boundary, distant])
    a, b = filter_calibration_clouds(all_core, raw)
    assert a[:len(core)].all() and not a[len(core):].any()
    assert b[:len(core)+len(raw_boundary)].all()
    assert not b[len(core)+len(raw_boundary):].any()
    with pytest.raises(ValueError, match="no_connected"):
        filter_calibration_clouds(np.eye(3), np.eye(3))


@pytest.mark.parametrize("label", ["plug_tail", "plug_head"])
@pytest.mark.parametrize("roll", [0., .9])
def test_coarse_recovers_tilt_and_slip_from_partial_side(label, roll):
    import open3d as o3d
    from scipy.spatial.transform import Rotation

    mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=.02, height=.1678, resolution=60, split=10)
    mesh.rotate(Rotation.from_rotvec([0., np.pi/2, 0.]).as_matrix(), center=(0, 0, 0))
    mesh.translate([.0061, 0., 0.])
    solver = EndpointCADRegistration(mesh)
    axis = np.array([.1, -.98, .15]); axis /= np.linalg.norm(axis)
    y = np.cross([0., 0., 1.], axis); y /= np.linalg.norm(y)
    rotation = np.column_stack([axis, y, np.cross(axis, y)]) @ Rotation.from_rotvec([roll, 0., 0.]).as_matrix()
    center = np.array([.03, -.07, .36])
    translation = center + .012*axis  # known axial slip
    axial = np.linspace(-.0778, -.025, 70) if label == "plug_tail" else np.linspace(.04, .09, 70)
    x, theta = np.meshgrid(axial, np.linspace(-.8, .8, 30))
    points = np.column_stack([x.ravel(), .02*np.cos(theta.ravel()), .02*np.sin(theta.ravel())]) @ rotation.T + translation
    points = np.delete(points, np.arange(500, 650), axis=0)
    points += np.random.default_rng(1).normal(0., .00015, points.shape)
    result = solver.fit_coarse(points[::4], points, center, axis, label)
    transform = result['t_camera_grasp']
    assert np.rad2deg(np.arccos(np.clip(transform[:3, 0] @ axis, -1, 1))) < 3.
    assert np.linalg.norm(transform[:3, 3]-translation) < .003
    assert np.linalg.norm(np.cross(transform[:3, 3]-center, transform[:3, 0])) < 1e-10
    assert result['surface_median_m'] < .001
    assert len(result['candidates']) == 8
    assert all(c['evaluations'] <= 30 for c in result['candidates'])
    assert result['cost'] == min(c['cost'] for c in result['candidates'])


def test_icp_plane_correction_preserves_anchors_and_reports_missing_matches():
    import open3d as o3d

    mesh = o3d.geometry.TriangleMesh.create_box(.1678, .04, .04)
    mesh.translate([-.0778, -.02, -.02])
    solver = EndpointCADRegistration(mesh)
    x,y = np.meshgrid(np.linspace(-.075, .08, 30), np.linspace(-.018, .018, 15))
    points = np.column_stack([x.ravel(), y.ravel(), np.full(x.size, .02)])
    initial = np.eye(4); initial[2,3] = .0015
    before = np.median(solver.surface_distance(points-initial[:3,3]))
    result = solver.refine_icp(points, initial, np.zeros(3), 'plug_tail')
    assert result['status'] == 'awaiting_visual_review'
    assert result['surface_median_m'] < before
    assert result['endpoint_displacement_m'] < .001
    assert result['clamp_axis_distance_m'] < .002
    assert len(result['history']) <= 20
    assert all(h['cost_after'] <= h['cost_before'] for h in result['history'])
    t = result['t_camera_grasp']
    np.testing.assert_allclose(t[:3,:3].T @ t[:3,:3], np.eye(3), atol=1e-10)
    # Removing reliable correspondences cannot turn distant observations into a success.
    failed = solver.refine_icp(points+[0,0,1], initial, np.zeros(3), 'plug_tail')
    assert failed['status'] == 'failed'
    assert failed['reason'] == 'insufficient_icp_correspondences'
    assert 't_camera_grasp' not in failed
    assert solver.refine_icp(points[:3], initial, np.zeros(3), 'plug_tail')['status'] == 'failed'
