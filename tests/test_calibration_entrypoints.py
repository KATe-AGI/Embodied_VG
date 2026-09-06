"""Calibration CLI isolation and single/batch argument compatibility."""
import sys
from pathlib import Path

import calibration_6d_batch as batch
import calibration_6d_single as single


def test_calibration_options_and_default_method(monkeypatch):
    pose = ['0', '0', '0', '0', '0', '0']
    command = ['single', '--rgb', 'frame_color.png', '--d2rgb', 'frame_d2rgb.npy', '--output-dir', 'out', '--robot-pose', *pose]
    monkeypatch.setattr(sys, 'argv', command)
    args = single.parse_args()
    assert args.registration_method == 'symmetric'
    assert args.clamp_center_end == [0., 0., .420]
    monkeypatch.setattr(sys, 'argv', command + ['--registration-method','calibration','--clamp-center-end','.01','.02','.42'])
    args = single.parse_args()
    assert args.registration_method == 'calibration'
    assert args.clamp_center_end == [.01,.02,.42]


def test_batch_forwards_calibration_options_without_changing_pose(monkeypatch):
    monkeypatch.setattr(sys,'argv',['batch','--input-dir','frames','--output-dir','out',
        '--robot-pose','0','0','0','0','0','0','--registration-method','calibration',
        '--clamp-center-end','.01','.02','.42'])
    args = batch.parse_args()
    frame = batch.frame_args(args, Path('frames/a_color.png'), Path('frames/a_d2rgb.npy'))
    assert frame.registration_method == 'calibration'
    assert frame.clamp_center_end == [.01,.02,.42]
    assert frame.robot_pose == args.robot_pose
    assert frame.rgb == Path('frames/a_color.png')
    assert 'input_dir' not in vars(frame)


def test_clamp_projection_is_independent_of_cad_semantic_points(tmp_path):
    import cv2
    import numpy as np
    image = np.zeros((240, 320, 3), np.uint8)
    mask = np.zeros(image.shape[:2], np.uint8)
    camera = dict(fx=100., fy=100., cx=160., cy=120.)
    path = tmp_path/'projection.png'
    single._save_rgb_projection(image, mask, {}, camera, path, np.array([0.,0.,.4]))
    actual = cv2.imread(str(path))
    np.testing.assert_array_equal(actual[120,160], [0,128,255])
    single._save_rgb_projection(image, mask, {}, camera, path)
    assert not cv2.imread(str(path)).any()


def test_clamp_markers_in_ply_and_failed_registration_html(tmp_path):
    import json
    import re
    import numpy as np
    from plug_vg.registration_review import write_registration_comparison_ply, write_interactive_review_html
    center = np.array([.03,-.07,.36])
    observed = np.array([[0.,0.,.4]])
    empty = np.empty((0,3))
    ply = tmp_path/'review.ply'
    write_registration_comparison_ply(ply, observed, empty, None, clamp_center_camera_m=center)
    text = ply.read_text(); values = np.loadtxt(text.split('end_header\n')[1].splitlines())
    np.testing.assert_allclose(values[0,:3], observed[0])
    marker = values[np.all(values[:,3:]==[255,128,0],axis=1),:3]
    assert len(marker)>1
    assert np.min(np.linalg.norm(marker-center,axis=1))<1e-9
    np.testing.assert_allclose(marker.mean(axis=0), center, atol=1e-9)
    html = tmp_path/'review.html'
    write_interactive_review_html(html, observed, None, observed, empty, None, None,
                                  'failed','no correspondences',{},7,clamp_center_camera_m=center)
    data = json.loads(re.search(r'const data = (.*);', html.read_text()).group(1))
    assert data['clamp_center_camera_m']==center.tolist()
    assert 'Clamp center (flange TCP)' in html.read_text()
    write_registration_comparison_ply(ply, observed, empty, None)
    assert 'element vertex 1\n' in ply.read_text()
