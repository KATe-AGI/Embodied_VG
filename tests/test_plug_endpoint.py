from types import SimpleNamespace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from plug_vg.placement import load_placement, preset_points
from plug_vg.plug_endpoint import choose_end_instance, fit_local_depth, locate_edge_point, projected_base_axis


class EndpointTests(unittest.TestCase):
    def setUp(self):
        self.config = load_placement()
        self.cfg = self.config['endpoint']
        self.camera = dict(fx=800., fy=800., cx=100., cy=100., depth_scale=.001)
        self.mask = np.zeros((200, 200), np.uint8)
        self.mask[30:171, 75:126] = 1

    def test_offsets_follow_normal_and_base_z_separately(self):
        n = np.array([.6, .8, 0.])
        p = np.array([.1, .2, .3])
        points = preset_points(p, n, self.config)
        np.testing.assert_allclose(points['preset'], p+.22*n)
        np.testing.assert_allclose(points['head_preset']-points['preset'], [0,0,.082])
        np.testing.assert_allclose(points['tail_preset']-points['preset'], [0,0,-.090])
        self.assertAlmostEqual(np.linalg.norm(points['head_preset']-points['tail_preset']), .172)

    def test_head_tail_direction_and_ring_does_not_move_midpoint(self):
        self.mask[90:110, 50:150] = 1
        head = locate_edge_point(self.mask, [0,-1], 'plug_head', self.cfg)
        tail = locate_edge_point(self.mask, [0,-1], 'plug_tail', self.cfg)
        self.assertAlmostEqual(head['edge_pixel'][0], 100, delta=1)
        self.assertLess(head['target_pixel'][1], 45)
        self.assertGreater(tail['target_pixel'][1], 155)
        for location in [head, tail]:
            u,v = location['target_pixel'].astype(int)
            self.assertTrue(location['safe_mask'][v,u])
            self.assertLessEqual(location['inset_actual_px'], self.cfg['max_target_retreat_px']+1)

    def test_slanted_plane_depth_boundary_background_and_holes(self):
        location = locate_edge_point(self.mask, [0,-1], 'plug_head', self.cfg)
        yy,xx=np.indices(self.mask.shape)
        rays=np.stack([(xx-100)/800,(yy-100)/800,np.ones_like(xx)],axis=-1)
        normal=np.array([.20,-.15,1.])
        z=.5/(rays@normal)
        depth=z*1000
        depth[self.mask==0]=1200  # Background must never participate.
        u,v=location['target_pixel'].astype(int)
        depth[v,u]=0  # Target pixel itself may lack depth.
        rng=np.random.default_rng(4)
        inside=np.argwhere(self.mask>0)
        for y,x in inside[rng.choice(len(inside),len(inside)//12,replace=False)]:depth[y,x]=1000
        fit=fit_local_depth(depth,location,self.camera,self.cfg)
        expected=rays[v,u]*z[v,u]
        np.testing.assert_allclose(fit['point_camera_m'],expected,atol=1e-6)
        pixels=fit['sample_pixels'].astype(int)
        self.assertTrue(np.all(location['safe_mask'][pixels[:,1],pixels[:,0]]))
        self.assertLess(fit['quality']['inlier_ratio'],1)
        self.assertFalse(fit['quality']['physical_accuracy_verified'])

    def test_slanted_end_rim_midpoint_is_not_extreme_corner(self):
        mask = np.zeros_like(self.mask)
        cv2.fillPoly(mask, [np.array([[75, 30], [125, 30], [125, 140], [75, 170]])], 1)
        location = locate_edge_point(mask, [0, -1], 'plug_tail', self.cfg)
        np.testing.assert_allclose(location['edge_pixel'], [100, 155], atol=3)

    def test_rotated_endpoint_lies_on_axial_centerline_and_contour(self):
        mask = np.zeros_like(self.mask)
        vertices = np.array([[60, 50], [150, 140], [125, 165], [35, 75]], dtype=np.int32)
        cv2.fillPoly(mask, [vertices], 1)

        location = locate_edge_point(mask, [1, 1], "plug_head", self.cfg)

        edge = location["edge_pixel"]
        self.assertAlmostEqual(edge.sum(), 290.0)
        self.assertEqual(cv2.pointPolygonTest(vertices, tuple(edge), False), 0)
        u, v = location["target_pixel"].astype(int)
        self.assertTrue(location["safe_mask"][v, u])
        self.assertGreater((edge - location["target_pixel"]) @ location["direction_image"], 0)

    def test_endpoint_exits_main_interior_before_later_mask_fragment(self):
        mask = self.mask.copy()
        # An open notch crosses the centerline, while the silhouette remains connected at the right.
        mask[145:160, :115] = 0

        location = locate_edge_point(mask, [0, -1], "plug_tail", self.cfg)

        self.assertAlmostEqual(location["edge_pixel"][1], 144)
        u, v = location["target_pixel"].astype(int)
        self.assertEqual(v, 141)
        self.assertTrue(location["safe_mask"][v, u])

    def test_curved_surface_uses_quadratic_holdout_fit(self):
        location=locate_edge_point(self.mask,[0,-1],'plug_head',self.cfg)
        yy,xx=np.indices(self.mask.shape);u,v=location['target_pixel']
        inverse=2 + .02*((xx-u)/14)**2 + .005*((yy-v)/14)**2
        depth=1000/inverse
        fit=fit_local_depth(depth,location,self.camera,self.cfg)
        self.assertEqual(fit['quality']['model'],'inverse_depth_quadratic')
        self.assertAlmostEqual(fit['point_camera_m'][2],.5,places=7)

    def test_missing_depth_rejected_and_one_sided_support_uses_median(self):
        location=locate_edge_point(self.mask,[0,-1],'plug_head',self.cfg)
        with self.assertRaisesRegex(ValueError,'Insufficient'):
            fit_local_depth(np.zeros(self.mask.shape),location,self.camera,self.cfg)
        depth=np.full(self.mask.shape,500.)
        _,v=location['target_pixel'].astype(int)
        depth[:v+1,:]=0
        fit = fit_local_depth(depth,location,self.camera,self.cfg)
        self.assertEqual(fit['quality']['model'], 'local_median')
        self.assertAlmostEqual(fit['point_camera_m'][2], .5)

    def test_median_fallback_without_any_surface_fit(self):
        location = locate_edge_point(self.mask, [0, -1], 'plug_head', self.cfg)
        depth = np.full(self.mask.shape, 500, dtype=np.uint16)
        with patch('plug_vg.plug_endpoint._robust_model', side_effect=ValueError('no fit')):
            fit = fit_local_depth(depth, location, self.camera, self.cfg)
        self.assertEqual(fit['quality']['model'], 'local_median')
        self.assertIsNone(fit['quality']['coefficients'])
        self.assertEqual(fit['quality']['inlier_samples'], fit['quality']['valid_samples'])
        self.assertAlmostEqual(fit['point_camera_m'][2], .5)

    def test_axis_mismatch_and_clipped_mask_rejected(self):
        with self.assertRaisesRegex(ValueError,'inconsistent'):
            locate_edge_point(self.mask,[1,0],'plug_head',self.cfg)
        self.mask[:30,75:126]=1
        with self.assertRaisesRegex(ValueError,'clipped'):
            locate_edge_point(self.mask,[0,-1],'plug_head',self.cfg)

    def test_axis_projection_uses_capture_rotation(self):
        t=np.eye(4)
        t[:3,:3]=np.array([[1,0,0],[0,0,-1],[0,1,0]])
        d=projected_base_axis(self.mask,np.full(self.mask.shape,500),self.camera,t,self.cfg)
        np.testing.assert_allclose(d,[0,1],atol=1e-12)
        with self.assertRaisesRegex(ValueError,'nearly'):
            projected_base_axis(self.mask,np.full(self.mask.shape,500),self.camera,np.eye(4),self.cfg)

    def test_three_duplicate_endpoints_keep_highest_confidence(self):
        import torch

        shifted = np.roll(self.mask, 1, axis=1)
        other = np.roll(self.mask, -1, axis=1)
        prediction = SimpleNamespace(
            names={0: "plug_head", 1: "plug_tail"},
            boxes=[SimpleNamespace(cls=torch.tensor([1]), conf=torch.tensor([score]))
                   for score in (0.69, 0.80, 0.93)],
            masks=SimpleNamespace(data=torch.tensor(np.stack((self.mask, other, shifted)))),
            orig_shape=self.mask.shape,
        )

        selected = choose_end_instance(prediction)

        self.assertEqual(selected["class_name"], "plug_tail")
        self.assertAlmostEqual(selected["confidence"], 0.93)
        self.assertEqual(selected["segmentation_candidate_count"], 3)
        np.testing.assert_array_equal(selected["mask"], shifted)

    def test_iou_at_threshold_and_confidence_tie_keep_first_mask(self):
        import torch

        mask = np.zeros_like(self.mask)
        mask[35:140, 90:117] = 1
        shifted = np.roll(mask, 3, axis=1)
        masks = (mask, shifted)
        self.assertEqual(np.count_nonzero(mask & shifted) / np.count_nonzero(mask | shifted), 0.80)
        for first, second in (masks, masks[::-1]):
            prediction = SimpleNamespace(
                names={0: "plug_head"},
                boxes=[SimpleNamespace(cls=torch.tensor([0]), conf=torch.tensor([0.875])) for _ in masks],
                masks=SimpleNamespace(data=torch.tensor(np.stack((first, second)))),
                orig_shape=mask.shape,
            )

            selected = choose_end_instance(prediction)

            self.assertEqual(selected["confidence"], 0.875)
            self.assertEqual(selected["segmentation_candidate_count"], 2)
            np.testing.assert_array_equal(selected["mask"], first)

    def test_one_pair_below_iou_threshold_remains_ambiguous(self):
        import torch

        masks = (self.mask, np.roll(self.mask, -3, axis=1), np.roll(self.mask, 3, axis=1))
        prediction = SimpleNamespace(
            names={0: "plug_head"},
            boxes=[SimpleNamespace(cls=torch.tensor([0]), conf=torch.tensor([score]))
                   for score in (0.99, 0.75, 0.80)],
            masks=SimpleNamespace(data=torch.tensor(np.stack(masks))),
            orig_shape=self.mask.shape,
        )

        with self.assertRaisesRegex(ValueError, "mask IoU 0.789 below 0.80"):
            choose_end_instance(prediction)

    def test_head_and_tail_candidates_remain_ambiguous(self):
        import torch

        prediction = SimpleNamespace(
            names={0: "plug_head", 1: "plug_tail"},
            boxes=[SimpleNamespace(cls=torch.tensor([cls_id]), conf=torch.tensor([confidence]))
                   for cls_id, confidence in ((0, 0.99), (1, 0.55))],
            masks=SimpleNamespace(data=torch.tensor(np.stack((self.mask, self.mask)))),
            orig_shape=self.mask.shape,
        )

        with self.assertRaisesRegex(ValueError, "across classes"):
            choose_end_instance(prediction)

    def test_single_frame_head_and_tail_outputs_use_capture_transform(self):
        import torch
        from infer_plug_endpoint_single import run
        from plug_vg.robot_transform import robot_pose_to_matrix

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            rgb = directory / 'plug.png'
            depth = directory / 'depth.npy'
            camera_path = directory / 'camera.yaml'
            hand_eye = directory / 'hand_eye.yaml'
            image = np.arange(200 * 200 * 3, dtype=np.uint8).reshape(200, 200, 3)
            cv2.imwrite(str(rgb), image)
            np.save(depth, np.full((200,200), 500, np.uint16))
            camera_path.write_text(json.dumps({**self.camera, 'image_width':200, 'image_height':200}))
            hand_eye.write_text(json.dumps({'transform_name':'T_end_camera', 'matrix_4x4':np.eye(4).tolist()}))
            pose=[.1,.2,.3,np.pi/2,0,0]
            roi_mask = np.zeros((200, 200), dtype=np.uint8)
            roi_mask[34:148, 95:126] = 1
            for class_id, name, cfg_args, head_args in [
                (0, 'plug_head', {"end2end": None, "nms": False}, {"end2end": True}),
                (1, 'plug_tail', {"nms": None}, {}),
            ]:
                prediction=SimpleNamespace(names={0:'plug_head',1:'plug_tail'},
                    boxes=[SimpleNamespace(cls=torch.tensor([class_id]),conf=torch.tensor([.99]))],
                    masks=SimpleNamespace(data=torch.tensor(roi_mask[None])),orig_shape=roi_mask.shape)
                model=SimpleNamespace()
                args=SimpleNamespace(rgb=rgb,d2rgb=depth,robot_pose=pose,camera_config=camera_path,
                    hand_eye_config=hand_eye,placement_config=Path('configs/placement/plug_placement.yaml'),
                    seg_weights=directory/'dummy.pt',conf=.5,imgsz=640,device='cpu',output_dir=directory,show=False)
                modules = {
                    'ultralytics': SimpleNamespace(YOLO=lambda _: model),
                    'ultralytics.cfg': SimpleNamespace(DEFAULT_CFG_DICT=cfg_args),
                }
                with patch.dict('sys.modules', modules), patch.object(
                    model, 'predict', return_value=[prediction], create=True
                ) as predict:
                    result, path = run(args)
                prepared, = predict.call_args.args
                self.assertEqual(prepared.shape, image.shape)
                expected_rgb = np.full_like(image, 114)
                expected_rgb[24:156, 80:160] = image[24:156, 80:160]
                np.testing.assert_array_equal(prepared, expected_rgb)
                self.assertEqual(predict.call_args.kwargs, dict(
                    conf=.5, imgsz=640, device='cpu', retina_masks=True, nms=False, verbose=False, **head_args
                ))
                self.assertEqual(result['input']['segmentation_roi_xyxy_px'], [80, 24, 160, 156])
                self.assertEqual(result['status'],'ok',result.get('reason'))
                self.assertEqual(result['class_name'],name)
                expected=(robot_pose_to_matrix(pose)@np.r_[result['point_camera_m'],1])[:3]
                np.testing.assert_allclose(result['point_base_m'],expected,atol=1e-8)
                self.assertEqual(json.loads(path.read_text())['status'],'ok')
                depth_overlay = Path(result['artifacts']['depth_fit_overlay'])
                self.assertTrue(depth_overlay.is_file())
                rendered_depth = cv2.imread(str(depth_overlay))
                self.assertGreater(rendered_depth.shape[0], 200)
                page=Path(result['artifacts']['review_html']).read_text()
                self.assertIn('Depth fit range and quality', page)
                self.assertIn('data:image/jpeg;base64,',page)
                self.assertIn('Plotly.newPlot',page)
                if class_id == 0:
                    self.assertGreater(result['localization']['target_pixel'][1], 138)
                else:
                    self.assertLess(result['localization']['target_pixel'][1], 45)

            # A mask that reaches the padded border cannot define a valid endpoint.
            roi_mask[24:34, 95:126] = 1
            prediction.masks.data = torch.tensor(roi_mask[None])
            with patch.dict('sys.modules', modules), patch.object(
                model, 'predict', return_value=[prediction], create=True
            ):
                clipped, _ = run(args)
            self.assertEqual(clipped['status'], 'failed')
            self.assertIn('segmentation ROI', clipped['reason'])
            self.assertNotIn('point_base_m', clipped)


if __name__=='__main__':unittest.main()
