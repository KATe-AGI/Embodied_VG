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
        np.testing.assert_allclose(points['head_preset']-points['preset'], [0,0,.080])
        np.testing.assert_allclose(points['tail_preset']-points['preset'], [0,0,-.087])
        self.assertAlmostEqual(np.linalg.norm(points['head_preset']-points['tail_preset']), .167)

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

    def test_curved_surface_uses_quadratic_holdout_fit(self):
        location=locate_edge_point(self.mask,[0,-1],'plug_head',self.cfg)
        yy,xx=np.indices(self.mask.shape);u,v=location['target_pixel']
        inverse=2 + .02*((xx-u)/14)**2 + .005*((yy-v)/14)**2
        depth=1000/inverse
        fit=fit_local_depth(depth,location,self.camera,self.cfg)
        self.assertEqual(fit['quality']['model'],'inverse_depth_quadratic')
        self.assertAlmostEqual(fit['point_camera_m'][2],.5,places=7)

    def test_missing_depth_and_one_sided_support_rejected(self):
        location=locate_edge_point(self.mask,[0,-1],'plug_head',self.cfg)
        with self.assertRaisesRegex(ValueError,'Insufficient'):
            fit_local_depth(np.zeros(self.mask.shape),location,self.camera,self.cfg)
        depth=np.full(self.mask.shape,500.)
        _,v=location['target_pixel'].astype(int)
        depth[:v+1,:]=0
        with self.assertRaisesRegex(ValueError,'extrapolation'):
            fit_local_depth(depth,location,self.camera,self.cfg)

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

    def test_class_identity_and_ambiguous_instances(self):
        import torch
        class Boxes(list):pass
        boxes=Boxes([SimpleNamespace(cls=torch.tensor([1]),conf=torch.tensor([.9]))])
        r=SimpleNamespace(names={0:'plug_head',1:'plug_tail'},boxes=boxes,
                          masks=SimpleNamespace(data=torch.tensor(self.mask[None])),orig_shape=self.mask.shape)
        selected=choose_end_instance(r)
        self.assertEqual(selected['class_name'],'plug_tail')
        boxes.append(SimpleNamespace(cls=torch.tensor([0]),conf=torch.tensor([.99])))
        with self.assertRaisesRegex(ValueError,'Expected one'):
            choose_end_instance(r)

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
            cv2.imwrite(str(rgb), np.full((200,200,3), 150, np.uint8))
            np.save(depth, np.full((200,200), 500, np.uint16))
            camera_path.write_text(json.dumps({**self.camera, 'image_width':200, 'image_height':200}))
            hand_eye.write_text(json.dumps({'transform_name':'T_end_camera', 'matrix_4x4':np.eye(4).tolist()}))
            pose=[.1,.2,.3,np.pi/2,0,0]
            for class_id, name in [(0,'plug_head'),(1,'plug_tail')]:
                prediction=SimpleNamespace(names={0:'plug_head',1:'plug_tail'},
                    boxes=[SimpleNamespace(cls=torch.tensor([class_id]),conf=torch.tensor([.99]))],
                    masks=SimpleNamespace(data=torch.tensor(self.mask[None])),orig_shape=self.mask.shape)
                model=SimpleNamespace(predict=lambda *args,**kwargs:[prediction])
                args=SimpleNamespace(rgb=rgb,d2rgb=depth,robot_pose=pose,camera_config=camera_path,
                    hand_eye_config=hand_eye,placement_config=Path('configs/placement/plug_placement.yaml'),
                    seg_weights=directory/'dummy.pt',conf=.5,imgsz=640,device='cpu',output_dir=directory,show=False)
                with patch.dict('sys.modules', {'ultralytics':SimpleNamespace(YOLO=lambda _:model)}):
                    result,path=run(args)
                self.assertEqual(result['status'],'ok',result.get('reason'))
                self.assertEqual(result['class_name'],name)
                expected=(robot_pose_to_matrix(pose)@np.r_[result['point_camera_m'],1])[:3]
                np.testing.assert_allclose(result['point_base_m'],expected,atol=1e-8)
                self.assertEqual(json.loads(path.read_text())['status'],'ok')
                page=Path(result['artifacts']['review_html']).read_text()
                self.assertIn('data:image/jpeg;base64,',page)
                self.assertIn('Plotly.newPlot',page)
                if class_id==0:self.assertGreater(result['localization']['target_pixel'][1],155)
                else:self.assertLess(result['localization']['target_pixel'][1],45)


if __name__=='__main__':unittest.main()
