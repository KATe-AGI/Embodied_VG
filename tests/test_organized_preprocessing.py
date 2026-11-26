import numpy as np
from plug_vg.organized_preprocessing import clean_depth_cloud

CAM=dict(fx=1000.,fy=1000.,cx=40.,cy=40.)

def test_flying_pixels_removed_and_surface_retained():
    d=np.zeros((80,80)); m=np.zeros_like(d,bool)
    m[10:70,10:70]=True;d[m]=.6
    d[10,10:70]=.65
    d[30,30]=.8
    r=clean_depth_cloud(d,m,CAM)
    assert not r.valid_mask[30,30]
    assert not r.valid_mask[10,10:70].any()
    assert r.valid_mask[12:68,12:68].mean()>.995
    assert np.allclose(r.points[:,2],.6)

def test_separate_surfaces_holes_and_shifted_ring_are_preserved():
    d=np.zeros((80,80));d[10:35,10:70]=.6;d[40:65,10:70]=.61
    d[20:23,20:23]=0
    r=clean_depth_cloud(d,d>0,CAM)
    assert not r.valid_mask[20:23,20:23].any()
    assert r.valid_mask[42:63,12:68].all()
    assert set(np.round(r.points[:,2],5))=={.6,.61}

def test_smoothing_reduces_plane_noise_with_bounded_motion():
    rng=np.random.default_rng(8);d=.6+rng.normal(0,.0004,(80,80));m=np.ones_like(d,bool)
    raw=d.copy();r=clean_depth_cloud(d,m,CAM)
    assert np.std(r.points[:,2])<.8*np.std(d)
    assert abs(r.points[:,2].mean()-.6)<.00005
    assert r.quality['movement_max_m']<.0011
    np.testing.assert_array_equal(raw,d)

def test_invalid_tiny_and_image_border():
    d=np.full((4,4),np.nan);d[1,1]=.6
    r=clean_depth_cloud(d,np.ones_like(d,bool),CAM)
    assert len(r.points)==0 and not r.valid_mask.any()
    r=clean_depth_cloud(np.full((4,4),.6),np.ones((4,4),bool),CAM)
    assert len(r.points)==16

def test_entrypoint_exposes_full_cloud_and_preserves_default():
    from plug_vg.visible_points import extract_visible_points_from_mask
    cam={**CAM,'depth_scale':.001,'image_width':80,'image_height':80}
    kwargs=dict(image_bgr=np.zeros((80,80,3),np.uint8),depth_raw=np.full((80,80),600),camera=cam,polygon_xy=[[10,10],[69,10],[69,69],[10,69]],min_depth_m=.1,max_depth_m=1.,min_points=1,voxel_size_m=.004)
    r=extract_visible_points_from_mask(**kwargs,point_filter='edge_clean')
    assert r.status=='ok' and len(r.full_points_camera_m)>len(r.visible_points_camera_m)
    assert r.mask[10,10] and r.quality['mask_erosion_px']==0
    a=extract_visible_points_from_mask(**kwargs)
    b=extract_visible_points_from_mask(**kwargs,point_filter='none')
    np.testing.assert_array_equal(a.visible_points_camera_m,b.visible_points_camera_m)


def test_dense_detached_small_fragment_removed():
    d=np.zeros((80,80));d[5:60,5:60]=.6;d[65:71,65:71]=.68
    r=clean_depth_cloud(d,d>0,CAM)
    assert not r.valid_mask[65:71,65:71].any()
    assert r.valid_mask[5:60,5:60].all()
    assert r.quality['component_removed']==36


def test_slanted_surface_keeps_extent_without_erosion():
    yy,xx=np.mgrid[:80,:80];d=.6+.0007*xx+.0003*yy
    r=clean_depth_cloud(d,np.ones_like(d,bool),CAM)
    assert r.valid_mask.mean()>.99
    assert r.valid_mask[0,0] and r.valid_mask[-1,-1]
    assert np.max(np.abs(r.depth_m[2:-2,2:-2]-d[2:-2,2:-2]))<1e-6
