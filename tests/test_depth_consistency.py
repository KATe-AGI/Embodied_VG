import numpy as np
from plug_vg.depth_consistency import depth_consistency_keep

def reference(depth, query):
    d=np.asarray(depth,np.float32);out=np.zeros(d.shape,bool)
    for y,x in zip(*np.nonzero(query)):
        center=d[y,x]
        if not np.isfinite(center) or center<.001 or center>=100:continue
        patch=d[max(0,y-2):y+3,max(0,x-2):x+3]
        bad=(~np.isfinite(patch))|(patch<.001)|(patch>=100)|(np.abs(patch-center)>np.float32(.001))
        out[y,x]=np.count_nonzero(bad)<=.8*patch.size
    return out

def test_matches_scalar_reference_at_borders_and_cropped_regions():
    rng=np.random.default_rng(4);d=rng.uniform(.699,.702,(31,29));d[5:9,7:11]=0;d[20,20]=np.nan
    for query in [np.ones(d.shape,bool),np.pad(np.ones((5,7),bool),((10,16),(8,14)))]:
        np.testing.assert_array_equal(depth_consistency_keep(d,query),reference(d,query))

def test_retains_step_edges_and_does_not_use_mask_as_zero_depth():
    d=np.full((15,15),.7);d[:,8:]=.72;q=np.zeros_like(d,bool);q[7,7]=True
    assert depth_consistency_keep(d,q)[7,7]
    np.testing.assert_array_equal(depth_consistency_keep(d,np.ones_like(d,bool)),np.ones_like(d,bool))

def test_isolated_depth_and_invalid_center_removed():
    d=np.full((15,15),.7);d[7,7]=.75;d[2,2]=np.nan;q=np.ones_like(d,bool)
    out=depth_consistency_keep(d,q)
    assert not out[7,7] and not out[2,2] and out[7,6]

def test_extraction_keeps_original_geometry_and_reports_rejections():
    from plug_vg.visible_points import extract_visible_points_from_mask
    cam=dict(fx=500.,fy=500.,cx=12.,cy=12.,depth_scale=.001,image_width=25,image_height=25)
    d=np.full((25,25),700,np.uint16);d[12,12]=900
    kw=dict(image_bgr=np.zeros((25,25,3),np.uint8),depth_raw=d,camera=cam,polygon_xy=[[0,0],[24,0],[24,24],[0,24]],min_depth_m=.1,max_depth_m=1.,min_points=1,mask_erosion_px=5,mad_z_threshold=0.,voxel_size_m=0.)
    base=extract_visible_points_from_mask(**kw);clean=extract_visible_points_from_mask(**kw,point_filter='depth_consistency')
    assert clean.quality['depth_consistency']['removed_points']==1
    np.testing.assert_array_equal(clean.visible_points_camera_m,base.visible_points_camera_m[base.visible_points_camera_m[:,2]<.8])
    np.testing.assert_array_equal(base.mask,clean.mask)
