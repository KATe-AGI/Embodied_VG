import numpy as np
from plug_vg.registration_cleanup import refine_registration_cloud
CAM=dict(fx=1000.,fy=1000.,cx=100.,cy=100.)
def patch(x0,y0,z,n=10):
 x,y=np.meshgrid(x0+np.arange(n)*.004,y0+np.arange(n)*.004)
 return np.c_[x.ravel(),y.ravel(),np.full(x.size,z)]
def test_frontal_surface_and_nearby_moving_ring_retained():
 body=patch(-.04,-.04,.7,20);ring=patch(-.02,-.02,.71,8)
 p=np.vstack([body,ring]);result,stats=refine_registration_cloud(p,CAM,np.ones((220,220),np.uint8),.004)
 np.testing.assert_array_equal(result,p)
 assert stats['grazing_removed']==0 and stats['detached_removed']==0

def test_remote_boundary_cluster_removed_but_interior_cluster_preserved():
 body=patch(-.04,-.04,.7,20);noise=patch(.065,-.01,.78,4)
 p=np.vstack([body,noise]);mask=np.zeros((220,220),np.uint8);mask[20:200,20:185]=1
 out,stats=refine_registration_cloud(p,CAM,mask,.004)
 assert len(out)==len(body) and stats['detached_removed']==len(noise)
 out,stats=refine_registration_cloud(p,CAM,np.ones((220,220),np.uint8),.004)
 assert len(out)==len(p)

def test_sparse_empty_nonfinite():
 p=np.empty((0,3));out,_=refine_registration_cloud(p,CAM,np.zeros((10,10)),.004);assert len(out)==0
 import pytest
 with pytest.raises(ValueError):refine_registration_cloud(np.array([[0.,0.,np.nan]]),CAM,np.zeros((10,10)),.004)

def test_grazing_surface_only_removed_at_boundary():
 x,y=np.meshgrid(np.arange(15)*.0004,np.arange(15)*.004)
 p=np.c_[x.ravel(),y.ravel(),.7+x.ravel()*10]
 xy=np.rint(np.c_[1000*p[:,0]/p[:,2]+100,1000*p[:,1]/p[:,2]+100]).astype(int)
 mask=np.zeros((220,220),np.uint8);mask[90:210,98:120]=1
 _,s=refine_registration_cloud(p,CAM,mask,.004);assert s['grazing_removed']>0
 out,s=refine_registration_cloud(p,CAM,np.ones((220,220),np.uint8),.004)
 assert len(out)==len(p)
