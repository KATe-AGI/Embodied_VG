"""Single depth-consistency rejection, following FoundationPose erode_depth.

Reference: NVlabs/FoundationPose Utils.py erode_depth_kernel.
Only the rejection rule is reproduced; no bilateral smoothing is applied.
Original depth (including outside the object mask) supplies neighbor evidence.
"""
import numpy as np


def depth_consistency_keep(depth_m, query_mask):
    """Fixed experiment: radius=2, depth difference=1mm, bad ratio > 0.8.

    Crop to queried pixels plus halo for efficiency; image-border neighborhoods
    count only in-image samples. Invalid centers never become valid. Retained
    samples are not moved or interpolated. Float32 matches the upstream kernel.
    """
    depth=np.asarray(depth_m,dtype=np.float32)
    query=np.asarray(query_mask,dtype=bool)
    if depth.ndim!=2 or depth.shape!=query.shape:
        raise ValueError('depth and query_mask must be matching 2D arrays')
    result=np.zeros(depth.shape,bool)
    yy,xx=np.nonzero(query)
    if len(yy)==0:return result
    y0=max(0,yy.min()-2);y1=min(depth.shape[0],yy.max()+3)
    x0=max(0,xx.min()-2);x1=min(depth.shape[1],xx.max()+3)
    d=depth[y0:y1,x0:x1];h,w=d.shape
    padded=np.pad(d,2,constant_values=np.nan)
    in_image=np.pad(np.ones(d.shape,bool),2,constant_values=False)
    bad=np.zeros(d.shape,np.int16);total=np.zeros(d.shape,np.int16)
    for dy in range(-2,3):
        for dx in range(-2,3):
            v=padded[2+dy:2+dy+h,2+dx:2+dx+w]
            inside=in_image[2+dy:2+dy+h,2+dx:2+dx+w]
            invalid=(~np.isfinite(v))|(v<.001)|(v>=100.)
            total+=inside
            bad+=inside&(invalid|(np.abs(v-d)>np.float32(.001)))
    keep=np.isfinite(d)&(d>=.001)&(d<100.)&(bad<=.8*total)
    result[y0:y1,x0:x1]=keep&query[y0:y1,x0:x1]
    return result
