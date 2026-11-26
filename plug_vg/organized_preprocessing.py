"""Conservative organized-depth cleaning; no CAD or pose reference is used."""
from dataclasses import dataclass
import numpy as np
from scipy.spatial import cKDTree

@dataclass(frozen=True)
class CleanResult:
    points: np.ndarray
    pixels: np.ndarray
    valid_mask: np.ndarray
    depth_m: np.ndarray
    quality: dict


def clean_depth_cloud(depth_m, mask, camera, min_depth=.1, max_depth=1., *, smooth=True):
    """Local depth support -> adaptive-radius outliers -> gated bilateral.

    Thresholds are an initial engineering profile, not a calibrated sensor model.
    All neighborhoods exclude invalid and out-of-mask measurements. No inpainting.
    """
    depth = np.asarray(depth_m, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if depth.ndim != 2 or mask.shape != depth.shape:
        raise ValueError('depth and mask must be matching 2D arrays')
    fx, fy = float(camera['fx']), float(camera['fy'])
    if not np.isfinite([fx, fy]).all() or min(fx, fy) <= 0:
        raise ValueError('focal lengths must be finite and positive')
    valid = mask & np.isfinite(depth) & (depth >= min_depth) & (depth <= max_depth)
    original_count = int(valid.sum())
    outdepth = np.zeros_like(depth)
    outmask = np.zeros_like(mask)
    profile = dict(depth_gate_base_m=.002, depth_gate_at_07m_m=.004,
                   support_window=5, min_support=3, min_support_fraction=.4,
                   radius_pixel_factor=3., radius_floor_m=.002,
                   radius_min_neighbors=3, smoothing_window=3,
                   smoothing_sigma_m=.0015, max_depth_movement_m=.001)
    if not original_count:
        return CleanResult(np.empty((0,3)), np.empty((0,2),int),outmask,outdepth,
                           dict(input_points=0,output_points=0,profile=profile))
    yy, xx = np.nonzero(mask)
    y0,y1,x0,x1=yy.min(),yy.max()+1,xx.min(),xx.max()+1
    d=depth[y0:y1,x0:x1].copy(); v=valid[y0:y1,x0:x1].copy()
    d[~v]=0
    def neighbors(a, radius):
        padded=np.pad(a,radius,mode='constant')
        h,w=a.shape
        for dy in range(-radius,radius+1):
            for dx in range(-radius,radius+1):
                if dy or dx:
                    yield dy,dx,padded[radius+dy:radius+dy+h,radius+dx:radius+dx+w]
    gate=.002+.002*(d/.7)**2
    support=np.zeros(d.shape,int); available=np.zeros(d.shape,int)
    for (_,_,nd),(_,_,nv) in zip(neighbors(d,2),neighbors(v,2)):
        available += nv
        support += nv & (np.abs(nd-d)<=gate)
    keep=v & (support>=3) & (support>=.4*available)
    # Reject small disconnected depth fragments, keeping multiple substantial surfaces.
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    n=int(keep.sum())
    component_removed=0
    component_min=min(64, original_count) if original_count < 128 else max(64, int(np.ceil(.005*original_count)))
    if n:
        ids=np.full(d.shape,-1,dtype=int);ids[keep]=np.arange(n)
        starts=[];ends=[]
        for dy,dx in [(0,1),(1,0),(1,1),(1,-1)]:
            h,w=d.shape
            sa=(slice(max(0,dy),h+min(0,dy)),slice(max(0,dx),w+min(0,dx)))
            sb=(slice(max(0,-dy),h-max(0,dy)),slice(max(0,-dx),w-max(0,dx)))
            a,b=ids[sa],ids[sb]
            good=(a>=0)&(b>=0)&(np.abs(d[sa]-d[sb])<=np.minimum(gate[sa],gate[sb]))
            starts.append(a[good]);ends.append(b[good])
        a=np.concatenate(starts);b=np.concatenate(ends)
        graph=coo_matrix((np.ones(len(a),dtype=np.uint8),(a,b)),shape=(n,n)).tocsr()
        _,labels=connected_components(graph,directed=False)
        counts=np.bincount(labels)
        accepted=counts[labels]>=component_min
        component_removed=int((~accepted).sum())
        keep[keep]=accepted
    after_support=int(keep.sum())
    ys,xs=np.nonzero(keep); z=d[ys,xs]
    pixels=np.column_stack([xs+x0,ys+y0])
    def project(z,pix):
        return np.column_stack([(pix[:,0]-camera['cx'])*z/fx,
                                (pix[:,1]-camera['cy'])*z/fy,z])
    points=project(z,pixels)
    if len(points):
        radius=np.maximum(.002,3*z/min(fx,fy))
        counts=cKDTree(points).query_ball_point(points,radius,return_length=True)-1
        rkeep=counts>=3
        keep[ys[~rkeep],xs[~rkeep]]=False
        pixels=pixels[rkeep]; ys=ys[rkeep]; xs=xs[rkeep]; z=z[rkeep]
    after_radius=len(z)
    original_z=z.copy()
    if smooth and len(z):
        numerator=d.copy(); denominator=np.ones_like(d)
        for (dy,dx,nd),(_,_,nv) in zip(neighbors(d,1),neighbors(keep,1)):
            delta=nd-d
            weights=np.exp(-.5*(dx*dx+dy*dy)-.5*(delta/.0015)**2)
            weights *= nv & (np.abs(delta)<=gate)
            numerator+=weights*nd; denominator+=weights
        candidate=numerator/denominator
        z+=np.clip(candidate[ys,xs]-z,-.001,.001)
    points=project(z,pixels)
    outmask[pixels[:,1],pixels[:,0]]=True
    outdepth[pixels[:,1],pixels[:,0]]=z
    movement=np.linalg.norm(project(z-original_z,pixels),axis=1)
    quality=dict(input_points=original_count,after_support=after_support,
                 after_radius=after_radius,output_points=len(z),
                 support_removed=original_count-after_support-component_removed,
                 component_removed=component_removed, component_min_points=component_min,
                 radius_removed=after_support-after_radius,
                 movement_mean_m=float(movement.mean()) if len(z) else 0.,
                 movement_max_m=float(movement.max()) if len(z) else 0.,
                 smooth=bool(smooth),profile=profile)
    return CleanResult(points,pixels,outmask,outdepth,quality)
