"""Geometry checks on uniformly sampled registration points, independent of CAD."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def refine_registration_cloud(points, camera, raw_mask, voxel_size_m):
    """One pass: boundary grazing surfaces, then remote boundary components.

    Near-grazing true silhouette surfaces may also be removed. The 20 mm
    attachment allowance preserves nearby independently moving ring surfaces;
    it is a task-specific conservative allowance, not a deformation model.
    """
    import cv2
    p=np.asarray(points,dtype=float)
    if p.ndim!=2 or p.shape[1]!=3 or not np.isfinite(p).all() or np.any(p[:,2]<=0):
        raise ValueError('points must be finite Nx3 with positive camera depth')
    n=len(p);keep=np.ones(n,bool)
    stats=dict(input_points=n,grazing_removed=0,detached_removed=0,output_points=n,
               boundary_band_px=10.,grazing_cosine=.2,attachment_allowance_m=.02,
               component_boundary_band_px=20.,component_boundary_fraction=.8)
    if n<12:return p.copy(),stats
    # Normal support is defined by the actual point spacing for voxel_size=0.
    spacing=float(voxel_size_m)
    if spacing<=0:
        spacing=float(np.median(cKDTree(p).query(p,k=2)[0][:,1]))
    radius=max(.003,1.75*spacing)
    stats['component_link_radius_m']=radius
    distance=cv2.distanceTransform(np.asarray(raw_mask,dtype=np.uint8),cv2.DIST_L2,5)
    xy=np.rint(np.column_stack([camera['fx']*p[:,0]/p[:,2]+camera['cx'],
                               camera['fy']*p[:,1]/p[:,2]+camera['cy']])).astype(int)
    inside=(xy[:,0]>=0)&(xy[:,0]<distance.shape[1])&(xy[:,1]>=0)&(xy[:,1]<distance.shape[0])
    edge=np.zeros(n);edge[inside]=distance[xy[inside,1],xy[inside,0]]
    tree=cKDTree(p);dist,idx=tree.query(p,k=12)
    q=p[idx];q=q-q.mean(axis=1)[:,None,:]
    eigen,vec=np.linalg.eigh(np.einsum('nki,nkj->nij',q,q))
    cosine=np.abs(np.einsum('ni,ni->n',vec[:,:,0],p/np.linalg.norm(p,axis=1)[:,None]))
    reliable=(eigen[:,1]>.1*np.maximum(eigen[:,2],1e-15))&(eigen[:,0]<.1*np.maximum(eigen.sum(axis=1),1e-15))&(dist[:,-1]<max(.025,5*spacing))
    grazing=(edge<=10)&(cosine<.2)&reliable
    keep[grazing]=False;stats['grazing_removed']=int(grazing.sum())
    indices=np.flatnonzero(keep);a=p[indices]
    if len(a):
        pairs=cKDTree(a).query_pairs(radius,output_type='ndarray')
        graph=coo_matrix((np.ones(len(pairs),dtype=np.uint8),(pairs[:,0],pairs[:,1])),shape=(len(a),len(a))).tocsr()
        nc,labels=connected_components(graph,directed=False)
        counts=np.bincount(labels);anchor=int(np.argmax(counts));anchor_tree=cKDTree(a[labels==anchor])
        components=[]
        for label in range(nc):
            if label==anchor:continue
            member=labels==label;global_ids=indices[member]
            gap=float(anchor_tree.query(a[member])[0].min())
            fraction=float(np.mean(edge[global_ids]<=20))
            rejected=gap>.02 and fraction>=.8
            if rejected:keep[global_ids]=False
            components.append(dict(points=int(member.sum()),gap_m=gap,boundary_fraction=fraction,removed=bool(rejected)))
        stats['components']=components
    stats['detached_removed']=int(n-keep.sum()-grazing.sum());stats['output_points']=int(keep.sum())
    return p[keep],stats
