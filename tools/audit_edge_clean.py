"""Reference-free cleaning audit; geometry proxies are not physical accuracy."""
from pathlib import Path
import sys,json,csv,time,argparse
import numpy as np
import cv2,yaml
from scipy.spatial import cKDTree
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from plug_vg.geometry import polygon_mask
from plug_vg.organized_preprocessing import clean_depth_cloud
from plug_vg.visible_points import _depth_to_camera_points,erode_mask,save_visible_points_ply
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def roughness(p):
    if len(p)<15:return None
    ids=np.linspace(0,len(p)-1,min(2000,len(p)),dtype=int)
    _,idx=cKDTree(p).query(p[ids],k=12)
    n=p[idx];c=n.mean(1);a=n-c[:,None,:]
    _,vec=np.linalg.eigh(np.einsum('nki,nkj->nij',a,a)/12)
    residual=np.abs(np.einsum('ni,ni->n',p[ids]-c,vec[:,:,0]))
    return dict(median_mm=float(np.median(residual)*1000),p95_mm=float(np.quantile(residual,.95)*1000))

def main():
    dataset=ROOT/'test_data/plug_camera_eval_20261106'
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    out=parser.parse_args().output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    cache=json.loads((ROOT/'.codex-runs/point_preprocess_20261106_01/segmentation_cache.json').read_text())
    rows=list(csv.DictReader((dataset/'manifest.csv').open()));results=[]
    for i,row in enumerate(rows):
        image=cv2.imread(str(dataset/row['rgb']));d=np.load(dataset/row['depth'])
        camera=yaml.safe_load((dataset/row['camera_config']).read_text())
        seg=cache[str(dataset/row['rgb'])];item=max(seg,key=lambda x:x['confidence'])
        mask=polygon_mask(item['polygon_xy'],d.shape)
        raw,pix,z=_depth_to_camera_points(mask,d,camera,.1,1.)
        base,_,_=_depth_to_camera_points(erode_mask(mask,5),d,camera,.1,1.)
        start=time.perf_counter();r=clean_depth_cloud(d*camera['depth_scale'],mask,camera);elapsed=time.perf_counter()-start
        removed=~r.valid_mask[pix[:,1].astype(int),pix[:,0].astype(int)]
        # Stable deep-mask pixels: retention is a proxy, not a ground-truth label.
        core=erode_mask(mask,10)>0;valid=np.isfinite(d)&(d*camera['depth_scale']>=.1)&(d*camera['depth_scale']<=1.)
        core&=valid
        result=dict(sample_id=row['sample_id'],group=row['group'],raw_count=len(raw),baseline_count=len(base),clean_count=len(r.points),seconds=elapsed,quality=r.quality,
                    core_retention=float(r.valid_mask[core].mean()) if core.any() else None,
                    raw_roughness=roughness(raw),baseline_roughness=roughness(base),clean_roughness=roughness(r.points))
        # Fixed raw PCA axis only for comparing extent; not a true anatomical endpoint.
        _,_,vt=np.linalg.svd(raw-np.median(raw,axis=0),full_matrices=False);axis=vt[0]
        for key,p in [('raw',raw),('baseline',base),('clean',r.points)]:
            result[key+'_axis_quantiles_mm']=(np.quantile(p@axis,[.01,.99])*1000).tolist() if len(p) else None
        results.append(result)
        dest=out/row['sample_id'];dest.mkdir()
        save_visible_points_ply(r.points,dest/'clean_full.ply',max_points=len(r.points))
        save_visible_points_ply(raw[removed],dest/'removed.ply',max_points=len(raw))
        np.savez_compressed(dest/'clean_depth_mask.npz',depth_m=r.depth_m,valid_mask=r.valid_mask)
        if i<6 or row['subset']!='original_156':
            # Same PCA camera and limits for all panels, mm units.
            center=np.median(raw,axis=0);basis=vt.T
            fig,axs=plt.subplots(2,3,figsize=(14,8))
            for col,(label,p) in enumerate([('Raw',raw),('Old erosion 5px',base),('Edge clean',r.points)]):
                q=(p-center)@basis*1000
                for j,dim in enumerate([1,2]):
                    ax=axs[j,col];ax.scatter(q[:,0],q[:,dim],s=.3,c='#176fa6',rasterized=True)
                    if col==2:
                        rem=(raw[removed]-center)@basis*1000
                        ax.scatter(rem[:,0],rem[:,dim],s=.5,c='#dc4638',alpha=.6)
                    allq=(raw-center)@basis*1000
                    lim=np.quantile(allq[:,[0,dim]],[0,1],axis=0)
                    ax.set_xlim(lim[0,0]-2,lim[1,0]+2);ax.set_ylim(lim[0,1]-2,lim[1,1]+2)
                    ax.set_aspect('equal');ax.set_title(label+(' + removed(red)' if col==2 else ''));ax.set_xlabel('PCA axis 1 / mm')
            fig.suptitle(row['sample_id']);fig.tight_layout();fig.savefig(dest/'comparison.png',dpi=130);plt.close(fig)
        if (i+1)%20==0:print(i+1,len(rows),flush=True)
    (out/'metrics.json').write_text(json.dumps(results,indent=2))
    summary={}
    for key in ['raw_count','baseline_count','clean_count','seconds','core_retention']:
        a=np.array([r[key] for r in results]);summary[key]=dict(mean=float(a.mean()),min=float(a.min()),max=float(a.max()))
    for key in ['raw_roughness','baseline_roughness','clean_roughness']:
        summary[key]={q:float(np.mean([r[key][q] for r in results])) for q in ['median_mm','p95_mm']}
    summary['frames']=len(results)
    summary['note']='No independent physical ground truth. Roughness and core retention are diagnostic proxies, not noise precision or pose accuracy.'
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
