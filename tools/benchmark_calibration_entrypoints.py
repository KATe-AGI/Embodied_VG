"""Alternating resident-model benchmark of both calibration entry methods."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import calibration_6d_batch as batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    sys.argv = ['benchmark', '--input-dir', str(ROOT/'test_gt_20260902'), '--robot-pose',
                '-0.014293', '0.460711', '0.742759', '2.167158', '0.044541', '-3.126827',
                '--output-dir', str(options.output/'frames'), '--device', '0']
    args = batch.parse_args()
    pairs = batch.discover_pairs(args.input_dir)
    assert len(pairs) == 5
    options.output.mkdir(parents=True, exist_ok=True)
    records, warmup = [], []

    def run_frame(method, pair, iteration):
        args.registration_method = method
        args.output_dir = options.output/'frames'/method
        result, _ = batch.run(batch.frame_args(args, *pair))
        return dict(method=method, iteration=iteration, frame=pair[0].stem, status=result['status'],
                    timing=result['timing'], t_camera_grasp=result.get('t_camera_grasp'))

    for method in ('symmetric', 'calibration'):
        for pair in pairs:
            warmup.append(run_frame(method, pair, -1))
    with (options.output/'measurements.jsonl').open('w') as stream:
        for iteration in range(20):
            methods = ('symmetric', 'calibration') if iteration % 2 == 0 else ('calibration', 'symmetric')
            for method in methods:
                for pair in pairs:
                    record = run_frame(method, pair, iteration)
                    records.append(record)
                    stream.write(json.dumps(record)+'\n'); stream.flush()
            print(f'Completed {iteration+1}/20 alternating five-frame batches', flush=True)
    summary = dict(gpu=torch.cuda.get_device_name(0), torch_version=torch.__version__,
                   warmup=warmup, rounds=20, frames_per_round=5, cuda_synchronized=True,
                   scope='segmentation + observation processing + registration + coordinate transforms; all statuses included',
                   methods={})
    for method in ('symmetric', 'calibration'):
        rows = [r for r in records if r['method'] == method]
        times = [r['timing']['warm_online_inference_s'] for r in rows]
        summary['methods'][method] = dict(count=len(rows), successes=sum(r['status']=='ok' for r in rows),
            p50_s=float(np.percentile(times,50)), p95_s=float(np.percentile(times,95)),
            stage_p50_s={key:float(np.median([r['timing'][key] for r in rows])) for key in rows[0]['timing']})
    a,b = summary['methods']['symmetric'],summary['methods']['calibration']
    summary['ratios'] = {key:b[key]/a[key] for key in ('p50_s','p95_s')}
    summary['performance_passed'] = all(v<=1.2 for v in summary['ratios'].values())
    (options.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='warmup'},indent=2))


if __name__ == '__main__':
    try:
        main()
    finally:
        batch.shutdown_calibration_workers()
