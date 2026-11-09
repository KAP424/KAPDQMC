"""Freeze a paired-boundary experiment without modifying earlier production."""
import argparse, datetime as dt, hashlib, json, math, tomllib
from pathlib import Path

ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def scalar(x):
    if isinstance(x,bool):return str(x).lower()
    if isinstance(x,(list,tuple)):return '['+', '.join(scalar(v) for v in x)+']'
    return json.dumps(x,ensure_ascii=False)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['pilot','production'],required=True)
    ap.add_argument('--allocation');args=ap.parse_args()
    dest=ROOT/'data/born_scaling/boundary_check'/args.stage;dest.mkdir(parents=True,exist_ok=True)
    path=dest/'plan.toml'
    if path.exists():raise RuntimeError('Frozen plan already exists')
    pilot=args.stage=='pilot'
    allocation={} if pilot else json.loads(Path(args.allocation).read_text(encoding='utf-8'))
    sizes=[6,18,36] if pilot else [6,12,18,24,30,36]
    models={}
    for rel in ['data/born_scaling/production/production_plan.toml',
                'data/born_scaling/production_beta3_large/production_beta3_large_plan.toml']:
        p=tomllib.loads((ROOT/rel).read_text(encoding='utf-8'))
        for job in p['jobs']:models[job['L']]=job
    sources=['run_born_boundary_check.jl','prepare_born_boundary_check.py','launch_born_boundary_check.py',
             'run_born_scaling.jl','original_swap_scaling.jl','fused_original_swap_kernel.jl','../../public/Geometry.jl']
    hashes={f:sha(ROOT/f) for f in sources}
    # Preserve the exact version even when a later separate experiment improves tooling.
    snapshots=ROOT/'data/born_scaling/boundary_check/source_snapshots';snapshots.mkdir(exist_ok=True)
    for f,h in hashes.items():
        snap=snapshots/(h+'_'+Path(f).name)
        if not snap.exists():snap.write_bytes((ROOT/f).read_bytes())
    top=dict(version=1,stage=args.stage,state='staggered',mu=1e-8,beta=3.0,
        created_at=dt.datetime.now(dt.timezone.utc).isoformat(),execution_workers=4,blas_threads=1,
        purpose='Paired p=1/3 and p=1/2 entropies with exact local-boundary-mask subgroup averaging',
        translations='p=1/m: all (tx,ty)=(i*L/m,j*L/m), i,j=0,...,m-1; 9/4 shifts, not all L^2',
        update='Original uniform E/S Metropolis swap, fused numerical kernel; rejected states retained',
        snapshot_format='Int32 little-endian original E order, L^2 indices per recorded sweep',
        analysis='One draw per recorded sweep after subgroup mean. All differences paired within the same E. At least 3 residual df for each reported fit.',
        fits=['constant','log','linear','log_plus_inverse_L','constant_plus_inverse_L'],
        L_min=[6,12,18],minimum_residual_dof=3,
        precision_target_D=0.004,precision_target_paired_geometry_difference=0.003,
        pilot_used_for_means=False)
    if not pilot:
        top['allocation_path']=str(Path(args.allocation).resolve());top['allocation_sha256']=sha(args.allocation)
    jobs=[]
    for L in sizes:
        old=models[L];burn=128 if pilot else 256
        keep=128 if pilot else int(allocation['retained_sweeps_by_L'][str(L)])
        every=4
        for chain,start in enumerate(['random','uniform_A','uniform_B','02_stripe'],1):
            run=f'{args.stage}_boundary_L{L}_beta3_chain{chain}'
            prefix=(dest/run).relative_to(ROOT).as_posix()
            job=dict(run_id=run,stage=args.stage,L=L,beta=3.0,mu=1e-8,chain=chain,start=start,
                seed=(2_400_000_000 if pilot else 2_500_000_000)+10000*L+chain,
                model_manifest=old['model_manifest'],model_manifest_sha256=old['model_manifest_sha256'],
                warmup_sweeps=burn,analysis_burnin=burn,retained_sweeps=keep,retained_records=keep//every,
                sweeps=burn+keep,measure_every=every,refresh_every=5,
                sample_path=prefix+'_samples.csv',receipt_path=prefix+'_receipt.toml',snapshot_path=prefix+'_E.bin')
            jobs.append(job)
    with path.open('x',encoding='utf-8',newline='\n') as stream:
        for k,v in top.items():stream.write(f'{k} = {scalar(v)}\n')
        stream.write('\n[source_sha256]\n')
        for k,v in hashes.items():stream.write(f'{scalar(k)} = {scalar(v)}\n')
        for job in jobs:
            stream.write('\n[[jobs]]\n')
            for k,v in job.items():stream.write(f'{k} = {scalar(v)}\n')
    print(path)

if __name__=='__main__':main()
