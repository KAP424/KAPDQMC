"""Freeze production after the independent boundary pilot; do not alter the pilot."""
import datetime as dt, hashlib, json, math, tomllib
from pathlib import Path
from prepare_born_boundary_check import ROOT, sha, scalar

def main():
    pilotpath=ROOT/'analysis/born_scaling/boundary_check/pilot/pilot_summary.json'
    pilot=json.loads(pilotpath.read_text(encoding='utf-8'))
    assert pilot['status']=='PASS' and pilot['stage']=='pilot'
    for f,h in pilot['files'].items():assert sha(ROOT/f)==h,f
    # This is a budget estimate, not a precision or convergence guarantee.
    # Pilot observed one draw / 4 sweeps; production records every sweep.
    # A factor two in variance budgets is a heuristic allowance for denser-series
    # correlation, not a bound (the cadence ratio can approach four for slow modes).
    # Actual ESS/MCSE is checked after the fixed production finishes.
    byL={r['L']:r for r in pilot['budget_evidence']}
    obs=['p3_group_D','p2_group_D','half_minus_third_group_D']
    targets=dict(zip(obs,[.004,.004,.003]))
    minimum={6:2048,12:1024,18:512,24:512,30:256,36:256}
    allocation={};evidence=[]
    for L in [6,12,18,24,30,36]:
        reference=next(size for size in [6,18,36] if size>=L)
        variance={name:max(byL[s][name+'_variance_scale'] for s in byL if s<=reference) for name in obs}
        needed=max(2*variance[name]/targets[name]**2 for name in obs)
        keep=max(minimum[L],128*math.ceil(needed/128))
        allocation[str(L)]=keep
        evidence.append(dict(L=L,pilot_reference=reference,variance_envelope=variance,records_per_chain=keep,
                             variance_margin=2,minimum_records=minimum[L]))
    output=ROOT/'data/born_scaling/boundary_check/production';output.mkdir(parents=True,exist_ok=True)
    planpath=output/'plan.toml';allocationpath=output/'allocation.json'
    if planpath.exists() or allocationpath.exists():raise RuntimeError('Production is already frozen')
    budget=dict(created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),retained_sweeps_by_L=allocation,
        measure_every=1,targets=targets,evidence=evidence,pilot_summary_sha256=sha(pilotpath),
        rule='Next-larger pilot variance envelope; factor 2 variance margin for cadence 4 to 1; round up 128; size-dependent minimum fixed before production',
        caveat='Pilot only estimates variance. No optional stopping; report actual MCSE, including any target misses.')
    allocationpath.write_text(json.dumps(budget,indent=2),encoding='utf-8')
    models={}
    for rel in ['data/born_scaling/production/production_plan.toml',
                'data/born_scaling/production_beta3_large/production_beta3_large_plan.toml']:
        for job in tomllib.loads((ROOT/rel).read_text(encoding='utf-8'))['jobs']:models[job['L']]=job
    source_files=['run_born_boundary_check.jl','run_born_boundary_check_fast.jl','prepare_born_boundary_production.py','prepare_born_boundary_check.py',
        'launch_born_boundary_production.py','run_born_scaling.jl','original_swap_scaling.jl','fused_original_swap_kernel.jl',
        '../../public/Geometry.jl','analyze_born_boundary_check.py','analyze_born_scaling.py']
    hashes={f:sha(ROOT/f) for f in source_files}
    snapshots=ROOT/'data/born_scaling/boundary_check/source_snapshots';snapshots.mkdir(exist_ok=True)
    for f,h in hashes.items():
        path=snapshots/(h+'_'+Path(f).name)
        if not path.exists():path.write_bytes((ROOT/f).read_bytes())
    top=dict(version=1,stage='production',state='staggered',mu=1e-8,beta=3.0,
        created_at=dt.datetime.now(dt.timezone.utc).isoformat(),execution_workers=4,blas_threads=1,
        update='Original uniform E/S Metropolis swap, fused kernel; all rejected states retained',
        translations='p=1/m: all (i*L/m,j*L/m), i,j=0,...,m-1; 9/4 shifts; shared local straight-edge mask exactly cancels',
        analysis='One draw per recorded sweep after subgroup mean. Geometry and origin contrasts paired within E. No pilot observations pooled.',
        entropy_kernel='Full correlation matrix once per state; mathematically equivalent purity Cholesky on each region; independently validated',
        analysis_burnin=256,measure_every=1,L_min=[6,12,18],minimum_residual_dof=3,
        fits=['constant','log','linear','log_plus_inverse_L','constant_plus_inverse_L'],
        precision_target_D=.004,precision_target_paired_geometry_difference=.003,
        allocation_path=allocationpath.relative_to(ROOT).as_posix(),allocation_sha256=sha(allocationpath),
        pilot_summary_path=pilotpath.relative_to(ROOT).as_posix(),pilot_summary_sha256=sha(pilotpath))
    evidence_files=[pilotpath,ROOT/'data/born_scaling/boundary_check/pilot/manifest.json',
        ROOT/'data/born_scaling/boundary_check/boundary_audit.json',
        ROOT/'data/born_scaling/boundary_check/independent_validation.json',
        ROOT/'analysis/born_scaling/boundary_check/kernel_validation_extended.toml',
        ROOT/'analysis/born_scaling/boundary_check/fast_kernel_validation.toml']
    jobs=[]
    for L in [6,12,18,24,30,36]:
        old=models[L];keep=allocation[str(L)]
        for chain,start in enumerate(['random','uniform_A','uniform_B','02_stripe'],1):
            run=f'production_boundary_L{L}_beta3_chain{chain}';prefix=(output/run).relative_to(ROOT).as_posix()
            jobs.append(dict(run_id=run,stage='production',L=L,beta=3.,mu=1e-8,chain=chain,start=start,
                seed=2_500_000_000+10000*L+chain,model_manifest=old['model_manifest'],
                model_manifest_sha256=old['model_manifest_sha256'],warmup_sweeps=256,analysis_burnin=256,
                retained_sweeps=keep,retained_records=keep,sweeps=256+keep,measure_every=1,refresh_every=5,
                sample_path=prefix+'_samples.csv',receipt_path=prefix+'_receipt.toml',snapshot_path=prefix+'_E.bin'))
    with planpath.open('x',encoding='utf-8',newline='\n') as stream:
        for k,v in top.items():stream.write(f'{k} = {scalar(v)}\n')
        for name, mapping in [('source_sha256',hashes),('evidence_sha256',
                {p.relative_to(ROOT).as_posix():sha(p) for p in evidence_files})]:
            stream.write('\n['+name+']\n')
            for k,v in mapping.items():stream.write(f'{scalar(k)} = {scalar(v)}\n')
        for job in jobs:
            stream.write('\n[[jobs]]\n')
            for k,v in job.items():stream.write(f'{k} = {scalar(v)}\n')
    print(json.dumps(budget,indent=2));print(planpath)

if __name__=='__main__':main()
