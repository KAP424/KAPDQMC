"""Short-pilot diagnostics/allocation suggestions only; no production observations."""
from pathlib import Path
import hashlib
import json
import math
import numpy as np
import analyze_born_scaling as core

ROOT=Path(__file__).resolve().parent
STAGE="pilot_beta3_large"
BURNS=(32,64,128,256)


def next_power_two(n):
    return 2**math.ceil(math.log2(max(2048,n)))


def main():
    old=json.loads((ROOT/"analysis/born_scaling/production/summary.json").read_text())
    if core.sha256(Path(core.__file__))!=old["analysis_source"]["executed_source_sha256"]:
        raise ValueError("Statistical helper changed since independently audited production")
    plan,groups,dependencies,d_error=core.load_complete_stage(STAGE)
    if tuple(plan["pilot_burnin_candidates"])!=BURNS:
        raise ValueError("Unexpected short-pilot burn-in grid")
    rows=[];blocks=[];covariances=[];timings=[];allocations=[];screen=[]
    for (L,beta,mu),chains in sorted(groups.items()):
        raw=np.stack([c["data"] for c in chains])
        if raw.shape[:2]!=(4,512):raise ValueError("Wrong frozen pilot size")
        by_burn={}
        for burn in BURNS:
            retained=raw[:,burn:,:]
            by_burn[burn]={}
            for observable,values in (("SA",retained[:,:,0]),("D",retained[:,:,3]),
                                       ("SA_minus_SC",retained[:,:,0]-retained[:,:,2])):
                diag,bb=core.error_summary(values)
                diag.update(L=L,beta=beta,mu=mu,burnin=burn,observable=observable,
                            role="Short independent pilot only; not production evidence",
                            first_half_mean=float(values[:,:values.shape[1]//2].mean()),
                            second_half_mean=float(values[:,values.shape[1]//2:].mean()))
                rows.append(diag);by_burn[burn][observable]=diag
                blocks.extend(dict(L=L,beta=beta,mu=mu,burnin=burn,observable=observable,**b) for b in bb)
            for size in core.BLOCK_LENGTHS:
                covariances.append(dict(L=L,beta=beta,mu=mu,burnin=burn,**core.covariance_checks(retained[:,:,:3],size)))
        for c in chains:
            receipt=c["receipt"]
            timings.append(dict(L=L,beta=beta,chain=c["job"]["chain"],start=c["job"]["start"],sweeps=512,
                                elapsed_seconds=receipt["elapsed_seconds"],seconds_per_sweep=receipt["elapsed_seconds"]/512,
                                concurrency=4,includes_initial_entropy_and_compile=True,
                                accepted=receipt["accepted"],proposed=receipt["proposed"],
                                inverse_drift=receipt["maximum_inverse_drift"],logweight_drift=receipt["maximum_logweight_drift"]))
        selected=next((burn for burn in BURNS if all(by_burn[later][obs]["initial_screen_pass"]
                      for later in BURNS if later>=burn for obs in ("SA","D"))),None)
        screen.append(dict(L=L,beta=beta,earliest_screen_pass_stable_over_later_cuts=selected,
                           burnin256_screen_pass=all(by_burn[256][o]["initial_screen_pass"] for o in ("SA","D")),
                           caution="A 512-sweep pilot cannot certify mixing or rule out unseen tails; production uses fresh seeds."))
        for burn in (128,256):
            diag=by_burn[burn]
            if any(diag[o]["MCSE_conservative"] is None for o in ("SA","D")):
                raise ValueError("Unresolved variance cannot determine a sample allocation")
            n=512-burn
            need_D=n*(diag["D"]["MCSE_conservative"]/.003)**2
            need_SA=n*(diag["SA"]["MCSE_conservative"]/.01)**2
            keep=next_power_two(max(need_D,need_SA))
            allocations.append(dict(L=L,beta=beta,pilot_burnin=burn,pilot_retained_per_chain=n,
                                    D_mean=diag["D"]["mean"],D_MCSE=diag["D"]["MCSE_conservative"],
                                    SA_mean=diag["SA"]["mean"],SA_MCSE=diag["SA"]["MCSE_conservative"],
                                    estimated_needed_D=need_D,estimated_needed_SA=need_SA,
                                    suggested_retained_per_chain=keep,
                                    estimated_production_D_MCSE=diag["D"]["MCSE_conservative"]*math.sqrt(n/keep),
                                    estimated_production_SA_MCSE=diag["SA"]["MCSE_conservative"]*math.sqrt(n/keep),
                                    role="Budget estimate only; no production plan or samples created"))
    out=ROOT/"analysis/born_scaling"/STAGE
    out.mkdir(parents=True,exist_ok=True)
    for name,items in (("diagnostics.csv",rows),("block_errors.csv",blocks),("region_covariance.csv",covariances),
                       ("chain_timings.csv",timings),("allocation_candidates.csv",allocations),("burnin_screen.csv",screen)):
        path=out/name;core.write_csv(path,items);dependencies[core.relative(path)]=core.sha256(path)
    dependencies[core.relative(Path(__file__))]=core.sha256(Path(__file__))
    dependencies[core.relative(Path(core.__file__))]=core.sha256(Path(core.__file__))
    summary=dict(status="PASS",stage=STAGE,role="Independent short-pilot diagnostics; no production pooling or fit selection",
                 cases=len(groups),chains=4*len(groups),raw_rows=6144,burnin_candidates=BURNS,
                 per_sample_D_identity_max_error=d_error,burnin_screens=screen,
                 allocations=allocations,files=dependencies,
                 conventions=dict(MCSE="Maximum raw-mean ESS and eligible block errors; no model target is used",
                                  blocks="At least 8 complete blocks per chain and at most 5% discarded trailing draws",
                                  allocation="n_pilot*(MCSE_pilot/absolute_target)^2 per chain, rounded up to a power of 2, minimum2048; budget estimate, not a guarantee",
                                  timing="Four concurrent workers, BLAS1; chain elapsed includes first entropy/compilation and excludes model load. Use the separate throughput benchmark before production.",
                                  inference="Finite short-pilot Rhat/ESS checks do not prove convergence; independently frozen production lengths and fresh seeds are required"))
    core.write_json(out/"summary.json",summary)
    print(json.dumps({k:v for k,v in summary.items() if k!="files"},indent=2))


if __name__=="__main__":main()
