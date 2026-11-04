"""Independent read-only audit of the frozen targeted Born IID experiment.

Run only after manifest.toml exists. Does not accept/reject physics from z scores:
the audit checks integrity and algebra, then reports paired IID uncertainty.
"""
from pathlib import Path
import csv
import hashlib
import json
import math
import os
import time
import tomllib
os.environ["OPENBLAS_NUM_THREADS"]="1"
os.environ["MKL_NUM_THREADS"]="1"
os.environ["OMP_NUM_THREADS"]="1"
import numpy as np
HERE=Path(__file__).resolve().parent
DATA=HERE/"iid_targeted"

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def readtoml(p):return tomllib.loads(Path(p).read_text())
def check_hashes(mapping):
    for path,h in mapping.items():
        assert sha(HERE/path)==h, path

def stat(x):
    x=np.asarray(x,float)
    mean=float(x.mean());se=float(x.std(ddof=1)/math.sqrt(len(x)))
    return dict(mean=mean,iid_SE=se,z_from_zero=mean/se if se>0 else None,n=len(x))

def rect(L,w,h):
    return np.array([2*(x+L*y)+s for y in range(h) for x in range(w) for s in (0,1)])

def initial(meta_path):
    meta=readtoml(meta_path);N=meta["Nsites"];n=meta["Nparticles"];L=meta["L"]
    name=next(f for f in meta["files"] if f.endswith("_Q.bin"))
    Q=np.fromfile(meta_path.parent/name,dtype="<f8").reshape(N,n,order="F")
    Q=np.linalg.qr(Q)[0];G=Q@Q.T
    regs=(rect(L,L//3,L),rect(L,L//3,2*L//3),rect(L,L,L//3))
    nums=np.array([G.diagonal()[R].sum() for R in regs])
    vs=np.array([nums[i]-np.sum(G[np.ix_(R,R)]**2) for i,R in enumerate(regs)])
    s2=[]
    for R in regs:
        e=np.linalg.eigvalsh(G[np.ix_(R,R)])
        s2.append(float(-np.log(e*e+(1-e)**2).sum()))
    cells=np.arange(N)//2;x=cells%L;y=cells//L
    dx=(x[:,None]-x[None,:])%L;dy=(y[:,None]-y[None,:])%L
    kx=np.minimum(np.minimum(dx,L-dx),L//3)/L
    ky=np.minimum(np.minimum(dy,L-dy),L//3)/L
    kd=kx*ky
    ks=((x[:,None]//(L//3)!=x[None,:]//(L//3)) &
        (y[:,None]//(L//3)!=y[None,:]//(L//3)))/9
    off=G.copy();np.fill_diagonal(off,0)
    return dict(Q=Q,G=G,regs=regs,N0=nums,V0=vs,S20=np.array(s2),Kx=kx,Ky=ky,KD=kd,KS=ks,
                off=off,offnorm=float((off*off).sum()),DV0=float((kd*G*G).sum()))

def attenuation(a,beta):
    j=np.arange(a+1,dtype=float);lw=np.zeros(a+1)
    lw[1:]=np.cumsum(2*(np.log(a-j[1:]+1)-np.log(j[1:]))-4*beta)
    pj=np.exp(lw-lw.max());pj/=pj.sum();t=math.exp(-2*beta)
    return float(pj@(2*j*(a-j)+j*j/t+(a-j)**2*t)/(a*a))

def independent_observables(model,E,beta,kappa):
    P=model["Q"].copy();P[E]*=math.exp(-2*beta);P=np.linalg.qr(P)[0]
    G=P@P.T;G2=G*G
    s=[];v=[];nums=[]
    for R in model["regs"]:
        ev=np.linalg.eigvalsh(G[np.ix_(R,R)])
        s.append(float(-np.log(ev*ev+(1-ev)**2).sum()))
        v.append(float((ev*(1-ev)).sum()))
        nums.append(float(ev.sum()))
    va=float((G2*model["Kx"]).sum());vc=float((G2*model["Ky"]).sum())
    dv=float((G2*model["KD"]).sum());vb=(2*va+vc)/3-dv
    probe=float((G*model["off"]).sum()/model["offnorm"])
    law=np.array(v)+(np.array(nums)-model["N0"])**2-model["V0"]
    names=("SA","SB","SC","D","VA","VB","VC","DV","VA_full","VB_full","VC_full",
           "DV_full","DV_subgroup","NA","NB","NC","channel_probe","channel_residual",
           "Vlaw_A_residual","Vlaw_B_residual","Vlaw_C_residual")
    values=[*s,(2*s[0]+s[2])/3-s[1],*v,(2*v[0]+v[2])/3-v[1],va,vb,vc,dv,
            float((G2*model["KS"]).sum()),*nums,probe,probe-kappa,*law]
    return dict(zip(names,values))

def main():
    started=time.perf_counter()
    manifest_path=DATA/"manifest.toml"
    assert manifest_path.exists(),"Production is incomplete: manifest.toml missing"
    manifest=readtoml(manifest_path);plan=readtoml(DATA/"plan.toml")
    assert manifest["status"]=="complete"
    assert manifest["plan_sha256"]==sha(DATA/"plan.toml")
    check_hashes(manifest["files"]);check_hashes(plan["source_sha256"]);check_hashes(plan["evidence_sha256"])
    models={int(m["L"]):initial(HERE/m["path"]) for m in plan["models"]}
    groups={};batches=[];snapshots=[];max_algebra_error=0.;totalrows=0
    for job in plan["jobs"]:
        receipt=readtoml(DATA/(job["id"]+"_receipt.toml"))
        assert receipt["status"]=="complete" and receipt["job"]==job
        assert receipt["plan_sha256"]==manifest["plan_sha256"]
        check_hashes(receipt["files"])
        with (DATA/(job["id"]+".csv")).open() as f: raw=list(csv.DictReader(f))
        assert len(raw)==job["records"]
        model=models[job["L"]];N,n=model["Q"].shape
        kappa=attenuation(n,job["beta"])
        assert abs(receipt["kappa"]-kappa)<1e-11
        assert np.max(abs(np.array(receipt["N0"])-model["N0"]))<1e-10
        assert np.max(abs(np.array(receipt["V0"])-model["V0"]))<1e-10
        numeric=[];hist=np.zeros(n+1,int)
        for index,row in enumerate(raw,1):
            assert (int(row["L"]),float(row["beta"]),int(row["batch"]),int(row["sample"]),int(row["seed"]))==(
                job["L"],job["beta"],job["batch"],index,job["seed"])
            F=list(map(int,row["F"].split(";")));E=list(map(int,row["E"].split(";")));j=int(row["j"])
            assert F==sorted(set(F)) and E==sorted(set(E)) and len(F)==len(E)==n
            assert 1<=min(F+E)<=max(F+E)<=N and len(set(F)&set(E))==j
            digest=hashlib.sha256(("F:"+row["F"]+"|E:"+row["E"]+"|j:"+str(j)).encode()).hexdigest()
            assert digest==row["record_sha256"]
            hist[j]+=1
            z={name:float(value) for name,value in row.items() if name not in ("F","E","record_sha256")}
            assert all(math.isfinite(v) for v in z.values())
            errors=[z["D"]-(2*z["SA"]+z["SC"])/3+z["SB"],
                    z["DV"]-(2*z["VA"]+z["VC"])/3+z["VB"],
                    z["DV_full"]-(2*z["VA_full"]+z["VC_full"])/3+z["VB_full"],
                    z["channel_residual"]-z["channel_probe"]+kappa]
            for k,label in enumerate("ABC"):
                errors.append(z[f"Vlaw_{label}_residual"]-z[f"V{label}"]-
                              (z[f"N{label}"]-model["N0"][k])**2+model["V0"][k])
            max_algebra_error=max(max_algebra_error,max(abs(e) for e in errors))
            assert min(z["SA"]-2*z["VA"],4*math.log(2)*z["VA"]-z["SA"])>-1e-9
            z["DV_lower_gap"]=z["DV_full"]-kappa*kappa*model["DV0"]
            z["SA_lower_gap"]=z["SA"]-2*kappa*kappa*model["V0"][0]
            z["SA_upper_gap"]=4*math.log(2)*model["V0"][0]-z["SA"]
            numeric.append(z)
        assert hist.tolist()==receipt["j_counts"]
        # Independently rebuild first/last postmeasurement states of batch 1.
        if job["batch"]==1:
            for at in (0,len(raw)-1):
                E=np.array(list(map(int,raw[at]["E"].split(";"))))-1
                predicted=independent_observables(model,E,job["beta"],kappa)
                err=max(abs(predicted[k]-numeric[at][k]) for k in predicted)
                assert err<1e-9,(job["id"],at,err)
                snapshots.append(dict(job=job["id"],sample=at+1,max_observable_error=err))
        meanG=np.fromfile(DATA/(job["id"]+"_meanG.bin"),dtype="<f8").reshape(N,N,order="F")
        meanG2=np.fromfile(DATA/(job["id"]+"_meanGsquare.bin"),dtype="<f8").reshape(N,N,order="F")
        probe=float((meanG*model["off"]).sum()/model["offnorm"])
        dv=float((meanG2*model["KD"]).sum())
        assert abs(probe-np.mean([r["channel_probe"] for r in numeric]))<1e-10
        assert abs(dv-np.mean([r["DV_full"] for r in numeric]))<1e-10
        batches.append(dict(job=job["id"],channel_residual=stat([r["channel_residual"] for r in numeric]),
                            meanG_probe_error=abs(probe-np.mean([r["channel_probe"] for r in numeric])),
                            meanG2_DV_error=abs(dv-np.mean([r["DV_full"] for r in numeric]))))
        groups.setdefault((job["L"],job["beta"]),[]).extend(numeric)
        totalrows+=len(raw)
    assert totalrows==manifest["records"]
    assert max_algebra_error<1e-9
    checks=[]
    for (L,beta),rows in sorted(groups.items()):
        names=("SA","D","DV_full","channel_probe","channel_residual","Vlaw_A_residual",
               "Vlaw_B_residual","Vlaw_C_residual","DV_lower_gap","SA_lower_gap","SA_upper_gap")
        stats={name:stat([r[name] for r in rows]) for name in names}
        for name in names:
            batchmeans=[np.mean([r[name] for r in rows if r["batch"]==b]) for b in range(1,5)]
            batchse=float(np.std(batchmeans,ddof=1)/2)
            stats[name]["batch_SE"]=batchse
            stats[name]["reported_MCSE"]=max(batchse,stats[name]["iid_SE"])
        checks.append(dict(L=L,beta=beta,kappa=attenuation(L*L,beta),statistics=stats))
    comparison=[]
    analysis_path=HERE/"iid_targeted_analysis/means.csv"
    if analysis_path.exists():
        with analysis_path.open() as f: published=list(csv.DictReader(f))
        index={(int(r["L"]),float(r["beta"]),r["observable"]):r for r in published}
        rename={"DV_lower_gap":"DV_full_lowergap","SA_lower_gap":"SA_lowergap","SA_upper_gap":"SA_uppergap"}
        for case in checks:
            for name,values in case["statistics"].items():
                r=index[case["L"],case["beta"],rename.get(name,name)]
                errors={field:abs(values[own]-float(r[field])) for own,field in (
                    ("mean","mean"),("iid_SE","SE_iid"),("batch_SE","SE_batches"),("reported_MCSE","MCSE_reported"))}
                assert max(errors.values())<1e-10
                comparison.append(dict(L=case["L"],beta=case["beta"],observable=name,errors=errors))
    report=dict(status="PASS",plan_sha256=manifest["plan_sha256"],manifest_sha256=sha(manifest_path),
                rows=totalrows,jobs=len(plan["jobs"]),max_CSV_algebra_error=max_algebra_error,
                independent_rebuilt_states=snapshots,batch_matrix_checks=batches,combined_checks=checks,
                independent_summary_comparisons=comparison,
                analysis_means_sha256=sha(analysis_path) if comparison else None,
                caveat="PASS certifies integrity/algebra/state reconstruction, not a p-value or successful asymptotic scaling. IID SE may underestimate rare tails.",
                audit_source_sha256=sha(__file__),runtime_seconds=time.perf_counter()-started)
    path=HERE/"validation/iid_targeted_independent_audit.json"
    path.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({k:report[k] for k in ("status","rows","jobs","max_CSV_algebra_error","runtime_seconds")}))
    for case in checks:
        print(case["L"],case["beta"],{k:round(case["statistics"][k]["z_from_zero"],3)
            for k in ("channel_residual","Vlaw_A_residual","Vlaw_B_residual","Vlaw_C_residual")})

if __name__=="__main__":main()
