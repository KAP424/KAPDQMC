"""Frozen-budget iid validation of channel identities and entropy/variance bounds.

This is a numerical check of independently derived exact statements, not a fit
for entropy corner coefficients. Matrix entries are never independent samples.
"""
from pathlib import Path
import csv
import hashlib
import json
import math
import tomllib
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "iid_targeted"
OUT = HERE / "iid_targeted_analysis"
OBS = ("SA", "SB", "SC", "D", "VA", "VB", "VC", "DV", "VA_full", "VB_full", "VC_full", "DV_full", "DV_subgroup", "NA", "NB", "NC", "channel_probe", "channel_residual", "Vlaw_A_residual", "Vlaw_B_residual", "Vlaw_C_residual")
CORE = ("channel_residual", "Vlaw_A_residual", "Vlaw_B_residual", "Vlaw_C_residual", "DV_full_lowergap", "SA_lowergap", "SA_uppergap")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def read_matrix(path, N):
    return np.fromfile(path, dtype=np.float64).reshape((N, N), order="F")


def statistics(values):
    flat = np.concatenate(values)
    mean = float(flat.mean())
    variance = float(flat.var(ddof=1))
    se = math.sqrt(variance / len(flat))
    batchmeans = np.array([v.mean() for v in values])
    batchse = float(batchmeans.std(ddof=1) / math.sqrt(len(values)))
    reported = max(se, batchse)
    absolute = np.sort(np.abs(flat))
    return dict(mean=mean, SE_iid=se, SE_batches=batchse, MCSE_reported=reported,
                z_mean_over_reported_SE=mean/reported if reported else None,
                n=len(flat), batches=len(values), iid_single_draw_variance=variance,
                minimum=float(flat.min()), median=float(np.median(flat)), p99=float(np.quantile(flat,.99)), maximum=float(flat.max()),
                largest_draw_fraction_absolute_sum=float(absolute[-1]/absolute.sum()) if absolute.sum() else 0,
                top_one_percent_fraction_absolute_sum=float(absolute[-max(1,math.ceil(len(flat)/100)):].sum()/absolute.sum()) if absolute.sum() else 0)


def main():
    manifest_path, plan_path = DATA / "manifest.toml", DATA / "plan.toml"
    manifest = tomllib.loads(manifest_path.read_text())
    plan = tomllib.loads(plan_path.read_text())
    assert manifest["status"] == "complete" and manifest["plan_sha256"] == sha(plan_path)
    sources = {"iid_targeted/manifest.toml":sha(manifest_path), "analyze_iid_targeted.py":sha(Path(__file__))}
    for mapping in (manifest["files"], plan["source_sha256"], plan["evidence_sha256"]):
        for name, expected in mapping.items():
            assert sha(HERE / name) == expected, name
            sources[name] = expected
    assert len({j["seed"] for j in plan["jobs"]}) == len(plan["jobs"]) == 24
    pilot_seeds={j["seed"] for j in tomllib.loads((HERE/"pilot/plan.toml").read_text())["jobs"]}
    assert not pilot_seeds.intersection(j["seed"] for j in plan["jobs"])
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError("Preserve existing successful analysis")
    records = {}
    reference_error, pure_bound_violation = 0.0, 0.0
    record_count = 0
    times = []
    for job in plan["jobs"]:
        receipt = tomllib.loads((DATA/(job["id"]+"_receipt.toml")).read_text())
        assert receipt["status"] == "complete" and receipt["job"] == job and receipt["plan_sha256"] == sha(plan_path)
        reference_error=max(reference_error,receipt["maximum_reference_entropy_error"])
        pure_bound_violation=max(pure_bound_violation,receipt["maximum_single_state_bound_violation"])
        with (DATA/(job["id"]+".csv")).open(newline="") as stream:
            rows=list(csv.DictReader(stream))
        assert len(rows)==job["records"]==512
        a,N=job["L"]**2,2*job["L"]**2
        values={name:np.array([float(r[name]) for r in rows]) for name in OBS}
        for index,row in enumerate(rows,1):
            E,F=[[int(i) for i in row[name].split(";")] for name in ("E","F")]
            j=int(row["j"])
            assert len(set(E))==len(E)==len(set(F))==len(F)==a
            assert all(1<=i<=N for i in E+F) and len(set(E)&set(F))==j
            assert int(row["L"])==job["L"] and float(row["beta"])==job["beta"]
            assert int(row["batch"])==job["batch"] and int(row["seed"])==job["seed"] and int(row["sample"])==index
            expected_hash=hashlib.sha256(f"F:{row['F']}|E:{row['E']}|j:{j}".encode()).hexdigest()
            assert expected_hash==row["record_sha256"]
        assert np.max(np.abs((2*values["SA"]+values["SC"])/3-values["SB"]-values["D"]))<1e-10
        assert np.max(np.abs((2*values["VA"]+values["VC"])/3-values["VB"]-values["DV"]))<1e-10
        kappa=receipt["kappa"]
        assert np.max(np.abs(values["channel_probe"]-kappa-values["channel_residual"]))<1e-12
        for ri,region in enumerate(("A","B","C")):
            S,V,number=values["S"+region],values["V"+region],values["N"+region]
            V0,N0=receipt["V0"][ri],receipt["N0"][ri]
            assert np.max(2*V-S)<1e-9 and np.max(S-4*math.log(2)*V)<1e-9
            assert np.max(np.abs(V+(number-N0)**2-V0-values[f"Vlaw_{region}_residual"]))<1e-10
            values[f"S{region}_lowergap"]=S-2*kappa**2*V0
            values[f"S{region}_uppergap"]=4*math.log(2)*V0-S
            values[f"V{region}_lowergap"]=V-kappa**2*V0
            values[f"V{region}_uppergap"]=V0-V
        values["DV_full_lowergap"]=values["DV_full"]-kappa**2*receipt["DV0_full"]
        G=read_matrix(DATA/(job["id"]+"_meanG.bin"),N)
        G2=read_matrix(DATA/(job["id"]+"_meanGsquare.bin"),N)
        assert np.min(G2-G**2)>=-1e-12 and abs(np.trace(G)-a)<1e-8
        records.setdefault((job["L"],job["beta"]),[]).append(dict(job=job,receipt=receipt,values=values,G=G,G2=G2,
            j=np.array([int(r["j"]) for r in rows])))
        record_count+=len(rows)
        for name in ("sampling_seconds","entropy_seconds","diagnostic_seconds"):
            times.append(dict(L=job["L"],beta=job["beta"],batch=job["batch"],phase=name,
                total_seconds=receipt[name],seconds_per_record=receipt[name]/len(rows)))
    assert record_count==manifest["records"]==12288
    means,batches,covariances,matrix_checks,bounds,jchecks=[],[],[],[],[],[]
    for (L,beta),entries in sorted(records.items()):
        entries=sorted(entries,key=lambda r:r["job"]["batch"])
        assert len(entries)==4
        r0=entries[0]["receipt"];kappa=r0["kappa"];N=2*L**2;a=L**2
        for name in entries[0]["values"]:
            stat=statistics([entry["values"][name] for entry in entries])
            means.append(dict(L=L,beta=beta,observable=name,**stat))
            for entry in entries:
                y=entry["values"][name]
                batches.append(dict(L=L,beta=beta,batch=entry["job"]["batch"],observable=name,mean=float(y.mean()),SE=float(y.std(ddof=1)/math.sqrt(len(y))),n=len(y)))
        core=np.concatenate([np.column_stack([entry["values"][name] for name in CORE]) for entry in entries])
        cov=np.cov(core,rowvar=False,ddof=1)/len(core)
        for i,name in enumerate(CORE):
            for j,other in enumerate(CORE):
                covariances.append(dict(L=L,beta=beta,first=name,second=other,iid_mean_covariance=float(cov[i,j])))
        G0=read_matrix(next((DATA/"models").glob(f"*_L{L}_*_G.bin")),N)
        meanG=sum(entry["G"] for entry in entries)/4
        meanG2=sum(entry["G2"] for entry in entries)/4
        expected=kappa*G0;np.fill_diagonal(expected,np.diag(G0))
        variance_mean=np.maximum(meanG2-meanG**2,0)/(2048-1)
        rms=float(np.sqrt(variance_mean.sum()))
        x,y=np.repeat(np.arange(a)%L,2),np.repeat(np.arange(a)//L,2)
        dx,dy=(x[:,None]-x[None,:])%L,(y[:,None]-y[None,:])%L
        kernel=np.minimum(np.minimum(dx,L-dx),L//3)*np.minimum(np.minimum(dy,L-dy),L//3)/L**2
        csv_DV=float(np.concatenate([entry["values"]["DV_full"] for entry in entries]).mean())
        matrix_DV=float(np.sum(meanG2*kernel))
        assert abs(csv_DV-matrix_DV)<1e-10
        matrix_checks.append(dict(L=L,beta=beta,kappa=kappa,Frobenius_error=float(np.linalg.norm(meanG-expected)),
            Frobenius_RMS_sampling_estimate=rms,Frobenius_error_over_RMS=float(np.linalg.norm(meanG-expected)/rms),
            matrix_DV_full=matrix_DV,CSV_DV_full=csv_DV,matrix_entries_used_as_independent_samples=False))
        for ri,region in enumerate(("A","B","C")):
            name="S"+region
            row=next(v for v in means if v["L"]==L and v["beta"]==beta and v["observable"]==name)
            bounds.append(dict(L=L,beta=beta,observable=name,initial_variance=r0["V0"][ri],
                deterministic_lower_bound=2*kappa**2*r0["V0"][ri],deterministic_upper_bound=4*math.log(2)*r0["V0"][ri],
                mean=row["mean"],MCSE_reported=row["MCSE_reported"]))
        row=next(v for v in means if v["L"]==L and v["beta"]==beta and v["observable"]=="DV_full")
        bounds.append(dict(L=L,beta=beta,observable="DV_full",initial_variance=r0["DV0_full"],
            deterministic_lower_bound=kappa**2*r0["DV0_full"],deterministic_upper_bound=None,
            mean=row["mean"],MCSE_reported=row["MCSE_reported"]))
        lw=np.zeros(a+1)
        for j in range(1,a+1):lw[j]=lw[j-1]+2*math.log((a-j+1)/j)-4*beta
        p=np.exp(lw-lw.max());p/=p.sum();index=np.arange(a+1)
        observed=np.concatenate([entry["j"] for entry in entries])
        target=float(p@index);variance=float(p@((index-target)**2));se=math.sqrt(variance/len(observed))
        jchecks.append(dict(L=L,beta=beta,exact_mean=target,sample_mean=float(observed.mean()),exact_mean_SD=se,
            standardized_mean_difference=float((observed.mean()-target)/se),minimum=int(observed.min()),maximum=int(observed.max()),
            zero_count=int(np.sum(observed==0)),expected_zero_count=float(len(observed)*p[0])))
    OUT.mkdir(exist_ok=True)
    outputs=[]
    for name,rows in (("means",means),("batches",batches),("core_covariance",covariances),("matrix_checks",matrix_checks),("bounds",bounds),("j_checks",jchecks),("timing",times)):
        path=OUT/(name+".csv");write_csv(path,rows);outputs.append(path)
    summary=dict(status="PASS",role="Completed fixed-budget targeted numerical validation; no scaling fits",records=record_count,
        parameters=6,batches_per_parameter=4,records_per_parameter=2048,core_observables=list(CORE),
        maximum_independent_entropy_error=reference_error,maximum_single_state_bound_violation=pure_bound_violation,
        all_seeds_distinct_and_disjoint_from_pilot=True,
        zero_expectation_checks=[row for row in means if row["observable"] in CORE[:4]],
        core_bound_checks=[row for row in means if row["observable"] in CORE[4:]],
        sources_sha256=sources,outputs_sha256={str(p.relative_to(HERE)):sha(p) for p in outputs},
        uncertainty="SE_iid uses independent records. SE_batches uses four independent batch means. MCSE_reported=max(SE_iid,SE_batches), a finite-sample safeguard, not guaranteed tail coverage.",
        limits=["All mean tests are finite-sample diagnostics. Multiple standardized residuals are correlated and do not constitute independent significance tests.",
            "The channel probe is one linear projection onto the initial offdiagonal projector; its expectation is exactly kappa. Matrix entries are not treated as independent observations.",
            "DV_full_lowergap subtracts a deterministic bound per sample, preserving its paired MCSE.",
            "Number-variance corner bounds do not establish an identical entropy corner coefficient.",
            "No asymptotic entropy-corner coefficient or finite-size scaling model is fitted.",
            "Unobserved rare tails can make empirical iid and batch errors optimistic."])
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(dict(status="PASS",records=record_count,maximum_independent_entropy_error=reference_error,maximum_single_state_bound_violation=pure_bound_violation),indent=2))


if __name__=="__main__":main()
