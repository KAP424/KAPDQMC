"""Independent beta=3 extension audit, prepared before final production.

No sampling and no pilot-mean reads. Default execution requires complete final
analysis and production manifests. --self-test uses synthetic arrays only.
Reuses the previously independent, frozen diagnostic implementation, never the
production analyzer. All regressions and predictive tests use 80-digit math.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
import tomllib

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "ai/analysis_beta_patterns/audit_dependencies"))
import mpmath as mp

DATA = ROOT / "analysis/born_scaling/beta3_large"
OUT = DATA / "independent_validation"
OBS = ("SA", "SB", "SC", "D", "SA_minus_SC")
NEW = [27,30,36]
FROZEN_KEEP = {27:2048,30:2560,36:2816}
FROZEN_PLAN_SHA256 = "3835a7f4af98fd0ae41bba5df2b94f96a48f0cbd6703b9d3b1ebfb0b7112bef9"
OLD = list(range(6,25,3))
ALL = OLD+NEW
BLOCKS = (16,32,64,128)
HASHES = {}
ERRORS = {}
COMPARISONS = 0


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def relative(path):
    return Path(os.path.relpath(path,ROOT)).as_posix()


def bind(path, expected=None):
    actual=sha(path)
    assert expected is None or actual==expected, ("SHA mismatch",path)
    HASHES[relative(path)]=actual
    return actual


def filemap(mapping, base=ROOT):
    for name,value in mapping.items():
        bind(base/name,value)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_toml(path):
    return tomllib.loads(Path(path).read_text(encoding="utf-8"))


def read_csv(path):
    bind(path)
    with Path(path).open(encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))


def check(name, actual, expected, atol=3e-12, rtol=3e-10):
    global COMPARISONS
    COMPARISONS+=1
    if expected is None:
        assert actual in (None,""), (name,actual)
        return
    assert actual not in (None,""), (name,actual,expected)
    actual,expected=float(actual),float(expected)
    error=abs(actual-expected)
    ERRORS[name]=max(ERRORS.get(name,0.),error)
    assert math.isfinite(actual) and error<=atol+rtol*abs(expected), (name,actual,expected,error)


def truth(value):
    assert value in (True,False,"True","False"), value
    return value in (True,"True")


def index(rows, fields):
    integer={"L","chain","block_length","draws_per_chain","L_min","L_max","slope_id","slope_id_a","slope_id_b"}
    result={tuple(int(r[k]) if k in integer else r[k] for k in fields):r for r in rows}
    assert len(result)==len(rows), ("duplicate keys",fields)
    return result


def load_independent_diagnostics():
    receipt_path=ROOT/"analysis/born_scaling_production_validation.json"
    receipt=read_json(receipt_path)
    assert receipt["status"]=="PASS"
    path=ROOT/receipt["validator_snapshot"]
    bind(path,receipt["files"][receipt["validator_snapshot"]]);bind(receipt_path)
    spec=importlib.util.spec_from_file_location("independent_old_statistics",path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def compare_diagnostics(row,d):
    for name,value in d.items():
        if name=="initial_screen_pass":
            assert truth(row[name])==value
        else:
            check("diagnostic_"+name,row[name],value)


def design(sizes, terms):
    return mp.matrix([[{"intercept":mp.mpf(1),"L":mp.mpf(L),"logL":mp.log(L),"invL":mp.mpf(1)/L}[t]
                       for t in terms] for L in sizes])


def weighted_fit(sizes,y,se,terms):
    with mp.workdps(80):
        X=design(sizes,terms)
        v=mp.matrix([mp.mpf(str(a)) for a in y])
        s=[mp.mpf(str(a)) for a in se]
        A=mp.matrix([[X[i,j]/s[i] for j in range(X.cols)] for i in range(X.rows)])
        c,_=mp.qr_solve(A,mp.matrix([v[i]/s[i] for i in range(len(y))]))
        cov=(A.T*A)**-1
        residual=v-X*c
        chi=sum((residual[i]/s[i])**2 for i in range(len(y)))
        floating=np.array(A.tolist(),dtype=float)
        return dict(coef=[float(a) for a in c],cov=np.array(cov.tolist(),dtype=float),
                    residual=[float(a) for a in residual],chi=float(chi),
                    rmse=float(mp.sqrt(sum(a*a for a in residual)/len(y))),
                    condition=float(np.linalg.cond(floating/np.linalg.norm(floating,axis=0))))


def self_test():
    L=[12,15,18,21,24,27]
    y=[.7+.23*math.log(l)+1.2/l for l in L]
    fit=weighted_fit(L,y,[.003]*len(L),("intercept","logL","invL"))
    assert max(abs(np.array(fit["coef"])-[.7,.23,1.2]))<1e-12
    H=np.array([[-1/3,1/3,0],[0,-1/3,1/3]])
    C=(H*np.array([.1,.2,.3])**2)@H.T
    assert abs(C[0,1]+.2**2/9)<1e-15
    helper=load_independent_diagnostics()
    for n in FROZEN_KEEP.values():
        t=np.arange(n,dtype=float)
        x=np.array([np.sin((.217+.013*j)*t)+.1*np.cos(.093*t) for j in range(4)])
        d,b=helper.diagnostics(x)
        assert d["draws_per_chain"]==n and math.isfinite(d["MCSE_conservative"])
        assert all(v["discarded_end_draws_per_chain"]==0 and v["eligible"] for v in b)
        checkpoints=sorted({k for k in (128,256,512,1024,2048,4096,8192,16384,32768) if k<=n}|{n})
        assert checkpoints[-1]==n
    print("PASS: synthetic regression/covariance and 2048/2560/2816-draw diagnostic checks; no production or pilot means read")


def load_new_data(summary, protocol):
    old_plan=read_toml(ROOT/"data/born_scaling/production/production_plan.toml")
    used_seeds={j["seed"] for j in old_plan["jobs"]}
    pilot_seeds=set();grouped={};retained_acceptance={};plan_records=[]
    total_records=0
    protocol_path=ROOT/"analysis/born_scaling/beta3_large_protocol.json"
    assert summary["source_stages"]==["production_beta3_large"]
    for stage in summary["source_stages"]:
        assert stage.startswith("production") and stage!="production"
        folder=ROOT/"data/born_scaling"/stage
        manifest_path=folder/f"{stage}_manifest.toml"
        manifest=read_toml(manifest_path);bind(manifest_path)
        assert manifest["status"]=="complete"
        filemap(manifest["files"])
        plan_path=folder/f"{stage}_plan.toml"
        bind(plan_path,manifest["plan_sha256"])
        assert sha(plan_path)==FROZEN_PLAN_SHA256
        plan=read_toml(plan_path)
        assert plan["stage"]==stage and plan["state"]=="staggered" and plan["mu"]==1e-8
        assert plan["update_kernel"]=="fused_rank_updates" and plan["burnin_discarded"]==0
        filemap(plan["source_sha256"])
        protocol_name=relative(protocol_path)
        bound=plan.get("analysis_protocol_sha256") if plan.get("analysis_protocol_path")==protocol_name else plan["source_sha256"].get(protocol_name)
        assert bound==sha(protocol_path)
        # Pilot plans are read only for seed separation and provenance. Their
        # observations and means are neither read nor used in this audit.
        for name,digest in plan.get("pilot_evidence",{}).items():
            path=ROOT/name;bind(path,digest)
            if path.suffix==".toml":
                evidence=read_toml(path)
                if "plan_path" in evidence:
                    pilot_path=ROOT/evidence["plan_path"]
                    bind(pilot_path,evidence["plan_sha256"])
                    pilot_seeds.update(j["seed"] for j in read_toml(pilot_path)["jobs"])
        stage_count=0
        for job in plan["jobs"]:
            assert job["L"] in NEW and job["beta"]==3. and job["mu"]==1e-8
            assert job["seed"] not in used_seeds
            used_seeds.add(job["seed"])
            burn,keep=job["analysis_burnin"],job["retained_sweeps"]
            assert burn==plan["analysis_burnin"] and keep>=2048 and job["sweeps"]==burn+keep
            assert burn==256 and keep==FROZEN_KEEP[job["L"]]
            assert job["measure_every"]==1
            run=job["run_id"]
            receipt_path=folder/f"{run}_receipt.toml"
            receipt=read_toml(receipt_path);bind(receipt_path)
            assert receipt["status"]=="complete" and receipt["job"]==job
            assert receipt["plan_sha256"]==sha(plan_path) and receipt["source_sha256"]==plan["source_sha256"]
            assert receipt["maximum_inverse_drift"]<=plan["max_inverse_drift_tolerance"]
            assert receipt["maximum_logweight_drift"]<=plan["max_logweight_drift_tolerance"]
            filemap(receipt["files"])
            data=[];accepted=proposed=0
            with (folder/f"{run}_samples.csv").open(encoding="utf-8-sig",newline="") as f:
                reader=csv.DictReader(f)
                assert reader.fieldnames==plan["columns"]
                for n,row in enumerate(reader,1):
                    assert int(row["sweep"])==n and int(row["record_index"])==n
                    for field in ("L","chain","seed","measure_every"):
                        assert int(row[field])==job[field]
                    for field in ("beta","mu"):
                        assert float(row[field])==job[field]
                    for field in ("stage","run_id","start"):
                        assert row[field]==job[field]
                    assert row["phase"]==plan["phase"]=="all_recorded_no_burnin_discard"
                    assert int(row["proposed"])==2*job["L"]**2 and 0<=int(row["accepted"])<=int(row["proposed"])
                    val=[float(row[o]) for o in ("SA","SB","SC","D","log_weight")]
                    assert np.isfinite(val).all() and min(val[:3])>=-1e-10
                    check("per_sample_D",val[3],(2*val[0]+val[2])/3-val[1],atol=1e-11)
                    if n>burn:
                        data.append(val);accepted+=int(row["accepted"]);proposed+=int(row["proposed"])
            assert n==job["sweeps"] and len(data)==keep
            initial=read_csv(folder/f"{run}_initial.csv")
            assert len(initial)==1 and int(initial[0]["sweep"])==0
            grouped.setdefault(job["L"],[]).append((job,np.array(data),receipt))
            retained_acceptance[run]=(accepted,proposed)
            stage_count+=n
        assert stage_count==manifest["total_records"]
        total_records+=stage_count
        plan_records.append(dict(stage=stage,path=relative(plan_path),sha256=sha(plan_path),burnin=plan["analysis_burnin"],jobs=len(plan["jobs"])))
    assert not used_seeds & pilot_seeds
    assert sorted(grouped)==NEW
    for L,chains in grouped.items():
        chains.sort(key=lambda a:a[0]["chain"])
        assert len(chains)==4
        assert [j["start"] for j,_,_ in chains]==["random","uniform_A","uniform_B","02_stripe"]
        assert len({j["model_manifest_sha256"] for j,_,_ in chains})==1
        assert len({len(x) for _,x,_ in chains})==1
    return grouped,retained_acceptance,plan_records,total_records


def validate_statistics(groups,acceptance,means,helper):
    tables={name:read_csv(DATA/f"{name}.csv") for name in ("diagnostics","block_errors","region_covariance","cumulative_diagnostics","chain_means","acceptance")}
    diag=index(tables["diagnostics"],("L","observable"))
    blocks=index(tables["block_errors"],("L","observable","block_length"))
    cov=index(tables["region_covariance"],("L","block_length"))
    prefixes=index(tables["cumulative_diagnostics"],("L","observable","draws_per_chain"))
    chains=index(tables["chain_means"],("L","observable","chain"))
    rates=index(tables["acceptance"],("L",))
    assert len(diag)==15 and len(blocks)==60 and len(cov)==12 and len(chains)==60 and len(rates)==3
    prefix_count=0;retained=0
    for L,group in sorted(groups.items()):
        raw=np.stack([x for _,x,_ in group]);burn=group[0][0]["analysis_burnin"]
        retained+=raw.shape[0]*raw.shape[1]
        values=np.concatenate((raw[:,:,:4],(raw[:,:,0]-raw[:,:,2])[:,:,None]),axis=2)
        point=means[(L,)]
        assert point["source"]=="new_production" and int(point["burnin"])==burn
        assert int(point["draws_per_chain"])==raw.shape[1] and int(point["n_chains"])==4
        screens={}
        for oi,observable in enumerate(OBS):
            x=values[:,:,oi];d,b=helper.diagnostics(x)
            compare_diagnostics(diag[(L,observable)],d)
            screens[observable]=d["initial_screen_pass"]
            for field,key in (("mean","mean"),("MCSE","MCSE_conservative"),("rhat_max","rhat_max"),("ESS_bulk","ESS_bulk"),("ESS_tail","ESS_tail")):
                check("new_mean_"+field,point[f"{observable}_{field}"],d[key])
            assert truth(point[f"{observable}_screen_pass"])==screens[observable]
            for block in b:
                row=blocks[(L,observable,block["block_length"])]
                for key,value in block.items():
                    if key=="eligible":assert truth(row[key])==value
                    else:check("block_"+key,row[key],value)
            for ci,(job,_,_) in enumerate(group):
                row=chains[(L,observable,ci+1)]
                assert int(row["seed"])==job["seed"] and row["start"]==job["start"]
                assert int(row["draws"])==len(x[ci]) and int(row["burnin"])==burn
                check("chain_mean",row["mean"],x[ci].mean())
            if observable in ("SA","D"):
                target=.01 if observable=="SA" else .003
                assert truth(point[f"{observable}_precision_target_met"])==(d["MCSE_conservative"]<=target)
                checkpoints=sorted({n for n in (128,256,512,1024,2048,4096,8192,16384,32768) if n<=x.shape[1]}|{x.shape[1]})
                for n in checkpoints:
                    row=prefixes[(L,observable,n)];calc,_=helper.diagnostics(x[:,:n])
                    compare_diagnostics(row,calc)
                    check("prefix_first",row["first_half_mean"],x[:,:n//2].mean())
                    check("prefix_second",row["second_half_mean"],x[:,n//2:n].mean())
                    prefix_count+=1
        assert truth(point["sampling_screen_pass"])==(screens["SA"] and screens["D"])
        for length in BLOCKS:
            count=raw.shape[1]//length
            b=raw[:,:count*length,:3].reshape(4,count,length,3).mean(axis=2).reshape(-1,3)
            centered=b-b.mean(axis=0);C=centered.T@centered/(len(b)*(len(b)-1))
            row=cov[(L,length)]
            for i,a in enumerate(OBS[:3]):
                for j,c in enumerate(OBS[:3]):check("region_covariance",row[f"cov_mean_{a}_{c}"],C[i,j],atol=1e-15)
            w=np.array([2/3,-1,1/3])
            check("D_covariance",row["D_variance_covariance"],w@C@w,atol=1e-15)
            check("D_variance_direct",row["D_variance_direct"],np.var(b@w,ddof=1)/len(b),atol=1e-15)
        ac=[acceptance[j["run_id"]] for j,_,_ in group]
        accepted,proposed=sum(a for a,_ in ac),sum(p for _,p in ac)
        row=rates[(L,)]
        assert int(row["accepted"])==accepted and int(row["proposed"])==proposed
        assert int(row["burnin"])==burn and int(row["retained_sweeps"])==raw.shape[0]*raw.shape[1]
        check("acceptance",row["pooled_acceptance"],accepted/proposed)
        check("mean_acceptance",point["acceptance"],accepted/proposed)
    assert len(prefixes)==prefix_count
    return retained


def validate_fits(means,protocol):
    defs=protocol["D_models"]|protocol["SA_models"]
    fits=index(read_csv(DATA/"fits.csv"),("model","L_min","L_max"))
    residuals=index(read_csv(DATA/"fit_residuals.csv"),("model","L_min","L_max","L"))
    covariances=index(read_csv(DATA/"fit_covariance.csv"),("model","L_min","L_max","term_a","term_b"))
    skipped=index(read_csv(DATA/"skipped_fits.csv"),("model","L_min","L_max"))
    expected_fits=set();expected_skipped=set();residual_count=cov_count=0
    for maximum in (24,max(ALL)):
        for minimum in protocol["L_min_candidates"]:
            sizes=[L for L in ALL if minimum<=L<=maximum]
            for model,terms in defs.items():
                key=(model,minimum,maximum)
                if len(sizes)<len(terms)+protocol["minimum_residual_degrees_of_freedom"]:
                    expected_skipped.add(key)
                    assert skipped[key]["reason"]=="fewer_than_three_residual_dof"
                    continue
                expected_fits.add(key);row=fits[key]
                observable="D" if model.startswith("D_") else "SA"
                y=[float(means[(L,)][f"{observable}_mean"]) for L in sizes]
                se=[float(means[(L,)][f"{observable}_MCSE"]) for L in sizes]
                assert all(e>0 for e in se)
                result=weighted_fit(sizes,y,se,terms)
                df=len(sizes)-len(terms)
                assert int(row["residual_dof"])==df and int(row["n_sizes"])==len(sizes)
                assert row["observable"]==observable and int(row["n_parameters"])==len(terms)
                assert truth(row["sampling_screen_pass"])==all(truth(means[(L,)]["sampling_screen_pass"]) for L in sizes)
                role="primary_large_size_window" if minimum==18 and maximum>24 else "sensitivity_window"
                assert row["window_role"]==role
                for name,val in (("chi_square",result["chi"]),("reduced_chi_square",result["chi"]/df),("residual_rmse",result["rmse"]),("weighted_design_condition",result["condition"])):
                    check("fit_"+name,row[name],val,atol=2e-9)
                assert truth(row["large_fit_residual"])==(result["chi"]/df>3.)
                assert not truth(row["small_window_caution"])
                for i,term in enumerate(terms):
                    check("coefficient",row[f"coefficient_{term}"],result["coef"][i],atol=2e-10)
                    check("coefficient_MCSE",row[f"MCSE_{term}"],math.sqrt(result["cov"][i,i]),atol=2e-10)
                    for j,term2 in enumerate(terms):
                        cr=covariances[key+(term,term2)]
                        check("fit_covariance",cr["covariance"],result["cov"][i,j],atol=2e-12)
                        check("fit_correlation",cr["correlation"],result["cov"][i,j]/math.sqrt(result["cov"][i,i]*result["cov"][j,j]))
                        cov_count+=1
                for i,L in enumerate(sizes):
                    rr=residuals[key+(L,)]
                    check("residual",rr["residual"],result["residual"][i])
                    check("standardized_residual",rr["standardized_residual"],result["residual"][i]/se[i],atol=3e-10)
                    check("fit_input_mean",rr["observed"],y[i]);check("fit_input_MCSE",rr["MCSE"],se[i])
                    residual_count+=1
    assert set(fits)==expected_fits and set(skipped)==expected_skipped
    assert len(residuals)==residual_count and len(covariances)==cov_count
    return len(fits),len(skipped)


def validate_slopes(means):
    slopes=read_csv(DATA/"local_slopes.csv")
    cov_rows=index(read_csv(DATA/"local_slope_covariance.csv"),("slope_id_a","slope_id_b"))
    pairs={"adjacent":list(zip(ALL[:-1],ALL[1:])),"width_six":[(L,L+6) for L in ALL if L+6 in ALL]}
    expected={(kind,axis,a,b) for kind in pairs for axis in ("log","linear") for a,b in pairs[kind]}
    assert {(r["kind"],r["axis"],int(r["L_left"]),int(r["L_right"])) for r in slopes}==expected
    assert len(slopes)==len(expected) and len(cov_rows)==len(slopes)**2
    M=np.array([float(means[(L,)]["D_mean"]) for L in ALL])
    variance=np.array([float(means[(L,)]["D_MCSE"])**2 for L in ALL])
    H=np.zeros((len(slopes),len(ALL)))
    for i,row in enumerate(slopes):
        assert int(row["slope_id"])==i
        a,b=int(row["L_left"]),int(row["L_right"])
        denominator=math.log(b/a) if row["axis"]=="log" else b-a
        H[i,ALL.index(a)]=-1/denominator;H[i,ALL.index(b)]=1/denominator
        check("slope_midpoint",row["L_midpoint"],(a+b)/2)
    C=(H*variance)@H.T;value=H@M
    for i,row in enumerate(slopes):
        check("slope",row["slope"],value[i]);check("slope_MCSE",row["MCSE"],math.sqrt(C[i,i]))
        for j in range(len(slopes)):
            check("slope_covariance",cov_rows[(i,j)]["covariance"],C[i,j],atol=1e-15)
    return len(slopes)


def holdout_test(means):
    frozen=ROOT/"analysis/born_scaling/beta3_large_holdout"
    summary_path=frozen/"summary.json"
    bind(summary_path,"f104eb228c47c27269011074620c579c3ff9de12935220819cfe918cf7608e74")
    summary=read_json(summary_path);filemap(summary["files"])
    assert summary["status"]=="FROZEN"
    predictions={(r["model"],int(r["L"])):r for r in read_csv(frozen/"predictions.csv")}
    covariance={(r["model_i"],int(r["L_i"]),r["model_j"],int(r["L_j"])):r for r in read_csv(frozen/"prediction_covariance.csv")}
    points=[];cov_rows=[];tests=[]
    with mp.workdps(80):
        for model in summary["models"]:
            residual=mp.matrix([mp.mpf(means[(L,)]["D_mean"])-mp.mpf(predictions[(model,L)]["D_prediction_decimal80"]) for L in NEW])
            C=mp.matrix([[mp.mpf(covariance[(model,a,model,b)]["prediction_covariance_decimal80"])
                          +(mp.mpf(means[(a,)]["D_MCSE"])**2 if a==b else 0) for b in NEW] for a in NEW])
            statistic=(residual.T*mp.lu_solve(C,residual))[0]
            for i,L in enumerate(NEW):
                points.append(dict(model=model,L=L,observed=float(means[(L,)]["D_mean"]),
                    new_MCSE=float(means[(L,)]["D_MCSE"]),frozen_prediction=float(predictions[(model,L)]["D_prediction"]),
                    residual=float(residual[i]),prediction_parameter_MCSE=float(predictions[(model,L)]["prediction_parameter_MCSE"]),
                    total_predictive_MCSE=float(mp.sqrt(C[i,i])),standardized_residual=float(residual[i]/mp.sqrt(C[i,i]))))
                for j,L2 in enumerate(NEW):
                    cov_rows.append(dict(model=model,L_i=L,L_j=L2,total_predictive_covariance=float(C[i,j])))
            tests.append(dict(model=model,new_sizes=NEW,statistic=float(statistic),nominal_degrees_of_freedom=len(NEW),
                              statistic_per_dof=float(statistic/len(NEW)),parameters_refitted_to_new_data=False))
    return points,cov_rows,dict(status="COMPUTED",tests=tests,
        formula="delta.T @ inverse(diag(new_MCSE^2)+frozen_parameter_prediction_covariance) @ delta",
        interpretation="Joint predictive discrepancy for two old-data models, not a proof of an asymptotic log/linear law",
        caveats=["Old fits used five points and two residual degrees of freedom; prediction variance excludes model bias",
                 "Nominal chi-square interpretation is approximate with estimated MCSE and finite non-Gaussian chains",
                 "Two models share old and new data, so their test statistics are correlated",
                 "L42 was not sampled by the cost decision; only the predefined 27/30/36 covariance submatrix is used"])


def write_csv(path,rows):
    assert not path.exists(), f"Refuse overwrite of independent result: {path}"
    with path.open("x",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    summary_path=DATA/"summary.json"
    if not summary_path.exists():
        raise RuntimeError("Complete final extension analysis required; pilot means are never used")
    summary=read_json(summary_path)
    assert summary["status"]=="PASS" and summary["new_sizes"]==NEW and summary["sizes"]==ALL
    filemap(summary["files"]);bind(summary_path)
    assert set(summary["analysis_sources"])=={"analyze_born_scaling.py","analyze_born_linear_crossover.py","analyze_born_beta3_large.py"}
    for source in summary["analysis_sources"].values():
        bind(ROOT/source["snapshot"],source["sha256"])
    previous=ROOT/"analysis/born_scaling/production/summary.json"
    bind(previous,"edb5db7350274bd779d6cc30c0396a8fce6ad6077c8a574c0ebd525ee951dea5")
    filemap(read_json(previous)["files"])
    protocol_path=ROOT/"analysis/born_scaling/beta3_large_protocol.json"
    protocol=read_json(protocol_path);bind(protocol_path)
    assert summary["protocol"]==protocol and protocol["minimum_residual_degrees_of_freedom"]==3
    groups,acceptance,plans,total_records=load_new_data(summary,protocol)
    helper=load_independent_diagnostics()
    means=index(read_csv(DATA/"means.csv"),("L",));assert sorted(k[0] for k in means)==ALL
    old_means=index([r for r in read_csv(previous.parent/"means.csv") if float(r["beta"])==3.],("L",))
    for L in OLD:
        row=means[(L,)]
        assert row["source"]=="previous_production"
        for obs in OBS:
            for field in ("mean","MCSE","rhat_max","ESS_bulk","ESS_tail"):
                assert float(row[f"{obs}_{field}"])==float(old_means[(L,)][f"{obs}_{field}"])
    retained=validate_statistics(groups,acceptance,means,helper)
    # Score frozen extrapolations before recalculating combined fits. No old
    # coefficient is refitted or selected by the new point's agreement.
    prediction_points,prediction_covariance,prediction_summary=holdout_test(means)
    fit_count,skipped_count=validate_fits(means,protocol)
    slope_count=validate_slopes(means)
    assert summary["new_chains"]==12 and summary["new_retained_observations"]==retained
    assert retained==29696 and total_records==32768
    assert summary["all_sampling_screen_pass"]==all(truth(r["sampling_screen_pass"]) for r in means.values())
    expected_misses={(L,o) for L in NEW for o,t in (("SA",.01),("D",.003)) if float(means[(L,)][f"{o}_MCSE"])>t}
    assert {(r["L"],r["observable"]) for r in summary["precision_misses"]}==expected_misses
    assert {(r["model"],r["L_min"],r["L_max"]) for r in summary["primary_fits"]}=={(m,18,36) for m in protocol["D_models"]|protocol["SA_models"]}
    check("summary_inverse_drift",summary["maximum_inverse_drift"],max(r["maximum_inverse_drift"] for cs in groups.values() for _,_,r in cs))
    check("summary_logweight_drift",summary["maximum_logweight_drift"],max(r["maximum_logweight_drift"] for cs in groups.values() for _,_,r in cs))
    for row in summary["new_means"]:
        for obs in OBS:
            check("summary_new_mean",row[f"{obs}_mean"],float(means[(row["L"],)][f"{obs}_mean"]))
            check("summary_new_MCSE",row[f"{obs}_MCSE"],float(means[(row["L"],)][f"{obs}_MCSE"]))
    assert not (OUT/"validation.json").exists(), "Independent final receipt already exists; do not silently overwrite"
    OUT.mkdir(parents=True,exist_ok=True)
    write_csv(OUT/"holdout_points.csv",prediction_points)
    write_csv(OUT/"holdout_covariance.csv",prediction_covariance)
    (OUT/"holdout_summary.json").write_text(json.dumps(prediction_summary,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")
    for name in ("holdout_points.csv","holdout_covariance.csv","holdout_summary.json"):
        bind(OUT/name)
    code_sha=sha(Path(__file__))
    snapshot=OUT/f"validate_born_beta3_large_{code_sha}.py"
    if snapshot.exists():assert sha(snapshot)==code_sha
    else:snapshot.write_bytes(Path(__file__).read_bytes())
    bind(snapshot,code_sha)
    result=dict(status="PASS",created_utc=datetime.now(timezone.utc).isoformat(),new_sizes=NEW,new_chains=12,
                retained_observations=retained,raw_new_records=total_records,fit_count=fit_count,skipped_fits=skipped_count,
                local_slopes=slope_count,full_slope_covariance_entries=slope_count**2,
                numeric_comparisons=COMPARISONS,maximum_absolute_differences=ERRORS,precision_decimal_digits=80,
                plans=plans,source_snapshot=relative(snapshot),holdout=prediction_summary,
                analysis_sources=summary["analysis_sources"],
                scope="Independent retained raw means/diagnostics, full slope covariance, 80-digit fits and frozen joint predictive tests; no new MC",
                limitations=["Computation agreement does not prove rare-tail coverage or asymptotic scaling",
                             "Estimated parameter/predictive MCSE excludes model/size-window bias"],files=HASHES)
    (OUT/"validation.json").write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({k:result[k] for k in ("status","new_sizes","new_chains","retained_observations","fit_count","local_slopes","numeric_comparisons","holdout")},indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--self-test",action="store_true")
    args=parser.parse_args()
    self_test() if args.self_test else main()
