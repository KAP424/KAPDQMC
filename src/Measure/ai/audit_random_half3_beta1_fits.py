"""Independent raw-data and high-precision fit audit, without importing main analysis.

python src/Measure/ai/audit_random_half3_beta1_fits.py --stage preliminary
python src/Measure/ai/audit_random_half3_beta1_fits.py --stage final

Uses 80-digit Decimal normal equations with partial-pivot Gaussian elimination.
This deliberately differs from the primary column-scaled Float64 SVD solver.
Only writes analysis_random_half3_beta1_independent/.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import itertools
import json
import math
import tomllib
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal, localcontext
from pathlib import Path

HERE=Path(__file__).resolve().parent
DATA=HERE/"data"
PRIMARY=HERE/"analysis_random_half3_beta1"
OUT=HERE/"analysis_random_half3_beta1_independent"
BITS="100110001011010110"
WINDOWS=((2,6),(4,8),(6,12),(8,16),(10,20),(12,24),(16,32))
SOURCES=("equilibrium_extended.csv","observations.csv","extraction_observations.csv",
         "random_half3_beta1_observations.csv","random_half3_beta1_equilibrium.csv")
D=Decimal

def sha(data):
    return hashlib.sha256(data).hexdigest()

def table(path):
    return list(csv.DictReader(path.open(encoding="utf-8-sig",newline="")))

def savecsv(path,rows):
    columns=list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w",encoding="utf-8",newline="") as file:
        writer=csv.DictWriter(file,fieldnames=columns)
        writer.writeheader(); writer.writerows(rows)

def parse_source(path,stage):
    raw=path.read_bytes()
    complete=raw
    partial=not raw.endswith(b"\n")
    if partial:
        if stage=="final":
            raise AssertionError(f"Unfinished CSV tail: {path}")
        complete=raw[:raw.rfind(b"\n")+1]
    rows=list(csv.DictReader(io.StringIO(complete.decode("utf-8-sig"))))
    return rows,dict(file=path.name,sha256=sha(raw),complete_prefix_sha256=sha(complete),
                     trimmed_partial_tail=partial,total_rows=len(rows))

def load_raw(stage):
    catalogue=table(DATA/"patterns.csv")
    pattern=next(r for r in catalogue if r["pattern"]=="random_half_3")
    assert pattern["occupied_bits"]==BITS and pattern["family"]=="general" and pattern["m"]=="3"
    runs={}
    for path in DATA.glob("random_half3_beta1_run_*.toml"):
        run=tomllib.loads(path.read_text(encoding="utf-8"))
        assert run["pattern_bits"]==BITS and run["beta"]==1
        assert "single-species natural-log Renyi2" in run["protocol"]
        assert "origin=(1,1)" in run["protocol"]
        if stage=="final":
            assert run["status"]=="complete"
        for name,digest in run["code_sha256"].items():
            assert sha((HERE/name).read_bytes())==digest,(path,name)
        runs[run["run_id"]]=run
    journal={}
    with (DATA/"extraction_provenance.tsv").open(encoding="utf-8-sig",newline="") as file:
        for row in csv.DictReader(file,delimiter="\t"):
            if row["pattern"]=="random_half_3" and D(row["beta"])==1:
                assert row["pattern_bits"]==BITS
                assert sha(row["observation_csv"].encode())==row["row_sha256"]
                journal[(int(row["L"]),float(row["theta"]))]=row
    merged={}; receipts=[]; duplicate_count=0; duplicate_error=D(0)
    identity_error=D(0); delta_error=D(0)
    for name in SOURCES:
        rawrows,receipt=parse_source(DATA/name,stage)
        selected=0
        for line,row in enumerate(rawrows,2):
            if row["pattern"]!="random_half_3" or D(row["beta"])!=1:
                continue
            selected+=1
            assert row["family"]=="general" and int(row["m"])==3
            size=int(row["L"]); theta=float(row["theta"])
            assert size%9==0 and theta>=0
            key=(size,theta)
            values={name:D(row[name]) for name in ("S_A","S_C","S_B","D","delta")}
            assert all(x.is_finite() for x in values.values())
            assert min(values[name] for name in ("S_A","S_C","S_B"))>=0
            reconstructed=(2*values["S_A"]+values["S_C"])/3-values["S_B"]
            identity_error=max(identity_error,abs(values["D"]-reconstructed))
            delta_error=max(delta_error,abs(values["delta"]-values["S_A"]+values["S_B"]))
            limit=D(row.get("premeasurement_excess") or row.get("premeasurement_limit_excess") or row["accessible_excess"])
            assert abs(limit)<D("1e-8")
            assert row.get("premeasurement_ground","true")=="true"
            if name.startswith("random_half3_beta1_"):
                run=runs[row["run_id"]]
                assert row["code_sha256"]==run["code_sha256"]["random_half3_beta1_experiments.jl"]
                assert D(row["rank_tol"])==D("1e-10")
                assert row["entropy_method"] in ("cholesky","eigen")
                assert size in run["L"] and theta in [float(t) for t in run["theta"]]
            elif name=="extraction_observations.csv":
                assert key in journal
                recorded=next(csv.DictReader(io.StringIO(
                    ",".join(rawrows[0].keys())+"\n"+journal[key]["observation_csv"])))
                assert all(D(recorded[k])==values[k] for k in values)
            item=dict(L=size,theta=theta,source=name,source_row=line,D_reconstructed=reconstructed,**values)
            if key in merged:
                duplicate_count+=1
                error=max(abs(values[k]-merged[key][k]) for k in values)
                duplicate_error=max(duplicate_error,error)
                assert error<D("1e-9"),(key,error)
            else:
                merged[key]=item
        receipts.append(dict(**receipt,selected_rows=selected))
    assert identity_error<D("1e-9") and delta_error<D("1e-9")
    return merged,receipts,dict(duplicate_rows=duplicate_count,max_duplicate_entropy_error=float(duplicate_error),
        max_D_reconstruction_error=float(identity_error),max_delta_reconstruction_error=float(delta_error),
        configuration_bits=BITS,run_ids=sorted(runs),all_selected_rows_general_m3_beta1=True)

def solve_elimination(matrix,rhs):
    # Only 2x2 through 5x5 systems, at 80 digits; no double precision solver involved.
    n=len(rhs); aug=[list(matrix[i])+[rhs[i]] for i in range(n)]
    for col in range(n):
        pivot=max(range(col,n),key=lambda row:abs(aug[row][col]))
        aug[col],aug[pivot]=aug[pivot],aug[col]
        assert aug[col][col]!=0
        for row in range(col+1,n):
            factor=aug[row][col]/aug[col][col]
            for j in range(col,n+1):
                aug[row][j]-=factor*aug[col][j]
    result=[D(0)]*n
    for i in reversed(range(n)):
        result[i]=(aug[i][-1]-sum(aug[i][j]*result[j] for j in range(i+1,n)))/aug[i][i]
    return result

def parameter_names(observable,model):
    names=["a"] if observable=="S_B" else []
    names.extend(("c","b"))
    if "inv" in model:
        names.append("d")
    if model.endswith("inv2"):
        names.append("e")
    return names

def high_precision_fit(points,observable,axis,model):
    rawB=observable=="S_B"
    names=parameter_names(observable,model)
    assert len(points)>=len(names)+2
    with localcontext() as context:
        context.prec=80
        x=[D(str(row[axis])) for row in points]
        y=[row[observable] for row in points]
        design=[]
        for v in x:
            row=[v] if rawB else []
            row.extend((-v.ln() if rawB else v.ln(),D(1)))
            if "inv" in model:
                row.append(1/v)
            if model.endswith("inv2"):
                row.append(1/v**2)
            design.append(row)
        count=len(names)
        gram=[[sum(r[i]*r[j] for r in design) for j in range(count)] for i in range(count)]
        rhs=[sum(row[i]*value for row,value in zip(design,y)) for i in range(count)]
        coef=solve_elimination(gram,rhs)
        residual=[value-sum(v*c for v,c in zip(row,coef)) for row,value in zip(design,y)]
        normal_error=max(abs(sum(row[j]*res for row,res in zip(design,residual))) for j in range(count))
        return dict(n=len(points),grid=",".join(format(float(v),"g") for v in x),
            **dict(zip(names,map(float,coef))),rmse=float((sum(r*r for r in residual)/len(x)).sqrt()),
            max_abs_residual=float(max(map(abs,residual))),normal_equation_residual=float(normal_error))

def all_fits(merged,plan):
    eq=sorted((r for (L,t),r in merged.items() if math.isinf(t)),key=lambda r:r["L"])
    finite={key:r for key,r in merged.items() if 2<=key[1]<=32}
    maxL=max(r["L"] for r in eq)
    output=[]; coverage=[]
    for upper in sorted({90,maxL}):
        for lower in plan["equilibrium_Lmin_checks"]:
            points=[r for r in eq if lower<=r["L"]<=upper]
            for obs in ("D","S_B"):
                prefix="area_" if obs=="S_B" else ""
                for model in (prefix+name for name in ("log","log_inv","log_inv2")):
                    if len(points)>=len(parameter_names(obs,model))+2:
                        result=high_precision_fit(points,obs,"L",model)
                        derived=high_precision_fit(points,"D_reconstructed","L",model) if obs=="D" else result
                        output.append(dict(domain="equilibrium",observable=obs,Lmin=lower,Lmax=upper,
                            fit_scope="equilibrium",model=model,c_from_reconstructed_D=derived["c"],**result))
    sizes=sorted({L for L,t in finite})
    for wi,(lo,hi) in enumerate(WINDOWS):
        grids={L:{t for size,t in finite if size==L and lo<=t<=hi} for L in sizes}
        complete={L:grid for L,grid in grids.items() if {lo,hi}<=grid and len(grid)>=5}
        pairs=[]
        for small,large in itertools.combinations(sorted(complete),2):
            common=complete[small]&complete[large]
            if len(common)>=5 and {lo,hi}<=common:
                pairs.append((large,small,common))
        chosen=max(pairs,key=lambda p:(p[0],p[1])) if pairs else None
        coverage.append(dict(window=f"{lo}-{hi}",complete_sizes=",".join(map(str,sorted(complete))),
            size_small=chosen[1] if chosen else "",size_large=chosen[0] if chosen else "",
            common_grid=",".join(format(t,"g") for t in sorted(chosen[2])) if chosen else ""))
        for L,grid in complete.items():
            for model in ("log","log_inv","log_inv2"):
                if len(grid)<len(parameter_names("D",model))+2:
                    continue
                full=high_precision_fit([finite[L,t] for t in sorted(grid)],"D","theta",model)
                output.append(dict(domain="time",observable="D",window_index=wi,window=f"{lo}-{hi}",L=L,
                    fit_scope="per_size_full_grid",model=model,**full))
        if chosen:
            large,small,grid=chosen
            for L in complete:
                if not grid<=complete[L]:
                    continue
                points=[finite[L,t] for t in sorted(grid)]
                for model in ("log","log_inv","log_inv2"):
                    if len(points)<len(parameter_names("D",model))+2:
                        continue
                    fitted=high_precision_fit(points,"D","theta",model)
                    derived=high_precision_fit(points,"D_reconstructed","theta",model)
                    output.append(dict(domain="time",observable="D",window_index=wi,window=f"{lo}-{hi}",L=L,
                        fit_scope="largest_pair_common_grid",model=model,primary_L=large,
                        comparison_small=small,c_from_reconstructed_D=derived["c"],**fitted))
    return output,coverage

def compare_primary(fits,coverage,merged,stage,receipts):
    primary_summary=json.loads((PRIMARY/"summary.json").read_text(encoding="utf-8"))
    sourcehash={Path(r["path"]).name:r["sha256"] for r in primary_summary["sources"]}
    stale=[r["file"] for r in receipts if sourcehash.get(r["file"])!=r["sha256"]]
    checks=[]; missing=[]
    eq={(r["observable"],int(r["requested_Lmin"]),int(r["requested_Lmax"]),r["model"]):r
        for r in table(PRIMARY/"equilibrium_fits.csv")}
    full={(r["window"],int(r["L"]),r["model"]):r for r in table(PRIMARY/"time_full_grid_fits.csv")}
    common={(r["window"],int(r["L"]),int(r["primary_L"]),int(r["comparison_small"]),r["model"]):r
        for r in table(PRIMARY/"time_common_grid_fits.csv")}
    for fit in fits:
        if fit["domain"]=="equilibrium":
            match=eq.get((fit["observable"],fit["Lmin"],fit["Lmax"],fit["model"]))
        elif fit["fit_scope"]=="per_size_full_grid":
            match=full.get((fit["window"],fit["L"],fit["model"]))
        else:
            match=common.get((fit["window"],fit["L"],fit["primary_L"],fit["comparison_small"],fit["model"]))
        identity={k:v for k,v in fit.items() if k in ("domain","observable","Lmin","Lmax","window","L","fit_scope","model")}
        if match is None or fit["grid"]!=match["grid"]:
            missing.append(dict(**identity,reason="primary row absent or different grid (possibly older snapshot)"))
            continue
        fields=[k for k in ("a","c","b","d","e","rmse","max_abs_residual") if k in fit]
        differences={k:abs(fit[k]-float(match["corner_c"] if k=="c" and fit["observable"]=="S_B" else match[k])) for k in fields}
        # Inverse-square coefficients carry x^2 units: their unscaled error can
        # be larger than the actual fitted-curve error on a high-L interval.
        # Bound the full prediction error without relying on cancellation.
        grid=[float(x) for x in fit["grid"].split(",")]
        prediction_bound=max(differences.get("a",0)*x+differences["c"]*abs(math.log(x))+
            differences["b"]+differences.get("d",0)/x+differences.get("e",0)/x**2 for x in grid)
        assert differences["c"]<1e-8 and prediction_bound<1e-8,(identity,differences,prediction_bound)
        assert differences["rmse"]<1e-9 and differences["max_abs_residual"]<1e-9
        checks.append(dict(**identity,grid=fit["grid"],independent_c=fit["c"],
            primary_c=float(match["corner_c"] if fit["observable"]=="S_B" else match["c"]),
            max_coefficient_or_residual_error=max(differences.values()),c_error=differences["c"],
            rmse_error=differences["rmse"],prediction_error_bound=prediction_bound,
            primary_scaled_design_condition=float(match["condition"])))
    selected=table(PRIMARY/"selected_observations.csv")
    for row in selected:
        key=int(row["L"]),float(row["theta"])
        assert key in merged
        assert all(abs(D(row[k])-merged[key][k])<D("1e-9") for k in ("S_A","S_C","S_B","D"))
    primary_coverage={r["window"]:r for r in table(PRIMARY/"time_window_coverage.csv")}
    coverage_errors=[]
    for row in coverage:
        p=primary_coverage[row["window"]]
        if any(str(row[k])!=p.get(k,"") for k in ("size_small","size_large","common_grid")):
            coverage_errors.append(row["window"])
    if stage=="final":
        assert primary_summary["stage"]=="final"
        assert not stale and not missing and not coverage_errors,(stale,missing,coverage_errors)
        assert len(selected)==len(merged)
        assert primary_summary["script_sha256"]==sha((HERE/"analyze_random_half3_beta1.py").read_bytes())
    return checks,dict(primary_stage=primary_summary["stage"],primary_stale_sources=stale,
        primary_selected_rows_verified=len(selected),unmatched_fit_rows=missing,
        coverage_windows_differing_from_primary=coverage_errors,
        primary_plan_sha256=primary_summary["plan_sha256"],
        primary_script_sha256=sha((HERE/"analyze_random_half3_beta1.py").read_bytes()),
        primary_summary_sha256=sha((PRIMARY/"summary.json").read_bytes()))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=("preliminary","final"),default="preliminary")
    args=parser.parse_args()
    planraw=(PRIMARY/"plan.json").read_bytes(); plan=json.loads(planraw)
    assert tuple(map(tuple,plan["time_windows"]))==WINDOWS
    merged,receipts,identity=load_raw(args.stage)
    required={(L,float(t)) for L in plan["planned_finite_sizes"] for t in plan["planned_times"]}
    for field in ("adaptive_finite_grids","adaptive_sampling_grids"):
        for addition in plan.get(field,[]):
            required|={(L,float(t)) for L in addition["sizes"] for t in addition["times"]}
    required|={(L,math.inf) for L in plan["planned_new_equilibrium_sizes"]+plan.get("adaptive_equilibrium_sizes",[])}
    missing_keys=sorted(required-merged.keys())
    if args.stage=="final":
        assert not missing_keys,missing_keys
    fits,coverage=all_fits(merged,plan)
    comparisons,primary=compare_primary(fits,coverage,merged,args.stage,receipts)
    if args.stage=="final":
        assert primary["primary_plan_sha256"]==sha(planraw)
    changed=[r["file"] for r in receipts if sha((DATA/r["file"]).read_bytes())!=r["sha256"]]
    if args.stage=="final":
        assert not changed
    OUT.mkdir(exist_ok=True)
    savecsv(OUT/"recomputed_fits.csv",fits)
    savecsv(OUT/"fit_comparisons.csv",comparisons)
    savecsv(OUT/"common_grid_coverage.csv",coverage)
    savecsv(OUT/"raw_selected_observations.csv",[merged[key] for key in sorted(merged)])
    report=dict(status="PASS" if args.stage=="final" else "PRELIMINARY_PASS",stage=args.stage,
        utc=datetime.now(timezone.utc).isoformat(),algorithm="80-digit Decimal normal equations, pivoted elimination; raw B uses explicit negative log column",
        script_sha256=sha(Path(__file__).read_bytes()),plan_sha256=sha(planraw),identity=identity,sources=receipts,
        sources_changed_during_audit=changed,raw_unique_rows=len(merged),fitted_rows=len(fits),
        compared_fit_rows=len(comparisons),missing_required_keys=[(L,str(t)) for L,t in missing_keys],
        max_compared_c_error=max(r["c_error"] for r in comparisons),
        max_compared_coefficient_or_residual_error=max(r["max_coefficient_or_residual_error"] for r in comparisons),
        max_compared_prediction_error_bound=max(r["prediction_error_bound"] for r in comparisons),
        max_c_change_from_reconstructing_D=max(abs(r["c"]-r["c_from_reconstructed_D"]) for r in fits if "c_from_reconstructed_D" in r),
        primary=primary,notes=["No entropy calculation or source-data changes.",
        "Checks implementation and identity only; does not validate the physical asymptotic fit ansatz or create statistical error bars."])
    bymodel=defaultdict(list)
    for row in comparisons:
        bymodel[(row["domain"],row["model"])].append(row)
    report["comparison_by_model"]=[dict(domain=domain,model=model,rows=len(rows),
        max_c_error=max(r["c_error"] for r in rows),
        max_prediction_error_bound=max(r["prediction_error_bound"] for r in rows))
        for (domain,model),rows in sorted(bymodel.items())]
    report["output_sha256"]={name:sha((OUT/name).read_bytes()) for name in
        ("recomputed_fits.csv","fit_comparisons.csv","common_grid_coverage.csv","raw_selected_observations.csv")}
    (OUT/"audit.json").write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps({key:report[key] for key in ("status","raw_unique_rows","fitted_rows","compared_fit_rows","max_compared_c_error","max_compared_coefficient_or_residual_error","missing_required_keys")},indent=2))

if __name__=="__main__":
    main()
