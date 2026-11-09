"""Independent beta-zero pilot/confirmation diagnostics, with frozen provenance.

No equilibrium value or literature target participates in the time-side ranking
or eligibility checks. Partial and sparse grids remain explicitly unresolved.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import io
import itertools
import json
import math
from pathlib import Path
import tomllib

import numpy as np

from audit_beta0_inventory import independent_fit, raw_slopes, grid, dump_csv, dump_json

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"analysis_beta0_study"
FIGURES=ROOT/"figs_beta0_study"
WINDOWS=((4.,7.),(4.,8.),(4.,9.),(4.5,9.),(4.,8.5),(5.,9.),(5.,10.))
SIZES=(72,108,144,180,216)
THRESHOLDS=dict(size_drift=.02,D_difference_to_fitted_log_span=.01,model_drift=.05,window_drift=.05)
FILES={}


def sha(data):return hashlib.sha256(data).hexdigest()


def tracked(path):
    data=path.read_bytes();FILES[path.relative_to(ROOT).as_posix()]=sha(data);return data


def snapshot(path,data=None):
    data=path.read_bytes() if data is None else data
    directory=OUT/"source_snapshots";directory.mkdir(exist_ok=True)
    saved=directory/f"{path.stem}_{sha(data)[:16]}{path.suffix}"
    if saved.exists() and saved.read_bytes()!=data:raise ValueError("Snapshot collision")
    if not saved.exists():saved.write_bytes(data)
    tracked(saved)
    return dict(original_path=path.relative_to(ROOT).as_posix(),sha256=sha(data),snapshot=saved.relative_to(ROOT).as_posix())


def half_grid(lo,hi):return [x/2 for x in range(round(lo*2),round(hi*2)+1)]


def ratio(value,denominator):return abs(value)/abs(denominator) if abs(denominator)>1e-15 else None


def read_confirmation(stage):
    path=OUT/"confirmation_plan.json"
    if not path.exists():
        if stage=="final":raise ValueError("Final analysis requires the frozen confirmation plan")
        return {}
    plan=json.loads(tracked(path))
    for relative,expected in plan["sources"].items():
        if sha(tracked(ROOT/relative))!=expected:raise ValueError(f"Frozen confirmation source changed: {relative}")
    if tuple(tuple(float(v) for v in w) for w in plan["windows"])!=WINDOWS:raise ValueError("Analysis windows disagree with frozen plan")
    primary=plan["primary"]
    if (primary["L"],primary["comparison_L"],primary["theta_min"],primary["theta_max"],primary["theta_step"],primary["model"])!=(216,180,4,9,.5,"log_inv1_inv2"):
        raise ValueError("Analysis primary differs from frozen plan")
    if plan["model_orders"]!=[1,2,3] or plan["order3_minimum_points"]!=11:raise ValueError("Analysis model rule differs from frozen plan")
    expected=dict(theta_min=4,theta_over_L_max=.05,size_coefficient_drift=.02,
        D_difference_over_fitted_log_span=.01,model_drift=.05,window_drift=.05,
        equilibrium_sensitivity=.05,postselection_relative_eq_time_gap=.05)
    if plan["thresholds"]!=expected:raise ValueError("Analysis thresholds differ from frozen plan")
    return plan


def read_inputs(stage):
    design=json.loads(tracked(OUT/"study_design.json"))
    for relative,expected in design["sources"].items():
        if sha(tracked(ROOT/relative))!=expected:raise ValueError(f"Design source changed: {relative}")
    provenance=json.loads(tracked(OUT/"baseline_provenance.json"))
    basebytes=tracked(OUT/"baseline_observations.csv")
    if sha(basebytes)!=provenance["baseline_sha256"]:raise ValueError("Frozen baseline changed")
    metadata=provenance["patterns"]
    aliases={name:canonical for canonical,meta in metadata.items() for name in meta["aliases"]}
    observations={};sources={};duplicates=[];live_sources=[];manifests={}
    for row in csv.DictReader(io.StringIO(basebytes.decode("utf-8-sig"))):
        name=row["canonical_pattern"];size=int(row["L"]);theta=float(row["theta"])
        row.update(pattern=name,L=size,theta=theta,m=int(row["m"]),beta=0.,
            ground_reachable=row["ground_reachable"].lower()=="true")
        for key in ("S_A","S_C","S_B","D"):row[key]=float(row[key])
        observations[name,size,theta]=row
        sources[row["source_id"]]=dict(baseline_point_id=row["source_id"],baseline_provenance_sha256=FILES["analysis_beta0_study/baseline_provenance.json"],new_sources=[])
    newpath=ROOT/"data/beta0_study_observations.csv"
    if newpath.exists():
        data=newpath.read_bytes()
        if not data.endswith(b"\n"):raise ValueError("New CSV has an incomplete trailing row; retry after writer flush")
        raw=csv.DictReader(io.StringIO(data.decode("utf-8-sig")));rows=list(raw);lines=data.decode("utf-8-sig").splitlines()
        ledgerpath=newpath.with_name(newpath.stem+"_provenance.tsv")
        ledgerbytes=ledgerpath.read_bytes();ledger=list(csv.DictReader(io.StringIO(ledgerbytes.decode()),delimiter="\t"))
        bynumber={int(r["output_row"]):r for r in ledger}
        if len(bynumber)!=len(ledger):raise ValueError("Duplicate ledger row numbers")
        live_sources.extend([snapshot(newpath,data),snapshot(ledgerpath,ledgerbytes)])
        for number,row in enumerate(rows,2):
            if number not in bynumber:raise ValueError(f"Missing ledger row {number}; retry after journal flush")
            journal=bynumber[number];rawline=lines[number-1]
            if journal["csv_row"]!=rawline or journal["row_sha256"]!=sha(rawline.encode()):raise ValueError("CSV row/ledger mismatch")
            runid=row["run_id"]
            if runid not in manifests:
                manifestpath=newpath.with_name(newpath.stem+"_run_"+runid+".toml")
                rawmanifest=manifestpath.read_bytes();manifest=tomllib.loads(rawmanifest.decode())
                if manifest["run_id"]!=runid or manifest["mode"]!="run":raise ValueError("Wrong run manifest")
                if stage=="final" and manifest["status"]!="complete":raise ValueError("Final analysis requires completed source runs")
                if "exact spectral-support" not in manifest["protocol"]:raise ValueError("Spectral support not declared")
                for relative,expected in manifest["code_sha256"].items():
                    source=Path(manifest.get("code_snapshots",{}).get(relative,ROOT/"data/provenance/code"/expected/relative))
                    if sha(tracked(source))!=expected:raise ValueError("Source-code snapshot mismatch")
                for kind in ("jobs","plan"):
                    original=manifest.get("job_file" if kind=="jobs" else "plan_file")
                    expected=manifest.get("job_file_sha256" if kind=="jobs" else "plan_sha256")
                    if original and expected:
                        saved=Path(manifest.get(kind+"_snapshot",ROOT/"data/provenance"/kind/expected/Path(original).name))
                        if sha(tracked(saved))!=expected:raise ValueError("Request snapshot mismatch")
                info=snapshot(manifestpath,rawmanifest);live_sources.append(info)
                manifests[runid]=dict(manifest=manifest,snapshot=info["snapshot"])
            manifest=manifests[runid]["manifest"]
            if row["code_sha256"]!=manifest["code_sha256"]["beta0_study_experiments.jl"]:raise ValueError("Row runner hash mismatch")
            if row["job_file_sha256"]!=manifest["job_file_sha256"]:raise ValueError("Row job hash mismatch")
            if row["projection_method"]!="exact_spectral_support" or float(row["rank_tol"])!=1e-10:raise ValueError("Wrong projection convention")
            name=aliases.get(row["pattern"])
            if name is None:raise ValueError(f"Unknown pattern: {row['pattern']}")
            meta=metadata[name];size=int(row["L"]);theta=float(row["theta"])
            if float(row["beta"])!=0 or int(row["m"])!=meta["m"] or row["occupied_bits"]!=meta["occupied_bits"]:raise ValueError("Pattern identity/beta mismatch")
            if int(row["class_id"])!=meta["occupancy_class"]:raise ValueError("Wrong occupancy class")
            reachable=row["ground_reachable"].lower()=="true"
            if reachable!=meta["ground_reachable"] or row["premeasurement_ground"].lower()!=str(reachable).lower():raise ValueError("Reachability metadata mismatch")
            if row["limit_state"]!=("ground_state" if reachable else "accessible_excited_limit"):raise ValueError("Limit-state label mismatch")
            values={key:float(row[key]) for key in ("S_A","S_C","S_B","D")}
            if not all(math.isfinite(value) for value in values.values()):raise ValueError("Nonfinite entropy")
            if abs(values["D"]-((2*values["S_A"]+values["S_C"])/3-values["S_B"]))>1e-8:raise ValueError("D identity mismatch")
            ident=f"{name}|{size}|{'Inf' if math.isinf(theta) else f'{theta:g}'}";key=(name,size,theta)
            evidence=dict(original_pattern=row["pattern"],source_snapshot=live_sources[0]["snapshot"],row=number,row_sha256=journal["row_sha256"],run_id=runid,manifest_snapshot=manifests[runid]["snapshot"])
            if key in observations:
                error=max(abs(observations[key][field]-values[field]) for field in values)
                duplicates.append(dict(pattern=name,L=size,theta=theta,max_abs_difference=error,consistent=error<=1e-8,new_row=number,source_id=ident))
                if error>1e-8:raise ValueError(f"Conflicting repeated entropy point: {key}")
            else:
                observations[key]=dict(canonical_pattern=name,pattern=name,m=meta["m"],L=size,theta=theta,beta=0.,
                    **values,ground_reachable=reachable,occupied_bits=meta["occupied_bits"],source_id=ident)
                sources[ident]=dict(baseline_point_id=None,new_sources=[])
            sources[ident]["new_sources"].append(evidence)
    return sorted(observations.values(),key=lambda r:(r["pattern"],r["L"],r["theta"])),metadata,design,sources,duplicates,live_sources,manifests


def analyze_points(observations,metadata,names):
    indexed=defaultdict(dict);limits=defaultdict(list)
    for row in observations:
        if row["pattern"] not in names:continue
        if math.isinf(row["theta"]):limits[row["pattern"]].append(row)
        else:indexed[row["pattern"],row["L"]][row["theta"]]=row
    coverage=[];timefits=[];limitfits=[];paired=[];pointdiffs=[]
    for name in names:
        for size in SIZES:
            points=indexed[name,size]
            for lo,hi in WINDOWS:
                required=half_grid(lo,hi);present=[t for t in required if t in points];missing=sorted(set(required)-set(present))
                coverage.append(dict(pattern=name,L=size,window=f"{lo:g}-{hi:g}",required_grid=grid(required),present_grid=grid(present),missing_grid=grid(missing),
                    complete=not missing,n=len(present),theta_over_L_max=hi/size,scale_screen=hi/size<=.05,
                    all_actual_theta=grid(points)))
                chosen=[points[t] for t in present]
                for order in (1,2,3):
                    fit=independent_fit(chosen,"theta",order)
                    if fit:
                        formal=not missing and (order<3 or len(chosen)>=11)
                        timefits.append(dict(pattern=name,L=size,window=f"{lo:g}-{hi:g}",requested_min=lo,requested_max=hi,
                            complete_grid=not missing,formal_robustness_eligible=formal,scale_screen=hi/size<=.05,
                            warning="incomplete_grid" if missing else "five_parameter_fit_has_fewer_than_11_points" if order==3 and len(chosen)<11 else "",
                            **fit))
        lower_bounds=(36,48,72) if metadata[name]["m"]==4 else (36,54,72)
        for lower,upper,order in itertools.product(lower_bounds,(144,216),(1,2)):
            points=[r for r in limits[name] if lower<=r["L"]<=upper]
            fit=independent_fit(points,"L",order)
            if fit:limitfits.append(dict(pattern=name,requested_min=lower,requested_max=upper,
                has_requested_upper=any(r["L"]==upper for r in points),state_role="equilibrium_ground",**fit))
        for smaller,larger in itertools.combinations(SIZES,2):
            a,b=indexed[name,smaller],indexed[name,larger]
            for theta in sorted(set(a)&set(b)):
                pointdiffs.append(dict(pattern=name,L_small=smaller,L_large=larger,theta=theta,D_small=a[theta]["D"],D_large=b[theta]["D"],D_difference=b[theta]["D"]-a[theta]["D"]))
            for lo,hi in WINDOWS:
                required=half_grid(lo,hi);common=[t for t in required if t in a and t in b]
                if len(common)<2:continue
                small=[a[t] for t in common];large=[b[t] for t in common]
                delta=np.array([y["D"]-x["D"] for x,y in zip(small,large)])
                span=float(np.ptp([r["D"] for r in large]));logrange=math.log(common[-1]/common[0])
                row=dict(pattern=name,L_small=smaller,L_large=larger,window=f"{lo:g}-{hi:g}",grid=grid(common),n=len(common),
                    complete_grid=common==required,scale_screen_both=hi/smaller<=.05,
                    raw_D_span_large=span,max_abs_D_difference=float(max(abs(delta))),
                    raw_size_difference_over_D_span=ratio(max(abs(delta)),span),
                    raw_change_of_size_difference_over_D_span=ratio(np.ptp(delta),span),
                    raw_endpoint_slope_small=(small[-1]["D"]-small[0]["D"])/logrange,
                    raw_endpoint_slope_large=(large[-1]["D"]-large[0]["D"])/logrange)
                for order in (1,2,3):
                    fa,fb=independent_fit(small,"theta",order),independent_fit(large,"theta",order)
                    if fa and fb:row.update({f"order{order}_c_small":fa["c_fit"],f"order{order}_c_large":fb["c_fit"],
                        f"order{order}_relative_size_drift":ratio(fb["c_fit"]-fa["c_fit"],fb["c_fit"]),
                        f"order{order}_D_difference_to_fitted_log_span":ratio(max(abs(delta)),fb["c_fit"]*logrange)})
                paired.append(row)
    return coverage,timefits,limitfits,paired,pointdiffs


def candidate_ranking(names,timefits,paired,slopes):
    """Exploratory rank uses the same L72/108 and theta4-9 for every pattern."""
    result=[]
    for name in names:
        pair=next((r for r in paired if r["pattern"]==name and r["L_small"]==72 and r["L_large"]==108 and r["window"]=="4-9"),None)
        primary=next((r for r in timefits if r["pattern"]==name and r["L"]==108 and r["window"]=="4-9" and r["inverse_order"]==2),None)
        models=[r for r in timefits if r["pattern"]==name and r["L"]==108 and r["window"]=="4-9" and r["formal_robustness_eligible"]]
        windows=[r for r in timefits if r["pattern"]==name and r["L"]==108 and r["inverse_order"]==2 and r["complete_grid"]]
        local=[r for r in slopes if r["pattern"]==name and r["domain"]=="time" and r["L"]==108 and r["pair_kind"]=="adjacent" and r["x_low"]>=4 and r["x_high"]<=9]
        c=primary["c_fit"] if primary else None
        modeldrift=max((ratio(r["c_fit"]-c,c) for r in models),default=None) if c else None
        windowdrift=max((ratio(r["c_fit"]-c,c) for r in windows),default=None) if c else None
        vals=[r["log_slope"] for r in local]
        scoreparts=[pair.get("raw_size_difference_over_D_span") if pair else None,modeldrift,windowdrift]
        score=max(scoreparts[0]/.02,scoreparts[1]/.05,scoreparts[2]/.05) if all(v is not None for v in scoreparts) else None
        result.append(dict(pattern=name,pilot_L_small=72,pilot_L_large=108,pilot_window="4-9",
            pilot_common_complete=pair["complete_grid"] if pair else False,
            pilot_score=score,pilot_score_definition="max(raw size difference / D span /0.02, model drift/0.05, window drift/0.05); exploratory ordering only",
            raw_size_difference_over_D_span=scoreparts[0],model_drift=modeldrift,window_drift=windowdrift,
            c_time_fit=c,raw_local_slope_min=min(vals,default=None),raw_local_slope_max=max(vals,default=None),
            raw_local_slope_first=vals[0] if vals else None,raw_local_slope_last=vals[-1] if vals else None,
            model_count=len(models),window_count=len(windows),scale_screen=False,
            status="exploratory_candidate_not_confirmed" if pair and pair["complete_grid"] else "incomplete_pilot_unresolved",
            equilibrium_used_for_ranking=False))
    result.sort(key=lambda r:(not r["pilot_common_complete"],r["pilot_score"] is None,r["pilot_score"] if r["pilot_score"] is not None else math.inf,r["pattern"]))
    for rank,row in enumerate(result,1):row["exploratory_rank"]=rank
    return result


def eligibility(names,timefits,limitfits,paired,confirmation):
    result=[]
    for name in names:
        fit=next((r for r in timefits if r["pattern"]==name and r["L"]==216 and r["window"]=="4-9" and r["inverse_order"]==2),None)
        pair=next((r for r in paired if r["pattern"]==name and r["L_small"]==180 and r["L_large"]==216 and r["window"]=="4-9"),None)
        models=[r for r in timefits if r["pattern"]==name and r["L"]==216 and r["window"]=="4-9" and r["formal_robustness_eligible"]]
        windows=[r for r in timefits if r["pattern"]==name and r["L"]==216 and r["inverse_order"]==2 and r["complete_grid"] and r["scale_screen"]]
        c=fit["c_fit"] if fit else None
        modeldrift=max((ratio(r["c_fit"]-c,c) for r in models),default=None) if c else None
        windowdrift=max((ratio(r["c_fit"]-c,c) for r in windows),default=None) if c else None
        third=[next((r for r in paired if r["pattern"]==name and r["L_small"]==small and r["L_large"]==large and r["window"]=="4-7"),None) for small,large in ((144,180),(180,216))]
        third_complete=all(r and r["complete_grid"] and "order2_relative_size_drift" in r for r in third)
        drifts=[r["order2_relative_size_drift"] for r in third] if third_complete else [None,None]
        third_decreasing=drifts[1]<drifts[0] if third_complete else None
        flags=dict(primary_complete=bool(fit and fit["complete_grid"] and fit["n"]>=11),
            pair_complete=bool(pair and pair["complete_grid"] and pair["scale_screen_both"]),
            models_complete=len(models)==3,windows_complete=len(windows)==len(WINDOWS),third_complete=third_complete)
        size=pair.get("order2_relative_size_drift") if pair else None
        raw=pair.get("order2_D_difference_to_fitted_log_span") if pair else None
        missing=[key for key,value in flags.items() if not value]
        signchanged=bool(c and (any(r["c_fit"]*c<=0 for r in models+windows) or (pair and pair.get("order2_c_small",c)*c<=0)))
        passed=not missing and all(value is not None and value<=limit for value,limit in ((size,.02),(raw,.01),(modeldrift,.05),(windowdrift,.05))) and third_decreasing is True and drifts[1]<=.02 and not signchanged
        eq=[r for r in limitfits if r["pattern"]==name and r["requested_max"]==216 and r["has_requested_upper"]]
        primary_eq=next((r for r in eq if r["requested_min"]==36 and r["inverse_order"]==1),None)
        eqc=primary_eq["c_fit"] if primary_eq else None
        confirmed=name in confirmation
        eqvalues=[r["c_fit"] for r in eq];timevalues=[r["c_fit"] for r in models+windows]
        eqdrift=max((ratio(value-eqc,eqc) for value in eqvalues),default=None) if eqc else None
        eqstable=len(eq)==6 and eqdrift is not None and eqdrift<=.05
        gap=(c-eqc)/abs(eqc) if c is not None and eqc is not None and eqc else None
        overlap=max(min(eqvalues),min(timevalues))<=min(max(eqvalues),max(timevalues)) if eqvalues and timevalues else None
        supported=confirmed and passed and eqstable and gap is not None and abs(gap)<=.05 and overlap is True
        result.append(dict(pattern=name,primary_L=216,comparison_L=180,primary_window="4-9",primary_model="log_inv1_inv2",
            c_time_fit=c,c_eq_fit=eqc,eq_model="log_inv1",eq_requested_window="36-216",
            eq_actual_grid=primary_eq["grid"] if primary_eq else "unavailable",
            eq_sensitivity_min=min((r["c_fit"] for r in eq),default=None),eq_sensitivity_max=max((r["c_fit"] for r in eq),default=None),
            time_sensitivity_min=min((r["c_fit"] for r in models+windows),default=None),time_sensitivity_max=max((r["c_fit"] for r in models+windows),default=None),
            **flags,size_drift=size,D_difference_to_fitted_log_span=raw,model_drift=modeldrift,window_drift=windowdrift,
            third_drift_144_180=drifts[0],third_drift_180_216=drifts[1],third_decreasing=third_decreasing,
            sign_changed=signchanged,missing_checks=";".join(missing),time_side_local_checks_pass=passed,
            confirmation_selected=confirmed,confirmation_local_checks_pass=confirmed and passed,
            status="confirmed_local_checks_not_asymptotic_proof" if confirmed and passed else "available_local_checks_candidate_not_confirmation_selected" if passed else "unresolved_missing_confirmation_data" if missing else "unresolved_time_side_drift",
            equilibrium_sensitivity=eqdrift,equilibrium_sensitivity_pass=eqstable,
            sensitivity_ranges_overlap=overlap,numerical_agreement_supported=supported,
            evidence_class="finite_window_numerical_support" if supported else "time_checks_pass_comparison_unresolved" if confirmed and passed else "confirmation_time_side_unresolved" if confirmed else "pilot_only_not_sufficiently_confirmed",
            time_minus_eq=(c-eqc) if c is not None and eqc is not None else None,
            relative_time_eq_gap=gap,
            comparison_order="Time-side eligibility evaluated before this equilibrium comparison; gap does not select fits"))
    return result


def make_figures(observations,names,metadata,timefits,limitfits,slopes,paired,primary):
    """Static scientific figures; unshifted raw D and independent axis limits."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    FIGURES.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,
        "axes.spines.right":False,"axes.grid":True,"grid.alpha":.2,"savefig.dpi":170})
    palette={72:"#9D9D9D",108:"#9065A8",144:"#4388B5",180:"#219E85",216:"#D47728"}
    def save(fig,name):
        for suffix in ("png","pdf"):
            path=FIGURES/f"{name}.{suffix}";fig.savefig(path,bbox_inches="tight");tracked(path)
        plt.close(fig)
    def predict(fit,x):
        x=np.asarray(x,dtype=float)
        return fit["c_fit"]*np.log(x)+fit["constant"]+sum(fit[f"inv{j}"]/x**j for j in range(1,fit["inverse_order"]+1))
    byname={row["pattern"]:row for row in primary}
    status_labels={"finite_window_numerical_support":"Numerical support within tested windows",
        "time_checks_pass_comparison_unresolved":"Time checks pass; comparison unresolved",
        "confirmation_time_side_unresolved":"Confirmation: unresolved time-side checks",
        "pilot_only_not_sufficiently_confirmed":"Pilot only: insufficient confirmation"}
    for name in names:
        state=byname[name];points=[r for r in observations if r["pattern"]==name]
        infinite=sorted((r for r in points if math.isinf(r["theta"]) and r["L"]>=18),key=lambda r:r["L"])
        fig,axes=plt.subplots(1,2,figsize=(12.6,4.7));left,right=axes
        left.plot(np.log([r["L"] for r in infinite]),[r["D"] for r in infinite],"o",mfc="white",mec="#286B9C",ms=7,label="Measured ground-state limit")
        eq=[r for r in limitfits if r["pattern"]==name and r["requested_max"]==216 and r["has_requested_upper"]]
        principal=next((r for r in eq if r["requested_min"]==36 and r["inverse_order"]==1),None)
        for fit in eq:
            if fit is principal:continue
            x=np.geomspace(fit["x_min"],fit["x_max"],180)
            left.plot(np.log(x),predict(fit,x),color="#8DAEC3",alpha=.5,lw=1)
        if principal:
            x=np.geomspace(principal["x_min"],principal["x_max"],180)
            left.plot(np.log(x),predict(principal,x),color="#286B9C",lw=2,
                label=f"log + 1/L; c={principal['c_fit']:.5f}")
        left.set(xlabel=r"$\ln L$",ylabel=r"$D=(2S_A+S_C)/3-S_B$",title="Equilibrium: independent L fits")
        left.legend(loc="best",fontsize=8)
        for size in (108,144,180,216):
            rows=sorted((r for r in points if r["L"]==size and 1<=r["theta"]<=10),key=lambda r:r["theta"])
            if rows:right.plot(np.log([r["theta"] for r in rows]),[r["D"] for r in rows],"o-",ms=3.5,lw=1,
                color=palette[size],mfc="white",label=f"L={size}")
        right.axvspan(math.log(4),math.log(9),color="#D47728",alpha=.08)
        fit=next((r for r in timefits if r["pattern"]==name and r["L"]==216 and r["window"]=="4-9" and r["inverse_order"]==2 and r["complete_grid"]),None)
        if fit:
            x=np.geomspace(4,9,200)
            right.plot(np.log(x),predict(fit,x),color="#363636",lw=2,ls="--",
                label=f"L216 fit c={fit['c_fit']:.5f}"+("; unresolved" if not state["confirmation_local_checks_pass"] else ""))
        else:right.text(.03,.05,"L216, theta 4-9 fit unavailable\nConfirmation remains unresolved",transform=right.transAxes,fontsize=9)
        right.set(xlabel=r"$\ln\theta$",ylabel=r"Raw $D(\theta,L)$",title="Nonequilibrium: same-time size checks")
        right.legend(loc="best",fontsize=8,ncol=2)
        fig.suptitle(f"{name}  |  class {metadata[name]['occupancy_class']}  |  beta=0\n{status_labels[state['evidence_class']]}",fontsize=12)
        fig.text(.5,.005,"Raw points are not shifted. Shading marks the planned time window; thin equilibrium lines show model/cut sensitivity, not statistical CI.",ha="center",fontsize=8)
        fig.tight_layout(rect=(0,.025,1,.91));save(fig,name+"_scaling")

        fig,axes=plt.subplots(1,2,figsize=(12.6,4.5));left,right=axes
        for size in (72,108,144,180,216):
            selected=[r for r in slopes if r["pattern"]==name and r["domain"]=="time" and r["L"]==size and r["pair_kind"]=="adjacent" and 1<=r["x_low"]<r["x_high"]<=10]
            if selected:left.plot([r["x_geometric_mean"] for r in selected],[r["log_slope"] for r in selected],"o-",color=palette[size],mfc="white",ms=3,lw=1,label=f"L={size}")
        left.axvspan(4,9,color="#D47728",alpha=.08)
        left.set(xlabel=r"$\sqrt{\theta_i\theta_{i+1}}$",ylabel=r"$\Delta D/\Delta\ln\theta$",title="Adjacent raw log slopes; no fitted corrections")
        left.legend(fontsize=8,ncol=3)
        available={(r["L"],r["theta"]):r for r in points if math.isfinite(r["theta"])}
        for smaller,larger,window,color in ((72,108,"4-9","#9065A8"),(144,180,"4-7","#4388B5"),(180,216,"4-9","#D47728")):
            pair=next((r for r in paired if r["pattern"]==name and r["L_small"]==smaller and r["L_large"]==larger and r["window"]==window),None)
            if not pair or not pair["raw_D_span_large"]:continue
            theta=[float(t) for t in pair["grid"].split(",")]
            values=[(available[larger,t]["D"]-available[smaller,t]["D"])/pair["raw_D_span_large"] for t in theta]
            right.plot(theta,values,"o-",mfc="white",ms=4,color=color,label=f"{smaller}/{larger}, window {window}"+(" (pilot)" if smaller==72 else ""))
        right.axhline(0,color="black",lw=.6)
        right.set(xlabel=r"$\theta$",ylabel=r"$[D(L_{large})-D(L_{small})]/\mathrm{span}(D_{large})$",title="Same-time size differences / measured window span")
        right.legend(fontsize=8,loc="best")
        fig.suptitle(name+" | beta=0 | "+status_labels[state["evidence_class"]],fontsize=11)
        fig.text(.5,.005,"Each size pair has its labeled window and its own raw-D-span denominator. L72/108 at late times is exploratory and outside the final scale screen.",ha="center",fontsize=8)
        fig.tight_layout(rect=(0,.025,1,.94));save(fig,name+"_time_diagnostics")

    selected=[r for r in primary if r["confirmation_selected"]]
    fig,axes=plt.subplots(math.ceil(len(selected)/2),2,figsize=(12.5,2.8*math.ceil(len(selected)/2)),squeeze=False)
    for ax,row in zip(axes.flat,selected):
        for xpos,value,lo,hi,color,marker,label in ((0,row["c_eq_fit"],row["eq_sensitivity_min"],row["eq_sensitivity_max"],"#286B9C","o","Equilibrium"),
                (1,row["c_time_fit"],row["time_sensitivity_min"],row["time_sensitivity_max"],"#D47728","s","Time")):
            if value is None:continue
            if lo is not None and hi is not None:ax.vlines(xpos,lo,hi,color=color,lw=2,alpha=.7)
            ax.plot(xpos,value,marker,ms=9 if xpos==0 else 6,mec=color,mew=1.7,
                mfc="white" if xpos==0 or not row["confirmation_local_checks_pass"] else color)
            ax.annotate(f"{value:.5f}",(xpos,value),xytext=(8,0),textcoords="offset points",fontsize=8,va="center",color=color)
        ax.set_xlim(-.5,1.65);ax.set_xticks((0,1),["Eq: L36-216","Time: L216, 4-9"])
        ax.set_ylabel("Log-fit parameter")
        ax.set_title(row["pattern"]+"\n"+status_labels[row["evidence_class"]],fontsize=10)
    for ax in list(axes.flat)[len(selected):]:ax.set_axis_off()
    fig.suptitle("beta=0: independently selected time fits and equilibrium comparison",fontsize=14)
    fig.text(.5,.005,"Whiskers are model/window sensitivity ranges, not statistical CI. Orange open squares mark unresolved time checks; each panel has its own y scale.",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.02,1,.975));save(fig,"coefficient_comparison")


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--stage",choices=("pilot","final"),default="pilot")
    parser.add_argument("--plots",action="store_true",help="Render final standalone scientific figures after complete-grid checks")
    args=parser.parse_args();OUT.mkdir(exist_ok=True)
    if args.plots and args.stage!="final":parser.error("Final figures require --stage final")
    observations,metadata,design,provenance,duplicates,live,manifests=read_inputs(args.stage)
    confirmation_plan=read_confirmation(args.stage)
    names=design["patterns"]
    coverage,timefits,limitfits,paired,pointdiffs=analyze_points(observations,metadata,names)
    slopes=raw_slopes([r for r in observations if r["pattern"] in names],metadata)
    ranking=candidate_ranking(names,timefits,paired,slopes)
    confirmation=confirmation_plan.get("patterns",[])
    primary=eligibility(names,timefits,limitfits,paired,confirmation)
    observed={(r["pattern"],r["L"],r["theta"]) for r in observations};required=[]
    for filename in ("pilot_jobs.csv","pilot_reused.csv","confirmation_jobs.csv","confirmation_reused.csv"):
        path=OUT/filename
        if not path.exists():continue
        for row in csv.DictReader(io.StringIO(tracked(path).decode("utf-8-sig"))):
            key=(row["pattern"],int(row["L"]),float(row["theta"]))
            required.append(dict(plan_file=filename,pattern=key[0],L=key[1],theta=key[2],present=key in observed))
    missing=[row for row in required if not row["present"]]
    if args.stage=="final" and missing:raise ValueError(f"Final analysis missing {len(missing)} frozen required keys")
    outputs={"study_observations.csv":observations,"study_duplicate_checks.csv":duplicates,"study_time_coverage.csv":coverage,
        "study_time_fits.csv":timefits,"study_equilibrium_fits.csv":limitfits,"study_size_comparisons.csv":paired,
        "study_same_theta_size_differences.csv":pointdiffs,"study_local_log_slopes.csv":slopes,
        "pilot_candidate_ranking.csv":ranking,"primary_coefficients.csv":primary,"study_required_coverage.csv":required}
    for filename,rows in outputs.items():dump_csv(OUT/filename,rows);tracked(OUT/filename)
    dump_json(OUT/"study_observation_provenance.json",dict(points=provenance,live_sources=live));tracked(OUT/"study_observation_provenance.json")
    if args.plots:make_figures(observations,names,metadata,timefits,limitfits,slopes,paired,primary)
    tracked(Path(__file__));tracked(ROOT/"audit_beta0_inventory.py")
    summary=dict(status="PASS",stage=args.stage,generated_utc=datetime.now(timezone.utc).isoformat(),files=FILES,
        raw_source_snapshots=live,source_run_status={run:info["manifest"]["status"] for run,info in manifests.items()},
        counts=dict(patterns=len(names),pooled_points=len(observations),new_run_count=len(manifests),duplicates=len(duplicates),
            time_fits=len(timefits),equilibrium_fits=len(limitfits),paired_checks=len(paired),raw_local_slopes=len(slopes),missing_required_keys=len(missing)),
        conventions=dict(beta=0,log="natural",D="(2SA+SC)/3-SB",equilibrium_selection="None; limit fits compared only after time-side decisions",
            time_windows=WINDOWS,primary=dict(L=216,comparison_L=180,window=[4,9],model="log_inv1_inv2"),
            threshold=THRESHOLDS,five_parameter_minimum_points=11,third_size_window=[4,7],third_sizes=[144,180,216],
            condition_and_range="Deterministic sensitivity diagnostics, not statistical confidence intervals",
            pilot_ranking="Same L72/108 and theta4-9 for all candidates, not certification; no equilibrium target or literature value",
            final_scale_screen="theta>=4 and theta/L<=0.05 for both compared sizes; an operational screen, not an asymptotic proof",
            missing_data="No confirmation data means candidate/unresolved, never a confirmed pass"),
        candidates=ranking,primary=primary,confirmation_patterns=confirmation,
        comparison_rule=confirmation_plan.get("comparison_rule","Not yet frozen"))
    dump_json(OUT/"study_summary.json",summary)
    print(json.dumps(dict(status="PASS",stage=args.stage,**summary["counts"],local_pass=sum(r["time_side_local_checks_pass"] for r in primary),confirmed_local_pass=sum(r["confirmation_local_checks_pass"] for r in primary),numerical_support=sum(r["numerical_agreement_supported"] for r in primary)),indent=2))


if __name__=="__main__":main()
