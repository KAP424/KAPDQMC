"""Read-only census of the pre-study beta=0 evidence.

The input allowlist is deliberately frozen: later beta0-study calculations and
this script's own output cannot flow back into this baseline. Duplicate entropy
values are checked before a point is pooled. Fits below are recomputed from raw
points, never selected using a target equilibrium coefficient or old pass flag.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"analysis_beta0_study"
# Current experiment files, before the new beta0 study. New files are excluded.
RAW_FILES=(
    "observations.csv","extended_observations.csv","extraction_observations.csv",
    "extraction_smoke_observations.csv","equilibrium_arrangements.csv","equilibrium_extended.csv",
    "ground_observations_baseline.csv","late_time_observations.csv",
    "ref_calibration_observations.csv","ref_calibration_equilibrium.csv","ref_calibration_late.csv",
    "beta_patterns_observations.csv","uniformA_dense_observations.csv",
    "occupancy_classes_observations.csv","irregular_followup_observations.csv",
    "irregular_followup_extended_equilibrium.csv","second_class_extra_observations.csv",
    "beta_patterns_folded_validation.csv","beta_patterns_validation.csv",
    "irregular_ground_validation.csv","irregular_m3_validation.csv",
    "occupancy_classes_validation.csv","ref_calibration_validation.csv",
    "second_class_extra_validation.csv","uniformA_dense_half_validation.csv",
    "uniformA_dense_validation.csv","numerical_sensitivity.csv")
HISTORICAL_DIRS=("analysis_uniformA_dense","analysis_beta_patterns","analysis_occupancy_classes",
    "analysis_irregular_followup","analysis_second_class_extra","analysis_equilibrium",
    "analysis_time_scale","analysis_ref_calibration","analysis_multi_pattern_entropy")
PREFERRED=("second_class_extra_observations.csv","irregular_followup_observations.csv",
    "irregular_followup_extended_equilibrium.csv","occupancy_classes_observations.csv",
    "beta_patterns_observations.csv","uniformA_dense_observations.csv",
    "ref_calibration_observations.csv","ref_calibration_equilibrium.csv","ref_calibration_late.csv")
SOURCES={}


def read(path):
    data=path.read_bytes();SOURCES[str(path.relative_to(ROOT)).replace('\\','/')]=hashlib.sha256(data).hexdigest()
    return data


def table(path):
    return list(csv.DictReader(read(path).decode("utf-8-sig").splitlines()))


def dump_csv(path,rows):
    fields=list(dict.fromkeys(k for row in rows for k in row)) or ["status"]
    with path.open("w",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def dump_json(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n",encoding="utf-8")


def grid(values):return ",".join(f"{x:g}" for x in sorted(values))


def point_id(name,size,time):return f"{name}|{size}|{'Inf' if math.isinf(time) else f'{time:g}'}"


def collect():
    registry={r["pattern"]:dict(m=int(r["m"]),bits=r["occupied_bits"]) for r in table(ROOT/"data/patterns.csv")}
    records=[];sources=[]
    for filename in RAW_FILES:
        path=ROOT/"data"/filename
        if not path.exists():raise ValueError(f"Frozen source missing: {path}")
        raw=table(path);selected=[]
        for number,row in enumerate(raw,2):
            if row.get("occupied_bits"):
                value=dict(m=int(row["m"]),bits=row["occupied_bits"])
                if row["pattern"] in registry and registry[row["pattern"]]!=value:
                    raise ValueError(f"Pattern changed mask: {row['pattern']}")
                registry[row["pattern"]]=value
            if float(row["beta"])==0:selected.append((number,row))
        sources.append(dict(path="data/"+filename,sha256=SOURCES["data/"+filename],rows=len(raw),beta0_rows=len(selected),
            role="validation_or_numerical_check" if "validation" in filename or filename=="numerical_sensitivity.csv" else "observation_or_reused_observation"))
        records.extend((filename,number,row) for number,row in selected)
    aliases=defaultdict(list)
    for name,meta in registry.items():aliases[meta["m"],meta["bits"]].append(name)
    canonical={key:min(names,key=lambda n:(not n.startswith("pure02_alternating"),n)) for key,names in aliases.items()}
    groups=defaultdict(list);metadata={};ground_flags=defaultdict(set)
    for filename,number,row in records:
        name=row["pattern"]
        if name not in registry:raise ValueError(f"No mask evidence for {name}")
        meta=registry[name];name=canonical[meta["m"],meta["bits"]]
        counts=[sum(int(meta["bits"][i])+int(meta["bits"][i+1])==q for i in range(0,len(meta["bits"]),2)) for q in range(3)]
        metadata[name]=dict(m=meta["m"],occupied_bits=meta["bits"],aliases=sorted(aliases[meta["m"],meta["bits"]]),
            cell_counts=counts,occupancy_class=1 if counts[1]==meta["m"]**2 else 2 if counts[1]==0 else 3)
        size,time=int(row["L"]),float(row["theta"])
        entropies={k:float(row.get(k,row.get(k.replace("_","")))) for k in ("S_A","S_C","S_B")}
        d=(2*entropies["S_A"]+entropies["S_C"])/3-entropies["S_B"]
        if row.get("D") and abs(float(row["D"])-d)>1e-8:raise ValueError("Weighted D identity failed")
        if not all(math.isfinite(x) for x in [*entropies.values(),d]):raise ValueError("Nonfinite entropy")
        for field in ("x0","y0","cut_x0","cut_y0"):
            if row.get(field) and int(row[field])!=1:raise ValueError("Baseline requires the original fixed cut origin")
        flag=row.get("premeasurement_ground",row.get("ground",""))
        if flag:ground_flags[name].add(flag.lower()=="true")
        if math.isinf(time) and not flag and row.get("excess_energy"):
            ground_flags[name].add(abs(float(row["excess_energy"]))<1e-8)
        source=dict(file="data/"+filename,row=number,sha256=SOURCES["data/"+filename],original_pattern=row["pattern"],
                    role="check" if "validation" in filename or filename=="numerical_sensitivity.csv" else "observation",
                    rank_tol=row.get("rank_tol",row.get("tol","")),run_id=row.get("run_id",""))
        groups[name,size,time].append(dict(**entropies,D=float(row["D"]) if row.get("D") else d,source=source))
    observations=[];mapping={};duplicates=[];conflicts=[]
    def preference(item):
        filename=Path(item["source"]["file"]).name
        return (0,PREFERRED.index(filename)) if filename in PREFERRED else (2 if item["source"]["role"]=="check" else 1,filename)
    for (name,size,time),items in sorted(groups.items()):
        items.sort(key=preference);selected=items[0]
        for item in items[1:]:
            errors={k:abs(item[k]-selected[k]) for k in ("S_A","S_C","S_B","D")}
            record=dict(canonical_pattern=name,L=size,theta=time,selected_file=selected["source"]["file"],
                selected_row=selected["source"]["row"],comparison_file=item["source"]["file"],comparison_row=item["source"]["row"],
                max_absolute_difference=max(errors.values()),consistent=max(errors.values())<=1e-8)
            duplicates.append(record)
            if not record["consistent"]:conflicts.append(record)
        flags=ground_flags[name]
        if len(flags)!=1:raise ValueError(f"Missing/conflicting reachability evidence: {name}: {flags}")
        reachable=next(iter(flags));metadata[name]["ground_reachable"]=reachable
        ident=point_id(name,size,time)
        observations.append(dict(canonical_pattern=name,pattern=name,m=metadata[name]["m"],L=size,theta=time,beta=0,
            **{key:selected[key] for key in ("S_A","S_C","S_B","D")},ground_reachable=reachable,
            occupied_bits=metadata[name]["occupied_bits"],source_count=len(items),source_id=ident,
            selected_source_file=selected["source"]["file"],selected_source_row=selected["source"]["row"],
            selected_source_sha256=selected["source"]["sha256"]))
        mapping[ident]=dict(canonical_pattern=name,L=size,theta="Inf" if math.isinf(time) else time,
            selected_source=selected["source"],all_sources=[x["source"] for x in items],
            duplicate_consistency_tolerance=1e-8)
    return observations,metadata,sources,mapping,duplicates,conflicts


TIME_WINDOWS=((2,4),(2,6),(3,6),(4,7),(4.5,7),(4,6.5),(5,7),(5,8),(4,8),(4,9),(5,9),(6,9),(6,10),(6,12))
MODEL_NAMES=("log","log_inv1","log_inv1_inv2","log_inv1_inv2_inv3")
HISTORICAL_FILES=(
    ("analysis_uniformA_dense/primary_coefficients.csv","uniform_A"),
    ("analysis_beta_patterns/primary_coefficients.csv",None),
    ("analysis_occupancy_classes/primary_coefficients.csv",None),
    ("analysis_irregular_followup/primary_coefficients.csv",None),
    ("analysis_second_class_extra/primary_parameters.csv",None),
    ("analysis_equilibrium/equilibrium_summary.csv",None),
    ("analysis_time_scale/all_patterns_summary.csv",None),
    ("analysis_ref_calibration/calibration_comparison.csv",None))


def load_frozen_baseline():
    """Read, hash-check and recheck duplicates without touching frozen artifacts."""
    baseline_path=OUT/"baseline_observations.csv"
    provenance=json.loads(read(OUT/"baseline_provenance.json"))
    if hashlib.sha256(read(baseline_path)).hexdigest()!=provenance["baseline_sha256"]:
        raise ValueError("Frozen baseline hash changed")
    for relative,expected in provenance["source_sha256"].items():
        if hashlib.sha256(read(ROOT/relative)).hexdigest()!=expected:
            raise ValueError(f"Frozen historical source changed: {relative}")
    observations=table(baseline_path)
    for row in observations:
        for field in ("S_A","S_C","S_B","D","theta"):row[field]=float(row[field])
        for field in ("m","L","source_count"):row[field]=int(row[field])
        row["ground_reachable"]=row["ground_reachable"].lower()=="true"
    # Recompute comparisons from historical sources. No serialization is done here.
    rebuilt,metadata,_,_,duplicates,conflicts=collect()
    if conflicts or len(rebuilt)!=len(observations):raise ValueError("Historical baseline audit no longer matches")
    keys=("canonical_pattern","m","L","theta","S_A","S_C","S_B","D","ground_reachable","occupied_bits")
    for actual,expected in zip(rebuilt,observations):
        if any(actual[key]!=expected[key] for key in keys):raise ValueError("Frozen baseline does not reproduce")
    return observations,metadata,provenance,duplicates


def independent_fit(points,axis,order):
    """Unweighted column-scaled SVD; condition/noise gain are not error bars."""
    points=sorted(points,key=lambda row:row[axis]);x=np.array([row[axis] for row in points],dtype=float)
    y=np.array([row["D"] for row in points],dtype=float)
    columns=[np.log(x),np.ones_like(x)]+[x**(-j) for j in range(1,order+1)]
    design=np.column_stack(columns);n,p=design.shape
    if n<p+2:return None
    scale=np.linalg.norm(design,axis=0);scaled=design/scale
    parameters=np.linalg.lstsq(scaled,y,rcond=None)[0]/scale
    residual=design@parameters-y;weights=np.linalg.pinv(scaled)[0]/scale[0]
    correction=design[:,2:]@parameters[2:] if p>2 else np.zeros(n)
    logspan=abs(parameters[0]*math.log(x[-1]/x[0]))
    return dict(model=MODEL_NAMES[order],inverse_order=order,n=n,parameters=p,residual_dof=n-p,
        x_min=float(x[0]),x_max=float(x[-1]),grid=grid(x),c_fit=float(parameters[0]),constant=float(parameters[1]),
        inv1=float(parameters[2]) if p>2 else 0.,inv2=float(parameters[3]) if p>3 else 0.,
        inv3=float(parameters[4]) if p>4 else 0.,rmse=float(np.sqrt(np.mean(residual**2))),
        max_abs_residual=float(np.max(np.abs(residual))),condition_scaled=float(np.linalg.cond(scaled)),
        condition_raw=float(np.linalg.cond(design)),c_noise_gain_l2=float(np.linalg.norm(weights)),
        c_noise_gain_l1=float(np.sum(abs(weights))),raw_D_span=float(np.ptp(y)),
        raw_endpoint_secant=float((y[-1]-y[0])/math.log(x[-1]/x[0])),
        correction_span=float(np.ptp(correction)),
        correction_to_fitted_log_span=float(np.ptp(correction)/logspan) if logspan else None)


def roles(meta):
    return dict(ground_reachable=meta["ground_reachable"],occupancy_class=meta["occupancy_class"],
        limit_state="ground_state" if meta["ground_reachable"] else "accessible_excited_limit",
        parameter_meaning="finite_window_fit_not_certified_corner_coefficient" if meta["ground_reachable"] else "fit_parameter_only_not_ground_state_corner_coefficient")


def raw_slopes(observations,metadata):
    groups=defaultdict(list);result=[]
    for row in observations:
        if math.isinf(row["theta"]):groups[row["canonical_pattern"],"limit",0].append(row)
        elif row["theta"]>0:groups[row["canonical_pattern"],"time",row["L"]].append(row)
    for (name,domain,size),points in sorted(groups.items()):
        axis="L" if domain=="limit" else "theta";points.sort(key=lambda row:row[axis])
        pairs=[("adjacent",a,b) for a,b in zip(points,points[1:])]
        pairs.extend(("doubling",a,b) for a,b in itertools.combinations(points,2) if b[axis]==2*a[axis])
        for kind,a,b in pairs:
            xa,xb=a[axis],b[axis]
            result.append(dict(pattern=name,**roles(metadata[name]),domain=domain,L=size if domain=="time" else "",
                pair_kind=kind,x_low=xa,x_high=xb,x_geometric_mean=math.sqrt(xa*xb),D_low=a["D"],D_high=b["D"],
                log_slope=(b["D"]-a["D"])/math.log(xb/xa),
                theta_over_L_max=xb/size if domain=="time" else "",source_id_low=a["source_id"],source_id_high=b["source_id"]))
    return result


def compute_fits(observations,metadata):
    groups=defaultdict(list)
    for row in observations:groups[row["canonical_pattern"]].append(row)
    limits=[];times=[];missing=[]
    for name,points in sorted(groups.items()):
        meta=metadata[name];infinite=[r for r in points if math.isinf(r["theta"])]
        lower_bounds=(24,36,48) if meta["m"]==4 else (18,36,54)
        upper_bounds=sorted(set([min(144,max(r["L"] for r in infinite))]+[r["L"] for r in infinite if r["L"]>144])) if infinite else []
        for lo,hi,order in itertools.product(lower_bounds,upper_bounds,range(3)):
            selected=[r for r in infinite if lo<=r["L"]<=hi]
            fit=independent_fit(selected,"L",order)
            common=dict(pattern=name,**roles(meta),domain="limit",requested_min=lo,requested_max=hi)
            if fit:limits.append(dict(**common,**fit))
            else:missing.append(dict(**common,model=MODEL_NAMES[order],n=len(selected),required=order+4))
        finite=[r for r in points if math.isfinite(r["theta"]) and r["theta"]>0]
        for size in sorted(set(r["L"] for r in finite)):
            for (lo,hi),order in itertools.product(TIME_WINDOWS,range(4)):
                selected=[r for r in finite if r["L"]==size and lo<=r["theta"]<=hi]
                fit=independent_fit(selected,"theta",order)
                common=dict(pattern=name,**roles(meta),domain="time",L=size,requested_min=lo,requested_max=hi)
                if fit:times.append(dict(**common,theta_over_L_max=fit["x_max"]/size,
                    operational_theta_over_L_le_005=fit["x_max"]/size<=.05,**fit))
                elif size>=72:missing.append(dict(**common,model=MODEL_NAMES[order],n=len(selected),required=order+4))
    return limits,times,missing


def size_comparisons(observations,metadata):
    byname=defaultdict(lambda:defaultdict(dict));result=[]
    for row in observations:
        if math.isfinite(row["theta"]) and row["theta"]>0 and row["L"]>=72:
            byname[row["canonical_pattern"]][row["L"]][row["theta"]]=row
    for name,sizes in sorted(byname.items()):
        for small,large in itertools.combinations(sorted(sizes),2):
            for lo,hi in TIME_WINDOWS:
                common=sorted(t for t in set(sizes[small])&set(sizes[large]) if lo<=t<=hi)
                if len(common)<2:continue
                a=[sizes[small][t] for t in common];b=[sizes[large][t] for t in common]
                difference=np.array([rb["D"]-ra["D"] for ra,rb in zip(a,b)])
                span=float(np.ptp([r["D"] for r in b]));endpoint=b[-1]["D"]-b[0]["D"]
                row=dict(pattern=name,**roles(metadata[name]),L_small=small,L_large=large,requested_min=lo,requested_max=hi,
                    grid=grid(common),n=len(common),theta_min=common[0],theta_max=common[-1],theta_over_L_small=common[-1]/small,
                    theta_over_L_large=common[-1]/large,max_abs_D_size_difference=float(max(abs(difference))),
                    D_difference_span=float(np.ptp(difference)),large_raw_D_span=span,
                    size_difference_over_raw_D_span=float(max(abs(difference))/span) if span else None,
                    change_of_size_difference_over_raw_D_span=float(np.ptp(difference)/span) if span else None,
                    small_raw_endpoint_slope=(a[-1]["D"]-a[0]["D"])/math.log(common[-1]/common[0]),
                    large_raw_endpoint_slope=endpoint/math.log(common[-1]/common[0]))
                for order in range(4):
                    fa,fb=independent_fit(a,"theta",order),independent_fit(b,"theta",order)
                    if fa and fb:
                        prefix=f"order{order}_"
                        row.update({prefix+"c_small":fa["c_fit"],prefix+"c_large":fb["c_fit"],
                            prefix+"absolute_c_drift":abs(fb["c_fit"]-fa["c_fit"]),
                            prefix+"relative_c_drift":abs(fb["c_fit"]-fa["c_fit"])/max(abs(fb["c_fit"]),1e-15)})
                result.append(row)
    return result


def historical_rows(metadata):
    alias={name:canonical for canonical,meta in metadata.items() for name in meta["aliases"]};result=[]
    for relative,default_pattern in HISTORICAL_FILES:
        path=ROOT/relative
        if not path.exists():continue
        for number,row in enumerate(table(path),2):
            if float(row.get("beta","nan"))!=0:continue
            name=alias.get(row.get("pattern",default_pattern),row.get("pattern",default_pattern))
            if name not in metadata:continue
            result.append(dict(pattern=name,source_file=relative,source_row=number,source_sha256=SOURCES[relative],
                trust_policy="historical context only; no flag imported into new diagnostics",
                historical_c_limit=row.get("c_eq",row.get("c_limit_fit",row.get("D_corrected_c",""))),
                historical_c_time=row.get("c_neq",row.get("c_time_fit",row.get("c_main108",row.get("c","")))),
                historical_L=row.get("primary_L",row.get("main_L","")),historical_window=row.get("primary_window",""),
                old_flag=row.get("local_diagnostic_pass",row.get("numeric_diagnostic_pass",row.get("stable_diagnostic",""))),
                full_historical_row_json=json.dumps(row,sort_keys=True)))
    return result


def summarize(observations,metadata,limits,times,slopes,comparisons):
    result=[]
    for name,meta in sorted(metadata.items()):
        points=[r for r in observations if r["canonical_pattern"]==name]
        finite=[r for r in points if math.isfinite(r["theta"])];infinite=[r for r in points if math.isinf(r["theta"])]
        sizes=sorted(set(r["L"] for r in finite));largest=max(sizes,default=None)
        finite_grids={str(size):grid(r["theta"] for r in finite if r["L"]==size) for size in sizes}
        primary_lower=24 if meta["m"]==4 else 18
        eq=[r for r in limits if r["pattern"]==name and r["requested_min"]==primary_lower and r["requested_max"]==144 and r["inverse_order"]==1]
        time=[r for r in times if r["pattern"]==name and r["requested_min"]==4 and r["requested_max"]==7 and r["inverse_order"]==2]
        primary=next((r for r in time if r["L"]==180),None)
        best=max(time,key=lambda r:r["L"],default=None)
        bestL=best["L"] if best else None
        other_models=[r["c_fit"] for r in times if best and r["pattern"]==name and r["L"]==bestL and r["requested_min"]==4 and r["requested_max"]==7 and r["inverse_order"] in (1,2,3)]
        other_windows=[r["c_fit"] for r in times if best and r["pattern"]==name and r["L"]==bestL and r["inverse_order"]==2 and (r["requested_min"],r["requested_max"]) in ((4,7),(4.5,7),(4,6.5),(5,7),(5,8))]
        selected_slopes=[r for r in slopes if r["pattern"]==name and r["domain"]=="time" and r["L"]==largest and r["pair_kind"]=="adjacent" and 4<=r["x_low"] and r["x_high"]<=7]
        paired=[r for r in comparisons if r["pattern"]==name and r["requested_min"]==4 and r["requested_max"]==7 and r["L_large"]==largest]
        nearest=max(paired,key=lambda r:r["L_small"],default=None)
        timevalue=best["c_fit"] if best else None;eqvalue=eq[0]["c_fit"] if eq else None
        result.append(dict(pattern=name,m=meta["m"],**roles(meta),aliases=";".join(meta["aliases"]),occupied_bits=meta["occupied_bits"],
            unique_points=len(points),source_rows=sum(r["source_count"] for r in points),finite_L_grid=grid(sizes),
            finite_theta_grids_json=json.dumps(finite_grids,sort_keys=True),max_finite_L=largest,
            max_L_theta_grid=finite_grids.get(str(largest),""),limit_L_grid=grid(r["L"] for r in infinite),
            baseline_limit_c_fit=eqvalue,baseline_limit_model="log_inv1",baseline_limit_grid=eq[0]["grid"] if eq else "unavailable",
            baseline_limit_sensitivity_min=min((r["c_fit"] for r in limits if r["pattern"]==name and r["requested_max"]==144 and r["inverse_order"] in (1,2)),default=None),
            baseline_limit_sensitivity_max=max((r["c_fit"] for r in limits if r["pattern"]==name and r["requested_max"]==144 and r["inverse_order"] in (1,2)),default=None),
            fixed_L180_time_4_7_c_fit=primary["c_fit"] if primary else None,
            largest_available_time_4_7_fit_L=bestL,largest_available_time_4_7_grid=best["grid"] if best else "unavailable",
            largest_available_time_4_7_c_fit=timevalue,time_minus_limit_fit=(timevalue-eqvalue) if timevalue is not None and eqvalue is not None else None,
            time_model_c_min=min(other_models,default=None),time_model_c_max=max(other_models,default=None),
            time_window_c_min=min(other_windows,default=None),time_window_c_max=max(other_windows,default=None),
            time_condition_scaled=best["condition_scaled"] if best else None,time_c_noise_gain_l2=best["c_noise_gain_l2"] if best else None,
            raw_local_slope_L=largest,raw_local_slope_window="adjacent raw points in 4-7",
            raw_local_slope_first=selected_slopes[0]["log_slope"] if selected_slopes else None,
            raw_local_slope_last=selected_slopes[-1]["log_slope"] if selected_slopes else None,
            raw_local_slope_min=min((r["log_slope"] for r in selected_slopes),default=None),
            raw_local_slope_max=max((r["log_slope"] for r in selected_slopes),default=None),
            raw_size_pair_small=nearest["L_small"] if nearest else None,raw_size_pair_large=largest,
            raw_size_common_grid=nearest["grid"] if nearest else "unavailable",
            raw_size_difference_over_raw_D_span=nearest["size_difference_over_raw_D_span"] if nearest else None,
            raw_size_slope_difference=(nearest["large_raw_endpoint_slope"]-nearest["small_raw_endpoint_slope"]) if nearest else None,
            raw_theta_over_L_smaller=nearest["theta_over_L_small"] if nearest else None,
            numeric_or_physical_pass="not_assigned_inventory_only",
            caution="Existing grids differ; raw slopes precede fits. Model/window ranges are sensitivity, not statistical confidence intervals. No equilibrium target used in selection."))
    return result


def diagnostics():
    observations,metadata,provenance,duplicates=load_frozen_baseline()
    slopes=raw_slopes(observations,metadata);limits,times,missing=compute_fits(observations,metadata)
    comparisons=size_comparisons(observations,metadata);history=historical_rows(metadata)
    inventory=summarize(observations,metadata,limits,times,slopes,comparisons)
    outputs={"inventory.csv":inventory,"inventory_local_log_slopes.csv":slopes,
        "inventory_limit_fits.csv":limits,"inventory_time_fits.csv":times,"inventory_missing_fits.csv":missing,
        "inventory_size_comparisons.csv":comparisons,"inventory_historical_coefficients.csv":history}
    for filename,rows in outputs.items():dump_csv(OUT/filename,rows)
    files={str((OUT/name).relative_to(ROOT)).replace('\\','/'):hashlib.sha256((OUT/name).read_bytes()).hexdigest() for name in outputs}
    files.update(SOURCES);files[str(Path(__file__).relative_to(ROOT)).replace('\\','/')]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    summary=dict(status="PASS",generated_utc=datetime.now(timezone.utc).isoformat(),
        baseline_sha256=provenance["baseline_sha256"],frozen_baseline_unchanged=True,files=files,
        counts=dict(patterns=len(metadata),reachable_patterns=sum(m["ground_reachable"] for m in metadata.values()),
            unique_points=len(observations),duplicate_comparisons=len(duplicates),local_log_slopes=len(slopes),limit_fits=len(limits),
            time_fits=len(times),size_comparisons=len(comparisons),historical_beta0_rows=len(history)),
        conventions=dict(D="(2 S_A + S_C) / 3 - S_B",beta=0,log="natural",
            fit_model="c_fit ln(x) + constant + sum_j a_j/x^j; unweighted, column-scaled SVD",
            minimum_residual_dof=2,baseline_limit="L<=144, lower bound24 for m4 else18, inverse order1",
            diagnostic_time_windows=TIME_WINDOWS,model_orders=[0,1,2,3],
            scale_screen="theta/L<=0.05 is recorded as an operational screen, not proof of a scaling limit",
            no_target_selection="Neither the equilibrium fit nor any literature value selects windows, models or candidate rankings",
            old_pass_flags="retained solely as historical provenance; not used for any current diagnostic",
            reachability="source rank/ground metadata, consistently checked across available sources; this inventory does not repeat Bloch rank calculations",
            blocked="Inf is the accessible excited-state limit; its fitted parameter is not a ground-state corner coefficient",
            duplicate_policy="equal raw points pooled once after checking S_A,S_C,S_B,D within1e-8; no independent-sample interpretation",
            uncertainty="model/window variation and condition/noise gain are deterministic diagnostics, never statistical CI",
            evidence_limit="finite-window fit agreement does not establish asymptotic log scaling; continuous-time size platforms remain to be tested"),
        inventory=inventory)
    dump_json(OUT/"inventory.json",summary)
    print(json.dumps(dict(status="PASS",**summary["counts"],baseline_sha256=provenance["baseline_sha256"]),indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-only",action="store_true")
    parser.add_argument("--diagnostics-only",action="store_true",help="Read frozen baseline and produce independent diagnostics; never rewrite baseline/provenance")
    args=parser.parse_args();OUT.mkdir(exist_ok=True)
    if args.baseline_only and args.diagnostics_only:parser.error("Choose only one mode")
    if (OUT/"baseline_observations.csv").exists() or args.diagnostics_only:
        if args.baseline_only:
            observations,metadata,provenance,duplicates=load_frozen_baseline()
            print(json.dumps(dict(status="PASS",mode="verify_existing_frozen_baseline",unique_points=len(observations),baseline_sha256=provenance["baseline_sha256"]),indent=2))
        else:diagnostics()
        return
    observations,metadata,sources,mapping,duplicates,conflicts=collect()
    dump_csv(OUT/"inventory_duplicate_checks.csv",duplicates)
    if conflicts:
        dump_csv(OUT/"inventory_conflicts.csv",conflicts)
        raise ValueError("Conflicting duplicates: baseline was not pooled or published")
    dump_csv(OUT/"baseline_observations.csv",observations)
    dump_csv(OUT/"inventory_sources.csv",sources)
    provenance=dict(status="PASS",generated_utc=datetime.now(timezone.utc).isoformat(),
        baseline_sha256=hashlib.sha256((OUT/"baseline_observations.csv").read_bytes()).hexdigest(),
        source_sha256=dict(SOURCES),patterns=metadata,points=mapping,
        duplicate_comparisons=len(duplicates),duplicate_max_abs_difference=max((r["max_absolute_difference"] for r in duplicates),default=0),
        policy="Frozen input allowlist; later beta0-study data and own outputs excluded. Equal raw points are never counted as independent samples.")
    dump_json(OUT/"baseline_provenance.json",provenance)
    print(json.dumps(dict(status="PASS",unique_points=len(observations),patterns=len(metadata),
        reachable_patterns=sum(m["ground_reachable"] for m in metadata.values()),duplicates=len(duplicates),
        max_duplicate_difference=provenance["duplicate_max_abs_difference"],baseline_sha256=provenance["baseline_sha256"]),indent=2))
    if args.baseline_only:return
    diagnostics()


if __name__=="__main__":main()
