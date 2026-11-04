"""Dense-beta uniform_A analysis with explicit scale-screening and provenance.

Exploration never selects a window using equilibrium agreement. The primary
L/window/model must be supplied explicitly, normally through --plan. A final
run additionally requires an explicit expected grid in that plan.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import itertools
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

HERE = Path(__file__).resolve().parent
MODELS = {"log_inv1": (1,), "log_inv1_inv2": (1, 2),
          "log_inv1_inv2_inv3": (1, 2, 3)}
DEFAULT_WINDOWS = [(4., 7.), (4.5, 7.), (4., 6.5), (5., 7.), (5., 8.), (4., 8.),
                   (5., 9.), (4., 10.), (5., 10.), (6., 10.), (6., 12.), (8., 12.)]
DEFAULT_INPUTS = [HERE/"data"/name for name in (
    "ground_observations_baseline.csv", "ref_calibration_observations.csv",
    "ref_calibration_equilibrium.csv", "ref_calibration_late.csv",
    "uniformA_dense_benchmark.csv", "uniformA_dense_observations.csv",
    "uniformA_dense_equilibrium.csv")]
BLUE, ORANGE = "#286B9C", "#D47728"


def sha_bytes(content):
    return hashlib.sha256(content).hexdigest()


def canonical(value):
    return round(float(value), 12)


def grid_string(values):
    return ",".join(f"{x:g}" for x in values)


def ratio(value, denominator):
    return float(value / denominator) if abs(denominator) >= 1e-10 else None


def dump_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row)) or ["status"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def relative_path(path):
    try:
        return str(path.resolve().relative_to(HERE))
    except ValueError:
        return str(path.resolve())


def read_sources(paths, stage, output, betas):
    selected, sources, duplicates = {}, [], []
    snapshots = output/"source_snapshots"
    snapshots.mkdir(parents=True, exist_ok=True)
    for path in paths:
        if not path.exists():
            sources.append(dict(path=str(path.resolve()), missing=True))
            continue
        original = path.read_bytes()
        digest = sha_bytes(original)
        content = original
        ignored = 0
        if content and not content.endswith(b"\n"):
            if stage == "final":
                raise ValueError(f"Incomplete CSV tail: {path}")
            end = content.rfind(b"\n") + 1
            ignored = len(content)-end
            content = content[:end]
        snapshot = snapshots/f"{path.stem}_{digest[:12]}.csv"
        snapshot.write_bytes(content)
        rows = list(csv.DictReader(io.StringIO(content.decode("utf-8-sig"))))
        sources.append(dict(path=str(path.resolve()), sha256=digest, size_bytes=len(original), rows=len(rows),
                            ignored_incomplete_tail_bytes=ignored,
                            parsed_snapshot=str(snapshot.resolve()), parsed_sha256=sha_bytes(content)))
        for number, row in enumerate(rows, 2):
            if row.get("pattern") != "uniform_A":
                continue
            beta = canonical(row["beta"])
            theta = canonical(row["theta"])
            size = int(row["L"])
            if beta not in betas or theta <= 0 or math.isnan(theta):
                continue
            values = {key:float(row[key]) for key in ("S_A", "S_C", "S_B", "D")}
            if not all(math.isfinite(x) for x in values.values()):
                raise ValueError(f"Nonfinite entropy: {path}:{number}")
            if abs((2*values["S_A"]+values["S_C"])/3-values["S_B"]-values["D"]) > 1e-9:
                raise ValueError(f"Entropy identity failed: {path}:{number}")
            ground = row.get("premeasurement_ground", row.get("ground", ""))
            if ground and ground.lower() != "true":
                raise ValueError(f"Input not ground-state accessible: {path}:{number}")
            excess = next((row[k] for k in ("premeasurement_excess", "premeasurement_limit_excess", "accessible_excess") if row.get(k)), None)
            if excess is not None and abs(float(excess)) > 1e-8:
                raise ValueError(f"Ground excess failed: {path}:{number}")
            key = size, theta, beta
            record = dict(pattern="uniform_A", L=size, theta=theta, beta=beta, **values,
                          source_file=relative_path(path), source_row=number, source_sha256=digest,
                          source_kind=row.get("source_kind", "existing_observation"),
                          underlying_source_file=row.get("source_path", row.get("source_file", "")),
                          underlying_source_row=row.get("source_row", ""),
                          underlying_source_sha256=row.get("source_sha256", ""), run_id=row.get("run_id", ""))
            if key in selected:
                previous = selected[key]
                error = max(abs(values[k]-previous[k]) for k in values)
                if error > 1e-9:
                    raise ValueError(f"Conflicting observations {key}: {error}")
                duplicates.append(dict(L=size, theta=theta, beta=beta,
                                       previous_file=previous["source_file"], previous_row=previous["source_row"],
                                       selected_file=record["source_file"], selected_row=number,
                                       max_entropy_difference=error))
            selected[key] = record
    return sorted(selected.values(), key=lambda r:(r["beta"],r["L"],r["theta"])), sources, duplicates


def fit(x, y, model):
    powers = MODELS[model]
    if len(x) < len(powers)+4:
        return None, "fewer_than_parameters_plus_two_points"
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    design = np.column_stack([np.log(x), np.ones(len(x))]+[x**(-p) for p in powers])
    scale = np.linalg.norm(design, axis=0)
    normalized = design/scale
    coefficients, _, rank, singular = np.linalg.lstsq(normalized, y, rcond=None)
    if rank != design.shape[1]:
        return None, "rank_deficient_fit"
    coefficients = coefficients/scale
    residual = y-design@coefficients
    result = dict(c=float(coefficients[0]), b=float(coefficients[1]),
                  n=len(x), parameter_count=design.shape[1], residual_dof=len(x)-design.shape[1],
                  x_min=float(min(x)), x_max=float(max(x)), grid=grid_string(x),
                  rmse=float(np.sqrt(np.mean(residual**2))), max_abs_residual=float(max(abs(residual))),
                  condition=float(singular[0]/singular[-1]))
    log_span = abs(result["c"])*math.log(max(x)/min(x))
    correction_span = 0.
    for i, p in enumerate(powers):
        result[f"inv{p}"] = float(coefficients[i+2])
        span = abs(coefficients[i+2])*(min(x)**(-p)-max(x)**(-p))
        correction_span += span
        result[f"inverse{p}_to_log_span"] = ratio(span, log_span)
    result["total_correction_to_log_span"] = ratio(correction_span,log_span)
    return result, None


def predict(result, x):
    x = np.asarray(x, dtype=float)
    return result["c"]*np.log(x)+result["b"]+sum(result.get(f"inv{p}",0)*x**(-p) for p in (1,2,3))


def analyze(rows, betas, windows, theta_min, ratio_max, eq_max):
    equilibrium = defaultdict(dict)
    finite = defaultdict(dict)
    for row in rows:
        target, key, x = (equilibrium,row["beta"],row["L"]) if math.isinf(row["theta"]) else (finite,(row["beta"],row["L"]),row["theta"])
        target[key][x] = row["D"]
    eq_fits, individual, pairs, missing = [], [], [], []
    for beta in betas:
        for low, model in itertools.product((18,36,54), ("log_inv1","log_inv1_inv2")):
            data = equilibrium.get(beta,{})
            x = sorted(size for size in data if low <= size <= eq_max)
            meta = dict(beta=beta, model=model, requested_Lmin=low, requested_Lmax=eq_max)
            result, reason = fit(x,[data[size] for size in x],model)
            if result:
                eq_fits.append(dict(**meta,**result,extent_complete=x[0]==low and x[-1]==eq_max))
            else:
                missing.append(dict(domain="equilibrium",**meta,reason=reason,n=len(x)))
        sizes = sorted(size for b,size in finite if b == beta)
        for size in sizes:
            data = finite[beta,size]
            for low,high in windows:
                x = sorted(t for t in data if low <= t <= high)
                meta = dict(beta=beta,L=size,window=f"{low:g}-{high:g}",theta_min=low,theta_max=high,
                            theta_over_L_min=low/size,theta_over_L_max=high/size,
                            theta_over_region_width_max=3*high/size,
                            scale_screen=low>=theta_min and high/size<=ratio_max)
                if not x or x[0] != low or x[-1] != high:
                    missing.append(dict(domain="time_individual",**meta,reason="both_window_endpoints_required",n=len(x)))
                    continue
                for model in MODELS:
                    result,reason=fit(x,[data[t] for t in x],model)
                    if result:
                        individual.append(dict(**meta,model=model,**result))
                    else:
                        missing.append(dict(domain="time_individual",**meta,model=model,reason=reason,n=len(x)))
        for small,large in itertools.combinations(sizes,2):
            a,b=finite[beta,small],finite[beta,large]
            for low,high in windows:
                x=sorted(t for t in a.keys()&b.keys() if low<=t<=high)
                meta=dict(beta=beta,L_small=small,L_large=large,window=f"{low:g}-{high:g}",theta_min=low,theta_max=high,
                          theta_over_L_small_max=high/small,theta_over_L_large_max=high/large,
                          theta_over_region_width_large_max=3*high/large,
                          scale_screen_large=low>=theta_min and high/large<=ratio_max,
                          scale_screen_both=low>=theta_min and high/small<=ratio_max)
                if not x or x[0]!=low or x[-1]!=high:
                    missing.append(dict(domain="time_pair",**meta,reason="both_common_window_endpoints_required",n=len(x)))
                    continue
                for model in MODELS:
                    fs,reason=fit(x,[a[t] for t in x],model)
                    fl,_=fit(x,[b[t] for t in x],model)
                    if fs is None or fl is None:
                        missing.append(dict(domain="time_pair",**meta,model=model,reason=reason,n=len(x)))
                        continue
                    delta=max(abs(a[t]-b[t]) for t in x)
                    pairs.append(dict(**meta,model=model,**fl,c_small=fs["c"],
                                      size_drift=ratio(abs(fl["c"]-fs["c"]),abs(fl["c"])),
                                      size_sign_changed=fl["c"]*fs["c"]<=0,
                                      max_absolute_D_difference=delta,
                                      D_difference_to_log_span=ratio(delta,abs(fl["c"])*math.log(high/low))))
    return eq_fits, individual, pairs, missing, equilibrium, finite


def explicit_primary(args, plan):
    result=dict(plan.get("primary",{}))
    by_beta=plan.get("primary_by_beta",result.pop("by_beta",{}))
    if args.primary_L is not None: result["L"]=args.primary_L
    if args.comparison_L is not None: result["comparison_L"]=args.comparison_L
    if args.primary_window is not None: result["window"]=args.primary_window
    if args.primary_model is not None: result["model"]=args.primary_model
    overrides={canonical(beta):dict(value) for beta,value in by_beta.items()}
    checks=[result] if result else []
    checks.extend({**result,**value} for value in overrides.values())
    for check in checks:
        if not all(k in check for k in ("L","window","model")):
            raise ValueError("Primary configuration requires L, window and model together")
        if check["model"] not in MODELS:
            raise ValueError("Unknown primary model")
    return dict(default=result,by_beta=overrides) if result or overrides else {}


def primary_for_beta(configuration,beta):
    if not configuration:
        return {}
    return {**configuration["default"],**configuration["by_beta"].get(beta,{})}


def primary_rows(betas, configuration, eq_fits, individual, pairs, finite, eq_max,theta_min,ratio_max):
    result=[]
    for beta in betas:
        primary=primary_for_beta(configuration,beta)
        eq=next((r for r in eq_fits if r["beta"]==beta and r["requested_Lmin"]==18 and r["model"]=="log_inv1" and r["extent_complete"]),None)
        row=dict(beta=beta,c_eq=eq["c"] if eq else None,eq_status="available" if eq else "missing_complete_18_to_max_extent",neq_status="primary_not_selected")
        eq_range=[r["c"] for r in eq_fits if r["beta"]==beta and r["extent_complete"]]
        row.update(eq_range_min=min(eq_range) if eq_range else None,eq_range_max=max(eq_range) if eq_range else None,eq_range_n=len(eq_range))
        if primary:
            window=grid_string(primary["window"]).replace(",","-")
            candidates=[r for r in pairs if r["beta"]==beta and r["L_large"]==primary["L"] and r["window"]==window and r["model"]==primary["model"]]
            if "comparison_L" in primary:
                candidates=[r for r in candidates if r["L_small"]==primary["comparison_L"]]
            chosen=max(candidates,key=lambda r:r["L_small"]) if candidates else None
            row.update(primary_L=primary["L"],primary_window=window,primary_model=primary["model"],neq_status="missing_common_grid_fit")
            if chosen:
                accepted=chosen["scale_screen_both"]
                row.update(neq_status="finite_window_estimate" if accepted else "excluded_by_operational_scale_screen",
                           c_neq=chosen["c"] if accepted else None,c_neq_diagnostic=chosen["c"],
                           comparison_L=chosen["L_small"],grid=chosen["grid"],n=chosen["n"],
                           size_drift=chosen["size_drift"],size_sign_changed=chosen["size_sign_changed"],
                           theta_over_L=chosen["theta_over_L_large_max"],
                           theta_over_L_comparison=chosen["theta_over_L_small_max"],
                           theta_over_region_width=chosen["theta_over_region_width_large_max"],
                           scale_screen_both=chosen["scale_screen_both"],
                           D_difference_to_log_span=chosen["D_difference_to_log_span"],
                           correction_to_log_span=chosen["total_correction_to_log_span"])
                model_checks=[r for r in pairs if r["beta"]==beta and r["L_small"]==chosen["L_small"] and r["L_large"]==chosen["L_large"] and r["window"]==window]
                window_checks=[r for r in individual if r["beta"]==beta and r["L"]==chosen["L_large"] and r["model"]==chosen["model"] and r["scale_screen"]]
                model_drifts=[ratio(abs(r["c"]-chosen["c"]),abs(chosen["c"])) for r in model_checks]
                window_drifts=[ratio(abs(r["c"]-chosen["c"]),abs(chosen["c"])) for r in window_checks]
                row.update(model_drift=max((x for x in model_drifts if x is not None),default=None),
                           window_drift=max((x for x in window_drifts if x is not None),default=None),
                           model_check_count=len(model_checks),window_check_count=len(window_checks),
                           model_sign_changed=any(r["c"]*chosen["c"]<=0 for r in model_checks),
                           window_sign_changed=any(r["c"]*chosen["c"]<=0 for r in window_checks),
                           window_check_scope="individual measured grids at primary L; each window has separate size-check status")
                band=[r["c"] for r in individual if r["beta"]==beta and r["L"]==chosen["L_large"] and r["scale_screen"]]
                row.update(neq_range_min=min(band) if band and accepted else None,
                           neq_range_max=max(band) if band and accepted else None,neq_range_n=len(band))
                grid=[float(x) for x in chosen["grid"].split(",")]
                third=[size for b,size in finite if b==beta and size not in (chosen["L_small"],chosen["L_large"])
                       and set(grid)<=finite[b,size].keys() and max(grid)/size<=ratio_max and min(grid)>=theta_min]
                if third:
                    larger=[size for size in third if size>chosen["L_large"]]
                    size=min(larger) if larger else max(third)
                    third_fit,_=fit(grid,[finite[beta,size][t] for t in grid],chosen["model"])
                    if size>chosen["L_large"]:
                        extra=ratio(abs(third_fit["c"]-chosen["c"]),abs(third_fit["c"]))
                        decreasing=extra<chosen["size_drift"] if extra is not None and chosen["size_drift"] is not None else None
                        side="above_primary"
                    else:
                        extra=ratio(abs(third_fit["c"]-chosen["c_small"]),abs(chosen["c_small"]))
                        decreasing=chosen["size_drift"]<extra if extra is not None and chosen["size_drift"] is not None else None
                        side="below_comparison"
                    row.update(third_L=size,c_third=third_fit["c"],additional_size_drift=extra,
                               third_size_position=side,size_drift_decreasing=decreasing)
                available_checks=row["model_check_count"]==len(MODELS) and row["window_check_count"]>=2
                third_missing="third_L" not in row
                third_pass=third_missing or (row.get("additional_size_drift") is not None and
                           row["additional_size_drift"]<=.02 and row.get("size_drift_decreasing") is True)
                passed=accepted and available_checks and all(row.get(k) is not None and row[k]<=limit for k,limit in
                         (("size_drift",.02),("D_difference_to_log_span",.01),("model_drift",.05),("window_drift",.05))) and third_pass and not any(row[k] for k in
                         ("size_sign_changed","model_sign_changed","window_sign_changed"))
                row.update(local_diagnostic_pass=passed,
                           reliability="local_checks_only_not_certified_asymptotic" if passed else "unresolved_diagnostic_estimate",
                           third_size_missing=third_missing,
                           third_size_status="missing_two_size_local_check_only" if third_missing else "checked_stable_and_decreasing" if third_pass else "checked_unstable_or_not_decreasing",
                           third_size_gate=None if third_missing else third_pass,
                           narrow_window_warning=chosen["theta_max"]/chosen["theta_min"]<2,
                           five_parameter_residual_dof=chosen["n"]-5,
                           local_diagnostic_missing_checks=not available_checks)
                if eq and row["c_neq"] is not None:
                    row.update(ratio_neq_to_eq=ratio(row["c_neq"],eq["c"]),
                               relative_gap=ratio(row["c_neq"]-eq["c"],abs(eq["c"])))
        result.append(row)
    return result


def sensitivity_rows(coefficients,eq_fits,individual,pairs):
    result=[]
    for coefficient in coefficients:
        beta=coefficient["beta"]
        eq_anchor=next((r for r in eq_fits if r["beta"]==beta and r["requested_Lmin"]==18 and r["model"]=="log_inv1"),None)
        if eq_anchor:
            for row in eq_fits:
                if row["beta"]!=beta: continue
                result.append(dict(domain="equilibrium",beta=beta,model=row["model"],window=f"{row['requested_Lmin']}-{row['requested_Lmax']}",
                                   grid=row["grid"],c=row["c"],reference_c=eq_anchor["c"],
                                   relative_drift=ratio(abs(row["c"]-eq_anchor["c"]),abs(eq_anchor["c"])),
                                   extent_complete=row["extent_complete"],reference_extent_complete=eq_anchor["extent_complete"]))
        if coefficient.get("c_neq_diagnostic") is None: continue
        for row in individual:
            if row["beta"]!=beta or row["L"]!=coefficient["primary_L"]: continue
            pair=next((r for r in pairs if r["beta"]==beta and r["L_large"]==row["L"] and r["L_small"]==coefficient["comparison_L"]
                       and r["window"]==row["window"] and r["model"]==row["model"]),None)
            same_window=row["window"]==coefficient["primary_window"]
            same_model=row["model"]==coefficient["primary_model"]
            result.append(dict(domain="time",beta=beta,L=row["L"],model=row["model"],window=row["window"],
                               grid=row["grid"],n=row["n"],residual_dof=row["residual_dof"],c=row["c"],
                               reference_c=coefficient["c_neq_diagnostic"],reference_grid=coefficient["grid"],
                               relative_drift=ratio(abs(row["c"]-coefficient["c_neq_diagnostic"]),abs(coefficient["c_neq_diagnostic"])),
                               comparison_kind="primary_full_grid_check" if same_window and same_model else "model" if same_window else "window" if same_model else "window_and_model",
                               scale_screen_at_primary_L=row["scale_screen"],theta_over_L=row["theta_over_L_max"],
                               comparison_L=coefficient["comparison_L"],
                               size_check_status="available_within_both_size_screens" if pair and pair["scale_screen_both"] else "available_but_comparison_size_outside_screen" if pair else "missing_common_grid_fit",
                               size_check_grid=pair["grid"] if pair else None,
                               size_drift=pair["size_drift"] if pair else None,
                               scale_screen_both_sizes=pair["scale_screen_both"] if pair else None))
    return result


def coverage_check(rows, plan, betas):
    present={(r["L"],r["theta"],r["beta"]) for r in rows}
    expected=set()
    for grid in plan.get("required_finite_grids",[]):
        sizes=grid.get("sizes",grid.get("L",[]))
        times=grid.get("times",grid.get("theta",[]))
        strengths=grid.get("betas",grid.get("beta",betas))
        expected.update((int(size),canonical(time),canonical(beta)) for size,time,beta in itertools.product(sizes,times,strengths))
    for grid in plan.get("required_equilibrium_grids",[]):
        sizes=grid.get("sizes",grid.get("L",[]))
        strengths=grid.get("betas",grid.get("beta",betas))
        if any(not math.isinf(float(t)) for t in grid.get("theta",[math.inf])):
            raise ValueError("Equilibrium required grids must have theta=Inf")
        expected.update((int(size),math.inf,canonical(beta)) for size,beta in itertools.product(sizes,strengths))
    missing=[dict(L=size,theta=time,beta=beta) for size,time,beta in sorted(expected-present)]
    return len(expected),missing


def save(fig,directory,name):
    for extension in ("png","pdf"):
        fig.savefig(directory/f"{name}.{extension}",bbox_inches="tight",facecolor="white")
    plt.close(fig)


def array(rows,key):
    return np.array([np.nan if r.get(key) is None else r[key] for r in rows],dtype=float)


def coefficient_curves(ax,coefficients,passed,failed,detailed=False):
    """Keep both series visible at coincident coordinates without moving data."""
    beta=array(coefficients,"beta")
    eq_style=dict(color=BLUE,lw=2.4,linestyle="-")
    time_style=dict(color=ORANGE,lw=1.5,linestyle=(0,(4,2.8)))
    eq_marker=dict(marker="o",ms=8.2,mfc="white",mec=BLUE,mew=1.5)
    time_marker=dict(marker="s",ms=4.2,mec=ORANGE,mew=1.2)
    ax.plot(beta,array(coefficients,"c_eq"),**eq_style,zorder=2)
    ax.plot(beta,array(coefficients,"c_neq"),**time_style,zorder=3)
    ax.plot(beta,array(coefficients,"c_eq"),ls="none",**eq_marker,zorder=4,clip_on=False)
    for block,face in ((failed,"white"),(passed,ORANGE)):
        ax.plot(array(block,"beta"),array(block,"c_neq"),ls="none",mfc=face,
                **time_marker,zorder=5,clip_on=False)
    handles=[Line2D([],[],**eq_style,**eq_marker,label="Equilibrium")]
    if detailed:
        handles.extend(Line2D([],[],**time_style,**time_marker,mfc=face,label=label)
                       for face,label in ((ORANGE,"Nonequilibrium: local checks"),
                                          ("white","Nonequilibrium: unresolved diagnostic")))
    else:
        handles.append(Line2D([],[],**time_style,**time_marker,mfc="white",label="Nonequilibrium"))
    return handles


def plots(rows,eq_fits,individual,pairs,coefficients,primary,windows,directory,stage,plot_betas,theta_min,ratio_max,eq_max):
    directory.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,
                         "axes.spines.right":False,"savefig.dpi":180,"pdf.fonttype":42})
    beta=array(coefficients,"beta")
    fig,axes=plt.subplots(1,3,figsize=(16,4.7),constrained_layout=True)
    ax=axes[0]
    passed=[r for r in coefficients if r.get("local_diagnostic_pass") and r.get("c_neq") is not None]
    failed=[r for r in coefficients if not r.get("local_diagnostic_pass") and r.get("c_neq") is not None]
    handles=coefficient_curves(ax,coefficients,passed,failed,detailed=True)
    eq_band=ax.fill_between(beta,array(coefficients,"eq_range_min"),array(coefficients,"eq_range_max"),color=BLUE,alpha=.14,label="Eq. model/window range",zorder=1)
    time_band=ax.fill_between(beta,array(coefficients,"neq_range_min"),array(coefficients,"neq_range_max"),color=ORANGE,alpha=.17,label="Time model/window range",zorder=1)
    ax.set(ylabel="Single-species fitted coefficient c",title="Explicit primary estimates; linear axes")
    bounds=np.concatenate([array(coefficients,k) for k in ("c_eq","c_neq","eq_range_min","neq_range_min")])
    bounds=bounds[np.isfinite(bounds)]
    if len(bounds):
        ax.set_ylim(bottom=min(-.025*float(max(abs(bounds))),1.05*float(min(bounds))))
    ax.axhline(0,color="0.55",lw=.6,alpha=.6,zorder=0)
    ax.legend(handles=handles+[eq_band,time_band],fontsize=8,handlelength=3)
    axes[1].plot(beta,array(coefficients,"ratio_neq_to_eq"),"s-",color=ORANGE,mfc="white",label="Nonequilibrium / equilibrium")
    axes[1].plot(array(passed,"beta"),array(passed,"ratio_neq_to_eq"),"s",color=ORANGE)
    axes[1].axhline(1,color=BLUE,lw=1,label="Equilibrium reference")
    axes[1].set(ylabel="Coefficient ratio",title="Small signals remain visible")
    axes[1].legend(fontsize=8)
    axes[2].plot(beta,100*array(coefficients,"relative_gap"),"s-",color=ORANGE,mfc="white",label="Relative gap")
    axes[2].plot(array(passed,"beta"),100*array(passed,"relative_gap"),"s",color=ORANGE)
    axes[2].axhline(0,color=BLUE,lw=1)
    axes[2].set(ylabel="(Time c - equilibrium c) / |equilibrium c| (%)",title="Comparison only; never a selection target")
    for ax in axes:
        ax.set_xlabel(r"Measurement strength $\beta$")
        ax.set_xlim(-.035,2.035)
        ax.set_xticks(np.linspace(0,2,9))
        ax.grid(axis="y",alpha=.25)
    selection="Primary not selected"
    if primary and not primary["by_beta"]:
        configured=primary["default"]
        selection=f"Time: L={configured['L']} vs {configured.get('comparison_L','largest lower complete L')}, theta={configured['window'][0]:g}..{configured['window'][1]:g}, {configured['model']}"
    elif primary:
        selection="Explicit per-beta time configurations; see primary_coefficients.csv"
    counts=f"Available: eq {sum(r.get('c_eq') is not None for r in coefficients)}/{len(coefficients)}, time {sum(r.get('c_neq') is not None for r in coefficients)}/{len(coefficients)}"
    if not any(r.get("c_neq") is not None for r in coefficients):
        axes[1].text(.5,.7,"No complete primary time estimate yet",transform=axes[1].transAxes,ha="center",fontsize=9)
    fig.suptitle(f"uniform_A dense beta: {stage}; {counts}\n{selection}; shaded available-fit ranges are not confidence intervals",fontsize=11)
    save(fig,directory,"coefficients_beta")

    fig,ax=plt.subplots(figsize=(7,4.6),constrained_layout=True)
    ax.fill_between(beta,array(coefficients,"eq_range_min"),array(coefficients,"eq_range_max"),
                    color=BLUE,alpha=.12)
    ax.fill_between(beta,array(coefficients,"neq_range_min"),array(coefficients,"neq_range_max"),
                    color=ORANGE,alpha=.14)
    handles=coefficient_curves(ax,coefficients,passed,failed)
    ax.set(xlabel=r"Measurement strength $\beta$",ylabel="Fitted corner coefficient c",xlim=(-.035,2.035))
    ax.set_xticks(np.linspace(0,2,9))
    if len(bounds):
        ax.set_ylim(bottom=min(-.025*float(max(abs(bounds))),1.05*float(min(bounds))))
    ax.axhline(0,color="0.55",lw=.6,alpha=.6,zorder=0)
    ax.grid(axis="y",alpha=.22)
    ax.legend(handles=handles,fontsize=9,handlelength=3)
    eq_limit=eq_max
    time_title="time configuration not selected"
    if primary and not primary["by_beta"]:
        configured=primary["default"]
        time_title=f"time L{configured['L']}, theta {configured['window'][0]:g}..{configured['window'][1]:g}"
    elif primary:
        time_title="time configurations specified per beta"
    ax.set_title(f"uniform_A: eq L18..{eq_limit:g}; {time_title}\n"
                 f"{stage}; time {len(passed)+len(failed)}/{len(coefficients)} available; one species, natural logarithms",fontsize=10)
    ax.text(.5,-.22,"Filled/open squares: local checks pass/unresolved.\nBands: available model/window ranges, not confidence intervals.",
            transform=ax.transAxes,ha="center",va="top",fontsize=8)
    save(fig,directory,"corner_coefficient_beta")

    fig,axes=plt.subplots(1,3,figsize=(16,4.7),constrained_layout=True)
    for ax,key,title in zip(axes,("size_drift","window_drift","model_drift"),
                            ("Common-grid size drift","Predefined-window sensitivity","Correction-model sensitivity")):
        ax.plot(beta,100*array(coefficients,key),"s-",color=ORANGE)
        ax.set(xlabel=r"$\beta$",ylabel="Relative drift (%)",title=title)
        ax.set_xlim(min(beta)-.02*(max(beta)-min(beta)),max(beta)+.02*(max(beta)-min(beta)))
        ax.grid(axis="y",alpha=.25)
    fig.suptitle(f"uniform_A: time-only diagnostics ({stage}); no equilibrium coefficient enters these differences")
    save(fig,directory,"time_diagnostics_beta")

    plotted=[]
    for value in plot_betas:
        block=[r for r in rows if r["beta"]==value]
        if not block: continue
        fig,axes=plt.subplots(1,2,figsize=(12,4.7),constrained_layout=True)
        eq=sorted([r for r in block if math.isinf(r["theta"]) and 18<=r["L"]<=eq_max],key=lambda r:r["L"])
        if eq:
            axes[0].plot(np.log([r["L"] for r in eq]),[r["D"] for r in eq],"o",color=BLUE,label="Equilibrium data")
        fitted=next((r for r in eq_fits if r["beta"]==value and r["requested_Lmin"]==18 and r["model"]=="log_inv1"),None)
        if fitted:
            x=np.geomspace(fitted["x_min"],fitted["x_max"],200)
            axes[0].plot(np.log(x),predict(fitted,x),color=BLUE,label=f"Fit on L={fitted['x_min']:g}..{fitted['x_max']:g}")
        finite=[r for r in block if math.isfinite(r["theta"])]
        coefficient=next(r for r in coefficients if r["beta"]==value)
        configured=primary_for_beta(primary,value)
        allowed_sizes=set()
        if configured:
            allowed_sizes.add(configured["L"])
            comparison=coefficient.get("comparison_L",configured.get("comparison_L"))
            if comparison is not None:
                allowed_sizes.add(comparison)
            if coefficient.get("third_L") is not None:
                allowed_sizes.add(coefficient["third_L"])
            else:
                larger=sorted({r["L"] for r in finite if r["L"]>configured["L"]
                               and r["theta"]>=theta_min and r["theta"]/r["L"]<=ratio_max})
                if larger:
                    allowed_sizes.add(larger[0])
        finite=[r for r in finite if r["L"] in allowed_sizes and r["theta"]>=theta_min and r["theta"]/r["L"]<=ratio_max]
        sizes=sorted({r["L"] for r in finite})
        colors=plt.get_cmap("viridis")(np.linspace(.1,.9,max(len(sizes),1)))
        for size,color in zip(sizes,colors):
            points=sorted([r for r in finite if r["L"]==size],key=lambda r:r["theta"])
            if points:
                axes[1].plot(np.log([r["theta"] for r in points]),[r["D"] for r in points],"o-",color=color,ms=3,lw=.7,label=f"L={size}")
        if configured and coefficient.get("c_neq") is not None:
            chosen=next(r for r in pairs if r["beta"]==value and r["L_large"]==configured["L"] and r["L_small"]==coefficient["comparison_L"] and r["window"]==coefficient["primary_window"] and r["model"]==configured["model"])
            x=np.geomspace(chosen["theta_min"],chosen["theta_max"],200)
            label="Explicit primary fit" if coefficient.get("local_diagnostic_pass") else "Primary fit: unresolved diagnostic"
            axes[1].plot(np.log(x),predict(chosen,x),color=ORANGE,lw=2,label=label)
        else:
            notice="Primary time fit not yet available" if stage=="preliminary" else "No primary time fit within scale screens"
            if not finite:
                notice="No time data within scale screens\n"+notice
            axes[1].text(.98,.05,notice,transform=axes[1].transAxes,ha="right",va="bottom",fontsize=8,
                         bbox=dict(facecolor="white",alpha=.85,edgecolor="none"))
        axes[0].set(xlabel=r"$\ln L$",ylabel="D",title="Equilibrium scaling")
        axes[1].set(xlabel=r"$\ln\theta$",ylabel="D",title=f"Time data: theta >= {theta_min:g}, theta/L <= {ratio_max:g}")
        for ax in axes:
            ax.set_xscale("linear")
            ax.grid(axis="y",alpha=.25)
            if ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=7,ncol=2)
        fig.suptitle(f"uniform_A, beta={value:g}: {stage}; curves outside the primary window remain diagnostics")
        name=f"scaling_beta_{value:g}".replace(".","p")
        save(fig,directory,name)
        plotted.append(name)
    return ["coefficients_beta","corner_coefficient_beta","time_diagnostics_beta"]+plotted


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs",nargs="+",type=Path,default=DEFAULT_INPUTS)
    parser.add_argument("--output",type=Path,default=HERE/"analysis_uniformA_dense")
    parser.add_argument("--figures",type=Path,default=HERE/"figures_uniformA_dense")
    parser.add_argument("--plan",type=Path,default=HERE/"analysis_uniformA_dense/plan.json")
    parser.add_argument("--stage",choices=("preliminary","final"),default="preliminary")
    parser.add_argument("--primary-L",type=int)
    parser.add_argument("--comparison-L",type=int)
    parser.add_argument("--primary-window",nargs=2,type=float)
    parser.add_argument("--primary-model",choices=tuple(MODELS))
    parser.add_argument("--theta-min",type=float)
    parser.add_argument("--max-theta-over-L",type=float)
    parser.add_argument("--eq-max",type=int)
    parser.add_argument("--plot-betas",nargs="+",type=float,default=[0,.5,1,1.5,2])
    parser.add_argument("--no-plots",action="store_true")
    args=parser.parse_args()
    plan_bytes=args.plan.read_bytes() if args.plan else b"{}"
    plan=json.loads(plan_bytes)
    args.theta_min=args.theta_min if args.theta_min is not None else float(plan.get("theta_min_screen",4))
    args.max_theta_over_L=args.max_theta_over_L if args.max_theta_over_L is not None else float(plan.get("theta_over_L_max_screen",.05))
    args.eq_max=args.eq_max if args.eq_max is not None else int(plan.get("equilibrium_primary",{}).get("window",[18,144])[1])
    betas=sorted(canonical(b) for b in plan.get("betas",[i/10 for i in range(21)]))
    windows=[tuple(map(float,w)) for w in plan.get("windows",DEFAULT_WINDOWS)]
    primary=explicit_primary(args,plan)
    for beta in betas:
        configured=primary_for_beta(primary,beta)
        if configured and tuple(configured["window"]) not in windows:
            raise ValueError("Primary window must be explicitly listed among the diagnostic windows")
    if args.stage=="final" and (not primary or not plan.get("required_finite_grids") or not plan.get("required_equilibrium_grids")):
        raise ValueError("Final stage requires explicit primary and required finite/equilibrium grids in --plan")
    args.output.mkdir(parents=True,exist_ok=True)
    rows,sources,duplicates=read_sources(args.inputs,args.stage,args.output,set(betas))
    expected,missing_keys=coverage_check(rows,plan,betas)
    if args.stage=="final" and missing_keys:
        raise ValueError(f"Incomplete required grid: {len(missing_keys)} keys")
    eq,individual,pairs,missing,eq_data,time_data=analyze(rows,betas,windows,args.theta_min,args.max_theta_over_L,args.eq_max)
    coefficients=primary_rows(betas,primary,eq,individual,pairs,time_data,args.eq_max,args.theta_min,args.max_theta_over_L)
    sensitivities=sensitivity_rows(coefficients,eq,individual,pairs)
    if args.stage=="final" and any(r.get("c_eq") is None or r.get("c_neq") is None for r in coefficients):
        raise ValueError("Final stage requires complete primary estimates and operational scale screening for all beta values")
    coverage=[]
    groups=defaultdict(list)
    for row in rows: groups[row["beta"],row["L"]].append(row)
    for (beta,size),block in sorted(groups.items()):
        times=sorted(r["theta"] for r in block if math.isfinite(r["theta"]))
        coverage.append(dict(beta=beta,L=size,equilibrium_present=any(math.isinf(r["theta"]) for r in block),
                             finite_n=len(times),times=grid_string(times),
                             theta_over_L_max=max(times)/size if times else None))
    for name,values in (("observations",rows),("duplicate_observations",duplicates),("coverage",coverage),
                        ("equilibrium_fits",eq),("time_individual_fits",individual),("time_pair_fits",pairs),
                        ("missing_fits",missing),("missing_required_keys",missing_keys),("primary_coefficients",coefficients),
                        ("time_sensitivity",sensitivities)):
        dump_csv(args.output/f"{name}.csv",values)
    figures=[] if args.no_plots else plots(rows,eq,individual,pairs,coefficients,primary,windows,args.figures,args.stage,args.plot_betas,
                                         args.theta_min,args.max_theta_over_L,args.eq_max)
    changed=[r["path"] for r in sources if not r.get("missing") and sha_bytes(Path(r["path"]).read_bytes())!=r["sha256"]]
    if args.stage=="final" and changed:
        raise ValueError(f"Inputs changed during final analysis: {changed}; rerun after scan completion")
    summary=dict(generated_utc=datetime.now(timezone.utc).isoformat(),stage=args.stage,
                 script_sha256=sha_bytes(Path(__file__).read_bytes()),plan_sha256=sha_bytes(plan_bytes),plan=plan,
                 arguments={k:str(v) if isinstance(v,Path) else [str(x) for x in v] if k=="inputs" else v for k,v in vars(args).items()},
                 sources=sources,sources_changed_during_analysis=changed,rows=len(rows),duplicates=len(duplicates),
                 betas=betas,windows=windows,primary=primary,required_keys=expected,
                 required_grid_complete=bool(expected) and not missing_keys,missing_required_keys=len(missing_keys),
                 primary_complete=all(r.get("c_eq") is not None and r.get("c_neq") is not None for r in coefficients),
                 fit_counts=dict(equilibrium=len(eq),time_individual=len(individual),time_pairs=len(pairs)),figures=figures,
                 conventions=dict(entropy="single-species natural-log Renyi-2",observable="D=(2SA+SC)/3-SB",
                                  time_fit="measured common grids, both window endpoints required, n>=parameters+2",
                                  primary="explicit external configuration; never selected using equilibrium closeness",
                                  scale_screen="theta_min>=configured minimum and theta_max/L<=configured maximum for BOTH primary/comparison sizes; operational only, not a proof",
                                  sensitivity="main time value and same-window model checks use the explicit common-grid size pair; shifted-window checks and shaded ranges use individual measured grids at primary L, with separate per-window size-check status; ranges are not statistical confidence intervals",
                                  axes="raw scaling uses explicit ln(L)/ln(theta) on linear axes; beta and coefficient axes are linear",
                                  raw_scaling_scope="equilibrium only within requested 18..eq_max; time only configured primary/comparison and available third size, with each point satisfying theta_min and theta/L screens; fit line only inside configured primary window",
                                  missing="missing/screen-excluded primary values remain gaps, never extrapolated across beta"))
    (args.output/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps({k:summary[k] for k in ("stage","rows","fit_counts","primary_complete","missing_required_keys","sources_changed_during_analysis")},indent=2))


if __name__=="__main__":
    main()
