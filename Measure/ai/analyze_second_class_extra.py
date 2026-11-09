"""Separate analysis of four additional class-2 occupation patterns.

Only numerical fit helpers are reused. A blocked initial configuration has an
accessible excited-state limit, not an equilibrium ground state. Public tables
and plots retain that distinction even if a numerical fit is locally stable.
All new output stays in this experiment's own directories.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import io
import itertools
import json
import math
from pathlib import Path
import re
import tomllib

import numpy as np
import analyze_occupancy_classes as old
from analyze_occupancy_classes import core, previous
import plot_occupancy_classes_patterns as schematic
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "analysis_second_class_extra"
FIGURES = HERE / "figures_second_class_extra"
ENTROPIES = ("S_A", "S_C", "S_B", "D")


def sha(path):
    return core.sha_bytes(path.read_bytes())


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def boolean(value):
    if isinstance(value, bool):
        return value
    if str(value).lower() not in ("true", "false"):
        raise ValueError(f"Expected an explicit boolean, got {value!r}")
    return str(value).lower() == "true"


def load_plan(path):
    content = path.read_bytes()
    plan = json.loads(content.decode("utf-8-sig"))
    # The old helper's 'equilibrium' key is an internal mathematical selector.
    private = dict(plan, pattern_metadata={name:dict(meta, equilibrium=meta["limit_fit"])
                        for name, meta in plan["pattern_metadata"].items()})
    names, catalog = old.catalog_from_plan(private)
    for name in names:
        raw = plan["pattern_metadata"][name]
        reachable = boolean(raw["ground_reachable"])
        state = "ground_state" if reachable else "accessible_excited_state"
        if raw["limit_state"] != state or catalog[name]["occupancy_class"] != 2:
            raise ValueError(f"Inconsistent class or physical state in plan: {name}")
        catalog[name].update(ground_reachable=reachable, limit_state=state,
                             projection_method=raw["projection_method"])
    if plan["primary"] != dict(L=180, comparison_L=144, window=[4,7], model="log_inv1_inv2"):
        raise ValueError("This experiment must retain the frozen previous time-fit protocol")
    if plan["windows"] != old.DEFAULT_WINDOWS:
        raise ValueError("Do not tune diagnostic windows after inspecting new entropies")
    if plan["theta_min_screen"] != 4 or plan["theta_over_L_max_screen"] != .05:
        raise ValueError("Operational scale screens differ from the frozen protocol")
    for relative, expected in plan.get("frozen_code_sha256", {}).items():
        if sha(HERE/relative) != expected:
            raise ValueError(f"Frozen numerical source changed: {relative}")
    return plan, content, names, catalog


def expected_keys(plan, names):
    keys = set()
    for kind in ("required_limit_grids", "required_finite_grids"):
        for grid in plan[kind]:
            for name, size, theta, beta in itertools.product(grid.get("patterns", names),
                    grid["L"], grid.get("theta", ["Inf"]), grid.get("beta", plan["betas"])):
                if kind == "required_limit_grids" and not math.isinf(float(theta)):
                    raise ValueError("Limit grid must have theta=Inf")
                keys.add((name, int(size), core.canonical(theta), core.canonical(beta)))
    if len(keys) != plan["expected_observation_count"]:
        raise ValueError("Planned observation count does not match its grids")
    return keys


def manifest_method(row, csv_path, plan, stage, dependencies, manifests):
    run_id=row.get("run_id", "")
    if not re.fullmatch(r"[A-Za-z0-9_-]+",run_id):raise ValueError("Missing or unsafe run id")
    path=HERE/"data"/f"second_class_extra_run_{run_id}.toml"
    if run_id not in manifests:
        content=path.read_bytes();manifest=tomllib.loads(content.decode("utf-8-sig"))
        dependencies[path]=content
        protocol=manifest.get("protocol","")
        if "Exact spectral-support finite theta or accessible Inf" not in protocol or "blocked Inf is an excited state" not in protocol:
            raise ValueError("Run manifest does not declare the exact-support protocol")
        if stage=="final" and manifest["status"]!="complete":raise ValueError("Final input contains an unfinished run")
        if manifest["run_id"]!=run_id or Path(manifest["output"]).resolve()!=csv_path.resolve():
            raise ValueError("Manifest run/output mismatch")
        for relative,digest in manifest["code_sha256"].items():
            snapshot=HERE/"data/provenance/code"/digest/relative
            data=snapshot.read_bytes()
            if core.sha_bytes(data)!=digest:raise ValueError(f"Code snapshot mismatch: {snapshot}")
            dependencies[snapshot]=data
            if relative in plan["frozen_code_sha256"] and digest!=plan["frozen_code_sha256"][relative]:
                raise ValueError(f"Run used different frozen source: {relative}")
        required=("second_class_extra_experiments.jl","second_class_extra_patterns.jl","fermion_experiment.jl","analysis_second_class_extra/plan.json")
        if any(relative not in manifest["code_sha256"] for relative in required):
            raise ValueError("Run manifest omits a required physical/protocol dependency")
        if manifest["code_sha256"][required[-1]]!=sha(OUTPUT/"plan.json"):
            raise ValueError("Run was not computed against the frozen analysis plan")
        manifests[run_id]=manifest
    manifest=manifests[run_id]
    if row.get("code_sha256")!=manifest["code_sha256"]["second_class_extra_experiments.jl"]:
        raise ValueError("Observation runner SHA differs from its manifest")
    if (row["pattern"] not in manifest["patterns"] or int(row["L"]) not in manifest["L"]
        or core.canonical(row["theta"]) not in {core.canonical(x) for x in manifest["theta"]}
        or core.canonical(row["beta"]) not in {core.canonical(x) for x in manifest["beta"]}):
        raise ValueError("Observation is outside its run manifest's declared job grid")
    return "strict_spectral_support",core.relative_path(path),core.sha_bytes(dependencies[path])


def input_sources(paths, catalog, expected, stage, plan, dependencies):
    """An independent reader permits blocked states without weakening old guards."""
    sources, selected, duplicates, manifests = [], {}, [], {}
    snapshots = OUTPUT/"source_snapshots"
    snapshots.mkdir(parents=True, exist_ok=True)
    for path in paths:
        if not path.exists():
            sources.append(dict(path=str(path.resolve()), missing=True))
            continue
        original = path.read_bytes()
        content, ignored = original, 0
        if content and not content.endswith(b"\n"):
            if stage == "final":
                raise ValueError(f"Incomplete CSV tail: {path}")
            end = content.rfind(b"\n")+1
            ignored, content = len(content)-end, content[:end]
        digest = core.sha_bytes(original)
        snapshot = snapshots/f"{path.stem}_{digest[:12]}.csv"
        snapshot.write_bytes(content)
        reader = csv.DictReader(io.StringIO(content.decode("utf-8-sig")))
        column_count = len(reader.fieldnames or [])
        if column_count != 41:
            raise ValueError(f"Expected the actual 41-column BP CSV schema, got {column_count}: {path}")
        raw = list(reader)
        sources.append(dict(path=str(path.resolve()), sha256=digest, rows=len(raw),
            input_column_count=column_count,
            ignored_incomplete_tail_bytes=ignored, parsed_snapshot=core.relative_path(snapshot),
            parsed_sha256=core.sha_bytes(content)))
        for number, row in enumerate(raw, 2):
            name = row["pattern"]
            if name not in catalog:
                raise ValueError(f"Unexpected pattern: {path}:{number}: {name}")
            meta = catalog[name]
            size, theta, beta = int(row["L"]), core.canonical(row["theta"]), core.canonical(row["beta"])
            key = name, size, theta, beta
            if key not in expected:
                raise ValueError(f"Unplanned observation must not silently enter a fit: {key}")
            if int(row["m"]) != meta["m"] or row["occupied_bits"] != meta["occupied_bits"]:
                raise ValueError(f"Wrong occupation mask: {key}")
            if size % (3*meta["m"]):
                raise ValueError(f"Incompatible cell/cut phase: {key}")
            for field in ("x0", "y0", "cut_x0", "cut_y0"):
                if row.get(field) and int(row[field]) != 1:
                    raise ValueError(f"Wrong cut origin: {key}")
            reachable = boolean(row.get("ground_reachable", row.get("premeasurement_ground")))
            if reachable != meta["ground_reachable"]:
                raise ValueError(f"Asymptotic GS reachability changed: {key}")
            excess = float(row["accessible_excess"])
            if not math.isfinite(excess) or (abs(excess) > 1e-8 if reachable else excess <= 1e-8):
                raise ValueError(f"Accessible-limit energy contradicts reachability: {key}")
            method,manifest_path,manifest_hash = manifest_method(row,path,plan,stage,dependencies,manifests)
            if row.get("projection_method") and row["projection_method"]!=method:
                raise ValueError("Explicit row projection method contradicts its manifest")
            if method != meta["projection_method"]:
                raise ValueError(f"Unexpected projection method: {key}: {method}")
            if not reachable and method != "strict_spectral_support":
                raise ValueError("Blocked data require preservation of exact initial spectral support")
            values = {field:float(row[field]) for field in ENTROPIES}
            if not all(math.isfinite(x) for x in values.values()):
                raise ValueError(f"Nonfinite entropy: {key}")
            if abs((2*values["S_A"]+values["S_C"])/3-values["S_B"]-values["D"]) > 1e-9:
                raise ValueError(f"Weighted D identity failed: {key}")
            record = dict(pattern=name, m=meta["m"], occupied_bits=meta["occupied_bits"],
                occupancy_class=2, L=size, theta=theta, beta=beta, x0=1, y0=1, **values,
                ground_reachable=reachable, limit_state=meta["limit_state"],
                state_role=meta["limit_state"]+"_limit" if math.isinf(theta) else "finite_imaginary_time",
                accessible_excess=excess, projection_method=method,
                projection_evidence="frozen_runner_and_code_snapshots_plus_run_manifest",
                run_manifest=manifest_path,run_manifest_sha256=manifest_hash,
                source_file=core.relative_path(path), source_row=number, source_sha256=digest,
                run_id=row.get("run_id", ""), code_sha256=row.get("code_sha256", ""),
                entropy_method=row.get("entropy_method", ""))
            if key in selected:
                prior = selected[key]
                error = max(abs(values[field]-prior[field]) for field in ENTROPIES)
                if error > 1e-9:
                    raise ValueError(f"Conflicting duplicate observations: {key}")
                duplicates.append(dict(pattern=name, L=size, theta=theta, beta=beta,
                    previous_file=prior["source_file"], previous_row=prior["source_row"],
                    selected_file=record["source_file"], selected_row=number, max_entropy_difference=error))
            selected[key] = record
    return sorted(selected.values(), key=lambda r:(r["pattern"],r["beta"],r["L"],r["theta"])), sources, duplicates


def public_row(name, meta, source, role):
    """No public eq/neq shorthand calls a blocked endpoint an equilibrium GS."""
    special = {"c_eq":"c_limit_fit", "c_neq":"c_time_fit", "c_neq_diagnostic":"c_time_diagnostic",
               "ratio_neq_to_eq":"ratio_time_to_limit_fit", "relative_gap":"relative_fit_gap",
               "local_diagnostic_pass":"numeric_diagnostic_pass"}
    row = {}
    for key, value in source.items():
        new = special.get(key, "limit_"+key[3:] if key.startswith("eq_") else
                          "time_"+key[4:] if key.startswith("neq_") else key)
        if key == "domain" and value == "equilibrium": value = "limit"
        row[new] = value
    if "numeric_diagnostic_pass" in row:
        row["ground_state_local_checks_pass"] = bool(meta["ground_reachable"] and row["numeric_diagnostic_pass"])
        row["reliability"] = ("blocked_scaling_unresolved_descriptive_fit_only" if not meta["ground_reachable"]
             else "local_checks_only_not_certified_asymptotic" if row["numeric_diagnostic_pass"]
             else "unresolved_finite_window_fit")
    return dict(pattern=name, m=meta["m"], ground_reachable=meta["ground_reachable"],
        limit_state=meta["limit_state"], state_role=role,
        parameter_meaning="finite_window_log_fit_not_certified_corner_coefficient" if meta["ground_reachable"]
            else "accessible_excited_state_fit_not_ground_state_corner_coefficient", **row)


def local_slopes(rows, catalog):
    groups, result = defaultdict(list), []
    for row in rows:
        domain = "limit" if math.isinf(row["theta"]) else "time"
        groups[row["pattern"],row["beta"],domain,None if domain=="limit" else row["L"]].append(row)
    for (name,beta,domain,size), values in groups.items():
        xkey = "L" if domain=="limit" else "theta"
        points = sorted(values,key=lambda r:r[xkey])
        for left,right in zip(points,points[1:]):
            x1,x2 = left[xkey],right[xkey]
            result.append(public_row(name,catalog[name],dict(beta=beta,domain=domain,L=size,
                x_left=x1,x_right=x2,x_geometric_mean=math.sqrt(x1*x2),D_left=left["D"],D_right=right["D"],
                local_log_slope=(right["D"]-left["D"])/math.log(x2/x1),
                diagnostic="adjacent raw secant; not an asymptotic coefficient"),domain))
    return result


def save(fig, stem, figures):
    core.save(fig, FIGURES, stem)
    figures.append(stem)


def state_label(meta):
    return "GS reachable; equilibrium endpoint" if meta["ground_reachable"] else "GS BLOCKED; accessible excited-state limit"


def limit_label(meta):
    return "Equilibrium entropy" if meta["ground_reachable"] else "Accessible-limit entropy (not GS)"


def axes_style(ax):
    ax.grid(axis="y",alpha=.22)
    if ax.get_legend_handles_labels()[0]:ax.legend(fontsize=8)


def catalogue(names,catalog,figures):
    fig,axes=plt.subplots(2,2,figsize=(10,11))
    fig.subplots_adjust(left=.08,right=.96,bottom=.20,top=.79,hspace=.80,wspace=.32)
    for ax,name in zip(axes.flat,names):
        meta=catalog[name]
        schematic.draw_panel(ax,dict(pattern=name,m=meta["m"],occupied_bits=meta["occupied_bits"],
                                     cell_counts=[meta["empty_cells"],0,meta["double_cells"]]))
        ax.text(.5,-.33,"GS reachable" if meta["ground_reachable"] else "GS BLOCKED: excited-state limit",
                transform=ax.transAxes,ha="center",fontsize=10)
    handles=[Line2D([],[],marker="os"[s],linestyle="none",ms=7,
        mec=schematic.COLORS[s],mfc=schematic.COLORS[s] if occupied else "white",
        label=f"{'AB'[s]} {'occupied' if occupied else 'empty'}") for s,occupied in itertools.product((0,1),(True,False))]
    fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.5,.95),ncol=4,frameon=False)
    fig.suptitle("Additional class-2 patterns: only empty / doubly occupied cells",y=.99,fontsize=14)
    fig.text(.5,.025,"Occupation tiles in cell coordinates. Reachability checked on planned sizes; no scaling guarantee.\n"
             "Blocked alternating patterns remain included as excited-limit comparisons, never as equilibrium GS data.",ha="center",fontsize=10)
    save(fig,"pattern_catalogue",figures)
    for name in names:
        meta=catalog[name]
        single,ax=plt.subplots(figsize=(5.8,6.4))
        single.subplots_adjust(left=.16,right=.94,bottom=.22,top=.72)
        schematic.draw_panel(ax,dict(pattern=name,m=meta["m"],occupied_bits=meta["occupied_bits"],
                                     cell_counts=[meta["empty_cells"],0,meta["double_cells"]]))
        single.legend(handles=handles,loc="upper center",bbox_to_anchor=(.5,.98),ncol=2,frameon=False)
        single.text(.5,.04,"Class 2: empty / double cells only.\n"+state_label(meta),ha="center",fontsize=10)
        save(single,f"{name}_pattern",figures)


def paired_and_slopes(rows,slopes,names,catalog,results,betas,stage,figures):
    for name,beta in itertools.product(names,betas):
        meta,result=catalog[name],results[name]
        block=[r for r in rows if r["pattern"]==name and r["beta"]==beta]
        limit=sorted((r for r in block if math.isinf(r["theta"])),key=lambda r:r["L"])
        parameter=next(r for r in result["coefficients"] if r["beta"]==beta)
        fitted=next((r for r in result["eq"] if r["beta"]==beta and r["requested_Lmin"]==meta["equilibrium"]["Lmin"] and r["model"]=="log_inv1"),None)
        time_fit=next((r for r in result["pairs"] if r["beta"]==beta and r["L_large"]==180 and r["L_small"]==144 and r["window"]=="4-7" and r["model"]=="log_inv1_inv2"),None)
        fig,axes=plt.subplots(1,2,figsize=(12.5,4.8),constrained_layout=True)
        axes[0].plot(np.log([r["L"] for r in limit]),[r["D"] for r in limit],"o",mfc="white",color=core.BLUE,label=limit_label(meta))
        if fitted:
            x=np.geomspace(fitted["x_min"],fitted["x_max"],200)
            label=f"Log + 1/L fit, L{fitted['x_min']:g}..{fitted['x_max']:g}"
            if not fitted["extent_complete"]:label+=" (incomplete)"
            axes[0].plot(np.log(x),core.predict(fitted,x),color=core.BLUE,label=label)
        for index,size in enumerate((144,180)):
            pts=sorted((r for r in block if r["L"]==size and math.isfinite(r["theta"])),key=lambda r:r["theta"])
            axes[1].plot(np.log([r["theta"] for r in pts]),[r["D"] for r in pts],
                marker="os"[index],linestyle=("-","--")[index],mfc="white",ms=6-index,
                color=("#5B3B8C","#148F8B")[index],lw=.9,label=f"L={size}")
        if time_fit:
            x=np.geomspace(4,7,200)
            status=("local numerical checks pass" if parameter.get("local_diagnostic_pass") else "UNRESOLVED") if meta["ground_reachable"] else "BLOCKED, scaling UNRESOLVED"
            axes[1].plot(np.log(x),core.predict(time_fit,x),color=core.ORANGE,lw=2,label=f"Fit theta4..7: {status}")
        for ax in axes:ax.set_ylabel("Raw D");axes_style(ax)
        axes[0].set(xlabel=r"$\ln L$",title=limit_label(meta))
        axes[1].set(xlabel=r"$\ln\theta$",title="Finite time; theta >= 4, theta/L <= 0.05")
        fig.suptitle(f"{name}; beta={beta:g}; {stage}\n{state_label(meta)}; raw points unshifted, preset inverse-scale fit corrections",fontsize=11)
        save(fig,f"{name}_scaling_beta_{beta:g}".replace(".","p"),figures)

        fig,axes=plt.subplots(2,2,figsize=(12,7.5),constrained_layout=True)
        axes[0,0].plot(np.log([r["L"] for r in limit]),[r["D"] for r in limit],"o-",mfc="white",color=core.BLUE,lw=1)
        sec=[r for r in slopes if r["pattern"]==name and r["beta"]==beta and r["domain"]=="limit"]
        axes[1,0].plot(np.log([r["x_geometric_mean"] for r in sec]),[r["local_log_slope"] for r in sec],"o-",mfc="white",color=core.BLUE,lw=1)
        for index,size in enumerate((144,180)):
            pts=sorted((r for r in block if r["L"]==size and math.isfinite(r["theta"])),key=lambda r:r["theta"])
            sec=[r for r in slopes if r["pattern"]==name and r["beta"]==beta and r["domain"]=="time" and r["L"]==size]
            style=dict(marker="os"[index],linestyle=("-","--")[index],mfc="white",color=("#5B3B8C","#148F8B")[index],lw=1,label=f"L={size}")
            axes[0,1].plot(np.log([r["theta"] for r in pts]),[r["D"] for r in pts],**style)
            axes[1,1].plot(np.log([r["x_geometric_mean"] for r in sec]),[r["local_log_slope"] for r in sec],**style)
        axes[0,0].set(title=limit_label(meta),xlabel=r"$\ln L$",ylabel="Raw D")
        axes[0,1].set(title="Finite-time entropy",xlabel=r"$\ln\theta$",ylabel="Raw D")
        axes[1,0].set(xlabel=r"$\ln\sqrt{L_iL_{i+1}}$",ylabel=r"$\Delta D/\Delta\ln L$")
        axes[1,1].set(xlabel=r"$\ln\sqrt{\theta_i\theta_{i+1}}$",ylabel=r"$\Delta D/\Delta\ln\theta$")
        for ax in axes.flat:axes_style(ax)
        for ax in axes[1]:ax.axhline(0,color=".5",lw=.7)
        fig.suptitle(f"{name}; beta={beta:g}; {stage}\n{state_label(meta)}. Raw adjacent-point slopes; no model, no asymptotic certification.",fontsize=11)
        save(fig,f"{name}_raw_slopes_beta_{beta:g}".replace(".","p"),figures)


def parameter_axis(ax,coefficients,meta,limit_only=False):
    beta=core.array(coefficients,"beta")
    ax.plot(beta,core.array(coefficients,"c_eq"),color=core.BLUE,lw=2.3,zorder=2)
    ax.plot(beta,core.array(coefficients,"c_eq"),"o",mfc="white",mec=core.BLUE,ms=8,mew=1.4,zorder=4,clip_on=False)
    ax.fill_between(beta,core.array(coefficients,"eq_range_min"),core.array(coefficients,"eq_range_max"),color=core.BLUE,alpha=.12)
    label="Equilibrium finite-window c" if meta["ground_reachable"] else "Accessible limit: c_fit"
    handles=[Line2D([],[],color=core.BLUE,marker="o",mfc="white",ms=8,label=label)]
    bounds=[core.array(coefficients,key) for key in ("c_eq","eq_range_min","eq_range_max")]
    if not limit_only:
        ax.plot(beta,core.array(coefficients,"c_neq"),color=core.ORANGE,ls="--",lw=1.5,zorder=3)
        ax.fill_between(beta,core.array(coefficients,"neq_range_min"),core.array(coefficients,"neq_range_max"),color=core.ORANGE,alpha=.12)
        for passed in (False,True):
            group=[r for r in coefficients if bool(meta["ground_reachable"] and r.get("local_diagnostic_pass"))==passed]
            ax.plot(core.array(group,"beta"),core.array(group,"c_neq"),"s",mfc=core.ORANGE if passed else "white",mec=core.ORANGE,ms=4.5,mew=1.2,zorder=5,clip_on=False)
        handles.append(Line2D([],[],color=core.ORANGE,ls="--",marker="s",mfc="white",ms=5,
            label="Time: open = unresolved" if meta["ground_reachable"] else "Time: blocked / unresolved c_fit"))
        if meta["ground_reachable"]:
            handles.append(Line2D([],[],color=core.ORANGE,ls="--",marker="s",mfc=core.ORANGE,ms=5,label="Time: filled = local checks pass"))
        bounds += [core.array(coefficients,key) for key in ("c_neq","neq_range_min","neq_range_max")]
    values=np.concatenate(bounds);values=values[np.isfinite(values)]
    if len(values):
        low,high=min(0.,float(min(values))),max(0.,float(max(values)))
        pad=.07*max(high-low,.01);ax.set_ylim(low-pad,high+pad)
    ax.axhline(0,color=".5",lw=.7);ax.grid(axis="y",alpha=.2)
    ax.set(xlabel=r"$\beta$",ylabel="Finite-window c" if meta["ground_reachable"] else r"Descriptive $c_{\rm fit}$",xlim=(-.04,2.04))
    ax.set_xticks([0,.5,1,1.5,2]);ax.legend(handles=handles,fontsize=8)


def parameter_figures(names,catalog,results,stage,figures):
    overview,all_axes=plt.subplots(len(names),2,figsize=(13,3.8*len(names)))
    overview.subplots_adjust(left=.08,right=.98,bottom=.065,top=.91,hspace=.57,wspace=.25)
    for index,name in enumerate(names):
        meta=catalog[name];coefficients=results[name]["coefficients"]
        single,single_axes=plt.subplots(1,2,figsize=(12.8,5),constrained_layout=True)
        for pair in (all_axes[index],single_axes):
            parameter_axis(pair[0],coefficients,meta)
            parameter_axis(pair[1],coefficients,meta,limit_only=True)
            low,high=meta["equilibrium"]["Lmin"],meta["equilibrium"]["Lmax"]
            pair[0].set_title(f"{name}\nLimit L{low}..{high}; time L180 theta4..7",fontsize=10)
            pair[1].set_title(state_label(meta)+"\nLimit parameter and size/model range; independent y scale",fontsize=10)
        single.suptitle(f"{name}; {stage}\nBands: model/window ranges, not confidence intervals. Blocked states are not corner-coefficient estimates.",fontsize=11)
        save(single,f"{name}_coefficients_beta",figures)
    overview.suptitle(f"Additional second-class configurations: {stage}\nGround-state and accessible excited-state endpoints explicitly distinguished\nBands: model/window ranges, not confidence intervals; independent y scales",fontsize=13,y=.985)
    overview.text(.5,.02,"Local numerical stability never removes a GS-reachability obstruction. All blocked time markers remain hollow.\n"
                  "Prespecified fits do not certify log scaling; no fitting to force agreement between endpoints.",ha="center",fontsize=10)
    save(overview,"parameters_beta",figures)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan",type=Path,default=OUTPUT/"plan.json")
    parser.add_argument("--stage",choices=("interim","final"),default="interim")
    parser.add_argument("--no-plots",action="store_true")
    parser.add_argument("--check-plan",action="store_true")
    args=parser.parse_args()
    plan,plan_bytes,names,catalog=load_plan(args.plan)
    expected=expected_keys(plan,names)
    if args.check_plan:
        print(json.dumps(dict(status="PASS",plan_sha256=core.sha_bytes(plan_bytes),required_keys=len(expected),patterns=names),indent=2));return
    OUTPUT.mkdir(exist_ok=True);FIGURES.mkdir(exist_ok=True)
    paths=[Path(p) if Path(p).is_absolute() else HERE/p for p in plan["inputs"]]
    dependencies={Path(module.__file__).resolve():Path(module.__file__).read_bytes() for module in (old,core,previous,schematic)}
    dependencies[Path(__file__).resolve()]=Path(__file__).read_bytes()
    for relative in plan.get("validation_sources",[]):
        path=HERE/relative;content=path.read_bytes();dependencies[path]=content
        validation=json.loads(content.decode("utf-8-sig"))
        if validation.get("status")!="PASS":raise ValueError(f"Input validation must pass: {relative}")
        for source,recorded in validation["files"].items():
            file=HERE/source;data=file.read_bytes()
            if core.sha_bytes(data)!=recorded:raise ValueError(f"Validation source changed: {source}")
            dependencies[file]=data
    rows,sources,duplicates=input_sources(paths,catalog,expected,args.stage,plan,dependencies)
    present={(r["pattern"],r["L"],r["theta"],r["beta"]) for r in rows}
    missing=[dict(pattern=n,L=s,theta=t,beta=b) for n,s,t,b in sorted(expected-present)]
    tables=defaultdict(list,observations=rows,duplicate_observations=duplicates,missing_required_keys=missing)
    results={};betas=plan["betas"]
    for name in names:
        meta=catalog[name];block=[r for r in rows if r["pattern"]==name]
        result=old.run_pattern(block,betas,[tuple(w) for w in plan["windows"]],plan,meta["equilibrium"],plan["theta_min_screen"],plan["theta_over_L_max_screen"])
        results[name]=result
        for title,field,role in (("limit_fits","eq","limit"),("time_individual_fits","individual","time"),
                ("time_pair_fits","pairs","time"),("primary_parameters","coefficients","paired_fit_parameters"),
                ("fit_sensitivity","sensitivity","diagnostic"),("missing_fits","missing","diagnostic")):
            tables[title].extend(public_row(name,meta,r,role) for r in result[field])
    complete=all(r.get("c_limit_fit") is not None and r.get("c_time_fit") is not None for r in tables["primary_parameters"])
    if args.stage=="final" and (missing or not complete or any(s.get("missing") for s in sources)):
        raise ValueError("Final analysis requires the complete frozen grid and all primary fits")
    tables["local_log_slopes"]=local_slopes(rows,catalog)
    for title,values in tables.items():core.dump_csv(OUTPUT/f"{title}.csv",values)
    figures=[]
    if not args.no_plots:
        plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,"axes.spines.right":False,"savefig.dpi":180,"pdf.fonttype":42})
        catalogue(names,catalog,figures)
        paired_and_slopes(rows,tables["local_log_slopes"],names,catalog,results,betas,args.stage,figures)
        parameter_figures(names,catalog,results,args.stage,figures)
    if args.plan.read_bytes()!=plan_bytes:raise ValueError("Plan changed during analysis")
    changed=[s["path"] for s in sources if not s.get("missing") and sha(Path(s["path"]))!=s["sha256"]]
    if args.stage=="final" and changed:raise ValueError("Input changed during final analysis")
    if any(path.read_bytes()!=data for path,data in dependencies.items()):raise ValueError("Dependency changed during analysis")
    summary=dict(stage=args.stage,generated_utc=datetime.now(timezone.utc).isoformat(),plan_sha256=core.sha_bytes(plan_bytes),
        script_sha256=sha(Path(__file__)),sources=sources,
        input_column_count=next((s["input_column_count"] for s in sources if not s.get("missing")),None),
        rows=len(rows),required_keys=len(expected),missing_required_keys=len(missing),
        primary_complete=complete,patterns=plan["pattern_metadata"],table_counts={k:len(v) for k,v in tables.items()},figures=figures,
        by_pattern={name:dict(ground_reachable=catalog[name]["ground_reachable"],limit_state=catalog[name]["limit_state"],
             numeric_pass_beta=[r["beta"] for r in results[name]["coefficients"] if r.get("local_diagnostic_pass")],
             interpretation="descriptive fits only; GS blocked" if not catalog[name]["ground_reachable"] else "finite-window diagnostics, not an asymptotic proof") for name in names},
        source_changes_during_run=changed,
        conventions=dict(D="(2*S_A+S_C)/3-S_B; independent entropies, unshifted raw observations",
            csv_schema_note="Raw BP CSV has 41 columns; frozen plan phrase 39-column was a descriptive typo, no numerical grid or source changed",
            blocked_limit="Exact initial spectral-support accessible excited state; not equilibrium GS",
            parameter="Log model parameter; blocked values are never ground-state corner coefficients",
            uncertainty="Model/window ranges, not statistical confidence intervals",
            local_checks="Same numerical thresholds as previous analysis; cannot certify a physical scaling limit"))
    write_json(OUTPUT/"summary.json",summary)
    outputs=[OUTPUT/f"{name}.csv" for name in tables]+[OUTPUT/"summary.json"]
    outputs += [FIGURES/f"{stem}.{suffix}" for stem in figures for suffix in ("png","pdf")]
    bound={core.relative_path(path).replace('\\','/'):core.sha_bytes(data) for path,data in dependencies.items()}
    bound[core.relative_path(args.plan).replace('\\','/')]=core.sha_bytes(plan_bytes)
    bound.update({core.relative_path(path).replace('\\','/'):sha(path) for path in outputs})
    bound.update({core.relative_path(Path(s["path"])).replace('\\','/'):s["sha256"] for s in sources if not s.get("missing")})
    receipt=dict(status="COMPLETE" if args.stage=="final" and not missing and complete else "INTERIM",stage=args.stage,
        generated_utc=datetime.now(timezone.utc).isoformat(),files=bound,
        scope="Source/plan/output binding and coverage; independent physical and fit validation remains separate")
    write_json(OUTPUT/"receipt.json",receipt)
    print(json.dumps({k:summary[k] for k in ("stage","rows","required_keys","missing_required_keys","primary_complete","table_counts")},indent=2))


if __name__=="__main__":main()
