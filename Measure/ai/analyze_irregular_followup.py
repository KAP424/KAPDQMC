"""Independent follow-up of the irregular 4x4 pattern and a 3x3 control.

The original six-pattern analysis is imported read-only. Its 24--144 equilibrium
fits remain the primary baseline even after held-out L=180,216 become available.
No output from that analysis is changed. Local secants and out-of-sample residuals
are diagnostics; neither a small residual nor a fitted log term certifies scaling.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import itertools
import json
import math
from pathlib import Path

import numpy as np
import analyze_occupancy_classes as old
from analyze_occupancy_classes import core, previous
import plot_occupancy_classes_patterns as catalogue_plot
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "analysis_irregular_followup"
FIGURES = HERE / "figures_irregular_followup"
COLORS = ("#286B9C", "#D47728", "#5B3B8C")
REPRESENTATIVE_BETAS = (0., 1., 2.)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha(path):
    return core.sha_bytes(path.read_bytes())


def same_output(path, expected):
    # This script deliberately cannot target any of the previous output folders.
    if path.resolve() != expected.resolve():
        raise ValueError(f"Use the dedicated follow-up directory: {expected}")


def selected(rows, name, beta, size=None):
    return [r for r in rows if r["pattern"] == name and r["beta"] == beta
            and (size is None or r["L"] == size)]


def local_secants(rows):
    """Adjacent measured-point slopes, with no model or interpolation fitted."""
    groups, output = defaultdict(list), []
    for row in rows:
        domain = "equilibrium" if math.isinf(row["theta"]) else "time"
        groups[row["pattern"], row["beta"], domain, None if domain == "equilibrium" else row["L"]].append(row)
    for (name, beta, domain, size), group in groups.items():
        xkey = "L" if domain == "equilibrium" else "theta"
        points = sorted(group, key=lambda r: r[xkey])
        for left, right in zip(points, points[1:]):
            x1, x2 = left[xkey], right[xkey]
            if x2 <= x1:
                raise ValueError("Duplicate or unordered abscissae in secant diagnostic")
            output.append(dict(pattern=name, beta=beta, domain=domain, L=size,
                               x_left=x1, x_right=x2, x_geometric_mean=math.sqrt(x1*x2),
                               D_left=left["D"], D_right=right["D"],
                               local_log_slope=(right["D"]-left["D"])/math.log(x2/x1),
                               diagnostic="adjacent raw-data secant, not asymptotic c"))
    return output


def mode_inputs(names, catalog, plan_bytes):
    """Read the separately validated fixed-momentum diagnostic, never BZ minima."""
    csv_path, json_path = OUTPUT/"mode_diagnostics.csv", OUTPUT/"mode_diagnostics.json"
    dependencies = {p:p.read_bytes() for p in (csv_path,json_path)}
    metadata = json.loads(dependencies[json_path].decode("utf-8-sig"))
    if metadata.get("status") != "PASS" or metadata.get("csv_sha256") != core.sha_bytes(dependencies[csv_path]):
        raise ValueError("Fixed-mode diagnostic must be PASS and bind its current CSV")
    if metadata.get("plan_sha256") != core.sha_bytes(plan_bytes):
        raise ValueError("Fixed-mode diagnostic was not validated against the current plan")
    records = list(csv.DictReader(dependencies[csv_path].decode("utf-8-sig").splitlines()))
    if len(records) != metadata["rows"] or set(r["pattern"] for r in records) != set(names):
        raise ValueError("Fixed-mode diagnostic row count or pattern set mismatch")
    parameters, curves, distances = [], [], []
    for name in names:
        declared = metadata["patterns"][name]
        if declared["occupied_bits"] != catalog[name]["occupied_bits"] or declared["m"] != catalog[name]["m"]:
            raise ValueError("Fixed-mode diagnostic occupation mask mismatch")
        block = [r for r in records if r["pattern"] == name]
        s = float(block[0]["ground_overlap_sigma_min"])
        gap = float(block[0]["positive_negative_energy_gap"])
        if not 0 < s <= 1 or gap <= 0:
            raise ValueError("Mode formula requires nonzero ground overlap at a gapped momentum")
        for row in block:
            if abs(float(row["kx"])-math.pi)>1e-12 or abs(float(row["ky"]))>1e-12:
                raise ValueError("This figure is exclusively the fixed (pi,0) supercell momentum")
            if abs(float(row["ground_overlap_sigma_min"])-s)>1e-14 or abs(float(row["positive_negative_energy_gap"])-gap)>1e-12:
                raise ValueError("Fixed-mode parameters changed across theta/beta rows")
        selected_points = sorted((r for r in block if float(r["beta"]) == 0 and math.isfinite(float(r["theta"]))),key=lambda r:float(r["theta"]))
        for row in selected_points:
            distance=float(row["premeasurement_ground_projector_frobenius"])
            if distance <= 0 or not math.isfinite(distance):
                raise ValueError("Finite-time fixed-mode projector distances must be positive for the log axis")
            distances.append(dict(pattern=name,theta=float(row["theta"]),projector_distance=distance))
        for beta in np.linspace(0,2,101):
            p=s*s/(s*s+math.exp(-4*float(beta))*(1-s*s))
            curves.append(dict(pattern=name,kx=math.pi,ky=0.,beta=float(beta),s_fixed_mode=s,
                minimum_F_principal_occupation=p,source="exact formula for filtered ground-state subspace; no fit"))
        parameters.append(dict(pattern=name,kx=math.pi,ky=0.,s_fixed_mode=s,gap_fixed_mode=gap,
            projector_distance_theta4=next(r["projector_distance"] for r in distances if r["pattern"]==name and r["theta"]==4),
            projector_distance_theta7=next(r["projector_distance"] for r in distances if r["pattern"]==name and r["theta"]==7)))
    return parameters,curves,distances,metadata,dependencies


def extension_diagnostics(rows, names, catalog, results, betas):
    """Freeze all baseline fits; test new sizes before fitting any extension."""
    extrapolation, extended, missing = [], [], []
    for name in names:
        config = catalog[name]["equilibrium"]
        baseline_max = config["Lmax"]
        for beta in betas:
            points = sorted((r for r in selected(rows, name, beta) if math.isinf(r["theta"])), key=lambda r: r["L"])
            held = [r for r in points if r["L"] > baseline_max]
            if not held:
                continue
            for fitted in results[name]["eq"]:
                if fitted["beta"] != beta or not fitted["extent_complete"]:
                    continue
                for point in held:
                    prediction = float(core.predict(fitted, [point["L"]])[0])
                    error = point["D"]-prediction
                    extrapolation.append(dict(pattern=name, beta=beta, heldout_L=point["L"],
                        train_Lmin=fitted["requested_Lmin"], train_Lmax=baseline_max,
                        train_grid=fitted["grid"], model=fitted["model"], c_train=fitted["c"],
                        D_observed=point["D"], D_prediction=prediction, residual=error,
                        abs_residual=abs(error), train_rmse=fitted["rmse"],
                        abs_residual_over_train_rmse=core.ratio(abs(error), fitted["rmse"]),
                        heldout_not_used_in_fit=True,
                        interpretation="prediction error; ratio to train RMSE is not a significance test"))
            extra_cuts = [60, 72, 96] if catalog[name]["m"] == 4 else [72, 90]
            lower_cuts = sorted(set(config["lower_cuts"]+extra_cuts))
            for high, low, model in itertools.product([baseline_max]+[r["L"] for r in held],
                                                      lower_cuts, ("log_inv1", "log_inv1_inv2")):
                block = [r for r in points if low <= r["L"] <= high]
                fitted, reason = core.fit([r["L"] for r in block], [r["D"] for r in block], model)
                meta = dict(pattern=name, beta=beta, model=model, requested_Lmin=low,
                            requested_Lmax=high, fit_role="refit_after_heldout_test" if high > baseline_max else "baseline_sensitivity")
                if fitted:
                    extended.append(dict(**meta, **fitted,
                        extent_complete=block[0]["L"] == low and block[-1]["L"] == high))
                else:
                    missing.append(dict(**meta, reason=reason, n=len(block)))
    return extrapolation, extended, missing


def save(fig, directory, stem, figures):
    core.save(fig, directory, stem)
    figures.append(stem)


def axes_style(ax):
    ax.grid(axis="y", alpha=.22)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=8)


def pattern_figure(names, catalog, directory, figures):
    fig, axes = plt.subplots(1, len(names), figsize=(4.5*len(names), 5.6))
    fig.subplots_adjust(left=.04, right=.98, bottom=.20, top=.70, wspace=.20)
    for ax, name in zip(np.atleast_1d(axes), names):
        meta = catalog[name]
        catalogue_plot.draw_panel(ax, dict(pattern=name, m=meta["m"], occupied_bits=meta["occupied_bits"],
            cell_counts=[meta["empty_cells"], meta["single_cells"], meta["double_cells"]]))
    handles = [Line2D([], [], marker="o" if sub == 0 else "s", color="none", markersize=7,
                      markerfacecolor=catalogue_plot.COLORS[sub] if occupied else "white",
                      markeredgecolor=catalogue_plot.COLORS[sub], label=f"{'AB'[sub]} {'occupied' if occupied else 'empty'}")
               for sub, occupied in itertools.product((0, 1), (True, False))]
    fig.legend(handles=handles, ncol=4, loc="upper center", bbox_to_anchor=(.5,.93), frameon=False)
    fig.suptitle("Irregular-pattern follow-up: periodic occupation tiles", y=.995, fontsize=15)
    fig.text(.5, .035, "Cell-coordinate schematic, not real-space bond geometry. Filled = occupied; hollow = empty.\n"
             "An odd 3 x 3 half-filled tile cannot contain only empty and double cells.", ha="center", fontsize=10)
    save(fig, directory, "pattern_catalogue", figures)


def raw_and_secants(rows, slopes, names, directory, figures, stage):
    for beta in REPRESENTATIVE_BETAS:
        fig, axes = plt.subplots(len(names), 2, figsize=(12, 3.4*len(names)), constrained_layout=True)
        for index, name in enumerate(names):
            points = sorted((r for r in selected(rows, name, beta) if math.isinf(r["theta"])), key=lambda r: r["L"])
            secants = [r for r in slopes if r["pattern"] == name and r["beta"] == beta and r["domain"] == "equilibrium"]
            ax, slope_ax = axes[index]
            if points:
                ax.plot(np.log([r["L"] for r in points]), [r["D"] for r in points], "o-", color=COLORS[index%3], lw=1, mfc="white")
                held = [r for r in points if r["L"] > 144]
                if held:
                    ax.plot(np.log([r["L"] for r in held]), [r["D"] for r in held], "s", color="black", ms=6, mfc="none", label="New L > 144")
            if secants:
                slope_ax.plot(np.log([r["x_geometric_mean"] for r in secants]), [r["local_log_slope"] for r in secants], "o-", color=COLORS[index%3], lw=1, mfc="white")
                slope_ax.axhline(0, color=".5", lw=.7)
            ax.set(xlabel=r"$\ln L$", ylabel="Raw D", title=name)
            slope_ax.set(xlabel=r"$\ln\sqrt{L_i L_{i+1}}$", ylabel=r"$\Delta D/\Delta\ln L$", title="Adjacent-point slope; no fit")
            axes_style(ax); axes_style(slope_ax)
        fig.suptitle(f"Equilibrium data and local slopes: beta={beta:g}; {stage}\nIndependent y scales; a slope plateau is a diagnostic, not a proof of the asymptotic limit", fontsize=12)
        save(fig, directory, f"equilibrium_raw_and_slopes_beta_{beta:g}".replace(".","p"), figures)


def paired_plots(rows, names, catalog, results, betas, directory, figures, stage, theta_min, ratio_max):
    for name, beta in itertools.product(names, betas):
        result, config = results[name], catalog[name]["equilibrium"]
        block = selected(rows, name, beta)
        coefficient = next((r for r in result["coefficients"] if r["beta"] == beta), {})
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), constrained_layout=True)
        eq = sorted((r for r in block if math.isinf(r["theta"])), key=lambda r: r["L"])
        old_points, held = [r for r in eq if r["L"] <= config["Lmax"]], [r for r in eq if r["L"] > config["Lmax"]]
        axes[0].plot(np.log([r["L"] for r in old_points]), [r["D"] for r in old_points], "o", color=core.BLUE, mfc="white", label="Equilibrium data")
        if held:
            axes[0].plot(np.log([r["L"] for r in held]), [r["D"] for r in held], "s", color="black", mfc="none", label="Held-out larger L")
        fitted = next((r for r in result["eq"] if r["beta"] == beta and r["requested_Lmin"] == config["Lmin"] and r["model"] == "log_inv1"), None)
        if fitted:
            x = np.geomspace(fitted["x_min"], fitted["x_max"], 200)
            extent_label = "" if fitted["extent_complete"] else " (incomplete)"
            axes[0].plot(np.log(x), core.predict(fitted,x), color=core.BLUE,
                         label=f"Baseline fit {fitted['x_min']:g}..{fitted['x_max']:g}{extent_label}")
            if held:
                x = np.geomspace(config["Lmax"], max(r["L"] for r in held), 100)
                axes[0].plot(np.log(x), core.predict(fitted,x), "--", color=core.BLUE, label="Frozen-fit extrapolation")
        for index, size in enumerate((144, 180)):
            pts = sorted((r for r in block if r["L"] == size and math.isfinite(r["theta"])
                          and r["theta"] >= theta_min and r["theta"]/size <= ratio_max), key=lambda r:r["theta"])
            if pts:
                axes[1].plot(np.log([r["theta"] for r in pts]), [r["D"] for r in pts],
                             marker=("o","s")[index], linestyle=("-","--")[index],
                             color=("#5B3B8C","#148F8B")[index], mfc="white", ms=(7,5)[index], lw=.8, label=f"L={size}")
        primary = core.primary_for_beta(result["primary"], beta)
        fit_time = next((r for r in result["pairs"] if r["beta"] == beta and r["L_large"] == primary.get("L")
                        and r["L_small"] == primary.get("comparison_L") and r["window"] == coefficient.get("primary_window")
                        and r["model"] == primary.get("model")), None)
        if fit_time and coefficient.get("c_neq") is not None:
            x = np.geomspace(fit_time["theta_min"], fit_time["theta_max"], 200)
            status = "local checks pass" if coefficient.get("local_diagnostic_pass") else "UNRESOLVED"
            axes[1].plot(np.log(x), core.predict(fit_time,x), color=core.ORANGE, lw=2, label=f"Finite-window fit: {status}")
        else:
            axes[1].text(.5, .5, "Primary fit unavailable", transform=axes[1].transAxes, ha="center")
        axes[0].set(xlabel=r"$\ln L$", ylabel="D", title="Equilibrium: baseline and held-out sizes")
        axes[1].set(xlabel=r"$\ln\theta$", ylabel="D", title=f"Time: theta >= {theta_min:g}, theta/L <= {ratio_max:g}")
        for ax in axes: axes_style(ax)
        fig.suptitle(f"{name}; beta={beta:g}; {stage}\nRaw points unshifted; inverse-scale corrections retained in predetermined fits", fontsize=11)
        save(fig, directory, f"{name}_scaling_beta_{beta:g}".replace(".","p"), figures)


def coefficient_figures(names, catalog, results, directory, figures, stage):
    # Always give equilibrium its own column, even when time estimates diverge.
    fig, axes = plt.subplots(len(names), 2, figsize=(13, 4*len(names)))
    fig.subplots_adjust(left=.075,right=.98,bottom=.065,top=.90,hspace=.48,wspace=.27)
    for index, name in enumerate(names):
        coefficients = results[name]["coefficients"]
        eq_config = catalog[name]["equilibrium"]
        primary = core.primary_for_beta(results[name]["primary"], coefficients[0]["beta"])
        eq_scope = f"Eq L={eq_config['Lmin']}..{eq_config['Lmax']}"
        time_scope = f"time L={primary['L']}, theta={primary['window'][0]:g}..{primary['window'][1]:g}"
        scope = f"{eq_scope}; {time_scope}"
        previous.coefficient_axis(axes[index,0], coefficients, f"{name}\n{scope}", show_bands=True)
        axes[index,0].title.set_fontsize(10)
        old.equilibrium_zoom_axis(axes[index,1], coefficients, show_bands=True)
        axes[index,1].set_title(f"Equilibrium L={eq_config['Lmin']}..{eq_config['Lmax']} and size/model range\nIndependent y scale",fontsize=10)
        zoom = old.needs_equilibrium_zoom(coefficients)
        if zoom:
            axes[index,0].text(.02,.98,"Time: unresolved finite-window fits", transform=axes[index,0].transAxes, va="top", fontsize=8)
        if index<len(names)-1:
            axes[index,0].set_xlabel("");axes[index,1].set_xlabel("")
        if not any(r.get("c_eq") is not None for r in coefficients):
            for ax in axes[index]:
                ax.text(.5,.5,"Pending data",transform=ax.transAxes,ha="center",fontsize=12)
        single, single_axes = plt.subplots(1,2,figsize=(12,4.4),constrained_layout=True)
        previous.coefficient_axis(single_axes[0], coefficients, f"Both finite-window estimates\n{scope}", detailed=True)
        single_axes[0].title.set_fontsize(10)
        old.equilibrium_zoom_axis(single_axes[1], coefficients, show_bands=True)
        single_axes[1].set_title(f"Equilibrium L={eq_config['Lmin']}..{eq_config['Lmax']}\nSize/model range; independent y scale",fontsize=10)
        single.suptitle(f"{name}; {stage}\nBands: model/window ranges, not confidence intervals; filled/open time squares: local pass/unresolved", fontsize=11)
        save(single,directory,f"{name}_coefficients_beta",figures)
    fig.suptitle(f"Follow-up coefficient comparison: {stage}\nFinite-window estimates, not certified angle coefficients.\nBands: model/window ranges, not confidence intervals; filled/open time squares: local pass/unresolved.",fontsize=12,y=.985)
    save(fig,directory,"coefficients_beta",figures)


def mode_figure(names, parameters, curves, distances, directory, figures):
    fig,axes=plt.subplots(1,2,figsize=(12.5,5.6))
    fig.subplots_adjust(left=.08,right=.98,bottom=.22,top=.78,wspace=.25)
    axes[0].axvspan(4,7,color=".6",alpha=.13,label=r"Entropy-fit window: $4\leq\theta\leq7$")
    for index,name in enumerate(names):
        points=[r for r in distances if r["pattern"]==name]
        label=name
        axes[0].plot([r["theta"] for r in points],[r["projector_distance"] for r in points],
            marker=("o","s","^")[index%3],linestyle=("-","--",":")[index%3],
            color=COLORS[index%3],mfc="white",ms=5,label=label)
        values=[r for r in curves if r["pattern"]==name]
        s=next(r["s_fixed_mode"] for r in parameters if r["pattern"]==name)
        axes[1].plot([r["beta"] for r in values],[r["minimum_F_principal_occupation"] for r in values],
            linestyle=("-","--",":")[index%3],color=COLORS[index%3],lw=2,
            label=f"{name}; s={s:.5f}")
    axes[0].set(xlabel=r"Projection time $\theta$",ylabel=r"$\|P_\theta-P_{\rm GS}\|_F$",yscale="log",
                title="Unmeasured mode: distance to ground subspace")
    axes[1].set(xlabel=r"Measurement strength $\beta$",ylabel=r"Smallest principal occupation in $F$",
                xlim=(0,2),ylim=(0,1),title=r"Ground-state mode: $p_F=s^2/[s^2+e^{-4\beta}(1-s^2)]$")
    for ax in axes:axes_style(ax)
    fig.suptitle(r"Single-mode mechanism diagnostic: $k=(\pi,0)$ in supercell reciprocal coordinates"+"\nAll three modes are gapped; the two m=4 patterns share the same energy gap",fontsize=12,y=.97)
    fig.text(.5,.055,"Left: independent projector calculation, beta=0; points joined as a guide. Right: exact formula, no fit.\n"
        "F = initially occupied sites; s = minimum ground/initial overlap singular value at this fixed k.\n"
        "Neither panel is an entropy or a scaling coefficient; fixed-k s is not the minimum over the Brillouin zone.",ha="center",fontsize=9,linespacing=1.5)
    save(fig,directory,"mode_crossover",figures)


def extrapolation_figures(rows, extrapolation, extended, names, results, catalog, directory, figures):
    for name, beta in itertools.product(names, REPRESENTATIVE_BETAS):
        errors = [r for r in extrapolation if r["pattern"] == name and r["beta"] == beta]
        if not errors:
            continue
        config = catalog[name]["equilibrium"]
        eq = sorted((r for r in selected(rows,name,beta) if math.isinf(r["theta"])),key=lambda r:r["L"])
        fig, axes = plt.subplots(1,3,figsize=(16,4.7),constrained_layout=True)
        axes[0].plot(np.log([r["L"] for r in eq]),[r["D"] for r in eq],"o",color="black",mfc="white",label="Raw equilibrium data")
        for index, model in enumerate(("log_inv1","log_inv1_inv2")):
            fit = next((r for r in results[name]["eq"] if r["beta"] == beta and r["requested_Lmin"] == config["Lmin"] and r["model"] == model and r["extent_complete"]),None)
            if not fit: continue
            for low,high,style in ((config["Lmin"],144,"-"),(144,max(r["L"] for r in eq),"--")):
                x=np.geomspace(low,high,150)
                axes[0].plot(np.log(x),core.predict(fit,x),style,color=COLORS[index],label=f"{model}: {'fit' if style=='-' else 'extrapolate'}")
            errs=[r for r in errors if r["model"]==model and r["train_Lmin"]==config["Lmin"]]
            axes[1].plot([r["heldout_L"] for r in errs],[r["residual"] for r in errs],("o-","s--")[index],color=COLORS[index],mfc="white",label=f"{model}, Lmin={config['Lmin']}")
            for point in sorted({r["heldout_L"] for r in errs}):
                spread=[r["residual"] for r in errors if r["model"]==model and r["heldout_L"]==point]
                axes[1].vlines(point,min(spread),max(spread),color=COLORS[index],alpha=.35,lw=6)
        for index, high in enumerate(sorted({r["requested_Lmax"] for r in extended if r["pattern"]==name and r["beta"]==beta})):
            for model,marker,style in (("log_inv1","o","-"),("log_inv1_inv2","s","--")):
                fits=sorted((r for r in extended if r["pattern"]==name and r["beta"]==beta and r["requested_Lmax"]==high and r["model"]==model and r["extent_complete"]),key=lambda r:r["requested_Lmin"])
                axes[2].plot([r["requested_Lmin"] for r in fits],[r["c"] for r in fits],marker=marker,linestyle=style,color=COLORS[index%3],mfc="white",label=f"Lmax={high}, inv{'1' if model=='log_inv1' else '1+2'}")
        axes[0].set(xlabel=r"$\ln L$",ylabel="D",title="Frozen original fit: L <= 144")
        axes[1].set(xlabel="Held-out L",ylabel="Observed D - predicted D",title="Out-of-sample error; bars: Lmin range")
        axes[2].set(xlabel="Fit Lmin",ylabel="Finite-window c",title="Refit sensitivity after adding data")
        for ax in axes:
            axes_style(ax)
        axes[1].axhline(0,color=".5",lw=.7);axes[2].axhline(0,color=".5",lw=.7)
        fig.suptitle(f"{name}; beta={beta:g}: held-out extrapolation and subsequent refits\nNew larger-L points never enter the frozen baseline fit. Sensitivity ranges are not statistical errors.",fontsize=12)
        save(fig,directory,f"{name}_extrapolation_beta_{beta:g}".replace(".","p"),figures)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan",type=Path,default=OUTPUT/"plan.json")
    parser.add_argument("--inputs",type=Path,nargs="+")
    parser.add_argument("--output",type=Path,default=OUTPUT)
    parser.add_argument("--figures",type=Path,default=FIGURES)
    parser.add_argument("--stage",choices=("interim","final"),default="interim")
    parser.add_argument("--plot-betas",type=float,nargs="+",default=[0,.5,1,1.5,2])
    parser.add_argument("--no-plots",action="store_true")
    args=parser.parse_args()
    same_output(args.output,OUTPUT);same_output(args.figures,FIGURES)
    plan_bytes=args.plan.read_bytes();plan=json.loads(plan_bytes.decode("utf-8-sig"))
    names,catalog=old.catalog_from_plan(plan)
    betas=sorted({core.canonical(x) for x in plan["betas"]})
    windows=[tuple(map(float,w)) for w in plan.get("windows",old.DEFAULT_WINDOWS)]
    theta_min=float(plan.get("theta_min_screen",4));ratio_max=float(plan.get("theta_over_L_max_screen",.05))
    paths=args.inputs or [Path(p) if Path(p).is_absolute() else HERE/p for p in plan.get("inputs",[])]
    if not paths: raise ValueError("The plan must provide input CSV paths")
    dependencies={Path(module.__file__).resolve():Path(module.__file__).read_bytes() for module in (old,core,previous,catalogue_plot)}
    mode_parameters,mode_curves,mode_distances,mode_metadata,mode_dependencies=mode_inputs(names,catalog,plan_bytes)
    dependencies.update(mode_dependencies)
    args.output.mkdir(parents=True,exist_ok=True);args.figures.mkdir(parents=True,exist_ok=True)
    rows,sources,duplicates=old.read_sources(paths,"final" if args.stage=="final" else "preliminary",args.output,catalog,set(betas))
    tables=defaultdict(list);tables["observations"]=rows;tables["duplicate_observations"]=duplicates
    tables["mode_parameters"]=mode_parameters;tables["mode_crossover_formula"]=mode_curves
    results={};required=0
    for name in names:
        block=[r for r in rows if r["pattern"]==name]
        pplan=previous.per_pattern_plan(plan,name,names)
        result=old.run_pattern(block,betas,windows,pplan,catalog[name]["equilibrium"],theta_min,ratio_max)
        count,missing=core.coverage_check(block,pplan,betas);required+=count
        result.update(required_keys=count,missing_required_keys=len(missing))
        tables["missing_required_keys"].extend(old.tag(name,catalog,missing))
        for output,field in (("equilibrium_fits","eq"),("time_individual_fits","individual"),("time_pair_fits","pairs"),
                             ("primary_coefficients","coefficients"),("time_sensitivity","sensitivity"),("missing_fits","missing")):
            tables[output].extend(old.tag(name,catalog,result[field]))
        results[name]=result
    complete=all(r.get("c_eq") is not None and r.get("c_neq") is not None for r in tables["primary_coefficients"])
    if args.stage=="final" and (required==0 or tables["missing_required_keys"] or not complete or any(s.get("missing") for s in sources)):
        raise ValueError("Final requires all plan keys, all primary estimates, and every source file")
    tables["local_log_slopes"]=local_secants(rows)
    tables["heldout_extrapolation"],tables["extended_equilibrium_fits"],tables["missing_extended_fits"]=extension_diagnostics(rows,names,catalog,results,betas)
    for title,values in tables.items():core.dump_csv(args.output/f"{title}.csv",values)
    figures=[]
    if not args.no_plots:
        plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,"axes.spines.right":False,"savefig.dpi":180,"pdf.fonttype":42})
        pattern_figure(names,catalog,args.figures,figures)
        raw_and_secants(rows,tables["local_log_slopes"],names,args.figures,figures,args.stage)
        paired_plots(rows,names,catalog,results,args.plot_betas,args.figures,figures,args.stage,theta_min,ratio_max)
        coefficient_figures(names,catalog,results,args.figures,figures,args.stage)
        extrapolation_figures(rows,tables["heldout_extrapolation"],tables["extended_equilibrium_fits"],names,results,catalog,args.figures,figures)
        mode_figure(names,mode_parameters,mode_curves,mode_distances,args.figures,figures)
    changed=[s["path"] for s in sources if not s.get("missing") and sha(Path(s["path"]))!=s["sha256"]]
    dependency_changed=[str(p) for p,content in dependencies.items() if p.read_bytes()!=content]
    plan_changed=args.plan.read_bytes()!=plan_bytes
    if args.stage=="final" and (changed or dependency_changed or plan_changed):
        raise ValueError("Source, dependency or plan changed during final analysis")
    summary=dict(generated_utc=datetime.now(timezone.utc).isoformat(),stage=args.stage,
        script_sha256=sha(Path(__file__)),plan_sha256=core.sha_bytes(plan_bytes),plan=plan,
        dependencies=[dict(path=str(p),sha256=core.sha_bytes(c)) for p,c in dependencies.items()],sources=sources,
        source_changes_during_run=changed,dependency_changes_during_run=dependency_changed,plan_changed_during_run=plan_changed,
        patterns=catalog,betas=betas,rows=len(rows),duplicates=len(duplicates),required_keys=required,
        missing_required_keys=len(tables["missing_required_keys"]),primary_complete=complete,
        table_counts={k:len(v) for k,v in tables.items()},figures=figures,
        mode_diagnostic=dict(status=mode_metadata["status"],momentum="(pi,0), supercell reciprocal coordinates",
            csv_sha256=core.sha_bytes(mode_dependencies[OUTPUT/"mode_diagnostics.csv"]),
            json_sha256=core.sha_bytes(mode_dependencies[OUTPUT/"mode_diagnostics.json"]),
            scope="Single gapped mode, neither entropy nor scaling coefficient; fixed-k overlap, not full-BZ minimum"),
        by_pattern={name:dict(local_pass_beta=[r["beta"] for r in results[name]["coefficients"] if r.get("local_diagnostic_pass")],
            required_keys=results[name]["required_keys"],missing_required_keys=results[name]["missing_required_keys"]) for name in names},
        conventions=dict(primary="Original per-pattern Lmin..144 log+constant+1/L; time L180 theta4..7 log+constant+1/theta+1/theta^2",
            local_slopes="Adjacent raw-data secants at geometric-mean abscissae; no fitted coefficient implied",
            heldout="All measured L>144 held out from every original fit; refits exported separately",
            scope="Finite-window sensitivity, not an asymptotic theorem or statistical confidence interval",
            measurement="D=(2SA+SC)/3-SB, independent entropies, own initially empty mask, unshifted raw D"))
    write_json(args.output/"summary.json",summary)
    outputs=[args.output/f"{name}.csv" for name in tables]+[args.output/"summary.json"]
    outputs += [args.figures/f"{stem}.{suffix}" for stem in figures for suffix in ("png","pdf")]
    receipt=dict(generated_utc=datetime.now(timezone.utc).isoformat(),stage=args.stage,
        status="COMPLETE" if args.stage=="final" and complete and not tables["missing_required_keys"] else "INTERIM",
        script_sha256=sha(Path(__file__)),plan_sha256=core.sha_bytes(plan_bytes),
        dependencies=[dict(path=core.relative_path(p),sha256=core.sha_bytes(c)) for p,c in dependencies.items()],
        outputs=[dict(path=core.relative_path(p),sha256=sha(p)) for p in outputs],
        scope="Source/plan/output binding and coverage. Independent physical and fit validation is separate.")
    write_json(args.output/"receipt.json",receipt)
    print(json.dumps({k:summary[k] for k in ("stage","rows","required_keys","missing_required_keys","primary_complete","table_counts","by_pattern")},indent=2))


if __name__=="__main__":
    main()
