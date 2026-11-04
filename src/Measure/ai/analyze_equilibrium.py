"""Analyze raw equilibrium entropies, perimeter terms, and corner subtraction.

This is separate from analyze_results.py and never modifies observations.csv.
Run after the independent equilibrium scan finishes:
    python src/Measure/ai/analyze_equilibrium.py

Requires only numpy and matplotlib. All entropies are single-species Renyi-2
entropies with natural logarithms. Fit-window ranges are not confidence bounds.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
REGIONS = ("S_A", "S_C", "S_B")
COLORS = ["#245E93", "#D18A17", "#B75530", "#71832F", "#A74679", "#5D6E80"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X"]
RAW_MODELS = ("area", "area_inv", "area_log", "area_log_inv")
D_MODELS = ("log", "log_inv", "area_log", "area_log_inv")
CANONICAL = {"S_A": "area_inv", "S_C": "area_inv", "S_B": "area_log_inv"}
MODEL_LABELS = {
    "area": r"$aL+b$", "area_inv": r"$aL+b+d/L$",
    "area_log": r"$aL+q\ln L+b$", "area_log_inv": r"$aL+q\ln L+b+d/L$",
    "log": r"$c\ln L+b$", "log_inv": r"$c\ln L+b+d/L$",
}


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11,
        "axes.labelsize": 11, "legend.fontsize": 8, "xtick.labelsize": 9,
        "ytick.labelsize": 9, "axes.spines.top": False, "axes.spines.right": False,
        "axes.axisbelow": True, "grid.color": "#DEE3E8", "grid.linewidth": .7,
        "axes.edgecolor": "#4D5963", "text.color": "#25313B",
        "lines.linewidth": 1.4, "lines.markersize": 4, "savefig.dpi": 180,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def grouped(rows, keys):
    result = defaultdict(list)
    for row in rows:
        result[tuple(row[k] for k in keys)].append(row)
    return result


def boolean(value):
    if value.lower() in ("true", "1"):
        return True
    if value.lower() in ("false", "0"):
        return False
    raise ValueError(f"Unrecognized Boolean {value!r}")


def read_sources(paths, ground_tol):
    rows, provenance, seen = [], [], {}
    discarded_finite = duplicates = 0
    text_fields = {"pattern", "family", "source", "source_file", "source_sha256",
                   "limit_fingerprint", "scan_stage", "grid_version"}
    required = {"pattern", "family", "m", "L", "theta", "beta", *REGIONS}
    for path in paths:
        source_bytes = path.read_bytes()
        reader = csv.DictReader(io.StringIO(source_bytes.decode("utf-8-sig"), newline=""))
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing {sorted(missing)}")
        count = 0
        for line, raw in enumerate(reader, 2):
            if not any(raw.values()):
                continue
            count += 1
            if not math.isinf(float(raw["theta"])):
                discarded_finite += 1
                continue
            row = {}
            for key, value in raw.items():
                if value is None or value == "":
                    continue
                if key in text_fields:
                    row[key] = value
                elif key == "premeasurement_ground":
                    row[key] = boolean(value)
                else:
                    try:
                        row[key] = float(value)
                    except ValueError:
                        row[key] = value
            for key in ("L", "m"):
                row[key] = int(row[key])
            if row["L"] % (3 * row["m"]):
                raise ValueError(f"{path}:{line}: region cuts are not aligned to the supercell")
            for key in (*REGIONS, "beta"):
                if not math.isfinite(row[key]):
                    raise ValueError(f"{path}:{line}: nonfinite {key}")
            if min(row[k] for k in REGIONS) < -1e-9 or row["beta"] < 0:
                raise ValueError(f"{path}:{line}: invalid entropy or beta")
            excess = row.get("premeasurement_excess", row.get("accessible_excess"))
            if excess is None or not math.isfinite(excess):
                raise ValueError("Premeasurement energy excess is required; measured energy is insufficient")
            ground = abs(excess) < ground_tol
            if "premeasurement_ground" in row and row["premeasurement_ground"] != ground:
                raise ValueError(f"{path}:{line}: ground-state label disagrees with unmeasured energy")
            row["state_class"] = "ground" if ground else "obstructed"
            row["premeasurement_excess"] = excess
            row["delta"] = row["S_A"] - row["S_B"]
            row["D"] = (2 * row["S_A"] + row["S_C"]) / 3 - row["S_B"]
            key = (row["pattern"], row["L"], row["beta"])
            if key in seen:
                previous = seen[key]
                if previous["state_class"] != row["state_class"] or any(
                    abs(previous[k] - row[k]) > 1e-10 for k in REGIONS
                ):
                    raise ValueError(f"Conflicting duplicates at {key}")
                duplicates += 1
                continue
            seen[key] = row
            rows.append(row)
        provenance.append(dict(path=str(path.resolve()), sha256=hashlib.sha256(source_bytes).hexdigest(),
                               rows_read=count))
    if not rows:
        raise ValueError("No equilibrium observations were read")
    return rows, provenance, dict(identical_duplicates_removed=duplicates, finite_rows_ignored=discarded_finite)


def basis(x, model):
    x = np.asarray(x, float)
    terms = {
        "area": ("a", "b"), "area_inv": ("a", "b", "d"),
        "area_log": ("a", "q", "b"), "area_log_inv": ("a", "q", "b", "d"),
        "log": ("q", "b"), "log_inv": ("q", "b", "d"),
    }[model]
    columns = {"a": x, "b": np.ones_like(x), "q": np.log(x), "d": 1 / x}
    return np.column_stack([columns[k] for k in terms]), terms


def least_squares(design, y, names):
    y = np.asarray(y, float)
    if len(y) < len(names) + 2:
        return None
    scale = np.linalg.norm(design, axis=0)
    if np.any(scale == 0):
        return None
    xs = design / scale
    coef, _, rank, _ = np.linalg.lstsq(xs, y, rcond=None)
    if rank < len(names):
        return None
    coef /= scale
    residual = y - design @ coef
    q = np.linalg.qr(xs, mode="reduced")[0]
    leverage = np.sum(q**2, axis=1)
    loo = float(np.sqrt(np.mean((residual / (1-leverage))**2))) if np.max(leverage) < 1 - 1e-10 else None
    return {
        **dict(zip(names, map(float, coef))), "n_observations": len(y), "n_parameters": len(names),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "max_abs_residual": float(np.max(np.abs(residual))),
        "loo_observation_rmse": loo, "max_leverage": float(np.max(leverage)),
        "condition_scaled_design": float(np.linalg.cond(xs)),
    }


def fit_one(x, y, model):
    x = np.asarray(x, float)
    design, names = basis(x, model)
    fitted = least_squares(design, y, names)
    if fitted is None:
        return None
    return dict(model=model, Lmin=float(min(x)), Lmax=float(max(x)), n_sizes=len(x), **fitted)


JOINT_NAMES = ("a_A", "a_C", "b_A", "b_C", "b_B", "d_A", "d_C", "d_B", "c")


def joint_design(x):
    """Rows ordered as all A, all C, all B; coefficients in JOINT_NAMES order."""
    x = np.asarray(x, float)
    n = len(x)
    design = np.zeros((3*n, 9))
    design[:n, 0] = x
    design[:n, 2] = 1
    design[:n, 5] = 1/x
    design[n:2*n, 1] = x
    design[n:2*n, 3] = 1
    design[n:2*n, 6] = 1/x
    design[2*n:, 0] = 2*x/3
    design[2*n:, 1] = x/3
    design[2*n:, 4] = 1
    design[2*n:, 7] = 1/x
    design[2*n:, 8] = -np.log(x)
    return design


def fit_joint(block):
    if len(block) < 5:
        return None
    x = np.array([r["L"] for r in block])
    y = np.concatenate([[r[k] for r in block] for k in REGIONS])
    design = joint_design(x)
    fitted = least_squares(design, y, JOINT_NAMES)
    if fitted is None:
        return None
    coef = np.array([fitted[k] for k in JOINT_NAMES])
    residual = (y-design@coef).reshape(3, len(x))
    for i, region in enumerate(REGIONS):
        fitted["rmse_"+region] = float(np.sqrt(np.mean(residual[i]**2)))
    # Leave all three entropies at the same L out together, not one correlated row.
    loo_errors = []
    for held_out in range(len(x)):
        keep = np.tile(np.arange(len(x)) != held_out, 3)
        other = least_squares(design[keep], y[keep], JOINT_NAMES)
        if other is None:
            break
        predicted = design[~keep] @ np.array([other[k] for k in JOINT_NAMES])
        loo_errors.extend(y[~keep]-predicted)
    fitted["loo_size_rmse"] = float(np.sqrt(np.mean(np.asarray(loo_errors)**2))) if len(loo_errors)==len(y) else None
    return dict(model="geometry_joint", Lmin=float(min(x)), Lmax=float(max(x)), n_sizes=len(x), **fitted)


def fit_all(rows, lmins):
    region_fits, difference_fits, joint_fits = [], [], []
    for (pattern, state_class, beta), block in sorted(grouped(rows, ("pattern", "state_class", "beta")).items()):
        used = set()
        for lmin in lmins:
            subset = sorted([r for r in block if r["L"] >= lmin], key=lambda r:r["L"])
            window = tuple(r["L"] for r in subset)
            if window in used or len(subset) < 4:
                continue
            used.add(window)
            metadata = dict(pattern=pattern, family=subset[0]["family"], m=subset[0]["m"],
                            state_class=state_class, beta=beta, requested_Lmin=lmin)
            x = [r["L"] for r in subset]
            for region in REGIONS:
                for model in RAW_MODELS:
                    result = fit_one(x, [r[region] for r in subset], model)
                    if result:
                        region_fits.append(dict(**metadata, region=region, corner_c=-result.get("q", 0), **result))
            for observable in ("delta", "D"):
                for model in D_MODELS:
                    result = fit_one(x, [r[observable] for r in subset], model)
                    if result:
                        difference_fits.append(dict(**metadata, observable=observable, c=result["q"], **result))
            joint = fit_joint(subset)
            if joint:
                joint_fits.append(dict(**metadata, **joint))
    return region_fits, difference_fits, joint_fits


def pick(candidates, reference_lmin, **criteria):
    matches = [r for r in candidates if all(r.get(k)==v for k,v in criteria.items()) and
               r["Lmin"] >= reference_lmin]
    return min(matches, key=lambda r:(r["Lmin"], -r["n_sizes"])) if matches else None


def add_fit_fields(row, prefix, result, keys):
    if result:
        row.update({prefix+key: result.get(key) for key in keys})


def summaries(rows, rf, df, jf, reference_lmin):
    result = []
    for (pattern, state_class, beta), block in sorted(grouped(rows, ("pattern", "state_class", "beta")).items()):
        criteria = dict(pattern=pattern, state_class=state_class, beta=beta)
        row = dict(**criteria, family=block[0]["family"], m=block[0]["m"],
                   observed_sizes=len(block), observed_Lmin=min(r["L"] for r in block),
                   observed_Lmax=max(r["L"] for r in block))
        selected = {}
        for region in REGIONS:
            f = pick(rf, reference_lmin, **criteria, region=region, model=CANONICAL[region])
            selected[region] = f
            add_fit_fields(row, region+"_", f, ("a","q","b","d","corner_c","Lmin","Lmax","n_sizes",
                                                "rmse","loo_observation_rmse","condition_scaled_design"))
            free = pick(rf, reference_lmin, **criteria, region=region, model="area_log_inv")
            add_fit_fields(row, region+"_free_log_", free, ("q","rmse","loo_observation_rmse"))
            windows = [r["a"] for r in rf if all(r[k]==v for k,v in criteria.items()) and
                       r["region"]==region and r["model"]==CANONICAL[region]]
            if windows:
                row[region+"_a_window_min"] = min(windows)
                row[region+"_a_window_max"] = max(windows)
            corners = [r["corner_c"] for r in rf if all(r[k]==v for k,v in criteria.items()) and
                       r["region"]==region and r["model"]==CANONICAL[region]]
            if corners:
                row[region+"_corner_window_min"] = min(corners)
                row[region+"_corner_window_max"] = max(corners)
        for model, prefix in (("log", "D_log_"),("log_inv", "D_corrected_"),("area_log_inv", "D_free_area_")):
            f = pick(df, reference_lmin, **criteria, observable="D", model=model)
            add_fit_fields(row, prefix, f, ("c","a","rmse","loo_observation_rmse","Lmin","Lmax","n_sizes",
                                           "condition_scaled_design"))
            windows = [r["c"] for r in df if all(r[k]==v for k,v in criteria.items()) and
                       r["observable"]=="D" and r["model"]==model]
            if windows:
                row[prefix+"window_min"] = min(windows)
                row[prefix+"window_max"] = max(windows)
                row[prefix+"window_count"] = len(windows)
        joint = pick(jf, reference_lmin, **criteria)
        add_fit_fields(row, "joint_", joint, (*JOINT_NAMES,"rmse","loo_size_rmse","Lmin","Lmax","n_sizes",
                                             "rmse_S_A","rmse_S_C","rmse_S_B","condition_scaled_design"))
        joint_c = [r["c"] for r in jf if all(r[k]==v for k,v in criteria.items())]
        if joint_c:
            row["joint_c_window_min"] = min(joint_c)
            row["joint_c_window_max"] = max(joint_c)
        if all(selected.values()):
            a, c, b = (selected[k]["a"] for k in REGIONS)
            row["area_B_from_strips"] = (2*a+c)/3
            row["independent_area_mismatch"] = b-(2*a+c)/3
            row["independent_combined_rmse"] = float(np.sqrt(np.mean([selected[k]["rmse"]**2 for k in REGIONS])))
            loo = [selected[k]["loo_observation_rmse"] for k in REGIONS]
            if all(value is not None for value in loo):
                row["independent_combined_loo_rmse"] = float(np.sqrt(np.mean(np.square(loo))))
                if joint and joint["loo_size_rmse"] is not None and row["independent_combined_loo_rmse"]>1e-14:
                    row["joint_to_independent_loo_ratio"] = joint["loo_size_rmse"]/row["independent_combined_loo_rmse"]
        result.append(row)
    return result


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w",encoding="utf-8",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save(fig, directory, stem):
    for ext in ("png","pdf"):
        fig.savefig(directory/f"{stem}.{ext}",bbox_inches="tight",facecolor="white")
    plt.close(fig)


def evaluate(result, x):
    design,names=basis(x,result["model"])
    return design@np.array([result[k] for k in names])


def raw_figures(rows, fits, directory, reference_lmin, selected_betas):
    for (pattern,state_class),block in sorted(grouped(rows,("pattern","state_class")).items()):
        fig,axes=plt.subplots(2,3,figsize=(13.5,7.2),constrained_layout=True,
                              gridspec_kw={"height_ratios":[2,1]})
        betas=[b for b in selected_betas if any(r["beta"]==b for r in block)]
        for i,beta in enumerate(betas):
            subset=sorted([r for r in block if r["beta"]==beta],key=lambda r:r["L"])
            x=np.array([r["L"] for r in subset])
            color,marker=COLORS[i%len(COLORS)],MARKERS[i%len(MARKERS)]
            for j,region in enumerate(REGIONS):
                y=np.array([r[region] for r in subset])
                axes[0,j].plot(np.log(x),y,linestyle="none",marker=marker,color=color,label=rf"$\beta={beta:g}$")
                fitted=pick(fits,reference_lmin,pattern=pattern,state_class=state_class,beta=beta,
                            region=region,model=CANONICAL[region])
                if fitted:
                    xx=np.linspace(fitted["Lmin"],fitted["Lmax"],200)
                    axes[0,j].plot(np.log(xx),evaluate(fitted,xx),color=color)
                    keep=x>=fitted["Lmin"]
                    axes[1,j].plot(x[keep],y[keep]-evaluate(fitted,x[keep]),marker=marker,color=color)
        for j,region in enumerate(REGIONS):
            axes[0,j].set_ylabel(rf"$S_2({region[-1]})$")
            axes[0,j].set_title(MODEL_LABELS[CANONICAL[region]])
            axes[0,j].set_ylim(bottom=0)
            axes[1,j].axhline(0,color="#596773",linewidth=.8)
            axes[1,j].set_ylabel("Data - fitted entropy")
            axes[1,j].ticklabel_format(axis="y",style="sci",scilimits=(-2,2))
            axes[0,j].set_xscale("linear")
            axes[0,j].set_xlabel(r"$\ln L$")
            axes[1,j].set_xlabel("System size L")
            for ax in axes[:,j]:
                ax.grid(axis="y")
        axes[0,0].legend(ncol=2)
        fig.suptitle(f"{pattern} / {state_class}: raw entropies and independent fits\n"
                     f"Markers: data; lines: fits restricted to L >= {reference_lmin:g}; one species")
        save(fig,directory,f"raw_entropy_{pattern}_{state_class}")


def coefficient_figures(summary,directory):
    for (pattern,state_class),block in sorted(grouped(summary,("pattern","state_class")).items()):
        block=sorted(block,key=lambda r:r["beta"])
        fig,axes=plt.subplots(1,3,figsize=(14,4.5),constrained_layout=True)
        for i,region in enumerate(REGIONS):
            points=[r for r in block if region+"_a" in r]
            if not points: continue
            x=[r["beta"] for r in points]
            y=[r[region+"_a"] for r in points]
            axes[0].plot(x,y,marker=MARKERS[i],color=COLORS[i],label=f"a_{region[-1]}")
            axes[0].vlines(x,[r[region+"_a_window_min"] for r in points],
                           [r[region+"_a_window_max"] for r in points],color=COLORS[i],alpha=.5)
        for prefix,label,color,marker in (("D_corrected_","D: log + 1/L",COLORS[0],"o"),
                                          ("S_B_","raw B: -log coefficient",COLORS[1],"s"),
                                          ("joint_","geometry joint fit",COLORS[3],"D")):
            key=prefix+("corner_c" if prefix=="S_B_" else "c")
            points=[r for r in block if key in r]
            if points:
                axes[1].plot([r["beta"] for r in points],[r[key] for r in points],
                             marker=marker,color=color,label=label)
                if prefix=="D_corrected_":
                    axes[1].vlines([r["beta"] for r in points],[r[prefix+"window_min"] for r in points],
                                   [r[prefix+"window_max"] for r in points],color=color,alpha=.5)
                elif prefix=="S_B_":
                    axes[1].vlines([r["beta"] for r in points],[r["S_B_corner_window_min"] for r in points],
                                   [r["S_B_corner_window_max"] for r in points],color=color,alpha=.5)
                else:
                    axes[1].vlines([r["beta"] for r in points],[r["joint_c_window_min"] for r in points],
                                   [r["joint_c_window_max"] for r in points],color=color,alpha=.5)
        points=[r for r in block if "independent_area_mismatch" in r]
        if points:
            axes[2].plot([r["beta"] for r in points],[r["independent_area_mismatch"] for r in points],
                         marker="o",color=COLORS[2])
        axes[2].axhline(0,color="#596773",linewidth=.8)
        axes[0].set_ylabel("Area coefficient a in S = aL + ...")
        axes[1].set_ylabel("Finite-size effective log coefficient c")
        axes[2].set_ylabel(r"$a_B-(2a_A+a_C)/3$")
        axes[0].legend(); axes[1].legend()
        for ax in axes:
            ax.set_xlabel(r"Measurement strength $\beta$")
            ax.grid(axis="y")
        qualification = ("Ground-state input; model agreement must be checked" if state_class == "ground"
                         else "Obstructed input; fitted c is not a ground-state corner estimate")
        fig.suptitle(f"{pattern} / {state_class}: area and logarithmic terms\n"
                     f"{qualification}; whiskers: window ranges, not confidence intervals")
        save(fig,directory,f"coefficients_{pattern}_{state_class}")


def difference_figures(rows,df,directory,reference_lmin,selected_betas):
    for (pattern,state_class),block in sorted(grouped(rows,("pattern","state_class")).items()):
        fig,axes=plt.subplots(1,3,figsize=(13.5,4.3),constrained_layout=True)
        betas=[b for b in selected_betas if any(r["beta"]==b for r in block)]
        for i,beta in enumerate(betas):
            subset=sorted([r for r in block if r["beta"]==beta],key=lambda r:r["L"])
            x=np.array([r["L"] for r in subset])
            y=np.array([r["D"] for r in subset])
            color,marker=COLORS[i%len(COLORS)],MARKERS[i%len(MARKERS)]
            axes[0].plot(np.log(x),y,linestyle="none",marker=marker,color=color,label=rf"$\beta={beta:g}$")
            f=pick(df,reference_lmin,pattern=pattern,state_class=state_class,beta=beta,observable="D",model="log_inv")
            if f:
                xx=np.geomspace(f["Lmin"],f["Lmax"],200)
                axes[0].plot(np.log(xx),evaluate(f,xx),color=color)
                keep=x>=f["Lmin"]
                axes[1].plot(x[keep],y[keep]-evaluate(f,x[keep]),marker=marker,color=color)
            windows=sorted([r for r in df if r["pattern"]==pattern and r["state_class"]==state_class and
                            r["beta"]==beta and r["observable"]=="D" and r["model"]=="log_inv"],
                           key=lambda r:r["Lmin"])
            if windows:
                axes[2].plot([r["Lmin"] for r in windows],[r["c"] for r in windows],marker=marker,color=color)
        sizes=np.array(sorted({r["L"] for r in block}))
        ticks=sizes[np.unique(np.linspace(0,len(sizes)-1,min(6,len(sizes))).astype(int))]
        axes[0].set_xscale("linear"); axes[0].set_xticks(np.log(ticks),[f"{v:.2f}" for v in np.log(ticks)])
        axes[0].set_xlabel(r"$\ln L$"); axes[0].set_ylabel(r"$D=(2S_A+S_C)/3-S_B$")
        axes[1].set_xlabel("System size L"); axes[1].set_ylabel("D data - corrected log fit")
        axes[1].axhline(0,color="#596773",linewidth=.8)
        axes[1].ticklabel_format(axis="y",style="sci",scilimits=(-2,2))
        axes[2].set_xlabel("Minimum fitted size Lmin"); axes[2].set_ylabel("Effective fitted c in D = c ln L + b + d/L")
        axes[0].legend(ncol=2)
        for ax in axes: ax.grid(axis="y")
        fig.suptitle(f"{pattern} / {state_class}: D fit, residuals, and window dependence")
        save(fig,directory,f"D_scaling_{pattern}_{state_class}")


def cross_pattern_figures(rows,summary,directory,comparison_size,selected_betas):
    for state_class in ("ground","obstructed"):
        candidates=sorted({r["pattern"] for r in rows if r["state_class"]==state_class})
        for beta in selected_betas:
            data={r["pattern"]:r for r in rows if r["state_class"]==state_class and
                  r["L"]==comparison_size and r["beta"]==beta}
            patterns=[p for p in candidates if p in data]
            if not patterns: continue
            fitted={r["pattern"]:r for r in summary if r["state_class"]==state_class and r["beta"]==beta}
            fig,axes=plt.subplots(1,3,figsize=(14,max(4,.45*len(patterns)+1.6)),
                                  sharey=True,constrained_layout=True)
            pos=np.arange(len(patterns))
            for i,region in enumerate(REGIONS):
                offset=(i-1)*.18
                axes[0].plot([data[p][region] for p in patterns],pos+offset,linestyle="none",
                             marker=MARKERS[i],color=COLORS[i],label=region)
                points=[(j,fitted[p]) for j,p in enumerate(patterns) if p in fitted and region+"_a" in fitted[p]]
                axes[1].plot([r[region+"_a"] for _,r in points],[j+offset for j,_ in points],
                             linestyle="none",marker=MARKERS[i],color=COLORS[i],label="a_"+region[-1])
            points=[(j,fitted[p]) for j,p in enumerate(patterns) if p in fitted and "D_corrected_c" in fitted[p]]
            axes[2].plot([r["D_corrected_c"] for _,r in points],[j for j,_ in points],linestyle="none",
                         marker="o",color=COLORS[0],label="D, log + 1/L")
            for j,r in points:
                axes[2].hlines(j,r["D_corrected_window_min"],r["D_corrected_window_max"],color=COLORS[0],alpha=.6)
            axes[0].set_yticks(pos,patterns)
            axes[0].invert_yaxis()
            axes[0].set_xlabel(f"Raw entropy at L={comparison_size}")
            axes[1].set_xlabel("Independent area coefficient a")
            axes[2].set_xlabel("Finite-size effective D coefficient c")
            for ax in axes:
                ax.grid(axis="x"); ax.legend(loc="best")
            fig.suptitle(f"{state_class} states only: cross-pattern comparison at beta={beta:g}\n"
                         f"{len(patterns)} patterns measured at this beta; no interpolation; logarithmic term shown separately")
            stem=f"cross_patterns_{state_class}_beta_{beta:g}".replace(".","p")
            save(fig,directory,stem)


def matched_pattern_figure(path,directory):
    if not path.exists():
        return None
    content=path.read_bytes()
    patterns=list(csv.DictReader(io.StringIO(content.decode("utf-8-sig"))))
    if not patterns:
        return None
    fig,axes=plt.subplots(1,len(patterns),figsize=(3.4*len(patterns),3.6),
                         squeeze=False,constrained_layout=True)
    for ax,pattern in zip(axes[0],patterns):
        period=int(pattern["m"])
        bits=[int(v) for v in pattern["occupied_bits"]]
        for y in range(period):
            for x in range(period):
                for sub in (0,1):
                    occupied=bits[sub+2*(x+period*y)]
                    ax.scatter(x+(sub-.5)*.28,y,marker="o" if sub==0 else "s",s=68,
                               facecolor=COLORS[0] if occupied else "white",edgecolor=COLORS[0],linewidth=1.2)
        ax.set_xlim(-.55,period-.45); ax.set_ylim(period-.45,-.55)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(f"{pattern['pattern']}\nA-B={pattern['sublattice_imbalance']}, "
                     f"double/empty cells={pattern['double_cells']}/{pattern['empty_cells']}",fontsize=10)
    fig.suptitle("Arrangement controls: matched counts within each pair\n"
                 "A: circle; B: square; filled/open: occupied/empty; cell indices, not physical geometry")
    save(fig,directory,"matched_pattern_catalogue")
    return dict(path=str(path.resolve()),sha256=hashlib.sha256(content).hexdigest(),rows=len(patterns))


def parse_args():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input",nargs="+",type=Path,
                        default=[HERE/"data/equilibrium_extended.csv",HERE/"data/equilibrium_arrangements.csv"])
    parser.add_argument("--output",type=Path,default=HERE/"analysis_equilibrium")
    parser.add_argument("--figures",type=Path,default=HERE/"figures_equilibrium")
    parser.add_argument("--lmins",nargs="+",type=float,default=[0,12,18,27,36,45])
    parser.add_argument("--reference-lmin",type=float,default=18.)
    parser.add_argument("--ground-tol",type=float,default=1e-8)
    parser.add_argument("--plot-betas",nargs="+",type=float,default=[0,.5,1,2,4])
    parser.add_argument("--comparison-L",type=int,default=90)
    parser.add_argument("--no-plots",action="store_true")
    return parser.parse_args()


def main():
    args=parse_args()
    rows,sources,checks=read_sources(args.input,args.ground_tol)
    args.output.mkdir(parents=True,exist_ok=True)
    args.figures.mkdir(parents=True,exist_ok=True)
    lmins=sorted(set(args.lmins+[args.reference_lmin]))
    rf,df,jf=fit_all(rows,lmins)
    summary=summaries(rows,rf,df,jf,args.reference_lmin)
    write_csv(args.output/"raw_region_fits.csv",rf)
    write_csv(args.output/"difference_fits.csv",df)
    write_csv(args.output/"geometry_joint_fits.csv",jf)
    write_csv(args.output/"equilibrium_summary.csv",summary)
    coverage=[]
    for (pattern,state_class),block in sorted(grouped(rows,("pattern","state_class")).items()):
        coverage.append(dict(pattern=pattern,state_class=state_class,family=block[0]["family"],m=block[0]["m"],
                             rows=len(block),sizes=sorted({r["L"] for r in block}),
                             betas=sorted({r["beta"] for r in block}),
                             max_abs_premeasurement_excess=max(abs(r["premeasurement_excess"]) for r in block)))
    write_csv(args.output/"coverage.csv",coverage)
    if not args.no_plots:
        style()
        raw_figures(rows,rf,args.figures,args.reference_lmin,args.plot_betas)
        coefficient_figures(summary,args.figures)
        difference_figures(rows,df,args.figures,args.reference_lmin,args.plot_betas)
        cross_pattern_figures(rows,summary,args.figures,args.comparison_L,args.plot_betas)
        pattern_source=matched_pattern_figure(HERE/"data/arrangement_selection.csv",args.figures)
    else:
        pattern_source=None
    receipt={
        "generated_utc":datetime.now(timezone.utc).isoformat(),"sources":sources,
        "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "software":{"python":sys.version,"numpy":np.__version__,"matplotlib":matplotlib.__version__},
        "arguments":{k:[str(p) for p in v] if k=="input" else str(v) if isinstance(v,Path) else v
                     for k,v in vars(args).items()},
        "rows":len(rows),"ground_rows":sum(r["state_class"]=="ground" for r in rows),
        "obstructed_rows":sum(r["state_class"]=="obstructed" for r in rows),
        "data_checks":checks,"coverage":coverage,
        "matched_pattern_catalogue_source":pattern_source,
        "fit_counts":{"raw_region":len(rf),"difference":len(df),"geometry_joint":len(jf)},
        "definitions":{
            "normalization":"single-species Renyi-2 entropy, natural logarithms",
            "state_class":"abs(premeasurement_excess) < ground_tol; NEVER classify by measured energy",
            "reference_window":"smallest available fitted Lmin >= reference_lmin, per model",
            "canonical_raw_A_C":"aL+b+d/L",
            "canonical_raw_B":"aL+q*ln(L)+b+d/L; corner coefficient c=-q",
            "D":"(2*S_A+S_C)/3-S_B; independent fit c*ln(L)+b+d/L",
            "joint_model":"SA=aA*L+bA+dA/L; SC=aC*L+bC+dC/L; SB=(2*aA+aC)*L/3-c*ln(L)+bB+dB/L",
            "area_mismatch":"independent aB-(2*aA+aC)/3; geometry is checked, not forced in raw fits",
            "leave_one_out":"raw fits leave one L out; joint fits also report leaving all 3 rows of one L out",
            "uncertainty":"fit-window range, not statistical confidence; numerical/model errors not inferred from RMSE",
            "window_range_scope":"all supported requested Lmin windows, including those below the reference window; inspect individual fits for matched-window comparisons",
        },
        "limitations":[
            "A small raw-entropy residual can coexist with an uncertain subleading logarithm.",
            "Ground-state energy equality does not establish identical zero-mode occupation.",
            "Every comparison is finite-size and specific to the selected initial-state/measurement protocol.",
            "The geometry joint model is a constrained diagnostic; independent fits and held-out residuals remain authoritative checks.",
            "Effective log coefficients are not established thermodynamic corner coefficients when raw-region models disagree or window drift remains substantial.",
            "Ground and obstructed states are never pooled in a fit or cross-pattern figure.",
        ],
    }
    with (args.output/"summary.json").open("w",encoding="utf-8") as handle:
        json.dump(receipt,handle,indent=2,ensure_ascii=False,allow_nan=False)
    print(json.dumps({"rows":len(rows),"classes":{"ground":receipt["ground_rows"],"obstructed":receipt["obstructed_rows"]},
                      "fit_counts":receipt["fit_counts"],"output":str(args.output)},indent=2))


if __name__=="__main__":
    main()
