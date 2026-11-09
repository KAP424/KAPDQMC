"""Independent fixed-window analysis of random_half_3 at beta=1.

Reads named CSV fields, snapshots and hashes inputs, and never calculates states.
The registered plan is analysis_random_half3_beta1/plan.json. Missing expanded
data are allowed in preliminary mode. Final mode checks the planned core grid.
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

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
MODELS = ("log", "log_inv", "log_inv2")
RAW_MODELS = ("area_log", "area_log_inv", "area_log_inv2")
TEXT = {"pattern", "family", "source", "source_file", "source_sha256", "sourcecode",
        "limit_fingerprint", "scan_stage", "grid_version"}
PALETTE = ["#547A98", "#AD7438", "#A65043", "#6F8543", "#765496", "#24678F", "#147A68", "#9E6685", "#43566E"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "h"]


def grouped(rows, keys):
    result = defaultdict(list)
    for row in rows:
        result[tuple(row[key] for key in keys)].append(row)
    return result


def ratio(numerator, denominator):
    return float(numerator / denominator) if abs(denominator) > 1e-12 else None


def read_sources(paths, stage, pattern, beta):
    seen, receipts = {}, []
    identity_error = 0.0
    for path in paths:
        if not path.exists():
            if stage == "final":
                raise FileNotFoundError(path)
            receipts.append(dict(path=str(path.resolve()), missing=True))
            continue
        data = path.read_bytes()
        original = data
        trimmed = False
        if data and not data.endswith(b"\n"):
            if stage == "final":
                raise ValueError(f"Incomplete CSV tail: {path}")
            data = data[:data.rfind(b"\n") + 1]
            trimmed = True
        parsed = list(csv.DictReader(io.StringIO(data.decode("utf-8-sig"))))
        selected = 0
        for source_row, raw in enumerate(parsed, 2):
            if raw.get("pattern") != pattern or float(raw["beta"]) != beta:
                continue
            selected += 1
            row = dict(pattern=pattern, beta=beta, L=int(raw["L"]), theta=float(raw["theta"]))
            for field in ("S_A", "S_C", "S_B", "D", "delta"):
                row[field] = float(raw[field])
                if not math.isfinite(row[field]):
                    raise ValueError(f"Nonfinite entropy {path}:{source_row}")
            identity_error = max(identity_error, abs(row["D"] - (2*row["S_A"] + row["S_C"])/3 + row["S_B"]))
            ground_excess = float(raw.get("premeasurement_excess") or raw.get("premeasurement_limit_excess") or raw["accessible_excess"])
            if abs(ground_excess) >= 1e-8:
                raise ValueError(f"Obstructed accessible limit {path}:{source_row}: {ground_excess}")
            if raw.get("premeasurement_ground", "true").lower() == "false":
                raise ValueError(f"Conflicting ground classification {path}:{source_row}")
            row.update(ground_excess=ground_excess, source_file=path.name, source_row=source_row)
            key = row["L"], row["theta"]
            if key in seen:
                mismatch = max(abs(row[field] - seen[key][field]) for field in ("S_A", "S_C", "S_B", "D"))
                if mismatch > 1e-9:
                    raise ValueError(f"Conflicting duplicate {key}, entropy difference {mismatch}")
            else:
                seen[key] = row
        receipts.append(dict(path=str(path.resolve()), sha256=hashlib.sha256(original).hexdigest(),
                             rows_read=len(parsed), selected_rows=selected, trimmed_partial_tail=trimmed))
    if identity_error > 1e-9:
        raise ValueError(f"D identity error {identity_error}")
    rows = sorted(seen.values(), key=lambda r: (r["L"], r["theta"]))
    return rows, receipts, identity_error


def design(x, model):
    columns, names = [], []
    if model.startswith("area_"):
        columns.append(x)
        names.append("a")
    columns.extend((np.log(x), np.ones(len(x))))
    names.extend(("c", "b"))
    if "inv" in model:
        columns.append(1/x)
        names.append("d")
    if model.endswith("inv2"):
        columns.append(1/x**2)
        names.append("e")
    return np.column_stack(columns), names


def fit(x, y, model):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    matrix, names = design(x, model)
    minimum = len(names) + 2
    if len(x) < minimum:
        return None, f"n={len(x)} < parameters+2={minimum}"
    scale = np.linalg.norm(matrix, axis=0)
    scaled = matrix / scale
    solution, _, rank, _ = np.linalg.lstsq(scaled, y, rcond=None)
    if rank != len(names):
        return None, "rank-deficient numerical design"
    coefficients = solution / scale
    residual = y - matrix @ coefficients
    result = dict(model=model, n=len(x), parameter_count=len(names),
                  x_min=float(min(x)), x_max=float(max(x)), grid=",".join(f"{v:g}" for v in x),
                  condition=float(np.linalg.cond(scaled)), rmse=float(np.sqrt(np.mean(residual**2))),
                  max_abs_residual=float(max(abs(residual))), **dict(zip(names, map(float, coefficients))))
    if model.startswith("area_"):
        result["corner_c"] = -result["c"]
    log_span = abs(result["c"] * math.log(max(x)/min(x)))
    result["inverse1_to_log_span"] = ratio(abs(result.get("d", 0)) * (1/min(x)-1/max(x)), log_span)
    result["inverse2_to_log_span"] = ratio(abs(result.get("e", 0)) * (1/min(x)**2-1/max(x)**2), log_span)
    return result, ""


def predict(fit_row, x):
    matrix, names = design(np.asarray(x, dtype=float), fit_row["model"])
    return matrix @ np.array([fit_row[name] for name in names])


def analyze_equilibrium(rows, plan, missing):
    maximum = max(r["L"] for r in rows)
    fits = []
    for high in sorted({90, maximum}):
        for low in plan["equilibrium_Lmin_checks"]:
            subset = [r for r in rows if low <= r["L"] <= high]
            if not subset:
                continue
            for observable, models in (("D", MODELS), ("S_B", RAW_MODELS)):
                for model in models:
                    meta = dict(domain="equilibrium", observable=observable, requested_Lmin=low, requested_Lmax=high)
                    result, reason = fit([r["L"] for r in subset], [r[observable] for r in subset], model)
                    if result:
                        fits.append(dict(**meta, **result))
                    else:
                        missing.append(dict(**meta, model=model, reason=reason))
    main = next(r for r in fits if r["observable"] == "D" and r["model"] == "log_inv"
                and r["requested_Lmin"] == 18 and r["requested_Lmax"] == maximum)
    legacy = next(r for r in fits if r["observable"] == "D" and r["model"] == "log_inv"
                  and r["requested_Lmin"] == 18 and r["requested_Lmax"] == 90)
    raw = next(r for r in fits if r["observable"] == "S_B" and r["model"] == "area_log_inv"
               and r["requested_Lmin"] == 18 and r["requested_Lmax"] == maximum)
    return fits, main, legacy, raw


def complete(rows, low, high, minimum=5):
    subset = sorted([r for r in rows if low <= r["theta"] <= high], key=lambda r: r["theta"])
    return subset if len(subset) >= minimum and subset[0]["theta"] == low and subset[-1]["theta"] == high else None


def analyze_time(rows, plan, eq, legacy, missing):
    by_size = {key[0]: values for key, values in grouped(rows, ("L",)).items()}
    full, rolling, comparisons, windows, pointwise = [], [], [], [], []
    for small, large in itertools.combinations(sorted(by_size), 2):
        first = {r["theta"]: r for r in by_size[small]}
        second = {r["theta"]: r for r in by_size[large]}
        for theta in sorted(set(first).intersection(second)):
            pointwise.append(dict(size_small=small, size_large=large, theta=theta,
                                  D_small=first[theta]["D"], D_large=second[theta]["D"],
                                  D_difference=second[theta]["D"]-first[theta]["D"]))
    for index, (low, high) in enumerate(plan["time_windows"]):
        meta = dict(window_index=index, window=f"{low:g}-{high:g}", theta_min=low, theta_max=high)
        blocks = {size: subset for size, values in by_size.items() if (subset := complete(values, low, high)) is not None}
        for size, subset in sorted(blocks.items()):
            for model in MODELS:
                result, reason = fit([r["theta"] for r in subset], [r["D"] for r in subset], model)
                if result:
                    full.append(dict(**meta, L=size, fit_scope="per_size_full_grid", **result))
                else:
                    missing.append(dict(domain="time_full_grid", **meta, L=size, model=model, reason=reason))
        chosen = None
        for large, small in itertools.combinations(sorted(blocks, reverse=True), 2):
            lookup = {size: {r["theta"]: r for r in blocks[size]} for size in (small, large)}
            times = sorted(set(lookup[small]).intersection(lookup[large]))
            if len(times) < 5 or times[0] != low or times[-1] != high:
                continue
            if chosen is None:
                chosen = small, large, times
            dsmall = np.array([lookup[small][t]["D"] for t in times])
            dlarge = np.array([lookup[large][t]["D"] for t in times])
            difference = dlarge - dsmall
            for model in MODELS:
                fs, reason_s = fit(times, dsmall, model)
                fl, reason_l = fit(times, dlarge, model)
                if fs is None or fl is None:
                    missing.append(dict(domain="time_size_pair", **meta, size_small=small, size_large=large,
                                        model=model, reason=reason_s or reason_l))
                    continue
                comparisons.append(dict(**meta, size_small=small, size_large=large, model=model,
                    n=len(times), grid=",".join(f"{t:g}" for t in times), c_small=fs["c"], c_large=fl["c"],
                    coefficient_difference=fl["c"]-fs["c"],
                    coefficient_size_drift_eq=ratio(abs(fl["c"]-fs["c"]), abs(eq["c"])),
                    coefficient_size_drift_time=ratio(abs(fl["c"]-fs["c"]), abs(fl["c"])),
                    max_D_difference=float(max(abs(difference))), rms_D_difference=float(np.sqrt(np.mean(difference**2))),
                    curve_difference_eq=ratio(float(max(abs(difference))), abs(eq["c"])*math.log(high/low)),
                    curve_difference_time=ratio(float(max(abs(difference))), abs(fl["c"])*math.log(high/low)),
                    relative_eq_error=ratio(fl["c"]-eq["c"], abs(eq["c"])),
                    rmse_large=fl["rmse"], inverse1_to_log_span_large=fl["inverse1_to_log_span"],
                    inverse2_to_log_span_large=fl["inverse2_to_log_span"], condition_large=fl["condition"]))
        status = dict(**meta, complete_sizes=",".join(map(str, sorted(blocks))))
        if chosen is None:
            status.update(status="missing_complete_common_grid", reason="Two complete sizes with at least five common times required")
            windows.append(status)
            continue
        small, large, times = chosen
        status.update(status="available", size_small=small, size_large=large, n_common=len(times),
                      common_grid=",".join(f"{t:g}" for t in times))
        eligible = []
        for size, subset in sorted(blocks.items()):
            lookup = {r["theta"]: r for r in subset}
            if not all(t in lookup for t in times):
                continue
            eligible.append(size)
            for model in MODELS:
                result, reason = fit(times, [lookup[t]["D"] for t in times], model)
                if result:
                    rolling.append(dict(**meta, L=size, primary_L=large, comparison_small=small,
                        fit_scope="largest_pair_common_grid", **result,
                        relative_eq_error=ratio(result["c"]-eq["c"], abs(eq["c"])),
                        relative_legacy_eq_error=ratio(result["c"]-legacy["c"], abs(legacy["c"]))))
                else:
                    missing.append(dict(domain="time_common_grid", **meta, L=size, model=model, reason=reason))
        status["eligible_common_grid_sizes"] = ",".join(map(str, eligible))
        windows.append(status)
    return full, rolling, comparisons, windows, pointwise


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def finite_differences(rows, plan):
    result = []
    for (size,), block in sorted(grouped(rows, ("L",)).items()):
        lookup = {r["theta"]: r["D"] for r in block}
        for time in plan["finite_difference_times"]:
            pair = all(t in lookup for t in (time, 2*time))
            triple = pair and 4*time in lookup
            row = dict(L=size, t=time, pair_available=pair, triple_available=triple,
                       D_t=lookup.get(time), D_2t=lookup.get(2*time), D_4t=lookup.get(4*time))
            row["s"] = (lookup[2*time]-lookup[time])/math.log(2) if pair else None
            row["c_eff"] = (2*lookup[4*time]-3*lookup[2*time]+lookup[time])/math.log(2) if triple else None
            result.append(row)
    return result


def main_window_summary(rolling, comparisons, windows, eq):
    """Compact table keeps all registered windows and the actual size pair."""
    result = []
    for window in windows:
        row = dict(window)
        row["equilibrium_c"] = eq["c"]
        if window["status"] == "available":
            for model in MODELS:
                large_fit = next((r for r in rolling if r["window_index"] == window["window_index"]
                                  and r["L"] == window["size_large"] and r["model"] == model), None)
                row["c_" + model] = large_fit["c"] if large_fit else None
            comparison = next(r for r in comparisons if r["window_index"] == window["window_index"]
                              and r["size_small"] == window["size_small"]
                              and r["size_large"] == window["size_large"] and r["model"] == "log_inv")
            for field in ("c_small", "c_large", "relative_eq_error", "coefficient_size_drift_eq",
                          "coefficient_size_drift_time", "max_D_difference", "curve_difference_eq",
                          "curve_difference_time", "rmse_large", "inverse1_to_log_span_large", "condition_large"):
                row[field] = comparison[field]
        result.append(row)
    return result


def findings_markdown(summary, eq_fits, windows, sparse_rolling, primary_comparisons):
    """Reproducible numeric tables; interpretation remains separate from fits."""
    eq = summary["main_equilibrium_fit"]
    maximum = int(eq["x_max"])
    def num(value, precision=7):
        return "unavailable" if value is None else f"{value:.{precision}f}"
    def pct(value):
        return "unavailable" if value is None else f"{100*value:+.3f}%"
    lines = [f"# random_half_3, beta=1: {summary['stage']} analysis tables", "",
             "All coefficients use natural logarithms and one-species Renyi-2 entropy. "
             "These deterministic fits have no statistical error bars or confidence intervals.", "",
             f"Main equilibrium reference: L18..{maximum}, corrected D coefficient {eq['c']:.9f}.", "",
             "## All registered time windows", "",
             "Main model:", "", "$$", r"D(\theta)=c\log\theta+b+\frac{d}{\theta}", "$$", "",
             "Each row uses exactly the same measured "
             "time grid at the two stated sizes. Relative error is signed; drift and curve difference are absolute. "
             "All percentages use the main equilibrium coefficient (curve difference also divides by the log time span).", "",
             "| Window | Sizes | c_small | c_large | Relative eq error | Size coefficient drift | Curve difference |", 
             "|---|---:|---:|---:|---:|---:|---:|"]
    for r in windows:
        lines.append(f"| {r['window']} | {r.get('size_small')}/{r.get('size_large')} | {num(r.get('c_small'))} | "
                     f"{num(r.get('c_large'))} | {pct(r.get('relative_eq_error'))} | "
                     f"{pct(r.get('coefficient_size_drift_eq'))} | {pct(r.get('curve_difference_eq'))} |")
    lines += ["", "## Equilibrium model sensitivity", "",
              f"All models below use L18..{maximum}. The raw-B corner coefficient is the negative of its fitted log coefficient.", "",
              "| Observable | Model | Corner coefficient | Area coefficient | RMSE | Scaled condition |",
              "|---|---|---:|---:|---:|---:|"]
    for r in eq_fits:
        if r["requested_Lmin"] == 18 and r["requested_Lmax"] == maximum:
            lines.append(f"| {r['observable']} | {r['model']} | {num(r.get('corner_c', r['c']))} | "
                         f"{num(r.get('a'))} | {r['rmse']:.3g} | {r['condition']:.3g} |")
    lines += ["", "## Equilibrium lower-cutoff sensitivity", "",
              "| L minimum | Corrected D coefficient | Independent raw-B coefficient |",
              "|---:|---:|---:|"]
    for low in summary["plan"]["equilibrium_Lmin_checks"]:
        first = next((r for r in eq_fits if r["requested_Lmin"] == low and r["requested_Lmax"] == maximum and r["model"] == "log_inv"), None)
        second = next((r for r in eq_fits if r["requested_Lmin"] == low and r["requested_Lmax"] == maximum and r["model"] == "area_log_inv"), None)
        lines.append(f"| {low} | {num(first['c'] if first else None)} | {num(second['corner_c'] if second else None)} |")
    lines += ["", "## Time-model and sampling sensitivity", "",
              "Primary grids include registered theta=9,11 samples when both selected sizes have them. "
              "The sparse diagnostic removes these samples before all fits. Four-parameter fits require at least six common points. "
              "A missing fit is not evidence for or against that model.", "",
              "| Window | L | Primary n | Pure log c | Primary corrected c | Original sparse corrected c | Primary four-parameter c | Sparse four-parameter c |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in windows:
        size = r.get("size_large")
        sparse = {x["model"]: x for x in sparse_rolling if x["window_index"] == r["window_index"] and x["L"] == size}
        lines.append(f"| {r['window']} | {size} | {r.get('n_common')} | {num(r.get('c_log'))} | {num(r.get('c_log_inv'))} | "
                     f"{num(sparse.get('log_inv', {}).get('c'))} | {num(r.get('c_log_inv2'))} | {num(sparse.get('log_inv2', {}).get('c'))} |")
    lines += ["", "## Intermediate-window model and size diagnostics", "",
              "These two pre-registered windows are shown for the requested model check; the full seven-window results above remain part of the analysis.", "",
              "| Window | Model | Sizes | c_small | c_large | Relative eq error | Size coefficient drift | RMSE | Inverse1/log span | Inverse2/log span |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in primary_comparisons:
        if r["window"] in ("6-12", "8-16"):
            lines.append(f"| {r['window']} | {r['model']} | {r['size_small']}/{r['size_large']} | "
                         f"{num(r['c_small'])} | {num(r['c_large'])} | {pct(r['relative_eq_error'])} | "
                         f"{pct(r['coefficient_size_drift_eq'])} | {r['rmse_large']:.3g} | "
                         f"{num(r['inverse1_to_log_span_large'], 3)} | {num(r['inverse2_to_log_span_large'], 3)} |")
    lines += ["", "## Receipt", "",
              f"Remaining planned finite keys: {len(summary['missing_planned_core_keys'])}; "
              f"remaining planned equilibrium sizes: {len(summary['missing_planned_equilibrium_sizes'])}.", "",
              "Exact source, script, and plan hashes are recorded in summary.json. "
              "All individual coefficient, inverse-power correction, residual, and conditioning diagnostics remain in the CSV tables. "
              "No time window was selected by closeness to the equilibrium reference.", ""]
    return "\n".join(lines)


def save(fig, directory, name):
    for extension in ("png", "pdf"):
        fig.savefig(directory / f"{name}.{extension}", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def ln_ticks(ax, values):
    """Label explicit natural-log coordinates on an ordinary linear axis."""
    positions = np.log(values)
    if len(positions) > 6:
        positions = positions[np.unique(np.linspace(0, len(positions)-1, 6).astype(int))]
    ax.set_xscale("linear")
    ax.set_xticks(positions, [f"{value:.2f}" for value in positions])


def make_plots(eq_rows, time_rows, eq_fits, eq, raw, rolling, comparisons, windows, pointwise, differences, sparse_rolling, plan, directory, stage):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
                         "axes.spines.top": False, "axes.spines.right": False, "legend.fontsize": 7,
                         "savefig.dpi": 180, "pdf.fonttype": 42})
    sizes = sorted({r["L"] for r in time_rows})
    colors = {size: PALETTE[i % len(PALETTE)] for i, size in enumerate(sizes)}
    markers = {size: MARKERS[i % len(MARKERS)] for i, size in enumerate(sizes)}
    largest_eq = max(r["L"] for r in eq_rows)
    main_fits = [r for r in eq_fits if r["requested_Lmin"] == 18 and r["requested_Lmax"] == largest_eq]
    grid = np.geomspace(18, largest_eq, 250)
    line_styles = {"log": "--", "log_inv": "-", "log_inv2": ":"}

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.7), constrained_layout=True)
    L = np.array([r["L"] for r in eq_rows]); D = np.array([r["D"] for r in eq_rows]); B = np.array([r["S_B"] for r in eq_rows])
    axes[0, 0].plot(np.log(L), D, "o", color="#24678F", label="Measured D", markersize=4)
    axes[0, 1].plot(np.log(L), B, "o", color="#24678F", label="Measured S_B", markersize=4)
    axes[1, 0].plot(np.log(L), B-raw["a"]*L, "o", color="#24678F", label="S_B - fitted a_B L", markersize=4)
    for result in main_fits:
        name = result["model"].removeprefix("area_")
        style = line_styles[name]
        if result["observable"] == "D":
            axes[0, 0].plot(np.log(grid), predict(result, grid), style, label=result["model"])
            subset = [r for r in eq_rows if 18 <= r["L"] <= largest_eq]
            values = [r["L"] for r in subset]
            axes[1, 1].plot(values, np.array([r["D"] for r in subset])-predict(result, values), marker=".", linestyle=style, label=result["model"])
        else:
            axes[0, 1].plot(np.log(grid), predict(result, grid), style, label=result["model"])
            subset = [r for r in eq_rows if 18 <= r["L"] <= largest_eq]
            values = [r["L"] for r in subset]
            axes[1, 2].plot(values, np.array([r["S_B"] for r in subset])-predict(result, values), marker=".", linestyle=style, label=result["model"])
    axes[1, 0].plot(np.log(grid), predict(raw, grid)-raw["a"]*grid, color="#243746", label="Independent raw-B fit")
    for observable, model, color, marker in (("D", "log_inv", "#24678F", "o"), ("S_B", "area_log_inv", "#AA5746", "s")):
        block = [r for r in eq_fits if r["observable"] == observable and r["model"] == model and r["requested_Lmax"] == largest_eq]
        axes[0, 2].plot([r["x_min"] for r in block], [r["c"] if observable == "D" else r["corner_c"] for r in block], marker=marker, color=color, label=observable)
    axes[0, 2].set(xlabel="Minimum fitted L", ylabel="Log coefficient", title="Lower-cutoff sensitivity, main models")
    axes[0, 0].set(title=f"D: c={eq['c']:.7g}", ylabel="D", xlabel=r"$\ln L$")
    axes[0, 1].set(title=f"Raw B: independent c_B={raw['corner_c']:.7g}", ylabel="S_B", xlabel=r"$\ln L$")
    axes[1, 0].set(title=f"Area-subtracted diagnostic; a_B={raw['a']:.6g}", ylabel="S_B - fitted a_B L", xlabel=r"$\ln L$")
    axes[1, 1].set(title="D residuals", ylabel="Data - fit", xlabel="L")
    axes[1, 2].set(title="Raw B residuals", ylabel="Data - fit", xlabel="L")
    ticks = [min(L)] + [v for v in (18, 36, 54, 90, 144, 180, 216) if min(L) < v <= max(L)]
    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
        ln_ticks(ax, ticks)
    for ax in axes.flat:
        ax.grid(axis="y", color="#DFE5EA", linewidth=.5); ax.legend()
    fig.suptitle(f"random_half_3, beta=1: independent equilibrium diagnostics ({stage})\nFits start at L=18; deterministic data, no statistical error bars")
    save(fig, directory, "equilibrium_reference")

    fig, axes = plt.subplots(1, 2, figsize=(11.7, 4.8), constrained_layout=True)
    axes[0].plot(np.log(L), D, "o", color="#24678F", markersize=4, label="Measured D")
    axes[0].plot(np.log(grid), predict(eq, grid), color="#243746",
                 label=f"c ln L + b + d/L; c={eq['c']:.6f}")
    axes[1].plot(np.log(L), B-raw["a"]*L, "s", color="#AA5746", markersize=4,
                 label="Measured S_B - fitted a_B L")
    axes[1].plot(np.log(grid), predict(raw, grid)-raw["a"]*grid, color="#243746",
                 label=f"-c_B ln L + b + d/L; c_B={raw['corner_c']:.6f}")
    axes[0].set(title="Area-cancelled entropy combination", ylabel="D = (2S_A + S_C)/3 - S_B")
    axes[1].set(title="Independent raw-B fit after area subtraction", ylabel="S_B - fitted a_B L")
    for ax in axes:
        ln_ticks(ax, ticks)
        ax.set_xlabel(r"$\ln L$")
        ax.grid(axis="y", color="#DFE5EA", linewidth=.5)
        ax.legend(loc="best", fontsize=8)
    fig.suptitle(f"random_half_3, beta=1: equilibrium scaling ({stage})\n"
                 f"Independent fits on L=18..{largest_eq}; raw-B area coefficient a_B={raw['a']:.7g}")
    save(fig, directory, "equilibrium_scaling")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), constrained_layout=True)
    for size, block in sorted((key[0], value) for key, value in grouped(time_rows, ("L",)).items()):
        block = sorted(block, key=lambda r: r["theta"])
        for ax, minimum in zip(axes, (2, 8)):
            selected = [r for r in block if r["theta"] >= minimum]
            if selected:
                ax.plot(np.log([r["theta"] for r in selected]), [r["D"] for r in selected], color=colors[size], marker=markers[size], markersize=3.5, linewidth=1, label=f"L={size}")
            ln_ticks(ax, [v for v in (2, 3, 4, 6, 8, 12, 16, 24, 32) if minimum <= v <= max(r["theta"] for r in time_rows)])
            ax.set(xlabel=r"$\ln\theta$", ylabel="D, no vertical shifts")
    axes[0].set_title("All measured times in the registered range")
    axes[1].set_title("Late-time detail, same unshifted observable")
    for ax in axes:
        ax.grid(axis="y", color="#DFE5EA", linewidth=.5); ax.legend(ncol=2)
    fig.suptitle(f"random_half_3, beta=1: finite-time size dependence ({stage})")
    save(fig, directory, "nonequilibrium_D")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6), constrained_layout=True)
    labels = [f"{lo:g}-{hi:g}" for lo, hi in plan["time_windows"]]
    for ax, model in zip(axes, MODELS):
        ax.axhspan(.9, 1.1, color="#D7E3EE", label="+/-10% reference band")
        ax.axhspan(.95, 1.05, color="#AAC5DA", label="+/-5% reference band")
        ax.axhline(1, color="#465B6A", linewidth=.8)
        for size in sizes:
            block = [r for r in rolling if r["L"] == size and r["model"] == model]
            if block:
                by_window = {r["window_index"]: r["c"]/eq["c"] for r in block}
                ax.plot(range(len(labels)), [by_window.get(i, np.nan) for i in range(len(labels))],
                        color=colors[size], marker=markers[size], linewidth=1, label=f"L={size}")
        unavailable = [i for i in range(len(labels)) if not any(r["model"] == model and r["window_index"] == i for r in rolling)]
        for index in unavailable:
            ax.axvspan(index-.45, index+.45, color="#EEEEEE", alpha=.7, zorder=0)
        if unavailable:
            ax.text(.02, .02, "Gray windows: fewer than parameters + 2 common points",
                    transform=ax.transAxes, fontsize=6.5, color="#555555")
        ax.set_xticks(range(len(labels)), labels, rotation=40)
        ax.set(title=model, xlabel="Fixed measured time window", ylabel="c_time / c_eq")
        ax.grid(axis="y", color="#DFE5EA", linewidth=.5)
        ax.legend(ncol=2, fontsize=6.7)
    fig.suptitle(f"random_half_3, beta=1: all fixed-window coefficients ({stage})\nEach window uses one identical measured grid across the plotted sizes; bands are tolerances, not confidence intervals")
    save(fig, directory, "rolling_coefficients")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), constrained_layout=True)
    refined_sizes = sorted({size for addition in plan.get("adaptive_sampling_grids", []) for size in addition["sizes"]})
    for ax, model in zip(axes, MODELS):
        ax.axhline(eq["c"], color="#888888", linewidth=.8, label="Equilibrium reference")
        for size in refined_sizes:
            for fits, style, suffix in ((rolling, "-", "primary grid"), (sparse_rolling, "--", "original sparse grid")):
                block = [r for r in fits if r["L"] == size and r["model"] == model]
                if block:
                    by_window = {r["window_index"]: r["c"] for r in block}
                    ax.plot(range(len(labels)), [by_window.get(i, np.nan) for i in range(len(labels))],
                            color=colors[size], marker=markers[size], linestyle=style,
                            markersize=4, linewidth=1, label=f"L={size}, {suffix}")
        ax.set_xticks(range(len(labels)), labels, rotation=40)
        ax.set(title=model, xlabel="Same fixed time window", ylabel="Fitted time coefficient")
        ax.grid(axis="y", color="#DFE5EA", linewidth=.5)
        ax.legend(ncol=1, fontsize=6.5)
    fig.suptitle(f"random_half_3, beta=1: registered time-grid sensitivity ({stage})\n"
                 "Primary and original sparse grids are fitted separately; no mixed-grid size comparisons")
    save(fig, directory, "time_grid_sensitivity")

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.7), constrained_layout=True)
    adjacent_pairs = list(zip(sizes[:-1], sizes[1:]))
    for small, large in adjacent_pairs:
        points = [r for r in pointwise if r["size_small"] == small and r["size_large"] == large]
        if points:
            axes[0, 0].plot([r["theta"] for r in points], [r["D_difference"] for r in points], color=colors[large], marker=markers[large], markersize=3, label=f"{small}/{large}")
        block = [r for r in comparisons if r["size_small"] == small and r["size_large"] == large and r["model"] == "log_inv"]
        if block:
            axes[0, 1].plot([r["window_index"] for r in block], [100*r["curve_difference_eq"] for r in block], color=colors[large], marker=markers[large], label=f"{small}/{large}")
            axes[1, 0].plot([r["window_index"] for r in block], [100*r["coefficient_size_drift_eq"] for r in block], color=colors[large], marker=markers[large], label=f"{small}/{large}")
    for model in MODELS:
        block = [r for r in rolling if r["L"] == r["primary_L"] and r["model"] == model]
        if block:
            axes[1, 1].plot([r["window_index"] for r in block], [r["inverse1_to_log_span"] + r["inverse2_to_log_span"] for r in block], marker="o", label=model)
    axes[0, 0].set(title="Direct common-time differences", xlabel="theta", ylabel="D_large - D_small")
    axes[0, 1].set(title="Window maximum curve difference", ylabel="max |Delta D| / (|c_eq| log span), %")
    axes[1, 0].set(title="Common-grid coefficient size drift", ylabel="|Delta c| / |c_eq|, %")
    axes[1, 1].set(title="Inverse-power correction / log span", ylabel="Absolute endpoint contributions ratio")
    for ax in (axes[0, 1], axes[1, 0], axes[1, 1]):
        ax.set_xticks(range(len(labels)), labels, rotation=40); ax.set_xlabel("Fixed time window")
    for ax in axes.flat:
        ax.grid(axis="y", color="#DFE5EA", linewidth=.5); ax.legend(ncol=2)
    fig.suptitle(f"random_half_3, beta=1: finite-size and correction checks ({stage})\nUnshifted curves; no collapse or asymptotic law is assumed")
    save(fig, directory, "finite_size_checks")

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.4), constrained_layout=True)
    for size in sizes:
        for ax, key in zip(axes[0], ("s", "c_eff")):
            block = [r for r in differences if r["L"] == size and r[key] is not None]
            if block:
                ax.plot([r["t"] for r in block], [r[key] for r in block], color=colors[size], marker=markers[size], label=f"L={size}")
    lookup = {(r["L"], r["t"]): r for r in differences}
    for small, large in adjacent_pairs:
        for ax, key in zip(axes[1], ("s", "c_eff")):
            values = [(time, lookup[large, time][key]-lookup[small, time][key])
                      for time in plan["finite_difference_times"]
                      if lookup[small, time][key] is not None and lookup[large, time][key] is not None]
            if values:
                ax.plot([t for t, _ in values], [v for _, v in values], color=colors[large], marker=markers[large], label=f"{small}/{large}")
    axes[0, 0].set(title="s(t) = [D(2t)-D(t)] / ln 2", ylabel="Local log slope")
    axes[0, 1].set(title="c_eff(t) = [2D(4t)-3D(2t)+D(t)] / ln 2", ylabel="1/theta-cancelled coefficient")
    axes[1, 0].set(title="Size difference in s(t)", ylabel="s_large - s_small")
    axes[1, 1].set(title="Size difference in c_eff(t)", ylabel="c_eff_large - c_eff_small")
    for ax in axes.flat:
        ax.set_xlabel("Base time t; only measured pairs/triples")
        ax.set_xticks(plan["finite_difference_times"])
        ax.grid(axis="y", color="#DFE5EA", linewidth=.5); ax.legend(ncol=2)
    fig.suptitle(f"random_half_3, beta=1: equilibrium-independent time diagnostics ({stage})\nNo fitted equilibrium coefficient, interpolation, or statistical error bars")
    save(fig, directory, "finite_difference_diagnostics")
    return ["equilibrium_reference", "equilibrium_scaling", "nonequilibrium_D", "rolling_coefficients", "time_grid_sensitivity", "finite_size_checks", "finite_difference_diagnostics"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, default=[HERE / "data/equilibrium_extended.csv",
        HERE / "data/observations.csv", HERE / "data/extraction_observations.csv", HERE / "data/random_half3_beta1_observations.csv",
        HERE / "data/random_half3_beta1_equilibrium.csv"])
    parser.add_argument("--plan", type=Path, default=HERE / "analysis_random_half3_beta1/plan.json")
    parser.add_argument("--output", type=Path, default=HERE / "analysis_random_half3_beta1")
    parser.add_argument("--figures", type=Path, default=HERE / "figures_random_half3_beta1")
    parser.add_argument("--stage", choices=("preliminary", "final"), default="preliminary")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    plan_bytes = args.plan.read_bytes(); plan = json.loads(plan_bytes)
    rows, sources, identity_error = read_sources(args.inputs, args.stage, plan["pattern"], plan["beta"])
    eq_rows = [r for r in rows if math.isinf(r["theta"])]
    time_rows = [r for r in rows if 2 <= r["theta"] <= max(hi for _, hi in plan["time_windows"])]
    expected = {(L, theta) for L in plan["planned_finite_sizes"] for theta in plan["planned_times"]}
    for addition in plan.get("adaptive_finite_grids", []):
        expected.update((L, theta) for L in addition["sizes"] for theta in addition["times"])
    for addition in plan.get("adaptive_sampling_grids", []):
        expected.update((L, theta) for L in addition["sizes"] for theta in addition["times"])
    present = {(r["L"], r["theta"]) for r in time_rows}
    missing_core = sorted(expected - present)
    required_eq = set(plan["planned_new_equilibrium_sizes"] + plan.get("adaptive_equilibrium_sizes", []))
    missing_eq = sorted(required_eq - {r["L"] for r in eq_rows})
    if args.stage == "final" and (missing_core or missing_eq):
        raise ValueError("The registered finite core grid or equilibrium maximum size is incomplete")
    missing = []
    eq_fits, eq, legacy, raw = analyze_equilibrium(eq_rows, plan, missing)
    full, rolling, comparisons, windows, pointwise = analyze_time(time_rows, plan, eq, legacy, missing)
    differences = finite_differences(time_rows, plan)
    window_summary = main_window_summary(rolling, comparisons, windows, eq)
    primary_pairs = {(r["window_index"], r["size_small"], r["size_large"]) for r in windows if r["status"] == "available"}
    primary_comparisons = [r for r in comparisons if (r["window_index"], r["size_small"], r["size_large"]) in primary_pairs]
    excluded_times = plan.get("sparse_original_grid_diagnostic", {}).get("excluded_added_times", [])
    sparse_missing = []
    _, sparse_rolling, sparse_comparisons, sparse_windows, _ = analyze_time(
        [r for r in time_rows if r["theta"] not in excluded_times], plan, eq, legacy, sparse_missing)
    sparse_summary = main_window_summary(sparse_rolling, sparse_comparisons, sparse_windows, eq)
    args.output.mkdir(parents=True, exist_ok=True); args.figures.mkdir(parents=True, exist_ok=True)
    for name, values in (("equilibrium_fits", eq_fits), ("time_full_grid_fits", full),
        ("time_common_grid_fits", rolling), ("size_window_comparisons", comparisons),
        ("time_window_coverage", windows), ("size_pointwise", pointwise), ("finite_difference_diagnostics", differences),
        ("unavailable_fits", missing), ("main_window_summary", window_summary),
        ("primary_model_comparisons", primary_comparisons),
        ("time_sparse_grid_fits", sparse_rolling), ("sparse_size_window_comparisons", sparse_comparisons),
        ("sparse_window_coverage", sparse_windows), ("sparse_main_window_summary", sparse_summary),
        ("sparse_unavailable_fits", sparse_missing)):
        write_csv(args.output / (name + ".csv"), values)
    write_csv(args.output / "selected_observations.csv", rows)
    figures = [] if args.no_plots else make_plots(eq_rows, time_rows, eq_fits, eq, raw, rolling,
        comparisons, windows, pointwise, differences, sparse_rolling, plan, args.figures, args.stage)
    changed = [r["path"] for r in sources if not r.get("missing") and
               hashlib.sha256(Path(r["path"]).read_bytes()).hexdigest() != r["sha256"]]
    if args.stage == "final" and changed:
        raise ValueError(f"Inputs changed during final analysis: {changed}")
    summary = dict(generated_utc=datetime.now(timezone.utc).isoformat(), stage=args.stage,
        sources=sources, sources_changed_during_analysis=changed,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        plan_sha256=hashlib.sha256(plan_bytes).hexdigest(), plan=plan,
        equilibrium_rows=len(eq_rows), finite_rows_in_analysis_range=len(time_rows),
        finite_sizes=sorted({r["L"] for r in time_rows}), equilibrium_sizes=sorted({r["L"] for r in eq_rows}),
        missing_planned_core_keys=missing_core, missing_planned_equilibrium_sizes=missing_eq,
        max_entropy_identity_error=identity_error,
        main_equilibrium_fit=eq, legacy_equilibrium_fit=legacy, independent_raw_B_fit=raw,
        raw_B_vs_D=ratio(abs(raw["corner_c"]-eq["c"]), abs(eq["c"])), figures=figures,
        software=dict(numpy=np.__version__, matplotlib=matplotlib.__version__),
        notes=["No statistical errors or confidence intervals: observations are deterministic.",
               "All registered windows and model availability are retained; no target-matching window selection.",
               "Time-size comparisons are fitted on identical measured grids; full-grid fits are separate diagnostics.",
               "Registered theta=9,11 densification is compared with separately fitted original sparse grids.",
               "A small residual alone cannot establish an asymptotic logarithm; inspect correction/model/window drift."])
    summary["narrative_markdown"] = findings_markdown(summary, eq_fits, window_summary, sparse_rolling, primary_comparisons)
    summary["report_section"] = "RESULTS.md#random-half3-beta1"
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(dict(stage=args.stage, equilibrium_sizes=summary["equilibrium_sizes"], finite_sizes=summary["finite_sizes"],
                          main_eq_c=eq["c"], raw_B_c=raw["corner_c"], missing_planned_core_keys=len(missing_core)), indent=2))


if __name__ == "__main__":
    main()
