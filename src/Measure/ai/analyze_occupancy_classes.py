"""Six-pattern occupancy-class analysis with explicit, per-pattern size grids.

Only this module's output directories are written. Audited numerical helpers in
analyze_uniformA_dense and plotting/raw-entropy helpers in analyze_beta_patterns
are imported read-only; no uniform-A entropy symmetry is assumed.
"""
from __future__ import annotations

import argparse
import csv
import io
import itertools
import json
import math
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import analyze_uniformA_dense as core
import analyze_beta_patterns as previous
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DEFAULT_WINDOWS = [[4, 7], [4.5, 7], [4, 6.5], [5, 7], [5, 8]]
DEFAULT_INPUTS = [HERE / "data" / name for name in (
    "ground_observations_baseline.csv", "ref_calibration_observations.csv",
    "ref_calibration_equilibrium.csv", "ref_calibration_late.csv",
    "beta_patterns_observations.csv", "occupancy_classes_observations.csv")]
CLASS_LABELS = {
    1: "One particle per cell",
    2: "Empty / double cells only",
    3: "Mixed cell occupancies",
}


def catalog_from_plan(plan):
    supplied = plan.get("pattern_metadata", {})
    configured = plan.get("patterns", list(supplied))
    if isinstance(configured, dict):
        supplied = {**configured, **supplied}
        names = list(configured)
    else:
        names = list(configured)
    if not names or len(names) != len(set(names)):
        raise ValueError("Plan needs a nonempty list of distinct pattern names")
    catalog = {}
    for name in names:
        raw = dict(supplied.get(name, {}))
        m, bits = int(raw["m"]), str(raw["occupied_bits"])
        if len(bits) != 2*m*m or set(bits) - {"0", "1"} or bits.count("1") != m*m:
            raise ValueError(f"{name}: mask must encode exactly half filling in an m by m supercell")
        counts = Counter(int(bits[i])+int(bits[i+1]) for i in range(0, len(bits), 2))
        category = 1 if counts[1] == m*m else 2 if counts[1] == 0 else 3
        declared = raw.get("occupancy_class", raw.get("class_id", category))
        if int(declared) != category:
            raise ValueError(f"{name}: declared class does not match actual cell occupancies")
        family = "special" if category == 1 else "general"
        if raw.get("family", family) != family:
            raise ValueError(f"{name}: family does not match actual cell occupancies")
        default_eq = dict(Lmin=24, Lmax=144, lower_cuts=[24, 36, 48]) if m == 4 else dict(
            Lmin=18, Lmax=144, lower_cuts=[18, 36, 54])
        eq = {**default_eq, **plan.get("equilibrium_by_pattern", {}).get(name, {}),
              **raw.get("equilibrium", {})}
        low, high = int(eq["Lmin"]), int(eq["Lmax"])
        cuts = sorted(set(int(x) for x in eq["lower_cuts"]) | {low})
        if low >= high or any(x % (3*m) or x < low or x >= high for x in cuts) or high % (3*m):
            raise ValueError(f"{name}: equilibrium bounds must be actual compatible sizes")
        catalog[name] = dict(m=m, occupied_bits=bits, family=family, occupancy_class=category,
                             class_label=raw.get("class_label", CLASS_LABELS[category]),
                             double_cells=counts[2], empty_cells=counts[0], single_cells=counts[1],
                             equilibrium=dict(Lmin=low, Lmax=high, lower_cuts=cuts))
    return names, catalog


def tag(name, catalog, rows):
    meta = catalog[name]
    return [dict(pattern=name, m=meta["m"], family=meta["family"],
                 occupancy_class=meta["occupancy_class"], **row) for row in rows]


def read_sources(paths, stage, output, catalog, betas):
    selected, sources, duplicates = {}, [], []
    snapshots = output / "source_snapshots"
    snapshots.mkdir(parents=True, exist_ok=True)
    for path in paths:
        if not path.exists():
            sources.append(dict(path=str(path.resolve()), missing=True))
            continue
        original = path.read_bytes()
        digest, content, ignored = core.sha_bytes(original), original, 0
        if content and not content.endswith(b"\n"):
            if stage == "final":
                raise ValueError(f"Incomplete CSV tail: {path}")
            end = content.rfind(b"\n")+1
            ignored, content = len(content)-end, content[:end]
        snapshot = snapshots / f"{path.stem}_{digest[:12]}.csv"
        snapshot.write_bytes(content)
        raw_rows = list(csv.DictReader(io.StringIO(content.decode("utf-8-sig"))))
        sources.append(dict(path=str(path.resolve()), sha256=digest, size_bytes=len(original),
                            rows=len(raw_rows), ignored_incomplete_tail_bytes=ignored,
                            parsed_snapshot=str(snapshot.resolve()), parsed_sha256=core.sha_bytes(content)))
        for number, row in enumerate(raw_rows, 2):
            name = row.get("pattern")
            if name not in catalog:
                continue
            beta, theta, size = core.canonical(row["beta"]), core.canonical(row["theta"]), int(row["L"])
            if beta not in betas:
                continue
            # The finite-time asymptotic-fit protocol excludes theta=0; preserve
            # such historical inputs in the source snapshot, not as fit points.
            if theta == 0:
                continue
            try:
                if theta < 0 or math.isnan(theta):
                    raise ValueError("Invalid imaginary time")
                meta = catalog[name]
                if size % (3*meta["m"]):
                    raise ValueError("Supercell/cut phase incompatible with L")
                if row.get("m") and int(row["m"]) != meta["m"]:
                    raise ValueError("Wrong supercell")
                if row.get("occupied_bits") and row["occupied_bits"] != meta["occupied_bits"]:
                    raise ValueError("Different occupation mask under the same name")
                for field in ("x0", "y0", "cut_x0", "cut_y0"):
                    if row.get(field) and int(row[field]) != 1:
                        raise ValueError("Different cut origin")
                values = {key: float(row[key]) for key in ("S_A", "S_C", "S_B", "D")}
                if not all(math.isfinite(value) for value in values.values()):
                    raise ValueError("Nonfinite entropy")
                if abs((2*values["S_A"]+values["S_C"])/3-values["S_B"]-values["D"]) > 1e-9:
                    raise ValueError("Weighted D identity failed")
                ground = row.get("premeasurement_ground", row.get("ground", ""))
                if ground and ground.lower() != "true":
                    raise ValueError("Not ground-state accessible")
                excess = next((row[key] for key in ("premeasurement_excess", "premeasurement_limit_excess",
                                                   "accessible_excess") if row.get(key)), None)
                if not ground and excess is None:
                    raise ValueError("Missing asymptotic reachability evidence")
                if excess is not None and (not math.isfinite(float(excess)) or abs(float(excess)) > 1e-8):
                    raise ValueError("Asymptotic energy excess failed")
                record = dict(pattern=name, m=meta["m"], family=meta["family"],
                              occupancy_class=meta["occupancy_class"], occupied_bits=meta["occupied_bits"],
                              L=size, theta=theta, beta=beta, x0=1, y0=1, **values,
                              premeasurement_ground=True,
                              premeasurement_excess=float(excess) if excess is not None else None,
                              source_file=core.relative_path(path), source_row=number, source_sha256=digest,
                              source_kind=row.get("source_kind", "existing_observation"),
                              underlying_source_file=row.get("source_path", row.get("source_file", "")),
                              underlying_source_row=row.get("source_row", ""),
                              underlying_source_sha256=row.get("source_sha256", ""),
                              run_id=row.get("run_id", ""), entropy_method=row.get("entropy_method", ""),
                              mask_evidence="input_occupied_bits" if row.get("occupied_bits") else "named_legacy_pattern",
                              phase_evidence="input_origin" if any(row.get(k) for k in ("x0", "y0", "cut_x0", "cut_y0"))
                              else "fixed_origin_legacy_protocol")
                key = name, size, theta, beta
                if key in selected:
                    old = selected[key]
                    error = max(abs(values[k]-old[k]) for k in values)
                    if error > 1e-9:
                        raise ValueError(f"Conflicting observations: max entropy difference {error}")
                    duplicates.append(dict(pattern=name, L=size, theta=theta, beta=beta,
                                           previous_file=old["source_file"], previous_row=old["source_row"],
                                           selected_file=record["source_file"], selected_row=number,
                                           max_entropy_difference=error))
                selected[key] = record
            except (ValueError, KeyError, TypeError) as error:
                # Failure is an explicit artifact and aborts analysis, never an
                # implicitly excluded pattern that would bias the class sample.
                core.dump_csv(output/"input_failure.csv", [dict(source=str(path), source_row=number,
                    pattern=name, L=size, beta=beta, theta=theta, failure=str(error))])
                raise ValueError(f"{path}:{number}: {error}") from error
    return sorted(selected.values(), key=lambda r: (r["pattern"], r["beta"], r["L"], r["theta"])), sources, duplicates


def equilibrium_fits(rows, betas, configuration):
    fits, missing = [], []
    high = configuration["Lmax"]
    for beta, low, model in itertools.product(betas, configuration["lower_cuts"],
                                              ("log_inv1", "log_inv1_inv2")):
        points = sorted((r for r in rows if r["beta"] == beta and math.isinf(r["theta"])
                         and low <= r["L"] <= high), key=lambda r: r["L"])
        x = [r["L"] for r in points]
        meta = dict(beta=beta, model=model, requested_Lmin=low, requested_Lmax=high)
        result, reason = core.fit(x, [r["D"] for r in points], model)
        if result:
            fits.append(dict(**meta, **result, extent_complete=x[0] == low and x[-1] == high))
        else:
            missing.append(dict(domain="equilibrium", **meta, n=len(x), reason=reason))
    return fits, missing


def primary_and_sensitivity(betas, primary, eq, individual, pairs, finite, eq_config, theta_min, ratio_max):
    """Adapt the immutable helper's primary-selector key, never its fit data.

    core.primary_rows uses requested_Lmin==18 solely as an equilibrium selector.
    A private metadata copy aliases the actual main lower bound to that selector.
    Every public equilibrium fit, primary window and sensitivity row retains its
    true bounds and measured grid. No x or y value, coefficient or fit is changed.
    """
    low, high = eq_config["Lmin"], eq_config["Lmax"]
    if low != 18 and any(r["requested_Lmin"] == 18 for r in eq):
        raise ValueError("Ambiguous private equilibrium selector")
    private_eq = [dict(r, requested_Lmin=18 if r["requested_Lmin"] == low else r["requested_Lmin"]) for r in eq]
    coefficients = core.primary_rows(betas, primary, private_eq, individual, pairs, finite,
                                     high, theta_min, ratio_max)
    for row in coefficients:
        fitted = next((r for r in eq if r["beta"] == row["beta"] and r["requested_Lmin"] == low
                       and r["model"] == "log_inv1" and r["extent_complete"]), None)
        row.update(eq_Lmin=low, eq_Lmax=high, eq_window=f"{low}-{high}", eq_model="log_inv1",
                   eq_grid=fitted["grid"] if fitted else None)
        if row["c_eq"] is None:
            row["eq_status"] = f"missing_complete_{low}_to_{high}_extent"
    # Time diagnostics do not depend on equilibrium. Generate these without any
    # equilibrium alias, then add explicit per-pattern equilibrium diagnostics.
    sensitivity = core.sensitivity_rows(coefficients, [], individual, pairs)
    for beta in betas:
        anchor = next((r for r in eq if r["beta"] == beta and r["requested_Lmin"] == low
                       and r["model"] == "log_inv1"), None)
        if not anchor:
            continue
        for r in eq:
            if r["beta"] == beta:
                sensitivity.append(dict(domain="equilibrium", beta=beta, model=r["model"],
                    window=f"{r['requested_Lmin']}-{high}", grid=r["grid"], c=r["c"], reference_c=anchor["c"],
                    relative_drift=core.ratio(abs(r["c"]-anchor["c"]), abs(anchor["c"])),
                    extent_complete=r["extent_complete"], reference_extent_complete=anchor["extent_complete"]))
    return coefficients, sensitivity


def run_pattern(rows, betas, windows, pattern_plan, eq_config, theta_min, ratio_max):
    args = argparse.Namespace(primary_L=None, comparison_L=None, primary_window=None, primary_model=None)
    primary = core.explicit_primary(args, pattern_plan)
    for beta in betas:
        chosen = core.primary_for_beta(primary, beta)
        if chosen and tuple(chosen["window"]) not in windows:
            raise ValueError("Primary window must be explicitly included among diagnostic windows")
    # core.analyze receives only finite observations: no hard-coded equilibrium
    # lower bound can enter this module's equilibrium fits.
    finite_rows = [r for r in rows if math.isfinite(r["theta"])]
    _, individual, pairs, missing, _, finite = core.analyze(
        finite_rows, betas, windows, theta_min, ratio_max, eq_config["Lmax"])
    missing = [r for r in missing if r["domain"] != "equilibrium"]
    eq, eq_missing = equilibrium_fits(rows, betas, eq_config)
    coefficients, sensitivity = primary_and_sensitivity(
        betas, primary, eq, individual, pairs, finite, eq_config, theta_min, ratio_max)
    return dict(primary=primary, eq=eq, individual=individual, pairs=pairs, coefficients=coefficients,
                sensitivity=sensitivity, missing=eq_missing+missing)


def needs_equilibrium_zoom(coefficients):
    """Display-only trigger; use available primary estimates, never a substitute L."""
    equilibrium = [abs(float(r["c_eq"])) for r in coefficients
                   if r.get("c_eq") is not None and math.isfinite(r["c_eq"])]
    time = [abs(float(r["c_neq"])) for r in coefficients
            if r.get("c_neq") is not None and math.isfinite(r["c_neq"])]
    return bool(equilibrium and time and max(time) > 10*max(equilibrium))


def time_fit_warning(coefficients):
    available = [r for r in coefficients if r.get("c_neq") is not None]
    if available and not any(r.get("local_diagnostic_pass") for r in available):
        return "Time: unresolved finite-window fits; full range retained"
    return "Time: finite-window fits; full range and local status retained"


def equilibrium_zoom_axis(ax, coefficients, show_bands=True):
    """A separate small panel leaves every point of the full-range plot visible."""
    beta, value = core.array(coefficients, "beta"), core.array(coefficients, "c_eq")
    ax.plot(beta, value, color=core.BLUE, lw=2.4, zorder=2)
    ax.plot(beta, value, "o", ms=6.5, mfc="white", mec=core.BLUE, mew=1.3,
            linestyle="none", zorder=3, clip_on=False)
    bound_keys = ["c_eq"]
    if show_bands:
        ax.fill_between(beta, core.array(coefficients, "eq_range_min"),
                        core.array(coefficients, "eq_range_max"), color=core.BLUE, alpha=.12, zorder=1)
        bound_keys += ["eq_range_min", "eq_range_max"]
    bounds = np.concatenate([core.array(coefficients, key) for key in bound_keys])
    bounds = bounds[np.isfinite(bounds)]
    low, high = min([0.]+bounds.tolist()), max([0.]+bounds.tolist())
    padding = .06*(high-low if high > low else .01)
    ax.set(xlim=(-.035, 2.035), ylim=(low-padding, high+padding), xlabel=r"$\beta$",
           ylabel="Equilibrium c", title="Equilibrium zoom: independent y scale")
    ax.set_xticks(np.linspace(0, 2, 5))
    ax.tick_params(labelsize=8)
    ax.title.set_fontsize(9)
    ax.yaxis.label.set_fontsize(9)
    ax.axhline(0, color=".55", lw=.6, alpha=.6, zorder=0)
    ax.grid(axis="y", alpha=.22)


def make_plots(rows, results, names, catalog, betas, directory, stage, plot_betas, theta_min, ratio_max):
    directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.dpi": 180, "pdf.fonttype": 42})
    figures = []
    for name in names:
        result, meta = results[name], catalog[name]
        coefficients, eq_config = result["coefficients"], meta["equilibrium"]
        configured = core.primary_for_beta(result["primary"], betas[0])
        selection = (f"eq L{eq_config['Lmin']}..{eq_config['Lmax']}; "
                     f"time L{configured['L']}, theta {configured['window'][0]:g}..{configured['window'][1]:g}") if configured else "Primary not configured"
        eq_n, time_n = (sum(r.get(k) is not None for r in coefficients) for k in ("c_eq", "c_neq"))
        zoom = needs_equilibrium_zoom(coefficients)
        title = f"{name}: {selection}\n{stage}; eq {eq_n}/{len(betas)}, time {time_n}/{len(betas)} available"
        note = "Finite-window D coefficients; filled/open time squares: local checks pass/unresolved.\nBands: model/window ranges, not confidence intervals."
        if zoom:
            fig, (ax, zoom_ax) = plt.subplots(2, 1, figsize=(7, 7), sharex=True,
                                            gridspec_kw={"height_ratios": [2, 1]})
            fig.subplots_adjust(left=.12, right=.98, bottom=.15, top=.82, hspace=.42)
            previous.coefficient_axis(ax, coefficients, title + "\n" + time_fit_warning(coefficients))
            ax.set_xlabel("")
            ax.set_ylabel("Fitted c (full range)")
            equilibrium_zoom_axis(zoom_ax, coefficients)
            fig.text(.5, .025, note, ha="center", va="bottom", fontsize=8)
        else:
            fig, ax = plt.subplots(figsize=(7, 4.6), constrained_layout=True)
            previous.coefficient_axis(ax, coefficients, title)
            ax.text(.5, -.22, note, transform=ax.transAxes, ha="center", va="top", fontsize=8)
        stem = f"{name}_corner_coefficient_beta"
        core.save(fig, directory, stem)
        figures.append(stem)

        fig, axes = plt.subplots(1, 3, figsize=(16, 4.7), constrained_layout=True)
        previous.coefficient_axis(axes[0], coefficients, "Finite-window D estimates", detailed=True)
        passed = [r for r in coefficients if r.get("local_diagnostic_pass")]
        for ax, key, factor, title in ((axes[1], "ratio_neq_to_eq", 1, "Nonequilibrium / equilibrium"),
                                      (axes[2], "relative_gap", 100, "Relative difference (%)")):
            ax.plot(core.array(coefficients, "beta"), factor*core.array(coefficients, key),
                    "s-", color=core.ORANGE, mfc="white", zorder=3)
            ax.plot(core.array(passed, "beta"), factor*core.array(passed, key), "s", color=core.ORANGE, zorder=4)
            ax.axhline(1 if factor == 1 else 0, color=core.BLUE, lw=1, zorder=1)
            ax.set(xlabel=r"$\beta$", ylabel=title, title=title, xlim=(-.035, 2.035))
            ax.set_xticks(np.linspace(0, 2, 9))
            ax.grid(axis="y", alpha=.25)
        fig.suptitle(f"{name}: {stage}; {selection}\nComparison never enters time gates; bands are sensitivity ranges, not confidence intervals", fontsize=11)
        stem = f"{name}_coefficient_diagnostics"
        core.save(fig, directory, stem)
        figures.append(stem)

        for beta in plot_betas:
            if beta not in betas:
                continue
            block = [r for r in rows if r["pattern"] == name and r["beta"] == beta]
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), constrained_layout=True)
            equilibrium = sorted((r for r in block if math.isinf(r["theta"]) and
                                   eq_config["Lmin"] <= r["L"] <= eq_config["Lmax"]), key=lambda r: r["L"])
            if equilibrium:
                axes[0].plot(np.log([r["L"] for r in equilibrium]), [r["D"] for r in equilibrium],
                             "o", color=core.BLUE, label="Equilibrium data")
            fitted = next((r for r in result["eq"] if r["beta"] == beta and
                           r["requested_Lmin"] == eq_config["Lmin"] and r["model"] == "log_inv1"), None)
            if fitted:
                x = np.geomspace(fitted["x_min"], fitted["x_max"], 200)
                axes[0].plot(np.log(x), core.predict(fitted, x), color=core.BLUE,
                             label=f"Fit: L={fitted['x_min']:g}..{fitted['x_max']:g}")
            coefficient = next(r for r in coefficients if r["beta"] == beta)
            configured = core.primary_for_beta(result["primary"], beta)
            finite = [r for r in block if math.isfinite(r["theta"])]
            allowed = set()
            if configured:
                allowed.add(configured["L"])
                comparison = coefficient.get("comparison_L") or configured.get("comparison_L")
                if comparison is not None:
                    allowed.add(comparison)
                if coefficient.get("third_L"):
                    allowed.add(coefficient["third_L"])
                else:
                    larger = sorted({r["L"] for r in finite if r["L"] > configured["L"] and
                                     r["theta"] >= theta_min and r["theta"]/r["L"] <= ratio_max})
                    if larger:
                        allowed.add(larger[0])
            finite = [r for r in finite if r["L"] in allowed and r["theta"] >= theta_min
                      and r["theta"]/r["L"] <= ratio_max]
            for size in sorted({r["L"] for r in finite}):
                index = sorted(allowed).index(size) % 3
                color = ("#5b3b8c", "#148f8b", "#657a20")[index]
                points = sorted((r for r in finite if r["L"] == size), key=lambda r: r["theta"])
                axes[1].plot(np.log([r["theta"] for r in points]), [r["D"] for r in points],
                             marker=("o", "s", "^")[index], linestyle=("-", "--", ":")[index],
                             color=color, ms=(7.2, 5.2, 3.2)[index], mfc="white" if index < 2 else color,
                             mew=1, lw=.8, zorder=4+index, label=f"L={size}")
            if configured and coefficient.get("c_neq") is not None:
                fitted = next(r for r in result["pairs"] if r["beta"] == beta and
                              r["L_large"] == configured["L"] and r["L_small"] == coefficient["comparison_L"]
                              and r["window"] == coefficient["primary_window"] and r["model"] == configured["model"])
                x = np.geomspace(fitted["theta_min"], fitted["theta_max"], 200)
                label = "Primary fit: local checks" if coefficient.get("local_diagnostic_pass") else "Primary fit: unresolved"
                axes[1].plot(np.log(x), core.predict(fitted, x), color=core.ORANGE, lw=2, zorder=3, label=label)
            else:
                axes[1].text(.98, .05, "Primary time fit not yet available", transform=axes[1].transAxes,
                             ha="right", va="bottom", fontsize=8)
            if not finite:
                axes[1].set_xlim(np.log(configured.get("window", [4, 7]) if configured else [4, 7]))
                axes[1].set_yticks([])
            axes[0].set(xlabel=r"$\ln L$", ylabel="D", title="Equilibrium scaling")
            axes[1].set(xlabel=r"$\ln\theta$", ylabel="D", title=f"Time data: theta >= {theta_min:g}, theta/L <= {ratio_max:g}")
            for ax in axes:
                ax.set_xscale("linear")
                ax.grid(axis="y", alpha=.25)
                if ax.get_legend_handles_labels()[0]:
                    ax.legend(fontsize=7, ncol=2)
            fig.suptitle(f"{name}, beta={beta:g}: {stage}; fits retain inverse-scale corrections", fontsize=11)
            stem = f"{name}_scaling_beta_{beta:g}".replace(".", "p")
            core.save(fig, directory, stem)
            figures.append(stem)

    zoom_any = any(needs_equilibrium_zoom(results[name]["coefficients"]) for name in names)
    fig = plt.figure(figsize=(12.8, 16 if zoom_any else 13))
    fig.subplots_adjust(left=.07, right=.98, bottom=.06, top=.89, wspace=.25, hspace=.40)
    outer = fig.add_gridspec(3, 2)
    for category in (1, 2, 3):
        members = [name for name in names if catalog[name]["occupancy_class"] == category]
        for index in range(2):
            slot = outer[category-1, index]
            if index < len(members):
                name = members[index]
                coefficients = results[name]["coefficients"]
                zoom = needs_equilibrium_zoom(coefficients)
                if zoom:
                    inner = slot.subgridspec(2, 1, height_ratios=[2, 1], hspace=.44)
                    ax = fig.add_subplot(inner[0])
                    zoom_ax = fig.add_subplot(inner[1], sharex=ax)
                else:
                    ax = fig.add_subplot(slot)
                eq_n, time_n = (sum(r.get(k) is not None for r in coefficients) for k in ("c_eq", "c_neq"))
                title = f"Class {category}: {name}\neq {eq_n}/{len(betas)}, time {time_n}/{len(betas)} available"
                if zoom:
                    title += "\n" + time_fit_warning(coefficients)
                previous.coefficient_axis(ax, coefficients,
                                          title, show_bands=False)
                if zoom:
                    ax.title.set_fontsize(9)
                    ax.set_xlabel("")
                    ax.set_ylabel("Fitted c (full range)")
                    ax.tick_params(labelbottom=False)
                    equilibrium_zoom_axis(zoom_ax, coefficients, show_bands=False)
                values = [float(r[k]) for r in coefficients
                          for k in ("c_eq", "c_neq") if r.get(k) is not None and math.isfinite(r[k])]
                low, high = min([0.]+values), max([0.]+values)
                padding = .05*(high-low if high > low else .01)
                ax.set_ylim(low-padding, high+padding)
                if not values:
                    ax.text(.5, .45, "No complete primary fits yet", transform=ax.transAxes,
                            ha="center", color=".35")
                    ax.set_yticks([])
            else:
                ax = fig.add_subplot(slot)
                ax.text(.5, .5, f"Class {category}: pattern pending", transform=ax.transAxes, ha="center")
                ax.set_axis_off()
    fig.suptitle(f"Finite-window D coefficients ({stage}, single species)\n"
                 "Rows: 1 = single cells; 2 = empty/double only; 3 = mixed occupancies\n"
                 "Independent y ranges; negative estimates retained; sensitivity ranges in individual diagnostics\n"
                 "Filled/open time squares = local checks pass/unresolved",
                 fontsize=11)
    core.save(fig, directory, "corner_coefficients_by_class")
    figures.append("corner_coefficients_by_class")
    return figures


def load_mpmath():
    try:
        import mpmath as mp
    except ImportError:
        # Reuse the already installed audit package without modifying it.
        dependency = HERE/"analysis_beta_patterns"/"audit_dependencies"
        if dependency.exists():
            sys.path.insert(0, str(dependency))
        try:
            import mpmath as mp
        except ImportError as error:
            raise SystemExit("Self-check requires mpmath, also available from the existing beta-pattern audit dependency directory.") from error
    return mp


def validate_current_fits(output):
    """Rebuild numerical fits with 80-digit QR, without calling any fit helper.

    This checks the published merged observations, every D fit including both
    members of each size pair, raw-entropy fits, and the actual primary anchors.
    It does not claim to reimplement source merging or all diagnostic gates.
    """
    mp = load_mpmath()
    mp.mp.dps = 80
    names = ["observations", "equilibrium_fits", "time_individual_fits", "time_pair_fits",
             "raw_entropy_fits", "primary_coefficients"]
    contents = {name: (output/f"{name}.csv").read_bytes() for name in names}
    summary_bytes = (output/"summary.json").read_bytes()
    summary = json.loads(summary_bytes.decode("utf-8"))
    if summary["script_sha256"] != core.sha_bytes(Path(__file__).read_bytes()):
        raise ValueError("Regenerate analysis after changing the script before validating its tables")
    tables = {name: list(csv.DictReader(io.StringIO(value.decode("utf-8-sig"))))
              for name, value in contents.items()}
    points = {(r["pattern"], float(r["beta"]), int(r["L"]), float(r["theta"])): r
              for r in tables["observations"]}
    models = {"log_inv1": [1], "log_inv1_inv2": [1, 2], "log_inv1_inv2_inv3": [1, 2, 3]}
    maximum_c, maximum_parameter, count = 0., 0., 0

    def check(row, domain, size=None, raw=False, small=False):
        nonlocal maximum_c, maximum_parameter, count
        grid = [float(x) for x in row["grid"].split(",")]
        if raw:
            order = {"area_log_inv1": 1, "area_log_inv2": 2}[row["model"]]
            powers = list(range(1, order+1))
            keys = ["a", "log_coefficient", "b"]+[f"inv{p}" for p in powers]
        else:
            powers = models[row["model"]]
            keys = ["c", "b"]+[f"inv{p}" for p in powers]
        assert len(grid) >= len(keys)+2
        x = [mp.mpf(str(t)) for t in grid]
        matrix = mp.matrix([([t, mp.log(t), mp.mpf(1)] if raw else [mp.log(t), mp.mpf(1)])
                            +[t**(-p) for p in powers] for t in x])
        values = []
        for t in grid:
            L, theta = (int(t), math.inf) if domain == "equilibrium" else (int(size), t)
            record = points[row["pattern"], float(row["beta"]), L, theta]
            values.append(mp.mpf(record[row["region"] if raw else "D"]))
        fitted, _ = mp.qr_solve(matrix, mp.matrix(values))
        targets = [("c_small", fitted[0])] if small else list(zip(keys, fitted))
        for key, value in targets:
            error = abs(float(value)-float(row[key]))
            maximum_parameter = max(maximum_parameter, error)
            if key in ("c", "c_small", "log_coefficient"):
                maximum_c = max(maximum_c, error)
                assert error <= 1e-8+abs(float(value))*1e-8, (domain, key, error)
            assert error <= 1e-6+abs(float(value))*1e-8, (domain, key, error)
        count += 1

    for row in tables["equilibrium_fits"]:
        check(row, "equilibrium")
    for row in tables["time_individual_fits"]:
        check(row, "time", row["L"])
    for row in tables["time_pair_fits"]:
        check(row, "time", row["L_large"])
        check(row, "time", row["L_small"], small=True)
    for row in tables["raw_entropy_fits"]:
        check(row, "equilibrium", raw=True)
    for row in tables["primary_coefficients"]:
        if row.get("c_eq"):
            matches = [r for r in tables["equilibrium_fits"] if
                       (r["pattern"], r["beta"], r["requested_Lmin"], r["model"]) ==
                       (row["pattern"], row["beta"], row["eq_Lmin"], "log_inv1")]
            assert len(matches) == 1 and float(matches[0]["c"]) == float(row["c_eq"])
        if row.get("c_neq"):
            matches = [r for r in tables["time_pair_fits"] if
                       (r["pattern"], r["beta"], r["L_small"], r["L_large"], r["window"], r["model"]) ==
                       (row["pattern"], row["beta"], row["comparison_L"], row["primary_L"],
                        row["primary_window"], row["primary_model"])]
            assert len(matches) == 1 and float(matches[0]["c"]) == float(row["c_neq"])
    assert all((output/f"{name}.csv").read_bytes() == value for name, value in contents.items())
    assert (output/"summary.json").read_bytes() == summary_bytes
    receipt = dict(status="PASS", stage=summary["stage"], generated_utc=datetime.now(timezone.utc).isoformat(),
                   precision_digits=80, method="Independent mpmath rectangular QR from merged observation CSV numeric strings; no fit helper called",
                   fit_count_including_small_pairs=count, max_c_or_raw_logcoefficient_error=maximum_c,
                   max_parameter_error=maximum_parameter, primary_anchor_rows=len(tables["primary_coefficients"]),
                   table_sha256={name: core.sha_bytes(value) for name, value in contents.items()},
                   summary_sha256=core.sha_bytes(summary_bytes), script_sha256=summary["script_sha256"],
                   scope="Numerical fits and true per-pattern equilibrium anchors; does not independently reimplement every local diagnostic gate")
    (output/"independent_fit_validation.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))


def selfcheck(output):
    mp = load_mpmath()
    mp.mp.dps = 80
    metadata = {
        "synthetic_special": dict(m=2, occupied_bits="10101001"),
        "synthetic_mixed": dict(m=3, occupied_bits="011010111001011000"),
        "synthetic_pure02": dict(m=4, occupied_bits="11"*8+"00"*8),
    }
    names, catalog = catalog_from_plan(dict(patterns=list(metadata), pattern_metadata=metadata))
    max_error, fit_count = 0., 0
    for x in ([4+i/2 for i in range(7)], list(range(24, 145, 12))):
        y = [.173*math.log(t)+.31+.07/t-.028/t**2+.009/t**3 for t in x]
        for model, powers in core.MODELS.items():
            result, reason = core.fit(x, y, model)
            assert result and not reason
            xx = [mp.mpf(str(t)) for t in x]
            matrix = mp.matrix([[mp.log(t), 1]+[t**(-p) for p in powers] for t in xx])
            expected, _ = mp.qr_solve(matrix, mp.matrix([mp.mpf(str(t)) for t in y]))
            actual = [result["c"], result["b"]]+[result[f"inv{p}"] for p in powers]
            error = max(abs(float(a)-float(b)) for a, b in zip(actual, expected))
            max_error, fit_count = max(max_error, error), fit_count+1
            assert error < 2e-8, (model, error)
    fixture = []
    for name in names:
        m = catalog[name]["m"]
        eq = catalog[name]["equilibrium"]
        target = .1*catalog[name]["occupancy_class"]
        for size in range(eq["Lmin"], eq["Lmax"]+1, 3*m):
            d = target*math.log(size)+.4+.05/size
            fixture.append(dict(pattern=name, m=m, occupied_bits=catalog[name]["occupied_bits"], L=size,
                                theta="Inf", beta=1, S_A=11+d, S_C=12+2*d,
                                S_B=(2*(11+d)+(12+2*d))/3-d, D=d, premeasurement_ground=True))
        for size, theta in itertools.product([144, 180], [4+i/2 for i in range(9)]):
            d = target*math.log(theta)+.4+.05/theta
            fixture.append(dict(pattern=name, m=m, occupied_bits=catalog[name]["occupied_bits"], L=size,
                                theta=theta, beta=1, S_A=11+d, S_C=12+2*d,
                                S_B=(2*(11+d)+(12+2*d))/3-d, D=d, premeasurement_ground=True))
    with tempfile.TemporaryDirectory(prefix="occupancy_class_check_") as temporary:
        root = Path(temporary)
        source = root/"synthetic.csv"
        core.dump_csv(source, fixture)
        rows, _, duplicates = read_sources([source, source], "final", root, catalog, {1.})
        assert len(rows) == len(fixture) and len(duplicates) == len(fixture)
        for name in names:
            eq = catalog[name]["equilibrium"]
            result = run_pattern([r for r in rows if r["pattern"] == name], [1.],
                                 [tuple(w) for w in DEFAULT_WINDOWS],
                                 dict(primary=dict(L=180, comparison_L=144, window=[4, 7], model="log_inv1_inv2")),
                                 eq, 4, .05)
            coefficient = result["coefficients"][0]
            assert abs(coefficient["c_eq"]-.1*catalog[name]["occupancy_class"]) < 1e-10
            assert abs(coefficient["c_neq"]-.1*catalog[name]["occupancy_class"]) < 1e-9
            assert coefficient["eq_Lmin"] == eq["Lmin"]
            assert all(int(float(r["window"].split("-")[0])) in eq["lower_cuts"]
                       for r in result["sensitivity"] if r["domain"] == "equilibrium")
        blocked = [dict(fixture[0], premeasurement_ground=False)]
        bad = root/"blocked.csv"
        core.dump_csv(bad, blocked)
        try:
            read_sources([bad], "final", root, catalog, {1.})
        except ValueError as error:
            assert "Not ground-state accessible" in str(error)
        else:
            raise AssertionError("Blocked initial state was accepted")
    output.mkdir(parents=True, exist_ok=True)
    receipt = dict(status="PASS", generated_utc=datetime.now(timezone.utc).isoformat(),
                   script_sha256=core.sha_bytes(Path(__file__).read_bytes()), independent_precision_digits=80,
                   independent_solver="mpmath QR on reconstructed raw synthetic rows",
                   independent_dependency=dict(name="mpmath", version=mp.__version__, path=str(Path(mp.__file__).resolve())),
                   independent_fit_count=fit_count, max_parameter_error=max_error,
                   checks=["all three occupancy classes computed from masks", "half-filling metadata",
                           "80-digit independent fit comparison", "pattern key isolation", "duplicate provenance",
                           "SA differs from SC", "m4 actual 24..144 equilibrium bounds and sensitivity rows",
                           "blocked state aborts with visible failure record"])
    (output/"framework_validation.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=HERE/"analysis_occupancy_classes"/"plan.json")
    parser.add_argument("--inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=HERE/"analysis_occupancy_classes")
    parser.add_argument("--figures", type=Path, default=HERE/"figures_occupancy_classes")
    parser.add_argument("--stage", choices=("preliminary", "final"), default="preliminary")
    parser.add_argument("--plot-betas", nargs="+", type=float, default=[0, .5, 1, 1.5, 2])
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--validate-fits", action="store_true", help="Independently check existing output tables with 80-digit QR")
    args = parser.parse_args()
    if args.self_check:
        selfcheck(args.output)
        return
    if args.validate_fits:
        validate_current_fits(args.output)
        return
    plan_bytes = args.plan.read_bytes()
    plan = json.loads(plan_bytes.decode("utf-8-sig"))
    names, catalog = catalog_from_plan(plan)
    if args.stage == "final" and Counter(catalog[n]["occupancy_class"] for n in names) != Counter({1: 2, 2: 2, 3: 2}):
        raise ValueError("Final requires exactly two actual patterns in each of the three occupancy classes")
    if any(n > 2 for n in Counter(catalog[name]["occupancy_class"] for name in names).values()):
        raise ValueError("The six-panel protocol accepts at most two patterns per class")
    dependencies = {Path(module.__file__).resolve(): Path(module.__file__).read_bytes() for module in (core, previous)}
    betas = sorted({core.canonical(x) for x in plan.get("betas", [i/4 for i in range(9)])})
    windows = [tuple(map(float, x)) for x in plan.get("windows", DEFAULT_WINDOWS)]
    theta_min, ratio_max = float(plan.get("theta_min_screen", 4)), float(plan.get("theta_over_L_max_screen", .05))
    paths = args.inputs or [Path(p) if Path(p).is_absolute() else HERE/p for p in plan.get("inputs", [])] or DEFAULT_INPUTS
    args.output.mkdir(parents=True, exist_ok=True)
    rows, sources, duplicates = read_sources(paths, args.stage, args.output, catalog, set(betas))
    tables, results, expected = defaultdict(list), {}, 0
    tables["observations"], tables["duplicate_observations"] = rows, duplicates
    for name in names:
        block = [r for r in rows if r["pattern"] == name]
        pattern_plan = previous.per_pattern_plan(plan, name, names)
        result = run_pattern(block, betas, windows, pattern_plan, catalog[name]["equilibrium"], theta_min, ratio_max)
        if args.stage == "final" and (not result["primary"] or not pattern_plan.get("required_finite_grids")
                                      or not pattern_plan.get("required_equilibrium_grids")):
            raise ValueError(f"{name}: final requires explicit primary and both required data grids")
        count, missing = core.coverage_check(block, pattern_plan, betas)
        expected += count
        result.update(required_keys=count, missing_required_keys=len(missing))
        tables["missing_required_keys"].extend(tag(name, catalog, missing))
        for output, field in (("equilibrium_fits", "eq"), ("time_individual_fits", "individual"),
                              ("time_pair_fits", "pairs"), ("missing_fits", "missing"),
                              ("primary_coefficients", "coefficients"), ("time_sensitivity", "sensitivity")):
            tables[output].extend(tag(name, catalog, result[field]))
        for beta, low, order, region in itertools.product(
                betas, catalog[name]["equilibrium"]["lower_cuts"], (1, 2), ("S_A", "S_C", "S_B")):
            fitted = previous.entropy_fit([r for r in block if r["beta"] == beta], region, low,
                                          catalog[name]["equilibrium"]["Lmax"], order)
            if fitted:
                tables["raw_entropy_fits"].extend(tag(name, catalog, [dict(beta=beta, **fitted)]))
        results[name] = result
    complete = all(r.get("c_eq") is not None and r.get("c_neq") is not None for r in tables["primary_coefficients"])
    if args.stage == "final" and (tables["missing_required_keys"] or not complete):
        raise ValueError("Final requires all requested data and primary estimates, not local diagnostic passes")
    grouped = defaultdict(list)
    for r in rows:
        grouped[r["pattern"], r["beta"], r["L"]].append(r)
    for (name, beta, size), group in sorted(grouped.items()):
        times = sorted(r["theta"] for r in group if math.isfinite(r["theta"]))
        tables["coverage"].append(dict(pattern=name, occupancy_class=catalog[name]["occupancy_class"],
            beta=beta, L=size, equilibrium_present=any(math.isinf(r["theta"]) for r in group),
            finite_n=len(times), times=core.grid_string(times), theta_over_L_max=max(times)/size if times else None))
    for title, values in tables.items():
        core.dump_csv(args.output/f"{title}.csv", values)
    figures = [] if args.no_plots else make_plots(rows, results, names, catalog, betas, args.figures,
                                                  args.stage, args.plot_betas, theta_min, ratio_max)
    changed = [s["path"] for s in sources if not s.get("missing") and
               core.sha_bytes(Path(s["path"]).read_bytes()) != s["sha256"]]
    dependency_changed = any(p.read_bytes() != content for p, content in dependencies.items())
    plan_changed = args.plan.read_bytes() != plan_bytes
    if args.stage == "final" and (changed or dependency_changed or plan_changed):
        raise ValueError("Source, plan or dependency changed during final analysis")
    summary = dict(generated_utc=datetime.now(timezone.utc).isoformat(), stage=args.stage,
        script_sha256=core.sha_bytes(Path(__file__).read_bytes()), plan_sha256=core.sha_bytes(plan_bytes), plan=plan,
        dependencies=[dict(path=str(p), sha256=core.sha_bytes(content)) for p, content in dependencies.items()],
        sources=sources, sources_changed_during_analysis=changed, dependency_changed_during_analysis=dependency_changed,
        plan_changed_during_analysis=plan_changed, patterns=catalog, betas=betas, windows=windows,
        primary={name: results[name]["primary"] for name in names}, rows=len(rows), duplicates=len(duplicates),
        required_keys=expected, required_grid_complete=bool(expected) and not tables["missing_required_keys"],
        missing_required_keys=len(tables["missing_required_keys"]), primary_complete=complete,
        primary_rows=len(tables["primary_coefficients"]), figures=figures,
        arguments={k: str(v) if isinstance(v, Path) else [str(x) for x in v] if k == "inputs" and v else v
                   for k, v in vars(args).items()},
        fit_counts={k: len(tables[k]) for k in ("equilibrium_fits", "time_individual_fits",
                                               "time_pair_fits", "raw_entropy_fits")},
        by_pattern={name: dict(required_keys=results[name]["required_keys"],
            missing_required_keys=results[name]["missing_required_keys"],
            available_eq=sum(r.get("c_eq") is not None for r in results[name]["coefficients"]),
            available_time=sum(r.get("c_neq") is not None for r in results[name]["coefficients"]),
            local_pass_beta=[r["beta"] for r in results[name]["coefficients"] if r.get("local_diagnostic_pass")])
            for name in names},
        conventions=dict(entropy="single-species natural-log Renyi-2; no SA=SC assumption",
            observable="D=(2SA+SC)/3-SB; fixed masks and cut origin; no phase averaging",
            equilibrium="per-pattern actual Lmin/Lmax/lower cuts; log+constant+inverse size; inverse-square sensitivity",
            equilibrium_selector_adapter="private metadata alias only for immutable primary helper; no fit/grid/value changes; all output bounds actual",
            time_fit="common measured grids, both endpoints required, at least parameter_count+2 points",
            local_gates="size<=2%, D_difference/log_span<=1%, model/window<=5%, same sign, full model set and >=2 windows; third size if present <=2% and decreasing",
            scope="finite-window estimates, not certified asymptotic angle coefficients; no selection by eq agreement",
            scale_screen="theta minimum and theta/L maximum apply to both primary and comparison; operational only",
            sensitivity="available model/window ranges, not confidence intervals",
            axes="explicit ln L and ln theta on linear axes; beta linear; overview class rows with independent y ranges and negative estimates retained"))
    (args.output/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("stage", "rows", "fit_counts", "required_keys",
                                             "missing_required_keys", "primary_complete", "by_pattern")}, indent=2))


if __name__ == "__main__":
    main()
