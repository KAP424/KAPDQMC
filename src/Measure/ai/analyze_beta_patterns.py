"""Independent-output analysis of three nonuniform ground-accessible patterns.

The numerical least-squares and local-diagnostic routines are imported read-only
from the audited uniform-A analysis. They are applied separately to each pattern;
neither uniform-A data nor its entropy symmetry assumptions are used here.
"""
from __future__ import annotations

import argparse
import csv
import io
import itertools
import json
import math
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import analyze_uniformA_dense as core
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
PATTERNS = {
    "one_B_2": {"m": 2, "family": "special", "occupied_bits": "10101001"},
    "diagonal_B_3": {"m": 3, "family": "special", "occupied_bits": "011010100110101001"},
    "random_half_3": {"m": 3, "family": "general", "occupied_bits": "100110001011010110"},
}
DEFAULT_INPUTS = [HERE / "data" / name for name in (
    "ground_observations_baseline.csv", "ref_calibration_observations.csv",
    "ref_calibration_equilibrium.csv", "ref_calibration_late.csv",
    "random_half3_beta1_observations.csv", "beta_patterns_observations.csv")]
DEFAULT_WINDOWS = [(4., 7.), (4.5, 7.), (4., 6.5), (5., 7.), (5., 8.), (6., 9.)]


def framework_selfcheck(output):
    """Synthetic ingestion/isolation checks, not physical entropy observations."""
    checks = []
    with tempfile.TemporaryDirectory(prefix="beta_patterns_framework_") as directory:
        temporary = Path(directory)
        fixture = []
        targets = {name: .1*(index+1) for index, name in enumerate(PATTERNS)}
        for name, size, theta in itertools.product(PATTERNS, (144, 180), np.arange(4., 7.01, .5)):
            value = targets[name]*math.log(theta) + .37 + .04/theta - .02/theta**2
            sa, sc = 10.+value, 11.+2*value
            fixture.append(dict(pattern=name, **PATTERNS[name], L=size, beta=1., theta=theta,
                                x0=1, y0=1, S_A=sa, S_C=sc, S_B=(2*sa+sc)/3-value, D=value,
                                premeasurement_ground=True, premeasurement_excess=0.))
        source = temporary/"synthetic.csv"
        core.dump_csv(source, fixture)
        observations, _, _ = read_sources([source], "final", temporary, set(PATTERNS), {1.})
        assert len(observations) == 42 and all(row["S_A"] != row["S_C"] for row in observations)
        checks.append("distinct patterns at identical L/theta/beta remain separate; SA != SC accepted")
        errors = []
        for name in PATTERNS:
            block = [row for row in observations if row["pattern"] == name]
            _, _, pairs, _, _, _ = core.analyze(block, [1.], [(4., 7.)], 4., .05, 144)
            fitted = next(row for row in pairs if row["model"] == "log_inv1_inv2")
            errors.append(abs(fitted["c"]-targets[name]))
        assert max(errors) < 1e-10
        checks.append("separate pattern fits recover different known coefficients on the same grid")
        duplicates, _, ledger = read_sources([source, source], "final", temporary, set(PATTERNS), {1.})
        assert len(duplicates) == 42 and len(ledger) == 42
        checks.append("identical duplicate source rows retained once with complete ledger")
        for label, changes in (("mask mismatch", {"occupied_bits": "01"*4}),
                               ("cut phase mismatch", {"x0": 2}),
                               ("ground reachability failure", {"premeasurement_ground": False}),
                               ("conflicting entropy duplicate", {"S_A": fixture[0]["S_A"]+.01, "S_C": fixture[0]["S_C"]+.01, "S_B": fixture[0]["S_B"]+.01})):
            changed = [dict(fixture[0], **changes)]
            bad = temporary/"invalid.csv"
            core.dump_csv(bad, changed)
            try:
                read_sources([source, bad], "final", temporary, set(PATTERNS), {1.})
            except ValueError:
                checks.append(f"{label} rejected")
            else:
                raise AssertionError(f"Failed to reject {label}")
    output.mkdir(parents=True, exist_ok=True)
    receipt = dict(status="PASS", synthetic_only=True, checks=checks, max_synthetic_c_error=max(errors),
                   generated_utc=datetime.now(timezone.utc).isoformat(), script_sha256=core.sha_bytes(Path(__file__).read_bytes()),
                   dependency_sha256=core.sha_bytes(Path(core.__file__).read_bytes()))
    (output/"framework_validation.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))


def read_sources(paths, stage, output, patterns, betas):
    selected, sources, duplicates = {}, [], []
    snapshots = output / "source_snapshots"
    snapshots.mkdir(parents=True, exist_ok=True)
    for path in paths:
        if not path.exists():
            sources.append(dict(path=str(path.resolve()), missing=True))
            continue
        original = path.read_bytes()
        digest = core.sha_bytes(original)
        content, ignored = original, 0
        if content and not content.endswith(b"\n"):
            if stage == "final":
                raise ValueError(f"Incomplete CSV tail: {path}")
            end = content.rfind(b"\n") + 1
            ignored, content = len(content) - end, content[:end]
        snapshot = snapshots / f"{path.stem}_{digest[:12]}.csv"
        snapshot.write_bytes(content)
        rows = list(csv.DictReader(io.StringIO(content.decode("utf-8-sig"))))
        sources.append(dict(path=str(path.resolve()), sha256=digest, size_bytes=len(original), rows=len(rows),
                            ignored_incomplete_tail_bytes=ignored, parsed_snapshot=str(snapshot.resolve()),
                            parsed_sha256=core.sha_bytes(content)))
        for number, row in enumerate(rows, 2):
            name = row.get("pattern")
            if name not in patterns:
                continue
            beta, theta, size = core.canonical(row["beta"]), core.canonical(row["theta"]), int(row["L"])
            if beta not in betas or theta <= 0 or math.isnan(theta):
                continue
            meta = PATTERNS[name]
            if size % (3 * meta["m"]):
                raise ValueError(f"Supercell/cut phase incompatible with L: {path}:{number}")
            if row.get("m") and int(row["m"]) != meta["m"]:
                raise ValueError(f"Wrong supercell: {path}:{number}")
            if row.get("occupied_bits") and row["occupied_bits"] != meta["occupied_bits"]:
                raise ValueError(f"Different occupation mask under the same name: {path}:{number}")
            for key in ("x0", "y0", "cut_x0", "cut_y0"):
                if row.get(key) and int(row[key]) != 1:
                    raise ValueError(f"Different cut origin: {path}:{number}")
            values = {key: float(row[key]) for key in ("S_A", "S_C", "S_B", "D")}
            if not all(math.isfinite(value) for value in values.values()):
                raise ValueError(f"Nonfinite entropy: {path}:{number}")
            if abs((2 * values["S_A"] + values["S_C"]) / 3 - values["S_B"] - values["D"]) > 1e-9:
                raise ValueError(f"Weighted entropy identity failed: {path}:{number}")
            ground = row.get("premeasurement_ground", row.get("ground", ""))
            if ground and ground.lower() != "true":
                raise ValueError(f"Not ground-state accessible: {path}:{number}")
            excess = next((row[key] for key in ("premeasurement_excess", "premeasurement_limit_excess", "accessible_excess")
                           if row.get(key)), None)
            if not ground and excess is None:
                raise ValueError(f"Missing asymptotic reachability evidence: {path}:{number}")
            if excess is not None and abs(float(excess)) > 1e-8:
                raise ValueError(f"Asymptotic energy excess failed: {path}:{number}")
            key = name, size, theta, beta
            record = dict(pattern=name, **meta, L=size, theta=theta, beta=beta, x0=1, y0=1, **values,
                          premeasurement_excess=float(excess) if excess is not None else None,
                          source_file=core.relative_path(path), source_row=number, source_sha256=digest,
                          source_kind=row.get("source_kind", "existing_observation"),
                          underlying_source_file=row.get("source_path", row.get("source_file", "")),
                          underlying_source_row=row.get("source_row", ""),
                          underlying_source_sha256=row.get("source_sha256", ""), run_id=row.get("run_id", ""),
                          entropy_method=row.get("entropy_method", ""),
                          mask_evidence="input_occupied_bits" if row.get("occupied_bits") else "named_legacy_pattern",
                          phase_evidence="input_origin" if any(row.get(k) for k in ("x0", "y0", "cut_x0", "cut_y0")) else "fixed_origin_legacy_protocol")
            if key in selected:
                previous = selected[key]
                error = max(abs(values[k] - previous[k]) for k in values)
                if error > 1e-9:
                    raise ValueError(f"Conflicting observations {key}: {error}")
                duplicates.append(dict(pattern=name, L=size, theta=theta, beta=beta,
                                       previous_file=previous["source_file"], previous_row=previous["source_row"],
                                       selected_file=record["source_file"], selected_row=number, max_entropy_difference=error))
            selected[key] = record
    return sorted(selected.values(), key=lambda r: (r["pattern"], r["beta"], r["L"], r["theta"])), sources, duplicates


def per_pattern_plan(plan, name, patterns):
    result = dict(plan)
    result["primary"] = {**plan.get("primary", {}), **plan.get("primary_by_pattern", {}).get(name, {})}
    result["primary_by_beta"] = plan.get("primary_by_pattern_beta", {}).get(name, {})
    for domain in ("required_finite_grids", "required_equilibrium_grids"):
        result[domain] = [grid for grid in plan.get(domain, []) if name in grid.get("patterns", patterns)]
    return result


def tagged(name, rows):
    return [dict(pattern=name, m=PATTERNS[name]["m"], family=PATTERNS[name]["family"], **row) for row in rows]


def entropy_fit(rows, region, low, high, inverse_order):
    points = sorted((row for row in rows if math.isinf(row["theta"]) and low <= row["L"] <= high), key=lambda r: r["L"])
    nparam = 3 + inverse_order
    if len(points) < nparam + 2:
        return None
    x = np.asarray([row["L"] for row in points], dtype=float)
    y = np.asarray([row[region] for row in points])
    design = np.column_stack([x, np.log(x), np.ones(len(x))] + [x ** (-p) for p in range(1, inverse_order + 1)])
    scale = np.linalg.norm(design, axis=0)
    fitted, _, rank, singular = np.linalg.lstsq(design / scale, y, rcond=None)
    if rank != nparam:
        return None
    fitted /= scale
    residual = y - design @ fitted
    return dict(region=region, model=f"area_log_inv{inverse_order}", requested_Lmin=low, requested_Lmax=high,
                a=float(fitted[0]), log_coefficient=float(fitted[1]),
                corner_c=-float(fitted[1]) if region == "S_B" else None,
                b=float(fitted[2]), **{f"inv{p}": float(fitted[2+p]) for p in range(1, inverse_order+1)},
                grid=core.grid_string(x), n=len(x), parameter_count=nparam, residual_dof=len(x)-nparam,
                rmse=float(np.sqrt(np.mean(residual**2))), max_abs_residual=float(max(abs(residual))),
                condition=float(singular[0]/singular[-1]), extent_complete=x[0] == low and x[-1] == high)


def coefficient_axis(ax, coefficients, title, detailed=False, show_bands=True):
    beta = core.array(coefficients, "beta")
    passed = [r for r in coefficients if r.get("local_diagnostic_pass") and r.get("c_neq") is not None]
    failed = [r for r in coefficients if not r.get("local_diagnostic_pass") and r.get("c_neq") is not None]
    handles = core.coefficient_curves(ax, coefficients, passed, failed, detailed=detailed)
    if show_bands:
        ax.fill_between(beta, core.array(coefficients, "eq_range_min"), core.array(coefficients, "eq_range_max"), color=core.BLUE, alpha=.12, zorder=1)
        ax.fill_between(beta, core.array(coefficients, "neq_range_min"), core.array(coefficients, "neq_range_max"), color=core.ORANGE, alpha=.14, zorder=1)
    bounds = np.concatenate([core.array(coefficients, key) for key in ("c_eq", "c_neq", "eq_range_min", "neq_range_min")])
    bounds = bounds[np.isfinite(bounds)]
    if len(bounds):
        ax.set_ylim(bottom=min(-.025*float(max(abs(bounds))), 1.05*float(min(bounds))))
    ax.axhline(0, color="0.55", lw=.6, alpha=.6, zorder=0)
    ax.set(xlim=(-.035, 2.035), xlabel=r"Measurement strength $\beta$", ylabel="Fitted coefficient c", title=title)
    ax.set_xticks(np.linspace(0, 2, 9))
    ax.grid(axis="y", alpha=.22)
    ax.legend(handles=handles, fontsize=8 if detailed else 9, handlelength=3)


def make_plots(rows, results, patterns, betas, directory, stage, plot_betas, theta_min, ratio_max, eq_max):
    directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.dpi": 180, "pdf.fonttype": 42})
    figures = []
    for name in patterns:
        result = results[name]
        coefficients, configuration = result["coefficients"], result["primary"]
        neq_n = sum(row.get("c_neq") is not None for row in coefficients)
        eq_n = sum(row.get("c_eq") is not None for row in coefficients)
        configured = core.primary_for_beta(configuration, betas[0])
        selection = f"eq L18..{eq_max}; time L{configured['L']}, theta {configured['window'][0]:g}..{configured['window'][1]:g}" if configured else "No primary configured"
        fig, ax = plt.subplots(figsize=(7, 4.6), constrained_layout=True)
        coefficient_axis(ax, coefficients, f"{name}: {selection}\n{stage}; eq {eq_n}/{len(betas)}, time {neq_n}/{len(betas)} available")
        ax.text(.5, -.22, "Filled/open time squares: local checks pass/unresolved.\nBands: available model/window ranges, not confidence intervals.",
                transform=ax.transAxes, ha="center", va="top", fontsize=8)
        stem = f"{name}_corner_coefficient_beta"
        core.save(fig, directory, stem)
        figures.append(stem)

        fig, axes = plt.subplots(1, 3, figsize=(16, 4.7), constrained_layout=True)
        coefficient_axis(axes[0], coefficients, "Finite-window estimates", detailed=True)
        beta = core.array(coefficients, "beta")
        passed = [row for row in coefficients if row.get("local_diagnostic_pass")]
        for ax, key, factor, title in ((axes[1], "ratio_neq_to_eq", 1, "Nonequilibrium / equilibrium"),
                                      (axes[2], "relative_gap", 100, "Relative difference (%)")):
            ax.plot(beta, factor*core.array(coefficients, key), "s-", color=core.ORANGE, mfc="white")
            ax.plot(core.array(passed, "beta"), factor*core.array(passed, key), "s", color=core.ORANGE)
            ax.axhline(1 if key == "ratio_neq_to_eq" else 0, color=core.BLUE, lw=1)
            ax.set(xlabel=r"$\beta$", ylabel=title, title=title, xlim=(-.035, 2.035))
            ax.set_xticks(np.linspace(0, 2, 9))
            ax.grid(axis="y", alpha=.25)
        fig.suptitle(f"{name}: {stage}; {selection}\nComparison panels never enter the local time diagnostics; bands are model/window ranges, not confidence intervals", fontsize=11)
        stem = f"{name}_coefficient_diagnostics"
        core.save(fig, directory, stem)
        figures.append(stem)

        for value in plot_betas:
            block = [row for row in rows if row["pattern"] == name and row["beta"] == value]
            if not block:
                continue
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), constrained_layout=True)
            eq = sorted((row for row in block if math.isinf(row["theta"]) and 18 <= row["L"] <= eq_max), key=lambda r: r["L"])
            if eq:
                axes[0].plot(np.log([row["L"] for row in eq]), [row["D"] for row in eq], "o", color=core.BLUE, label="Equilibrium data")
            fitted = next((row for row in result["eq"] if row["beta"] == value and row["requested_Lmin"] == 18 and row["model"] == "log_inv1"), None)
            if fitted:
                x = np.geomspace(fitted["x_min"], fitted["x_max"], 200)
                axes[0].plot(np.log(x), core.predict(fitted, x), color=core.BLUE, label=f"Fit: L={fitted['x_min']:g}..{fitted['x_max']:g}")
            coefficient = next(row for row in coefficients if row["beta"] == value)
            configured = core.primary_for_beta(configuration, value)
            finite = [row for row in block if math.isfinite(row["theta"])]
            allowed = set()
            if configured:
                allowed.add(configured["L"])
                comparison = coefficient.get("comparison_L", configured.get("comparison_L"))
                if comparison is not None:
                    allowed.add(comparison)
                if coefficient.get("third_L") is not None:
                    allowed.add(coefficient["third_L"])
                else:
                    larger = sorted({row["L"] for row in finite if row["L"] > configured["L"] and row["theta"] >= theta_min and row["theta"]/row["L"] <= ratio_max})
                    if larger:
                        allowed.add(larger[0])
            finite = [row for row in finite if row["L"] in allowed and row["theta"] >= theta_min and row["theta"]/row["L"] <= ratio_max]
            sizes = sorted({row["L"] for row in finite})
            # Keep the size encoding stable while preliminary grids fill in.
            # Nested open markers reveal coincident curves without moving data.
            style_order = sorted(allowed)
            colors = ("#5b3b8c", "#148f8b", "#657a20")
            for size in sizes:
                index = style_order.index(size) % 3
                color = colors[index]
                points = sorted((row for row in finite if row["L"] == size), key=lambda r: r["theta"])
                axes[1].plot(np.log([row["theta"] for row in points]), [row["D"] for row in points],
                             marker=("o", "s", "^")[index], linestyle=("-", "--", ":")[index],
                             color=color, ms=(7.2, 5.2, 3.2)[index], lw=.8, mew=1,
                             mfc="white" if index < 2 else color, zorder=4+index, label=f"L={size}")
            if configured and coefficient.get("c_neq") is not None:
                chosen = next(row for row in result["pairs"] if row["beta"] == value and row["L_large"] == configured["L"] and row["L_small"] == coefficient["comparison_L"] and row["window"] == coefficient["primary_window"] and row["model"] == configured["model"])
                x = np.geomspace(chosen["theta_min"], chosen["theta_max"], 200)
                label = "Primary fit: local checks" if coefficient.get("local_diagnostic_pass") else "Primary fit: unresolved diagnostic"
                axes[1].plot(np.log(x), core.predict(chosen, x), color=core.ORANGE, lw=2, zorder=3, label=label)
            else:
                axes[1].text(.98, .05, "Primary time fit not yet available", transform=axes[1].transAxes,
                             ha="right", va="bottom", fontsize=8, bbox=dict(facecolor="white", alpha=.85, edgecolor="none"))
            if not finite:
                limits = configured.get("window", [theta_min, theta_min*1.75]) if configured else [theta_min, theta_min*1.75]
                axes[1].set_xlim(np.log(limits))
                axes[1].set_yticks([])
                axes[1].text(.5, .5, "No time observations within the configured size / scale screen",
                             transform=axes[1].transAxes, ha="center", fontsize=8)
            axes[0].set(xlabel=r"$\ln L$", ylabel="D", title="Equilibrium scaling")
            axes[1].set(xlabel=r"$\ln\theta$", ylabel="D", title=f"Time data: theta >= {theta_min:g}, theta/L <= {ratio_max:g}")
            for ax in axes:
                ax.set_xscale("linear")
                ax.grid(axis="y", alpha=.25)
                if ax.get_legend_handles_labels()[0]:
                    ax.legend(fontsize=7, ncol=2)
            fig.suptitle(f"{name}, beta={value:g}: {stage}; curves outside the primary window remain diagnostics", fontsize=11)
            stem = f"{name}_scaling_beta_{value:g}".replace(".", "p")
            core.save(fig, directory, stem)
            figures.append(stem)

    fig, axes = plt.subplots(1, len(patterns), figsize=(5.3*len(patterns), 4.7), squeeze=False)
    fig.subplots_adjust(left=.06, right=.99, bottom=.15, top=.79, wspace=.27)
    for ax, name in zip(axes[0], patterns):
        coefficient_axis(ax, results[name]["coefficients"], name, show_bands=False)
    primary_values = [float(row[key]) for name in patterns for row in results[name]["coefficients"]
                      for key in ("c_eq", "c_neq") if row.get(key) is not None and math.isfinite(row[key])]
    low, high = min([0.] + primary_values), max([0.] + primary_values)
    padding = .05 * (high-low if high > low else .01)
    for ax in axes[0]:
        ax.set_ylim(low-padding, high+padding)
    fig.suptitle(f"Finite-window D coefficients ({stage}, single species); filled/open time squares = local checks pass/unresolved\nSame y scale in all panels; sensitivity ranges shown in individual diagnostics", fontsize=11)
    core.save(fig, directory, "corner_coefficients_comparison")
    figures.append("corner_coefficients_comparison")
    return figures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=HERE/"analysis_beta_patterns")
    parser.add_argument("--figures", type=Path, default=HERE/"figures_beta_patterns")
    parser.add_argument("--plan", type=Path, default=HERE/"analysis_beta_patterns/plan.json")
    parser.add_argument("--stage", choices=("preliminary", "final"), default="preliminary")
    parser.add_argument("--plot-betas", nargs="+", type=float, default=[0, .5, 1, 1.5, 2])
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--self-check", action="store_true", help="Run only synthetic ingestion and pattern-isolation checks")
    args = parser.parse_args()
    if args.self_check:
        framework_selfcheck(args.output)
        return
    plan_bytes = args.plan.read_bytes()
    plan = json.loads(plan_bytes)
    dependency_bytes = Path(core.__file__).read_bytes()
    patterns = plan.get("patterns", list(PATTERNS))
    if not patterns or len(set(patterns)) != len(patterns) or any(name not in PATTERNS for name in patterns):
        raise ValueError("Expected unique locked pattern names")
    for name in patterns:
        if plan.get("pattern_bits", {}).get(name, PATTERNS[name]["occupied_bits"]) != PATTERNS[name]["occupied_bits"]:
            raise ValueError(f"Unexpected mask for {name}")
    betas = sorted({core.canonical(value) for value in plan.get("betas", [i/4 for i in range(9)])})
    windows = [tuple(map(float, window)) for window in plan.get("windows", DEFAULT_WINDOWS)]
    theta_min, ratio_max = float(plan.get("theta_min_screen", 4)), float(plan.get("theta_over_L_max_screen", .05))
    eq_primary = plan.get("equilibrium_primary", {"window": [18, 144], "model": "log_inv1"})
    if eq_primary.get("window", [18, 144])[0] != 18 or eq_primary.get("model", "log_inv1") != "log_inv1":
        raise ValueError("This protocol uses the explicit equilibrium primary 18..max with log_inv1")
    eq_max = int(eq_primary.get("window", [18, 144])[1])
    paths = args.inputs or [Path(path) if Path(path).is_absolute() else HERE/path for path in plan.get("inputs", [])] or DEFAULT_INPUTS
    args.output.mkdir(parents=True, exist_ok=True)
    rows, sources, duplicates = read_sources(paths, args.stage, args.output, set(patterns), set(betas))
    tables = defaultdict(list)
    tables["observations"], tables["duplicate_observations"] = rows, duplicates
    results, expected = {}, 0
    config_args = argparse.Namespace(primary_L=None, comparison_L=None, primary_window=None, primary_model=None)
    for name in patterns:
        pattern_plan = per_pattern_plan(plan, name, patterns)
        primary = core.explicit_primary(config_args, pattern_plan)
        for value in betas:
            specified = core.primary_for_beta(primary, value)
            if specified and tuple(specified["window"]) not in windows:
                raise ValueError(f"Primary window must occur in the diagnostics: {name}")
        if args.stage == "final" and (not primary or not pattern_plan.get("required_finite_grids") or not pattern_plan.get("required_equilibrium_grids")):
            raise ValueError(f"Final requires primary and both required domains for {name}")
        block = [row for row in rows if row["pattern"] == name]
        count, missing_keys = core.coverage_check(block, pattern_plan, betas)
        expected += count
        tables["missing_required_keys"].extend(tagged(name, missing_keys))
        eq, individual, pairs, missing, _, finite = core.analyze(block, betas, windows, theta_min, ratio_max, eq_max)
        coefficients = core.primary_rows(betas, primary, eq, individual, pairs, finite, eq_max, theta_min, ratio_max)
        sensitivities = core.sensitivity_rows(coefficients, eq, individual, pairs)
        for title, values in (("equilibrium_fits", eq), ("time_individual_fits", individual), ("time_pair_fits", pairs),
                              ("missing_fits", missing), ("primary_coefficients", coefficients), ("time_sensitivity", sensitivities)):
            tables[title].extend(tagged(name, values))
        for value, low, order, region in itertools.product(betas, (18, 36, 54), (1, 2), ("S_A", "S_C", "S_B")):
            raw = entropy_fit([row for row in block if row["beta"] == value], region, low, eq_max, order)
            if raw:
                tables["raw_entropy_fits"].extend(tagged(name, [dict(beta=value, **raw)]))
        results[name] = dict(eq=eq, individual=individual, pairs=pairs, coefficients=coefficients, primary=primary,
                             required_keys=count, missing_required_keys=len(missing_keys))
    if args.stage == "final" and tables["missing_required_keys"]:
        raise ValueError(f"Incomplete required grids: {len(tables['missing_required_keys'])} keys")
    complete = all(row.get("c_eq") is not None and row.get("c_neq") is not None for row in tables["primary_coefficients"])
    if args.stage == "final" and not complete:
        raise ValueError("Final requires every explicit primary pair and operational size screen, not a local pass")
    groups = defaultdict(list)
    for row in rows:
        groups[row["pattern"], row["beta"], row["L"]].append(row)
    for (name, beta, size), block in sorted(groups.items()):
        times = sorted(row["theta"] for row in block if math.isfinite(row["theta"]))
        tables["coverage"].append(dict(pattern=name, beta=beta, L=size, equilibrium_present=any(math.isinf(row["theta"]) for row in block),
                                       finite_n=len(times), times=core.grid_string(times), theta_over_L_max=max(times)/size if times else None))
    for title, values in tables.items():
        core.dump_csv(args.output/f"{title}.csv", values)
    figures = [] if args.no_plots else make_plots(rows, results, patterns, betas, args.figures, args.stage, args.plot_betas, theta_min, ratio_max, eq_max)
    changed = [source["path"] for source in sources if not source.get("missing") and core.sha_bytes(Path(source["path"]).read_bytes()) != source["sha256"]]
    dependency_changed = Path(core.__file__).read_bytes() != dependency_bytes
    plan_changed = args.plan.read_bytes() != plan_bytes
    if args.stage == "final" and (changed or dependency_changed or plan_changed):
        raise ValueError("Source, dependency or plan changed during final analysis; rerun after freeze")
    summary = dict(generated_utc=datetime.now(timezone.utc).isoformat(), stage=args.stage,
                   script_sha256=core.sha_bytes(Path(__file__).read_bytes()),
                   dependencies=[dict(path=str(Path(core.__file__).resolve()), sha256=core.sha_bytes(dependency_bytes))],
                   dependency_changed_during_analysis=dependency_changed, plan_changed_during_analysis=plan_changed,
                   plan_sha256=core.sha_bytes(plan_bytes), plan=plan, sources=sources, sources_changed_during_analysis=changed,
                   arguments={key: str(value) if isinstance(value, Path) else [str(path) for path in value] if key == "inputs" and value else value for key, value in vars(args).items()},
                   patterns={name: PATTERNS[name] for name in patterns}, betas=betas, windows=windows,
                   primary={name: results[name]["primary"] for name in patterns},
                   rows=len(rows), duplicates=len(duplicates), required_keys=expected,
                   required_grid_complete=bool(expected) and not tables["missing_required_keys"],
                   missing_required_keys=len(tables["missing_required_keys"]), primary_complete=complete,
                   primary_rows=len(tables["primary_coefficients"]),
                   fit_counts={key: len(tables[key]) for key in ("equilibrium_fits", "time_individual_fits", "time_pair_fits", "raw_entropy_fits")},
                   by_pattern={name: dict(required_keys=results[name]["required_keys"], missing_required_keys=results[name]["missing_required_keys"],
                                           available_eq=sum(row.get("c_eq") is not None for row in results[name]["coefficients"]),
                                           available_time=sum(row.get("c_neq") is not None for row in results[name]["coefficients"]),
                                           local_pass_beta=[row["beta"] for row in results[name]["coefficients"] if row.get("local_diagnostic_pass")]) for name in patterns},
                   figures=figures,
                   conventions=dict(entropy="single-species natural-log Renyi-2; no SA=SC assumption",
                                    observable="D=(2SA+SC)/3-SB; masks and cut origin fixed, no phase averaging",
                                    raw_entropy_fits="a L + log_coefficient ln L + b + inverse powers; corner_c=-log_coefficient only for SB, not used in local time gates",
                                    time_fit="true common grids; both window endpoints and n>=parameters+2 required",
                                    primary="explicit configuration; never selected by equilibrium agreement; local failure retains the finite-window estimate",
                                    local_gates="same inherited rules as audited uniformA: size<=2%, raw D difference/log span<=1%, model/window<=5%, same sign, required models/windows; third size if present must have <=2% and decreasing drift",
                                    scale_screen="both primary/comparison satisfy theta_min and theta/L; operational condition, not proof of microscopic or asymptotic scale separation",
                                    sensitivity="same-window model checks use common-grid pair; window checks/bands use individual measured grids at primary L and report separate size-check availability; not confidence intervals",
                                    axes="explicit ln L / ln theta on linear scaling axes; beta/coefficient axes linear; missing primary beta values remain gaps"))
    (args.output/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("stage", "rows", "fit_counts", "required_keys", "missing_required_keys", "primary_complete", "by_pattern")}, indent=2))


if __name__ == "__main__":
    main()
