"""Independent larger-size beta=3 analysis; never overwrites the old production.

Only complete frozen stages are accepted. Pilot observations are not pooled.
Uses the original validated diagnostics, and column-scaled QR for finite-window
fits. Model/window rules are in the separately frozen beta3_large_protocol.json.
"""
from pathlib import Path
import argparse
import json
import math
import tomllib
from datetime import datetime, timezone

import numpy as np
import analyze_born_scaling as base
import analyze_born_linear_crossover as regression

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "analysis/born_scaling/beta3_large"
FIG = ROOT / "figures/born_scaling/beta3_large"
PROTOCOL_PATH = ROOT / "analysis/born_scaling/beta3_large_protocol.json"


def frozen_inputs(stages):
    previous_path = ROOT / "analysis/born_scaling/production/summary.json"
    previous = json.loads(previous_path.read_text(encoding="utf-8"))
    if previous["status"] != "PASS":
        raise ValueError("Previous complete production required")
    dependencies = base.verify_file_map(previous["files"])
    for path in (previous_path, PROTOCOL_PATH):
        dependencies[base.relative(path)] = base.sha256(path)
    provenance = OUT.parent / "provenance"
    provenance.mkdir(parents=True, exist_ok=True)
    analysis_sources = {}
    for path in (Path(base.__file__), Path(regression.__file__), Path(__file__)):
        digest = base.sha256(path)
        snapshot = provenance / f"{path.stem}_{digest}.py"
        if snapshot.exists():
            if base.sha256(snapshot) != digest:
                raise ValueError("Changed content-addressed analysis source snapshot")
        else:
            with snapshot.open("xb") as stream:
                stream.write(path.read_bytes())
        dependencies[base.relative(snapshot)] = digest
        analysis_sources[base.relative(path)] = dict(sha256=digest, snapshot=base.relative(snapshot))
    validation_path = ROOT / "analysis/born_scaling_production_validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if validation["status"] != "PASS":
        raise ValueError("Previous independent validation required")
    dependencies[base.relative(validation_path)] = base.sha256(validation_path)
    previous_plan = tomllib.loads((ROOT / "data/born_scaling/production/production_plan.toml").read_text(encoding="utf-8"))
    seeds = {job["seed"] for job in previous_plan["jobs"]}
    groups = {}
    plans = []
    max_identity_error = 0.
    pilot_seeds = set()
    for stage in stages:
        if not stage.startswith("production") or stage == "production":
            raise ValueError("Only independent new production stages may be added")
        plan, new_groups, files, identity_error = base.load_complete_stage(stage)
        if plan.get("update_kernel") != "fused_rank_updates":
            raise ValueError("Changed update kernel")
        protocol_name = base.relative(PROTOCOL_PATH)
        bound_protocol = plan.get("analysis_protocol_sha256") if plan.get("analysis_protocol_path") == protocol_name else plan["source_sha256"].get(protocol_name)
        if bound_protocol != base.sha256(PROTOCOL_PATH):
            raise ValueError("Analysis protocol is not bound to the frozen production plan")
        if "throughput_benchmark_path" in plan:
            benchmark_path = ROOT / plan["throughput_benchmark_path"]
            if base.sha256(benchmark_path) != plan["throughput_benchmark_sha256"]:
                raise ValueError("Changed throughput evidence")
            benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
            if benchmark["status"] != "PASS" or benchmark["recommended_workers"] != plan["execution_workers"]:
                raise ValueError("Production worker count differs from frozen timing decision")
            dependencies.update(base.verify_file_map(benchmark["files"]))
            dependencies[base.relative(benchmark_path)] = base.sha256(benchmark_path)
        dependencies.update(files)
        plans.append(plan)
        max_identity_error = max(max_identity_error, identity_error)
        # Bind the source of the allocation, and explicitly test seed independence.
        for evidence_name, digest in plan.get("pilot_evidence", {}).items():
            evidence_path = ROOT / evidence_name
            if base.sha256(evidence_path) != digest:
                raise ValueError("Changed pilot evidence")
            dependencies[evidence_name] = digest
            if evidence_path.suffix == ".toml":
                evidence = tomllib.loads(evidence_path.read_text(encoding="utf-8"))
                if "plan_path" in evidence:
                    pilot_path = ROOT / evidence["plan_path"]
                    if base.sha256(pilot_path) != evidence["plan_sha256"]:
                        raise ValueError("Changed pilot plan")
                    dependencies[base.relative(pilot_path)] = base.sha256(pilot_path)
                    pilot = tomllib.loads(pilot_path.read_text(encoding="utf-8"))
                    pilot_seeds.update(job["seed"] for job in pilot["jobs"])
                    dependencies.update(base.verify_file_map(evidence["files"]))
        for (L, beta, mu), chains in new_groups.items():
            if L <= 24 or L % 3 or beta != 3. or mu != 1e-8 or (L, beta, mu) in groups:
                raise ValueError("Wrong or duplicate extension case")
            for c in chains:
                job = c["job"]
                if job["seed"] in seeds:
                    raise ValueError("Repeated old/new production seed")
                seeds.add(job["seed"])
                burn = job["analysis_burnin"]
                keep = job["retained_sweeps"]
                if keep < 2048 or burn != plan["analysis_burnin"] or job["sweeps"] != burn + keep:
                    raise ValueError("Retention differs from frozen allocation")
                select = c["sweeps"] > burn
                if select.sum() != keep:
                    raise ValueError("Missing retained sweeps")
                c["production_data"] = c["data"][select]
                c["retained_accepted"] = int(c["accepted_counts"][select].sum())
                c["retained_proposed"] = int(c["proposed_counts"][select].sum())
            groups[(L, beta, mu)] = chains
    if seeds & pilot_seeds:
        raise ValueError("Pilot seed reused in production")
    points = []
    for row in base.read_csv(ROOT / "analysis/born_scaling/production/means.csv"):
        if float(row["beta"]) == 3. and row["grid"] == "main":
            point = dict(L=int(row["L"]), beta=3., mu=1e-8, source="previous_production",
                         sampling_screen_pass=row["sampling_screen_pass"] == "True")
            for o in ("SA", "SB", "SC", "D", "SA_minus_SC"):
                for field in ("mean", "MCSE", "rhat_max", "ESS_bulk", "ESS_tail"):
                    point[f"{o}_{field}"] = float(row[f"{o}_{field}"])
            points.append(point)
    if sorted(p["L"] for p in points) != [6, 9, 12, 15, 18, 21, 24]:
        raise ValueError("Unexpected previous size grid")
    return points, groups, plans, dependencies, max_identity_error, analysis_sources


def summarize_new(groups):
    points, diagnostics, blocks, covariance, cumulative, chain_means, acceptance = [], [], [], [], [], [], []
    for (L, beta, mu), chains in sorted(groups.items()):
        raw = np.stack([c["production_data"] for c in chains])
        burn = chains[0]["job"]["analysis_burnin"]
        common = dict(L=L, beta=beta, mu=mu, burnin=burn)
        point = dict(**common, source="new_production", n_chains=4, draws_per_chain=raw.shape[1])
        values = dict(SA=raw[:, :, 0], SB=raw[:, :, 1], SC=raw[:, :, 2], D=raw[:, :, 3],
                      SA_minus_SC=raw[:, :, 0]-raw[:, :, 2])
        for observable, x in values.items():
            diag, batch = base.error_summary(x)
            diagnostics.append(dict(**common, observable=observable, **diag))
            blocks.extend(dict(**common, observable=observable, **b) for b in batch)
            for field, source in (("mean", "mean"), ("MCSE", "MCSE_conservative"),
                                  ("rhat_max", "rhat_max"), ("ESS_bulk", "ESS_bulk"), ("ESS_tail", "ESS_tail")):
                point[f"{observable}_{field}"] = diag[source]
            point[f"{observable}_screen_pass"] = diag["initial_screen_pass"]
            for ci, chain in enumerate(chains):
                chain_means.append(dict(**common, observable=observable, chain=chain["job"]["chain"],
                    seed=chain["job"]["seed"], start=chain["job"]["start"], mean=float(x[ci].mean()), draws=len(x[ci])))
            if observable in ("SA", "D"):
                point[f"{observable}_precision_target_met"] = bool(diag["MCSE_conservative"] is not None and
                    diag["MCSE_conservative"] <= base.PRECISION_TARGETS[observable])
                checkpoints = sorted({n for n in (128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768)
                                      if n <= x.shape[1]} | {x.shape[1]})
                for n in checkpoints:
                    d, _ = base.error_summary(x[:, :n])
                    cumulative.append(dict(**common, observable=observable, **d,
                        first_half_mean=float(x[:, :n//2].mean()), second_half_mean=float(x[:, n//2:n].mean())))
        for length in base.BLOCK_LENGTHS:
            covariance.append(dict(**common, **base.covariance_checks(raw[:, :, :3], length)))
        accepted = sum(c["retained_accepted"] for c in chains)
        proposed = sum(c["retained_proposed"] for c in chains)
        point["acceptance"] = accepted / proposed
        point["sampling_screen_pass"] = point["SA_screen_pass"] and point["D_screen_pass"]
        acceptance.append(dict(**common, retained_sweeps=raw.shape[0]*raw.shape[1], accepted=accepted,
                               proposed=proposed, pooled_acceptance=accepted/proposed))
        points.append(point)
    return points, dict(diagnostics=diagnostics, block_errors=blocks, region_covariance=covariance,
                        cumulative_diagnostics=cumulative, chain_means=chain_means, acceptance=acceptance)


def fit_all(points, protocol):
    fits, residuals, covariance, skipped = [], [], [], []
    definitions = protocol["D_models"] | protocol["SA_models"]
    regression.MODELS = {name: tuple(terms) for name, terms in definitions.items()}
    for maximum in (24, max(p["L"] for p in points)):
        for minimum in protocol["L_min_candidates"]:
            selected = [p for p in points if minimum <= p["L"] <= maximum]
            for model, terms in definitions.items():
                observable = "D" if model.startswith("D_") else "SA"
                key = dict(beta=3., mu=1e-8, observable=observable, model=model, L_min=minimum, L_max=maximum,
                    window_role="primary_large_size_window" if minimum == protocol["primary_large_size_window_L_min"] and maximum > 24 else "sensitivity_window")
                if len(selected) < len(terms) + protocol["minimum_residual_degrees_of_freedom"]:
                    skipped.append(dict(**key, n_sizes=len(selected), reason="fewer_than_three_residual_dof"))
                    continue
                if any(p[f"{observable}_MCSE"] is None or p[f"{observable}_MCSE"] <= 0 for p in selected):
                    skipped.append(dict(**key, n_sizes=len(selected), reason="unresolved_or_nonpositive_MCSE_in_window"))
                    continue
                L = np.array([p["L"] for p in selected], dtype=float)
                y = np.array([p[f"{observable}_mean"] for p in selected])
                se = np.array([p[f"{observable}_MCSE"] for p in selected])
                result, delta = regression.fit(L, y, se, model)
                fits.append(dict(**key, sampling_screen_pass=all(p["sampling_screen_pass"] for p in selected), **result))
                for p, r in zip(selected, delta):
                    residuals.append(dict(**key, L=p["L"], observed=p[f"{observable}_mean"],
                        MCSE=p[f"{observable}_MCSE"], residual=float(r),
                        standardized_residual=float(r/p[f"{observable}_MCSE"])))
                X = regression.design(L, model) / se[:, None]
                scale = np.linalg.norm(X, axis=0)
                _, R = np.linalg.qr(X/scale, mode="reduced")
                transform = np.linalg.solve(R, np.eye(len(terms))) / scale[:, None]
                cov = transform @ transform.T
                for i, a in enumerate(terms):
                    for j, b in enumerate(terms):
                        covariance.append(dict(**key, term_a=a, term_b=b, covariance=float(cov[i, j]),
                            correlation=float(cov[i, j]/math.sqrt(cov[i, i]*cov[j, j]))))
    return fits, residuals, covariance, skipped


def local_slopes(points):
    slopes, covariances = [], []
    sizes = [p["L"] for p in points]
    means = np.array([p["D_mean"] for p in points])
    missing = np.array([p["D_MCSE"] is None for p in points])
    variances = np.array([p["D_MCSE"]**2 if p["D_MCSE"] is not None else 0. for p in points])
    all_rows = []
    for kind in ("adjacent", "width_six"):
        pairs = list(zip(sizes[:-1], sizes[1:])) if kind == "adjacent" else [(L, L+6) for L in sizes if L+6 in sizes]
        for axis in ("log", "linear"):
            H = np.zeros((len(pairs), len(sizes)))
            for i, (a, b) in enumerate(pairs):
                denominator = math.log(b/a) if axis == "log" else b-a
                H[i, sizes.index(a)] = -1/denominator
                H[i, sizes.index(b)] = 1/denominator
            values = H @ means
            cov = (H * variances) @ H.T
            for i, (a, b) in enumerate(pairs):
                slopes.append(dict(slope_id=len(slopes), kind=kind, axis=axis, L_left=a, L_right=b, L_midpoint=(a+b)/2,
                    slope=float(values[i]), MCSE=None if np.any((H[i] != 0) & missing) else math.sqrt(cov[i, i])))
                all_rows.append(H[i])
    H = np.stack(all_rows)
    covariance = (H * variances) @ H.T
    for i, a in enumerate(slopes):
        for j, b in enumerate(slopes):
            covariances.append(dict(slope_id_a=i, slope_id_b=j,
                kind_a=a["kind"], axis_a=a["axis"], left_a=a["L_left"], right_a=a["L_right"],
                kind_b=b["kind"], axis_b=b["axis"], left_b=b["L_left"], right_b=b["L_right"],
                covariance=None if a["MCSE"] is None or b["MCSE"] is None else float(covariance[i, j])))
    return slopes, covariances


def make_plots(points, fits, slopes, cumulative, protocol):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "figure.dpi": 160})
    FIG.mkdir(parents=True, exist_ok=True)
    paths = []
    maximum = max(p["L"] for p in points)
    minimum = protocol["primary_large_size_window_L_min"]
    styles = {
        "D_log": ("#257899", "--", r"$c\ln L+b$"),
        "D_linear": ("#D17A28", ":", r"$aL+b$"),
        "D_log_invL": ("#708532", "-", r"$c\ln L+b+d/L$"),
        "D_linear_log_diagnostic": ("#96618B", "-.", r"$aL+c\ln L+b$")}

    def errors(values):
        return np.array([np.nan if v is None else v for v in values], dtype=float)

    def save(fig, name):
        for suffix in ("png", "pdf"):
            path = FIG / f"{name}.{suffix}"
            fig.savefig(path, bbox_inches="tight")
            paths.append(path)
        plt.close(fig)

    def scatter(ax, observable, logarithmic):
        for source, marker, color, label in (("previous_production", "o", "#286B9C", "Previous L <= 24"),
                ("new_production", "s", "#D17A28", "New larger sizes")):
            selected = [p for p in points if p["source"] == source]
            x = np.array([p["L"] for p in selected])
            ax.errorbar(np.log(x) if logarithmic else x, [p[f"{observable}_mean"] for p in selected],
                yerr=errors([p[f"{observable}_MCSE"] for p in selected]), fmt=marker, mfc="white", mec=color,
                color=color, ms=6, capsize=2.5, label=label, zorder=5)

    fig, axes = plt.subplots(2, 2, figsize=(13.8, 10.0), constrained_layout=True)
    for ax, logarithmic in zip(axes[0], (True, False)):
        scatter(ax, "D", logarithmic)
        grid = np.linspace(minimum, maximum, 250)
        for model, (color, linestyle, label) in styles.items():
            selected = [f for f in fits if f["model"] == model and f["L_min"] == minimum and f["L_max"] == maximum]
            if not selected:
                continue
            fit = selected[0]
            coefficient = np.array([fit[f"coefficient_{term}"] for term in regression.MODELS[model]])
            ax.plot(np.log(grid) if logarithmic else grid, regression.design(grid, model) @ coefficient,
                color=color, ls=linestyle, lw=1.7, label=label+f", chi2/dof={fit['reduced_chi_square']:.3g}")
        ax.set(xlabel=r"$\ln L$" if logarithmic else r"$L$", ylabel=r"$\overline{D}$",
               title=f"beta=3 | fits restricted to L={minimum}–{maximum}")
        ax.legend(fontsize=8.5)
        ax.grid(alpha=.18)
    for ax, axis in zip(axes[1], ("log", "linear")):
        for kind, marker, color, shift in (("adjacent", "o", "#286B9C", 0.), ("width_six", "s", "#D17A28", .12)):
            selected = [s for s in slopes if s["kind"] == kind and s["axis"] == axis]
            ax.errorbar([s["L_midpoint"]+shift for s in selected], [s["slope"] for s in selected],
                yerr=errors([s["MCSE"] for s in selected]), fmt=marker, mfc="white", color=color,
                capsize=2, ms=5, label="Adjacent sizes" if kind == "adjacent" else "Intervals of width 6")
        ax.set(xlabel="Interval midpoint", ylabel=r"$\Delta D/\Delta\ln L$" if axis == "log" else r"$\Delta D/\Delta L$",
               title="Raw local slopes | shared endpoints imply covariance")
        ax.grid(alpha=.18)
        ax.legend(fontsize=9)
    fig.suptitle("Conditional half-output Born average | beta=3, mu/t=1e-8", fontsize=15)
    unresolved = any(p["D_MCSE"] is None or p["SA_MCSE"] is None for p in points)
    fig.supxlabel("Error bars: estimated MCSE. Curves are finite-window descriptions, not established asymptotic laws." +
                  (" Missing error bars: MCSE unresolved." if unresolved else ""), fontsize=10)
    save(fig, "scaling_and_local_slopes")

    fig, axes = plt.subplots(2, 2, figsize=(13.6, 9.2), constrained_layout=True)
    for model, (color, linestyle, label) in styles.items():
        selected = [f for f in fits if f["model"] == model and f["L_max"] == maximum]
        axes[0, 0].plot([f["L_min"] for f in selected], [f["reduced_chi_square"] for f in selected],
                       marker="o", mfc="white", color=color, ls=linestyle, label=label)
        if "logL" in regression.MODELS[model]:
            axes[0, 1].errorbar([f["L_min"] for f in selected], [f["coefficient_logL"] for f in selected],
                yerr=[f["MCSE_logL"] for f in selected], marker="o", mfc="white", color=color,
                ls=linestyle, capsize=2, label=label)
    axes[0, 0].axhline(3, color="#555555", ls=":", lw=1, label="Misfit marker = 3")
    axes[0, 0].set(yscale="log", xlabel=r"$L_{\min}$", ylabel="chi2/dof", title=f"Fit-window sensitivity | Lmax={maximum}")
    axes[0, 1].set(xlabel=r"$L_{\min}$", ylabel="Fitted log coefficient", title="Model-conditional statistical errors only")
    scatter(axes[1, 0], "SA", False)
    axes[1, 0].set(xlabel="L", ylabel=r"$\overline{S_2(A)}$", title="Strip entropy control")
    for L in sorted({r["L"] for r in cumulative}):
        selected = [r for r in cumulative if r["L"] == L and r["observable"] == "D"]
        axes[1, 1].errorbar([r["draws_per_chain"] for r in selected], [r["mean"] for r in selected],
            yerr=errors([r["MCSE_conservative"] for r in selected]), marker="o", mfc="white", capsize=2,
            label=f"L={L}"+(" (some MCSE unresolved)" if any(r["MCSE_conservative"] is None for r in selected) else ""))
    axes[1, 1].set(xscale="log", xlabel="Retained sweeps per chain", ylabel=r"$\overline{D}$",
        title="Nested-prefix check | four independent starts")
    from matplotlib.ticker import NullFormatter
    prefix_ticks = [128, 256, 512, 1024, 2048, max(r["draws_per_chain"] for r in cumulative)]
    axes[1, 1].set_xticks(sorted(set(prefix_ticks)), labels=[str(n) for n in sorted(set(prefix_ticks))])
    axes[1, 1].xaxis.set_minor_formatter(NullFormatter())
    for ax in axes.flat:
        ax.grid(alpha=.18)
        ax.legend(fontsize=8.5)
    fig.suptitle("beta=3 extension | sampling and scaling are separate checks", fontsize=15)
    fig.supxlabel("All eligible cuts reported (at least three residual degrees of freedom). Prefix estimates are correlated.", fontsize=10)
    save(fig, "fit_and_sampling_diagnostics")
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stages", nargs="+", required=True)
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    points, groups, plans, dependencies, identity, analysis_sources = frozen_inputs(args.stages)
    new, tables = summarize_new(groups)
    points = sorted(points+new, key=lambda p: p["L"])
    fits, residuals, fit_cov, skipped = fit_all(points, protocol)
    slopes, slope_cov = local_slopes(points)
    tables.update(means=points, fits=fits, fit_residuals=residuals, fit_covariance=fit_cov,
                  skipped_fits=skipped, local_slopes=slopes, local_slope_covariance=slope_cov)
    OUT.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        path = OUT / f"{name}.csv"
        base.write_csv(path, rows)
        dependencies[base.relative(path)] = base.sha256(path)
    paths = make_plots(points, fits, slopes, tables["cumulative_diagnostics"], protocol)
    dependencies.update({base.relative(path): base.sha256(path) for path in paths})
    summary = dict(status="PASS", created_utc=datetime.now(timezone.utc).isoformat(), protocol=protocol,
        source_stages=args.stages, sizes=[p["L"] for p in points], new_sizes=[p["L"] for p in new],
        new_chains=4*len(new), new_retained_observations=sum(4*p["draws_per_chain"] for p in new),
        per_sample_D_identity_max_error=identity, all_sampling_screen_pass=all(p["sampling_screen_pass"] for p in points),
        precision_misses=[dict(L=p["L"], observable=o, MCSE=p[f"{o}_MCSE"], target=base.PRECISION_TARGETS[o])
                          for p in new for o in ("SA", "D") if not p[f"{o}_precision_target_met"]],
        maximum_inverse_drift=max(c["receipt"]["maximum_inverse_drift"] for cs in groups.values() for c in cs),
        maximum_logweight_drift=max(c["receipt"]["maximum_logweight_drift"] for cs in groups.values() for c in cs),
        primary_fits=[f for f in fits if f["window_role"] == "primary_large_size_window"],
        new_means=new, files=dependencies, analysis_sources=analysis_sources,
        inference="Complete-input analysis; PASS verifies computation, not a scaling law or complete rare-event coverage")
    base.write_json(OUT / "summary.json", summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ("files", "protocol")}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
