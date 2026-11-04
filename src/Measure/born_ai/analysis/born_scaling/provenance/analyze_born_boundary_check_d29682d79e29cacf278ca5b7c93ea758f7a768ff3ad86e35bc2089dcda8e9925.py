"""Paired analysis of the Born boundary-cancellation experiment.

The p=1/3 and p=1/2 geometries use their own exactly specified translation
subgroups (nine and four shifts). Translations are correlated measurements of
one state, never extra Monte Carlo draws. Every contrast is formed per retained
record before its uncertainty is estimated. Old production is never pooled.
"""
from pathlib import Path
import argparse
import csv
import json
import math
import os
import tomllib
from datetime import datetime, timezone

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np
import analyze_born_scaling as base

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "analysis/born_scaling/boundary_check"
FIG = ROOT / "figures/born_scaling/boundary_check"
SIZES = (6, 12, 18, 24, 30, 36)
REGIONS = ("SA", "SB", "SC", "D")
MODELS = {
    "log": ("intercept", "logL"),
    "log_invL": ("intercept", "logL", "invL"),
    "linear": ("intercept", "L"),
    "constant": ("intercept",),
    "invL": ("intercept", "invL"),
}


def read_metadata(path):
    path = Path(path)
    return (tomllib.loads(path.read_text(encoding="utf-8-sig")) if path.suffix == ".toml"
            else json.loads(path.read_text(encoding="utf-8-sig")))


def resolve_path(value, relative_to=ROOT):
    path = Path(value)
    return path if path.is_absolute() else relative_to / path


def remember(path, dependencies):
    path = Path(path).resolve()
    try:
        name = base.relative(path)
    except ValueError:
        # Explicit allocation/evidence files may live on another Windows drive.
        name = path.as_posix()
    dependencies[name] = base.sha256(path)


def verify_mapping(mapping, dependencies, relative_to=ROOT):
    for name, expected in mapping.items():
        path = resolve_path(name, relative_to)
        digest = expected.get("sha256") if isinstance(expected, dict) else expected
        if base.sha256(path) != digest:
            raise ValueError(f"Input hash mismatch: {path}")
        remember(path, dependencies)


def verify_frozen_evidence(plan, dependencies):
    """Bind design evidence and its underlying files before reading any observations."""
    verify_mapping(plan.get("evidence_sha256", {}), dependencies)
    for stem in ("allocation", "pilot_summary"):
        path_key, hash_key = f"{stem}_path", f"{stem}_sha256"
        if (path_key in plan) != (hash_key in plan):
            raise ValueError(f"Incomplete {stem} path/hash binding")
        if path_key in plan:
            verify_mapping({plan[path_key]: plan[hash_key]}, dependencies)
    for name in plan.get("evidence_sha256", {}):
        path = resolve_path(name)
        evidence = read_metadata(path)
        if "status" in evidence and evidence["status"].lower() not in ("pass", "complete", "completed"):
            raise ValueError(f"Frozen evidence has not passed: {path}")
        for field in ("source_sha256", "source_hashes", "reference_hashes"):
            verify_mapping(evidence.get(field, {}), dependencies)
        for field in ("files", "inputs"):
            for relative_name, digest in evidence.get(field, {}).items():
                # Geometry receipts use sibling filenames. Analysis/manifest
                # maps use paths relative to born_ai; this is an explicit
                # schema rule, not an arbitrary search for a matching file.
                parent = path.parent if Path(relative_name).parent == Path(".") else ROOT
                verify_mapping({relative_name: digest}, dependencies, parent)


def csv_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        yield from csv.DictReader(stream)


def sample_path(job):
    for key in ("samples_path", "sample_path", "output_csv", "samples_csv", "csv_path", "output"):
        if key in job:
            return resolve_path(job[key])
    raise ValueError("Plan job needs an explicit sample CSV path; input globbing is forbidden")


def integer(row, key):
    value = float(row[key])
    if not value.is_integer():
        raise ValueError(f"Noninteger {key}")
    return int(value)


def read_chain(path, job, plan):
    """Return chain x record-ready arrays without treating shifts as samples."""
    records = {}
    identity_error = 0.0
    burnin = int(job.get("warmup_sweeps", job.get("analysis_burnin",
        plan.get("warmup_sweeps", plan.get("analysis_burnin", 256)))))
    header_identity = None
    for row in csv_rows(path):
        L, beta, mu = integer(row, "L"), float(row["beta"]), float(row["mu"])
        chain, seed = integer(row, "chain"), integer(row, "seed")
        identity = (L, beta, mu, chain, seed, row["start"])
        if header_identity is None:
            header_identity = identity
        if identity != header_identity:
            raise ValueError(f"Mixed chain identity in {path}")
        if L not in SIZES or beta != 3.0 or mu != 1e-8:
            raise ValueError("Unexpected production family")
        for key, actual in (("L", L), ("beta", beta), ("mu", mu), ("chain", chain), ("seed", seed)):
            if key in job and float(job[key]) != actual:
                raise ValueError(f"Plan/sample disagreement for {key}")
        sweep = integer(row, "sweep")
        if sweep <= burnin:
            continue
        record = integer(row, "record_index")
        step = integer(row, "measure_every")
        den = integer(row, "p_den")
        if integer(row, "p_num") != 1 or den not in (2, 3):
            raise ValueError("Only the preregistered reciprocal geometries are supported")
        tx, ty = integer(row, "tx"), integer(row, "ty")
        if tx not in range(0, L, L // den) or ty not in range(0, L, L // den):
            raise ValueError("Translation is not in the prescribed subgroup")
        if integer(row, "translation_index") != 1 + tx // (L // den) + den * (ty // (L // den)):
            raise ValueError("Translation index differs from the frozen enumeration")
        if row.get("stage") != job.get("stage", plan.get("stage")) or row["start"] != job.get("start", row["start"]):
            raise ValueError("Stage or starting state differs from plan")
        vals = np.array([float(row[o]) for o in REGIONS])
        if not np.isfinite(vals).all():
            raise ValueError("Nonfinite entropy")
        error = abs(vals[3] - ((1 - 1 / den) * vals[0] + vals[2] / den - vals[1]))
        identity_error = max(identity_error, float(error))
        if error > 1e-10:
            raise ValueError("Per-record D identity failed")
        metadata = (sweep, step, row.get("accepted"), row.get("proposed"), row.get("log_weight"))
        if integer(row, "proposed") != 2 * L * L * step or not 0 <= integer(row, "accepted") <= integer(row, "proposed"):
            raise ValueError("Recorded proposal/acceptance counts do not cover exactly one measurement interval")
        entry = records.setdefault(record, dict(metadata=metadata, translations={}))
        if entry["metadata"] != metadata:
            raise ValueError("Translations in one record have different sampler metadata")
        key = (den, tx, ty)
        if key in entry["translations"]:
            raise ValueError("Duplicate geometry/translation in one record")
        entry["translations"][key] = vals
    if not records or header_identity is None:
        raise ValueError("Empty retained chain")
    L, beta, mu, chain, seed, start = header_identity
    ids = sorted(records)
    if ids != list(range(ids[0], ids[0] + len(ids))):
        raise ValueError("Missing record index")
    meta = [records[i]["metadata"] for i in ids]
    step = meta[0][1]
    sweeps = [m[0] for m in meta]
    if any(m[1] != step for m in meta) or any(b - a != step for a, b in zip(sweeps, sweeps[1:])):
        raise ValueError("Nonuniform retained measurement cadence")
    if sweeps[0] != burnin + step:
        raise ValueError("Retained observations must begin immediately after frozen burn-in")
    expected = job.get("retained_records", job.get("retained_measurements", job.get("measurements")))
    if expected is not None and len(ids) != int(expected):
        raise ValueError(f"Expected {expected} retained records, observed {len(ids)}")
    if "measure_every" in job and step != int(job["measure_every"]):
        raise ValueError("Measurement cadence differs from plan")
    arrays = {}
    for den in (3, 2):
        shifts = [(tx, ty) for ty in range(0, L, L // den) for tx in range(0, L, L // den)]
        arrays[den] = np.array([[records[r]["translations"][(den, tx, ty)]
                                for tx, ty in shifts] for r in ids])
    if any(len(records[r]["translations"]) != 13 for r in ids):
        raise ValueError("A retained record must contain exactly 9 + 4 geometry/shift rows")
    accepted = sum(int(m[2]) for m in meta) if all(m[2] is not None for m in meta) else None
    proposed = sum(int(m[3]) for m in meta) if all(m[3] is not None for m in meta) else None
    return dict(L=L, beta=beta, mu=mu, chain=chain, seed=seed, start=start,
                arrays=arrays, draws=len(ids), measure_every=step, burnin=burnin,
                first_sweep=sweeps[0], last_sweep=sweeps[-1], identity_error=identity_error,
                accepted=accepted, proposed=proposed, path=base.relative(path))


def production_inputs(plan_path, manifest_path, stage="production"):
    dependencies = {}
    plan, manifest = read_metadata(plan_path), read_metadata(manifest_path)
    if manifest.get("status", "").lower() not in ("complete", "completed", "pass"):
        raise ValueError("A complete production manifest is required")
    if plan.get("stage") != stage or manifest.get("stage") != stage:
        raise ValueError("Pilot and production data must be analyzed separately")
    if not plan.get("jobs"):
        raise ValueError("No jobs in frozen production plan")
    for path in (plan_path, manifest_path):
        remember(path, dependencies)
    for container in (manifest, plan):
        for field in ("files", "source_sha256", "source_hashes"):
            verify_mapping(container.get(field, {}), dependencies)
    verify_frozen_evidence(plan, dependencies)
    if "plan_sha256" in manifest and manifest["plan_sha256"] != base.sha256(plan_path):
        raise ValueError("Manifest binds a different plan")
    groups, seen = {}, set()
    for job in plan["jobs"]:
        path = sample_path(job)
        if base.relative(path) not in dependencies:
            raise ValueError(f"Raw input is not hash-bound by plan/manifest: {path}")
        receipt_path = resolve_path(job["receipt_path"])
        receipt = read_metadata(receipt_path)
        if (base.relative(receipt_path) not in dependencies or receipt.get("status") != "complete"
                or receipt["plan_sha256"] != base.sha256(plan_path)):
            raise ValueError("Completed chain receipt does not bind this plan")
        for drift in ("maximum_inverse_drift", "maximum_logweight_drift"):
            if not 0 <= float(receipt[drift]) < 1e-5:
                raise ValueError("Numerical drift threshold failed")
        if float(receipt["legacy_entropy_max_difference"]) >= 1e-8:
            raise ValueError("Legacy entropy identity check failed")
        if int(receipt["records"]) != int(job["sweeps"]) // int(job["measure_every"]):
            raise ValueError("Incomplete chain record count")
        chain = read_chain(path, job, plan)
        chain["receipt"] = receipt
        if chain["seed"] in seen:
            raise ValueError("Repeated production seed")
        seen.add(chain["seed"])
        groups.setdefault((chain["L"], chain["beta"], chain["mu"]), []).append(chain)
    if sorted(k[0] for k in groups) != ([6, 18, 36] if stage == "pilot" else list(SIZES)):
        raise ValueError("The complete preregistered size grid is required")
    for chains in groups.values():
        chains.sort(key=lambda c: c["chain"])
        if len(chains) != 4 or len({c["draws"] for c in chains}) != 1:
            raise ValueError("Exactly four equal-length chains are required for each size")
        if len({c["chain"] for c in chains}) != 4:
            raise ValueError("Repeated chain ID")
    return groups, plan, manifest, dependencies


def observables(chains):
    """Contrasts are formed on paired states, including their cross-geometry covariance."""
    data = {den: np.stack([c["arrays"][den] for c in chains]) for den in (3, 2)}
    result = {}
    for den, raw in data.items():
        for j, region in enumerate(REGIONS):
            result[f"p{den}_origin_{region}"] = raw[:, :, 0, j]
            result[f"p{den}_group_{region}"] = raw[:, :, :, j].mean(axis=2)
            result[f"p{den}_group_minus_origin_{region}"] = (
                result[f"p{den}_group_{region}"] - result[f"p{den}_origin_{region}"])
        for ti in range(raw.shape[2]):
            result[f"p{den}_shift{ti}_D"] = raw[:, :, ti, 3]
    for scope in ("origin", "group"):
        for region in REGIONS:
            result[f"half_minus_third_{scope}_{region}"] = (
                result[f"p2_{scope}_{region}"] - result[f"p3_{scope}_{region}"])
    return result


def summarize(groups):
    means, diagnostics, batches, chain_means, prefix, covariance, acceptance = [], [], [], [], [], [], []
    for (L, beta, mu), chains in sorted(groups.items()):
        common = dict(L=L, beta=beta, mu=mu, measure_every=chains[0]["measure_every"],
                      burnin=chains[0]["burnin"])
        values = observables(chains)
        for name, x in values.items():
            diag, block = base.error_summary(x)
            means.append(dict(**common, observable=name, mean=diag["mean"], MCSE=diag["MCSE_conservative"],
                n_chains=x.shape[0], draws_per_chain=x.shape[1], rhat_max=diag["rhat_max"],
                ESS_bulk=diag["ESS_bulk"], ESS_tail=diag["ESS_tail"],
                screen_pass=diag["initial_screen_pass"], deterministic=False))
            diagnostics.append(dict(**common, observable=name, **diag))
            batches.extend(dict(**common, observable=name, **b) for b in block)
            for ci, chain in enumerate(chains):
                chain_means.append(dict(**common, observable=name, chain=chain["chain"],
                    seed=chain["seed"], start=chain["start"], mean=float(x[ci].mean()), draws=x.shape[1]))
            if name in ("p3_group_D", "p2_group_D", "half_minus_third_group_D",
                        "p3_group_minus_origin_D", "p2_group_minus_origin_D"):
                checkpoints = sorted({n for n in (128, 256, 512, 1024, 2048, 4096) if n <= x.shape[1]} | {x.shape[1]})
                for n in checkpoints:
                    d, _ = base.error_summary(x[:, :n])
                    prefix.append(dict(**common, observable=name, retained_per_chain=n,
                        mean=d["mean"], MCSE=d["MCSE_conservative"], rhat_max=d["rhat_max"],
                        ESS_bulk=d["ESS_bulk"], ESS_tail=d["ESS_tail"],
                        first_half_mean=float(x[:, :n//2].mean()), second_half_mean=float(x[:, n//2:n].mean())))
        # Preserve whole covariance matrices at each block length; there is no
        # fictitious matrix assembled from separate maximum scalar errors.
        names = [name for name in values if "_shift" not in name]
        raw = np.stack([values[name] for name in names], axis=-1)
        for length in base.BLOCK_LENGTHS:
            nblocks = raw.shape[1] // length
            if nblocks < 2:
                continue
            keep = nblocks * length
            block = raw[:, :keep].reshape(raw.shape[0], nblocks, length, len(names)).mean(axis=2).reshape(-1, len(names))
            matrix = np.cov(block, rowvar=False, ddof=1) / len(block)
            eligible = nblocks >= 8 and (raw.shape[1] - keep) / raw.shape[1] <= .05
            for i, first in enumerate(names):
                for j, second in enumerate(names):
                    covariance.append(dict(**common, block_length=length, blocks_per_chain=nblocks,
                        eligible=eligible, discarded_per_chain=raw.shape[1]-keep,
                        first=first, second=second, covariance_mean=float(matrix[i, j])))
        accepted = sum(c["accepted"] for c in chains) if all(c["accepted"] is not None for c in chains) else None
        proposed = sum(c["proposed"] for c in chains) if all(c["proposed"] is not None for c in chains) else None
        acceptance.append(dict(**common, recorded_accepted=accepted, recorded_proposed=proposed,
            acceptance=accepted/proposed if proposed else None, records_per_chain=chains[0]["draws"],
            first_retained_sweep=chains[0]["first_sweep"], last_retained_sweep=chains[0]["last_sweep"]))
    return means, dict(diagnostics=diagnostics, block_errors=batches, chain_means=chain_means,
                       cumulative_diagnostics=prefix, covariance=covariance, acceptance=acceptance)


def baseline_inputs(path, receipt_path, dependencies):
    if receipt_path:
        receipt = read_metadata(receipt_path)
        if receipt.get("status", "").lower() not in ("pass", "complete", "completed"):
            raise ValueError("Baseline validation did not pass")
        remember(receipt_path, dependencies)
        verify_mapping(receipt.get("files", {}), dependencies, Path(receipt_path).parent)
        for field in ("source_sha256", "source_hashes", "reference_hashes"):
            verify_mapping(receipt.get(field, {}), dependencies)
    remember(path, dependencies)
    points = {}
    for row in csv_rows(path):
        L, den = integer(row, "L"), integer(row, "p_den")
        if L not in SIZES:
            continue
        if integer(row, "p_num") != 1 or den not in (2, 3) or float(row["beta"]) != 0 or float(row["mu"]) != 1e-8:
            raise ValueError("Mixed baseline family")
        if (L, den) in points:
            raise ValueError("Duplicate baseline geometry")
        vals = {o: float(row.get(o, row.get(o.replace("S", "S_")))) for o in REGIONS}
        if abs(vals["D"] - ((1-1/den)*vals["SA"]+vals["SC"]/den-vals["SB"])) > 1e-10:
            raise ValueError("Baseline D identity failed")
        points[(L, den)] = vals
    if sorted(points) != sorted((L, den) for L in SIZES for den in (2, 3)):
        raise ValueError("Incomplete baseline")
    result = []
    for L in SIZES:
        for den in (3, 2):
            for region in REGIONS:
                result.append(dict(L=L, beta=0.0, mu=1e-8, observable=f"p{den}_group_{region}",
                    mean=points[(L, den)][region], MCSE=0., deterministic=True, screen_pass=True))
        for region in REGIONS:
            result.append(dict(L=L, beta=0.0, mu=1e-8, observable=f"half_minus_third_group_{region}",
                mean=points[(L, 2)][region]-points[(L, 3)][region], MCSE=0., deterministic=True, screen_pass=True))
    return result


def fits_and_slopes(means):
    fits, slopes = [], []
    for name, terms in MODELS.items():
        base.MODEL_TERMS[f"boundary_{name}"] = terms
    fit_observables = {
        "p3_group_D": ("log", "log_invL", "linear"),
        "p2_group_D": ("log", "log_invL", "linear"),
        "half_minus_third_group_D": ("constant", "log", "invL", "log_invL"),
        "half_minus_third_group_SA": ("constant", "log", "invL"),
        "half_minus_third_group_SC": ("constant", "log", "invL"),
    }
    for beta in (0., 3.):
        for observable, models in fit_observables.items():
            selected = sorted([r for r in means if r["beta"] == beta and r["observable"] == observable], key=lambda r:r["L"])
            sizes = np.array([r["L"] for r in selected])
            values = np.array([r["mean"] for r in selected])
            errors = np.array([r["MCSE"] if r["MCSE"] is not None else np.nan for r in selected])
            for cutoff in (6, 12, 18):
                for model in models:
                    scope = dict(beta=beta, observable=observable, model=model, requested_L_min=cutoff)
                    if not np.isfinite(errors[sizes >= cutoff]).all():
                        fits.append(dict(**scope, status="sampling_uncertainty_unresolved"))
                        continue
                    if any(not row["screen_pass"] for row in selected if row["L"] >= cutoff):
                        fits.append(dict(**scope, status="sampling_screen_not_passed"))
                        continue
                    result = base.finite_size_fit(sizes, values, errors, f"boundary_{model}", L_min=cutoff, min_residual_dof=3)
                    result.pop("model", None)
                    fits.append(dict(**scope, **result))
            for i in range(len(sizes)-1):
                width = math.log(sizes[i+1]/sizes[i])
                endpoint_screens = bool(selected[i]["screen_pass"] and selected[i+1]["screen_pass"])
                slopes.append(dict(beta=beta, observable=observable, L_left=int(sizes[i]), L_right=int(sizes[i+1]),
                    L_mid=float(math.sqrt(sizes[i]*sizes[i+1])), slope=float((values[i+1]-values[i])/width),
                    MCSE=float(math.hypot(errors[i], errors[i+1])/width),
                    endpoint_sampling_screens_pass=endpoint_screens,
                    interpretation="eligible finite interval, not asymptotic proof" if endpoint_screens else
                        "endpoint sampling unresolved; descriptive value only, no scaling inference",
                    note="Different L are independent; adjacent slopes share one endpoint and are correlated"))
    return fits, slopes


def slope_covariances(means, slopes):
    """Shared endpoints generate covariance even though separate L runs are independent."""
    rows = []
    for beta, observable in sorted({(s["beta"], s["observable"]) for s in slopes}):
        selected = [s for s in slopes if s["beta"] == beta and s["observable"] == observable]
        variances = {r["L"]:r["MCSE"]**2 for r in means if r["beta"] == beta and
                     r["observable"] == observable and r["MCSE"] is not None}
        for first in selected:
            width1 = math.log(first["L_right"]/first["L_left"])
            weight1 = {first["L_left"]:-1/width1, first["L_right"]:1/width1}
            for second in selected:
                width2 = math.log(second["L_right"]/second["L_left"])
                weight2 = {second["L_left"]:-1/width2, second["L_right"]:1/width2}
                if any(L not in variances for L in weight1.keys() | weight2.keys()):
                    covariance = None
                else:
                    covariance = sum(weight1.get(L, 0)*weight2.get(L, 0)*variances[L] for L in weight1.keys() | weight2.keys())
                rows.append(dict(beta=beta, observable=observable,
                    first_left=first["L_left"], first_right=first["L_right"],
                    second_left=second["L_left"], second_right=second["L_right"],
                    covariance=covariance, error_role="Propagation of scalar conservative mean MCSEs; not model uncertainty"))
    return rows


def finite_fit(fits, beta, observable, model, cutoff=6):
    return next((f for f in fits if f["beta"] == beta and f["observable"] == observable and
                 f["model"] == model and f["requested_L_min"] == cutoff and
                 f["status"] == "finite_window_diagnostic_only"), None)


def predict(fit, grid):
    coeff = fit["coefficient"]
    return (coeff.get("intercept", 0) + coeff.get("logL", 0)*np.log(grid) +
            coeff.get("L", 0)*grid + coeff.get("invL", 0)/grid)


def precision_checks(means, plan):
    """A frozen precision target is distinct from the chain convergence screens."""
    targets = {"p3_group_D": float(plan["precision_target_D"]),
               "p2_group_D": float(plan["precision_target_D"]),
               "half_minus_third_group_D": float(plan["precision_target_paired_geometry_difference"])}
    checks = []
    for row in means:
        if row["beta"] != 3. or row["observable"] not in targets:
            continue
        target = targets[row["observable"]]
        met = bool(row["MCSE"] is not None and math.isfinite(row["MCSE"]) and row["MCSE"] <= target)
        row["precision_target"] = target
        row["precision_target_met"] = met
        row["precision_and_sampling_screen_pass"] = bool(met and row["screen_pass"])
        checks.append(dict(L=row["L"], beta=row["beta"], observable=row["observable"],
            MCSE=row["MCSE"], precision_target=target, precision_target_met=met,
            sampling_screen_pass=row["screen_pass"],
            precision_and_sampling_screen_pass=row["precision_and_sampling_screen_pass"]))
    return targets, checks


def make_figures(means, fits, slopes, tables, directory):
    plt = base.plotting()
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    colors = {3: "#246A9B", 2: "#CE742B"}
    markers = {3: dict(marker="o", mfc="white", ms=7), 2: dict(marker="s", mfc="#CE742B", ms=4)}

    def series(beta, observable):
        rows = sorted([r for r in means if r["beta"] == beta and r["observable"] == observable], key=lambda r:r["L"])
        return np.array([r["L"] for r in rows]), np.array([r["mean"] for r in rows]), np.array([r["MCSE"] for r in rows], dtype=float)

    def save(fig, name):
        for suffix in ("png", "pdf"):
            path = directory / f"{name}.{suffix}"
            fig.savefig(path, bbox_inches="tight")
            paths.append(path)
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12.4, 9), constrained_layout=True)
    grid = np.linspace(6, 36, 300)
    for column, beta in enumerate((0., 3.)):
        ax = axes[0, column]
        for den in (3, 2):
            name = f"p{den}_group_D"
            sizes, vals, errors = series(beta, name)
            ax.errorbar(np.log(sizes), vals, yerr=errors if beta else None, color=colors[den],
                        linestyle="none", capsize=3, label=fr"$p=1/{den}$", **markers[den])
            fit = finite_fit(fits, beta, name, "log_invL")
            if fit:
                ax.plot(np.log(grid), predict(fit, grid), color=colors[den], lw=1.5,
                        label=fr"log + $1/L$: $c={fit['coefficient']['logL']:.3f}$")
            plain = finite_fit(fits, beta, name, "log")
            if plain:
                ax.plot(np.log(grid), predict(plain, grid), color=colors[den], ls=":", lw=1.2)
        ax.set(xlabel=r"$\ln L$", ylabel=r"$\overline{D}_p$", title=fr"$\beta={beta:g}$: same corner angles, two shapes")
        ax.legend(fontsize=8.5)
        ax.text(.03, .04, "Solid: log + 1/L; dotted: log only\nAll-size fits; finite-window diagnostics", transform=ax.transAxes, fontsize=8.5)
        ax = axes[1, column]
        for den in (3, 2):
            selected = [r for r in slopes if r["beta"] == beta and r["observable"] == f"p{den}_group_D"]
            ax.errorbar([r["L_mid"] for r in selected],
                        [r["slope"] if r["endpoint_sampling_screens_pass"] else np.nan for r in selected],
                        yerr=[r["MCSE"] for r in selected] if beta else None, color=colors[den],
                        linestyle="-" if den == 3 else "--", capsize=3, label=fr"$p=1/{den}$", **markers[den])
            failed = [r for r in selected if not r["endpoint_sampling_screens_pass"]]
            if failed:
                ax.errorbar([r["L_mid"] for r in failed], [r["slope"] for r in failed],
                            yerr=[r["MCSE"] for r in failed], color=colors[den], marker="x", ms=8,
                            linestyle="none", capsize=3, label=fr"$p=1/{den}$: unresolved endpoints")
        ax.set(xlabel=r"$\sqrt{L_iL_{i+1}}$", ylabel=r"$\Delta D_p / \Delta\ln L$", title="Local log slopes; shared endpoints are correlated")
        ax.legend(fontsize=9)
    save(fig, "geometry_scaling")

    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.6), constrained_layout=True)
    ax = axes[0, 0]
    for beta, color, marker in ((0., "#555555", "^"), (3., "#276C9B", "o")):
        sizes, vals, errors = series(beta, "half_minus_third_group_D")
        ax.errorbar(sizes, vals, yerr=errors if beta else None, color=color, marker=marker,
                    mfc="white", capsize=3, label=fr"$\beta={beta:g}$")
        for model, style in (("constant", ":"), ("invL", "--")):
            fit = finite_fit(fits, beta, "half_minus_third_group_D", model)
            if fit:
                ax.plot(grid, predict(fit, grid), color=color, ls=style, lw=1)
    ax.set(xlabel="L", ylabel=r"$\overline{D}_{1/2}-\overline{D}_{1/3}$", title="Paired shape contrast: plateau is an asymptotic test")
    ax.text(.03, .04, "Dotted: constant; dashed: constant + 1/L\nA finite-size drift is not an uncancelled area term", transform=ax.transAxes, fontsize=8)
    ax.legend(fontsize=9)
    ax = axes[0, 1]
    for den in (3, 2):
        sizes, vals, errors = series(3., f"p{den}_group_minus_origin_D")
        ax.errorbar(sizes, vals, yerr=errors, color=colors[den], ls="none", capsize=3,
                    label=fr"$p=1/{den}$", **markers[den])
    ax.axhline(0, color="#555555", lw=1)
    ax.set(xlabel="L", ylabel=r"$D_{\rm subgroup}-D_{\rm origin}$", title=r"$\beta=3$: paired translation check (1 MCSE)")
    ax.legend(fontsize=9)
    ax = axes[1, 0]
    for region, style, color in (("SA", "-", "#276C9B"), ("SC", "--", "#CE742B")):
        sizes, vals, errors = series(3., f"half_minus_third_group_{region}")
        ax.errorbar(sizes, vals, yerr=errors, marker="o" if region == "SA" else "s", mfc="white",
                    capsize=3, color=color, ls=style, label=region)
    ax.set(xlabel="L", ylabel=r"$\overline{S}_{p=1/2}-\overline{S}_{p=1/3}$", title=r"$\beta=3$: equal-boundary smooth strips")
    ax.legend(fontsize=9)
    ax = axes[1, 1]
    for den in (3, 2):
        sizes, _, origin = series(3., f"p{den}_origin_D")
        _, _, group = series(3., f"p{den}_group_D")
        ax.plot(sizes, group/origin, color=colors[den], ls="-" if den == 3 else "--",
                label=fr"$p=1/{den}$ ({den**2} shifts)", **markers[den])
    ax.axhline(1, color="#555555", lw=1)
    ax.set(xlabel="L", ylabel="MCSE(subgroup) / MCSE(origin)", title="Precision per retained state; extra entropy cost excluded")
    ax.legend(fontsize=9)
    save(fig, "paired_boundary_checks")

    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.7), constrained_layout=True)
    ax = axes[0, 0]
    for den in (3, 2):
        for model, offset, style in (("log", -.18, ":"), ("log_invL", .18, "-")):
            selected = [f for f in fits if f["beta"] == 3. and f["observable"] == f"p{den}_group_D" and
                        f["model"] == model and f["status"] == "finite_window_diagnostic_only"]
            ax.errorbar([f["requested_L_min"]+offset+(den-2)*.15 for f in selected],
                        [f["coefficient"]["logL"] for f in selected],
                        yerr=[f["coefficient_MCSE"]["logL"] for f in selected], color=colors[den],
                        ls=style, capsize=3, label=fr"$p=1/{den}$, {model.replace('_', ' + ')}", **markers[den])
    ax.set(xlabel=r"Requested $L_{\min}$ (small display offsets)", ylabel=r"$c_{\rm fit}$", xticks=[6,12,18],
           title=r"$\beta=3$: at least three residual degrees of freedom")
    ax.text(.02, .03, "Missing fits: too few sizes; no lowered df threshold", transform=ax.transAxes, fontsize=8)
    ax.legend(fontsize=8)
    ax = axes[0, 1]
    for den in (3, 2):
        for model, style in (("log", ":"), ("log_invL", "-"), ("linear", "--")):
            selected = [f for f in fits if f["beta"] == 3. and f["observable"] == f"p{den}_group_D" and
                        f["model"] == model and f["status"] == "finite_window_diagnostic_only"]
            ax.plot([f["requested_L_min"] for f in selected], [f["reduced_chi_square"] for f in selected],
                    color=colors[den], ls=style, label=fr"$p=1/{den}$, {model.replace('_', ' + ')}", **markers[den])
    ax.set(xlabel=r"$L_{\min}$", ylabel=r"$\chi^2/\mathrm{dof}$", yscale="log", title="Residual mismatch is separate from statistical MCSE")
    ax.legend(fontsize=8)
    ax = axes[1, 0]
    for den in (3, 2):
        sizes, _, _ = series(3., f"p{den}_group_D")
        rows = sorted([r for r in means if r["beta"] == 3. and r["observable"] == f"p{den}_group_D"], key=lambda r:r["L"])
        ax.plot(sizes, [r["rhat_max"] for r in rows], color=colors[den], ls="-" if den == 3 else "--",
                label=fr"$p=1/{den}$", **markers[den])
    ax.axhline(1.01, color="#777777", ls=":", label="screen 1.01")
    ax.set(xlabel="L", ylabel=r"Maximum rank/folded $\widehat R$", title="Four independent starts; a screen is not a proof")
    ax.legend(fontsize=9)
    ax = axes[1, 1]
    maxL = max(r["L"] for r in means)
    for den in (3, 2):
        rows = [r for r in tables["cumulative_diagnostics"] if r["L"] == maxL and r["observable"] == f"p{den}_group_D"]
        final = rows[-1]["mean"]
        ax.errorbar([r["retained_per_chain"] for r in rows], [r["mean"]-final for r in rows],
                    yerr=[r["MCSE"] for r in rows], color=colors[den], ls="-" if den == 3 else "--",
                    capsize=3, label=fr"$p=1/{den}$", **markers[den])
    ax.axhline(0, color="#777777", lw=1)
    ax.set(xlabel="Retained records per chain", ylabel="Prefix mean minus final mean", xscale="log",
           title=f"L={maxL}: nested prefixes are correlated")
    ax.legend(fontsize=9)
    save(fig, "fit_and_sampling_diagnostics")
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("pilot", "production"), default="production")
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--baseline-receipt", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--figures", type=Path, default=FIG)
    args = parser.parse_args()
    args.plan = args.plan or ROOT / "data/born_scaling/boundary_check" / args.stage / "plan.toml"
    args.manifest = args.manifest or args.plan.with_name("manifest.json")
    args.output = args.output or OUT / args.stage
    if args.stage == "production" and not args.baseline:
        parser.error("Production analysis requires the independent --baseline CSV")
    groups, plan, manifest, dependencies = production_inputs(args.plan, args.manifest, args.stage)
    means, tables = summarize(groups)
    if args.stage == "production":
        for path in (args.baseline, args.baseline_receipt):
            if path is None or base.relative(path) not in dependencies:
                raise ValueError("Production baseline CSV and receipt must be bound by the frozen plan evidence")
        means += baseline_inputs(args.baseline, args.baseline_receipt, dependencies)
        fits, slopes = fits_and_slopes(means)
    else:
        fits, slopes = [], []
    targets, precision = precision_checks(means, plan)
    tables["local_slope_covariance"] = slope_covariances(means, slopes)
    tables["precision_checks"] = precision
    args.output.mkdir(parents=True, exist_ok=True)
    snapshots = ROOT / "analysis/born_scaling/provenance"
    snapshots.mkdir(parents=True, exist_ok=True)
    for source in (Path(__file__), Path(base.__file__)):
        snapshot = snapshots / f"{source.stem}_{base.sha256(source)}.py"
        if snapshot.exists():
            if base.sha256(snapshot) != base.sha256(source):
                raise ValueError("Modified content-addressed source snapshot")
        else:
            snapshot.write_bytes(source.read_bytes())
        remember(snapshot, dependencies)
    outputs = []
    for name, rows in dict(means=means, local_slopes=slopes, **tables).items():
        if not rows:
            continue
        path = args.output / f"{name}.csv"
        base.write_csv(path, rows)
        outputs.append(path)
    path = args.output / "fits.json"
    base.write_json(path, fits)
    outputs.append(path)
    paths = make_figures(means, fits, slopes, tables, args.figures) if args.stage == "production" else []
    relevant = [r for r in means if r["beta"] == 3. and r["observable"] in
                ("p3_group_D", "p2_group_D", "half_minus_third_group_D", "p3_group_SA", "p2_group_SA")]
    budget = []
    for (L, beta, mu), chains in sorted(groups.items()):
        item = dict(L=L, records_per_chain=chains[0]["draws"], measure_every=chains[0]["measure_every"],
            seconds_per_sweep=float(np.mean([c["receipt"]["elapsed_seconds"] / c["receipt"]["job"]["sweeps"] for c in chains])),
            entropy_seconds_per_record=float(np.mean([c["receipt"]["entropy_seconds"] / c["receipt"]["records"] for c in chains])),
            variance_scale_units="MCSE squared times retained records per chain; MCSE pools four chains",
            cadence_caveat="Changing measurement cadence changes autocorrelation per record; extrapolated budgets need a margin and production diagnostics")
        for name in ("p3_group_D", "p2_group_D", "half_minus_third_group_D"):
            row = next(r for r in means if r["L"] == L and r["beta"] == 3 and r["observable"] == name)
            item[f"{name}_MCSE"] = row["MCSE"]
            item[f"{name}_variance_scale"] = row["MCSE"]**2 * chains[0]["draws"] if row["MCSE"] is not None else None
        budget.append(item)
    summary = dict(status="PASS", stage=args.stage, created_utc=datetime.now(timezone.utc).isoformat(),
        role="Paired finite-size boundary-cancellation checks, not certification of asymptotic corner universality",
        interpretation_limits=[
            "Nine/four subgroup shifts are correlated measurements of one state, not independent samples.",
            "The local straight-edge bond-count identity does not remove nonlocal shape corrections.",
            "A nonconstant shape contrast over finite L does not by itself prove unequal asymptotic corner coefficients.",
            "No fit is reported with fewer than three residual degrees of freedom.",
            "Individual conservative MCSE is the maximum of ESS and eligible batch estimates; saved covariance matrices are separate block estimates.",
            "Beta=0 baseline is deterministic; it has no Monte Carlo error bars.",
        ],
        sizes=sorted(k[0] for k in groups), beta=3, n_chains_per_size=4,
        retained_records=sum(c["draws"] for chains in groups.values() for c in chains),
        raw_entropy_rows=sum(c["draws"]*13 for chains in groups.values() for c in chains),
        max_D_identity_error=max(c["identity_error"] for chains in groups.values() for c in chains),
        key_sampling_screens_pass=all(r["screen_pass"] for r in relevant),
        failed_key_screens=[dict(L=r["L"], observable=r["observable"]) for r in relevant if not r["screen_pass"]],
        precision_targets=targets, precision_checks=precision,
        all_precision_targets_met=all(row["precision_target_met"] for row in precision),
        precision_misses=[row for row in precision if not row["precision_target_met"]],
        paired_corner_coefficient_diagnostics=[{
            **fit,
            "interpretation": "Finite-window delta-c from paired per-state geometry differences. A poor fit cannot identify a universal corner-coefficient difference; even a good fit is not an asymptotic proof."
        } for fit in fits if fit["observable"] == "half_minus_third_group_D" and fit["model"] == "log_invL"],
        plan_path=base.relative(args.plan), manifest_path=base.relative(args.manifest),
        budget_evidence=budget,
        files=dependencies, outputs={base.relative(p):base.sha256(p) for p in outputs+paths})
    base.write_json(args.output / ("pilot_summary.json" if args.stage == "pilot" else "summary.json"), summary)
    print(json.dumps({key:summary[key] for key in ("status", "retained_records", "raw_entropy_rows", "key_sampling_screens_pass")}, indent=2))


if __name__ == "__main__":
    main()
