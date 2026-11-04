"""Independent, read-only production audit. No sampling or production imports.

Default execution requires the complete production manifest and final analysis.
--self-test exercises only deterministic synthetic arrays, without reading draws.
The receipt certifies reproducibility of stated statistics, not tail coverage,
asymptotic scaling, or a rigorous Monte Carlo confidence interval.
"""
from pathlib import Path
from statistics import NormalDist
from datetime import datetime, timezone
import argparse
import csv
import hashlib
import json
import math
import os
import sys
import tomllib

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis/born_scaling/production"
sys.path.insert(0, str(ROOT.parent / "ai/analysis_beta_patterns/audit_dependencies"))
import mpmath as mp

BLOCKS = (16, 32, 64, 128)
OBS = ("SA", "SB", "SC", "D", "SA_minus_SC")
TERMS = {
    "D_log": ("intercept", "logL"),
    "D_log_invL": ("intercept", "logL", "invL"),
    "D_linear_log_diagnostic": ("intercept", "L", "logL"),
    "SA_area": ("intercept", "L"),
    "SA_area_log": ("intercept", "L", "logL"),
}
MAX_ERRORS = {}
FILES = {}
CHECKS = 0


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rel(path):
    return Path(os.path.relpath(path, ROOT)).as_posix()


def bind(path, expected=None):
    path = Path(path).resolve()
    actual = digest(path)
    if expected is not None and actual != expected:
        raise AssertionError(f"SHA mismatch: {path}")
    FILES[rel(path)] = actual
    return actual


def filemap(mapping, base=ROOT):
    for name, expected in mapping.items():
        bind(base / name, expected)


def read_toml(path):
    return tomllib.loads(Path(path).read_text(encoding="utf-8"))


def rows(name):
    path = OUT / name
    bind(path)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def boolean(value):
    if isinstance(value, bool):
        return value
    if value not in ("True", "False", "true", "false"):
        raise AssertionError(f"Unknown boolean {value}")
    return value.lower() == "true"


def close(name, actual, expected, atol=2e-12, rtol=2e-10):
    global CHECKS
    CHECKS += 1
    if expected is None:
        assert actual in (None, ""), (name, actual)
        return
    assert actual not in (None, ""), (name, actual, expected)
    actual, expected = float(actual), float(expected)
    error = abs(actual - expected)
    MAX_ERRORS[name] = max(MAX_ERRORS.get(name, 0.), error)
    assert math.isfinite(actual) and error <= atol + rtol * abs(expected), (name, actual, expected, error)


def ranked_normal(x):
    """Stable-sort tie groups, independently of the analyzer's unique mapping."""
    flat = x.ravel()
    order = np.argsort(flat, kind="stable")
    sorted_x = flat[order]
    boundaries = np.r_[0, np.flatnonzero(sorted_x[1:] != sorted_x[:-1]) + 1, len(flat)]
    z = np.empty(len(flat))
    normal = NormalDist()
    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
        rank = (lo + hi + 1) / 2
        z[order[lo:hi]] = normal.inv_cdf((rank - .375) / (len(flat) + .25))
    return z.reshape(x.shape)


def variance(x):
    m, n = x.shape
    means = np.mean(x, axis=1)
    W = float(np.sum((x - means[:, None]) ** 2) / (m * (n - 1)))
    B = float(n * np.sum((means - means.mean()) ** 2) / (m - 1))
    return W, ((n - 1) * W + B) / n


def ess(x):
    m, n = x.shape
    W, V = variance(x)
    if V == 0:
        return None, V
    centered = x - x.mean(axis=1, keepdims=True)
    length = 2 ** math.ceil(math.log2(2 * n - 1))
    f = np.fft.rfft(centered, n=length, axis=1)
    ac = np.mean(np.fft.irfft(f * f.conjugate(), n=length, axis=1)[:, :n], axis=0) / n
    rho = (V - W + ac) / V
    rho[0] = 1.
    last, total = math.inf, 0.
    for i in range(0, n - 1, 2):
        pair = float(rho[i] + rho[i + 1])
        if pair <= 0:
            break
        last = min(last, pair)
        total += last
    return m * n / max(1., 2 * total - 1.), V


def rhat(x):
    W, V = variance(x)
    return math.sqrt(V / W) if W > 0 else None


def block_info(x, length):
    m, n = x.shape
    count = n // length
    b = x[:, :count * length].reshape(m, count, length).mean(axis=2)
    return dict(block_length=length, blocks_per_chain=count, total_blocks=b.size,
                discarded_end_draws_per_chain=n - count * length,
                block_mean=float(b.mean()), full_mean=float(x.mean()),
                SE=float(np.std(b, ddof=1) / math.sqrt(b.size)),
                eligible=count >= 8 and (n - count * length) / n <= .05)


def diagnostics(x):
    # No expected production observable is constant. Fail explicitly rather than
    # silently validate a numerically unresolved rare-tail case as deterministic.
    assert x.ndim == 2 and x.shape[0] == 4 and np.isfinite(x).all()
    assert np.ptp(x) > 1e-12, "Observed-constant case needs explicit unresolved audit"
    half = x.shape[1] // 2
    split = np.vstack((x[:, :half], x[:, -half:]))
    rank = ranked_normal(split)
    folded = ranked_normal(abs(split - np.median(split)))
    bulk, _ = ess(rank)
    raw, V = ess(split)
    qlo, qhi = np.quantile(split, [.05, .95])
    lower, _ = ess((split <= qlo).astype(float))
    upper, _ = ess((split <= qhi).astype(float))
    assert lower is not None and upper is not None
    blocks = [block_info(x, b) for b in BLOCKS]
    eligible = [b["SE"] for b in blocks if b["eligible"]]
    se = math.sqrt(V / raw)
    rh_rank, rh_fold = rhat(rank), rhat(folded)
    rh = max(rh_rank, rh_fold)
    return dict(mean=float(np.mean(x)), n_chains=x.shape[0], draws_per_chain=x.shape[1],
                observed_range=float(np.ptp(x)), rhat_rank=rh_rank, rhat_folded=rh_fold,
                rhat_max=rh, ESS_bulk=bulk, ESS_tail=min(lower, upper), ESS_mean=raw,
                ESS_tail_lower=lower, ESS_tail_upper=upper, MCSE_ess=se,
                MCSE_conservative=max([se] + eligible),
                tail_quantile_05=qlo, tail_quantile_95=qhi,
                eligible_block_SE_max_min_ratio=max(eligible) / min(eligible),
                initial_screen_pass=bool(rh <= 1.01 and bulk >= 400 and min(lower, upper) >= 400)), blocks


def compare_diag(row, calculated):
    for key, value in calculated.items():
        if key == "initial_screen_pass":
            assert boolean(row[key]) == value
        else:
            close(key, row[key], value)


def keyed(table, fields):
    def value(r, field):
        if field in ("L", "chain", "block_length", "draws_per_chain", "L_left", "L_right", "chain_index"):
            return int(r[field])
        if field in ("beta", "mu", "threshold", "L_min"):
            return float(r[field])
        return r[field]
    result = {tuple(value(r, k) for k in fields): r for r in table}
    assert len(result) == len(table), (fields, "duplicate table key")
    return result


def mp_fit(sizes, y, se, model):
    with mp.workdps(80):
        def term(L, name):
            return {"intercept": mp.mpf(1), "L": L, "logL": mp.log(L), "invL": 1 / L}[name]
        weighted = any(e != 0 for e in se)
        assert not weighted or all(e > 0 for e in se)
        L = [mp.mpf(str(v)) for v in sizes]
        sigma = [mp.mpf(str(e)) if weighted else mp.mpf(1) for e in se]
        X = mp.matrix([[term(l, t) for t in TERMS[model]] for l in L])
        A = mp.matrix([[X[i, j] / sigma[i] for j in range(X.cols)] for i in range(X.rows)])
        v = mp.matrix([mp.mpf(str(v)) for v in y])
        b = mp.matrix([v[i] / sigma[i] for i in range(len(y))])
        c, _ = mp.qr_solve(A, b)
        residual = v - X * c
        rss = sum((residual[i] / sigma[i]) ** 2 for i in range(len(y)))
        cov = (A.T * A) ** -1 if weighted else None
        return dict(coefficient=[float(v) for v in c], residual=[float(v) for v in residual],
                    chi_square=float(rss) if weighted else None,
                    reduced_chi_square=float(rss / (len(y) - X.cols)) if weighted else None,
                    se=[float(mp.sqrt(cov[i, i])) for i in range(X.cols)] if weighted else None,
                    rmse=float(mp.sqrt(sum(r * r for r in residual) / len(y))))


def self_test():
    # Fixed synthetic deterministic arrays, not an additional MC simulation.
    t = np.arange(256.)
    x = np.array([np.sin(t * (.217 + j * .013)) + .1 * np.cos(t * .093) for j in range(4)])
    z = ranked_normal(np.array([[1., 2., 2.], [3., 4., 5.]]))
    assert z[0, 1] == z[0, 2] and np.isfinite(z).all()
    d, blocks = diagnostics(x)
    assert 0 < d["ESS_bulk"] <= x.size and d["MCSE_conservative"] >= d["MCSE_ess"]
    assert sum(b["eligible"] for b in blocks) == 2
    L = np.arange(6, 25, 3)
    y = .7 + .155 * np.log(L) + 1.3 / L
    f = mp_fit(L, y, np.full(len(L), .002), "D_log_invL")
    assert max(abs(np.array(f["coefficient"]) - [.7, .155, 1.3])) < 1e-12
    print("PASS: deterministic self-test; production files were not read")


def main():
    manifest_path = ROOT / "data/born_scaling/production/production_manifest.toml"
    summary_path = OUT / "summary.json"
    if not manifest_path.exists() or not summary_path.exists():
        raise RuntimeError("Final manifest and analysis summary required; audit will not consume an in-progress grid")
    manifest = read_toml(manifest_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "complete" and summary["status"] == "PASS" and summary["stage"] == "production"
    assert summary["source_stages"] == ["production"]
    filemap(manifest["files"])
    filemap(summary["files"])
    bind(manifest_path); bind(summary_path)
    provenance = summary["analysis_source"]
    bind(ROOT / provenance["source_snapshot"], provenance["executed_source_sha256"])
    plan_path = manifest_path.parent / "production_plan.toml"
    bind(plan_path, manifest["plan_sha256"])
    plan = read_toml(plan_path)
    assert plan["stage"] == "production" and plan["analysis_burnin"] == 256
    assert plan["state"] == "staggered" and plan["mu"] == 1e-8 and plan["burnin_discarded"] == 0
    assert plan["update_kernel"] == "fused_rank_updates"
    assert digest(plan_path) == "3f3929b15eec47c5720b7f481240ab66f36e843a4a09761cf98f7658ae427909"
    filemap(plan["source_sha256"])
    jobs = plan["jobs"]
    assert len(jobs) == 240 and len({j["seed"] for j in jobs}) == 240
    expected_cases = {(L, b, "main") for L in range(6, 25, 3) for b in (.5, 1., 1.5, 2., 2.5, 3.)}
    expected_cases |= {(L, b, "dense") for L in (6, 12, 18) for b in (.25, .75, 1.25, 1.75, 2.25, 2.75)}
    assert {(j["L"], j["beta"], j["grid"]) for j in jobs} == expected_cases
    assert sum(j["retained_sweeps"] for j in jobs) == 958464
    assert sum(j["analysis_burnin"] for j in jobs) == 61440
    for stage in ("pilot", "pilot_extra", "smallcheck"):
        path = ROOT / f"data/born_scaling/{stage}/{stage}_plan.toml"
        if path.exists():
            old = read_toml(path); bind(path)
            assert not {j["seed"] for j in jobs} & {j["seed"] for j in old["jobs"]}
    for model in plan["models"]:
        filemap(model["files"])
        state = read_toml(ROOT / model["model_manifest"])
        assert state["Nsites"] == 2 * model["L"] ** 2 and state["Nparticles"] == model["L"] ** 2
        assert state["null_singular_count"] == 2 and state["mu"] == 1e-8
        assert state["eigenvector_residual"] < 1e-10 and state["orthogonality_error"] < 1e-10

    grouped = {}
    retained_acceptance = {}
    for job in jobs:
        run = job["run_id"]
        receipt_path = manifest_path.parent / f"{run}_receipt.toml"
        receipt = read_toml(receipt_path); bind(receipt_path)
        assert receipt["status"] == "complete" and receipt["job"] == job
        assert receipt["plan_sha256"] == digest(plan_path) and receipt["source_sha256"] == plan["source_sha256"]
        assert receipt["maximum_inverse_drift"] <= plan["max_inverse_drift_tolerance"]
        assert receipt["maximum_logweight_drift"] <= plan["max_logweight_drift_tolerance"]
        filemap(receipt["files"])
        data = []
        accepted_keep = proposed_keep = 0
        path = manifest_path.parent / f"{run}_samples.csv"
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            assert reader.fieldnames == plan["columns"]
            for index, row in enumerate(reader, 1):
                for key in ("L", "chain", "seed", "measure_every"):
                    assert int(row[key]) == job[key]
                for key in ("beta", "mu"):
                    assert float(row[key]) == job[key]
                for key in ("run_id", "stage", "start"):
                    assert row[key] == job[key]
                assert int(row["sweep"]) == index and int(row["record_index"]) == index
                assert row["phase"] == "all_recorded_no_burnin_discard" and job["measure_every"] == 1
                assert int(row["proposed"]) == 2 * job["L"] ** 2
                assert 0 <= int(row["accepted"]) <= int(row["proposed"])
                value = [float(row[k]) for k in ("SA", "SB", "SC", "D", "log_weight")]
                assert np.isfinite(value).all() and min(value[:3]) >= -1e-10
                close("per_sample_D", value[3], (2 * value[0] + value[2]) / 3 - value[1], atol=1e-11)
                if index > 256:
                    data.append(value)
                    accepted_keep += int(row["accepted"])
                    proposed_keep += int(row["proposed"])
        assert index == job["sweeps"] == 256 + job["retained_sweeps"]
        assert len(data) == job["retained_sweeps"]
        with (manifest_path.parent / f"{run}_initial.csv").open(encoding="utf-8-sig", newline="") as stream:
            initial = list(csv.DictReader(stream))
            assert len(initial) == 1 and int(initial[0]["sweep"]) == 0
        grouped.setdefault((job["L"], job["beta"]), []).append((job, np.asarray(data)))
        retained_acceptance[run] = (accepted_keep, proposed_keep)
    assert manifest["total_records"] == 1019904 and len(grouped) == 60

    means = keyed(rows("means.csv"), ("L", "beta"))
    diag_rows = keyed(rows("diagnostics.csv"), ("L", "beta", "observable"))
    block_rows = keyed(rows("block_errors.csv"), ("L", "beta", "observable", "block_length"))
    chain_rows = keyed(rows("chain_means.csv"), ("L", "beta", "observable", "chain"))
    cov_rows = keyed(rows("region_covariance.csv"), ("L", "beta", "block_length"))
    cumulative = keyed(rows("cumulative_diagnostics.csv"), ("L", "beta", "observable", "draws_per_chain"))
    tails = keyed(rows("tail_counts.csv"), ("L", "beta", "observable", "threshold", "chain_index"))
    acceptance_rows = keyed(rows("acceptance.csv"), ("L", "beta"))
    assert len(means) == 67 and len(diag_rows) == 300 and len(block_rows) == 1200 and len(chain_rows) == 1200
    assert len(cov_rows) == 240 and len(tails) == 960
    assert len(acceptance_rows) == 60
    expected_prefixes = sum(2 * sum(n <= chains[0][0]["retained_sweeps"] for n in (128,256,512,1024,2048,4096,8192)) for chains in grouped.values())
    assert len(cumulative) == expected_prefixes
    computed_means = {}
    for key, chains in sorted(grouped.items()):
        chains.sort(key=lambda pair: pair[0]["chain"])
        assert [j["start"] for j, _ in chains] == ["random", "uniform_A", "uniform_B", "02_stripe"]
        assert len({j["model_manifest_sha256"] for j, _ in chains}) == 1
        acceptance = acceptance_rows[key]
        ac = [retained_acceptance[j["run_id"]] for j, _ in chains]
        accepted, proposed = sum(a for a, _ in ac), sum(p for _, p in ac)
        assert int(acceptance["accepted"]) == accepted and int(acceptance["proposed"]) == proposed
        assert int(acceptance["burnin"]) == 256 and acceptance["scope"] == "retained_sweeps_only"
        assert int(acceptance["retained_sweeps_total"]) == sum(j["retained_sweeps"] for j, _ in chains)
        close("acceptance", acceptance["pooled_acceptance"], accepted / proposed)
        close("chain_acceptance_min", acceptance["chain_acceptance_min"], min(a / p for a, p in ac))
        close("chain_acceptance_max", acceptance["chain_acceptance_max"], max(a / p for a, p in ac))
        raw = np.stack([x for _, x in chains])
        value = np.concatenate((raw[:, :, :4], (raw[:, :, 0] - raw[:, :, 2])[:, :, None]), axis=2)
        mean_row = means[key]
        assert int(mean_row["burnin"]) == 256 and int(mean_row["draws_per_chain"]) == raw.shape[1]
        current = {}
        for oi, observable in enumerate(OBS):
            x = value[:, :, oi]
            d, blocks = diagnostics(x)
            compare_diag(diag_rows[key + (observable,)], d)
            current[observable] = d
            for field, source in (("mean", "mean"), ("MCSE", "MCSE_conservative"), ("rhat_max", "rhat_max"), ("ESS_bulk", "ESS_bulk"), ("ESS_tail", "ESS_tail")):
                close("mean_table_" + field, mean_row[f"{observable}_{field}"], d[source])
            assert boolean(mean_row[f"{observable}_screen_pass"]) == d["initial_screen_pass"]
            for b in blocks:
                target = block_rows[key + (observable, b["block_length"])]
                for field, val in b.items():
                    if field == "eligible":
                        assert boolean(target[field]) == val
                    else:
                        close("block_" + field, target[field], val)
            for ci, (job, _) in enumerate(chains):
                row = chain_rows[key + (observable, ci + 1)]
                assert int(row["seed"]) == job["seed"] and row["start"] == job["start"]
                close("chain_mean", row["mean"], x[ci].mean())
            if observable in ("SA", "D"):
                for n in (128, 256, 512, 1024, 2048, 4096, 8192):
                    if n > x.shape[1]:
                        continue
                    row = cumulative[key + (observable, n)]
                    cd, _ = diagnostics(x[:, :n]); compare_diag(row, cd)
                    first = x[:, :n // 2].mean(); second = x[:, n // 2:n].mean()
                    close("prefix_first", row["first_half_mean"], first)
                    close("prefix_second", row["second_half_mean"], second)
                    close("prefix_difference", row["half_mean_difference"], second - first)
                for threshold in (.01, .1):
                    for ci in range(4):
                        row = tails[key + (observable, threshold, ci)]
                        hits = np.flatnonzero((abs(x[ci]) if observable == "D" else x[ci]) > threshold)
                        assert int(row["hits"]) == len(hits)
                        close("first_tail_hit", row["first_hit_draw"], int(hits[0] + 1) if len(hits) else None)
                target = .003 if observable == "D" else .01
                assert boolean(mean_row[f"{observable}_absolute_precision_target_met"]) == (d["MCSE_conservative"] <= target)
        assert boolean(mean_row["sampling_screen_pass"]) == (current["SA"]["initial_screen_pass"] and current["D"]["initial_screen_pass"])
        computed_means[key] = current
        for length in BLOCKS:
            b = raw[:, :, :3].reshape(4, -1, length, 3).mean(axis=2).reshape(-1, 3)
            centered = b - b.mean(axis=0)
            cov = (centered.T @ centered) / (len(b) * (len(b) - 1))
            row = cov_rows[key + (length,)]
            for i, a in enumerate(OBS[:3]):
                for j, c in enumerate(OBS[:3]):
                    close("covariance", row[f"cov_mean_{a}_{c}"], cov[i, j], atol=1e-15)
            w = np.array([2 / 3, -1, 1 / 3])
            close("D_covariance", row["D_variance_covariance"], w @ cov @ w, atol=1e-15)
            close("D_direct_variance", row["D_variance_direct"], np.var(b @ w, ddof=1) / len(b), atol=1e-15)

    baseline_path = ROOT / "analysis/born_scaling_state_audit.csv"
    bind(baseline_path)
    with baseline_path.open(newline="") as stream:
        baseline = [r for r in csv.DictReader(stream) if r["state"] == "staggered_mu1e-8" and int(r["L"]) in range(6, 25, 3)]
    assert len(baseline) == 7
    for row in baseline:
        key = (int(row["L"]), 0.)
        for obs, source in zip(OBS, ("S_A", "S_B", "S_C", "D", "S_A_minus_S_C")):
            close("beta0_baseline", means[key][f"{obs}_mean"], float(row[source]))
            assert float(means[key][f"{obs}_MCSE"]) == 0
            assert means[key][f"{obs}_rhat_max"] == "" and boolean(means[key]["deterministic"])

    fit_rows = rows("fits.csv")
    residual_rows = keyed(rows("fit_residuals.csv"), ("beta", "model", "L_min", "L"))
    assert len(fit_rows) == 70
    assert {(float(r["beta"]), r["model"], int(float(r["L_min"]))) for r in fit_rows} == {(b, model, lo) for b in (0.,.5,1.,1.5,2.,2.5,3.) for model in TERMS for lo in (6,9)}
    assert len(residual_rows) == 455
    for row in fit_rows:
        beta, lower, model, observable = float(row["beta"]), int(float(row["L_min"])), row["model"], row["observable"]
        sizes = list(range(lower, 25, 3))
        y = [float(means[(L, beta)][f"{observable}_mean"]) for L in sizes]
        se = [float(means[(L, beta)][f"{observable}_MCSE"]) for L in sizes]
        reference = mp_fit(sizes, y, se, model)
        assert int(row["n_sizes"]) == len(sizes) and int(row["residual_dof"]) == len(sizes) - len(TERMS[model])
        assert row["fit_status"] == "finite_window_diagnostic_only"
        assert boolean(row["sampling_screen_pass"]) == all(boolean(means[(L, beta)]["sampling_screen_pass"]) for L in sizes)
        for i, term in enumerate(TERMS[model]):
            close("fit_coefficient", row[f"coefficient_{term}"], reference["coefficient"][i], atol=2e-9)
            close("fit_parameter_MCSE", row[f"MCSE_{term}"], reference["se"][i] if beta else None, atol=1e-10)
        for field in ("chi_square", "reduced_chi_square"):
            close("fit_" + field, row[field], reference[field], atol=2e-8)
        close("fit_rmse", row["residual_rmse"], reference["rmse"])
        assert boolean(row["large_fit_residual"]) == (reference["reduced_chi_square"] is not None and reference["reduced_chi_square"] > 3.)
        for i, L in enumerate(sizes):
            r = residual_rows[(beta, model, float(lower), L)]
            close("fit_residual", r["residual"], reference["residual"][i], atol=2e-11)
            close("fitted", r["fitted"], y[i] - reference["residual"][i], atol=2e-11)
    local = rows("D_local_log_slopes.csv")
    assert len(local) == 42
    for row in local:
        beta = float(row["beta"])
        left, right = int(row["L_left"]), int(row["L_right"])
        a, b = means[(left, beta)], means[(right, beta)]
        delta = math.log(right / left)
        close("local_slope", row["D_local_log_slope"], (float(b["D_mean"]) - float(a["D_mean"])) / delta)
        close("local_slope_MCSE", row["MCSE"], math.hypot(float(a["D_MCSE"]), float(b["D_MCSE"])) / delta)
    assert summary["stochastic_cases"] == 60 and summary["deterministic_cases"] == 7
    assert summary["retained_observations"] == 958464 and summary["chains"] == 240
    assert summary["all_sampling_screen_pass"] == all(boolean(r["sampling_screen_pass"]) for r in means.values())
    pooled_accepted = sum(a for a, _ in retained_acceptance.values())
    pooled_proposed = sum(p for _, p in retained_acceptance.values())
    assert summary["acceptance"]["accepted"] == pooled_accepted and summary["acceptance"]["proposed"] == pooled_proposed
    close("summary_acceptance", summary["acceptance"]["pooled_acceptance"], pooled_accepted / pooled_proposed)
    expected_precision_misses = {(L, beta, obs) for (L, beta), row in means.items() if beta > 0 for obs, target in (("SA",.01),("D",.003)) if float(row[f"{obs}_MCSE"]) > target}
    assert {(r["L"],r["beta"],r["observable"]) for r in summary["precision_misses"]} == expected_precision_misses
    residual_keys = {(float(r["beta"]),r["observable"],r["model"],int(float(r["L_min"]))) for r in fit_rows if boolean(r["large_fit_residual"])}
    assert {(r["beta"],r["observable"],r["model"],r["L_min"]) for r in summary["large_fit_residuals"]} == residual_keys
    assert {(r["beta"],r["L_min"]) for r in summary["primary_D_large_residuals"]} == {(b,lo) for b,o,m,lo in residual_keys if m == "D_log_invL" and lo == 6}

    # Freeze this independently executed validator as well as the analysis code.
    source_sha = digest(Path(__file__))
    source_snapshot = ROOT / f"analysis/born_scaling/provenance/validate_production_{source_sha}.py"
    if source_snapshot.exists():
        assert digest(source_snapshot) == source_sha
    else:
        source_snapshot.write_bytes(Path(__file__).read_bytes())
    bind(source_snapshot, source_sha)
    result = dict(status="PASS", created_utc=datetime.now(timezone.utc).isoformat(),
                  scope="Independent raw-data retention, statistics, 80-decimal QR fits, and provenance audit; no additional Monte Carlo",
                  completed_cases=60, completed_chains=240, retained_observations=958464,
                  raw_records=1019904, numeric_comparisons=CHECKS, maximum_absolute_differences=MAX_ERRORS,
                  independent_fit_decimal_precision=80, fit_count=len(fit_rows),
                  analysis_source=provenance, validator_snapshot=rel(source_snapshot),
                  limitations=["Recomputes the specified empirical diagnostics; does not prove absence of an unobserved rare tail",
                               "Fit parameter MCSE excludes finite-size/model error and uncertainty in estimated MCSE weights",
                               "No logarithmic law or universal corner coefficient is certified by this audit"], files=FILES)
    output = ROOT / "analysis/born_scaling_production_validation.json"
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("status", "completed_cases", "completed_chains", "retained_observations", "numeric_comparisons", "fit_count")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    self_test() if args.self_test else main()
