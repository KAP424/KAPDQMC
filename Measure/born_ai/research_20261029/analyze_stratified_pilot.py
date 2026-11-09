"""Read-only pilot analysis and candidate allocation; never launches production."""
from pathlib import Path
import csv
import hashlib
import json
import math
import tomllib
import numpy as np

HERE = Path(__file__).resolve().parent
PILOT = HERE / "pilot"
OUT = HERE / "pilot_analysis"
OBS = ("SA", "SB", "SC", "D", "VA", "VB", "VC", "DV", "VA_full", "VB_full", "VC_full", "DV_full", "DV_subgroup")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def logz(a, beta):
    lw = [0.0]
    for j in range(1, a + 1):
        lw.append(lw[-1] + 2 * math.log((a - j + 1) / j) - 4 * beta)
    peak = max(lw)
    return peak + math.log(math.fsum(math.exp(v - peak) for v in lw))


def array(path, N):
    return np.fromfile(path, dtype=np.float64).reshape((N, N), order="F")


def main():
    manifest_path, plan_path = PILOT / "manifest.toml", PILOT / "plan.toml"
    manifest = tomllib.loads(manifest_path.read_text())
    plan = tomllib.loads(plan_path.read_text())
    assert manifest["status"] == "complete" and manifest["plan_sha256"] == digest(plan_path)
    sources = {"pilot/manifest.toml": digest(manifest_path), "analyze_stratified_pilot.py": digest(Path(__file__))}
    for mapping in (manifest["files"], plan["source_sha256"]):
        for name, expected in mapping.items():
            assert digest(HERE / name) == expected, name
            sources[name] = expected
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError("Preserve existing analysis")
    OUT.mkdir(exist_ok=True)
    jobs, strata, points, matrices, allocations, comparisons = [], [], [], [], [], []
    for job in plan["jobs"]:
        path = PILOT / (job["id"] + ".csv")
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == job["records"]
        a, N = job["L"]**2, 2 * job["L"]**2
        for index, row in enumerate(rows, 1):
            E, F = [[int(i) for i in row[name].split(";")] for name in ("E", "F")]
            j = int(row["j"])
            assert len(E) == len(set(E)) == len(F) == len(set(F)) == a
            assert all(1 <= i <= N for i in E + F)
            assert job["jlo"] <= j <= job["jhi"] and len(set(E) & set(F)) == j
            assert int(row["sample"]) == index and int(row["seed"]) == job["seed"]
            text = f"F:{row['F']}|E:{row['E']}|j:{j}"
            assert hashlib.sha256(text.encode()).hexdigest() == row["record_sha256"]
            assert abs((2 * float(row["SA"]) + float(row["SC"])) / 3 - float(row["SB"]) - float(row["D"])) < 1e-12
            assert abs((2 * float(row["VA"]) + float(row["VC"])) / 3 - float(row["VB"]) - float(row["DV"])) < 1e-12
        values = np.array([[float(row[name]) for name in OBS] for row in rows])
        cost = np.array([sum(float(row[name]) for name in ("sampling_seconds", "entropy_seconds", "fluctuation_seconds")) for row in rows])
        G = array(PILOT / (job["id"] + "_meanG.bin"), N)
        G2 = array(PILOT / (job["id"] + "_meanGsquare.bin"), N)
        assert np.min(G2 - G**2) >= -1e-12
        assert abs(np.trace(G) - a) < 1e-9
        entry = dict(job=job, values=values, mean=values.mean(axis=0), variance=values.var(axis=0, ddof=1),
                     G=G, G2=G2, median_cost=float(np.median(cost)), mean_cost=float(np.mean(cost)))
        jobs.append(entry)
        for oi, observable in enumerate(OBS):
            y = values[:, oi]
            absolute = np.sort(np.abs(y))
            top = max(1, math.ceil(len(y) / 10))
            strata.append(dict(L=job["L"], beta=job["beta"], z=job["z"], case=job["case"], method=job["method"],
                jlo=job["jlo"], jhi=job["jhi"], weight=job["stratum_probability"], n=len(y), observable=observable,
                mean=float(y.mean()), variance=float(y.var(ddof=1)), empirical_SE=float(y.std(ddof=1) / math.sqrt(len(y))),
                minimum=float(y.min()), median=float(np.median(y)), p95=float(np.quantile(y, .95)), maximum=float(y.max()),
                largest_draw_fraction_absolute_sum=float(absolute[-1] / absolute.sum()) if absolute.sum() else 0,
                top_tenth_fraction_absolute_sum=float(absolute[-top:].sum() / absolute.sum()) if absolute.sum() else 0,
                first_half_mean=float(y[:len(y)//2].mean()), second_half_mean=float(y[len(y)//2:].mean()),
                seconds_per_sample_median=entry["median_cost"], seconds_per_sample_mean=entry["mean_cost"],
                distinct_observed_j=len(set(int(row["j"]) for row in rows))))
    for L, case in sorted({(entry["job"]["L"], entry["job"]["case"]) for entry in jobs}):
        selected = [entry for entry in jobs if entry["job"]["L"] == L and entry["job"]["case"] == case]
        strata_jobs = [entry for entry in selected if entry["job"]["method"] == "stratified"]
        iid = next(entry for entry in selected if entry["job"]["method"] == "iid_latent")
        weights = np.array([entry["job"]["stratum_probability"] for entry in strata_jobs])
        assert abs(weights.sum() - 1) < 1e-12
        ns = np.array([entry["job"]["records"] for entry in strata_jobs])
        mu = np.array([entry["mean"] for entry in strata_jobs])
        variance = np.array([entry["variance"] for entry in strata_jobs])
        weighted_mean = weights @ mu
        mean_variance = (weights**2 / ns) @ variance
        total = int(ns.sum())
        iid_variance = weights @ (variance + (mu - weighted_mean)**2)
        beta, z = iid["job"]["beta"], iid["job"]["z"]
        for index, observable in enumerate(OBS):
            strat_se = math.sqrt(mean_variance[index])
            iid_se = math.sqrt(iid["variance"][index] / iid["job"]["records"])
            points.append(dict(L=L, beta=beta, z=z, case=case, observable=observable,
                stratified_mean=float(weighted_mean[index]), stratified_empirical_SE=strat_se,
                iid_mean=float(iid["mean"][index]), iid_empirical_SE=iid_se,
                independent_difference_z=float((weighted_mean[index] - iid["mean"][index]) / math.hypot(strat_se, iid_se)),
                iid_variance_reconstructed_from_strata=float(iid_variance[index]),
                equal_budget_variance_gain_using_strata=float(iid_variance[index] / (total * mean_variance[index])),
                equal_budget_variance_gain_using_observed_iid=float(iid["variance"][index] / (total * mean_variance[index])),
                n_stratified=total, n_iid=iid["job"]["records"]))
        a, N = L**2, 2 * L**2
        g_path = next((PILOT / "models").glob(f"*_L{L}_*_G.bin"))
        G0 = array(g_path, N)
        kappa = 1 - (1 - math.exp(-2 * beta))**2 * math.exp(logz(a - 1, beta) - logz(a, beta))
        expected = kappa * G0
        np.fill_diagonal(expected, np.diag(G0))
        meanG = sum(weight * entry["G"] for weight, entry in zip(weights, strata_jobs))
        matrix_mean_variance = sum(weight**2 * (entry["G2"] - entry["G"]**2) / (entry["job"]["records"] - 1) for weight, entry in zip(weights, strata_jobs))
        rms_sampling = float(np.sqrt(np.maximum(matrix_mean_variance, 0).sum()))
        x, y = np.repeat(np.arange(a) % L, 2), np.repeat(np.arange(a) // L, 2)
        dx, dy = (x[:, None] - x[None, :]) % L, (y[:, None] - y[None, :]) % L
        kernel = np.minimum(np.minimum(dx, L-dx), L//3) * np.minimum(np.minimum(dy, L-dy), L//3) / L**2
        D0 = float(np.sum(np.square(G0) * kernel))
        di = OBS.index("DV_full")
        matrices.append(dict(L=L, beta=beta, case=case, kappa=kappa,
            Frobenius_error_meanG=float(np.linalg.norm(meanG - expected)), Frobenius_RMS_sampling_estimate=rms_sampling,
            Frobenius_error_over_RMS=float(np.linalg.norm(meanG - expected) / rms_sampling),
            DV_initial=D0, exact_dephasing_DV_lower_bound=kappa**2 * D0,
            DV_full_stratified_mean=float(weighted_mean[di]), DV_full_empirical_SE=float(math.sqrt(mean_variance[di])),
            limitation="Entry errors are correlated. Frobenius ratio is descriptive, not a normal z score or confidence test."))
        # Candidate allocation is made from pilot only, with a 50% equal-stratum
        # defensive component. It is not a frozen production plan or a promise.
        primary = {"D": max(1e-4, .05*abs(weighted_mean[OBS.index("D")])),
                   "SA": max(2e-4, .02*abs(weighted_mean[OBS.index("SA")])),
                   "DV_full": max(1e-5, .03*abs(weighted_mean[di]))}
        score = np.sqrt(sum((weights*np.sqrt(variance[:, OBS.index(name)])/target)**2 for name, target in primary.items()))
        fraction = .1 + .5 * score / score.sum()
        required = max(float(np.sum(weights**2*variance[:, OBS.index(name)]/fraction)/target**2) for name, target in primary.items())
        counts = np.maximum(128, np.ceil(2 * required * fraction / 64).astype(int) * 64)
        allocations.append(dict(L=L, beta=beta, case=case, targets={name: float(value) for name, value in primary.items()},
            stratum_counts=counts.tolist(), total=int(counts.sum()),
            estimated_steady_seconds=float(sum(count*entry["median_cost"] for count, entry in zip(counts, strata_jobs))),
            predicted_MCSE={name:float(math.sqrt(np.sum(weights**2*variance[:,OBS.index(name)]/counts))) for name in primary},
            rule="2x pilot-derived sample budget; 50% equal-stratum defensive allocation, minimum128 each, multiples64. Pilot-only proposal requiring independent production and tail checks."))
    dump_csv(OUT / "strata.csv", strata)
    dump_csv(OUT / "means.csv", points)
    dump_csv(OUT / "dephasing_checks.csv", matrices)
    summary = dict(status="PASS", stage="pilot_only", records=sum(entry["job"]["records"] for entry in jobs),
        candidate_allocations=allocations, sources_sha256=sources,
        limitations=["64 samples per stratum do not establish tail coverage or calibrated confidence intervals.",
            "Independent draws eliminate Markov autocorrelation but not unobserved rare events within a stratum.",
            "Tail j>=4 is sampled conditionally over its full range, never omitted; finite precision still applies.",
            "Pilot findings select cost/variance budgets only, not an intended mean or scaling trend.",
            "Variance gains are empirical pilot estimates. Comparison to reconstructed iid variance is less sensitive to missed between-stratum events, but within-stratum tails remain unresolved.",
            "D and SA are origin entropies. The positive full-translation kernel applies to number variance DV_full, not to entropy D.",
            "No asymptotic scaling or corner coefficient is fitted in this pilot."],
        outputs_sha256={path.name:digest(path) for path in OUT.glob("*.csv")})
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(dict(status="PASS", records=summary["records"], cases=len(allocations),
                          proposed_records=sum(item["total"] for item in allocations),
                          proposed_steady_seconds=sum(item["estimated_steady_seconds"] for item in allocations)), indent=2))


if __name__ == "__main__":
    main()
