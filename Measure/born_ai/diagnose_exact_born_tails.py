"""Post hoc diagnostics for fixed-design exact/MC runs; never changes samples.

The exact IID standard error uses the enumerated population variance. It is not
a Gaussian coverage guarantee when the mean is controlled by very rare events.
"""
import csv
import hashlib
import json
import math
from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "exact_born"
OBSERVABLES = ("S_A", "S_B", "S_C", "A_minus_B", "D")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    manifest_path = DATA / "exact_born_manifest.toml"
    manifest = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    for key, expected in manifest["files"].items():
        assert sha(ROOT / key) == expected, key
    summary = read_csv(DATA / "exact_born_summary.csv")
    iid_summary = {(int(r["L"]), float(r["beta"]), r["observable"]): r
                   for r in summary if r["method"] == "iid"}
    moments, tails = [], []
    threshold = 0.01
    for L in (2, 3):
        for beta in (0., .5, 1., 2., 3.):
            stem = f"exact_born_L{L}_beta{str(beta).replace('.', 'p')}"
            exact = read_csv(DATA / (stem + "_trajectories.csv"))
            probabilities = [float(row["probability"]) for row in exact]
            assert abs(math.fsum(probabilities) - 1) < 1e-11
            for obs in OBSERVABLES:
                values = [float(row[obs]) for row in exact]
                mean = math.fsum(p * x for p, x in zip(probabilities, values))
                variance = math.fsum(p * (x - mean)**2 for p, x in zip(probabilities, values))
                if max(values) - min(values) < 1e-12:
                    variance = 0.0
                iid = iid_summary[L, beta, obs]
                n = int(iid["n"])
                oracle_se = math.sqrt(variance / n)
                difference = float(iid["mean"]) - mean
                moments.append(dict(L=L, beta=beta, ensemble="fixed_half", observable=obs,
                                    exact_mean=mean, exact_variance=variance, iid_n=n,
                                    oracle_iid_SE=oracle_se, iid_mean=float(iid["mean"]),
                                    iid_empirical_SE=float(iid["SE"]),
                                    iid_empirical_z=iid["z_score"],
                                    iid_oracle_z=difference / oracle_se if oracle_se else ""))
                if L == 2 and beta == 3:
                    rare = [abs(x) > threshold for x in values]
                    rare_mass = math.fsum(p for p, flag in zip(probabilities, rare) if flag)
                    contribution = math.fsum(p*x for p, x, flag in zip(probabilities, values, rare) if flag)
                    tails.append(dict(L=L, beta=beta, observable=obs,
                                      definition="abs(observable)>0.01; post-hoc explanatory threshold",
                                      threshold=threshold, configurations=sum(rare), probability_mass=rare_mass,
                                      exact_mean_contribution=contribution, iid_n=n,
                                      iid_expected_hits=n * rare_mass,
                                      iid_zero_hit_probability=math.exp(n * math.log1p(-rare_mass)),
                                      iid_observed_hits=0, local_observed_hits=0,
                                      iid_observed_mean_contribution=0.0,
                                      local_observed_mean_contribution=0.0))
    # This is an explanation of one observed anomaly, not a selection filter.
    with (DATA / "exact_born_L2_beta3p0_mc.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            method = row["method"]
            for tail in tails:
                value = float(row[tail["observable"]])
                if abs(value) > threshold:
                    tail[method + "_observed_hits"] += 1
                    tail[method + "_observed_mean_contribution"] += value / 30_000
    moment_path = DATA / "exact_born_exact_moments.csv"
    tail_path = DATA / "exact_born_tail_cases.csv"
    write_csv(moment_path, moments)
    write_csv(tail_path, tails)
    files = [manifest_path, moment_path, tail_path,
             Path(__file__).resolve()]
    receipt = dict(status="complete", audit_status="PASS", selected_case=dict(L=2, beta=3),
                   selection_policy="Post hoc explanation of the empirical-SE anomaly; no samples, seeds, or sample counts changed.",
                   oracle_SE="sqrt(exact population variance / iid_n), only valid for independent draws; not a Gaussian coverage guarantee",
                   empirical_SE="Unchanged observed block-mean SE; absence of rare events can make it drastically underestimate true sampling variation.",
                   tail_definition="For each observable, abs(observable)>0.01. Threshold chosen after observing the L2 beta3 anomaly.",
                   files={str(p.relative_to(ROOT)).replace("\\", "/"): sha(p) for p in files},
                   tails=tails)
    (DATA / "exact_born_tail_diagnostics.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    for row in moments:
        if row["L"] == 2 and row["beta"] == 3:
            print(row)
    for row in tails:
        print(row)


if __name__ == "__main__":
    main()
