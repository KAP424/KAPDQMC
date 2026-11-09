"""Exact finite-system variance of unbiased, known-normalizer importance sampling.

Only reads existing fixed-half trajectory tables. The observable is ALWAYS the
beta=3 postmeasurement entropy, including when sampling from another beta.
No samples, self-normalization, parameter fitting, or production edits occur.
"""
import csv
import hashlib
import json
import math
from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "exact_born"
OUT = ROOT / "analysis"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_table(L, beta, manifest, files):
    path = DATA / f"exact_born_L{L}_beta{str(beta).replace('.', 'p')}_trajectories.csv"
    rel = str(path.relative_to(ROOT)).replace("\\", "/")
    assert sha(path) == manifest["files"][rel]
    files[rel] = sha(path)
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    result = {r["E_indices"]: r for r in rows}
    assert len(result) == len(rows)
    assert abs(math.fsum(float(r["probability"]) for r in rows) - 1) < 1e-11
    return result


def main():
    manifest_path = DATA / "exact_born_manifest.toml"
    manifest = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "complete" and manifest["ensemble"] == "fixed_half"
    files = {str(manifest_path.relative_to(ROOT)).replace("\\", "/"): sha(manifest_path)}
    rows, checks = [], []
    for L in (2, 3):
        orbital_hashes = {r["ground_orbital_sha256"] for r in manifest["exact_cases"] if r["L"] == L}
        assert len(orbital_hashes) == 1, "all beta distributions must use exactly the same orbitals"
        tables = {beta: read_table(L, beta, manifest, files) for beta in (0., .5, 1., 2., 3.)}
        keys = list(tables[3.])
        assert all(set(table) == set(keys) for table in tables.values())
        target = [float(tables[3.][key]["probability"]) for key in keys]
        proposals = [("direct", 3., 1., target)]
        for beta in (0., .5, 1., 2.):
            alternative = [float(tables[beta][key]["probability"]) for key in keys]
            proposals.append(("pure", beta, 0., alternative))
            for eta in (.5, .1):
                q = [eta * p + (1 - eta) * a for p, a in zip(target, alternative)]
                proposals.append(("mixture", beta, eta, q))
        for observable in ("S_A", "D"):
            values = [float(tables[3.][key][observable]) for key in keys]
            mean = math.fsum(p * o for p, o in zip(target, values))
            direct_variance = math.fsum(p * (o - mean)**2 for p, o in zip(target, values))
            for kind, beta, eta, q in proposals:
                assert min(q) > 0
                weights = [p / prob for p, prob in zip(target, q)]
                expectation = math.fsum(prob * weight * value for prob, weight, value in zip(q, weights, values))
                variance = math.fsum(prob * (weight * value - mean)**2
                                     for prob, weight, value in zip(q, weights, values))
                second_weight_moment = math.fsum(prob * weight**2 for prob, weight in zip(q, weights))
                assert abs(expectation - mean) < 1e-13
                assert abs(math.fsum(prob * weight for prob, weight in zip(q, weights)) - 1) < 1e-11
                if kind == "mixture":
                    assert max(weights) <= 1 / eta + 1e-12
                rows.append(dict(L=L, target_beta=3., observable=observable,
                                 proposal=kind, alternative_beta=beta, eta_target=eta,
                                 configurations=len(keys), exact_target_mean=mean,
                                 estimator_expectation=expectation,
                                 single_sample_variance=variance,
                                 direct_single_sample_variance=direct_variance,
                                 variance_reduction=direct_variance / variance,
                                 theoretical_weight_ESS_fraction=1 / second_weight_moment,
                                 max_weight=max(weights), min_weight=min(weights),
                                 standard_error_at_30000=math.sqrt(variance / 30_000)))
        checks.append(dict(L=L, ground_orbital_sha256=next(iter(orbital_hashes)), configurations=len(keys)))
    csv_path = OUT / "importance_beta3_variance.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    files[str(csv_path.relative_to(ROOT)).replace("\\", "/")] = sha(csv_path)
    files[str(Path(__file__).resolve().relative_to(ROOT)).replace("\\", "/")] = sha(Path(__file__))
    receipt = dict(status="complete", audit_status="PASS", rows=len(rows), files=files, checks=checks,
                   proposal="q(E)=eta*p_beta3(E)+(1-eta)*p_beta_prime(E); eta=0 is pure proposal",
                   estimator="X(E)=p_beta3(E)/q(E) * O_beta3(E)",
                   unbiasedness="E_q[X]=sum_E q(E)*(p3(E)/q(E))*O3(E)=sum_E p3(E)*O3(E); normalized p and q are known, no self-normalization",
                   variance="sum_E q(E)*(X(E)-mu3)^2; exact enumeration, not a sample estimate",
                   weight_ESS="Asymptotic normalized weight ESS fraction = (E_q[w])^2/E_q[w^2] = 1/E_q[w^2]; not observable-specific efficiency",
                   normalizers="The existing fixed-half ensemble has analytic Z_a(beta)=sum_j binomial(a,j)^2 exp(-4 beta j) for orthonormal rank-a orbitals on 2a sites",
                   limitations="Finite L2/L3 proof of concept only; no wall-clock speed comparison, high-dimensional overlap or variance improvement is guaranteed. Per-draw entropy remains O_beta3.")
    (OUT / "importance_beta3_variance.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    for L in (2, 3):
        for observable in ("S_A", "D"):
            print(f"L={L} {observable}")
            for r in rows:
                if r["L"] == L and r["observable"] == observable:
                    print(f"  {r['proposal']:7s} beta'={r['alternative_beta']:g} eta={r['eta_target']:g} "
                          f"variance={r['single_sample_variance']:.9g} VR={r['variance_reduction']:.6g} "
                          f"weightESS={r['theoretical_weight_ESS_fraction']:.6g} maxw={r['max_weight']:.6g}")


if __name__ == "__main__":
    main()
