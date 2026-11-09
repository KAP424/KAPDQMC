"""Independent NumPy audit of the new geometry identity and beta-zero table.

The local-cut surrogate is not the full entropy. Fit residuals are numerical
diagnostics; beta=0 is deterministic and no Monte Carlo errors are assigned.
"""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/born_scaling/boundary_check"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def indicator(L, nx, ny, tx, ty):
    inside = np.zeros((L, L), dtype=bool)
    inside[np.ix_((np.arange(nx) + tx) % L, (np.arange(ny) + ty) % L)] = True
    # Bond labels: A(x,y)-B(x,y), A(x,y)-B(x+1,y), A(x,y)-B(x,y-1).
    outside_endpoint = (inside, np.roll(inside, -1, axis=0), np.roll(inside, 1, axis=1))
    return np.stack([
        np.stack((inside & ~other, ~inside & other), axis=-1)
        for other in outside_endpoint
    ], axis=-2).astype(np.int64)


def weighted_cuts(L, m, tx, ty):
    width = L // m
    return ((m - 1) * indicator(L, width, L, tx, ty)
            + indicator(L, L, width, tx, ty)
            - m * indicator(L, width, L-width, tx, ty))


def main():
    destination = OUT / "independent_validation.json"
    if destination.exists():
        raise FileExistsError("Preserve the existing validation receipt")
    audit_path = OUT / "boundary_audit.json"
    audit = json.loads(audit_path.read_text())
    assert audit["status"] == "PASS"
    for group in ("source_hashes", "reference_hashes"):
        for relative, expected in audit[group].items():
            assert sha(ROOT / relative) == expected, relative
    for relative, expected in audit["files"].items():
        assert sha(OUT / relative) == expected
    rows = list(csv.DictReader((OUT / "geometry_baseline.csv").open()))
    sizes = audit["sizes"]
    assert len(rows) == 2 * len(sizes)
    independent = []
    for L in sizes:
        for m in (3, 2):
            w = L // m
            total = sum(weighted_cuts(L, m, tx, ty)
                        for tx in range(0, L, w) for ty in range(0, L, w))
            assert np.count_nonzero(total) == 0
            # Exhaust all translations in a second implementation at small L.
            full = None
            if L <= 12:
                full = sum(weighted_cuts(L, m, tx, ty)
                           for tx in range(L) for ty in range(L))
                assert np.count_nonzero(full) == 0
            independent.append(dict(L=L, m=m, subgroup_max_integer_residual=int(abs(total).max()),
                                    full_max_integer_residual=None if full is None else int(abs(full).max())))
    fits = []
    differences = []
    for L in sizes:
        pair = {int(r["p_den"]):r for r in rows if int(r["L"]) == L}
        for m, row in pair.items():
            a, b, c, d = map(float, (row["S_A"],row["S_B"],row["S_C"],row["D"]))
            assert abs((1-1/m)*a+c/m-b-d) < 1e-12
            assert abs(float(row["D_cholesky"])-d) < 1e-9
            assert float(row["max_entropy_method_difference"]) < 1e-9
        differences.append(dict(L=L, D13_minus_D12=float(pair[3]["D"])-float(pair[2]["D"]),
                                SA13_minus_SA12=float(pair[3]["S_A"])-float(pair[2]["S_A"])))
    for lower in (6, 12, 18, 24):
        for m in (3,2):
            subset = [r for r in rows if int(r["p_den"]) == m and int(r["L"]) >= lower]
            x = np.array([float(r["L"]) for r in subset])
            y = np.array([float(r["D"]) for r in subset])
            for model in ("log", "log_invL"):
                columns = [np.ones_like(x),np.log(x)]
                if model == "log_invL":
                    columns.append(1/x)
                X = np.array(columns).T
                coef = np.linalg.lstsq(X,y,rcond=None)[0]
                residual = y-X@coef
                fits.append(dict(p_num=1,p_den=m,L_min=lower,L_max=max(sizes),model=model,
                                 c=float(coef[1]),intercept=float(coef[0]),
                                 inverse_L_coefficient=None if model=="log" else float(coef[2]),
                                 max_absolute_residual=float(abs(residual).max()),
                                 residual_RMS=float(np.sqrt(np.mean(residual**2))),
                                 residual_dof=len(x)-len(columns)))
    receipt = dict(status="PASS",independent_integer_checks=independent,
                   beta0_finite_window_unweighted_fits=fits,shape_differences=differences,
                   scope="No full-entropy identity is claimed. Deterministic fits have no sampling error bars and do not establish an asymptotic law by themselves.",
                   inputs={p.name:sha(p) for p in (audit_path,OUT/"geometry_baseline.csv")},
                   validator_sha256=sha(Path(__file__)))
    destination.write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(receipt,indent=2))


if __name__ == "__main__":
    main()
