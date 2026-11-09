"""Slope extraction with the same 1/L correction the ground state needs.

The ground state itself has a raw local slope 0.2447 at L >= 21, which is 2%
below c/4 = 0.25; fitting  S2 = A ln L + b + d/L  recovers A = 0.25001.
So the measured data must be fitted the same way before being compared to 0.25.
"""

import json
import os
import numpy as np
import onedim_born as ob

# data_fixed_big.json (32768 samples/point) is what RESULTS_1D.md tabulates;
# data_fixed.json (8192) is the earlier scan whose beta = 1 and 1.5 values the
# document explicitly retracts.  Reading the old file here reproduced the
# retracted 1.12 and 0.84 without any warning.
_FIXED = "data_fixed_big.json"
data = [r for r in json.load(open(_FIXED if os.path.exists(_FIXED)
                                  else "data_fixed.json"))
        if "S2_half" in r]
print(f"source: {_FIXED if os.path.exists(_FIXED) else 'data_fixed.json'}")

print("ground-state calibration (deterministic, L = 9..199 step 2):")
Ls = np.arange(9, 200, 2)
Sg = np.array([ob.renyi2(ob.chain_state(int(L))[0], np.arange(int(L))) for L in Ls])
for lmin in (21, 45):
    m = Ls >= lmin
    A = np.column_stack([np.log(Ls[m]), np.ones(m.sum()), 1.0 / Ls[m]])
    coef, *_ = np.linalg.lstsq(A, Sg[m], rcond=None)
    print(f"   L>={lmin}: A = {coef[0]:.6f}   (exact c/4 = 0.25)")

print()
print("measured, three-parameter fit  S2 = A ln L + b + d/L:")
print(f"{'beta':>5} {'L_min':>6} {'A':>9} {'+-':>7} {'A/0.25':>8} {'d':>9}")
rows = []
for beta in sorted({r["beta"] for r in data}):
    pts = sorted([r for r in data if r["beta"] == beta], key=lambda r: r["L"])
    for lmin in (21, 27):
        use = [r for r in pts if r["L"] >= lmin]
        if len(use) < 5:
            continue
        x = np.array([r["L"] for r in use], dtype=float)
        y = np.array([r["S2_half"] for r in use])
        e = np.array([r["S2_half_sem"] for r in use])
        A = np.column_stack([np.log(x), np.ones(len(x)), 1.0 / x])
        W = np.diag(1.0 / e**2)
        cov = np.linalg.inv(A.T @ W @ A)
        coef = cov @ (A.T @ W @ y)
        err = np.sqrt(np.diag(cov))
        rows.append((beta, lmin, coef[0], err[0], coef[2]))
        print(f"{beta:5.2f} {lmin:6d} {coef[0]:9.4f} {err[0]:7.4f} "
              f"{coef[0] / 0.25:8.3f} {coef[2]:9.3f}")

print()
print("same fit for the GROUND STATE restricted to the same L sets:")
for lmin in (21, 27):
    use = sorted({r["L"] for r in data if r["L"] >= lmin})
    Qs = [ob.chain_state(L) for L in use]
    y = np.array([ob.renyi2(Q, np.arange(Q.shape[0] // 2)) for Q, _ in Qs])
    x = np.array(use, dtype=float)
    A = np.column_stack([np.log(x), np.ones(len(x)), 1.0 / x])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    print(f"   L>={lmin}: A = {coef[0]:.6f}")

print()
print("ratio of measured A to the ground-state A on the SAME L set:")
gsA = {}
for lmin in (21, 27):
    use = sorted({r["L"] for r in data if r["L"] >= lmin})
    y = np.array([ob.renyi2(ob.chain_state(L)[0], np.arange(L)) for L in use])
    x = np.array(use, dtype=float)
    A = np.column_stack([np.log(x), np.ones(len(x)), 1.0 / x])
    gsA[lmin] = np.linalg.lstsq(A, y, rcond=None)[0][0]
for beta, lmin, a, ea, _ in rows:
    print(f"   beta={beta:4.2f} L>={lmin}: A/gsA = {a/gsA[lmin]:.3f} +- {ea/gsA[lmin]:.3f}")
