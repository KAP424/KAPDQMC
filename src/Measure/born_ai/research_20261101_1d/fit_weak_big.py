"""Weak-side ln L coefficient with the 32768-sample-per-point data."""

import json
import numpy as np
import onedim_born as ob

big = json.load(open("data_large_L.json"))
small = [r for r in json.load(open("data_fixed_big.json")) if "S2_half" in r]


def fit3(L, y, e):
    L = np.array(L, dtype=float)
    y = np.array(y)
    e = np.array(e)
    A = np.column_stack([np.log(L), np.ones(len(L)), 1.0 / L])
    W = np.diag(1.0 / e**2)
    cov = np.linalg.inv(A.T @ W @ A)
    c = cov @ (A.T @ W @ y)
    return c[0], np.sqrt(cov[0, 0]), c[2]


def gs_A(Ls):
    y = np.array([ob.renyi2(ob.chain_state(L)[0], np.arange(L)) for L in Ls])
    A = np.column_stack([np.log(np.array(Ls, float)), np.ones(len(Ls)),
                         1.0 / np.array(Ls, float)])
    return np.linalg.lstsq(A, y, rcond=None)[0][0]


print("three-parameter fit  A ln L + b + d/L   (32768 samples per small-L point)")
print(f"{'beta':>5} {'L range':>10} {'A':>9} {'err':>7} {'A/0.25':>8} {'d':>8}")
for beta in (0.25, 0.5):
    pts = sorted([r for r in small if r["beta"] == beta and r["L"] >= 21]
                 + [r for r in big if r["beta"] == beta], key=lambda r: r["L"])
    L = [p["L"] for p in pts]
    y = [p.get("S2_half", p.get("S2")) for p in pts]
    e = [p.get("S2_half_sem", p.get("sem")) for p in pts]
    a, ea, d = fit3(L, y, e)
    print(f"{beta:5.2f} {min(L):5d}-{max(L):5d} {a:9.5f} {ea:7.5f} "
          f"{a/0.25:8.4f} {d:8.3f}")
    for lmin in (27, 45, 63):
        sel = [i for i, l in enumerate(L) if l >= lmin]
        if len(sel) < 4:
            continue
        a, ea, d = fit3([L[i] for i in sel], [y[i] for i in sel], [e[i] for i in sel])
        print(f"      L>={lmin:3d}: A={a:.5f}+-{ea:.5f}  A/0.25={a/0.25:.4f}  d={d:.3f}")

print()
print("ground state on the same L sets:")
print(f"  L=21-127: A = {gs_A([21, 27, 33, 45, 63, 91, 127]):.7f}")
print(f"  L=27-127: A = {gs_A([27, 33, 45, 63, 91, 127]):.7f}")

print()
print("raw adjacent slopes, L = 45 -> 127:")
for beta in (0.25, 0.5):
    pts = sorted([r for r in big if r["beta"] == beta], key=lambda r: r["L"])
    for i in range(len(pts) - 1):
        r1, r2 = pts[i], pts[i + 1]
        s = (r2["S2"] - r1["S2"]) / np.log(r2["L"] / r1["L"])
        g = (r2["S2_gs"] - r1["S2_gs"]) / np.log(r2["L"] / r1["L"])
        print(f"  beta={beta:4.2f} L={r1['L']:3d}->{r2['L']:3d}: "
              f"measured={s:.5f}  gs={g:.5f}")
