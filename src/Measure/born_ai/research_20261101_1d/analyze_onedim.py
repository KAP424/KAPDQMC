"""Analysis of the 1D Born-measurement scan.

Produces
  - the z-collapse table and its L-drift
  - the fixed-beta slope ratio against the exact CFT value c/4 = 0.25
    (Renyi-2 of a half-ring in a periodic c=1 chain: S2 = (c/4) ln L + const)
  - figures
"""

import json
import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SIZES = [9, 15, 21, 27, 33, 45, 63]
SLOPE_CFT = 0.25  # d S2 / d ln L for a half-ring, c = 1, Renyi-2


def load(name):
    """Load a scan, preferring the high-statistics `_big` version.

    RESULTS_1D.md quotes the 32768-sample (fixed beta) and 32768-262144-sample
    (matched z) runs, which live in data_*_big.json.  This script used to read
    the original 4096/8192-sample files, so every table it printed had error
    bars 2-16x larger than the documented ones and the beta = 1, 1.5 slope
    ratios came out as the noisy 1.12 / 0.84 the document explicitly retracts.
    make_figures.py already preferred the _big files; this matches it.
    """
    big = name.replace(".json", "_big.json")
    path = big if os.path.exists(big) else name
    print(f"[load] {path}")
    with open(path) as fh:
        return [r for r in json.load(fh) if "S2_half" in r]


def wls_slope(x, y, yerr):
    w = 1.0 / yerr**2
    S0, S1, S2 = w.sum(), (w * x).sum(), (w * x * x).sum()
    T0, T1 = (w * y).sum(), (w * x * y).sum()
    det = S0 * S2 - S1 * S1
    a = (S0 * T1 - S1 * T0) / det
    return a


# ---------------------------------------------------------------- collapse
m = load("data_matched.json")
zs = sorted({round(r["z"], 6) for r in m})
print("=" * 78)
print("1D, half-ring S2, fixed z = L exp(-2 beta)")
print(f"{'z':>8} " + "".join(f"{'L=' + str(L):>10}" for L in SIZES))
for z in zs:
    line = f"{z:8.3f} "
    for L in SIZES:
        sel = [r for r in m if abs(r["z"] - z) < 1e-6 and r["L"] == L]
        line += f"{sel[0]['S2_half']:10.4f}" if sel else f"{'--':>10}"
    print(line)

print()
print("S2 / z^2   (theorem: -> 4f(1-f) + 4V0/a^2, f = 1/2)")
print(f"{'z':>8} " + "".join(f"{'L=' + str(L):>10}" for L in SIZES))
for z in zs:
    line = f"{z:8.3f} "
    for L in SIZES:
        sel = [r for r in m if abs(r["z"] - z) < 1e-6 and r["L"] == L]
        line += f"{sel[0]['S2_half'] / z**2:10.4f}" if sel else f"{'--':>10}"
    print(line)

print()
print("L-drift of the collapse:  S2(z,L) = f(z)[1 + g ln(L/21)],  z <= 3")
sub = [r for r in m if r["z"] <= 3.0]
rows = []
for z in sorted({round(r["z"], 6) for r in sub}):
    pts = [r for r in sub if abs(r["z"] - z) < 1e-6]
    if len(pts) < 4:
        continue
    ref = np.mean([r["S2_half"] for r in pts])
    g = wls_slope(np.log(np.array([r["L"] for r in pts]) / 21.0),
                  np.array([r["S2_half"] / ref - 1 for r in pts]),
                  np.array([r["S2_half_sem"] for r in pts]) / ref)
    rows.append((z, ref, g))
    print(f"  z={z:7.3f}  f(z)={ref:8.5f}   g={g:+.4f}")
print(f"  mean g = {np.mean([g for _, _, g in rows]):+.4f}")

# ---------------------------------------------------------------- weak side
f = load("data_fixed.json")
print()
print("=" * 78)
print("1D, fixed beta:  S2(L) vs ground state, slope ratio to c/4 = 0.25")
print(f"{'beta':>6} {'L_min':>6} {'slope':>9} {'+-':>7} {'ratio':>8} {'gs_slope':>9}")
out = []
for beta in sorted({r["beta"] for r in f}):
    pts = sorted([r for r in f if r["beta"] == beta], key=lambda r: r["L"])
    if len(pts) < 4:
        continue
    for lmin in (15, 21, 27):
        use = [r for r in pts if r["L"] >= lmin]
        if len(use) < 4:
            continue
        x = np.log([r["L"] for r in use])
        y = np.array([r["S2_half"] for r in use])
        e = np.array([r["S2_half_sem"] for r in use])
        w = 1.0 / e**2
        S0, S1, S2 = w.sum(), (w * x).sum(), (w * x * x).sum()
        T0, T1 = (w * y).sum(), (w * x * y).sum()
        det = S0 * S2 - S1 * S1
        slope = (S0 * T1 - S1 * T0) / det
        eslope = np.sqrt(S0 / det)
        gs = np.array([r["S2_half_gs"] for r in use])
        gw = 1.0 / np.ones_like(gs)
        G0, G1, G2 = gw.sum(), (gw * x).sum(), (gw * x * x).sum()
        U0, U1 = (gw * gs).sum(), (gw * x * gs).sum()
        gd = G0 * G2 - G1 * G1
        gslope = (G0 * U1 - G1 * U0) / gd
        out.append((beta, lmin, slope, eslope, slope / gslope, gslope))
        print(f"{beta:6.2f} {lmin:6d} {slope:9.4f} {eslope:7.4f} "
              f"{slope / gslope:8.3f} {gslope:9.4f}")

print()
print("combined over beta in {0.5, 1.0, 1.5}, L >= 21")
print("  NOTE: this is the RAW-slope ratio (measured / ground state on the same")
print("  window), with no 1/L correction.  The ground state itself is 2% below")
print("  c/4 at these sizes, so this is NOT the A/(c/4) of RESULTS_1D.md -- that")
print("  needs the three-parameter fit in fit_weak_big.py / slope_fit.py.  It is")
print("  also mixes asymptotic (0.5) with crossover (1.0, 1.5) betas, which is")
print("  why the pooled number sits above 1.")
sel = [o for o in out if o[1] == 21 and o[0] in (0.5, 1.0, 1.5)]
w = np.array([1 / o[3] ** 2 for o in sel])
r = np.array([o[4] for o in sel])
print(f"  ratio = {np.sum(w * r) / w.sum():.4f} +- {1 / np.sqrt(w.sum()):.4f}")
for o in sel:
    print(f"    beta={o[0]:4.2f}: {o[4]:.4f}")
print(f"  (ground-state raw slope on the same window: {sel[0][5]:.4f}, CFT c/4 = {SLOPE_CFT})")

# ---------------------------------------------------------------- figure
fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.8))
colors = plt.cm.viridis(np.linspace(0, 0.9, len(SIZES)))

a = ax[0]
for i, L in enumerate(SIZES):
    pts = sorted([r for r in m if r["L"] == L], key=lambda r: r["z"])
    if not pts:
        continue
    a.errorbar([r["z"] for r in pts], [r["S2_half"] for r in pts],
               yerr=[r["S2_half_sem"] for r in pts], marker="o", ms=4,
               lw=1.0, capsize=2, color=colors[i], label=f"L={L}")
zz = np.logspace(-2, 1.6, 100)
a.plot(zz, zz**2, "k--", lw=1.2, label=r"$z^2$ (small-$z$ law)")
for L in SIZES:
    pts = [r for r in m if r["L"] == L]
    if pts:
        a.plot([150], [pts[-1]["S2_half_gs"]], "v", color="k", ms=6, alpha=0.7)
a.set_xscale("log")
a.set_yscale("log")
a.set_xlabel(r"$z = L\,e^{-2\beta}$")
a.set_ylabel(r"$\overline{S_2}$ (half ring)")
a.set_title("(a) collapse in $z$")
a.legend(fontsize=7, ncol=2)
a.grid(alpha=0.3, which="both")

b = ax[1]
for i, L in enumerate(SIZES):
    pts = sorted([r for r in f if r["L"] == L], key=lambda r: r["beta"])
    if not pts:
        continue
    b.errorbar([r["beta"] for r in pts], [r["S2_half"] for r in pts],
               yerr=[r["S2_half_sem"] for r in pts], marker="o", ms=4,
               lw=1.0, capsize=2, color=colors[i], label=f"L={L}")
gs_pts = sorted([(r["L"], r["S2_half_gs"]) for r in f])
b.plot([p[0] for p in gs_pts], [p[1] for p in gs_pts], "k-", lw=1.4,
       label="ground state")
b.set_xlabel(r"$\beta$")
b.set_ylabel(r"$\overline{S_2}$ (half ring)")
b.set_title(r"(b) fixed $\beta$")
b.legend(fontsize=7, ncol=2)
b.grid(alpha=0.3)

c = ax[2]
for beta in sorted({r["beta"] for r in f}):
    pts = sorted([r for r in f if r["beta"] == beta], key=lambda r: r["L"])
    if len(pts) < 3:
        continue
    c.plot([r["L"] for r in pts], [r["S2_half"] for r in pts], "o-", ms=4,
           lw=1.0, label=fr"$\beta={beta}$")
c.plot([p[0] for p in gs_pts], [p[1] for p in gs_pts], "k-", lw=1.6,
       label="ground state")
c.set_xscale("log")
c.set_xlabel(r"$L$")
c.set_ylabel(r"$\overline{S_2}$")
c.set_title(r"(c) $\ln L$ growth at fixed $\beta$")
c.legend(fontsize=7)
c.grid(alpha=0.3, which="both")

fig.tight_layout()
# Not onedim_summary.png: that is make_figures.py's four-panel figure, which
# RESULTS_1D.md embeds.  Writing the same name here silently replaced it with
# this three-panel diagnostic whenever this script was run on its own.
fig.savefig("figures/onedim_analysis.png", dpi=150)
print("\nwrote figures/onedim_analysis.png")
