"""Figures for RESULTS_1D.md.

(a) collapse in z = L exp(-2 beta)
(b) the same data re-cut at fixed z: the collapse is NOT L-independent
(c) fixed beta: ln L growth, extended to L = 127
(d) fitted ln L coefficient vs beta, compared with the exact CFT value c/4 = 0.25
"""

import json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import onedim_born as ob

C_CFT = 0.25

# Prefer the 32768-sample-per-point datasets when they exist; fall back to the
# original 4096/8192-sample ones.
import os

def _load(primary, fallback):
    return primary if os.path.exists(primary) else fallback

matched = [r for r in json.load(open(_load("data_matched_big.json",
                                           "data_matched.json")))
           if "S2_half" in r]
fixed = [r for r in json.load(open(_load("data_fixed_big.json",
                                         "data_fixed.json")))
         if "S2_half" in r]
large = json.load(open("data_large_L.json"))
exact9 = json.load(open("data_exact_L9.json"))

SIZES = [9, 15, 21, 27, 33, 45, 63]
cmap = plt.cm.viridis(np.linspace(0.05, 0.9, len(SIZES)))
col = {L: cmap[i] for i, L in enumerate(SIZES)}


def fit3(L, y, e):
    L = np.array(L, dtype=float)
    y = np.array(y)
    e = np.array(e)
    A = np.column_stack([np.log(L), np.ones(len(L)), 1.0 / L])
    W = np.diag(1.0 / e**2)
    cov = np.linalg.inv(A.T @ W @ A)
    c = cov @ (A.T @ W @ y)
    return c[0], np.sqrt(cov[0, 0])


fig, ax = plt.subplots(2, 2, figsize=(12.5, 9.2))

# ---------------------------------------------------------------- (a)
a = ax[0, 0]
for L in SIZES:
    pts = sorted([r for r in matched if r["L"] == L], key=lambda r: r["z"])
    a.errorbar([r["z"] for r in pts], [r["S2_half"] for r in pts],
               yerr=[r["S2_half_sem"] for r in pts], marker="o", ms=4.5,
               lw=1.1, capsize=2, color=col[L], label=f"$L={L}$")
zz = np.logspace(-2, 1.8, 100)
a.plot(zz, zz**2, "k--", lw=1.3, label=r"$z^2$ (small-$z$ law)")
a.plot(zz, 1.02 * zz**2, ":", color="gray", lw=1.0)
for L in SIZES:
    pts = [r for r in matched if r["L"] == L]
    if pts:
        a.plot([1.6e2], [pts[-1]["S2_half_gs"]], "v", color=col[L], ms=6)
a.set_xscale("log")
a.set_yscale("log")
a.set_xlabel(r"$z = L\,e^{-2\beta}$")
a.set_ylabel(r"$\overline{S_2}$  (half ring, $f=1/2$)")
a.set_title(r"(a) collapse in $z$:  $z\lesssim0.1$ only", fontsize=11)
a.legend(fontsize=7.5, ncol=2, loc="lower right")
a.grid(alpha=0.3, which="both")

# ---------------------------------------------------------------- (b)
b = ax[0, 1]
Z_SHOW = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
zcol = plt.cm.plasma(np.linspace(0.05, 0.85, len(Z_SHOW)))
def _chi2(z):
    pts = [r for r in matched if abs(r["z"] - z) < 1e-3]
    if len(pts) < 3:
        return float("nan")
    v = np.array([p["S2_half"] for p in pts])
    e = np.array([p["S2_half_sem"] for p in pts])
    w = 1 / e**2
    m = (w * v).sum() / w.sum()
    return float((w * (v - m) ** 2).sum() / (len(v) - 1))


chi2 = {z: _chi2(z) for z in Z_SHOW}
for i, z in enumerate(Z_SHOW):
    pts = sorted([r for r in matched if abs(r["z"] - z) < 1e-3],
                 key=lambda r: r["L"])
    if len(pts) < 3:
        continue
    Ls = [p["L"] for p in pts]
    b.errorbar(Ls, [p["S2_half"] / z**2 for p in pts],
               yerr=[p["S2_half_sem"] / z**2 for p in pts],
               marker="o", ms=5, lw=1.3, capsize=2, color=zcol[i],
               label=rf"$z={z:g}$   ($\chi^2_{{\rm dof}}={chi2[z]:.0f}$)")
b.axhline(1.0, color="k", ls="--", lw=1.2)
b.text(9.5, 1.04, r"theorem: $4f(1-f)+4V_0/a^2\to1$", fontsize=8)
b.set_xscale("log")
b.set_yscale("log")
b.set_ylim(4e-4, 6)
b.set_xlabel(r"$L$")
b.set_ylabel(r"$\overline{S_2}\,/\,z^2$")
b.set_title(r"(b) at fixed $z$, $\overline{S_2}$ is NOT $L$-independent", fontsize=11)
b.legend(fontsize=7.5, loc="lower left")
b.grid(alpha=0.3, which="both")

# ---------------------------------------------------------------- (c)
c = ax[1, 0]
gsl = [r for r in fixed if r["L"] == SIZES[0]][0]
Ls_all = np.array(SIZES + [91, 127])
gs_S = np.array([ob.renyi2(ob.chain_state(int(L))[0], np.arange(int(L)))
                 for L in Ls_all])
c.plot(Ls_all, gs_S, "k-", lw=2.0, label="ground state", zorder=5)
for beta, mk in ((0.25, "o"), (0.5, "s"), (1.0, "D")):
    pts = sorted([r for r in fixed if r["beta"] == beta and r["L"] >= 21],
                 key=lambda r: r["L"])
    pts += sorted([r for r in large if r["beta"] == beta], key=lambda r: r["L"])
    L = [p["L"] for p in pts]
    y = [p.get("S2_half", p.get("S2")) for p in pts]
    e = [p.get("S2_half_sem", p.get("sem")) for p in pts]
    c.errorbar(L, y, yerr=e, marker=mk, ms=5, lw=1.2, capsize=2,
               label=rf"$\beta={beta}$")
    a_, ea = fit3(L, y, e)
    xs = np.logspace(np.log10(min(L)), np.log10(max(L)), 50)
    c.plot(xs, a_ * np.log(xs) + (np.mean(np.array(y) - a_ * np.log(L))),
           ":", lw=1.0, color=c.lines[-1].get_color())
c.set_xscale("log")
c.set_xlabel(r"$L$")
c.set_ylabel(r"$\overline{S_2}$")
c.set_title(r"(c) fixed $\beta$: $\ln L$ growth persists", fontsize=11)
c.legend(fontsize=8, loc="lower right")
c.grid(alpha=0.3, which="both")

# ---------------------------------------------------------------- (d)
d = ax[1, 1]
rows = []
for beta in (0.25, 0.5):
    pts = sorted([r for r in fixed if r["beta"] == beta and r["L"] >= 21],
                 key=lambda r: r["L"])
    pts += sorted([r for r in large if r["beta"] == beta], key=lambda r: r["L"])
    L = [p["L"] for p in pts]
    y = [p.get("S2_half", p.get("S2")) for p in pts]
    e = [p.get("S2_half_sem", p.get("sem")) for p in pts]
    a_, ea = fit3(L, y, e)
    rows.append((beta, a_ / C_CFT, ea / C_CFT, True))
for beta in (1.0, 1.5, 2.0, 3.0):
    pts = sorted([r for r in fixed if r["beta"] == beta and r["L"] >= 21],
                 key=lambda r: r["L"])
    a_, ea = fit3([p["L"] for p in pts], [p["S2_half"] for p in pts],
                  [p["S2_half_sem"] for p in pts])
    rows.append((beta, a_ / C_CFT, ea / C_CFT, False))

for beta, r, er, ok in rows:
    d.errorbar([beta], [r], yerr=[er], marker="o" if ok else "o",
               mfc="C0" if ok else "white", mec="C0" if ok else "C1",
               color="C0" if ok else "C1", ms=7, capsize=3, lw=1.3)
d.axhline(1.0, color="k", lw=1.4)
d.axhspan(0.96, 1.04, color="k", alpha=0.08)
d.set_xscale("log")
d.set_xticks([0.25, 0.5, 1.0, 1.5, 2.0, 3.0])
d.set_xticklabels(["0.25", "0.5", "1", "1.5", "2", "3"])
d.minorticks_off()
d.set_xlim(0.2, 3.8)
d.set_xlabel(r"$\beta$")
d.set_ylabel(r"$A\,/\,(c/4)$   with  $c/4 = 0.25$")
d.set_title(r"(d) fitted $\ln L$ coefficient vs exact CFT value", fontsize=11)
d.text(0.27, 1.75, "filled: $z\\gg1$ (weak side)\nopen:   $z$ crossover / $z$-controlled\n        — not asymptotic",
       fontsize=8, va="top")
d.set_ylim(0, 1.85)
d.grid(alpha=0.3, which="both")

fig.tight_layout()
fig.savefig("figures/onedim_summary.png", dpi=150)
print("wrote figures/onedim_summary.png")
