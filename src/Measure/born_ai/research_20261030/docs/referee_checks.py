"""Referee-side diagnostics for the matched-z collapse claim.

Every panel answers a specific objection in CRITIQUE.md, using only the data
already in data_matched_z/ and baseline_D.json.  Nothing here refits the
physics; it tests whether the reported quantities are independent.

Usage:  D:/app/conda/python.exe referee_checks.py
"""

import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # parent: research_20261030/
FIG = os.path.join(HERE, "figures")

COL = {6: "#440154", 12: "#31688e", 18: "#35b779",
       24: "#fde725", 30: "#e6601f", 36: "#b02020"}


def load():
    base = json.load(open(os.path.join(HERE, "baseline_D.json")))
    kL = np.array(sorted(int(k) for k in base), float)
    kD = np.array([base[str(int(k))]["D"] for k in kL])

    def Dgs(L):
        return float(np.interp(np.log(L), np.log(kL), kD))

    rows = []
    for f in sorted(glob.glob(os.path.join(HERE, "data_matched_z", "*.csv"))):
        a = np.genfromtxt(f, delimiter=",", names=True)
        if a.size == 0:
            continue
        D = a["D"]
        L = int(a["L"][0])
        rows.append(dict(L=L, z=float(a["z"][0]), m=float(D.mean()),
                         s=float(D.std(ddof=1) / np.sqrt(D.size)),
                         n=int(D.size), neff=int((a["j"] >= 1).sum())))
    for r in rows:
        r["Dgs"] = Dgs(r["L"])
        r["h"] = r["m"] / r["Dgs"]
    return rows, kL, kD, Dgs


def crossing(rs, thr=0.5):
    r = sorted(rs, key=lambda q: q["z"])
    zz = np.log([q["z"] for q in r])
    hh = np.array([q["h"] for q in r])
    idx = np.where(np.diff(np.sign(hh - thr)))[0]
    if not len(idx):
        return None
    i = idx[0]
    return float(np.exp(zz[i] + (thr - hh[i]) * (zz[i + 1] - zz[i]) / (hh[i + 1] - hh[i])))


def main():
    rows, kL, kD, Dgs = load()
    byL = {}
    for r in rows:
        byL.setdefault(r["L"], []).append(r)

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.4))

    # ---------------- (a) tautology: z* is DERIVED from f and Dgs -------------
    ax = axes[0, 0]
    zs = sorted({round(r["z"], 3) for r in rows})
    fz = {z: np.mean([r["m"] for r in rows if abs(r["z"] - z) < 0.02]) for z in zs}
    lz = np.log(zs)
    fv = np.array([fz[z] for z in zs])

    Ls = np.array(sorted(byL), float)
    zmeas = np.array([crossing(byL[int(L)]) for L in Ls])
    ok = ~np.isnan(zmeas)
    zpred = np.array([np.exp(np.interp(Dgs(L) / 2, fv, lz)) for L in Ls])

    lg = np.linspace(np.log(6), np.log(36), 200)
    ax.plot(np.exp(lg), 5.403 / 30 * np.exp(lg), "k--", lw=1.0,
            label=r"$z_*=0.180\,L$ (report Sec. 2.4)")
    ax.plot(np.exp(lg), 0.342 * np.exp(lg) ** 0.806, ":", color="gray", lw=1.2,
            label=r"$z_*=0.342\,L^{0.81}$ (report's own best fit)")
    ax.plot(np.exp(lg), np.interp(lg, np.log(Ls), zpred), "-", color="C3", lw=1.4,
            label=r"$f^{-1}(D_{\rm gs}/2)$: derived, not measured")
    ax.errorbar(Ls[ok], zmeas[ok], fmt="o", ms=7, color="#2b6cb0",
                label=r"$z_*(L)$ from $h=1/2$ crossing")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$L$")
    ax.set_ylabel(r"$z_*(L)$")
    ax.set_title("(a) " + r"$z_*$ is fixed by $D_{\rm gs}(L)$ and $f(z)$ alone"
                 + "\nagreement within 9%: C3 carries no independent information")
    ax.legend(fontsize=7.2, frameon=False, loc="upper left")

    # ---------------- (b) collapse failure in u = z/z* ----------------------
    ax = axes[0, 1]
    zstar = lambda L: 0.185 * L
    for L in sorted(byL):
        sel = sorted(byL[L], key=lambda q: q["z"])
        u = [r["z"] / zstar(L) for r in sel]
        ax.errorbar(u, [r["h"] for r in sel],
                    yerr=[r["s"] / r["Dgs"] for r in sel],
                    marker="o", ms=4, lw=1.0, capsize=2,
                    color=COL.get(L, "k"), label=f"L={L}")
    ax.axhline(0.5, color="k", lw=0.7, ls=":")
    ax.axvline(1.0, color="k", lw=0.7, ls=":")
    ax.set_xscale("log")
    ax.set_ylim(-0.05, 1.15)
    ax.set_xlabel(r"$u = z/z_*(L)$")
    ax.set_ylabel(r"$\overline{D_2}/D_{\rm gs}(L)$")
    ax.set_title("(b) no collapse in $u$: at $u\\approx0.3$, $h$ runs 0.074 (L=6)\n"
                 "to 0.210 (L=24) -- a factor 2.8")
    ax.legend(fontsize=7.2, frameon=False, ncol=2)
    ax.annotate("", xy=(0.32, 0.210), xytext=(0.32, 0.074),
                arrowprops=dict(arrowstyle="<->", color="crimson", lw=1.4))
    ax.text(0.34, 0.135, "2.8x", color="crimson", fontsize=9)

    # ---------------- (c) small-z amplitude is under-sampled ----------------
    ax = axes[1, 0]
    for zt, mk in ((0.089, "o"), (0.357, "s")):
        sel = sorted([r for r in rows if abs(r["z"] - zt) < 0.02], key=lambda q: q["L"])
        Ls2 = [r["L"] for r in sel]
        A = [r["m"] / r["z"] ** 2 for r in sel]
        ax.plot(Ls2, A, mk + "-", color="C0" if zt < 0.1 else "C1",
                label=fr"$z={zt:.3f}$", ms=6)
        if zt < 0.1:
            for r, a in zip(sel, A):
                ax.annotate(f"$n_{{j\\geq1}}$={r['neff']}", (r["L"], a),
                            textcoords="offset points", xytext=(4, 4), fontsize=7)
    ax.axhline(0.18071, color="k", ls="--", lw=1.0,
               label=r"$A=0.18071$ (exact $L=3$ enumeration)")
    ax.set_xlabel(r"$L$")
    ax.set_ylabel(r"$A \equiv \overline{D_2}/z^2$")
    ax.set_title("(c) the '$z^2$ law with $A=0.18$ for every $L$' claim:\n"
                 "measured $A$ spans 0.124-0.248 on 8-29 effective samples")
    ax.legend(fontsize=7.5, frameon=False)

    # ---------------- (d) h>1 where h must equal 1 --------------------------
    ax = axes[1, 1]
    for L in sorted(byL):
        sel = sorted(byL[L], key=lambda q: q["z"])
        z = [r["z"] for r in sel]
        dh = [r["h"] - 1 for r in sel]
        er = [r["s"] / r["Dgs"] for r in sel]
        ax.errorbar(z, dh, yerr=er, marker="o", ms=4, lw=1.0, capsize=2,
                    color=COL.get(L, "k"), label=f"L={L}")
    ax.axhline(0.0, color="k", lw=1.0)
    ax.axvspan(10, 1000, color="0.9", zorder=0)
    ax.set_xscale("log")
    ax.set_xlabel(r"$z$")
    ax.set_ylabel(r"$\overline{D_2}/D_{\rm gs}(L)-1$")
    ax.set_title("(d) at $z\\to\\infty$ measurement is OFF so $h\\to1$ exactly;\n"
                 "the largest-$z$ points sit 1-3.5% high (1-2$\\sigma$)")
    ax.legend(fontsize=7.2, frameon=False, ncol=2, loc="lower right")
    ax.text(16, -0.72, "shaded: no $\\beta{=}0$ anchor\nin this dataset",
            fontsize=8, color="0.35")

    fig.tight_layout()
    out = os.path.join(FIG, "referee_checks.png")
    fig.savefig(out, dpi=200)
    fig.savefig(out.replace(".png", ".pdf"))
    print("wrote", out)

    # ------------- numbers that go with the figure -------------------------
    print("\n=== (a) tautology check ===")
    for L, a, b in zip(Ls, zmeas, zpred):
        if np.isnan(a):
            continue
        print(f"  L={int(L):2d}  z*_measured={a:7.3f}  z*_derived={b:7.3f}  ratio={b/a:.2f}")

    print("\n=== (b) collapse residual in u ===")
    rec = sorted([(r["z"] / zstar(r["L"]), r["h"], r["L"]) for r in rows])
    for u0 in (0.08, 0.16, 0.32, 0.72):
        grp = [(u, h) for (u, h, L) in rec if abs(np.log(u / u0)) < 0.25]
        if len(grp) > 1:
            hs = [h for _, h in grp]
            print(f"  u~{u0:.2f}: n={len(hs)} h in [{min(hs):.3f},{max(hs):.3f}] spread={max(hs)/max(min(hs),1e-6):.1f}x")

    print("\n=== (c) effective samples on the small-z lines ===")
    for r in sorted(rows, key=lambda q: (q["z"], q["L"])):
        if r["z"] < 0.4:
            print(f"  L={r['L']:2d} z={r['z']:.3f} n={r['n']:4d} n(j>=1)={r['neff']:4d} A={r['m']/r['z']**2:.3f}")

    print("\n=== (d) weak-measurement excess h-1 ===")
    for r in sorted(rows, key=lambda q: -q["z"])[:8]:
        print(f"  L={r['L']:2d} z={r['z']:8.2f} h-1={r['h']-1:+.4f}")


if __name__ == "__main__":
    main()
