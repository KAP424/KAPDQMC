"""
Structure of the crossover, after the threshold-dependence finding.

An adversarial check showed that the 'crossover exponent' p of z_h(L) ~ L^p
depends on the threshold h used to define it (0.59 ... 1.19 for h = 0.3 ... 0.8),
so there is no single crossover scale z*(L).  This script asks the two questions
that remain well posed, using only measured points (no cross-L interpolation):

  1. Where does each level h = D_bar/D_gs sit, per L, and how well is that
     crossing actually bracketed by data?  (A crossing interpolated across a
     factor 5-7 gap in z is not a measurement.)
  2. At FIXED beta -- the physical question -- how does D_bar grow with ln L
     at the largest sizes, and is the deficit Delta = D_gs - D_bar a function
     of beta alone (additive: log coefficient preserved) or proportional to
     D_gs (multiplicative: coefficient renormalised)?

Duplicates: (L=18, beta=1.0) exists in both data_matched_z/ and
data_matched_largez/ as independent runs; samples are POOLED per (L, beta).

Usage: D:/app/conda/python.exe crossover_structure.py
"""

import glob
import json
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
RNG = np.random.default_rng(20261031)
NBOOT = 2000
MAX_BRACKET = 1.25          # max ln-z width of a bracket counted as measured


def read_csv(path):
    """Read one data CSV, or return None if it holds no samples.

    np.genfromtxt *raises* IndexError on a zero-byte file (it does not return an
    empty array), so a shard that is still being written -- or a run that died
    before its first flush -- used to abort the whole analysis.  A header-only
    file returns a zero-length array with the right dtype, which the callers
    also have to skip.  Both cases mean "no samples here".
    """
    try:
        a = np.genfromtxt(path, delimiter=",", names=True)
    except IndexError:
        return None
    if a.size == 0 or a.dtype.names is None:
        return None
    return a


def load_points():
    base = json.load(open(os.path.join(HERE, "baseline_D.json")))
    pool = defaultdict(list)
    zval = {}
    for d in ("data_matched_z", "data_matched_largez", "data_gapfill"):
        for f in sorted(glob.glob(os.path.join(HERE, d, "*.csv"))):
            a = read_csv(f)
            if a is None:
                continue
            key = (int(a["L"][0]), round(float(a["beta"][0]), 4))
            pool[key].append(np.asarray(a["D"], dtype=float))
            zval[key] = float(a["z"][0])
    pts = []
    for (L, b), arrs in sorted(pool.items()):
        D = np.concatenate(arrs)
        pts.append(dict(L=L, beta=b, z=zval[(L, b)], D=D, n=D.size,
                        nfiles=len(arrs), Dgs=float(base[str(L)]["D"]),
                        mean=float(D.mean()),
                        sem=float(D.std(ddof=1) / np.sqrt(D.size))))
    return pts, base


def crossing(zs, hs, level):
    """First upward crossing of `level`, linear in ln z; returns (z, width)."""
    lz = np.log(zs)
    for i in range(len(hs) - 1):
        if hs[i] < level <= hs[i + 1]:
            w = lz[i + 1] - lz[i]
            x = lz[i] + (level - hs[i]) * w / (hs[i + 1] - hs[i])
            return float(np.exp(x)), float(w)
    return None, None


def per_L(pts):
    out = defaultdict(list)
    for p in pts:
        out[p["L"]].append(p)
    for L in out:
        out[L].sort(key=lambda p: p["z"])
    return out


def main():
    pts, base = load_points()
    byL = per_L(pts)
    dup = [p for p in pts if p["nfiles"] > 1]
    print(f"{len(pts)} distinct (L,beta) points; pooled duplicates: "
          + ", ".join(f"L={p['L']} beta={p['beta']} ({p['nfiles']} files, n={p['n']})"
                      for p in dup))

    # ---------------- 1. level crossings with bracket widths ----------------
    levels = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    print("\n" + "=" * 78)
    print("1. LEVEL CROSSINGS z_h(L)   [width = ln-z gap of the bracketing points]")
    print("=" * 78)
    zh = {h: {} for h in levels}
    for h in levels:
        row = []
        for L in sorted(byL):
            zs = np.array([p["z"] for p in byL[L]])
            hs = np.array([p["mean"] / p["Dgs"] for p in byL[L]])
            z, w = crossing(zs, hs, h)
            if z is None:
                row.append(f"L{L}: --")
                continue
            ok = w <= MAX_BRACKET
            zh[h][L] = (z, w, ok)
            row.append(f"L{L}: {z:6.2f}{'' if ok else '*'}")
        print(f"  h={h:.1f}  " + "  ".join(row))
    print("  (* = bracket wider than ln z = %.2f, i.e. interpolated, not measured)"
          % MAX_BRACKET)

    # bootstrap exponents using only well-bracketed crossings
    print("\n  exponent p(h) from well-bracketed crossings, bootstrap over samples:")
    boot_means = []
    for _ in range(NBOOT):
        bm = {}
        for p in pts:
            idx = RNG.integers(0, p["n"], p["n"])
            bm[(p["L"], p["beta"])] = p["D"][idx].mean()
        boot_means.append(bm)
    p_of_h = {}
    for h in levels:
        Ls = sorted(L for L, (z, w, ok) in zh[h].items() if ok)
        if len(Ls) < 3:
            print(f"    h={h:.1f}: only {len(Ls)} well-bracketed sizes {Ls} -> no fit")
            continue
        x = np.log(Ls)
        y = np.log([zh[h][L][0] for L in Ls])
        p0 = np.polyfit(x, y, 1)[0]
        ps = []
        for bm in boot_means:
            yy = []
            for L in Ls:
                zs = np.array([q["z"] for q in byL[L]])
                hs = np.array([bm[(q["L"], q["beta"])] / q["Dgs"] for q in byL[L]])
                z, _ = crossing(zs, hs, h)
                yy.append(np.log(z) if z else np.nan)
            if np.all(np.isfinite(yy)):
                ps.append(np.polyfit(x, yy, 1)[0])
        ps = np.array(ps)
        p_of_h[h] = (p0, ps.std(), Ls)
        print(f"    h={h:.1f}: p = {p0:.3f} +- {ps.std():.3f}   sizes {Ls}")

    # ---------------- 2. fixed beta ----------------
    print("\n" + "=" * 78)
    print("2. FIXED BETA: h(L), deficit Delta = D_gs - D_bar, and ln L slope")
    print("=" * 78)
    betas = [0.5, 1.0, 1.5, 2.0, 3.0]
    fixed = {}
    for b in betas:
        sel = sorted([p for p in pts if abs(p["beta"] - b) < 2e-3], key=lambda p: p["L"])
        if not sel:
            continue
        fixed[b] = sel
        print(f"\n  beta = {b}")
        print(f"    {'L':>3} {'z':>8} {'D_bar':>8} {'sem':>7} {'D_gs':>8} "
              f"{'h':>7} {'Delta':>8}")
        for p in sel:
            print(f"    {p['L']:3d} {p['z']:8.2f} {p['mean']:8.4f} {p['sem']:7.4f} "
                  f"{p['Dgs']:8.4f} {p['mean']/p['Dgs']:7.3f} {p['Dgs']-p['mean']:+8.4f}")
        big = [p for p in sel if p["L"] >= 24]
        if len(big) >= 2:
            x = np.log([p["L"] for p in big])
            y = np.array([p["mean"] for p in big])
            s = np.array([p["sem"] for p in big])
            W = np.diag(1 / s ** 2)
            X = np.column_stack([x, np.ones_like(x)])
            cov = np.linalg.inv(X.T @ W @ X)
            c = cov @ X.T @ W @ y
            ygs = np.array([p["Dgs"] for p in big])
            cgs = np.polyfit(x, ygs, 1)[0]
            print(f"    L>=24 slope dD_bar/dlnL = {c[0]:.3f} +- {np.sqrt(cov[0,0]):.3f}"
                  f"   (ground state over same L: {cgs:.3f};"
                  f"  ratio {c[0]/cgs:.2f} +- {np.sqrt(cov[0,0])/cgs:.2f})")
            # additive vs multiplicative deficit
            dl = ygs - y
            w = 1 / s ** 2
            add = np.sum(w * dl) / np.sum(w)
            chi_add = np.sum(w * (dl - add) ** 2)
            k = np.sum(w * dl * ygs) / np.sum(w * ygs ** 2)
            chi_mul = np.sum(w * (dl - k * ygs) ** 2)
            print(f"    deficit: additive const {add:+.4f} (chi2 {chi_add:.2f}),"
                  f"  multiplicative {k:.3f}*D_gs (chi2 {chi_mul:.2f}),"
                  f"  dof {len(big)-1}")

    # ---------------- 3. edge of the strong-side collapse ----------------
    print("\n" + "=" * 78)
    print("3. DOES D_bar = f(z) STILL HOLD AT z = 5.93 AND 10.55?  (matched z)")
    print("=" * 78)
    for zt in (5.9343, 10.5498):
        sel = sorted([p for p in pts if abs(p["z"] - zt) / zt < 2e-3], key=lambda p: p["L"])
        if len(sel) < 2:
            continue
        ref = sel[0]
        print(f"\n  z = {zt}:   reference L={ref['L']}  D_bar = {ref['mean']:.4f} +- {ref['sem']:.4f}")
        for p in sel:
            pull = (p["mean"] - ref["mean"]) / np.hypot(p["sem"], ref["sem"]) if p is not ref else 0.0
            print(f"    L={p['L']:2d} beta={p['beta']:.4f}  D_bar = {p['mean']:.4f} +- {p['sem']:.4f}"
                  f"   h = {p['mean']/p['Dgs']:.3f}   vs ref: {pull:+.2f} sigma")

    # ---------------- figure ----------------
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 8.4))
    cmap = plt.get_cmap("viridis")

    ax = axes[0, 0]
    for k, h in enumerate([0.3, 0.5, 0.7, 0.9]):
        col = cmap(k / 3.2)
        Ls = sorted(zh[h])
        for L in Ls:
            z, w, ok = zh[h][L]
            ax.plot(L, z, "o" if ok else "x", color=col, ms=6 if ok else 7,
                    mfc=col if ok else "none")
        ok_L = [L for L in Ls if zh[h][L][2]]
        if len(ok_L) >= 2:
            ax.plot(ok_L, [zh[h][L][0] for L in ok_L], "-", color=col, lw=1.1,
                    label=rf"$h={h}$")
    Lr = np.array([6, 40])
    ax.plot(Lr, 0.03 * Lr ** 2, "k--", lw=0.9, label=r"slope 2 (fixed $\beta$)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("L"); ax.set_ylabel(r"$z_h(L)$")
    ax.set_title("(a) level crossings; x = interpolated across a wide gap")
    ax.legend(fontsize=7, frameon=False)

    ax = axes[0, 1]
    hs = sorted(p_of_h)
    ax.errorbar(hs, [p_of_h[h][0] for h in hs], yerr=[p_of_h[h][1] for h in hs],
                marker="o", ms=5, lw=1.2, capsize=3, color="k")
    ax.axhline(2.0, color="crimson", ls="--", lw=1.0,
               label=r"$p=2$: rate of $z$ at fixed $\beta$")
    ax.axhline(1.0, color="0.6", ls=":", lw=0.9)
    ax.set_xlabel(r"threshold $h$ defining $z_h$")
    ax.set_ylabel(r"exponent $p$ in $z_h \propto L^p$")
    ax.set_title("(b) no single crossover exponent")
    ax.legend(fontsize=7, frameon=False, loc="upper left")

    ax = axes[1, 0]
    for k, b in enumerate([0.5, 1.0, 1.5, 2.0, 3.0]):
        if b not in fixed:
            continue
        sel = fixed[b]
        ax.errorbar([p["L"] for p in sel], [p["mean"] / p["Dgs"] for p in sel],
                    yerr=[p["sem"] / p["Dgs"] for p in sel], marker="o", ms=4,
                    lw=1.1, capsize=2, color=cmap(k / 4.5), label=rf"$\beta={b}$")
    ax.axhline(1.0, color="k", lw=0.7, ls=":")
    ax.set_xlabel("L"); ax.set_ylabel(r"$\overline{D_2}/D_{\rm gs}(L)$")
    ax.set_title(r"(c) fixed $\beta$: plateaus at $\beta=1.5, 2$ for $L\geq 24$")
    ax.legend(fontsize=7, frameon=False)

    ax = axes[1, 1]
    Lg = np.array(sorted(int(k) for k in base))
    ax.plot(np.log(Lg), [base[str(L)]["D"] for L in Lg], "k-", lw=1.4,
            label=r"$D_{\rm gs}(L)$")
    for k, b in enumerate([0.5, 1.0, 1.5, 2.0, 3.0]):
        if b not in fixed:
            continue
        sel = fixed[b]
        ax.errorbar(np.log([p["L"] for p in sel]), [p["mean"] for p in sel],
                    yerr=[p["sem"] for p in sel], marker="o", ms=4, lw=1.0,
                    capsize=2, color=cmap(k / 4.5), label=rf"$\beta={b}$")
    ax.set_xlim(np.log(5), np.log(62))
    ax.set_xlabel(r"$\ln L$"); ax.set_ylabel(r"$\overline{D_2}$")
    ax.set_title(r"(d) fixed $\beta$ vs the ground state")
    ax.legend(fontsize=7, frameon=False, loc="upper left")

    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "crossover_structure.png"), dpi=200)
    fig.savefig(os.path.join(FIG, "crossover_structure.pdf"))
    plt.close(fig)
    print("\nfigure ->", os.path.join(FIG, "crossover_structure.png"))


if __name__ == "__main__":
    main()
