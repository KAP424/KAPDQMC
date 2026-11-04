"""
One-page overview of the final picture (figures/summary.png).

  (a) strong side: D_bar vs z for every L -- one curve in z until D_bar reaches
      the ground-state value of that L; small z: (16/81) z^2.
  (b) small z is a volume-law residual: region entropies / [4 f(1-f) z^2] and
      D / [(16/81) z^2] at z = 0.089 are ~1 for every L.
  (c) weak side: D_bar / D_gs at fixed beta plateaus for L >= 24.
  (d) no single crossover exponent: p(h) of z_h ~ L^p rises with the threshold.

Inputs: data_matched_z/, data_matched_largez/, data_gapfill/ (via
crossover_structure.load_points), baseline_D.json.
Usage: D:/app/conda/python.exe summary_figure.py
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import crossover_structure as C

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
RNG = np.random.default_rng(20261031)
VOL = {"SA": 8 / 9, "SB": 56 / 81, "SC": 8 / 9, "D": 16 / 81}
COL = {6: "#440154", 12: "#3b528b", 18: "#21918c", 24: "#5ec962", 30: "#e6a01f", 36: "#b02020"}


def p_of_h(pts, levels=(0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), nboot=400):
    byL = C.per_L(pts)
    use = {}
    for h in levels:
        Ls = []
        for L in sorted(byL):
            zs = np.array([p["z"] for p in byL[L]])
            hs = np.array([p["mean"] / p["Dgs"] for p in byL[L]])
            z, w = C.crossing(zs, hs, h)
            if z is not None and w <= C.MAX_BRACKET:
                Ls.append(L)
        if len(Ls) >= 3:
            use[h] = Ls
    boots = {h: [] for h in use}
    for _ in range(nboot):
        bm = {(p["L"], p["beta"]): p["D"][RNG.integers(0, p["n"], p["n"])].mean() for p in pts}
        for h, Ls in use.items():
            y = []
            for L in Ls:
                zs = np.array([q["z"] for q in byL[L]])
                hs = np.array([bm[(q["L"], q["beta"])] / q["Dgs"] for q in byL[L]])
                z, _ = C.crossing(zs, hs, h)
                y.append(np.log(z) if z else np.nan)
            if np.all(np.isfinite(y)):
                boots[h].append(np.polyfit(np.log(Ls), y, 1)[0])
    out = {}
    for h, Ls in use.items():
        y = []
        for L in Ls:
            zs = np.array([q["z"] for q in byL[L]])
            hs = np.array([q["mean"] / q["Dgs"] for q in byL[L]])
            y.append(np.log(C.crossing(zs, hs, h)[0]))
        out[h] = (np.polyfit(np.log(Ls), y, 1)[0], float(np.std(boots[h])))
    return out


def main():
    pts, base = C.load_points()
    byL = C.per_L(pts)
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.6))

    ax = axes[0, 0]
    for L in sorted(byL):
        sel = byL[L]
        ax.errorbar([p["z"] for p in sel], [p["mean"] for p in sel], yerr=[p["sem"] for p in sel],
                    marker="o", ms=3.6, lw=1.0, capsize=1.5, color=COL.get(L, "k"), label=f"L={L}")
        ax.plot([600], [base[str(L)]["D"]], marker="<", ms=7, color=COL.get(L, "k"))
    zz = np.geomspace(0.06, 3.0, 40)
    ax.plot(zz, VOL["D"] * zz ** 2, "k--", lw=1.1, label=r"$(16/81)\,z^2$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylim(5e-4, 1.2)
    ax.set_xlabel(r"$z = L^2 e^{-2\beta}$"); ax.set_ylabel(r"$\overline{D_2}$")
    ax.set_title(r"(a) strong side: one curve in $z$;  $\triangleleft$ = $D_{\rm gs}(L)$")
    ax.legend(fontsize=6.5, frameon=False, ncol=2, loc="lower right")

    ax = axes[0, 1]
    import glob
    # Pool every file sharing an L -- a point can be split across shard files
    # (L30_beta4.6108.csv.shardKof14.csv), and keying by L alone silently kept
    # only the last shard, which inflated L=30's error bar fourfold.
    pool = {}
    for f in glob.glob(os.path.join(HERE, "data_matched_z", "*.csv")):
        a = C.read_csv(f)
        if a is None:
            continue
        z = float(a["z"][0])
        if abs(z - 0.089) / 0.089 >= 0.01:
            continue
        pool.setdefault(int(a["L"][0]), []).append(a)
    data = {}
    for L, arrs in pool.items():
        z2 = float(arrs[0]["z"][0]) ** 2
        data[L] = {c: np.concatenate([a[c] for a in arrs]) / z2 / VOL[c]
                   for c in VOL}
    off = {"SA": -0.9, "SC": -0.3, "SB": 0.3, "D": 0.9}
    lab = {"SA": r"$S_2(A)$ / $\frac{8}{9}z^2$", "SC": r"$S_2(C)$ / $\frac{8}{9}z^2$",
           "SB": r"$S_2(B)$ / $\frac{56}{81}z^2$", "D": r"$D_2$ / $\frac{16}{81}z^2$"}
    for c, col in (("SA", "#2a78d6"), ("SC", "#1baf7a"), ("SB", "#eb6834"), ("D", "k")):
        Ls = sorted(data)
        ax.errorbar(np.array(Ls) + off[c], [data[L][c].mean() for L in Ls],
                    yerr=[data[L][c].std(ddof=1) / np.sqrt(data[L][c].size) for L in Ls],
                    marker="o", ms=4.5, lw=0, elinewidth=1.1, capsize=2, color=col, label=lab[c])
    ax.axhline(1.0, color="0.3", lw=0.9, ls="--")
    ax.set_ylim(0.0, 2.0)
    ax.set_xlabel("L"); ax.set_ylabel("measured / volume-fraction law, $z = 0.089$")
    ax.set_title(r"(b) small $z$: $S_2(R) \approx 4f_R(1-f_R)\,z^2$ -- a volume law")
    ax.legend(fontsize=7, frameon=False, ncol=2, loc="upper left")

    ax = axes[1, 0]
    cmap = plt.get_cmap("viridis")
    for k, b in enumerate((0.5, 1.0, 1.5, 2.0, 3.0)):
        sel = sorted([p for p in pts if abs(p["beta"] - b) < 2e-3], key=lambda p: p["L"])
        ax.errorbar([p["L"] for p in sel], [p["mean"] / p["Dgs"] for p in sel],
                    yerr=[p["sem"] / p["Dgs"] for p in sel], marker="o", ms=4, lw=1.1, capsize=2,
                    color=cmap(k / 4.5), label=rf"$\beta = {b}$")
    ax.axhline(1.0, color="k", lw=0.7, ls=":")
    ax.set_xlabel("L"); ax.set_ylabel(r"$\overline{D_2}/D_{\rm gs}(L)$")
    ax.set_title(r"(c) weak side: fixed-$\beta$ plateaus ($\beta\leq1.5$, $L\geq24$)")
    ax.legend(fontsize=7, frameon=False)

    ax = axes[1, 1]
    ph = p_of_h(pts)
    hs = sorted(ph)
    ax.errorbar(hs, [ph[h][0] for h in hs], yerr=[ph[h][1] for h in hs], marker="o", ms=5,
                lw=1.2, capsize=3, color="k")
    ax.axhline(2.0, color="crimson", ls="--", lw=1.0, label=r"$p = 2$: fixed-$\beta$ direction")
    ax.set_xlabel(r"threshold $h$ defining $z_h$"); ax.set_ylabel(r"$p$ in $z_h \propto L^p$")
    ax.set_title("(d) no single crossover exponent")
    ax.legend(fontsize=7, frameon=False, loc="upper left")

    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "summary.png"), dpi=200)
    fig.savefig(os.path.join(FIG, "summary.pdf"))
    plt.close(fig)
    print("p(h):", {h: (round(v[0], 3), round(v[1], 3)) for h, v in ph.items()})
    print("figure ->", os.path.join(FIG, "summary.png"))


if __name__ == "__main__":
    main()
