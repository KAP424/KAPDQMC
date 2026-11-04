"""
Corrected mechanism figures.

Fixes three errors found by the independent number audit (docs/NUMBER_AUDIT.md):

1. The j-stratum decomposition must come from ONE dataset (data_matched_z) and
   must not claim j = 0 gives a product state.  j = |E cap F| is the overlap of
   the MEASUREMENT SET with the AUXILIARY Slater determinant, not with the
   physical post-measurement support.  At j = 0 the auxiliary set equals the
   complement of F, so the filtered state is the near-product branch
   exp(-2 beta N_E)|Psi> with all particles of the dominant Fock component on
   E^c; it is a product state ONLY in the beta -> infinity limit, and at finite
   beta it has D ~ A z^2.  The data confirm this: <D | j=0> is not zero and at
   small z it carries most of the mean.

2. The annealed volume-law comparison must use L that are multiples of 3
   (w = L/3 must be an integer).  L = 10, 20, 40 are not valid sizes.

3. The small-z coefficient: A(L) = lim D_bar/z^2 = 16/81 + 4 D_V0(L)/L^4 for
   L >= 6 (exact, small_z_mechanism.py).  L = 3 is lower (0.1807) because 20%
   of its O(q) weight sits on rank-deficient outputs.  The reference line is
   therefore (16/81) z^2, not the old 0.18 z^2.
"""

import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from crossover_structure import read_csv

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
A_EXACT_L3 = 14.63775 / 81.0


def load_z():
    """One row per (L, beta) of data_matched_z/, pooling shard files.

    fig_small_z() draws one marker per row, so one row per FILE put a
    shard-split point on the plot several times at the same z, each with a
    fraction of the statistics.
    """
    pool = {}
    for f in sorted(glob.glob(os.path.join(HERE, "data_matched_z", "*.csv"))):
        a = read_csv(f)
        if a is None:
            continue
        key = (int(a["L"][0]), round(float(a["beta"][0]), 4))
        e = pool.setdefault(key, dict(L=key[0], beta=float(a["beta"][0]),
                                      z=float(a["z"][0]), D=[], j=[]))
        e["D"].append(np.asarray(a["D"], dtype=float))
        e["j"].append(np.asarray(a["j"], dtype=float))
    rows = []
    for key in sorted(pool):
        e = pool[key]
        rows.append(dict(L=e["L"], beta=e["beta"], z=e["z"],
                         D=np.concatenate(e["D"]), j=np.concatenate(e["j"])))
    return rows


def fig_j(rows):
    """Latent-j strata at beta = 3, i.e. z = 0.089, 0.357, 0.803 for L = 6, 12, 18.

    (a) sampled frequency of j (bars) against the exact law
        P(j) = C(a,j)^2 t^{2j} / Z_a (black ticks), a = L^2, t = e^{-2 beta};
    (b) fraction of the Born mean D_bar carried by each stratum (bootstrap errors).
    j >= 3 is pooled.  A linear category axis keeps j = 0 visible; the earlier
    log-axis version placed j = 0 at x = 0 and silently dropped it.
    """
    from math import lgamma
    cols = {6: "#1f77b4", 12: "#ff7f0e", 18: "#2ca02c"}
    rng = np.random.default_rng(3)
    width = 0.26
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    for k, L in enumerate((6, 12, 18)):
        sel = [r for r in rows if r["L"] == L and abs(r["beta"] - 3.0) < 0.01]
        if not sel:
            continue
        D = np.concatenate([r["D"] for r in sel])
        jc = np.minimum(np.concatenate([r["j"] for r in sel]).astype(int), 3)
        beta, z = sel[0]["beta"], sel[0]["z"]
        a = L * L
        jj = np.arange(a + 1)
        logC = lgamma(a + 1) - np.array([lgamma(x + 1) + lgamma(a - x + 1) for x in jj])
        lw = 2.0 * logC - 4.0 * beta * jj
        w = np.exp(lw - lw.max())
        w /= w.sum()
        p_exact = np.array([w[0], w[1], w[2], w[3:].sum()])
        p_samp = np.array([(jc == c).mean() for c in range(4)])
        share = np.array([D[jc == c].sum() for c in range(4)]) / D.sum()
        boot = np.empty((400, 4))
        for b in range(400):
            idx = rng.integers(0, D.size, D.size)
            Db, jb = D[idx], jc[idx]
            boot[b] = [Db[jb == c].sum() / Db.sum() for c in range(4)]
        x = np.arange(4) + (k - 1) * width
        lab = rf"$L={L}$, $z={z:.3f}$"
        axes[0].bar(x, p_samp, width, color=cols[L], alpha=0.85, label=lab)
        axes[0].plot(x, p_exact, "_", color="k", ms=13, mew=2)
        axes[1].bar(x, share, width, yerr=boot.std(axis=0), capsize=2,
                    color=cols[L], alpha=0.85, label=lab)
        print(f"  L={L} z={z:.3f} n={D.size}: P_samp={np.round(p_samp, 4)} "
              f"P_exact={np.round(p_exact, 4)} share={np.round(share, 3)} "
              f"+-{np.round(boot.std(axis=0), 3)}")
    axes[0].plot([], [], "_", color="k", ms=13, mew=2,
                 label=r"exact $P(j)\propto\binom{a}{j}^2 e^{-4\beta j}$")
    for ax in axes:
        ax.set_xticks(range(4))
        ax.set_xticklabels(["0", "1", "2", r"$\geq 3$"])
        ax.set_xlabel(r"latent $j=|E\cap F|$")
    axes[0].set_yscale("log")
    axes[0].set_ylim(3e-5, 2.0)
    axes[0].set_ylabel(r"probability $P(j)$")
    axes[0].set_title(r"(a) $\beta=3$: sampled $P(j)$ (bars) vs exact (ticks)")
    axes[0].legend(fontsize=7, frameon=False, loc="upper right")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_ylabel(r"fraction of $\overline{D_2}$ from stratum $j$")
    axes[1].set_title(r"(b) share of the Born mean carried by each $j$")
    axes[1].legend(fontsize=7, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "j_decomposition.png"), dpi=200)
    fig.savefig(os.path.join(FIG, "j_decomposition.pdf"))
    plt.close(fig)


def fig_annealed():
    Ls = np.array([12, 18, 24, 30, 36, 42, 48, 54, 60])
    w = Ls // 3
    vol = 2 * w * (Ls - 2 * w) * np.log(2)
    fig, ax = plt.subplots(figsize=(5.8, 4.3))
    ax.plot(Ls, vol, "k-", lw=1.6, label=r"annealed Gaussian: $2w(L-2w)\ln 2$ (volume law)")
    # One point per L: pool files sharing an L so a shard-split point is not
    # drawn as several half-statistics points at the same x.
    pool = {}
    for f in sorted(glob.glob(os.path.join(HERE, "data_matched_z", "*.csv"))):
        a = read_csv(f)
        if a is None or abs(float(a["beta"][0]) - 3.0) >= 0.01:
            continue
        pool.setdefault(int(a["L"][0]), []).append(a["D"])
    Ls = sorted(pool)
    xs = np.array(Ls)
    ys = np.array([np.concatenate(pool[L]).mean() for L in Ls])
    es = np.array([np.concatenate(pool[L]).std(ddof=1) / np.sqrt(np.concatenate(pool[L]).size)
                   for L in Ls])
    ax.errorbar(xs, ys, yerr=es, marker="s", ms=5, lw=1.2, capsize=2,
                color="crimson", label=r"Born average, $\beta=3$")
    ax.set_xlabel("L")
    ax.set_ylabel(r"$D_2$")
    ax.set_yscale("log")
    ax.set_title("First-moment state is not the measured ensemble")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "annealed_vs_born.png"), dpi=200)
    fig.savefig(os.path.join(FIG, "annealed_vs_born.pdf"))
    plt.close(fig)


def fig_small_z(rows):
    fig, ax = plt.subplots(figsize=(5.8, 4.3))
    for L, col in ((6, "#1f77b4"), (12, "#ff7f0e"), (18, "#2ca02c"), (24, "#d62728")):
        sel = sorted([r for r in rows if r["L"] == L and r["z"] < 2.2], key=lambda r: r["z"])
        if not sel:
            continue
        z = np.array([r["z"] for r in sel])
        m = np.array([r["D"].mean() for r in sel])
        e = np.array([r["D"].std(ddof=1) / np.sqrt(r["D"].size) for r in sel])
        ax.errorbar(z, m, yerr=e, marker="o", ms=4, lw=1.1, capsize=2, color=col, label=rf"$L={L}$")
    zz = np.geomspace(0.06, 2.2, 40)
    ax.plot(zz, 16 / 81 * zz ** 2, "k--", lw=1.3, label=r"$(16/81)\,z^2$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$z=L^2e^{-2\beta}$")
    ax.set_ylabel(r"$\overline{D_2}$")
    ax.set_title(r"Small $z$: $\overline{D_2}\propto z^2$, same amplitude for every $L$")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "small_z_law.png"), dpi=200)
    fig.savefig(os.path.join(FIG, "small_z_law.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    rows = load_z()
    fig_j(rows)
    fig_annealed()
    fig_small_z(rows)
    print("figures ->", FIG)
