"""
Final integrated analysis of the Born-averaged corner entropy.

Produces the two-parameter crossover picture

    D_bar(L, beta) = D_gs(L) * h( z / z*(L) ),   z = L^2 e^{-2 beta},

and the small-z law D_bar = A z^2 phi(z) with A = 14.63775/81 fixed by exact
L=3 enumeration.

Inputs: data_matched_z/ (five matched-z lines + beta anchors, L=6..36),
        data_matched_largez/ (direct z=43.85 points), baseline_D.json.
"""

import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# read_csv() returns None for a zero-byte or header-only data file instead of
# letting np.genfromtxt raise IndexError on it.
from crossover_structure import read_csv

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
A_EXACT = 14.63775 / 81.0


def load():
    base = json.load(open(os.path.join(HERE, "baseline_D.json")))
    ks = np.array(sorted(int(k) for k in base))
    vs = np.array([base[str(k)]["D"] for k in ks])
    # Pool every file sharing (L, beta): a point can be split across shard files
    # (L30_beta4.6108.csv.shardKof14.csv), and one row per file would give that
    # point a sqrt(Nshards)-times larger error bar and a duplicate marker in the
    # figure.  (L=18, beta=1.0) also exists twice as independent runs.
    pool, zval = {}, {}
    for d in ("data_matched_z", "data_matched_largez"):
        for f in sorted(glob.glob(os.path.join(HERE, d, "*.csv"))):
            a = read_csv(f)
            if a is None:
                continue
            key = (int(a["L"][0]), round(float(a["beta"][0]), 4))
            pool.setdefault(key, []).append(np.asarray(a["D"], dtype=float))
            zval[key] = float(a["z"][0])
    rows = []
    for (L, beta), arrs in sorted(pool.items()):
        D = np.concatenate(arrs)
        rows.append(dict(L=L, beta=beta, z=zval[(L, beta)], n=int(D.size),
                         mean=float(D.mean()),
                         sem=float(D.std(ddof=1) / np.sqrt(D.size)),
                         Dgs=float(np.interp(np.log(L), np.log(ks), vs))))
    # L=36, beta=3 uses the CSV only, as headline_numbers.py does.  The legacy
    # production value 0.24214 +- 0.00284 used to be appended as a second
    # hard-coded row for the same point; it is an independent MCMC measurement
    # (BORN_SCALING_RESULTS.md) that agrees with the CSV to 0.1 sigma, and
    # dropping it moves z*(36) by 0.4%.  See bessel_check.load_rows.
    for r in rows:
        r["h"] = r["mean"] / r["Dgs"]
    return rows, ks, vs


def zstar_per_L(rows):
    """z where h = 1/2, per L (ln z interpolation)."""
    out = {}
    for L in sorted({r["L"] for r in rows}):
        sel = sorted([r for r in rows if r["L"] == L], key=lambda r: r["z"])
        if len(sel) < 3:
            continue
        zz = np.log([r["z"] for r in sel])
        hh = np.array([r["h"] for r in sel])
        idx = np.where(np.diff(np.sign(hh - 0.5)))[0]
        if len(idx):
            i = idx[0]
            out[L] = float(np.exp(zz[i] + (0.5 - hh[i]) * (zz[i + 1] - zz[i])
                                 / (hh[i + 1] - hh[i])))
    return out


def main():
    rows, ks, vs = load()
    rows.sort(key=lambda r: (r["z"], r["L"]))
    zs = zstar_per_L(rows)

    print("== crossover scales ==")
    for L, z in zs.items():
        print(f"  L={L:2d}  z*={z:6.3f}  z*/L={z/L:6.3f}")
    Ls = np.array(sorted(zs))
    Z = np.array([zs[L] for L in Ls])
    c_lin = np.sum(Z * Ls) / np.sum(Ls * Ls)
    print(f"  linear fit: z* = {c_lin:.4f} L")
    print(f"  -> L*(beta) = {c_lin:.3f} e^(2 beta):")
    for b in (1.0, 1.5, 2.0, 2.5, 3.0):
        print(f"       beta={b}: L*={c_lin*np.exp(2*b):7.1f}")

    # ---------------- figure: h vs z/z* collapse ----------------------
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.3))
    col = {6: "#440154", 12: "#31688e", 18: "#35b779", 24: "#fde725",
           30: "#e6601f", 36: "#b02020"}
    ax = axes[0]
    for L in sorted({r["L"] for r in rows}):
        sel = sorted([r for r in rows if r["L"] == L], key=lambda r: r["z"])
        ax.errorbar([r["z"] for r in sel], [r["h"] for r in sel],
                    yerr=[r["sem"] / r["Dgs"] for r in sel], marker="o", ms=3.6,
                    lw=1.05, capsize=1.5, color=col.get(L, "k"), label=f"L={L}")
    ax.axhline(1.0, color="k", lw=0.7, ls=":")
    ax.set_xscale("log")
    ax.set_ylim(-0.05, 1.2)
    ax.set_xlabel(r"$z=L^2e^{-2\beta}$")
    ax.set_ylabel(r"$\overline{D_2}/D_{\rm gs}(L)$")
    ax.set_title(r"(a) crossover moves to larger $z$ with $L$")
    ax.legend(fontsize=5.8, frameon=False, ncol=2, loc="upper left")

    ax = axes[1]
    for L in sorted({r["L"] for r in rows}):
        sel = sorted([r for r in rows if r["L"] == L], key=lambda r: r["z"])
        zs_L = zs.get(L, c_lin * L)
        ax.errorbar([r["z"] / zs_L for r in sel], [r["h"] for r in sel],
                    yerr=[r["sem"] / r["Dgs"] for r in sel], marker="o", ms=3.6,
                    lw=1.05, capsize=1.5, color=col.get(L, "k"), label=f"L={L}")
    ax.axhline(0.5, color="k", lw=0.7, ls=":")
    ax.set_xscale("log")
    ax.set_ylim(-0.05, 1.2)
    ax.set_xlabel(r"$z/z_*(L)$,   $z_*=%.3f\,L$" % c_lin)
    ax.set_ylabel(r"$\overline{D_2}/D_{\rm gs}(L)$")
    ax.set_title("(b) scaling collapse")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "final_collapse.png"), dpi=220)
    fig.savefig(os.path.join(FIG, "final_collapse.pdf"))
    plt.close(fig)

    # ---------------- figure: phi(z) ------------------------------
    fig, ax = plt.subplots(figsize=(5.6, 4.3))
    for L in sorted({r["L"] for r in rows}):
        sel = sorted([r for r in rows if r["L"] == L and r["z"] < 6], key=lambda r: r["z"])
        ax.errorbar([r["z"] for r in sel], [r["mean"] / (A_EXACT * r["z"] ** 2) for r in sel],
                    yerr=[r["sem"] / (A_EXACT * r["z"] ** 2) for r in sel],
                    marker="o", ms=3.8, lw=1.05, capsize=1.5, color=col.get(L, "k"),
                    label=f"L={L}")
    zz = np.geomspace(0.08, 6, 60)
    ax.plot(zz, 1 - 0.45 * zz, "k--", lw=1.2, label=r"$1-0.45z$")
    ax.set_xscale("log")
    ax.set_ylim(-0.1, 1.9)
    ax.set_xlabel(r"$z$")
    ax.set_ylabel(r"$\overline{D_2}/(A z^2)$,   $A=14.638/3^4$")
    ax.set_title(r"$\phi(z)$ is $L$-independent (collapse of the shape)")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "final_phi.png"), dpi=220)
    fig.savefig(os.path.join(FIG, "final_phi.pdf"))
    plt.close(fig)

    json.dump({"zstar": {str(k): v for k, v in zs.items()},
               "zstar_linear_coef": float(c_lin), "A_exact": A_EXACT},
              open(os.path.join(HERE, "final_fit.json"), "w"), indent=1)
    print("\nfigures ->", FIG)


if __name__ == "__main__":
    main()
