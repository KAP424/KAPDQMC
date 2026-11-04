"""
Two-regime check of the threshold exponents p(h) (SUMMARY.md section 2.4).

If the strong side obeys D_bar = f(z) with f independent of L, a contour
h = D_bar/D_gs(L) at low h satisfies f(z_h) = h * D_gs(L), so z_h moves only
because D_gs(L) grows.  If the weak side obeys D_bar = D_gs(L) - Delta(beta)
with Delta independent of L, a high-h contour sits at Delta(beta_h) =
(1 - h) D_gs(L), beta_h barely moves, and z_h = L^2 exp(-2 beta_h) ~ L^2.
This script predicts p(h) at both ends from those two forms alone and prints
the measured values for comparison.

Usage: D:/app/conda/python.exe two_regime_check.py
"""

import numpy as np

import crossover_structure as C

# The measured p(h) is NOT hard-coded here.  An earlier version carried fixed
# constants (0.585/0.648/0.787/0.913 and the z_h arrays for h=0.8, 0.9) that
# were read off before the sample extension, and they silently went stale while
# the rest of the paper moved to the bootstrap numbers.  Compute them from the
# CSVs with the same crossing rule crossover_structure.py uses.
LEVELS_LOW = (0.3, 0.4, 0.5, 0.6)
LEVELS_HIGH = (0.8, 0.9)
MAX_BRACKET = C.MAX_BRACKET


def deficit(pts, beta, Lmin=24):
    """Delta(beta) = D_gs - D_bar, inverse-variance averaged over L >= Lmin.

    This used to be a hard-coded {1.5: 0.0345, 2.0: 0.1523}, read off before the
    sample extension; with the current data the same fit gives 0.0372 and
    0.1439, 8% and 6% away.  Same staleness as the p(h) constants above.
    """
    sel = [p for p in pts if abs(p["beta"] - beta) < 2e-3 and p["L"] >= Lmin]
    if len(sel) < 2:
        raise ValueError(f"need >= 2 sizes with L >= {Lmin} at beta={beta}")
    w = np.array([1 / p["sem"] ** 2 for p in sel])
    d = np.array([p["Dgs"] - p["mean"] for p in sel])
    return float((w * d).sum() / w.sum())


def measured_crossings(pts, h):
    """z_h(L) for every L whose bracket is narrow enough to count as measured."""
    out = {}
    for L, sel in C.per_L(pts).items():
        sel = sorted(sel, key=lambda q: q["z"])
        zs = np.array([q["z"] for q in sel])
        hs = np.array([q["mean"] / q["Dgs"] for q in sel])
        z, w = C.crossing(zs, hs, h)
        if z is not None and w <= MAX_BRACKET:
            out[L] = z
    return out


def pooled_f(pts, zmax=11.0):
    groups = {}
    for p in pts:
        if p["z"] > zmax:
            continue
        k = next((g for g in groups if abs(g - p["z"]) / g < 2e-3), p["z"])
        groups.setdefault(k, []).append(p)
    zs = sorted(g for g in groups if len({q["L"] for q in groups[g]}) >= 2)
    f = []
    for g in zs:
        w = np.array([1 / q["sem"] ** 2 for q in groups[g]])
        m = np.array([q["mean"] for q in groups[g]])
        f.append(float(np.sum(w * m) / np.sum(w)))
    return np.array(zs), np.array(f)


def main():
    pts, base = C.load_points()
    dgs = {int(k): v["D"] for k, v in base.items()}
    zs, f = pooled_f(pts)
    print("pooled f(z): " + ", ".join(f"{z:.3f}:{v:.4f}" for z, v in zip(zs, f)))

    print("\nlow h: f(z_h) = h D_gs(L), f pooled over L")
    for h in LEVELS_LOW:
        mz = measured_crossings(pts, h)
        Ls = sorted(mz)
        if len(Ls) < 3:
            print(f"  h={h}: only {len(Ls)} well-bracketed sizes {Ls} -> no fit")
            continue
        zp = [np.exp(np.interp(np.log(h * dgs[L]), np.log(f), np.log(zs))) for L in Ls]
        pp = np.polyfit(np.log(Ls), np.log(zp), 1)[0]
        pm = np.polyfit(np.log(Ls), np.log([mz[L] for L in Ls]), 1)[0]
        print(f"  h={h}: predicted p = {pp:.3f}   measured p = {pm:.3f}   sizes {Ls}")

    print("\nhigh h: Delta(beta_h) = (1 - h) D_gs(L), ln Delta linear in beta between 1.5 and 2")
    DELTA = {b: deficit(pts, b) for b in (1.5, 2.0)}
    print("  Delta(beta), L >= 24: " + ", ".join(f"beta={b}: {d:.4f}"
                                                for b, d in DELTA.items()))
    (b1, d1), (b2, d2) = sorted(DELTA.items())
    k = np.log(d2 / d1) / (b2 - b1)
    for h in LEVELS_HIGH:
        mz = measured_crossings(pts, h)
        Ls = sorted(mz)
        if len(Ls) < 3:
            print(f"  h={h}: only {len(Ls)} well-bracketed sizes {Ls} -> no fit")
            continue
        bh = [b1 + np.log((1 - h) * dgs[L] / d1) / k for L in Ls]
        zp = [L * L * np.exp(-2 * b) for L, b in zip(Ls, bh)]
        pp = np.polyfit(np.log(Ls), np.log(zp), 1)[0]
        pm = np.polyfit(np.log(Ls), np.log([mz[L] for L in Ls]), 1)[0]
        print(f"  h={h}: predicted p = {pp:.3f}   measured p = {pm:.3f}   sizes {Ls}   "
              f"beta_h = " + ", ".join(f"{b:.2f}" for b in bh))


if __name__ == "__main__":
    main()
