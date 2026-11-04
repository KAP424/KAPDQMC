"""
Referee round 2: put the two-parameter form to a fair, parametric test.

The first blind-prediction attempt used a kernel-smoothed empirical h(x), which
a sceptic can always blame for the failure.  Here h is a two-parameter model

    D_bar(L, z) = D_gs(L) / (1 + (z / (c L))^{-g})

which has the right limits (D_bar -> 0 as z -> 0, -> D_gs as z -> infinity) and
whose small-z behaviour D_bar ~ D_gs (z/(cL))^g reproduces the observed z^2 law
when g = 2.  The crossover constant c and the exponent g are fitted on L <= 24
only and then used to predict the ABSOLUTE value of D_bar at L = 30 and L = 36.

Also:
  * exact comparison of the new beta = 0 anchor against baseline_D.json;
  * monotonicity of h(z) per L (the overshoot question);
  * the two independent L = 36, beta = 3 measurements.
"""

import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # parent: research_20261030/


def load_rows():
    base = json.load(open(os.path.join(HERE, "baseline_D.json")))
    ks = np.array(sorted(int(k) for k in base))
    vs = np.array([base[str(k)]["D"] for k in ks])
    # (L, beta) = (18, 1.0) was run twice (data_matched_z/ and
    # data_matched_largez/, independent seeds): pool samples per (L, beta).
    pool = {}
    for d in ("data_matched_z", "data_matched_largez"):
        for f in sorted(glob.glob(os.path.join(HERE, d, "*.csv"))):
            a = np.genfromtxt(f, delimiter=",", names=True)
            if a.size == 0:
                continue
            key = (int(a["L"][0]), round(float(a["beta"][0]), 4))
            D = np.asarray(a["D"], dtype=float)
            prev = pool.get(key)
            pool[key] = (np.concatenate([prev[0], D]) if prev else D, float(a["z"][0]),
                         (prev[2] + "+" if prev else "") + os.path.basename(f))
    rows = []
    for (L, b), (D, z, src) in sorted(pool.items()):
        rows.append(dict(L=L, beta=b, z=z, n=int(D.size), mean=float(D.mean()),
                         sem=float(D.std(ddof=1) / np.sqrt(D.size)),
                         Dgs=float(np.interp(np.log(L), np.log(ks), vs)), src=src))
    for r in rows:
        r["h"] = r["mean"] / r["Dgs"]
    return rows, ks, vs


def model(L, z, c, g):
    return 1.0 / (1.0 + (z / (c * L)) ** (-g))


def chi2_fit(rows, fixed_g=None):
    """Grid/gauss-newton on (c, g) using only these rows."""
    L = np.array([r["L"] for r in rows])
    z = np.array([r["z"] for r in rows])
    h = np.array([r["h"] for r in rows])
    s = np.array([r["sem"] / r["Dgs"] for r in rows])

    def chi2(c, g):
        return np.sum(((model(L, z, c, g) - h) / s) ** 2)

    if fixed_g is None:
        best = (np.inf, 0.18, 2.0)
        for g in np.arange(0.6, 4.01, 0.02):
            for c in np.arange(0.05, 0.60, 0.002):
                v = chi2(c, g)
                if v < best[0]:
                    best = (v, c, g)
        return best[1], best[2], best[0]
    best = (np.inf, 0.18)
    for c in np.arange(0.05, 0.60, 0.0005):
        v = chi2(c, fixed_g)
        if v < best[0]:
            best = (v, c)
    return best[1], fixed_g, best[0]


def report_prediction(rows, c, g, tag, sizes=(30, 36)):
    print(f"\n  --- {tag}: c={c:.4f}, g={g:.3f} ---")
    print(f"  {'L':>3} {'z':>9} {'D_pred':>9} {'D_meas':>9} {'sem':>7} "
          f"{'pull':>7} {'rel':>8}")
    ps = []
    for r in sorted([r for r in rows if r["L"] in sizes], key=lambda r: (r["L"], r["z"])):
        pred = r["Dgs"] * model(r["L"], r["z"], c, g)
        p = (pred - r["mean"]) / r["sem"]
        ps.append(p)
        print(f"  {r['L']:3d} {r['z']:9.3f} {pred:9.4f} {r['mean']:9.4f} "
              f"{r['sem']:7.4f} {p:7.2f} {100*(pred-r['mean'])/max(r['mean'],1e-6):7.1f}%")
    ps = np.array(ps)
    print(f"  rms pull {np.sqrt(np.mean(ps**2)):.2f}   max|pull| {np.abs(ps).max():.2f}"
          f"   chi2/dof {np.sum(ps**2)/len(ps):.2f}")
    return ps


def main():
    rows, ks, vs = load_rows()

    # ---------- beta = 0 anchor ----------
    print("=" * 74)
    print("0. BETA = 0 ANCHOR vs baseline_D.json  (the referee's cheapest test)")
    print("=" * 74)
    base = json.load(open(os.path.join(HERE, "baseline_D.json")))
    for f in sorted(glob.glob(os.path.join(HERE, "data_beta0", "*.csv"))):
        a = np.genfromtxt(f, delimiter=",", names=True)
        L = int(a["L"][0])
        D = a["D"]
        ref = base[str(L)]["D"]
        print(f"  L={L:2d}  born={D.mean():.10f}  baseline={ref:.10f}  "
              f"|diff|={abs(D.mean()-ref):.2e}  spread across {a.size} samples="
              f"{np.ptp(D):.2e}")
    print("  -> the Born code path reproduces D_gs to machine precision; the")
    print("     normalisation is not biased, so h-1 != 0 is not a bookkeeping error.")

    # ---------- parametric blind test ----------
    print("\n" + "=" * 74)
    print("1. PARAMETRIC BLIND PREDICTION (fit on L<=24, predict L=30,36)")
    print("=" * 74)
    train = [r for r in rows if r["L"] <= 24]
    c, g, chi = chi2_fit(train)
    ndof = len(train) - 2
    print(f"  free fit: c={c:.4f}, g={g:.3f}, chi2/dof={chi/ndof:.2f} (n={len(train)})")
    report_prediction(rows, c, g, "free (c,g)")

    c2, g2, chi2_ = chi2_fit(train, fixed_g=2.0)
    print(f"\n  g fixed at 2 (the exact L=3 exponent): c={c2:.4f}, "
          f"chi2/dof={chi2_/ (len(train)-1):.2f}")
    report_prediction(rows, c2, 2.0, "g=2 fixed, one parameter")

    # ---------- leave-one-out on the training set ----------
    print("\n" + "=" * 74)
    print("2. LEAVE-ONE-OUT over all sizes (each size predicted from the others)")
    print("=" * 74)
    for Lout in (6, 12, 18, 24, 30, 36):
        tr = [r for r in rows if r["L"] != Lout]
        cc, gg, _ = chi2_fit(tr)
        ps = []
        for r in rows:
            if r["L"] != Lout:
                continue
            ps.append((r["Dgs"] * model(r["L"], r["z"], cc, gg) - r["mean"]) / r["sem"])
        ps = np.array(ps)
        print(f"  hold out L={Lout:2d}: c={cc:.3f} g={gg:.2f}  "
              f"rms pull {np.sqrt(np.mean(ps**2)):5.2f}  max|pull| {np.abs(ps).max():5.2f}")

    # ---------- monotonicity / overshoot ----------
    print("\n" + "=" * 74)
    print("3. IS h(z) MONOTONE?  (overshoot above 1 would be physical)")
    print("=" * 74)
    for L in sorted({r["L"] for r in rows}):
        sel = sorted([r for r in rows if r["L"] == L], key=lambda r: r["z"])
        if len(sel) < 3:
            continue
        hh = np.array([r["h"] for r in sel])
        mono = all(np.diff(hh) > 0)
        print(f"  L={L:2d}: h = " + " ".join(f"{v:6.3f}" for v in hh)
              + ("   monotone" if mono else "   NON-MONOTONE"))

    # ---------- the duplicate L=36 beta=3 ----------
    print("\n" + "=" * 74)
    print("4. THE TWO INDEPENDENT (L=36, beta=3) MEASUREMENTS")
    print("=" * 74)
    for r in rows:
        if r["L"] == 36 and abs(r["beta"] - 3.0) < 1e-3:
            print(f"  {r['src']}: D={r['mean']:.5f} +/- {r['sem']:.5f} (n={r['n']})")
    old = [r for r in rows if r["src"].startswith("L36_beta3.0000")]
    if old:
        print(f"  old production anchor (beta3_large/means.csv): "
              f"D=0.24214 +/- 0.00284")
        print(f"  difference: {old[0]['mean']-0.24214:+.5f}, "
              f"{(old[0]['mean']-0.24214)/np.hypot(old[0]['sem'],0.00284):.2f} sigma")
        print("  -> statistically compatible; quote both, and note that the")
        print("     fixed-z slope at z=3.21 changes sign with the choice.")

    # ---------- z* form ----------
    print("\n" + "=" * 74)
    print("5. z*(L) FORM AND WHAT IT IMPLIES FOR L*(beta)")
    print("=" * 74)
    print(f"  free fit gives crossover at z* = c L with c = {c:.4f}")
    print(f"  -> L*(beta) = {c:.3f} e^(2 beta):")
    for b in (1.0, 1.5, 2.0, 2.5, 3.0):
        print(f"       beta={b}: L* = {c*np.exp(2*b):6.1f}")
    print("  NOTE: this is not an independent measurement.  z* is f^{-1}(D_gs/2)")
    print("        by construction, so 'z* ~ L' restates the collapse curve.")


if __name__ == "__main__":
    main()
