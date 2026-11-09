"""
Referee-mandated checks, run against the final data state.

1. Blind prediction: fit the crossover constant on L <= 24 only, then predict
   the ABSOLUTE value of D_bar at L = 30 and L = 36 and compare to their SEM.
   The referee's point is that z*(L) is not independently measured -- it is
   f^{-1}(D_gs/2) for whatever curve f describes the data -- so the only way to
   give z* empirical content is out-of-sample prediction.

2. z*(L) power-law fit: p and its sigma, to see whether p = 1 survives.

3. chi2/dof versus a constant, per fixed-z row.

4. c2 model dependence: vary the finite-size correction term.

5. A(z) = D/z^2 at each z, to test the "universal A ~ 0.18" claim.

6. h > 1 excess, with the beta = 0 anchor now confirming D_gs exactly.
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


def zstar(rows, Ls=None):
    out = {}
    for L in sorted({r["L"] for r in rows}):
        if Ls is not None and L not in Ls:
            continue
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


def fit_power(Ls, Z):
    """ln z* = ln A + p ln L, ordinary least squares."""
    x = np.log(Ls); y = np.log(Z)
    n = len(x)
    xm, ym = x.mean(), y.mean()
    p = np.sum((x - xm) * (y - ym)) / np.sum((x - xm) ** 2)
    a = ym - p * xm
    resid = y - (a + p * x)
    s2 = np.sum(resid ** 2) / (n - 2)
    sp = np.sqrt(s2 / np.sum((x - xm) ** 2))
    return float(np.exp(a)), float(p), float(sp), float(np.sqrt(np.mean(resid ** 2)))


def main():
    rows, ks, vs = load_rows()

    # ---------- 1. blind prediction from L <= 24 ----------
    print("=" * 72)
    print("1. BLIND PREDICTION: fit on L<=24, predict L=30, L=36")
    print("=" * 72)
    train = [r for r in rows if r["L"] <= 24]
    zs_tr = zstar(train)
    Ltr = np.array(sorted(zs_tr))
    Ztr = np.array([zs_tr[L] for L in Ltr])
    c_tr = float(np.sum(Ztr * Ltr) / np.sum(Ltr * Ltr))
    print(f"  training sizes {list(Ltr)}: z* = {c_tr:.4f} L   (per-L: "
          + ", ".join(f"{zs_tr[L]/L:.3f}" for L in Ltr) + ")")

    # Rescale each training curve by its own z*(L) and fit h(x) as a smooth
    # monotone function of x = z/z*; then h_pred(z) = h(z/(c_tr L)).
    def h_of_x():
        xs, ys = [], []
        for r in train:
            x = np.log(r["z"] / zs_tr[r["L"]])
            xs.append(x); ys.append(r["h"])
        return np.array(xs), np.array(ys)

    xs, ys = h_of_x()
    # local linear regression in log x with a Gaussian kernel, width 0.5 dex
    def h_pred(L, z):
        xt = np.log(z / (c_tr * L))
        w = np.exp(-0.5 * ((xs - xt) / 0.45) ** 2)
        W = w.sum()
        if W < 1e-6:
            return 1.0 if xt > 0 else 0.0
        xb = np.sum(w * xs) / W; yb = np.sum(w * ys) / W
        sxx = np.sum(w * (xs - xb) ** 2)
        if sxx < 1e-8:
            return float(yb)
        slope = np.sum(w * (xs - xb) * (ys - yb)) / sxx
        return float(yb + slope * (xt - xb))

    print(f"\n  {'L':>3} {'z':>9} {'D_pred':>9} {'D_meas':>9} {'sig':>7} "
          f"{'pull':>7} {'rel.err':>8}")
    pulls = []
    for r in sorted([r for r in rows if r["L"] > 24], key=lambda r: (r["L"], r["z"])):
        pred = r["Dgs"] * h_pred(r["L"], r["z"])
        err = pred - r["mean"]
        pull = err / r["sem"]
        pulls.append(pull)
        print(f"  {r['L']:3d} {r['z']:9.3f} {pred:9.4f} {r['mean']:9.4f} "
              f"{r['sem']:7.4f} {pull:7.2f} {100*err/r['mean']:7.1f}%")
    pulls = np.array(pulls)
    print(f"\n  out-of-sample: rms pull = {np.sqrt(np.mean(pulls**2)):.2f}, "
          f"mean pull = {pulls.mean():+.2f}, max |pull| = {np.max(np.abs(pulls)):.2f}")

    # ---------- 2. z* power law on all six crossings ----------
    print("\n" + "=" * 72)
    print("2. z*(L) FORM: linear vs power law (all six crossings)")
    print("=" * 72)
    zs_all = zstar(rows)
    La = np.array(sorted(zs_all))
    Za = np.array([zs_all[L] for L in La])
    A, p, sp, rms_p = fit_power(La, Za)
    c_lin = float(np.sum(Za * La) / np.sum(La * La))
    rms_lin = float(np.sqrt(np.mean((Za - c_lin * La) ** 2)))
    print(f"  z*(L):  " + ", ".join(f"L={L}:{zs_all[L]:.3f}" for L in La))
    print(f"  power law: z* = {A:.4f} L^{p:.4f}  +/-  p_sigma {sp:.4f}, "
          f"rms {rms_p:.4f};  p=1 is {abs(1-p)/sp:.1f} sigma away")
    print(f"  linear:    z* = {c_lin:.4f} L,                    rms {rms_lin:.4f}")
    print(f"  at beta=3: linear -> L*={c_lin*np.exp(6):.1f}, "
          f"power -> L*={A*np.exp(6):.1f}")

    # ---------- 3. chi2/dof vs constant, per row ----------
    print("\n" + "=" * 72)
    print("3. FIXED-z ROWS: chi2/dof against a constant")
    print("=" * 72)
    for zt in sorted({round(r["z"], 3) for r in rows}):
        sel = [r for r in rows if abs(r["z"] - zt) < 1e-3]
        if len(sel) < 2:
            continue
        m = np.array([r["mean"] for r in sel]); s = np.array([r["sem"] for r in sel])
        w = 1 / s ** 2
        mu = np.sum(w * m) / np.sum(w)
        chi2 = np.sum(w * (m - mu) ** 2)
        dof = len(sel) - 1
        spread = 100 * (m.max() - m.min()) / m.min()
        print(f"  z={zt:8.3f}  N={len(sel)}  chi2/dof={chi2/dof:5.2f}  "
              f"spread={spread:6.1f}%  mean={mu:.4f}")

    # ---------- 4. c2 model dependence ----------
    print("\n" + "=" * 72)
    print("4. GROUND-STATE c2: model dependence")
    print("=" * 72)
    base = json.load(open(os.path.join(HERE, "baseline_D.json")))
    Lall = np.array(sorted(int(k) for k in base))
    Dall = np.array([base[str(k)]["D"] for k in Lall])
    for lo in (6, 12, 24, 30, 36):
        m = Lall >= lo
        L = Lall[m]; D = Dall[m]
        out = []
        for name, extra in (("pure log", lambda L: np.zeros_like(L, dtype=float)),
                            ("+a/L", lambda L: 1.0 / L),
                            ("+a/L^0.5", lambda L: 1.0 / np.sqrt(L)),
                            ("+a/L^1.5", lambda L: 1.0 / L ** 1.5)):
            X = np.column_stack([np.log(L), extra(L), np.ones_like(L)])
            coef, *_ = np.linalg.lstsq(X, D, rcond=None)
            r = D - X @ coef
            out.append((name, coef[0], np.sqrt(np.mean(r ** 2))))
        print(f"  L>={lo:2d}:  " + "   ".join(f"{n}={c:.4f}(rms {rr:.0e})"
                                             for n, c, rr in out))
    print("  -> model spread is the honest error bar, not the in-sample sigma.")

    # ---------- 5. A(z) ----------
    print("\n" + "=" * 72)
    print("5. IS A = D/z^2 UNIVERSAL?  (A_exp = 0.18071 from L=3)")
    print("=" * 72)
    AE = 14.63775 / 81.0
    for zt in sorted({round(r["z"], 3) for r in rows}):
        if zt > 4:
            continue
        sel = sorted([r for r in rows if abs(r["z"] - zt) < 1e-3], key=lambda r: r["L"])
        vals = [r["mean"] / r["z"] ** 2 for r in sel]
        errs = [r["sem"] / r["z"] ** 2 for r in sel]
        print(f"  z={zt:7.3f}: " + "  ".join(
            f"L{r['L']}:{v:.3f}+/-{e:.3f}" for r, v, e in zip(sel, vals, errs)))
        print(f"            spread (max/min) = {max(vals)/min(vals):.2f}x")

    # ---------- 6. h > 1 excess ----------
    print("\n" + "=" * 72)
    print("6. LARGE-z EXCESS h-1  (beta=0 anchor now confirms D_gs exactly)")
    print("=" * 72)
    for r in sorted([r for r in rows if r["z"] > 15], key=lambda r: r["z"]):
        exc = r["mean"] - r["Dgs"]
        print(f"  L={r['L']:2d} z={r['z']:8.2f}  h-1 = {100*(r['h']-1):+6.2f}%  "
              f"+/- {100*r['sem']/r['Dgs']:.2f}%  ({exc/r['sem']:+.2f} sigma)")


if __name__ == "__main__":
    main()
