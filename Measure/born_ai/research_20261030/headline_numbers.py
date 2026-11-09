"""Recompute every headline number in the paper and SUMMARY.md.

These were previously produced by ad-hoc commands typed at a prompt, which meant
each new batch of samples required re-deriving them by hand (and risked quoting
stale values).  Everything here reads the CSVs through one pooled loader, so a
point split across shard files counts once with its full statistics.

Usage:
    D:/app/conda/python.exe headline_numbers.py
"""

import glob
import json
import math
import os
from collections import defaultdict

import numpy as np

import crossover_structure as C

HERE = os.path.dirname(os.path.abspath(__file__))
COLS = ("D", "SA", "SB", "SC")
DIRS = ("data_matched_z", "data_matched_largez", "data_gapfill")


def load_pooled():
    """One record per (L, beta), pooling every file that shares the key.

    A point can be split across shard files (L30_beta4.6108.csv.shardKof14.csv).
    Keying by L alone -- as some earlier scripts did -- kept only the last file
    and inflated that point's error bar fourfold.
    """
    pool, zval = defaultdict(list), {}
    for d in DIRS:
        for f in sorted(glob.glob(os.path.join(HERE, d, "*.csv"))):
            # A shard that is still being written can be empty or header-only;
            # genfromtxt raises on that rather than returning an empty array.
            try:
                a = np.genfromtxt(f, delimiter=",", names=True)
            except IndexError:
                continue
            if a.size == 0:
                continue
            key = (int(a["L"][0]), round(float(a["beta"][0]), 4))
            zval[key] = float(a["z"][0])
            pool[key].append(a)
    pts = []
    for (L, beta), arrs in sorted(pool.items()):
        vals = {c: np.concatenate([a[c] for a in arrs]) for c in COLS}
        n = vals["D"].size
        stats = {c: (float(v.mean()), float(v.std(ddof=1) / np.sqrt(n)))
                 for c, v in vals.items()}
        rel = max(stats[c][1] / abs(stats[c][0]) for c in COLS)
        # `mean`/`sem` are D's, since D is the quantity every headline number
        # is about; per-region values live in `stats` so a ratio computed for
        # SA/SB/SC cannot silently reuse D's mean.
        pts.append(dict(L=L, beta=beta, z=zval[(L, beta)], D=vals["D"], n=n,
                        mean=stats["D"][0], sem=stats["D"][1], stats=stats,
                        rel=float(rel), nfiles=len(arrs)))
    return pts


def error_stats(pts):
    rels = np.array([p["rel"] for p in pts])
    worst = max(pts, key=lambda p: p["rel"])
    print(f"\n== error bars ({len(pts)} points) ==")
    for t in (0.02, 0.03, 0.04, 0.05, 0.08):
        print(f"  <= {t*100:.0f}%: {int((rels <= t).sum())}/{len(pts)}")
    print(f"  median {np.median(rels)*100:.2f}%   worst {worst['rel']*100:.1f}% "
          f"(L={worst['L']} z={worst['z']:.3f} n={worst['n']})")
    print("  above 5%:")
    for p in sorted(pts, key=lambda p: -p["rel"]):
        if p["rel"] > 0.05:
            print(f"    L={p['L']:2d} z={p['z']:8.3f} n={p['n']:6d} "
                  f"{p['rel']*100:5.1f}%")


def group_by_z(pts, rel=0.01):
    """Cluster points into matched-z rows.

    Rounding z to 3 decimals is not enough: the same nominal row carries
    z = 5.934 and 5.935 (different L round differently), and 10.549 vs 10.551.
    Exact keys split those rows in two and silently halve the group.
    """
    byz = defaultdict(list)
    zs = sorted({p["z"] for p in pts})
    reps = []
    for z in zs:
        if reps and abs(z - reps[-1]) / reps[-1] < rel:
            continue
        reps.append(z)
    for p in pts:
        k = min(reps, key=lambda r: abs(p["z"] - r))
        byz[k].append(p)
    return dict(sorted(byz.items()))


def ratio_table(pts):
    """Measured / leading-order per region, weighted over L, at fixed z.

    Leading order is 4 f_R (1-f_R) z^2, i.e. the coefficients 8/9, 56/81, 8/9
    and 16/81 for A, B, C, D (paper section 3.8).
    """
    th = {"SA": 8 / 9, "SB": 56 / 81, "SC": 8 / 9, "D": 16 / 81}
    print("\n== measured / leading order (weighted over L) ==")
    for z0 in (0.089, 0.357, 0.803):
        sel = [p for p in pts if abs(p["z"] - z0) < 0.001]
        if not sel:
            continue
        cells = []
        for c in ("SA", "SB", "SC", "D"):
            r = np.array([p["stats"][c][0] / (th[c] * p["z"] ** 2) for p in sel])
            e = np.array([p["stats"][c][1] / (th[c] * p["z"] ** 2) for p in sel])
            w = 1 / e ** 2
            cells.append(f"{c}={float((w*r).sum()/w.sum()):.3f}"
                         f"+-{float(1/np.sqrt(w.sum())):.3f}")
        print(f"  z={z0:.3f}: " + "  ".join(cells))


def per_z_chi2(pts):
    """chi2/dof of each matched-z row about its weighted mean."""
    print("\n== per-z chi2/dof about a constant ==")
    byz = group_by_z(pts)
    out = []
    for z, sel in byz.items():
        if len(sel) < 2 or z > 11:
            continue
        m = np.array([p["mean"] for p in sel])
        e = np.array([p["sem"] for p in sel])
        w = 1 / e ** 2
        mu = (w * m).sum() / w.sum()
        chi2 = (w * (m - mu) ** 2).sum()
        dof = len(sel) - 1
        out.append((z, chi2 / dof, len(sel)))
        print(f"  z={z:8.3f}  chi2/dof={chi2/dof:5.2f}  ({len(sel)} pts)")
    return out


def blind_prediction(pts):
    """Predict L=30,36 from L<=24 at the same z; report pull statistics."""
    pulls = []
    for z, sel in group_by_z(pts).items():
        if z > 11:
            continue
        ref = [p for p in sel if p["L"] <= 24]
        tgt = [p for p in sel if p["L"] > 24]
        if not ref or not tgt:
            continue
        w = 1 / np.array([p["sem"] for p in ref]) ** 2
        mu = (w * np.array([p["mean"] for p in ref])).sum() / w.sum()
        emu = 1 / np.sqrt(w.sum())
        for p in tgt:
            pulls.append((p["mean"] - mu) / np.hypot(p["sem"], emu))
    pulls = np.array(pulls)
    if pulls.size == 0:
        print("\n== blind prediction: no L>24 points ==")
        return
    chi2 = (pulls ** 2).sum()
    print(f"\n== blind prediction (L<=24 -> L>24) ==")
    print(f"  {pulls.size} points, rms pull {np.sqrt((pulls**2).mean()):.2f}, "
          f"max |pull| {np.abs(pulls).max():.2f}, "
          f"chi2/dof {chi2/pulls.size:.2f}")
    print(f"  negative pulls: {int((pulls < 0).sum())}/{pulls.size}")


def g_drift(pts, ref=18.0):
    """Fit D = f(z)[1 + g ln(L/ref)]: f free per z, g one parameter.

    The profile-likelihood interval at dchi2 = 1 comes from scanning g, since
    the f(z) values are nuisance parameters solved in closed form at each g.
    Only the matched-z rows (z <= 11) enter: the beta-anchored rows at
    z = 16, 44, 119, ... exist at different betas for different L, so they are
    not a test of L-dependence at fixed z and would bias g.
    """
    # z <= 11 keeps only the matched-z rows.  The beta-anchored rows at
    # z = 16, 44, 119, ... sit at a different beta for each L, so they are not
    # a fixed-z comparison; including them flips the large-z g to +0.04.
    rows = {z: sel for z, sel in group_by_z(pts).items()
            if len(sel) >= 2 and z <= 11}

    def chi2_at(g, subset=None):
        tot = 0.0
        for z, sel in rows.items():
            if subset and not subset(z):
                continue
            m = np.array([p["mean"] for p in sel])
            e = np.array([p["sem"] for p in sel])
            x = 1 + g * np.log(np.array([p["L"] for p in sel]) / ref)
            w = 1 / e ** 2
            f = (w * x * m).sum() / (w * x * x).sum()
            tot += (w * (m - f * x) ** 2).sum()
        return tot

    def dof_of(subset=None):
        n = 0
        for z, sel in rows.items():
            if subset and not subset(z):
                continue
            n += len(sel) - 1          # one f(z) fitted per group
        return max(n - 1, 1)

    def fit(subset=None, label=""):
        lo, hi = -0.30, 0.30
        gs = np.linspace(lo, hi, 1201)
        cs = np.array([chi2_at(g, subset) for g in gs])
        g0 = gs[cs.argmin()]
        c0 = cs.min()
        ok = gs[cs <= c0 + 1.0]
        d0 = chi2_at(0.0, subset)
        dof = dof_of(subset)
        sig = np.sqrt(max(d0 - c0, 0))
        print(f"  {label:26s} g={g0:+.4f}  [{ok.min():+.4f},{ok.max():+.4f}]  "
              f"dchi2(g=0)={d0-c0:5.1f} ({sig:.1f}sigma)  chi2/dof={c0/dof:.2f}")
        return g0

    print("\n== g drift, D = f(z)[1 + g ln(L/18)] ==")
    fit(None, "all matched z")
    fit(lambda z: z >= 1.4, "z >= 1.4")
    fit(lambda z: z >= 3.2, "z >= 3.2")
    saved = rows
    rows = {z: [p for p in sel if p["L"] != 6]
            for z, sel in saved.items()}
    rows = {z: sel for z, sel in rows.items() if len(sel) >= 2}
    fit(None, "drop L=6")
    rows = saved


def _chi2_sf(x, k):
    """Survival function of chi-square with k dof (no scipy in this env)."""
    # regularized upper incomplete gamma Q(k/2, x/2), series/continued fraction
    a, xx = k / 2.0, x / 2.0
    if xx <= 0:
        return 1.0
    if xx < a + 1:
        # P(a, x) = x^a e^{-x} / Gamma(a+1) * sum_n x^n / ((a+1)...(a+n)).
        # The prefactor needs Gamma(a+1), not Gamma(a): the first version used
        # lgamma(a), overestimated P by a factor a, and returned NEGATIVE
        # survival probabilities for small chi2 (e.g. -0.13 for chi2=3.8, k=4).
        # The values it was validated on all fell in the continued-fraction
        # branch below, so the series branch was never exercised.
        term, s, n = 1.0, 1.0, 0
        while True:
            n += 1
            term *= xx / (a + n)
            s += term
            if abs(term) < abs(s) * 1e-15 or n > 10000:
                break
        return 1.0 - s * np.exp(-xx + a * np.log(xx) - math.lgamma(a + 1))
    # Lentz continued fraction
    tiny = 1e-300
    b, c, d = xx + 1 - a, 1 / tiny, 1 / (xx + 1 - a)
    h = d
    for i in range(1, 10000):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-15:
            break
    return h * np.exp(-xx + a * np.log(xx) - math.lgamma(a))


def _chi2_sigma(x, k):
    """Equivalent two-sided Gaussian sigma for a chi-square statistic."""
    p = _chi2_sf(x, k)
    if p >= 1 or p <= 0:
        return 0.0
    # invert the normal tail without scipy: bisection on erfc
    lo, hi = 0.0, 40.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if 0.5 * math.erfc(mid / math.sqrt(2)) > p / 2:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def single_scale_test(pts, levels=(0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), nboot=2000):
    """Is a single z_*(L) enough?  Test ln z_h = G(h) + s_L.

    Crossing z_h is interpolated per L, then ln z_h is modelled as a smooth
    function G of h plus one shift s_L per L.  A single crossover scale means
    every threshold crosses at the same L-dependence, i.e. s_L == 0.

    WEIGHTS.  The first version weighted each crossing by 1/(bracket width) in
    ln z.  That was a heuristic with an arbitrary exponent, and the answer was
    entirely determined by it: with weight = width**(-p) the significance ran
    from 1.7 sigma (p = 0) to 3.0 sigma (p = 2), and the quoted 3.0 sigma was
    the p = 2 slice.  Worse, a second bug computed the chi-square as
    sum w^2 r^2 while the fit minimised sum w r^2, so the fit and the statistic
    being reported did not even correspond to the same objective.

    The weight now comes from error propagation instead.  For a crossing
    interpolated at level h between measured points i and i+1,

        t = (h - h_i)/(h_{i+1} - h_i),   ln z = ln z_i + t (ln z_{i+1} - ln z_i)
        dlnz/dh_i     = -(1-t) (Dlnz/Dh),   dlnz/dh_{i+1} = -t (Dlnz/Dh)

    with Var(h) = sem^2 / Dgs^2, so

        Var(ln z) = [(1-t)^2 Var(h_i) + t^2 Var(h_{i+1})] (Dlnz/Dh)^2 .

    Dlnz/Dh is the inverse local slope: a crossing taken across a wide gap in z
    for a small change in h is exactly the ill-determined one, so this
    reproduces the intent of the old heuristic without choosing an exponent.

    SIGNIFICANCE.  The propagated errors treat the crossings as independent,
    but successive crossings of the SAME L are built from the same measured
    points, so the nominal chi2(5) reference is too optimistic.  The p-value
    therefore comes from a residual bootstrap resampled in whole-L blocks,
    which preserves that within-L correlation, and the nominal number is
    reported only alongside it.
    """
    byL = C.per_L(pts)

    def crossings(points_by_L):
        """(h, L, ln z, Var(ln z)) for every well-bracketed crossing."""
        out = []
        for h in levels:
            for L in sorted(points_by_L):
                sel = points_by_L[L]
                zs = np.array([p["z"] for p in sel])
                hs = np.array([p["mean"] / p["Dgs"] for p in sel])
                sd = np.array([p["sem"] / p["Dgs"] for p in sel])
                lz = np.log(zs)
                for i in range(len(hs) - 1):
                    if hs[i] < h <= hs[i + 1]:
                        w = lz[i + 1] - lz[i]
                        dh = hs[i + 1] - hs[i]
                        if w > C.MAX_BRACKET:
                            break
                        t = (h - hs[i]) / dh
                        slope = w / dh            # Dlnz/Dh
                        var = ((1 - t) ** 2 * sd[i] ** 2 + t ** 2 * sd[i + 1] ** 2) * slope ** 2
                        if var > 0 and np.isfinite(var):
                            out.append((h, L, float(lz[i] + t * w), float(var)))
                        break
        return out

    data = crossings(byL)
    Ls = sorted({d[1] for d in data})
    hs_all = sorted({d[0] for d in data})

    def build(dat, free, weights):
        """Design matrix for ln z_h = G(h) + s_L, both blocks linear.

        G is piecewise linear through the crossing levels; s_L is one shift per
        L.  The two blocks are degenerate by a constant (G -> G + c,
        s_L -> s_L - c), so one s column is dropped to keep the system full
        rank.  That changes only the split between G and s, not the fitted
        subspace, so chi-square is unaffected.

        Both blocks are linear, so one lstsq solves the whole thing; the
        coordinate descent in the first version was never needed.
        """
        R, y, w = [], [], []
        for (h, L, lnz, var) in dat:
            r = np.zeros(len(hs_all) + (len(Ls) - 1 if free else 0))
            j = int(np.clip(np.searchsorted(hs_all, h) - 1, 0, len(hs_all) - 2))
            t = (h - hs_all[j]) / (hs_all[j + 1] - hs_all[j])
            r[j], r[j + 1] = 1 - t, t
            if free and L != Ls[0]:
                r[len(hs_all) + Ls.index(L) - 1] = 1.0
            R.append(r)
            y.append(lnz)
            w.append(1.0 / var if weights else 1.0)
        return np.array(R), np.array(y), np.array(w)

    def chi2(dat, free, weights=True):
        A, y, w = build(dat, free, weights)
        sw = np.sqrt(w)
        coef, *_ = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)
        r = y - A @ coef
        return float((w * r * r).sum()), coef

    def lr(dat, weights=True):
        c0, _ = chi2(dat, False, weights)
        c1, coef = chi2(dat, True, weights)
        s = {Ls[0]: 0.0}
        for i, L in enumerate(Ls[1:]):
            s[L] = float(coef[len(hs_all) + i])
        return max(c0 - c1, 0.0), s

    dchi2, shift = lr(data)
    k = len(Ls) - 1
    c0_w, _ = chi2(data, False)
    c1_w, _ = chi2(data, True)
    c0_eq, _ = chi2(data, False, weights=False)
    c1_eq, _ = chi2(data, True, weights=False)
    dchi2_eq, _ = lr(data, weights=False)

    # Permutation test.  Under the null (every L shares one G(h)) the residuals
    # ln z - G(h) are exchangeable across L at each h.  Permuting them within an
    # h level destroys the L-dependent structure while preserving the residual
    # magnitudes, which is exactly the null we want to reject.
    #
    # A residual BLOCK bootstrap (resampling one L's whole residual vector) was
    # tried first and is wrong here: the residuals of the null fit still contain
    # the shift s_L, so respawning them reproduces the signal and gives p = 1.
    # A version weighted by propagated crossing errors was also tried and gives
    # chi2/dof ~ 600 for the shifted model: the piecewise-linear G(h) is
    # misspecified at that precision (residuals ~18 sigma), so the weighted
    # chi-square measures model error, not the presence of s_L.  Equal weights
    # avoid attributing that model error to the physics.
    rng = np.random.default_rng(20261102)
    A, y, w = build(data, False, False)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    fitted = A @ coef
    res = y - fitted
    # positions grouped by h level
    by_h = {}
    for j, d in enumerate(data):
        by_h.setdefault(d[0], []).append(j)
    boot = []
    for _ in range(nboot):
        rp = res.copy()
        for h, js in by_h.items():
            rp[js] = res[rng.permutation(js)]
        dat = [(h, L, float(fitted[j] + rp[j]), var)
               for j, (h, L, lnz, var) in enumerate(data)]
        boot.append(lr(dat, weights=False)[0])
    boot = np.array(boot)

    print("\n== single-scale test: ln z_h = G(h) + s_L ==")
    print(f"  {len(data)} crossings, L={Ls}, h={hs_all}")
    n = len(data)
    dof0 = n - len(hs_all)
    dofs = n - len(hs_all) - (len(Ls) - 1)
    print(f"  equal weights (no arbitrary exponent):")
    print(f"    pure z collapse: chi2/dof = {c0_eq:.1f}/{dof0}")
    print(f"    free shift     : chi2/dof = {c1_eq:.1f}/{dofs}")
    print(f"    likelihood ratio: dchi2 = {dchi2_eq:.1f} for {k} params")
    if boot.size:
        pv = float((boot >= dchi2_eq).mean())
        # Resolution floor: with B draws the smallest reportable p is 1/(B+1).
        floor = 1.0 / (boot.size + 1)
        print(f"    permutation test over L at fixed h ({boot.size} draws): "
              f"null dchi2 median {np.median(boot):.1f}, p95 {np.percentile(boot, 95):.1f}")
        if pv == 0:
            print(f"    -> p < {floor:.1e} (no permutation reached the observed "
                  f"dchi2; nominal chi2(5) would say {_chi2_sigma(dchi2_eq, k):.1f}sigma)")
        else:
            print(f"    -> p = {pv:.3f}")
    # Goodness of fit with the propagated errors, as a check on the MODEL rather
    # than on the physics.  If this is huge the piecewise-linear G(h) does not
    # describe the crossings, and a weighted chi-square would be measuring model
    # error, not a scale mismatch -- so those weights are not used above.
    print(f"  goodness-of-fit with propagated errors: pure {c0_w:.0f}/{dof0}, "
          f"shifted {c1_w:.0f}/{dofs}")
    if c1_w > 3 * dofs:
        print("    -> G(h) is misspecified at the propagated precision "
              "(residuals far exceed the crossing errors), so the weighted")
        print("       chi-square is dominated by model error.  Not used for the test.")
    print("  shifts s_L = " + ", ".join(f"{L}:{shift[L]:+.2f}" for L in Ls))


def main():
    pts = load_pooled()
    print(f"loaded {len(pts)} pooled points, "
          f"{sum(p['n'] for p in pts)} samples total")
    error_stats(pts)
    ratio_table(pts)
    per_z_chi2(pts)
    blind_prediction(pts)
    g_drift(pts)
    p2, base = C.load_points()
    single_scale_test(p2)


if __name__ == "__main__":
    main()
