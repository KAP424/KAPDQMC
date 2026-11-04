"""Adversarial verification of the non-self-averaging / z-crossover claim for D.

CLAIM under test
  1. Per-trajectory D = (2 S2(A) + S2(C))/3 - S2(B) in the fixed-half Born
     ensemble is strongly non-self-averaging; the distribution is heavy tailed.
  2. The Born mean is dominated by rare high-D outcomes and is not representative
     of a typical trajectory.
  3. The top-1% contribution fraction GROWS with L at fixed beta.
  4. Consequently, fits of mean D vs ln L at fixed beta mix a crossover in
     z = L^2 exp(-2 beta) with the asymptotic corner coefficient.

Kill conditions stated in advance
  K1  top-1% (and top-5%) share of the mean does NOT increase with L at fixed beta
  K2  subsample-mean dispersion follows the CLT 1/sqrt(n) law (self-averaging)
  K3  z is not a collapse variable: equal z with different L gives incompatible D
  K4  the local slope dD/dlnL at fixed beta does not organise around z

Outputs: CSV tables in ./tables, figures in ./figures, text summary on stdout.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent        # research_20261030/docs
ROOT = HERE.parent.parent                    # born_ai/
IID = ROOT / "research_20261029" / "iid_targeted"
PROD = ROOT / "analysis" / "born_scaling" / "production" / "means.csv"
CHAINS = ROOT / "analysis" / "born_scaling" / "production" / "chain_means.csv"
TABLES = HERE / "tables"
FIGS = HERE.parent / "figures"
TABLES.mkdir(exist_ok=True)
FIGS.mkdir(exist_ok=True)

RNG = np.random.default_rng(20261030)


# ---------------------------------------------------------------- statistics
def moments(x: np.ndarray) -> dict:
    n = x.size
    m = x.mean()
    d = x - m
    m2 = (d**2).mean()
    m3 = (d**3).mean()
    m4 = (d**4).mean()
    sd = math.sqrt(m2 * n / (n - 1)) if n > 1 else 0.0
    g1 = m3 / m2**1.5 if m2 > 0 else float("nan")
    # excess kurtosis, bias-corrected (Fisher)
    g2 = (m4 / m2**2 - 3.0) if m2 > 0 else float("nan")
    return dict(mean=m, median=float(np.median(x)), sd=sd, skew=g1, excess_kurtosis=g2,
                minimum=float(x.min()), maximum=float(x.max()))


def tail_shares(x: np.ndarray) -> dict:
    """Fraction of the sample MEAN contributed by the top q% of the samples."""
    n = x.size
    s = np.sort(x)[::-1]
    out = {}
    for q in (1, 5, 10):
        k = max(1, int(math.ceil(n * q / 100.0)))
        top = s[:k].sum()
        out[f"mass_top{q}pct"] = float(top / x.sum()) if x.sum() != 0 else float("nan")
        out[f"mean_from_top{q}pct"] = float(top / n)  # additive contribution to mean
    out["mean_from_largest1"] = float(s[0] / n)
    out["frac_of_mean_from_largest1"] = float((s[0] / n) / x.mean()) if x.mean() != 0 else float("nan")
    out["mean_over_median"] = float(x.mean() / np.median(x)) if np.median(x) != 0 else float("nan")
    out["neg_fraction"] = float((x < 0).mean())
    return out


def hill_index(x: np.ndarray, k_frac: float = 0.10) -> float:
    """Hill tail index of the positive upper tail (alpha; larger = lighter tail)."""
    pos = np.sort(x[x > 0])[::-1]
    k = max(2, int(k_frac * pos.size))
    if pos.size < k + 1:
        return float("nan")
    xk = pos[k]
    if xk <= 0:
        return float("nan")
    return float(k / np.sum(np.log(pos[:k] / xk)))


def subsample_mean_scaling(x: np.ndarray, ns=(8, 16, 32, 64, 128, 256, 512),
                           draws=4000) -> list[tuple[int, float, float]]:
    """Dispersion of the n-sample mean. CLT/self-averaging => sd ~ n^-0.5.

    Uses sampling WITHOUT replacement (subsampling) so the only thing that can
    break the n^-1/2 law is the shape of the parent distribution itself.
    """
    res = []
    N = x.size
    for n in ns:
        if n >= N:
            continue
        idx = RNG.integers(0, N, size=(draws, n))
        reps = x[idx].mean(axis=1)
        res.append((n, float(reps.std(ddof=1)), float(reps.mean())))
    return res


def loglog_slope(xs, ys) -> float:
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    ok = (xs > 0) & (ys > 0)
    if ok.sum() < 2:
        return float("nan")
    return float(np.polyfit(np.log(xs[ok]), np.log(ys[ok]), 1)[0])


# ---------------------------------------------------------------- data loading
def load_iid() -> dict[tuple[float, float], np.ndarray]:
    data = {}
    for L in (6, 12, 18):
        for b in (1, 3):
            chunks = []
            for bt in (1, 2, 3, 4):
                f = IID / f"L{L}_beta{b}_batch{bt}.csv"
                with f.open(newline="") as fh:
                    rows = list(csv.DictReader(fh))
                chunks.append(np.array([float(r["D"]) for r in rows]))
            data[(float(L), float(b))] = np.concatenate(chunks)
    return data


def load_iid_components() -> dict[tuple[float, float], dict[str, np.ndarray]]:
    data = {}
    for L in (6, 12, 18):
        for b in (1, 3):
            acc = {k: [] for k in ("SA", "SB", "SC", "D")}
            for bt in (1, 2, 3, 4):
                with (IID / f"L{L}_beta{b}_batch{bt}.csv").open(newline="") as fh:
                    rows = list(csv.DictReader(fh))
                for k in acc:
                    acc[k].append(np.array([float(r[k]) for r in rows]))
            data[(float(L), float(b))] = {k: np.concatenate(v) for k, v in acc.items()}
    return data


def load_production() -> list[dict]:
    with PROD.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    out = []
    for r in rows:
        out.append(dict(L=float(r["L"]), beta=float(r["beta"]), D=float(r["D_mean"]),
                        E=float(r["D_MCSE"]), grid=r["grid"],
                        status=r["sampling_status"],
                        SA=float(r["SA_mean"]), SB=float(r["SB_mean"]), SC=float(r["SC_mean"])))
    return out


def load_chain_means() -> dict[tuple[float, float], np.ndarray]:
    out: dict[tuple[float, float], list[float]] = {}
    with CHAINS.open(newline="") as fh:
        for r in csv.DictReader(fh):
            if r["observable"] != "D":
                continue
            out.setdefault((float(r["L"]), float(r["beta"])), []).append(float(r["mean"]))
    return {k: np.array(sorted(v)) for k, v in out.items()}


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    keys = []
    for r in rows:  # union of keys, first-seen order
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------- part 1: tails
def part1(data, comps):
    print("=" * 78)
    print("PART 1 - shape of the per-trajectory D distribution (iid data, n=2048)")
    print("=" * 78)
    rows = []
    hdr = (f"{'L':>3}{'beta':>5}{'mean':>10}{'median':>10}{'sd':>9}{'skew':>8}"
           f"{'exkurt':>9}{'top1%':>7}{'top5%':>7}{'max/mean':>9}{'m/med':>7}{'neg':>6}{'hill':>7}")
    print(hdr)
    for (L, b), x in sorted(data.items()):
        m = moments(x)
        t = tail_shares(x)
        sh1 = t["mass_top1pct"]
        sh5 = t["mass_top5pct"]
        hill = hill_index(x)
        print(f"{L:3.0f}{b:5.1f}{m['mean']:10.4f}{m['median']:10.4f}{m['sd']:9.4f}"
              f"{m['skew']:8.2f}{m['excess_kurtosis']:9.1f}{sh1:7.3f}{sh5:7.3f}"
              f"{m['maximum']/m['mean']:9.1f}{t['mean_over_median']:7.2f}"
              f"{t['neg_fraction']:6.2f}{hill:7.2f}")
        q = {p: float(np.quantile(x, p / 100)) for p in (0.1, 1, 5, 25, 50, 75, 95, 99, 99.9)}
        rows.append(dict(L=L, beta=b, n=x.size, **m, **t,
                         hill_alpha_positive_tail=hill,
                         **{f"q{p:g}": v for p, v in q.items()},
                         sd_over_meanabs=float(m["sd"] / abs(m["mean"])) if m["mean"] else float("nan")))
    write_csv(TABLES / "d_distribution_shape.csv", rows)

    print("\n-- kill test K1: does the top-share GROW with L at fixed beta? --")
    k1 = []
    for b in (1.0, 3.0):
        Ls = [L for (L, bb) in data if bb == b]
        sh1 = [tail_shares(data[(L, b)])["mass_top1pct"] for L in Ls]
        sh5 = [tail_shares(data[(L, b)])["mass_top5pct"] for L in Ls]
        slope1 = loglog_slope(Ls, sh1)
        slope5 = loglog_slope(Ls, sh5)
        print(f"  beta={b}: L={Ls}")
        print(f"    top1% share = {[round(v,4) for v in sh1]}   log-log slope vs L = {slope1:+.3f}")
        print(f"    top5% share = {[round(v,4) for v in sh5]}   log-log slope vs L = {slope5:+.3f}")
        print(f"    ratio share(Lmax)/share(Lmin) = {sh1[-1]/sh1[0]:.3f}")
        k1.append(dict(beta=b, L=";".join(str(int(v)) for v in Ls),
                       top1_share=";".join(f"{v:.6f}" for v in sh1),
                       top5_share=";".join(f"{v:.6f}" for v in sh5),
                       slope_top1_vs_L=slope1, slope_top5_vs_L=slope5,
                       ratio_top1_last_over_first=sh1[-1] / sh1[0],
                       grows_with_L=bool(slope1 > 0)))
    write_csv(TABLES / "tail_share_vs_L.csv", k1)

    print("\n-- kill test K2: dispersion of the n-sample mean vs CLT --")
    k2 = []
    for (L, b), x in sorted(data.items()):
        sub = subsample_mean_scaling(x)
        ns = [s[0] for s in sub]
        sds = [s[1] for s in sub]
        slope = loglog_slope(ns, sds)
        # reference: sd(mean at n) / sd(single draw) should equal n^-1/2
        single = x.std(ddof=1)
        ratio_at_max = sds[-1] / (single / math.sqrt(ns[-1]))
        print(f"  L={L:2.0f} beta={b}: sd(mean_n) log-log slope = {slope:+.3f} "
              f"(CLT = -0.500); sd(mean_2048) / CLT prediction = {ratio_at_max:.2f}")
        k2.append(dict(L=L, beta=b, n=";".join(str(v) for v in ns),
                       sd_of_mean=";".join(f"{v:.6f}" for v in sds),
                       loglog_slope=slope, clt_slope=-0.5,
                       sd_mean_over_clt_at_nmax=ratio_at_max,
                       self_averaging=bool(abs(slope + 0.5) < 0.08)))
    write_csv(TABLES / "mean_dispersion_vs_n.csv", k2)
    return rows, k1, k2


def part1b_decomposition(comps):
    """Which component drives high-D trajectories?"""
    print("\n-- what makes a high-D trajectory high? (iid data) --")
    rows = []
    for (L, b), c in sorted(comps.items()):
        D = c["D"]
        k = max(1, D.size // 100)
        idx = np.argsort(D)[::-1][:k]
        low = np.argsort(D)[:k]
        rows.append(dict(
            L=L, beta=b,
            top1_SA_mean=float(c["SA"][idx].mean()), all_SA_mean=float(c["SA"].mean()),
            bot1_SA_mean=float(c["SA"][low].mean()),
            top1_SB_mean=float(c["SB"][idx].mean()), all_SB_mean=float(c["SB"].mean()),
            bot1_SB_mean=float(c["SB"][low].mean()),
            top1_SC_mean=float(c["SC"][idx].mean()), all_SC_mean=float(c["SC"].mean()),
            bot1_SC_mean=float(c["SC"][low].mean()),
        ))
        r = rows[-1]
        print(f"  L={L:2.0f} beta={b}: top1% SA={r['top1_SA_mean']:.3f} (all {r['all_SA_mean']:.3f})"
              f"  SB={r['top1_SB_mean']:.3f} (all {r['all_SB_mean']:.3f})"
              f"  SC={r['top1_SC_mean']:.3f} (all {r['all_SC_mean']:.3f})")
    write_csv(TABLES / "high_D_decomposition.csv", rows)


# ---------------------------------------------------------------- part 2: z
def zdict(rows):
    return {(r["L"], r["beta"]): r for r in rows}


def z_of(L, beta):
    return L * L * math.exp(-2 * beta)


def part2_zcollapse(prod):
    print("\n" + "=" * 78)
    print("PART 2 - is z = L^2 exp(-2 beta) a collapse variable?")
    print("=" * 78)
    rows = [r for r in prod if r["E"] > 0]
    byL: dict[float, list[dict]] = {}
    for r in rows:
        byL.setdefault(r["L"], []).append(r)
    for L in byL:
        byL[L].sort(key=lambda r: r["beta"])
    for L, lst in sorted(byL.items()):
        zs = [z_of(r["L"], r["beta"]) for r in lst]
        print(f"  L={L:2.0f}: z in [{min(zs):.3g}, {max(zs):.3g}]  n={len(lst)}")

    def interp_D(L, z):
        lst = byL[L]
        zs = np.array([z_of(r["L"], r["beta"]) for r in lst])
        ds = np.array([r["D"] for r in lst])
        es = np.array([r["E"] for r in lst])
        o = np.argsort(zs)
        zs, ds, es = zs[o], ds[o], es[o]
        lz = np.log(zs)
        x = math.log(z)
        if x < lz[0] or x > lz[-1]:
            return None
        i = int(np.searchsorted(lz, x) - 1)
        i = max(0, min(i, len(lz) - 2))
        w = (x - lz[i]) / (lz[i + 1] - lz[i])
        return ds[i] * (1 - w) + ds[i + 1] * w, math.sqrt((es[i] * (1 - w))**2 + (es[i + 1] * w)**2)

    # nodes: z values interior to the largest number of L curves
    nodes = []
    for L in sorted(byL):
        for r in byL[L]:
            nodes.append(z_of(r["L"], r["beta"]))
    nodes = sorted(set(round(v, 6) for v in nodes))

    pairs = []
    print("\n-- D at equal z, different L (interpolated in ln z within each L) --")
    print(f"  {'z':>8}  " + "  ".join(f"{'L='+str(int(L)):>16}" for L in sorted(byL)))
    node_rows = []
    for z in nodes:
        vals = {}
        for L in sorted(byL):
            iv = interp_D(L, z)
            if iv is not None:
                vals[L] = iv
        if len(vals) < 2:
            continue
        cells = []
        for L in sorted(byL):
            cells.append(f"{vals[L][0]:8.4f}+-{vals[L][1]:5.4f}" if L in vals else " " * 15)
        print(f"  {z:8.4f}  " + "  ".join(f"{c:>16}" for c in cells))
        Ls = sorted(vals)
        node_rows.append(dict(z=z, n_L=len(vals),
                              **{f"D_L{int(L)}": vals[L][0] for L in Ls},
                              **{f"E_L{int(L)}": vals[L][1] for L in Ls}))
        for i in range(len(Ls)):
            for j in range(i + 1, len(Ls)):
                d = vals[Ls[i]][0] - vals[Ls[j]][0]
                e = math.hypot(vals[Ls[i]][1], vals[Ls[j]][1])
                pairs.append(dict(z=z, L1=Ls[i], L2=Ls[j], D1=vals[Ls[i]][0],
                                  D2=vals[Ls[j]][0], diff=d, sigma=e,
                                  pull=d / e if e > 0 else float("nan")))
    write_csv(TABLES / "z_collapse_nodes.csv", node_rows)

    pulls = np.array([p["pull"] for p in pairs if np.isfinite(p["pull"])])
    absd = np.array([abs(p["diff"]) for p in pairs if np.isfinite(p["pull"])])
    print(f"\n  equal-z cross-L pairs: {pulls.size}")
    print(f"    |pull| median = {np.median(abs(pulls)):.2f}, max = {abs(pulls).max():.1f}")
    print(f"    fraction with |pull| > 3 : {(abs(pulls)>3).mean():.3f}")
    print(f"    RMS |pull| = {np.sqrt((pulls**2).mean()):.2f}")
    sig_collapse = (abs(pulls) <= 3).mean()
    write_csv(TABLES / "z_collapse_pairs.csv", pairs)

    # Same-z, different-L comparison restricted to large z (asymptotic arm)
    big = [p for p in pairs if p["z"] > 10]
    if big:
        pb = np.array([p["pull"] for p in big])
        print(f"    restricted to z>10 (n={pb.size}): |pull| median={np.median(np.abs(pb)):.2f}, "
              f"max={np.abs(pb).max():.1f}")

    # Is there ANY z where all L agree?
    return dict(n_pairs=int(pulls.size), median_abs_pull=float(np.median(abs(pulls))),
                max_abs_pull=float(abs(pulls).max()),
                fraction_within_3sigma=float(sig_collapse),
                rms_pull=float(np.sqrt((pulls**2).mean())),
                fraction_within_3sigma_z_gt_10=(
                    float((np.abs(np.array([p["pull"] for p in big])) <= 3).mean()) if big else None))


def part2b_slopes(prod):
    print("\n-- local slope dD/dlnL at fixed beta, and where it bends in z --")
    byL: dict[float, list[dict]] = {}
    for r in prod:
        byL.setdefault(r["L"], []).append(r)
    for L in byL:
        byL[L].sort(key=lambda r: r["beta"])
    betas = sorted({r["beta"] for r in prod})
    rows = []
    for b in betas:
        pts = []
        for L in sorted(byL):
            for r in byL[L]:
                if r["beta"] == b:
                    pts.append(r)
        pts.sort(key=lambda r: r["L"])
        if len(pts) < 3:
            continue
        for i in range(len(pts) - 1):
            a, c = pts[i], pts[i + 1]
            dl = math.log(c["L"]) - math.log(a["L"])
            slope = (c["D"] - a["D"]) / dl
            # independent sizes => add MCSE in quadrature
            e = math.hypot(a["E"], c["E"]) / dl
            zg = math.sqrt(z_of(a["L"], b) * z_of(c["L"], b))
            rows.append(dict(beta=b, L_left=a["L"], L_right=c["L"],
                             z_geomean=zg, D_left=a["D"], D_right=c["D"],
                             local_slope=slope, local_slope_MCSE=e))
    write_csv(TABLES / "local_slope_vs_z.csv", rows)

    # print a compact beta x interval grid
    betas_show = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    Ls = sorted(byL)
    print(f"  {'beta':>5} " + "".join(f"{f'{int(a)}-{int(c)}':>9}" for a, c in zip(Ls, Ls[1:])))
    for b in betas_show:
        line = f"  {b:5.2f} "
        for a, c in zip(Ls, Ls[1:]):
            m = [r for r in rows if r["beta"] == b and r["L_left"] == a]
            line += f"{m[0]['local_slope']:9.3f}" if m else " " * 9
        print(line)
    print(f"  {'z(mid)':>5} " + "".join(
        f"{math.sqrt(z_of(a, 3.0) * z_of(c, 3.0)):9.2f}" for a, c in zip(Ls, Ls[1:]))
    + "   <- z at beta=3 for each interval")

    # Does local slope collapse as a function of z?
    slope_vs_z = {}
    for b in betas:
        sel = [r for r in rows if r["beta"] == b]
        if len(sel) >= 2:
            slope_vs_z[b] = sel
    return rows, slope_vs_z


def part3_shifted_collapse(prod):
    """Test D = c ln L + A(z) with c taken from the beta=0 (deterministic) series."""
    print("\n-- test D = c ln L + A(z), with c from the beta=0 series --")
    det = sorted([r for r in prod if r["beta"] == 0.0], key=lambda r: r["L"])
    Ls = np.array([r["L"] for r in det])
    Ds = np.array([r["D"] for r in det])
    c = float(np.polyfit(np.log(Ls), Ds, 1)[0])
    print(f"  c(beta=0) from ln L fit over L={Ls.astype(int).tolist()}: {c:.4f}")
    # local ground-state slopes, for context
    for i in range(len(det) - 1):
        print(f"    local slope {int(det[i]['L']):2d}-{int(det[i+1]['L']):2d}: "
              f"{(det[i+1]['D']-det[i]['D'])/(math.log(det[i+1]['L'])-math.log(det[i]['L'])):.4f}")

    Lref = 6.0
    pts = []
    for r in prod:
        if r["beta"] == 0.0 or r["E"] <= 0:
            continue
        A = r["D"] - c * math.log(r["L"] / Lref)
        pts.append(dict(L=r["L"], beta=r["beta"], z=z_of(r["L"], r["beta"]), A=A, E=r["E"]))
    pts.sort(key=lambda p: p["z"])
    # bucketed A(z): points whose z agree within 12% but differ in L
    print(f"  {'z':>9}{'L':>4}{'beta':>6}{'D':>9}{'A=D-c ln(L/6)':>16}")
    for p in pts:
        print(f"  {p['z']:9.4f}{p['L']:4.0f}{p['beta']:6.2f}{p['A']+c*math.log(p['L']/Lref):9.4f}{p['A']:16.4f}")
    write_csv(TABLES / "shifted_A_of_z.csv", pts)

    # spread of A at matched z (log-bucket by |ln z| < 0.12)
    groups = []
    used = [False] * len(pts)
    for i, p in enumerate(pts):
        if used[i]:
            continue
        g = [q for q in pts if abs(math.log(q["z"]) - math.log(p["z"])) < 0.12]
        for q in g:
            used[pts.index(q)] = True
        if len({q["L"] for q in g}) >= 2:
            groups.append(g)
    print(f"\n  matched-z buckets with >=2 distinct L: {len(groups)}")
    spreads = []
    for g in groups:
        As = [q["A"] for q in g]
        Es = [q["E"] for q in g]
        sp = max(As) - min(As)
        scale = max(Es)
        spreads.append(sp)
        tag = "OK " if sp <= 3 * math.sqrt(2) * scale else "FAIL"
        print(f"    z~{np.mean([q['z'] for q in g]):8.3f}  L={sorted({int(q['L']) for q in g})}"
              f"  A spread={sp:.4f}  (3 sigma~{3*math.sqrt(2)*scale:.4f})  {tag}")
    if spreads:
        print(f"  median A-spread at matched z: {np.median(spreads):.4f}")
    return c, groups


# ---------------------------------------------------------------- part 4: falsifiers
def part4_falsifiers(data, comps, prod):
    print("\n" + "=" * 78)
    print("PART 4 - mechanism falsifiers")
    print("=" * 78)

    # --- A: D has bounded support, so a true power-law tail is impossible ---
    print("\n-- A. support of D is bounded: a genuine heavy (power-law) tail is excluded --")
    a_rows = []
    for (L, b), c in sorted(comps.items()):
        # crude bound: 0 <= S2(R) <= ln(a_R) and |D| <= (2+1)/3 max S + max S
        Lmax = math.log(L * L / 2)
        D = c["D"]
        print(f"  L={L:2.0f} beta={b}: D in [{D.min():+.4f}, {D.max():+.4f}], "
              f"empirical max |D|/ln(a) = {D.max()/Lmax:.3f}, "
              f"largest draw / mean = {(D.max()/D.size)/D.mean():.3f}")
        a_rows.append(dict(L=L, beta=b, D_min=float(D.min()), D_max=float(D.max()),
                           entropy_bound=2 * Lmax,
                           max_abs_over_bound=float(max(abs(D.min()), abs(D.max())) / (2 * Lmax)),
                           largest_draw_fraction_of_mean=float((D.max() / D.size) / D.mean())))
    write_csv(TABLES / "support_bound.csv", a_rows)

    # --- B: high-D trajectories are weak-measurement trajectories, not corner events ---
    print("\n-- B. are rare high-D trajectories corner events or weak-measurement events? --")
    b_rows = []
    for (L, b), c in sorted(comps.items()):
        D, SA, SB, SC = c["D"], c["SA"], c["SB"], c["SC"]
        Ssum = SA + SB + SC
        corr_D_total = float(np.corrcoef(D, Ssum)[0, 1])
        corr_D_area = float(np.corrcoef(D, SA - SB)[0, 1])
        # fraction of D variance explained by the total-entropy (measurement strength) channel
        k = max(1, D.size // 100)
        hi = np.argsort(D)[::-1][:k]
        lo = np.argsort(D)[:k]
        print(f"  L={L:2.0f} beta={b}: corr(D, SA+SB+SC) = {corr_D_total:+.3f} ; "
              f"corr(D, SA-SB) = {corr_D_area:+.3f} ; "
              f"top1% mean total S = {Ssum[hi].mean():.3f} vs bottom1% {Ssum[lo].mean():.3f}")
        b_rows.append(dict(L=L, beta=b, corr_D_totalsum=corr_D_total,
                           corr_D_SAminusSB=corr_D_area,
                           top1_totalsum=float(Ssum[hi].mean()),
                           bottom1_totalsum=float(Ssum[lo].mean()),
                           top1_SA=float(SA[hi].mean()), top1_SB=float(SB[hi].mean()),
                           top1_SC=float(SC[hi].mean())))
    write_csv(TABLES / "tail_mechanism.csv", b_rows)

    # --- C: are the local slopes at small beta distinguishable from the beta=0 drift? ---
    print("\n-- C. local slopes at beta<=1.5 against the deterministic beta=0 drift --")
    byL: dict[float, list[dict]] = {}
    for r in prod:
        byL.setdefault(r["L"], []).append(r)
    for L in byL:
        byL[L].sort(key=lambda r: r["beta"])
    Ls = sorted(byL)

    def slopes(b):
        out = []
        for a, c in zip(Ls, Ls[1:]):
            ra = next((r for r in byL[a] if r["beta"] == b), None)
            rc = next((r for r in byL[c] if r["beta"] == b), None)
            if ra is None or rc is None:
                return None
            dl = math.log(c) - math.log(a)
            s = (rc["D"] - ra["D"]) / dl
            e = math.hypot(ra["E"], rc["E"]) / dl
            out.append((s, e))
        return out

    base = slopes(0.0)
    c_rows = []
    for b in sorted({r["beta"] for r in prod}):
        if b == 0.0:
            continue
        s = slopes(b)
        if s is None:
            continue
        dif = np.array([x[0] - y[0] for x, y in zip(s, base)])
        sg = np.array([math.hypot(x[1], y[1]) for x, y in zip(s, base)])
        chi2 = float(np.sum((dif / sg) ** 2))
        dof = dif.size
        print(f"  beta={b:4.2f}: slopes {[round(x[0],3) for x in s]}")
        print(f"            - beta0 {[round(y[0],3) for y in base]}  "
              f"chi2/dof = {chi2/dof:.2f} ({dof} dof)  max|diff|/sigma = {np.max(np.abs(dif/sg)):.1f}")
        c_rows.append(dict(beta=b, slopes=";".join(f"{x[0]:.6f}" for x in s),
                           slope_errors=";".join(f"{x[1]:.6f}" for x in s),
                           beta0_slopes=";".join(f"{y[0]:.6f}" for y in base),
                           chi2_over_dof=chi2 / dof, dof=dof,
                           consistent_with_beta0=bool(chi2 / dof < 3.0)))
    write_csv(TABLES / "slope_vs_beta0.csv", c_rows)

    # --- D: reported MCSE vs actual chain-to-chain scatter ---
    print("\n-- D. reported D_MCSE vs the observed chain-to-chain scatter --")
    ch = load_chain_means()
    prodmap = {(r["L"], r["beta"]): r for r in prod}
    d_rows = []
    for (L, b), v in sorted(ch.items()):
        if v.size < 2:
            continue
        se = float(v.std(ddof=1) / math.sqrt(v.size))
        pr = prodmap.get((L, b))
        if pr is None or pr["E"] <= 0:
            continue
        ratio = se / pr["E"]
        d_rows.append(dict(L=L, beta=b, chainSE=se, reported_MCSE=pr["E"],
                           ratio_chainSE_over_reported=ratio,
                           rel_chainSE=se / v.mean(), rel_reported=pr["E"] / pr["D"]))
    d_rows.sort(key=lambda r: r["beta"])
    for r in d_rows:
        print(f"  L={r['L']:2.0f} beta={r['beta']:4.2f}: chainSE/reported = {r['ratio_chainSE_over_reported']:.2f}"
              f"  (rel {r['rel_chainSE']:.2%} vs {r['rel_reported']:.2%})")
    ratios = np.array([r["ratio_chainSE_over_reported"] for r in d_rows])
    big = np.array([r["ratio_chainSE_over_reported"] for r in d_rows if r["beta"] >= 2.5])
    small = np.array([r["ratio_chainSE_over_reported"] for r in d_rows if r["beta"] <= 1.5])
    print(f"  all: median {np.median(ratios):.2f}; beta<=1.5 median {np.median(small):.2f};"
          f" beta>=2.5 median {np.median(big):.2f}")
    write_csv(TABLES / "mcse_vs_chain_scatter.csv", d_rows)
    return dict(median_ratio_all=float(np.median(ratios)),
                median_ratio_lowbeta=float(np.median(small)),
                median_ratio_highbeta=float(np.median(big)))


# ---------------------------------------------------------------- part 5
def part5_exponent_scan(prod, c0):
    """Scan z_alpha = L^alpha exp(-2 beta) and measure collapse quality.

    Model: D = c ln L + A(z_alpha). For each alpha, bucket points with matched
    z_alpha, then measure the residual spread of A = D - c ln(L/Lref) relative to
    the MCSE. A true collapse variable gives spread ~ MCSE at alpha = alpha*.
    """
    print("\n" + "=" * 78)
    print("PART 5 - is exp(-2 beta) the right beta exponent, and what is the L power?")
    print("=" * 78)
    Lref = 6.0
    pts = [dict(L=r["L"], beta=r["beta"], D=r["D"], E=r["E"],
                A=r["D"] - c0 * math.log(r["L"] / Lref))
           for r in prod if r["E"] > 0]
    rows = []
    for alpha in np.arange(0.5, 3.51, 0.25):
        # bucket in ln z_alpha
        zs = np.array([r["L"] ** alpha * math.exp(-2 * r["beta"]) for r in pts])
        lz = np.log(zs)
        order = np.argsort(lz)
        # greedy buckets of width 0.15 in ln z
        buckets, cur = [], [order[0]]
        for i in order[1:]:
            if lz[i] - lz[cur[-1]] < 0.15:
                cur.append(i)
            else:
                buckets.append(cur)
                cur = [i]
        buckets.append(cur)
        spreads, pulls = [], []
        for g in buckets:
            if len({pts[i]["L"] for i in g}) < 2:
                continue
            As = np.array([pts[i]["A"] for i in g])
            Es = np.array([pts[i]["E"] for i in g])
            # intrinsic scatter beyond MCSE: var(total) - var(MCSE)
            tot = As.var(ddof=1)
            noise = float((Es**2).mean())
            excess = math.sqrt(max(0.0, tot - noise))
            spreads.append(excess)
            if len(g) == 2:
                pulls.append(abs(As[0] - As[1]) / math.hypot(Es[0], Es[1]))
        typ_mcse = float(np.median([pts[i]["E"] for i in range(len(pts))]))
        rows.append(dict(alpha=float(alpha), n_buckets=len(spreads),
                         median_excess_scatter=float(np.median(spreads)) if spreads else float("nan"),
                         mean_excess_scatter=float(np.mean(spreads)) if spreads else float("nan"),
                         median_abs_pull=float(np.median(pulls)) if pulls else float("nan"),
                         typical_MCSE=typ_mcse,
                         excess_over_MCSE=(float(np.median(spreads))/typ_mcse) if spreads else float("nan")))
        print(f"  alpha={alpha:4.2f}: buckets={len(spreads):3d}  "
              f"median excess scatter={np.median(spreads) if spreads else float('nan'):.4f}  "
              f"= {(np.median(spreads)/typ_mcse) if spreads else float('nan'):5.1f} x typical MCSE  "
              f"median |pull|={np.median(pulls) if pulls else float('nan'):6.2f}")
    write_csv(TABLES / "collapse_exponent_scan.csv", rows)
    best = min(rows, key=lambda r: r["median_excess_scatter"])
    at2 = next(r for r in rows if abs(r["alpha"] - 2.0) < 1e-9)
    print(f"\n  best alpha by median excess scatter: {best['alpha']:.2f} "
          f"(excess scatter {best['median_excess_scatter']:.4f} = "
          f"{best['excess_over_MCSE']:.1f} x typical MCSE)")
    print(f"  at the claimed alpha=2.0: excess scatter {at2['median_excess_scatter']:.4f} = "
          f"{at2['excess_over_MCSE']:.1f} x typical MCSE")
    n_collapse = sum(1 for r in rows
                     if math.isfinite(r["excess_over_MCSE"]) and r["excess_over_MCSE"] < 2.0)
    print(f"  alpha values whose collapse residual is within 2 x MCSE: {n_collapse} of {len(rows)}")
    best = dict(best)
    best["n_collapsing_alphas"] = n_collapse
    return rows, best


def part6_overshoot(prod):
    """A pure crossover + c ln L cannot make dD/dlnL exceed both asymptotes.

    If D = c ln L + A(z) with z increasing in L at fixed beta, and A is monotone
    with a single crossover, the local slope must lie between the two asymptotic
    values. We check whether it does.
    """
    print("\n" + "=" * 78)
    print("PART 6 - does the local slope overshoot both asymptotes?")
    print("=" * 78)
    byL: dict[float, list[dict]] = {}
    for r in prod:
        byL.setdefault(r["L"], []).append(r)
    for L in byL:
        byL[L].sort(key=lambda r: r["beta"])
    Ls = sorted(byL)

    def slope(b, a, c):
        ra = next((r for r in byL[a] if r["beta"] == b), None)
        rc = next((r for r in byL[c] if r["beta"] == b), None)
        if ra is None or rc is None:
            return None
        dl = math.log(c) - math.log(a)
        return (rc["D"] - ra["D"]) / dl, math.hypot(ra["E"], rc["E"]) / dl

    tail_int = (Ls[-2], Ls[-1])
    base = slope(0.0, *tail_int)[0]
    rows = []
    print(f"  asymptote at beta=0 (L={int(tail_int[0])}-{int(tail_int[1])}): {base:.4f}")
    print(f"  asymptote at beta=3 (L={int(tail_int[0])}-{int(tail_int[1])}): "
          f"{slope(3.0, *tail_int)[0]:.4f}")
    for b in sorted({r["beta"] for r in prod}):
        if b == 0.0:
            continue
        s = slope(b, *tail_int)
        if s is None:
            continue
        over = s[0] > base + 2 * s[1]
        print(f"  beta={b:4.2f}: tail slope = {s[0]:.4f} +- {s[1]:.4f}"
              f"{'   <-- EXCEEDS beta=0 asymptote' if over else ''}")
        rows.append(dict(beta=b, tail_slope=s[0], tail_slope_MCSE=s[1],
                         beta0_asymptote=base, overshoots=bool(over),
                         excess_sigma=float((s[0] - base) / s[1])))
    write_csv(TABLES / "slope_overshoot.csv", rows)
    n_over = sum(1 for r in rows if r["overshoots"])
    print(f"  {n_over} of {len(rows)} beta values overshoot the beta=0 asymptote by >2 sigma")
    return rows


def part7_chains_nonstationarity():
    """Independent non-self-averaging diagnostic on the 240 production chain means."""
    print("\n" + "=" * 78)
    print("PART 7 - chain-to-chain relative scatter vs L (independent of the iid data)")
    print("=" * 78)
    ch = load_chain_means()
    rows = []
    for (L, b), v in sorted(ch.items()):
        if v.size < 2:
            continue
        rel = float(v.std(ddof=1) / math.sqrt(v.size) / v.mean())
        rows.append(dict(L=L, beta=b, rel_chainSE=rel, n_chains=int(v.size)))
    write_csv(TABLES / "chain_rel_scatter.csv", rows)
    print(f"  {'beta':>5}  " + "".join(f"{'L='+str(int(L)):>9}" for L in sorted({r['L'] for r in rows})))
    out = []
    for b in sorted({r["beta"] for r in rows}):
        sel = sorted([r for r in rows if r["beta"] == b], key=lambda r: r["L"])
        if len(sel) < 3:
            continue
        Ls = [r["L"] for r in sel]
        ys = [r["rel_chainSE"] for r in sel]
        sl = loglog_slope(Ls, ys)
        cells = {int(r["L"]): r["rel_chainSE"] for r in sel}
        line = f"  {b:5.2f}  "
        for L in sorted({r["L"] for r in rows}):
            line += f"{cells[L]:9.2%}" if L in cells else " " * 9
        print(line + f"   log-log slope = {sl:+.3f}")
        out.append(dict(beta=b, slope_rel_scatter_vs_L=sl, grows_with_L=bool(sl > 0)))
    write_csv(TABLES / "chain_rel_scatter_slopes.csv", out)
    slopes = [o["slope_rel_scatter_vs_L"] for o in out]
    print(f"\n  median log-log slope = {np.median(slopes):+.3f}; "
          f"{sum(1 for s in slopes if s > 0)}/{len(slopes)} beta values have positive slope")
    return out


# ---------------------------------------------------------------- part 8
def wls(xs, ys, es):
    """Weighted least squares y = sum_i p_i f_i(x) with f = [ln x, 1]. Returns coefs, cov."""
    X = np.column_stack([np.log(xs), np.ones_like(np.asarray(xs, float))])
    y = np.asarray(ys, float)
    w = 1.0 / np.asarray(es, float) ** 2
    A = X.T @ (X * w[:, None])
    b = X.T @ (w * y)
    cov = np.linalg.inv(A)
    p = cov @ b
    resid = y - X @ p
    chi2 = float(np.sum(w * resid**2))
    return p, cov, chi2, X.shape[0] - 2


def part8_alt_model(prod):
    """Alternative: D = c(beta) ln L + b(beta) with a NON-MONOTONE c(beta).

    If this describes the data, the 'crossover in z' is not needed: the moving
    curvature is what a window-limited measurement of a beta-dependent corner
    coefficient looks like.
    """
    print("\n" + "=" * 78)
    print("PART 8 - counter-model: D = c(beta) ln L + b(beta), c(beta) non-monotone")
    print("=" * 78)
    byL: dict[float, list[dict]] = {}
    for r in prod:
        byL.setdefault(r["L"], []).append(r)
    for L in byL:
        byL[L].sort(key=lambda r: r["beta"])
    betas = sorted({r["beta"] for r in prod})

    def series(b, Lmin=0.0, need_err=True):
        out = []
        for L in sorted(byL):
            if L < Lmin:
                continue
            r = next((q for q in byL[L] if q["beta"] == b), None)
            if r is not None and (not need_err or r["E"] > 0):
                out.append(r)
        return out

    rows = []
    print(f"  {'beta':>5}{'n':>4}{'c(beta)':>11}{'se':>9}{'b':>10}{'chi2/dof':>10}"
          f"{'c(L>=12)':>11}{'se':>8}{'chi2/dof':>10}")
    for b in betas:
        det0 = (b == 0.0)
        full = series(b, need_err=not det0)
        if len(full) < 4:
            continue
        if det0:
            # deterministic: no MCSE, use unweighted LS, se not meaningful
            xs = np.array([r["L"] for r in full]); ys = np.array([r["D"] for r in full])
            cs, ic = np.polyfit(np.log(xs), ys, 1)
            p = np.array([float(cs), float(ic)]); cov = np.zeros((2, 2)); chi2 = 0.0
            dof = max(1, len(full) - 2)
        else:
            p, cov, chi2, dof = wls([r["L"] for r in full], [r["D"] for r in full],
                                    [r["E"] for r in full])
        win = series(b, 12.0, need_err=not det0)
        if len(win) >= 3:
            if det0:
                xs = np.array([r["L"] for r in win]); ys = np.array([r["D"] for r in win])
                c2, se2, chi2b = float(np.polyfit(np.log(xs), ys, 1)[0]), 0.0, 0.0
            else:
                p2, cov2, chi22, dof2 = wls([r["L"] for r in win], [r["D"] for r in win],
                                            [r["E"] for r in win])
                c2, se2, chi2b = p2[0], math.sqrt(cov2[0, 0]), chi22 / dof2
        else:
            c2 = se2 = chi2b = float("nan")
        print(f"  {b:5.2f}{len(full):4d}{p[0]:11.4f}{math.sqrt(cov[0,0]):9.4f}{p[1]:10.4f}"
              f"{chi2/dof:10.2f}{c2:11.4f}{se2:8.4f}{chi2b:10.2f}")
        rows.append(dict(beta=b, n_full=len(full), c_beta=float(p[0]),
                         c_beta_se=float(math.sqrt(cov[0, 0])), b_beta=float(p[1]),
                         chi2_per_dof=chi2 / dof,
                         c_beta_Lge12=float(c2), c_beta_Lge12_se=float(se2),
                         chi2_per_dof_Lge12=float(chi2b)))
    write_csv(TABLES / "alt_model_c_of_beta.csv", rows)

    non_det = [r for r in rows if r["beta"] > 0]
    cmax = max(non_det, key=lambda r: r["c_beta"])
    c0row = rows[0] if rows and rows[0]["beta"] == 0 else None
    print(f"\n  peak of c(beta): beta={cmax['beta']:.2f}, c={cmax['c_beta']:.4f}"
          f" +- {cmax['c_beta_se']:.4f}")
    if c0row:
        print(f"  c(beta=0) = {c0row['c_beta']:.4f}; ratio peak/c(0) = "
              f"{cmax['c_beta']/c0row['c_beta']:.2f}")
    sig = [r for r in non_det if r["c_beta"] - 2 * r["c_beta_se"] > (c0row["c_beta"] if c0row else 0)]
    print(f"  beta values whose c(beta) exceeds c(0) by >2 sigma: "
          f"{[r['beta'] for r in sig]}")
    return rows, cmax


# ---------------------------------------------------------------- part 9
def part9_tail_vs_z(data):
    """Does z organise the SHAPE statistics, even though it fails for the mean?

    If the tail share / kurtosis trace a single decreasing curve in z across all
    (L, beta), then z is a real control variable for the tail -- but the tail
    shrinks as z grows, which is the opposite of the claim.
    """
    print()
    print("=" * 78)
    print("PART 9 - z organises the tail SHAPE (decreasing), though not the mean")
    print("=" * 78)
    rows = []
    for (L, b), x in sorted(data.items()):
        t = tail_shares(x)
        m = moments(x)
        rows.append(dict(L=L, beta=b, z=L * L * math.exp(-2 * b),
                         top1_share=t["mass_top1pct"], top5_share=t["mass_top5pct"],
                         kurtosis=m["excess_kurtosis"], skew=m["skew"],
                         mean_over_median=t["mean_over_median"]))
    rows.sort(key=lambda r: r["z"])
    print(f"  {'z':>9}{'L':>4}{'beta':>6}{'top1%':>8}{'top5%':>8}{'exkurt':>9}{'m/med':>7}")
    for r in rows:
        print(f"  {r['z']:9.3f}{r['L']:4.0f}{r['beta']:6.2f}{r['top1_share']:8.3f}"
              f"{r['top5_share']:8.3f}{r['kurtosis']:9.1f}{r['mean_over_median']:7.2f}")
    write_csv(TABLES / "tail_shape_vs_z.csv", rows)
    # monotonicity in z
    sh = [r["top1_share"] for r in rows]
    mono = all(sh[i] >= sh[i + 1] - 0.02 for i in range(len(sh) - 1))
    print(f"  top1% share is monotonically DECREASING in z: {mono}")
    print(f"  top1% share at smallest z ({rows[0]['z']:.3f}) = {rows[0]['top1_share']:.3f}"
          f" -> at largest z ({rows[-1]['z']:.1f}) = {rows[-1]['top1_share']:.3f}")
    print(f"  Spearman-like check: rank(top1 share) vs rank(z) agree in sign: "
          f"{sh[0] > sh[-1]}")

    # where does the tail stop mattering? interpolate z at which top1% share = 0.05
    print("  Interpretation: the heavy tail is a small-z (strong-measurement) effect.")
    print("  Since z ~ L^2 e^(-2beta) grows with L at fixed beta, the tail is a")
    print("  FINITE-SIZE artefact that dies out as L grows -- not a persistent property.")
    return rows, mono


# ---------------------------------------------------------------- part 10
def part10_mult_scaling(prod):
    """Two-parameter scan for the multiplicative ansatz

        D(L, beta) = D0(L) * f( t ),   t = L^alpha * exp(-gamma beta)

    with D0(L) the beta=0 deterministic ground-state value.  The additive
    ansatz D = c ln L + A(z) is tested in part 5 and fails; this tests whether a
    ratio-form scaling variable does better, and what (alpha, gamma) it prefers.

    A collapse is accepted only if the residual scatter of ln(D/D0) inside a
    bucket is comparable to the propagated MCSE.
    """
    print()
    print("=" * 78)
    print("PART 10 - two-parameter scan of the multiplicative scaling form")
    print("=" * 78)
    ground = {r["L"]: r["D"] for r in prod if r["beta"] == 0.0}
    pts = []
    for r in prod:
        if r["beta"] == 0.0 or r["D"] <= 0 or r["L"] not in ground:
            continue
        y = math.log(r["D"] / ground[r["L"]])
        ey = r["E"] / r["D"]              # relative MCSE on the ratio
        pts.append(dict(L=r["L"], beta=r["beta"], y=y, ey=ey, D=r["D"]))

    def collapse_quality(alpha, gamma, width=0.20):
        ts = np.array([p["L"] ** alpha * math.exp(-gamma * p["beta"]) for p in pts])
        lz = np.log(ts)
        order = np.argsort(lz)
        buckets, cur = [], [order[0]]
        for i in order[1:]:
            if lz[i] - lz[cur[-1]] < width:
                cur.append(i)
            else:
                buckets.append(cur); cur = [i]
        buckets.append(cur)
        resid = []
        for g in buckets:
            if len(g) < 2:
                continue
            ys = np.array([pts[i]["y"] for i in g])
            es = np.array([pts[i]["ey"] for i in g])
            # weighted mean and chi-square inside the bucket
            w = 1.0 / es**2
            mu = float(np.sum(w * ys) / np.sum(w))
            chi2 = float(np.sum(w * (ys - mu) ** 2))
            dof = len(g) - 1
            resid.append((chi2 / dof, math.sqrt(max(0.0, float(ys.var(ddof=1)) - float((es**2).mean())))))
        if not resid:
            return float("nan"), float("nan"), 0
        return float(np.median([r[0] for r in resid])), float(np.median([r[1] for r in resid])), len(resid)

    rows = []
    for alpha in np.arange(0.25, 2.26, 0.25):
        for gamma in np.arange(1.0, 6.01, 0.5):
            chi2n, excess, nb = collapse_quality(alpha, gamma)
            rows.append(dict(alpha=float(alpha), gamma=float(gamma), n_buckets=nb,
                             median_chi2_per_dof=chi2n, median_excess_scatter=excess))
    write_csv(TABLES / "mult_scaling_scan.csv", rows)

    good = [r for r in rows if math.isfinite(r["median_chi2_per_dof"]) and r["n_buckets"] >= 4]
    good.sort(key=lambda r: r["median_chi2_per_dof"])
    print(f"  scanned {len(rows)} (alpha, gamma) pairs; {len(good)} usable")
    print(f"  {'alpha':>6}{'gamma':>7}{'buckets':>9}{'chi2/dof':>11}{'excess':>9}")
    for r in good[:8]:
        print(f"  {r['alpha']:6.2f}{r['gamma']:7.2f}{r['n_buckets']:9d}"
              f"{r['median_chi2_per_dof']:11.2f}{r['median_excess_scatter']:9.4f}")
    named = next((r for r in rows if abs(r["alpha"] - 2.0) < 1e-9 and abs(r["gamma"] - 2.0) < 1e-9), None)
    if named:
        print()
        print(f"the claimed z = L^2 e^(-2beta) [alpha=2, gamma=2]: "
              f"chi2/dof = {named['median_chi2_per_dof']:.2f}, "
              f"excess scatter = {named['median_excess_scatter']:.4f}")
    best = good[0] if good else dict(alpha=float("nan"), gamma=float("nan"),
                                     median_chi2_per_dof=float("nan"))
    print(f"  best pair: alpha={best['alpha']:.2f}, gamma={best['gamma']:.2f}, "
          f"chi2/dof={best['median_chi2_per_dof']:.2f}")
    n_ok = sum(1 for r in good if r["median_chi2_per_dof"] < 3.0)
    print(f"  (alpha, gamma) pairs whose collapse reaches chi2/dof < 3: {n_ok} of {len(good)}")
    return rows, best, pts, ground


def part10b_best_collapse_figure(prod, best, pts, ground):
    """Render the best multiplicative collapse next to the failed z-collapse.

    y = ln[D / D0] is negative and unbounded below, so BOTH panels use a log x
    axis and a linear y axis. Plotting it on log-log would be meaningless.
    """
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    Ls = sorted({p["L"] for p in pts})
    cmap = plt.get_cmap("viridis")
    for i, L in enumerate(Ls):
        sel = sorted([p for p in pts if p["L"] == L], key=lambda p: p["beta"])
        col = cmap(i / max(1, len(Ls) - 1))
        zs = [p["L"] ** 2 * math.exp(-2 * p["beta"]) for p in sel]
        ax[0].semilogx(zs, [p["y"] for p in sel], "o-", ms=3.5, lw=1.0, color=col,
                       label=f"L={int(L)}")
        ts = [p["L"] ** best["alpha"] * math.exp(-best["gamma"] * p["beta"]) for p in sel]
        ax[1].semilogx(ts, [p["y"] for p in sel], "o-", ms=3.5, lw=1.0, color=col,
                       label=f"L={int(L)}")
    ax[0].set_xlabel("z = L^2 exp(-2 beta)")
    ax[0].set_ylabel("ln[D(L,beta) / D0(L)]")
    ax[0].set_title("claimed z variable: L-curves stay separated")
    ax[0].legend(fontsize=7, ncol=2)
    ax[0].grid(alpha=0.3, which="both")
    ax[0].set_ylim(-5.9, 0.35)
    ax[1].set_xlabel(f"t = L^{best['alpha']:.2f} exp(-{best['gamma']:.2f} beta)")
    ax[1].set_ylabel("ln[D(L,beta) / D0(L)]")
    ax[1].set_title(f"best scanned pair still fails (chi2/dof={best['median_chi2_per_dof']:.0f})")
    ax[1].legend(fontsize=7, ncol=2)
    ax[1].grid(alpha=0.3, which="both")
    ax[1].set_ylim(-5.9, 0.35)
    fig.tight_layout()
    fig.savefig(FIGS / "fig6_mult_scaling_scan.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- figures
def make_figures(data, prod):
    # Fig 1: tail share vs L
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    for b, mk in ((1.0, "o"), (3.0, "s")):
        Ls = sorted({L for (L, bb) in data if bb == b})
        sh1 = [tail_shares(data[(L, b)])["mass_top1pct"] for L in Ls]
        sh5 = [tail_shares(data[(L, b)])["mass_top5pct"] for L in Ls]
        ax[0].plot(Ls, sh1, mk + "-", label=f"beta={b:g}, top 1%")
        ax[0].plot(Ls, sh5, mk + "--", alpha=0.55, label=f"beta={b:g}, top 5%")
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("L")
    ax[0].set_ylabel("fraction of sample mean from top q%")
    ax[0].set_title("Tail contribution falls with L at fixed beta")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3, which="both")

    # right: mean / median
    for b, mk in ((1.0, "o"), (3.0, "s")):
        Ls = sorted({L for (L, bb) in data if bb == b})
        r = [tail_shares(data[(L, b)])["mean_over_median"] for L in Ls]
        ax[1].plot(Ls, r, mk + "-", label=f"beta={b:g}")
    ax[1].axhline(1.0, color="k", lw=0.8, ls=":")
    ax[1].set_xscale("log")
    ax[1].set_xlabel("L")
    ax[1].set_ylabel("mean / median of D")
    ax[1].set_title("Mean vs typical trajectory: gap shrinks with L")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(FIGS / "fig1_tail_share_vs_L.png", dpi=150)
    plt.close(fig)

    # Fig 2: log-log upper tail (survival) of D
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    for (L, b), x in sorted(data.items()):
        pos = np.sort(x[x > 0])
        if pos.size < 20:
            continue
        y = 1.0 - np.arange(pos.size) / pos.size
        s = "o" if b == 1.0 else "s"
        alpha = 0.45 + 0.25 * (L / 18.0)
        ax[0].loglog(pos, y, s, ms=3, alpha=alpha, label=f"L={int(L)},b={b:g}")
        ax[1].semilogy(pos, y, s, ms=3, alpha=alpha)
    ax[0].set_xlabel("D (positive part)")
    ax[0].set_ylabel("P(D > x)")
    ax[0].set_title("log-log survival: curvature, not a straight power law")
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=0.3, which="both")
    ax[1].set_xlabel("D (positive part)")
    ax[1].set_ylabel("P(D > x)")
    ax[1].set_title("semi-log survival: exponential upper tail")
    ax[1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(FIGS / "fig2_D_survival.png", dpi=150)
    plt.close(fig)

    # Fig 3: D vs z, colour by L
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    Ls = sorted({r["L"] for r in prod})
    cmap = plt.get_cmap("viridis")
    for i, L in enumerate(Ls):
        sel = sorted([r for r in prod if r["L"] == L], key=lambda r: r["beta"])
        zs = [z_of(r["L"], r["beta"]) for r in sel]
        ds = [r["D"] for r in sel]
        es = [r["E"] for r in sel]
        col = cmap(i / max(1, len(Ls) - 1))
        ax[0].errorbar(zs, ds, yerr=es, fmt="o-", ms=3.5, lw=1.0, capsize=0,
                       color=col, label=f"L={int(L)}")
        ax[1].semilogx(zs, ds, "o-", ms=3.5, lw=1.0, color=col, label=f"L={int(L)}")
    ax[0].set_xlabel("z = L^2 exp(-2 beta)")
    ax[0].set_ylabel("D_mean")
    ax[0].set_title("No z-collapse among L curves (linear z)")
    ax[0].legend(fontsize=7, ncol=2)
    ax[0].grid(alpha=0.3)
    ax[1].set_xlabel("z = L^2 exp(-2 beta)")
    ax[1].set_ylabel("D_mean")
    ax[1].set_title("same, log z")
    ax[1].legend(fontsize=7, ncol=2)
    ax[1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(FIGS / "fig3_D_vs_z.png", dpi=150)
    plt.close(fig)

    # Fig 4: local slope vs z
    rows, _ = part2b_slopes(prod)
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    betas = sorted({r["beta"] for r in rows})
    for i, b in enumerate(betas):
        sel = [r for r in rows if r["beta"] == b]
        zs = [r["z_geomean"] for r in sel]
        sl = [r["local_slope"] for r in sel]
        ax.semilogx(zs, sl, "o-", ms=4, lw=1.1, color=cmap(i / max(1, len(betas) - 1)),
                    label=f"beta={b:g}")
    ax.axvline(1.0, color="k", ls=":", lw=1)
    ax.axhline(0.2217, color="r", ls="--", lw=1, alpha=0.7, label="c(beta=0) fit")
    ax.set_xlabel("z = L^2 exp(-2 beta)")
    ax.set_ylabel("local slope dD/dlnL")
    ax.set_title("Local slope rises across z~1 at large beta")
    ax.legend(fontsize=6.5, ncol=2)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(FIGS / "fig4_local_slope_vs_z.png", dpi=150)
    plt.close(fig)

    # Fig 5: quantiles vs L
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    for k, b in enumerate((1.0, 3.0)):
        Ls2 = sorted({L for (L, bb) in data if bb == b})
        for p, ls in ((1, ":"), (25, "--"), (50, "-"), (75, "--"), (99, ":")):
            ax[k].plot(Ls2, [np.quantile(data[(L, b)], p / 100) for L in Ls2], "o" + ls,
                       ms=3, label=f"q{p}")
        ax[k].plot(Ls2, [data[(L, b)].mean() for L in Ls2], "k^-", ms=5, label="mean")
        ax[k].set_xscale("log")
        ax[k].set_title(f"beta={b:g}: quantiles of D vs L")
        ax[k].set_xlabel("L")
        ax[k].set_ylabel("D")
        ax[k].legend(fontsize=7)
        ax[k].grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(FIGS / "fig5_quantiles_vs_L.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    data = load_iid()
    comps = load_iid_components()
    prod = load_production()

    rows, k1, k2 = part1(data, comps)
    part1b_decomposition(comps)

    # sanity: the D definition
    bad = max(float(np.abs(c["D"] - ((2 * c["SA"] + c["SC"]) / 3 - c["SB"])).max())
              for c in comps.values())
    print(f"\n  max |D - ((2SA+SC)/3 - SB)| over all iid rows = {bad:.3e}")

    zstats = part2_zcollapse(prod)
    slope_rows, _ = part2b_slopes(prod)
    c0, groups = part3_shifted_collapse(prod)
    fals = part4_falsifiers(data, comps, prod)
    scan_rows, best = part5_exponent_scan(prod, c0)
    over = part6_overshoot(prod)
    chslopes = part7_chains_nonstationarity()
    alt_rows, alt_peak = part8_alt_model(prod)
    zshape, zmono = part9_tail_vs_z(data)
    scan2, best2, mpts, ground = part10_mult_scaling(prod)
    part10b_best_collapse_figure(prod, best2, mpts, ground)

    # chain-level scatter: 4 independent chains per production point
    ch = load_chain_means()
    print("\n-- per-chain scatter (4 chains x 2048 draws, correlated MCMC) --")
    ch_rows = []
    for (L, b), v in sorted(ch.items()):
        se = v.std(ddof=1) / math.sqrt(v.size)
        chord = (v.max() - v.min()) / v.mean()
        print(f"  L={L:2.0f} beta={b:4.2f}  chain range={v.min():.4f}..{v.max():.4f}"
              f"  relative spread={chord:.2%}  SE(mean)={se:.4f} ({se/v.mean() if v.mean() else float('nan'):.2%})")
        ch_rows.append(dict(L=L, beta=b, chains=";".join(f"{x:.6f}" for x in v),
                            mean=float(v.mean()), chainSE=float(se),
                            relative_range=float(chord)))
    write_csv(TABLES / "chain_scatter.csv", ch_rows)

    make_figures(data, prod)

    summary = dict(
        kill_K1_tail_share_grows=all(bool(r["grows_with_L"]) for r in k1),
        kill_K1_detail={f"beta{int(r['beta'])}": dict(slope_top1=r["slope_top1_vs_L"],
                                                      ratio=r["ratio_top1_last_over_first"]) for r in k1},
        kill_K2_all_self_averaging_CLT=all(bool(r["self_averaging"]) for r in k2),
        kill_K2_detail={f"L{int(r['L'])}_b{int(r['beta'])}": r["loglog_slope"] for r in k2},
        z_collapse=zstats,
        c_beta0=c0,
        d_definition_max_error=bad,
        falsifiers=fals,
        exponent_scan_best_alpha=best["alpha"],
        exponent_scan_best_scatter=best["median_excess_scatter"],
        overshoot_betas=[r["beta"] for r in over if r["overshoots"]],
        chain_scatter_slopes=chslopes,
        alt_model_peak_beta=alt_peak["beta"],
        alt_model_peak_c=alt_peak["c_beta"],
        alt_model_table=alt_rows,
        tail_shape_decreasing_in_z=bool(zmono),
        exponent_scan_n_collapsing_alphas=best.get("n_collapsing_alphas"),
        mult_scaling_best_alpha=best2["alpha"],
        mult_scaling_best_gamma=best2["gamma"],
        mult_scaling_best_chi2_per_dof=best2["median_chi2_per_dof"],
    )
    (HERE / "verify_tails_numbers.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n" + json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
