"""Adversarial verification of the claimed decay exponent of mean(|G_E,ij|^2).

CLAIM UNDER TEST
  (i)   mean|G_E,ij|^2 ~ r^-alpha with alpha ~ 3.4-3.6 at beta=1 and ~ 1.0-1.5 at
        beta=3 (same L), i.e. strong measurement makes the correlation-square longer
        ranged;
  (ii)  the coefficient of ln L in D_V is fixed by alpha = 4 (the free-Dirac
        baseline) and "saturates to a constant for alpha > 2";
  (iii) hence alpha < 2 at beta=3 means D_V grows faster than ln L, invalidating the
        log-corner picture for strong measurement.

TESTS
  A. Recompute alpha from the raw binary matrices with six binnings and a GLS fit
     weighted by the 4-batch covariance; map the fit-window systematic.
     CALIBRATE the estimator on the ground state, whose continuum decay is exactly
     |G| ~ r^-2 hence |G|^2 ~ r^-4: whatever alpha the estimator returns there is
     the finite-size + geometry bias, and any claimed alpha must be read against it.
  B. Build a synthetic correlator |G_ij|^2 = c r^-alpha on the ACTUAL honeycomb
     torus with the ACTUAL region geometry and the sum rule sum|G|^2 = L^2, and
     measure D_V(L) for L = 6..192. Establish the true D_V(alpha) law and test
     statement (ii) directly.
  C. Recompute alpha using only pairs with r < L/3, and after removing the
     sum-rule-forced uniform floor, to test for a finite-size crossover.
  D. Decompose sum|G_ij|^2 into mean^2 and trajectory variance, and compare the
     measured D_V(L) against the ground-state ln L baseline.

All geometry / kernel conventions are validated against the receipts' DV0_full
before they are used.  Run:  python -I verify_decay.py
"""
import csv
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent        # research_20261030/docs
BORN = HERE.parent.parent                    # born_ai/
DATA = BORN / "research_20261029" / "iid_targeted"
BASELINE = BORN / "research_20261029" / "baseline"
FIG = HERE.parent / "figures"
OUT = HERE / "tables"

# --- palette: dataviz reference instance, validated all-pairs PASS (light) ----
BLUE = "#2a78d6"     # beta = 1
ORANGE = "#eb6834"   # beta = 3
AQUA = "#1baf7a"     # ground control
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8880"
GRID = "#d9d7d0"

# ----------------------------------------------------------------------------
# Lattice: matches run_iid_targeted.jl site labelling.
#   site i (0-based): cell = i//2, sub = i%2, x = cell % L, y = cell // L
#   r = x*a1 + y*a2 + d[sub], a1=(1,0), a2=(1/2,sqrt3/2), d=(0,0),(0,1/sqrt3)
# ----------------------------------------------------------------------------
A1 = np.array([1.0, 0.0])
A2 = np.array([0.5, np.sqrt(3.0) / 2.0])
DSUB = np.array([[0.0, 0.0], [0.0, 1.0 / np.sqrt(3.0)]])


def cell_coords(L):
    N = 2 * L * L
    i = np.arange(N)
    cell = i // 2
    return cell % L, cell // L, i % 2


def positions(L):
    x, y, sub = cell_coords(L)
    return x[:, None] * A1 + y[:, None] * A2 + DSUB[sub]


def min_image_dist(L, pos=None):
    """Minimum-image Euclidean distance on the L x L honeycomb torus."""
    if pos is None:
        pos = positions(L)
    best = np.full((len(pos), len(pos)), np.inf)
    for n1 in (-1, 0, 1):
        for n2 in (-1, 0, 1):
            shift = (n1 * A1 + n2 * A2) * L
            diff = pos[:, None, :] - pos[None, :, :] + shift
            best = np.minimum(best, np.sqrt((diff ** 2).sum(-1)))
    return best


def dv_kernel(L, p=1.0 / 3.0):
    """h_x(dx) h_y(dy) with h(d)=min(d,L-d,w)/L, w=pL.

    Translation-averaged D_V identity:  D_V^full = sum_ij |G_ij|^2 h_x h_y.
    """
    w = max(1, int(round(p * L)))
    x, y, _ = cell_coords(L)
    dx = np.abs(x[:, None] - x[None, :])
    dx = np.minimum(dx, L - dx)
    dy = np.abs(y[:, None] - y[None, :])
    dy = np.minimum(dy, L - dy)
    return (np.minimum(dx, w) / L) * (np.minimum(dy, w) / L)


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def load_matrix(L, beta, name="meanGsquare", batch=1):
    N = 2 * L * L
    return np.fromfile(DATA / f"L{L}_beta{beta}_batch{batch}_{name}.bin",
                       dtype=np.float64).reshape((N, N), order="F")


def load_batches(L, beta, name="meanGsquare"):
    return [load_matrix(L, beta, name, b) for b in (1, 2, 3, 4)]


def load_mean(L, beta, name="meanGsquare"):
    b = load_batches(L, beta, name)
    return sum(b) / len(b)


def ground_matrix(L):
    N = 2 * L * L
    return np.fromfile(DATA / "models" / f"iid_targeted_L{L}_staggered_mu1.0e-8_G.bin",
                       dtype=np.float64).reshape((N, N))


# ----------------------------------------------------------------------------
# A. alpha estimation
# ----------------------------------------------------------------------------
def shell_table(M, R, rmax=None):
    N = M.shape[0]
    off = ~np.eye(N, dtype=bool)
    if rmax is not None:
        off &= (R <= rmax)
    vals = np.unique(np.round(R[off], 10))
    rows = []
    for v in vals:
        m = (np.abs(R - v) < 1e-9) & off
        n = int(m.sum())
        if n:
            rows.append((float(v), n, float(M[m].mean())))
    return rows


def bin_table(M, R, edges):
    N = M.shape[0]
    off = ~np.eye(N, dtype=bool)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (R >= lo) & (R < hi) & off
        n = int(m.sum())
        if n >= 20:
            rows.append((float(R[m].mean()), n, float(M[m].mean())))
    return rows


def gls_alpha(batch_tables):
    """GLS fit of ln<|G|^2> = ln c - alpha ln r using the 4-batch covariance.

    For a single realisation (ground-state control) there is no batch scatter, so
    this degenerates to OLS and the reported error is the OLS slope error.
    """
    k = len(batch_tables[0])
    R = np.array([t[0] for t in batch_tables[0]], float)
    Mv = np.array([[t[i][2] for i in range(k)] for t in batch_tables], float)
    good = np.all(np.isfinite(Mv), axis=0) & np.all(Mv > 0, axis=0)
    R, Mv = R[good], Mv[:, good]
    if len(R) < 4:
        return float("nan"), float("nan"), len(R)
    Y = np.log(Mv)
    X = np.column_stack([np.log(R), np.ones_like(R)])
    if Y.shape[0] > 1:
        S = np.cov(Y, rowvar=False, ddof=1)
        Sinv = np.diag(1.0 / np.diag(S)) if S.ndim == 2 else np.array([[1.0 / float(S)]])
        A = X.T @ Sinv @ X
        coef = np.linalg.solve(A, X.T @ Sinv @ Y.mean(axis=0))
        return -coef[0], float(np.sqrt(np.linalg.inv(A)[0, 0])), len(R)
    coef, *_ = np.linalg.lstsq(X, Y[0], rcond=None)
    resid = Y[0] - X @ coef
    s2 = float(resid @ resid) / max(1, len(R) - 2)
    cov = s2 * np.linalg.inv(X.T @ X)
    return -coef[0], float(np.sqrt(cov[0, 0])), len(R)


SCHEMES = {
    "shell_exact": None,
    "geom_8": np.geomspace(0.9, 9.0, 9),
    "geom_12": np.geomspace(0.9, 9.0, 13),
    "linear_0.5": np.arange(0.9, 9.0, 0.5),
    "linear_1.0": np.arange(0.9, 9.0, 1.0),
    "wide_5": np.geomspace(0.5, 12.0, 6),
}


# ----------------------------------------------------------------------------
# B. synthetic D_V on the real torus / region geometry
# ----------------------------------------------------------------------------
def synthetic_dv(L, alpha, p=1.0 / 3.0):
    """D_V for |G_ij|^2 = c r^-alpha normalised to sum|G|^2 = L^2.

    Enumerated by cell displacement (L^2 cells x 4 sublattice pairs), so it runs to
    large L without dense N x N matrices.
    """
    w = max(1, int(round(p * L)))
    DX, DY = np.meshgrid(np.arange(L), np.arange(L), indexing="ij")
    DX = DX.ravel()
    DY = DY.ravel()
    hx = np.minimum(np.minimum(DX, L - DX), w) / L
    hy = np.minimum(np.minimum(DY, L - DY), w) / L
    hh = hx * hy
    base = DX[:, None] * A1 + DY[:, None] * A2
    tot_w = 0.0
    tot_hw = 0.0
    for sa in (0, 1):
        for sb in (0, 1):
            delta = DSUB[sb] - DSUB[sa]
            rbest = np.full(len(base), np.inf)
            for n1 in (-1, 0, 1):
                for n2 in (-1, 0, 1):
                    shift = (n1 * A1 + n2 * A2) * L
                    d = base + delta + shift
                    rbest = np.minimum(rbest, np.sqrt((d ** 2).sum(1)))
            diag = rbest < 1e-12
            safe = np.where(diag, 1.0, rbest)
            wgt = np.where(diag, 0.0, safe ** (-alpha))
            tot_w += float(wgt.sum())
            tot_hw += float((wgt * hh).sum())
    return float(L * L * tot_hw / tot_w)


def local_exponent(Ls, vals):
    Ls = np.asarray(Ls, float)
    vals = np.asarray(vals, float)
    return np.diff(np.log(vals)) / np.diff(np.log(Ls))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


# ----------------------------------------------------------------------------
def main():
    FIG.mkdir(exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    rep = {}

    # ---- 0. convention check against the receipts --------------------------
    rec = {6: 0.12940559252380907, 12: 0.183034645377678, 18: 0.21154446337187238}
    rep["kernel_vs_receipt_rel_error"] = {
        str(L): float((np.sum(ground_matrix(L) ** 2 * dv_kernel(L)) - rec[L]) / rec[L])
        for L in (6, 12, 18)}
    assert max(abs(v) for v in rep["kernel_vs_receipt_rel_error"].values()) < 1e-12

    Rs = {L: min_image_dist(L) for L in (6, 12, 18)}

    # ---- A. alpha, calibrated against the ground state ---------------------
    alpha = {}
    for L in (12, 18):
        R = Rs[L]
        for tag, bt in (("ground(true_alpha=4)", [ground_matrix(L) ** 2]),
                        ("beta1", load_batches(L, 1)),
                        ("beta3", load_batches(L, 3))):
            per = {}
            for nm, edges in SCHEMES.items():
                tables = [shell_table(b, R) if edges is None else bin_table(b, R, edges)
                          for b in bt]
                a, e, n = gls_alpha(tables)
                per[nm] = dict(alpha=a, err=e, nbin=n)
            alpha[f"L{L}_{tag}"] = per

    bias = {f"L{L}": {nm: 4.0 - alpha[f"L{L}_ground(true_alpha=4)"][nm]["alpha"]
                      for nm in SCHEMES} for L in (12, 18)}
    corrected = {f"L{L}_beta{b}": {nm: alpha[f"L{L}_beta{b}"][nm]["alpha"] + bias[f"L{L}"][nm]
                                   for nm in SCHEMES}
                 for L in (12, 18) for b in (1, 3)}

    # per-scheme spread = the systematic that the claim does not quote
    spread = {}
    for L in (12, 18):
        for b in (1, 3):
            v = [alpha[f"L{L}_beta{b}"][nm]["alpha"] for nm in SCHEMES]
            spread[f"L{L}_beta{b}"] = dict(min=min(v), max=max(v), range=max(v) - min(v),
                                           mean=float(np.mean(v)),
                                           median=float(np.median(v)))

    # window scan
    window = {}
    for L in (12, 18):
        R = Rs[L]
        for b in (1, 3):
            rows = []
            for lo in (0.9, 1.3, 1.8):
                for hi in (3.0, 4.5, 6.0, 8.0, 10.0, 14.0):
                    if hi > 0.62 * L:
                        continue
                    edges = np.geomspace(lo, hi, 9)
                    tables = [bin_table(m, R, edges) for m in load_batches(L, b)]
                    if len(tables[0]) < 3:
                        continue
                    a, e, n = gls_alpha(tables)
                    rows.append(dict(rmin=lo, rmax=hi, alpha=a, nbin=n))
            window[f"L{L}_beta{b}"] = rows

    # synthetic r^-4 through the same estimator: isolates geometry/finite-size bias
    synth_bias = {}
    for L in (12, 18):
        R = Rs[L]
        with np.errstate(divide="ignore"):
            A = np.where(np.isfinite(R), R, 1.0) ** (-4.0)
        np.fill_diagonal(A, 0.0)
        per = {}
        for nm, edges in SCHEMES.items():
            t = shell_table(A, R) if edges is None else bin_table(A, R, edges)
            rr = np.array([x[0] for x in t if x[2] > 0])
            yy = np.array([x[2] for x in t if x[2] > 0])
            per[nm] = float(-np.polyfit(np.log(rr), np.log(yy), 1)[0])
        synth_bias[f"L{L}"] = per

    # ---- C. crossover / sum-rule floor subtraction -------------------------
    crossover = {}
    for L in (6, 12, 18):
        R = Rs[L]
        N = 2 * L * L
        for b in (1, 3):
            M = load_mean(L, b)
            off = ~np.eye(N, dtype=bool)
            c = float(M[off].sum()) / (N * (N - 1))
            row = dict(floor=c, offsum=float(M[off].sum()))
            for frac, key in ((1 / 3, "r<L/3"), (0.5, "r<L/2"), (None, "all")):
                rm = None if frac is None else frac * L
                t_raw = shell_table(M, R, rmax=rm)
                t_sub = shell_table(M - c, R, rmax=rm)

                def fit(t):
                    rr = np.array([x[0] for x in t if x[2] > 0], float)
                    yy = np.array([x[2] for x in t if x[2] > 0], float)
                    return float(-np.polyfit(np.log(rr), np.log(yy), 1)[0])
                if len(t_raw) >= 4:
                    row[key] = dict(nshell=len(t_raw), raw=fit(t_raw), floor_sub=fit(t_sub))
            # far-field flatness test: mean at r>L/3 relative to the forced floor
            far = (R > L / 3) & off
            if far.any():
                row["far_over_floor"] = float(M[far].mean() / c)
                row["near_over_floor"] = float(
                    M[(R > 0.4) & (R < 1.3) & off].mean() / c)
            crossover[f"L{L}_beta{b}"] = row

    # ---- B. synthetic D_V(L) ----------------------------------------------
    Ls = [6, 8, 10, 12, 16, 18, 24, 32, 36, 48, 64, 96, 128, 192]
    alphas = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    synth = {al: [synthetic_dv(L, al) for L in Ls] for al in alphas}
    synth_exp = {al: float(local_exponent(Ls, synth[al])[-1]) for al in alphas}
    synth_exp3 = {al: float(local_exponent(Ls[-4:], synth[al][-4:])[-1]) for al in alphas}

    chk = []
    for L in (6, 12):
        R = Rs[L]
        with np.errstate(divide="ignore"):
            A = np.where(np.isfinite(R), R, 1.0) ** (-2.0)
        np.fill_diagonal(A, 0.0)
        A = A / A.sum() * (L * L)
        chk.append(abs(synthetic_dv(L, 2.0) - float(np.sum(A * dv_kernel(L))))
                   / synthetic_dv(L, 2.0))
    rep["synthetic_vs_dense_rel_error"] = chk
    assert max(chk) < 1e-10

    # ---- D. variance decomposition + data D_V -----------------------------
    kappa = {1: {6: 0.41162308951066984, 12: 0.4179433201601924, 18: 0.41907591751450923},
             3: {6: 0.005383603305987195, 12: 0.006601246939284988, 18: 0.008005492050918495}}
    data_dv = {}
    vardec = {}
    for L in (6, 12, 18):
        N = 2 * L * L
        off = ~np.eye(N, dtype=bool)
        K = dv_kernel(L)
        for b in (1, 3):
            mats = load_batches(L, b)
            d = np.array([float(np.sum(m * K)) for m in mats])
            data_dv[f"L{L}_beta{b}"] = dict(
                DV=float(d.mean()), DV_batch_sd=float(d.std(ddof=1)),
                lower_bound=float(kappa[b][L] ** 2 * rec[L]))
            if L >= 12:
                M2 = load_mean(L, b, "meanGsquare")
                M = load_mean(L, b, "meanG")
                V = M2 - M * M
                vardec[f"L{L}_beta{b}"] = dict(
                    sum_G2=float(M2[off].sum()),
                    sum_mean_sq=float((M * M)[off].sum()),
                    sum_variance=float(V[off].sum()),
                    variance_fraction=float(V[off].sum() / M2[off].sum()))

    base_dv = {}
    gc = BASELINE / "ground_corners.csv"
    if gc.exists():
        for r in csv.DictReader(gc.open()):
            if abs(float(r["p"]) - 1 / 3) < 1e-12 and float(r["mu"]) == 0.0:
                base_dv[int(r["L"])] = float(r["DV"])

    # ---- E. D_V split into a coherent (mean^2) and a trajectory-variance part --
    dv_split = {}
    for L in (6, 12, 18):
        N = 2 * L * L
        K = dv_kernel(L) * (~np.eye(N, dtype=bool))
        for b in (1, 3):
            M2 = load_mean(L, b, "meanGsquare")
            M = load_mean(L, b, "meanG")
            V = M2 - M * M
            tot = float(np.sum(M2 * K))
            coh = float(np.sum((M * M) * K))
            var = float(np.sum(V * K))
            offsum = float(M2[~np.eye(N, dtype=bool)].sum())
            dv_split[f"L{L}_beta{b}"] = dict(
                DV=tot, DV_coherent=coh, DV_variance=var,
                coherent_fraction=coh / tot, variance_fraction=var / tot,
                offsum=offsum, DV_over_offsum=tot / offsum)
    # the uniform (structureless) reference: kernel mass / pair count
    unif = {L: float(dv_kernel(L).sum() / (2 * L * L * (2 * L * L - 1)))
            for L in (6, 12, 18)}

    # ground state's own binned exponent and the resulting estimator bias
    ground_fit = {}
    for L in (12, 18):
        R = Rs[L]
        A0 = ground_matrix(L) ** 2
        for nm, edges in SCHEMES.items():
            t = shell_table(A0, R) if edges is None else bin_table(A0, R, edges)
            rr = np.array([x[0] for x in t if x[2] > 0])
            yy = np.array([x[2] for x in t if x[2] > 0])
            ground_fit.setdefault(f"L{L}", {})[nm] = float(-np.polyfit(np.log(rr), np.log(yy), 1)[0])
    lin_bias = {f"L{L}": 4.0 - ground_fit[f"L{L}"]["linear_1.0"] for L in (12, 18)}


    Ls_data = [6, 12, 18]
    data_exp = {}
    for b in (1, 3):
        vals = [data_dv[f"L{L}_beta{b}"]["DV"] for L in Ls_data]
        data_exp[f"beta{b}"] = [float(x) for x in local_exponent(Ls_data, vals)]
    base_exp = [float(x) for x in local_exponent([6, 12, 18],
                                                 [base_dv[L] for L in (6, 12, 18)])] if base_dv else []

    # D_V / ln L track record against the ground-state baseline
    dv_lnL = {b: [data_dv[f"L{L}_beta{b}"]["DV"] / np.log(L) for L in Ls_data]
              for b in (1, 3)}
    gs_lnL = [base_dv[L] / np.log(L) for L in (6, 12, 18)] if base_dv else []

    rep.update(
        alpha=alpha, bias_measured_on_ground=bias, alpha_bias_corrected=corrected,
        scheme_spread=spread, synthetic_r4_estimator_bias=synth_bias,
        ground_alpha_by_scheme=ground_fit,
        ground_linear_fit_bias=lin_bias,
        window_scan=window, crossover=crossover,
        synthetic={"Ls": Ls, "alpha": alphas,
                   "DV": {str(a): [float(x) for x in synth[a]] for a in alphas},
                   "exponent_large": {str(a): synth_exp[a] for a in alphas},
                   "exponent_last3": {str(a): synth_exp3[a] for a in alphas}},
        data_dv=data_dv, data_DV_exponent=data_exp,
        ground_DV_baseline=base_dv, ground_DV_exponent=base_exp,
        DV_over_lnL=dict(measured=dv_lnL, ground=gs_lnL),
        variance_decomposition=vardec, DV_split=dv_split,
        uniform_kernel_reference={str(k): v for k, v in unif.items()},
        figure=str(FIG / "verify_decay.png"),
    )

    # ---- tables -----------------------------------------------------------
    rows = []
    for L in (12, 18):
        for tag in ("ground(true_alpha=4)", "beta1", "beta3"):
            for nm in SCHEMES:
                r = alpha[f"L{L}_{tag}"][nm]
                rows.append(dict(L=L, series=tag, scheme=nm, nbin=r["nbin"],
                                 alpha=r["alpha"], err_gls=r["err"]))
    write_csv(OUT / "alpha_by_binning.csv", rows)

    write_csv(OUT / "alpha_window_scan.csv",
              [dict(L=k.split("_")[0][1:], beta=k.split("_beta")[1], **r)
               for k, v in window.items() for r in v])

    write_csv(OUT / "alpha_crossover.csv",
              [dict(L=k.split("_")[0][1:], beta=k.split("_beta")[1], window=w,
                    nshell=d["nshell"], alpha_raw=d["raw"], alpha_floor_sub=d["floor_sub"])
               for k, v in crossover.items() for w, d in v.items()
               if isinstance(d, dict)])

    write_csv(OUT / "synthetic_DV_vs_alpha.csv",
              [dict(L=L, alpha=al, DV=synth[al][i])
               for al in alphas for i, L in enumerate(Ls)])

    write_csv(OUT / "measured_DV.csv",
              [dict(L=int(k.split("_")[0][1:]), beta=k.split("_beta")[1],
                    DV=v["DV"], DV_batch_sd=v["DV_batch_sd"], lower_bound=v["lower_bound"])
               for k, v in data_dv.items()])

    write_csv(OUT / "DV_split_coherent_variance.csv",
              [dict(L=int(k.split("_")[0][1:]), beta=k.split("_beta")[1],
                    DV=v["DV"], DV_coherent=v["DV_coherent"], DV_variance=v["DV_variance"],
                    coherent_fraction=v["coherent_fraction"],
                    variance_fraction=v["variance_fraction"],
                    offsum=v["offsum"], DV_over_offsum=v["DV_over_offsum"],
                    uniform_reference=unif[int(k.split("_")[0][1:])])
               for k, v in dv_split.items()])

    # ---------------------- figure -----------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
        "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.8,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })

    def style(ax, title):
        ax.set_title(title, fontsize=9.5, color=INK, loc="left")
        ax.grid(True, color=GRID, lw=0.6, alpha=0.9)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    fig, axes = plt.subplots(2, 2, figsize=(12.6, 9.4))

    # P1: radial profiles --------------------------------------------------
    ax = axes[0, 0]
    for L, ls, mk in ((12, "--", "o"), (18, "-", "s")):
        R = Rs[L]
        for tag, M, col in (("ground", ground_matrix(L) ** 2, AQUA),
                            (r"$\beta$=1", load_mean(L, 1), BLUE),
                            (r"$\beta$=3", load_mean(L, 3), ORANGE)):
            t = shell_table(M, R, rmax=0.62 * L)
            r = np.array([x[0] for x in t])
            y = np.array([x[2] for x in t])
            ax.loglog(r, y, ls, color=col, marker=mk, ms=3, lw=1.0, mfc="white",
                      mew=0.9, label=f"L={L} {tag}" if L == 18 else None)
    ax.set_xlabel("r  (lattice units)")
    ax.set_ylabel(r"$\langle\,|G_{E,ij}|^2\,\rangle$")
    style(ax, r"A. Radial profile — shells scatter, the $\beta$=3 tail is flat")
    ax.legend(fontsize=7.5, frameon=False, loc="lower left")

    # P2: alpha vs window ---------------------------------------------------
    ax = axes[0, 1]
    ax.axhspan(3.4, 3.6, color=BLUE, alpha=0.10, lw=0)
    ax.axhspan(1.0, 1.5, color=ORANGE, alpha=0.14, lw=0)
    ax.axhline(4.0, color=INK, ls=":", lw=1.1)
    ax.text(2.95, 4.07, r"ground truth $\alpha=4$", fontsize=7.5, color=INK2, ha="right")
    for L, mk, ls in ((12, "s", "--"), (18, "o", "-")):
        for b, col in ((1, BLUE), (3, ORANGE)):
            rows_ = [w for w in window[f"L{L}_beta{b}"] if w["rmin"] == 0.9]
            ax.plot([w["rmax"] for w in rows_], [w["alpha"] for w in rows_],
                    mk + ls, color=col, ms=4.5, lw=1.2, mfc="white", mew=1.2,
                    label=fr"L={L} $\beta$={b}")
    ax.set_xlabel(r"$r_{\max}$ of fit window  ($r_{\min}=0.9$)")
    ax.set_ylabel(r"fitted $\alpha$")
    style(ax, r"A. Exponent vs window: no plateau, L=12 and L=18 disagree")
    ax.legend(fontsize=7.5, frameon=False, ncol=2)
    ax.set_ylim(0, 4.7)

    # P3: synthetic D_V(L) --------------------------------------------------
    ax = axes[1, 0]
    cmap = plt.get_cmap("Blues")
    for i, al in enumerate(alphas):
        c = cmap(0.32 + 0.62 * i / (len(alphas) - 1))
        ax.loglog(Ls, synth[al], "-", color=c, lw=1.3, marker="o", ms=2.8,
                  label=fr"$\alpha$={al}")
    ax.set_xlabel("L")
    ax.set_ylabel(r"$D_V(L)$")
    style(ax, r"B. Synthetic $D_V$ on the real torus, $\sum|G|^2=L^2$")
    ax.legend(fontsize=6.6, frameon=False, ncol=2, loc="upper left")

    # P4: growth exponent + measured D_V ------------------------------------
    ax = axes[1, 1]
    p = [synth_exp3[al] for al in alphas]
    ax.plot(alphas, p, "o-", color=BLUE, ms=5, lw=1.6, label=r"synthetic $d\ln D_V/d\ln L$")
    ax.plot(alphas, np.maximum(0.0, 4.0 - np.array(alphas)), "k--", lw=1.0,
            label=r"$4-\alpha$")
    ax.axvspan(0.0, 2.0, color=ORANGE, alpha=0.10, lw=0)
    ax.axvspan(2.0, 4.5, color=BLUE, alpha=0.08, lw=0)
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.axvline(4.5, color=MUTED, ls=":", lw=1.0)
    ax.text(2.6, 1.72, "claim says\n$D_V$ constant here", fontsize=7, color=INK2, ha="center")
    ax.text(4.75, 1.30, r"saturates only" "\n" r"above $\alpha\approx4.5$",
            fontsize=7, color=INK2, ha="left")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"growth exponent of $D_V$")
    style(ax, r"B. $D_V\sim L^{\max(0,4-\alpha)}$ — the $\alpha>2$ rule fails")
    ax.legend(fontsize=7, frameon=False, loc="upper right")

    fig.tight_layout(pad=1.6)
    fig.savefig(FIG / "verify_decay.png", dpi=160)
    plt.close(fig)

    # second figure: the measured D_V versus the ln L baseline
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    if base_dv:
        bL = sorted(base_dv)
        ax.loglog(bL, [base_dv[L] for L in bL], "-", color=INK2, lw=1.5,
                  label=r"ground state (true $\ln L$)")
    for b, col, mk in ((1, BLUE, "o"), (3, ORANGE, "s")):
        V = [data_dv[f"L{L}_beta{b}"]["DV"] for L in Ls_data]
        sd = [data_dv[f"L{L}_beta{b}"]["DV_batch_sd"] for L in Ls_data]
        ax.errorbar(Ls_data, V, yerr=sd, fmt=mk, color=col, ms=7, capsize=3, lw=1.3,
                    label=fr"measured $\beta$={b}")
    ax.set_xlabel("L")
    ax.set_ylabel(r"$D_V(L)$")
    style(ax, r"$D_V$: $\beta$=1 tracks the $\ln L$ baseline, $\beta$=3 rises steeply")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "verify_decay_DV.png", dpi=160)
    plt.close(fig)

    rep["figures"] = [str(FIG / "verify_decay.png"), str(FIG / "verify_decay_DV.png")]
    print(json.dumps(rep, indent=1, default=float))
    return rep


if __name__ == "__main__":
    main()
