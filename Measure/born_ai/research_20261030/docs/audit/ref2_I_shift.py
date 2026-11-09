"""Referee (I)-d: model-free single-scale test.  h_i = H(ln z_i - s_L) with free H (piecewise linear on K knots) and
free shifts s_L (any z*(L), not necessarily a power law).  Restricted to the crossover window h in [hlo,hhi], L>=Lmin."""
import sys; sys.path.insert(0, "d:/JuliaDQMC/code/KAPDQMC/src/Measure/born_ai/research_20261030/docs/audit")
import numpy as np, ref2_common as c
pts, base = c.load()

def build(hlo, hhi, Lmin):
    P = [p for p in pts.values() if p["L"] >= Lmin and hlo <= p["mean"] / p["Dgs"] <= hhi]
    return sorted(P, key=lambda p: (p["L"], p["z"]))

def hat_basis(u, knots):
    B = np.zeros((len(u), len(knots)))
    for i, x in enumerate(u):
        j = np.clip(np.searchsorted(knots, x) - 1, 0, len(knots) - 2)
        t = (x - knots[j]) / (knots[j + 1] - knots[j]); B[i, j] = 1 - t; B[i, j + 1] = t
    return B

def fit(P, K, shifts_free, power_p=None, iters=60):
    L = np.array([p["L"] for p in P]); lz = np.log([p["z"] for p in P]); y = np.array([p["mean"] / p["Dgs"] for p in P])
    s = np.array([p["sem"] / p["Dgs"] for p in P]); Ls = sorted(set(L)); ref = Ls[0]
    sh = {l: 0.0 for l in Ls}
    if power_p is not None:
        sh = {l: power_p * np.log(l / ref) for l in Ls}
    def solve(sh):
        u = lz - np.array([sh[l] for l in L]); knots = np.quantile(u, np.linspace(0, 1, K)) + 1e-9 * np.arange(K)
        B = hat_basis(u, knots); W = 1 / s ** 2
        # tiny curvature penalty to keep unconstrained knots (few points) well posed
        A = B.T @ (B * W[:, None]) + 1e-6 * np.eye(K); bvec = np.linalg.solve(A, B.T @ (W * y))
        return float((W * (y - B @ bvec) ** 2).sum()), bvec, knots
    chi, _, _ = solve(sh)
    if shifts_free:
        grid = np.linspace(-3.5, 3.5, 281)
        for it in range(iters):
            old = chi
            for l in Ls[1:]:
                best = None
                for g in grid:
                    t = dict(sh); t[l] = g; cc = solve(t)[0]
                    if best is None or cc < best[0]: best = (cc, g)
                sh[l] = best[1]
            chi = solve(sh)[0]
            if abs(old - chi) < 1e-6: break
        npar = K + len(Ls) - 1
    elif power_p is None:
        npar = K
    else:
        npar = K
    return chi, len(P) - npar, sh, Ls

for (hlo, hhi, Lmin, K) in ((0.25, 1.15, 12, 6), (0.25, 1.15, 12, 7), (0.35, 1.15, 12, 6), (0.25, 1.15, 18, 6), (0.45, 1.15, 12, 6)):
    P = build(hlo, hhi, Lmin)
    chi0, dof0, _, Ls = fit(P, K, False)                      # all L same shift (pure z collapse)
    chiS, dofS, sh, _ = fit(P, K, True)                       # free shift per L
    print("window h in [%.2f,%.2f], L>=%d, K=%d knots, N=%d points, Ls=%s" % (hlo, hhi, Lmin, K, len(P), Ls))
    print("   pure z-collapse (no shifts): chi2=%.1f dof=%d (%.2f)" % (chi0, dof0, chi0 / max(dof0, 1)))
    print("   free shift per L (single-scale, any z*(L)): chi2=%.1f dof=%d (%.2f); shifts ln z*: %s" % (
        chiS, dofS, chiS / max(dofS, 1), {int(l): round(v, 2) for l, v in sh.items()}))
    best = min(((fit(P, K, False, power_p=pp)[0], pp) for pp in np.linspace(0, 2.2, 45)))
    print("   power-law shifts z*~L^p: best p=%.2f chi2=%.1f (dof ~%d)" % (best[1], best[0], len(P) - K - 1))
