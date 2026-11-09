"""Referee (I)-d3: better optimiser for the single-scale test; optional free amplitude per L; knot-count scan."""
import sys; sys.path.insert(0, "d:/JuliaDQMC/code/KAPDQMC/src/Measure/born_ai/research_20261030/docs/audit")
import numpy as np, ref2_common as c
pts, base = c.load()
rng = np.random.default_rng(5)

def hat(u, kn):
    j = np.clip(np.searchsorted(kn, u) - 1, 0, len(kn) - 2); t = (u - kn[j]) / (kn[j + 1] - kn[j])
    B = np.zeros((len(u), len(kn))); B[np.arange(len(u)), j] = 1 - t; B[np.arange(len(u)), j + 1] = t
    return B

class Prob:
    def __init__(s, hlo, hhi, Lmin, K, Lmax=99):
        P = sorted([p for p in pts.values() if Lmin <= p["L"] <= Lmax and hlo <= p["mean"] / p["Dgs"] <= hhi], key=lambda p: (p["L"], p["z"]))
        s.L = np.array([p["L"] for p in P]); s.lz = np.log([p["z"] for p in P]); s.y = np.array([p["mean"] / p["Dgs"] for p in P])
        s.s = np.array([p["sem"] / p["Dgs"] for p in P]); s.Ls = sorted(set(s.L)); s.K = K; s.N = len(P)
        s.idx = {l: np.where(s.L == l)[0] for l in s.Ls}
        # FIXED knots on a common u-grid (shifts move data across them): span of lz - possible shifts
        s.kn = np.linspace(s.lz.min() - 0.2, s.lz.max() + 0.2, K)
    def chi(s, sh, amp=None):
        u = s.lz - np.array([sh[l] for l in s.L]); B = hat(u, s.kn)
        a = np.ones(s.N) if amp is None else np.array([amp[l] for l in s.L])
        W = 1 / s.s ** 2; Bm = B * a[:, None]
        A = Bm.T @ (Bm * W[:, None]) + 1e-8 * np.eye(s.K); b = np.linalg.solve(A, Bm.T @ (W * s.y))
        return float((W * (s.y - Bm @ b) ** 2).sum())
    def run(s, free_amp=False, restarts=6):
        best = (1e99, None, None)
        for r in range(restarts):
            sh = {l: (0.0 if r == 0 else rng.normal(0, 0.4) * np.log(l / s.Ls[0]) / np.log(3)) for l in s.Ls}
            if r == 1: sh = {l: 0.8 * np.log(l / s.Ls[0]) for l in s.Ls}
            if r == 2: sh = {l: 1.8 * np.log(l / s.Ls[0]) for l in s.Ls}
            sh[s.Ls[0]] = 0.0; amp = {l: 1.0 for l in s.Ls}
            cur = s.chi(sh, amp if free_amp else None)
            for it in range(40):
                old = cur
                for l in s.Ls:
                    if l != s.Ls[0]:
                        for g in np.linspace(-1.2, 1.2, 49):
                            t = dict(sh); t[l] = sh[l] + g * (0.5 ** (it // 8)); cc = s.chi(t, amp if free_amp else None)
                            if cc < cur: cur, sh = cc, t
                    if free_amp and l != s.Ls[0]:
                        for g in np.linspace(-0.1, 0.1, 41):
                            t = dict(amp); t[l] = amp[l] + g * (0.5 ** (it // 8)); cc = s.chi(sh, t)
                            if cc < cur: cur, amp = cc, t
                if old - cur < 1e-7: break
            if cur < best[0]: best = (cur, dict(sh), dict(amp))
        return best

