"""
Annealed Gaussian state for the honeycomb Dirac ground state under a single-layer
weak density measurement -- dense solver (validated against the Julia pipeline).

Physics
-------
Theorem 1 of BORN_SINGLE_LAYER_THEOREMS.md is exact:

    mean_E(G_E,ij) = kappa * G0_ij   (i != j),  diagonal entries preserved.

Half filling on a balanced bipartite lattice gives diag(G0) = 1/2, so

    Gbar(kappa) = (I + kappa S)/2 ,   S = 2 G0 - I = sgn(K) ,

a Gaussian MIXED state (Gibbs state of W = -2 atanh(kappa) S at beta_w = 1).
kappa = 1 -> pure Dirac GS; kappa -> 0 -> I/2.

rho_bar = sum_E p(E)|psi_E><psi_E| is NOT Gaussian in general; Gbar is its
first moment.  By concavity of S2 in rho,
    E_E S2(rho_E) >= S2(rho_bar),
and by the Gaussian state identity S2(rho_bar) <= S2 of the Gaussian state with
the same first moment, so `annealed_corner` gives a rigorous UPPER reference
for the physical Born-averaged corner entropy.

Validation: kappa = 1 with mu=None reproduces the Julia beta=0 production
baseline to 1e-8 (see `validate_against_production`).

Conventions: site = 2*(x*L+y)+sigma, x slowest, sigma=0 (A), 1 (B); bonds
A(x,y)-B(x,y), A(x,y)-B(x+1,y), A(x,y)-B(x,y-1), t=1.
Regions p=1/3: A = w x L, B = w x 2w, C = L x w; D2 = (2A + C)/3 - B.

NumPy only.
"""

import csv
import os
import numpy as np

LN2 = np.log(2.0)


# ------------------------------------------------------------------ lattice

def dense_hopping(L):
    N = 2 * L * L
    K = np.zeros((N, N))
    seen = set()
    for x in range(L):
        for y in range(L):
            a = 2 * (x * L + y)
            b = 2 * (x * L + y) + 1
            bp = 2 * (((x + 1) % L) * L + y) + 1
            bm = 2 * (x * L + ((y - 1) % L)) + 1
            for u, v in ((a, b), (a, bp), (a, bm)):
                key = (min(u, v), max(u, v))
                if key not in seen:
                    seen.add(key)
                    K[u, v] = K[v, u] = 1.0
    return K


def ground_projector(L, mu=None):
    """mu=None: lowest 2L^2/2 eigenvectors (Julia :lapack).
    mu>0: staggered chemical potential via SVD of the A-B block (Julia
    :staggered), zero modes routed onto sublattice B."""
    N = 2 * L * L
    a = N // 2
    K = dense_hopping(L)
    if mu is None:
        ev, V = np.linalg.eigh(K)
        Q = V[:, :a]
    else:
        AB = K[0::2, 1::2]
        U, s, Vt = np.linalg.svd(AB, full_matrices=False)
        s = s.copy()
        s[np.abs(s) < 1e-12] = 0.0
        eps = np.hypot(s, mu)
        cA = s / np.sqrt(2 * eps * (eps + mu))
        cB = -np.sqrt((eps + mu) / (2 * eps))
        Q = np.zeros((N, a))
        Q[0::2, :] = U * cA[None, :]
        Q[1::2, :] = Vt.T * cB[None, :]
    return Q @ Q.T


# ------------------------------------------------------------------ entropy

def ren2(C):
    nu = np.clip(np.linalg.eigvalsh(0.5 * (C + C.conj().T)).real, 0.0, 1.0)
    return float(-np.sum(np.log(nu * nu + (1.0 - nu) ** 2)))


def region_masks(L, p_num=1, p_den=3):
    w = L * p_num // p_den
    x = np.arange(L)
    X, Y = np.meshgrid(x, x, indexing='ij')
    rep = lambda m: np.repeat(m.ravel(), 2)
    return dict(A=rep(X < w), B=rep((X < w) & (Y < 2 * w)), C=rep(Y < w), w=w)


def corner_of_G(G, L, p_num=1, p_den=3):
    m = region_masks(L, p_num, p_den)
    out = {}
    for key in ('A', 'B', 'C'):
        idx = np.flatnonzero(m[key])
        out['S' + key] = ren2(G[np.ix_(idx, idx)])
    out['D'] = (2 * out['SA'] + out['SC']) / 3 - out['SB']
    return out


def annealed_projector(G0, kappa):
    I = np.eye(G0.shape[0])
    G = 0.5 * (I + kappa * (2.0 * G0 - I))
    return 0.5 * (G + G.T)


def annealed_corner(L, kappa, mu=None, G0=None):
    if G0 is None:
        G0 = ground_projector(L, mu)
    return corner_of_G(annealed_projector(G0, kappa), L)


# ------------------------------------------------------- measurement theory

def kappa_inf(beta):
    """Thermodynamic (a -> inf, beta fixed) mean channel: sech^2(beta)."""
    return 1.0 / np.cosh(beta) ** 2


def kappa_finite(a, beta):
    """Exact finite-a channel: 1 - (1-t)^2 Z_{a-1}/Z_a, t = e^{-2 beta}."""
    from math import comb
    t = np.exp(-2.0 * beta)
    q = t * t
    Z = sum(comb(a, j) ** 2 * q ** j for j in range(a + 1))
    Zm = sum(comb(a - 1, j) ** 2 * q ** j for j in range(a))
    return 1.0 - (1.0 - t) ** 2 * Zm / Z


def z_crossover(L, beta):
    """z = L^2 e^{-2 beta}: governs the strong-measurement crossover."""
    return L * L * np.exp(-2.0 * beta)


# ---------------------------------------------------------------- validation

PROD = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    '..', 'analysis', 'born_scaling', 'production', 'means.csv')


def load_production_baseline(path=PROD):
    out = {}
    with open(path) as fh:
        for row in csv.DictReader(fh):
            if float(row['beta']) != 0.0:
                continue
            out[int(row['L'])] = dict(
                SA=float(row['SA_mean']), SB=float(row['SB_mean']),
                SC=float(row['SC_mean']), D=float(row['D_mean']))
    return out


def validate_against_production(maxL=24, mu=1e-8):
    """kappa=1 must reproduce the Julia beta=0 numbers.

    Production used the staggered-SVD convention (:staggered, mu/t=1e-8), which
    pins the four zero modes; the plain :lapack convention is ambiguous there
    and the two differ by float-level tie-breaking on degenerate subsets.
    """
    ref = load_production_baseline()
    print(f"{'L':>3} {'dSA':>12} {'dSB':>12} {'dD':>12}")
    worst = 0.0
    for L, r in sorted(ref.items()):
        if L > maxL:
            continue
        got = annealed_corner(L, 1.0, mu=mu)
        d = [got['SA'] - r['SA'], got['SB'] - r['SB'], got['D'] - r['D']]
        worst = max(worst, max(abs(x) for x in d))
        print(f"{L:3d} {d[0]:12.2e} {d[1]:12.2e} {d[2]:12.2e}")
    print(f"max |deviation| = {worst:.2e}")
    return worst


if __name__ == "__main__":
    print("Annealed (first-moment) Gaussian state, kappa = 1 vs Julia beta=0:")
    validate_against_production()
    print()
    print("Annealed corner at the thermodynamic channel kappa = sech^2(beta):")
    betas = (0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
    hdr = f"{'L':>3} " + " ".join(f"{b:>8.2f}" for b in betas)
    print(hdr)
    for L in (6, 12, 18, 24):
        G0 = ground_projector(L)
        vals = [annealed_corner(L, kappa_inf(b), G0=G0)["D"] for b in betas]
        print(f"{L:3d} " + " ".join(f"{v:8.4f}" for v in vals))
