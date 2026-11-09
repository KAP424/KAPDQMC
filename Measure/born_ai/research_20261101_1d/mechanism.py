"""Mechanism of the 1D drift at fixed z.

The exact identity gives  sum_S det(Q_S)^2 * 2 * sum_sep X^2 = a^2 + 4 V0,
i.e. the *perturbative* prediction <S2>/z^2 -> 1.  The true value is smaller.
Here we check whether the shortfall comes from outcomes where t*|X| is not
small (near-singular Q_S), where S2 saturates instead of following 2 q X^2.
"""

import numpy as np
from itertools import combinations
import onedim_born as ob


def run(L, z, mu=1e-8):
    Q, ev = ob.chain_state(L, mu=mu)
    N, a = Q.shape
    beta = 0.5 * np.log(L / z)
    reg = ob.half_ring_region(N)
    inR = np.zeros(N, dtype=bool)
    inR[reg] = True
    q = np.exp(-4 * beta)
    t = np.exp(-2 * beta)
    rows = []
    for E in combinations(range(N), a):
        E = np.array(E)
        S = np.setdiff1d(np.arange(N), E)
        d = np.linalg.det(Q[S, :])
        d2 = d * d
        s2 = ob.entropy_trajectory(Q, E, beta, {"h": reg})["h"]
        if d2 > 1e-30:
            X = Q[E, :] @ np.linalg.inv(Q[S, :])
            sel = inR[E][:, None] != inR[S][None, :]
            th = 2.0 * np.sum(X[sel] ** 2)
            xmax = t * np.linalg.norm(X, 2)
        else:
            th, xmax = np.nan, np.inf
        rows.append((d2, s2, th, xmax))
    d2 = np.array([r[0] for r in rows])
    s2 = np.array([r[1] for r in rows])
    th = np.array([r[2] for r in rows])
    xm = np.array([r[3] for r in rows])
    tot_w = d2.sum()
    exact = np.sum(d2 * s2) / tot_w / z**2
    pert = np.nansum(d2 * q * th) / tot_w / z**2
    V0 = ob.ground_state_number_variance(Q, reg)
    print(f"L={L:3d} z={z:5.2f} beta={beta:6.3f}  exact S2/z^2={exact:.5f}   "
          f"perturbative={pert:.5f}   (a^2+4V0)/a^2={(a*a+4*V0)/a**2:.5f}")
    # split by how non-perturbative the outcome is
    for thr in (0.3, 1.0, 3.0):
        m = xm > thr
        wbad = d2[m].sum() / tot_w
        s2bad = np.sum(d2[m] * s2[m]) / tot_w / z**2
        print(f"      t*||X|| > {thr:4.1f}: weight={wbad:7.4f}  contributes "
              f"{s2bad:8.5f} of S2/z^2  (perturbative would give "
              f"{np.nansum(d2[m]*q*th[m])/tot_w/z**2:8.5f})")


for L in (7, 9):
    for z in (0.3, 1.0):
        run(L, z)
    print()
