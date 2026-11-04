"""Does the same-first-moment Gaussian conserve particle number?

The Born ensemble is built from a state with EXACTLY a = 4 particles, and the
measurement M_E = exp(-2 beta N_E) is particle-number conserving, so every
|psi_E> has <N> = 4 and Var(N) = 0.  The mixture rho_bar therefore also has
Var(N) = 0.

The Gaussian state with correlator Gbar = kappa G0 + (1-kappa)/2 I does NOT
conserve particle number in general (only when it is a Gibbs state of the
number operator, i.e. G0 proportional to I).  Its Var(N) > 0.  That is an
independent, structural reason why it cannot be the max-entropy comparison
state: the correct maximum-entropy state subject to the SAME FIRST MOMENT
*and* the conserved charge has strictly less entropy than the naive Gaussian.
"""
import itertools
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from fermionic_partial_trace import slater_fock, S2_rho, S2_gauss, pop  # noqa
import exact_l2_gaussian as m                                          # noqa

d = np.load(m.NPZ, allow_pickle=True)
Q = np.ascontiguousarray(d["Q"])
N, A_PART = 8, 4
psi0 = slater_fock(N, Q)
outs = list(itertools.combinations(range(N), A_PART))
nvec = np.array([pop(s) for s in range(1 << N)])


def ann_all(v):
    a = []
    for i in range(N):
        o = np.zeros_like(v)
        for s in range(1 << N):
            if (s >> i) & 1:
                o[s & ~(1 << i)] += (-1) ** pop(s & ((1 << i) - 1)) * v[s]
        a.append(o)
    return a


def corr(v):
    a = ann_all(v)
    return np.array([[np.vdot(a[i], a[j]) for j in range(N)] for i in range(N)])


print("=" * 84)
print("Does the first-moment Gaussian conserve particle number?  (L=2, a=4)")
print("=" * 84)
print(f"{'beta':>5} {'S2(rho_bar)':>12} {'S2(rho_G)':>11} {'Tr rb^2':>10} "
      f"{'Tr rG^2':>10} {'VarN(rho_bar)':>14} {'VarN(rho_G)':>12} "
      f"{'<N>(rho_G)':>11}")
for beta in (0.25, 0.5, 1.0, 2.0, 3.0):
    t = np.exp(-2.0 * beta)
    w = np.empty(len(outs))
    vecs = []
    for k, E in enumerate(outs):
        mask = sum(1 << i for i in E)
        ne = np.array([pop(s & mask) for s in range(1 << N)])
        f = psi0 * (t ** ne)
        w[k] = np.vdot(f, f).real
        vecs.append(f / np.sqrt(w[k]))
    p = w / w.sum()

    rho_bar = np.zeros((1 << N, 1 << N), dtype=complex)
    Gbar = np.zeros((N, N), dtype=complex)
    for pk, v in zip(p, vecs):
        rho_bar += pk * np.outer(v, v.conj())
        Gbar += pk * corr(v)
    rho_bar = 0.5 * (rho_bar + rho_bar.conj().T)
    rho_G = m.gaussian_fock(N, Gbar)

    diag_bar = np.real(np.diag(rho_bar))
    diag_G = np.real(np.diag(rho_G))
    var_bar = float(diag_bar @ nvec**2 - (diag_bar @ nvec) ** 2)
    var_G = float(diag_G @ nvec**2 - (diag_G @ nvec) ** 2)
    mean_G = float(diag_G @ nvec)
    print(f"{beta:5.2f} {S2_rho(rho_bar):12.6f} {S2_gauss(Gbar):11.6f} "
          f"{np.trace(rho_bar@rho_bar).real:10.6f} "
          f"{np.trace(rho_G@rho_G).real:10.6f} {var_bar:14.2e} "
          f"{var_G:12.6f} {mean_G:11.6f}")
print()
print("rho_bar has Var(N) = 0 to roundoff (the charge is exactly conserved).")
print("rho_G has Var(N) >> 0: the naive Gaussian does NOT respect the charge.")
print("=> the 'maximum entropy given the first moment' state is NOT rho_G;")
print("   conditioning on the conserved charge raises the achievable entropy")
print("   and the inequality  S2(rho_bar) >= S2(rho_G)  is the wrong direction.")
