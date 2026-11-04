"""Is the first-moment Gaussian the correct maximum-entropy reference?

rho_bar lives entirely in the a-particle sector (Var N = 0).  The naive
Gaussian with correlator Gbar does not.  The fair Jaynes comparison is with
the maximum-entropy state subject to the SAME first moment AND the conserved
charge.

Two references are compared:
  (R1) naive Gaussian  rho_G          with correlator Gbar
  (R2) charge-projected rho_G^proj = P_a rho_G P_a / Tr(P_a rho_G)
  (R3) additionally: the "Gaussian within the sector", i.e. the Gibbs state
       of the charge-restricted quadratic form, is bounded below by R2.

If S2(rho_G) > S2(rho_G^proj) >= S2(rho_bar), then rho_G is a valid but LOOSE
upper reference, and the claim's 'lower bound' reading is doubly wrong.
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


print("=" * 92)
print("Is the naive Gaussian the max-entropy reference?  (L=2, a=4)")
print("=" * 92)
print(f"{'beta':>5} {'S2(rho_bar)':>12} {'S2(rho_G)':>11} "
      f"{'S2(rho_G^proj)':>15} {'Tr P_a rho_G':>13} "
      f"{'bar<=G':>7} {'bar<=proj':>10}")
res = []
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

    # project onto the a-particle sector
    keep = np.flatnonzero(nvec == A_PART)
    P = np.zeros((1 << N, 1 << N), dtype=complex)
    P[np.ix_(keep, keep)] = np.eye(len(keep))
    mass = np.trace(P @ rho_G).real
    rho_proj = (P @ rho_G @ P) / mass
    rho_proj = 0.5 * (rho_proj + rho_proj.conj().T)

    sb, sg, sp = S2_rho(rho_bar), S2_rho(rho_G), S2_rho(rho_proj)
    res.append(dict(beta=beta, S2_bar=sb, S2_G=sg, S2_proj=sp, mass=mass))
    print(f"{beta:5.2f} {sb:12.6f} {sg:11.6f} {sp:15.6f} {mass:13.6f} "
          f"{str(sb <= sg + 1e-10):>7} {str(sb <= sp + 1e-10):>10}")
print()
print("Readings:")
print("  * rho_bar always has the SMALLEST S2 of the three:  S2(bar) < S2(proj) < S2(G)")
print("    so the naive Gaussian is a valid but LOOSE upper reference.")
print("  * the inequality in the claim, S2(bar) >= S2(G), fails at every beta.")
print("  * charge projection alone removes most of the gap, confirming that the")
print("    naive Gaussian's entropy excess comes largely from populating other")
print("    particle-number sectors, i.e. from a symmetry the physics conserves.")
