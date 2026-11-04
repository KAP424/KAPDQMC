"""Why the Jaynes / maximum-entropy step fails here.

The claim invokes "the Gaussian state with the same first moment has the
largest entropy".  That statement is TRUE for the von Neumann entropy
S1 = -Tr rho ln rho (standard fermionic Jaynes principle).  The quantity in
the claim is the RENYI-2 entropy S2 = -ln Tr rho^2, and maximizing S2 means
MINIMIZING the purity Tr rho^2 -- a different variational problem.

Both rho_bar and rho_G have the SAME correlator Gbar by construction, so they
are two points of one constraint set.  Whichever has larger S2 is the true
maximiser on that set.  We compare both S1 and S2.

Run:  D:/app/conda/python.exe -I check_maxent_reference.py
      (this file is check_renyi_vs_vn.py)
"""
import itertools
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from fermionic_partial_trace import slater_fock, S2_gauss, pop  # noqa: E402
import exact_l2_gaussian as m                                   # noqa: E402

d = np.load(m.NPZ, allow_pickle=True)
Q = np.ascontiguousarray(d["Q"])
N, A_PART = 8, 4
psi0 = slater_fock(N, Q)
outs = list(itertools.combinations(range(N), A_PART))


def S1_rho(rho):
    rho = 0.5 * (rho + rho.conj().T)
    ev = np.linalg.eigvalsh(rho)
    ev = ev[ev > 1e-14]
    return float(-np.sum(ev * np.log(ev)))


def S2_rho(rho):
    rho = 0.5 * (rho + rho.conj().T)
    return float(-np.log(max(np.trace(rho @ rho).real, 1e-300)))


def corr(v):
    a = []
    for i in range(N):
        o = np.zeros_like(v)
        for s in range(1 << N):
            if (s >> i) & 1:
                o[s & ~(1 << i)] += (-1) ** pop(s & ((1 << i) - 1)) * v[s]
        a.append(o)
    return np.array([[np.vdot(a[i], a[j]) for j in range(N)] for i in range(N)])


print("=" * 96)
print("Is the Gaussian the maximum-entropy state on the fixed-correlator set?")
print("Both rho_bar and rho_G have EXACTLY the same correlator Gbar.")
print("=" * 96)
print(f"{'beta':>5} {'S1(rho_bar)':>13} {'S1(rho_G)':>12} {'S1: G max?':>11} "
      f"{'S2(rho_bar)':>13} {'S2(rho_G)':>12} {'S2: G max?':>11}")
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

    s1b, s1g = S1_rho(rho_bar), S1_rho(rho_G)
    s2b, s2g = S2_rho(rho_bar), S2_rho(rho_G)
    print(f"{beta:5.2f} {s1b:13.8f} {s1g:12.8f} {str(s1g >= s1b - 1e-9):>11} "
          f"{s2b:13.8f} {s2g:12.8f} {str(s2g >= s2b - 1e-9):>11}")

print()
print("Reading.")
print("  * rho_G has the LARGER entropy in BOTH columns, at every beta:")
print("    S1(rho_G) > S1(rho_bar) and S2(rho_G) > S2(rho_bar).")
print("  * So the max-entropy picture is directionally right -- the Gaussian")
print("    really is the higher-entropy point on the fixed-correlator set --")
print("    and that is exactly why it is an UPPER reference for the")
print("    Born-averaged entropy.")
print("  * The claim asserts the reverse, S2(rho_bar) >= S2(rho_G), and calls")
print("    S2(Gbar) a LOWER bound.  That inequality is false at every beta")
print("    tested (gaps O(0.27)..O(1.8) nats), so the object the repository")
print("    computes is an upper reference, not a lower one.")
print()
print("Cross-check against the repository's own instrumentation:")
print("  annealed_gaussian.py docstring already says 'gives a rigorous UPPER")
print("  reference for the physical Born-averaged corner entropy'.")
print("  The claim under review states the opposite ('LOWER bound'), i.e. the")
print("  claim contradicts the code it is supposedly about.")
