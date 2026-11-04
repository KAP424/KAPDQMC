r"""
Task 2: at L = 2 honeycomb, is the Born-averaged rho_bar bounded BELOW (in S2)
by the Gaussian state having the same one-body correlator?

    claim tested:   S2(rho_bar) >= S2(rho_Gbar)
             i.e.   Tr rho_bar^2 <= Tr rho_Gbar^2

Two levels are checked, because they are different statements:

  LEVEL-FULL  -- the whole N = 8 site system.
      rho_bar = sum_E p(E) |psi_E><psi_E| is an explicit 256 x 256 matrix.
      rho_Gbar is the Gaussian state whose 8x8 correlator is
      Gbar = sum_E p(E) G_E.
      Gbar is exactly the first moment of rho_bar, so this is a clean
      "same first moment" comparison.

  LEVEL-REGION -- the region R in {A, B, C}, which is what the physics wants.
      rho_bar^R = sum_E p(E) rho_E,R, and every rho_E,R is GAUSSIAN with
      correlator G_E[R,R], so the mixture can be built exactly in the
      2^|R| Fock space (|R| <= 4 here).  rho_Gbar^R is Gaussian with
      correlator Gbar[R,R].

Data source: the shipped exact model Q in
research_20261029/exact_strata/L2_conditional_probabilities.npz, and the
geometry A/B/C = (2x1 block, 1x1 block, 1x2 block) matching p = 1/2.

Run:  D:/app/conda/python.exe -I exact_l2_gaussian.py
"""

import itertools
import json
import os
from math import comb

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
import sys
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from fermionic_partial_trace import ptrace_fermi  # noqa: E402
NPZ = os.path.join(HERE, "..", "research_20261029", "exact_strata",
                   "L2_conditional_probabilities.npz")
N, A_PART = 8, 4
LN2 = np.log(2.0)


# ------------------------------------------------------------------ entropy

def ren2_from_corr(C):
    nu = np.clip(np.linalg.eigvalsh(0.5 * (C + C.conj().T)).real, 0.0, 1.0)
    return float(-np.sum(np.log(nu * nu + (1.0 - nu) ** 2)))


def ren2_from_rho(rho):
    rho = 0.5 * (rho + rho.conj().T)
    return float(-np.log(max(np.trace(rho @ rho).real, 1e-300)))


def purity_from_rho(rho):
    rho = 0.5 * (rho + rho.conj().T)
    return float(np.trace(rho @ rho).real)


# --------------------------------------------------------- Fock-space tools

def popcount(x):
    return bin(x).count("1")


def gaussian_fock(nsit, C):
    """Density matrix of the Gaussian state with correlator C, in the Fock
    space of `nsit` spinless modes (occupation basis, bit i = site i).

    Exact product form, no matrix exponential and no log of near-degenerate
    occupations:  with d_k = sum_i V[i,k]^* c_i the C eigenbasis,

        rho = prod_k [ nu_k d_k^dag d_k + (1 - nu_k) (1 - d_k^dag d_k) ] .

    The factors commute, so the order is irrelevant, and each factor is
    bounded by 1, which keeps the construction well conditioned even when
    every nu_k sits at 0 or 1.
    """
    nu, V = np.linalg.eigh(0.5 * (C + C.conj().T))
    nu = np.clip(nu.real, 0.0, 1.0)
    dim = 1 << nsit
    rho = np.eye(dim, dtype=complex)
    for k in range(nsit):
        # d_k^dag d_k on Fock space
        # d_k^dag d_k on Fock space
        n_k = np.zeros((dim, dim), dtype=complex)
        for i in range(nsit):
            for j in range(nsit):
                c = np.conj(V[i, k]) * V[j, k]
                if abs(c) > 1e-16:
                    n_k += c * one_body_op(nsit, i, j)
        rho = rho @ (nu[k] * n_k + (1 - nu[k]) * (np.eye(dim) - n_k))
    rho = 0.5 * (rho + rho.conj().T)
    return rho / np.trace(rho).real


_OP_CACHE = {}


def one_body_op(nsit, i, j):
    """c_i^dag c_j as a dense matrix on the Fock space."""
    key = (nsit, i, j)
    if key in _OP_CACHE:
        return _OP_CACHE[key]
    dim = 1 << nsit
    M = np.zeros((dim, dim))
    for s in range(dim):
        if not (s >> j) & 1:
            continue
        sg = (-1) ** popcount(s & ((1 << j) - 1))          # remove j
        t0 = s & ~(1 << j)
        sg *= (-1) ** popcount(t0 & ((1 << i) - 1))        # add i
        t = t0 | (1 << i)
        if not (t0 >> i) & 1:
            M[t, s] += sg
    _OP_CACHE[key] = M
    return M


def slater_fock_vector(nsit, Q):
    """Fock vector of the Slater determinant spanned by Q's columns."""
    npart = Q.shape[1]
    amps = np.zeros(1 << nsit, dtype=complex)
    for occ in itertools.combinations(range(nsit), npart):
        rows = np.array(occ)
        amps[sum(1 << i for i in occ)] = np.linalg.det(Q[rows, :])
    return amps / np.linalg.norm(amps)


def corr_of_fock(nsit, rho):
    """One-body correlator <c_i^dag c_j> of a Fock-space density matrix."""
    C = np.zeros((nsit, nsit), dtype=complex)
    for i in range(nsit):
        for j in range(nsit):
            C[i, j] = np.trace(rho @ one_body_op(nsit, i, j))
    return C


def partial_trace(rho, nsit, keep):
    """Trace out every site not in `keep`; keep is a sorted site list."""
    keep = list(keep)
    rest = [i for i in range(nsit) if i not in set(keep)]
    dimk = 1 << len(keep)
    dimr = 1 << len(rest)
    # index maps: new index k <-> full index
    def full_index(k, r):
        s = 0
        for b, site in enumerate(keep):
            if (k >> b) & 1:
                s |= 1 << site
        for b, site in enumerate(rest):
            if (r >> b) & 1:
                s |= 1 << site
        return s
    out = np.zeros((dimk, dimk), dtype=complex)
    for k1 in range(dimk):
        for k2 in range(dimk):
            acc = 0.0
            for r in range(dimr):
                acc += rho[full_index(k1, r), full_index(k2, r)]
            out[k1, k2] = acc
    return out


# ----------------------------------------------------------------- geometry

def regions_p_half():
    """A/B/C for L = 2, p = 1/2, exactly as research_20261029.make_geometry."""
    def rect(L, w, h, tx=0, ty=0):
        return np.array([2 * ((x + tx) % L + L * ((y + ty) % L)) + s
                         for y in range(h) for x in range(w) for s in (0, 1)])
    return dict(A=rect(2, 1, 2), B=rect(2, 1, 1), C=rect(2, 2, 1))


def kappa_finite(a, beta):
    t = np.exp(-2.0 * beta)
    q = t * t
    Z = sum(comb(a, j) ** 2 * q ** j for j in range(a + 1))
    Zm = sum(comb(a - 1, j) ** 2 * q ** j for j in range(a))
    return 1.0 - (1.0 - t) ** 2 * Zm / Z


# --------------------------------------------------------------------- main

def main():
    data = np.load(NPZ, allow_pickle=True)
    Q = np.ascontiguousarray(data["Q"])
    G0 = Q @ Q.conj().T
    regs = regions_p_half()
    outcomes = list(itertools.combinations(range(N), A_PART))

    print("=" * 80)
    print("TASK 2 -- exact L=2: rho_bar vs the Gaussian state with the same first moment")
    print("=" * 80)
    print(f"N = {N} sites, a = {A_PART} particles, outcomes C(8,4) = {len(outcomes)}")
    print(f"regions: " + ", ".join(f"{k}={v.tolist()}" for k, v in regs.items()))
    print(f"diag(G0) = {np.round(np.diag(G0).real, 8)}")
    print()

    psi0 = slater_fock_vector(N, Q)
    ns = np.array([popcount(s) for s in range(1 << N)])
    pnum = np.abs(psi0) ** 2
    print(f"Fock |Psi_0>: norm = {np.linalg.norm(psi0):.15f}, "
          f"support = {int(np.sum(pnum > 1e-14))} of {1 << N}")
    print(f"  <N> = {pnum @ ns:.12f}, Var(N) = {pnum @ ns**2 - (pnum @ ns)**2:.3e}")
    print(f"  ||G(psi0) - G0||_max = {np.abs(corr_of_fock(N, np.outer(psi0, psi0.conj())) - G0).max():.2e}")
    print()

    report = {"N": N, "a": A_PART, "n_outcomes": len(outcomes), "cases": []}

    for beta in (0.5, 1.0, 2.0):
        t = np.exp(-2.0 * beta)
        d = np.array([popcount(s) for s in range(1 << N)])
        # per-outcome Born weight and filtered pure Fock vector
        w = np.empty(len(outcomes))
        vecs = []
        GE = []
        for k, E in enumerate(outcomes):
            mask = sum(1 << i for i in E)
            ne = np.array([popcount(s & mask) for s in range(1 << N)])
            filt = psi0 * (t ** ne)
            w[k] = np.vdot(filt, filt).real
            v = filt / np.sqrt(w[k])
            vecs.append(v)
            GE.append(corr_of_fock(N, np.outer(v, v.conj())))
        p = w / w.sum()

        # ---- full-system rho_bar
        rho_bar = np.zeros((1 << N, 1 << N), dtype=complex)
        for pk, v in zip(p, vecs):
            rho_bar += pk * np.outer(v, v.conj())
        rho_bar = 0.5 * (rho_bar + rho_bar.conj().T)

        # ---- first moment
        Gbar = sum(pk * Gk for pk, Gk in zip(p, GE))
        Gbar = 0.5 * (Gbar + Gbar.conj().T)

        # ---- Gaussian with the same first moment
        rho_G = gaussian_fock(N, Gbar)

        # ---- checks
        G_of_barmix = corr_of_fock(N, rho_bar)
        G_of_G = corr_of_fock(N, rho_G)
        kap = kappa_finite(A_PART, beta)

        S2_barmix = ren2_from_rho(rho_bar)
        S2_G = ren2_from_rho(rho_G)
        S2_G_corrformula = ren2_from_corr(Gbar)
        tr2_b, tr2_G = purity_from_rho(rho_bar), purity_from_rho(rho_G)

        # mean TRAJECTORY region entropies
        mean_traj = {k: float(p @ np.array([ren2_from_corr(G[np.ix_(regs[k0], regs[k0])])
                                            for G, k0 in [(G, k) for G in GE]]))
                     for k in ("A", "B", "C")}

        print("-" * 80)
        print(f"beta = {beta}   t = e^(-2 beta) = {t:.6e}   kappa_finite = {kap:.10f}")
        print(f"  ||G(rho_bar) - Gbar||_max      = {np.abs(G_of_barmix - Gbar).max():.2e}"
              f"   (Gbar IS the first moment of rho_bar)")
        print(f"  ||G(rho_G) - Gbar||_max        = {np.abs(G_of_G - Gbar).max():.2e}")
        print(f"  ||Gbar - [kappa G0 + (1-kappa)/2 I]||_max = "
              f"{np.abs(Gbar - (kap * G0 + (1 - kap) / 2 * np.eye(N))).max():.2e}")
        print()
        print(f"  LEVEL-FULL  S2(rho_bar)  = {S2_barmix:.12f}")
        print(f"  LEVEL-FULL  S2(rho_Gbar) = {S2_G:.12f}   "
              f"(correlator formula: {S2_G_corrformula:.12f})")
        print(f"  Tr rho_bar^2 = {tr2_b:.10e},  Tr rho_G^2 = {tr2_G:.10e}")
        print(f"  rank(rho_bar) = {int(np.sum(np.linalg.eigvalsh(rho_bar) > 1e-12))}"
              f" of {1 << N}")
        print(f"  ||rho_bar - rho_G||_F = {np.linalg.norm(rho_bar - rho_G):.6e}")
        print()
        print(f"  S2(rho_bar) - S2(rho_G) = {S2_barmix - S2_G:+.6e}"
              f"   -> claim {'HOLDS' if S2_barmix >= S2_G - 1e-10 else 'FAILS'}")
        print()

        # ---- region level
        print(f"  {'R':>3} {'|R|':>4} {'mean_E S2(rho_E,R)':>21} {'S2(rho_bar^R)':>17} "
              f"{'S2(Gbar_R)':>14} {'barR - Gbar':>14}")
        rows = {}
        for key in ("A", "B", "C"):
            idx = regs[key]
            dimk = 1 << len(idx)
            rho_bar_R = np.zeros((dimk, dimk), dtype=complex)
            for pk, G in zip(p, GE):
                rho_bar_R += pk * gaussian_fock(len(idx), G[np.ix_(idx, idx)])
            rho_bar_R = 0.5 * (rho_bar_R + rho_bar_R.conj().T)
            S2_barR = ren2_from_rho(rho_bar_R)
            S2_GR = ren2_from_corr(Gbar[np.ix_(idx, idx)])
            # Cross-check with the JW-CORRECT fermionic partial trace of each
            # pure trajectory.  A naive partial trace that ignores the
            # Jordan-Wigner string of reordering the kept modes to the front
            # gives the right diagonal but wrong coherences, and hence a wrong
            # entropy, whenever the kept modes are not contiguous.  That is why
            # the sign factor is mandatory here.  This route is independent of
            # gaussian_fock(): it works directly on the 2^N Fock vectors.
            from fermionic_partial_trace import ptrace_fermi
            chk_R = np.zeros((dimk, dimk), dtype=complex)
            for pk, v in zip(p, vecs):
                chk_R += pk * ptrace_fermi(v, N, idx)
            chk = np.abs(chk_R - rho_bar_R).max()
            rows[key] = dict(mean_traj=mean_traj[key], S2_barmix=S2_barR,
                             S2_G=S2_GR, pttrace_check=float(chk))
            print(f"  {key:>3} {len(idx):4d} {mean_traj[key]:21.10f} {S2_barR:17.10f} "
                  f"{S2_GR:14.10f} {S2_barR - S2_GR:+14.3e}")
            print(f"      fermionic partial-trace cross-check = {chk:.2e}")
        print()

        Db = (2 * rows["A"]["S2_barmix"] + rows["C"]["S2_barmix"]) / 3 - rows["B"]["S2_barmix"]
        Dg = (2 * rows["A"]["S2_G"] + rows["C"]["S2_G"]) / 3 - rows["B"]["S2_G"]
        Dm = (2 * rows["A"]["mean_traj"] + rows["C"]["mean_traj"]) / 3 - rows["B"]["mean_traj"]
        print(f"  corner D:  mean_traj = {Dm:.10f}   S2(rho_bar) = {Db:.10f}   "
              f"S2(Gbar) = {Dg:.10f}")
        print()

        report["cases"].append(dict(
            beta=beta, t=float(t), kappa=float(kap),
            S2_full_barmix=S2_barmix, S2_full_G=S2_G,
            Tr2_barmix=tr2_b, Tr2_G=tr2_G,
            frob_dist=float(np.linalg.norm(rho_bar - rho_G)),
            regions={k: v for k, v in rows.items()},
            D_mean_traj=Dm, D_barmix=Db, D_G=Dg))

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'beta':>5} {'S2(rho_bar)':>15} {'S2(rho_Gbar)':>15} {'diff':>14} {'holds':>7}")
    ok = True
    for r in report["cases"]:
        h = r["S2_full_barmix"] >= r["S2_full_G"] - 1e-10
        ok &= h
        print(f"{r['beta']:5.2f} {r['S2_full_barmix']:15.10f} {r['S2_full_G']:15.10f} "
              f"{r['S2_full_barmix']-r['S2_full_G']:+14.3e} {str(h):>7}")
    print()
    print("FULL-SYSTEM: S2(rho_bar) >= S2(rho_Gbar) at every beta:", ok)
    print()
    print(f"{'beta':>5} {'R':>3} {'mean_E S2(rho_E,R)':>20} {'S2(rho_bar^R)':>17} "
          f"{'S2(Gbar_R)':>14} {'holds':>7}")
    okR = True
    for r in report["cases"]:
        for k in ("A", "B", "C"):
            g = r["regions"][k]
            h = g["S2_barmix"] >= g["S2_G"] - 1e-10
            okR &= h
            print(f"{r['beta']:5.2f} {k:>3} {g['mean_traj']:20.10f} "
                  f"{g['S2_barmix']:17.10f} {g['S2_G']:14.10f} {str(h):>7}")
    print()
    print("REGION: S2(rho_bar^R) >= S2(Gbar_R) everywhere:", okR)
    print()
    print("Jensen/concavity direction check (mean trajectory vs mixture):")
    for r in report["cases"]:
        for k in ("A", "B", "C"):
            g = r["regions"][k]
            print(f"  beta={r['beta']:4.2f} {k}: mean_traj - S2(barmix) = "
                  f"{g['mean_traj']-g['S2_barmix']:+.3e}"
                  f"   ({'OK (concave path)' if g['mean_traj']>=g['S2_barmix']-1e-10 else 'VIOLATION'})")
    print()

    with open(os.path.join(HERE, "exact_l2_gaussian.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    print("wrote exact_l2_gaussian.json")


if __name__ == "__main__":
    main()
