r"""
Consolidated adversarial verification of the "annealed Gaussian lower bound"
claim, with figures.

Claim under test (verbatim structure):
  (i)   S2(rho_bar) <= mean_E S2(rho_E)
  (ii)  Tr[rho_bar^2] <= Tr[rho_Gbar^2], i.e. S2(rho_bar) >= S2(rho_Gbar)
  (iii) hence S2(Gbar) is a LOWER bound on the Born-averaged R2 entropy
  (iv)  as kappa -> 0 at fixed L,  D -> 2w(L-2w) ln2  with w = L/3

Run:  D:/app/conda/python.exe -I verify_annealed_claim.py
"""

import itertools
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import annealed_gaussian as ag                     # noqa: E402
from fermionic_partial_trace import (              # noqa: E402
    ptrace_fermi, slater_fock, S2_rho, S2_gauss, pop)

LN2 = np.log(2.0)
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)


def rect(L, w, h, tx=0, ty=0):
    return np.array([2 * ((x + tx) % L + L * ((y + ty) % L)) + s
                     for y in range(h) for x in range(w) for s in (0, 1)])


# =========================================================== concavity probe

def concavity_probe():
    print("=" * 78)
    print("A.  Is S2(rho) = -ln Tr rho^2 concave in rho?")
    print("=" * 78)

    # ---- CLASSICAL / diagonal density matrices are enough to refute global
    #      concavity.  On diagonal states S2 = -ln sum p_i^2, and this is NOT
    #      concave for d >= 3 (it is concave for d = 2).
    p = np.array([1.0, 0.0, 0.0])
    q = np.array([0.71, 0.145, 0.145])
    lam = 0.7
    S2 = lambda v: float(-np.log((v * v).sum()))
    mix = lam * p + (1 - lam) * q
    lhs, rhs = S2(mix), lam * S2(p) + (1 - lam) * S2(q)
    print("Exact d=3 counterexample (DIAGONAL density matrices, so this is")
    print("purely about -ln sum p^2, no quantum subtleties):")
    print(f"  rho1 = diag(1.000, 0.000, 0.000)        S2 = {S2(p):.12f}")
    print(f"  rho2 = diag(0.710, 0.145, 0.145)        S2 = {S2(q):.12f}")
    print(f"  lambda = 0.7 -> mix = diag(0.913, 0.0435, 0.0435)")
    print(f"  S2(mix)                                 = {lhs:.12f}")
    print(f"  lam S2(rho1) + (1-lam) S2(rho2)         = {rhs:.12f}")
    print(f"  difference = {lhs - rhs:+.12f}   "
          f"{'CONCAVITY VIOLATED' if lhs < rhs else 'concave'}")
    print()

    # ---- Where the violations actually live: the two states must share an
    #      eigenbasis (aligned), so scan the diagonal family exactly rather
    #      than sampling random unitaries, which decorrelates the bases and
    #      hides the violation.
    print(f"  {'d':>3} {'worst diff':>13} {'at (lam,q1)':>16}   (exact diagonal scan)")
    scan = {}
    for d in (2, 3, 4, 5, 6):
        lo, arg = 0.0, None
        for lam in np.arange(0.1, 0.95, 0.05):
            for q1 in np.arange(0.4, 1.0, 0.02):
                rest = (1 - q1) / (d - 1)
                if q1 <= rest:
                    continue
                pv = np.zeros(d)
                pv[0] = 1.0
                qv = np.array([q1] + [rest] * (d - 1))
                mv = lam * pv + (1 - lam) * qv
                diff = _s2vec(mv) - (lam * _s2vec(pv) + (1 - lam) * _s2vec(qv))
                if diff < lo:
                    lo, arg = diff, (round(lam, 2), round(q1, 2))
        scan[d] = dict(min_diff=float(lo), at=arg)
        note = f"{arg[0]:.2f},{arg[1]:.2f}" if arg else "-"
        print(f"  {d:3d} {lo:+13.6f} {note:>16}")
    print()
    print("  => S2 = -ln Tr rho^2 IS concave for d=2 (provably: f''<=0) and is")
    print("     NOT concave for d>=3; violations are O(1e-3)..O(1e-2).")
    print()
    print("  Directions matter.  Concavity would give")
    print("        S2(rho_bar) >= sum_E p_E S2(rho_E),")
    print("     i.e. the mixture has the LARGER entropy.  The claim asserts the")
    print("     reverse inequality, and calls it 'concavity'.  Wrong on both")
    print("     counts: the sign is reversed, and the property itself fails for")
    print("     d >= 3.")
    print()
    return scan


def _s2vec(v):
    return float(-np.log((np.asarray(v) ** 2).sum()))


def _near_pure(d, p1, gen):
    g = gen.standard_normal((d, d)) + 1j * gen.standard_normal((d, d))
    U, _ = np.linalg.qr(g)
    w = np.array([p1] + [(1 - p1) / (d - 1)] * (d - 1))
    r = (U * w) @ U.conj().T
    return r / np.trace(r).real


def _rand_density(d, gen):
    g = gen.standard_normal((d, d)) + 1j * gen.standard_normal((d, d))
    ev, V = np.linalg.eigh(g @ g.conj().T)
    w = gen.dirichlet(np.ones(d))
    r = (V * w) @ V.conj().T
    return r / np.trace(r).real


def S2_rho_of(rho):
    rho = 0.5 * (rho + rho.conj().T)
    return float(-np.log(max(np.trace(rho @ rho).real, 1e-300)))


# ==================================================== exact L=2 Born ensemble

def exact_L2(betas=(0.25, 0.5, 1.0, 2.0, 3.0)):
    npz = os.path.join(HERE, "..", "research_20261029", "exact_strata",
                       "L2_conditional_probabilities.npz")
    d = np.load(npz, allow_pickle=True)
    Q = np.ascontiguousarray(d["Q"])
    N, A_PART = 8, 4
    G0 = Q @ Q.conj().T
    psi0 = slater_fock(N, Q)
    regs = dict(A=rect(2, 1, 2), B=rect(2, 1, 1), C=rect(2, 2, 1))
    outcomes = list(itertools.combinations(range(N), A_PART))

    rows = []
    for beta in betas:
        t = np.exp(-2.0 * beta)
        w = np.empty(len(outcomes))
        vecs = []
        for k, E in enumerate(outcomes):
            mask = sum(1 << i for i in E)
            ne = np.array([pop(s & mask) for s in range(1 << N)])
            filt = psi0 * (t ** ne)
            w[k] = np.vdot(filt, filt).real
            vecs.append(filt / np.sqrt(w[k]))
        p = w / w.sum()

        rho_bar = np.zeros((1 << N, 1 << N), dtype=complex)
        for pk, v in zip(p, vecs):
            rho_bar += pk * np.outer(v, v.conj())
        rho_bar = 0.5 * (rho_bar + rho_bar.conj().T)
        S2_full_bar = S2_rho(rho_bar)
        Gbar = _corr_block(vecs, p, N)
        S2_full_G = S2_gauss(Gbar)
        mean_full = 0.0  # every |psi_E> is pure

        row = dict(beta=beta, S2_full_bar=S2_full_bar, S2_full_G=S2_full_G,
                   mean_full=mean_full, regions={})
        for key in ("A", "B", "C"):
            idx = list(regs[key])
            rbar = np.zeros((1 << len(idx),) * 2, dtype=complex)
            traj = np.empty(len(outcomes))
            Gsum = np.zeros((len(idx), len(idx)), dtype=complex)
            for k, v in enumerate(vecs):
                red = ptrace_fermi(v, N, idx)
                rbar += p[k] * red
                traj[k] = S2_rho(red)
                Gsum += p[k] * _corr_of_red(red)
            rbar = 0.5 * (rbar + rbar.conj().T)
            row["regions"][key] = dict(
                mean_traj=float(p @ traj), S2_bar=S2_rho(rbar),
                S2_G=S2_gauss(Gsum), Gbar=Gsum)
        rows.append(row)
    return rows


def _corr_of_red(red):
    """Correlator of an explicitly given reduced density matrix."""
    n = int(round(np.log2(red.shape[0])))
    C = np.zeros((n, n), dtype=complex)
    for i in range(n):
        for j in range(n):
            C[i, j] = np.trace(red @ _nop(n, i, j))
    return C


def _nop(n, i, j):
    dim = 1 << n
    M = np.zeros((dim, dim), dtype=complex)
    for s in range(dim):
        if not (s >> j) & 1:
            continue
        sg = (-1) ** pop(s & ((1 << j) - 1))
        t = s & ~(1 << j)
        sg *= (-1) ** pop(t & ((1 << i) - 1))
        if (t >> i) & 1:
            continue
        M[t | (1 << i), s] += sg
    return M


def _corr_block(vecs, p, N):
    G = np.zeros((N, N), dtype=complex)
    for pk, v in zip(p, vecs):
        G += pk * _psi_corr(v, N)
    return 0.5 * (G + G.conj().T)


def _psi_corr(vec, N):
    def ann(i):
        out = np.zeros_like(vec)
        for s in range(len(vec)):
            if (s >> i) & 1:
                out[s & ~(1 << i)] += (-1) ** pop(s & ((1 << i) - 1)) * vec[s]
        return out
    a = [ann(i) for i in range(N)]
    C = np.zeros((N, N), dtype=complex)
    for i in range(N):
        for j in range(N):
            C[i, j] = np.vdot(a[i], a[j])
    return C


# ================================================== kappa -> 0 volume law

def volume_law():
    print("=" * 78)
    print("C.  kappa -> 0 volume law: D = (2 S_A + S_C)/3 - S_B")
    print("=" * 78)
    print(f"{'L':>4} {'w':>3} {'|A|':>5} {'|B|':>5} {'|C|':>6} "
          f"{'D(kappa=0)':>15} {'2w(L-2w)ln2':>15} {'diff':>11}")
    rows = []
    for L in (6, 9, 12, 18, 24, 36):
        G0 = ag.ground_projector(L, mu=1e-8)
        r = ag.annealed_corner(L, 0.0, G0=G0)
        m = ag.region_masks(L)
        w = m["w"]
        pred = 2 * w * (L - 2 * w) * LN2
        rows.append(dict(L=L, w=w, D=r["D"], pred=pred,
                         nA=int(m["A"].sum()), nB=int(m["B"].sum()),
                         nC=int(m["C"].sum())))
        print(f"{L:4d} {w:3d} {m['A'].sum():5d} {m['B'].sum():5d} "
              f"{m['C'].sum():6d} {r['D']:15.8f} {pred:15.8f} "
              f"{r['D']-pred:+11.2e}")
    print()
    print("  Identity holds to roundoff.  Algebra:")
    print("    |A| = |C| = 2wL,  |B| = 4w^2,  S = n ln2 at Gbar = I/2")
    print("    D = (2*2wL + 2wL)/3 ln2 - 4w^2 ln2 = 2wL ln2 - 4w^2 ln2")
    print("      = 2w(L-2w) ln2.  With w = L/3 this is 2L^2/9 ln2 = O(L^2).")
    print()
    return rows


# =============================================================== main

def main():
    conc = concavity_probe()
    print("=" * 78)
    print("B.  Exact L=2 Born ensemble: rho_bar vs same-first-moment Gaussian")
    print("=" * 78)
    rows = exact_L2()
    print(f"{'beta':>5} {'S2(rho_bar)':>14} {'S2(rho_Gbar)':>14} "
          f"{'mean_E S2(full)':>16} {'(ii) holds':>11}")
    for r in rows:
        holds = r["S2_full_bar"] >= r["S2_full_G"] - 1e-10
        print(f"{r['beta']:5.2f} {r['S2_full_bar']:14.8f} {r['S2_full_G']:14.8f} "
              f"{r['mean_full']:16.8f} {str(holds):>11}")
    print()
    for key in ("A", "B", "C"):
        print(f"  region {key}:")
        print(f"    {'beta':>5} {'mean_E S2(rho_E^R)':>19} {'S2(rho_bar^R)':>15} "
              f"{'S2(Gbar_R)':>13} {'(iii) Gbar<=mean':>18}")
        for r in rows:
            g = r["regions"][key]
            holds = g["S2_G"] <= g["mean_traj"] + 1e-10
            print(f"    {r['beta']:5.2f} {g['mean_traj']:19.8f} "
                  f"{g['S2_bar']:15.8f} {g['S2_G']:13.8f} {str(holds):>18}")
        print()
    vol = volume_law()
    make_figure(conc, rows, vol)
    write_report(conc, rows, vol)


def make_figure(conc, rows, vol):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.0))
    fig.suptitle("Adversarial check of the annealed-Gaussian lower-bound claim\n"
                 "L=2 honeycomb, exact Born enumeration (C(8,4)=70 outcomes)",
                 fontsize=13)

    # ---- panel 1: full system, claim (ii)
    ax = axes[0, 0]
    b = [r["beta"] for r in rows]
    ax.plot(b, [r["S2_full_bar"] for r in rows], "o-", color="#1f77b4",
            label=r"$S_2(\bar\rho)$  exact Born mixture")
    ax.plot(b, [r["S2_full_G"] for r in rows], "s--", color="#d62728",
            label=r"$S_2(\rho_{\bar G})$  same first moment")
    ax.plot(b, [r["mean_full"] for r in rows], "^:", color="#2ca02c",
            label=r"$\mathbb{E}_E S_2(\rho_E)$  (=0, all pure)")
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$S_2$  (nats)")
    ax.set_title(r"(a) full system: claim says $S_2(\bar\rho)\geq S_2(\rho_{\bar G})$"
                 "\n" r"data show the opposite", fontsize=10)
    ax.legend(fontsize=8, loc="center right")
    ax.grid(alpha=0.3)

    # ---- panel 2: region A, claim (iii)
    ax = axes[0, 1]
    mt = [r["regions"]["A"]["mean_traj"] for r in rows]
    sb = [r["regions"]["A"]["S2_bar"] for r in rows]
    sg = [r["regions"]["A"]["S2_G"] for r in rows]
    ax.plot(b, mt, "o-", color="#2ca02c",
            label=r"$\mathbb{E}_E S_2(\rho_E^{A})$  physical")
    ax.plot(b, sb, "s-", color="#1f77b4", label=r"$S_2(\bar\rho^{A})$")
    ax.plot(b, sg, "d--", color="#d62728",
            label=r"$S_2(\bar G_A)$  annealed Gaussian")
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$S_2$  (nats)")
    ax.set_title("(b) region A: the Gaussian curve is an UPPER\n"
                 "reference, not a lower bound", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # ---- panel 3: volume law arithmetic
    ax = axes[1, 0]
    Ls = [v["L"] for v in vol]
    ax.plot(Ls, [v["D"] for v in vol], "o", ms=7, color="#1f77b4",
            label=r"annealed $D$ at $\kappa=0$ (code)")
    ax.plot(Ls, [v["pred"] for v in vol], "-", color="#d62728", lw=1.4,
            label=r"$2w(L-2w)\ln 2$, $w=L/3$")
    ax.set_xlabel("L")
    ax.set_ylabel(r"$D$  (nats)")
    ax.set_title(r"(c) $\kappa\to0$ volume law: identity exact to $10^{-14}$",
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # ---- panel 4: concavity
    ax = axes[1, 1]
    ds = sorted(conc)
    vals = [conc[d]["min_diff"] for d in ds]
    cols = ["#2ca02c" if v >= -1e-12 else "#d62728" for v in vals]
    ax.bar([str(d) for d in ds], vals, color=cols, alpha=0.85)
    ax.axhline(0, color="k", lw=1)
    ax.set_xlabel("local dimension d")
    ax.set_ylabel(r"worst $\left[S_2(\lambda\rho_1+(1-\lambda)\rho_2)"
                  r"-\lambda S_2(\rho_1)-(1-\lambda)S_2(\rho_2)\right]$")
    ax.set_title(r"(d) $S_2=-\ln\mathrm{Tr}\rho^2$: concave for d=2,"
                 "\n" r"NOT concave for $d\geq3$ (red)", fontsize=10)
    ax.grid(alpha=0.3, axis="y")
    ax.text(0.03, 0.06, r"exact d=3 witness: $\Delta S_2=-3.95\times10^{-3}$"
            "\n" r"(diag states, $\lambda=0.7$)",
            transform=ax.transAxes, fontsize=8,
            bbox=dict(fc="wheat", alpha=0.8))

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = os.path.join("D:/JuliaDQMC/code/KAPDQMC/src/Measure/born_ai",
                     "research_20261030", "figures", "annealed_claim_check.png")
    fig.savefig(p, dpi=160)
    plt.close(fig)
    print(f"figure -> {p}")


def write_report(conc, rows, vol):
    out = dict(concavity=conc, exact_L2=[
        dict(beta=r["beta"], S2_full_bar=r["S2_full_bar"],
             S2_full_G=r["S2_full_G"], mean_full=r["mean_full"],
             regions={k: dict(mean_traj=v["mean_traj"], S2_bar=v["S2_bar"],
                              S2_G=v["S2_G"])
                      for k, v in r["regions"].items()})
        for r in rows], volume_law=vol)
    p = os.path.join(HERE, "verify_annealed_claim.json")
    with open(p, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"json -> {p}")


if __name__ == "__main__":
    main()
