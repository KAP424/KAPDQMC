"""
Strong-measurement (small-z) amplitude A(L) = lim_{q->0} D_bar / z^2 of the
Born-averaged corner entropy, computed exactly, not fitted.

Structure (Cauchy-Binet): p(E) = sum_F |det Q_F|^2 t^{2|E cap F|} / Z_a, so to
leading order in q = t^2 = z^2/L^4 the Born average splits into

  * full-rank outputs S = E^c (det Q_S != 0): weight det(Q_S)^2, post-measurement
    state = product state on S plus Thouless amplitudes t X, X = Q_E Q_S^{-1};
    every occupation eigenvalue is near 0 or 1, so S2(R) = 2 V(R) + O(nu^2) with
    V(R) = t^2 * sum_{(e,c) separated by R} X_ec^2.   [the typical, j = 0, branch]
  * rank-deficient ("node") outputs, det Q_S = 0 with rank n-1: p(E) = q W_S,
    W_S = product of the n-1 nonzero singular values squared, and O(1) entropy.

Closed form of the full-rank part claimed by the legacy expansion
(research_20261029/theory_review.md section 8):  C_R = r(M - r) + 4 V0(R),
which on the D combination gives 16 L^4/81 + 4 D_V0(L), i.e.
A_bright(L) = 16/81 + 4 D_V0(L)/L^4.  16/81 is exactly the D combination of a
uniformly random pair of sites (area fractions 1/3, 1/3, 2/9): a VOLUME-law
residual, not a corner term.

Checks:
  L = 3   exact enumeration of all C(18,9) outputs: full-rank sum, node sum,
          total vs the known 14.63775, and E||X||^2 = n^2 (Cauchy-Binet).
  L >= 6  exact projection-DPP sampling of S (Householder sequential algorithm):
          full-rank part vs A_bright(L); scan for exact zeros of X (nodes).

Usage: D:/app/conda/python.exe small_z_mechanism.py
"""

import glob
import itertools
import json
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import annealed_gaussian as ag
from crossover_structure import read_csv

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
RNG = np.random.default_rng(20261031)
A_RANDOM_PAIR = 16.0 / 81.0


def ground(L):
    G0 = ag.ground_projector(L, mu=1e-8)
    ev, V = np.linalg.eigh(G0)
    Q = V[:, ev > 0.5]
    assert Q.shape[1] == L * L
    masks = ag.region_masks(L)
    return Q, G0, {k: masks[k] for k in "ABC"}


def V0(G0, m):
    return float(np.sum(G0[np.ix_(m, ~m)] ** 2))


def bright_formula(G0, masks, L):
    M = G0.shape[0]
    C = {k: masks[k].sum() * (M - masks[k].sum()) + 4 * V0(G0, masks[k]) for k in "ABC"}
    CD = (2 * C["A"] + C["C"]) / 3 - C["B"]
    DV0 = (2 * V0(G0, masks["A"]) + V0(G0, masks["C"])) / 3 - V0(G0, masks["B"])
    return C, CD, DV0


def separated(W, re, ce):
    """sum of W[e, c] over pairs with exactly one of (e, c) in the region."""
    return float(re @ W @ (1.0 - ce) + (1.0 - re) @ W @ ce)


def full_rank_terms(Q, S, masks):
    """2 * separated |X|^2 per region for the output with observed set S."""
    N = Q.shape[0]
    E = np.setdiff1d(np.arange(N), S)
    X = np.linalg.solve(Q[S].T, Q[E].T).T          # X = Q_E Q_S^{-1}
    W = X * X
    out = {k: 2.0 * separated(W, masks[k][E].astype(float), masks[k][S].astype(float))
           for k in "ABC"}
    return out, float(W.sum()), X


def ren2_regions(G, masks):
    return {k: ag.ren2(G[np.ix_(masks[k], masks[k])]) for k in "ABC"}


def node_limit_state(Q, S, sv_tol):
    """t -> 0 limit of the post-measurement state for a rank-(n-1) output."""
    N, n = Q.shape
    E = np.setdiff1d(np.arange(N), S)
    U, s, Vt = np.linalg.svd(Q[S])
    keep = s > sv_tol
    assert keep.sum() == n - 1
    v = Vt[~keep][0]                                 # null vector of Q_S
    Phi = np.zeros((N, n))
    Phi[S, : n - 1] = U[:, keep] if U.shape[1] == n else U[:, : n - 1]
    Phi[E, n - 1] = Q[E] @ v
    Phi, _ = np.linalg.qr(Phi)
    W_S = float(np.prod(s[keep] ** 2))
    return Phi @ Phi.T, W_S


# ------------------------------------------- the closed form is an identity
def identity_random_states(rng, sizes=(4, 5, 6), trials=2):
    """sum over ALL outputs S of the adjugate form det(Q_S)^2 * 2*sum_sep X^2
    equals r(M - r) + 4 V0(R), for random particle-hole symmetric states
    G = (1 + S)/2 (so <N_R> = r/2 for every region) and random regions."""
    worst = 0.0
    for n in sizes:
        N = 2 * n
        for _ in range(trials):
            U, _ = np.linalg.qr(rng.normal(size=(n, n)))
            Sm = np.zeros((N, N)); Sm[:n, n:] = U; Sm[n:, :n] = U.T
            G = (np.eye(N) + Sm) / 2
            ev, V = np.linalg.eigh(G)
            Q = V[:, ev > 0.5]
            R = rng.random(N) < 0.4
            r = R.sum()
            closed = r * (N - r) + 4 * float(np.sum(G[np.ix_(R, ~R)] ** 2))
            tot = 0.0
            for S in itertools.combinations(range(N), n):
                S = np.array(S)
                E = np.setdiff1d(np.arange(N), S)
                u, sv, vt = np.linalg.svd(Q[S])
                adjS = np.array([np.prod(np.delete(sv, k)) for k in range(n)])
                adj = np.linalg.det(u) * np.linalg.det(vt.T) * (vt.T * adjS) @ u.T
                W = (Q[E] @ adj) ** 2
                re, ce = R[E].astype(float), R[S].astype(float)
                tot += 2 * (re @ W @ (1 - ce) + (1 - re) @ W @ ce)
            worst = max(worst, abs(tot - closed))
    return worst


# --------------------------------------------------------------- L = 3 exact
def exact_L3():
    L = 3
    Q, G0, masks = ground(L)
    N, n = Q.shape
    C_br, CD_br, DV0 = bright_formula(G0, masks, L)
    acc_full = {k: 0.0 for k in "ABC"}
    acc_node = {k: 0.0 for k in "ABC"}
    acc_adj = {k: 0.0 for k in "ABC"}
    wnode = 0.0
    wsum = 0.0
    xnorm = 0.0
    nnode = 0
    nrank2 = 0
    for S in itertools.combinations(range(N), n):
        S = np.array(S)
        s = np.linalg.svd(Q[S], compute_uv=False)
        if s[-1] > 1e-6:
            w = float(np.prod(s ** 2))
            terms, wx, _ = full_rank_terms(Q, S, masks)
            for k in "ABC":
                acc_full[k] += w * terms[k]
            wsum += w
            xnorm += w * wx
        elif s[-2] > 1e-6:
            G, W_S = node_limit_state(Q, S, 1e-6)
            S2 = ren2_regions(G, masks)
            # adjugate continuation of det^2 * 2*sum_sep X^2 at a rank-(n-1) output:
            # 2 W_S [x(1-y) + y(1-x)], x = weight of the expelled particle (on E) in R,
            # y = weight of the hole (null left vector of Q_S, on S) in R
            E = np.setdiff1d(np.arange(N), S)
            U, sv, Vt = np.linalg.svd(Q[S])
            part = np.zeros(N); part[E] = (Q[E] @ Vt[-1]) ** 2
            hole = np.zeros(N); hole[S] = U[:, -1] ** 2
            for k in "ABC":
                acc_node[k] += W_S * S2[k]
                x, y = part[masks[k]].sum(), hole[masks[k]].sum()
                acc_adj[k] += 2.0 * W_S * (x * (1 - y) + y * (1 - x))
            wnode += W_S
            nnode += 1
        else:
            nrank2 += 1
    comb = lambda d: (2 * d["A"] + d["C"]) / 3 - d["B"]
    CD_full, CD_node = comb(acc_full), comb(acc_node)
    return dict(L=3, CD_full=CD_full, CD_node=CD_node, CD_total=CD_full + CD_node,
                CD_bright_formula=CD_br, DV0=DV0, weight_sum_full=wsum,
                E_X2_full=xnorm, n2=n * n, n_node_outputs=nnode, n_rank_le_n2=nrank2,
                C_full={k: acc_full[k] for k in "ABC"},
                C_node_adjugate={k: acc_adj[k] for k in "ABC"}, node_weight=wnode,
                C_bright_formula={k: C_br[k] for k in "ABC"})


# --------------------------------------------------------------- DPP sampling
def sample_dpp(Q, rng):
    V = Q.copy()
    N, n = V.shape
    F = np.empty(n, dtype=int)
    for k in range(n):
        p = np.einsum("ij,ij->i", V, V)
        p = np.clip(p, 0.0, None)
        i = rng.choice(N, p=p / p.sum())
        F[k] = i
        r = V[i].copy()
        alpha = -np.copysign(np.linalg.norm(r), r[0])
        r[0] -= alpha
        r /= np.linalg.norm(r)
        V = (V - 2.0 * np.outer(V @ r, r))[:, 1:]
    return np.sort(F)


def sampled(L, nsamp, node_tol=1e-6):
    Q, G0, masks = ground(L)
    N, n = Q.shape
    C_br, CD_br, DV0 = bright_formula(G0, masks, L)
    a_vals, x_vals, tiny = [], [], 0
    logx = []
    t0 = time.time()
    for k in range(nsamp):
        S = sample_dpp(Q, RNG)
        terms, wx, X = full_rank_terms(Q, S, masks)
        CD = (2 * terms["A"] + terms["C"]) / 3 - terms["B"]
        a_vals.append(CD / L ** 4)
        x_vals.append(wx / L ** 4)
        ax = np.abs(X).ravel()
        tiny += int(np.sum(ax < node_tol))
        if k < 20:
            logx.append(np.log10(ax + 1e-300))
    a_vals, x_vals = np.array(a_vals), np.array(x_vals)
    return dict(L=L, n=nsamp, A_full=float(a_vals.mean()),
                A_full_sem=float(a_vals.std(ddof=1) / np.sqrt(nsamp)),
                A_bright_formula=float(CD_br / L ** 4), DV0=DV0,
                X2_over_L4=float(x_vals.mean()),
                X2_over_L4_sem=float(x_vals.std(ddof=1) / np.sqrt(nsamp)),
                X2_max_over_L4=float(x_vals.max()),
                tiny_X_fraction=tiny / (nsamp * n * n), seconds=time.time() - t0,
                logx=np.concatenate(logx))


def main():
    print("=" * 76)
    print("Identity check on random particle-hole symmetric states, N = 8, 10, 12")
    print("=" * 76)
    print(f"  max |sum_S adjugate form - (r(M-r) + 4 V0)| = {identity_random_states(RNG):.1e}")
    print()
    print("=" * 76)
    print("L = 3: exact enumeration of all C(18,9) = 48620 outputs")
    print("=" * 76)
    ex = exact_L3()
    print(f"  full-rank weight sum        = {ex['weight_sum_full']:.12f}   (+ node outputs: weight O(q))")
    print(f"  full-rank sum det^2 ||X||^2 = {ex['E_X2_full']:.6f}   (Cauchy-Binet incl. nodes: n^2 = {ex['n2']})")
    print(f"  node outputs (rank n-1)     = {ex['n_node_outputs']},  rank <= n-2: {ex['n_rank_le_n2']}")
    print(f"  node weight sum W_S         = {ex['node_weight']:.6f}   (full + node = {ex['E_X2_full'] + ex['node_weight']:.6f}, Cauchy-Binet n^2 = 81)")
    for k in "ABC":
        tot = ex['C_full'][k] + ex['C_node_adjugate'][k]
        print(f"  C_{k}: full-rank {ex['C_full'][k]:10.5f} + node adjugate {ex['C_node_adjugate'][k]:9.5f}"
              f" = {tot:10.5f}   closed form r(M-r)+4V0 = {ex['C_bright_formula'][k]:10.5f}   diff {tot - ex['C_bright_formula'][k]:+.1e}")
    print(f"  C_D full-rank part  = {ex['CD_full']:.10f}")
    print(f"  C_D node part       = {ex['CD_node']:.10f}")
    print(f"  C_D total           = {ex['CD_total']:.10f}   (known lim D/q = 14.6377513890)")
    print(f"  C_D bright formula  = {ex['CD_bright_formula']:.10f}   (= 16 + 4 D_V0(3), legacy 16.2998418127)")
    print(f"  -> A(3) = {ex['CD_total']/81:.6f};  bright formula/81 = {ex['CD_bright_formula']/81:.6f};"
          f"  16/81 = {A_RANDOM_PAIR:.6f}")

    print("\n" + "=" * 76)
    print("L >= 6: projection-DPP sampling (node weight, and the heavy tail of ||X||^2)")
    print("=" * 76)
    runs = []
    for L, ns in ((6, 40000), (9, 4000), (12, 1500), (18, 300)):
        r = sampled(L, ns)
        runs.append(r)
        print(f"  L={L:2d} n={ns:5d}: E||X||^2/L^4 = {r['X2_over_L4']:.4f} (max single {r['X2_max_over_L4']:.0f})"
              f"   A_full sample mean = {r['A_full']:.4f}   closed form = {r['A_bright_formula']:.5f}"
              f"   |X|<1e-6 fraction = {r['tiny_X_fraction']:.1e}   [{r['seconds']:.0f}s]")
    print("  Cauchy-Binet fixes E||X||^2 = n^2 exactly (full-rank part = n^2 minus node weight);"
          " sample means below 1 are the heavy tail, not missing weight.")

    print("\n" + "=" * 76)
    print("Closed form A(L) = 16/81 + 4 D_V0(L)/L^4 (node weight <= 1e-4 for L >= 6)")
    print("=" * 76)
    closed = {}
    for L in (6, 9, 12, 18, 24, 30, 36):
        _, G0, masks = ground(L)
        _, CD, DV0 = bright_formula(G0, masks, L)
        closed[L] = CD / L ** 4
        print(f"  L={L:2d}: D_V0 = {DV0:.5f}   A = {closed[L]:.6f}")

    print("\n" + "=" * 76)
    print("Born data at z = 0.089: per-region volume-fraction law S2(R)/z^2 -> 4 f_R (1 - f_R)")
    print("=" * 76)
    vol = {"SA": 8 / 9, "SB": 56 / 81, "SC": 8 / 9}
    # Pool every file sharing an L: a point can be split across shard files
    # (L30_beta4.6108.csv.shardKof14.csv), and keying by L alone kept only the
    # last shard, inflating L=30's error bar fourfold.
    pool = {}
    for f in glob.glob(os.path.join(HERE, "data_matched_z", "*.csv")):
        a = read_csv(f)
        if a is None:
            continue
        z = float(a["z"][0])
        if abs(z - 0.089) / 0.089 > 0.01:
            continue
        pool.setdefault(int(a["L"][0]), []).append(a)
    data = {}
    for L, arrs in pool.items():
        z2 = float(arrs[0]["z"][0]) ** 2
        vals = {c: np.concatenate([a[c] for a in arrs]) / z2
                for c in ("D", "SA", "SB", "SC")}
        data[L] = {c: (v.mean(), v.std(ddof=1) / np.sqrt(v.size))
                   for c, v in vals.items()}
    for c in ("SA", "SB", "SC", "D"):
        m = np.array([data[L][c][0] for L in sorted(data)])
        e = np.array([data[L][c][1] for L in sorted(data)])
        w = 1 / e ** 2
        mu, se = np.sum(w * m) / np.sum(w), 1 / np.sqrt(np.sum(w))
        th = vol.get(c, A_RANDOM_PAIR)
        print(f"  {c}: pooled over L={sorted(data)}: {mu:.4f} +- {se:.4f}   leading order {th:.4f}"
              f"   ratio {mu/th:.3f} +- {se/th:.3f}")

    # ------------------------------------------------------------- figure
    fig, axes = plt.subplots(1, 3, figsize=(14.6, 4.3))
    ax = axes[0]
    Ls = sorted(closed)
    ax.plot(Ls, [closed[L] for L in Ls], "s-", ms=5, lw=1.2, color="#2a78d6",
            label=r"exact: $16/81 + 4D_{V0}(L)/L^4$")
    ax.axhline(A_RANDOM_PAIR, color="k", lw=0.9, ls="--", label=r"$16/81$: random-pair geometry")
    ax.plot([3], [ex["CD_total"] / 81], "*", ms=13, color="crimson",
            label=r"$L=3$ exact $=0.1807$ (20% node weight)")
    ax.errorbar(sorted(data), [data[L]["D"][0] for L in sorted(data)],
                yerr=[data[L]["D"][1] for L in sorted(data)], marker="^", ms=5, lw=0,
                elinewidth=1.0, capsize=2, color="0.4", label=r"Born data, $z=0.089$")
    ax.set_xlabel("L"); ax.set_ylabel(r"$A = \lim_{z\to0}\overline{D_2}/z^2$")
    ax.set_ylim(0.0, 0.32)
    ax.set_title(r"(a) small-$z$ amplitude")
    ax.legend(fontsize=6.8, frameon=False, loc="upper right")

    ax = axes[1]
    for c, col, mk in (("SA", "#2a78d6", "o"), ("SC", "#1baf7a", "s"), ("SB", "#eb6834", "D")):
        ax.errorbar(sorted(data), [data[L][c][0] for L in sorted(data)],
                    yerr=[data[L][c][1] for L in sorted(data)], marker=mk, ms=5, lw=0,
                    elinewidth=1.0, capsize=2, color=col, label=rf"$S_2({c[1]})/z^2$")
    ax.axhline(8 / 9, color="#2a78d6", lw=1.0, ls="--", label=r"$4f(1-f)=8/9$  (A, C)")
    ax.axhline(56 / 81, color="#eb6834", lw=1.0, ls="--", label=r"$4f(1-f)=56/81$  (B)")
    ax.set_xlabel("L"); ax.set_ylabel(r"$\overline{S_2(R)}/z^2$ at $z=0.089$")
    ax.set_ylim(0.3, 1.4)
    ax.set_title("(b) region entropies follow the volume fraction")
    ax.legend(fontsize=6.8, frameon=False, loc="upper left", ncol=2)

    ax = axes[2]
    ax.semilogy([3], [ex["node_weight"] / 81], "*", ms=13, color="crimson", label="L=3 exact node weight")
    ax.semilogy([r["L"] for r in runs], [max(r["tiny_X_fraction"], 1e-9) for r in runs], "o",
                ms=6, color="#2a78d6", label=r"$L\geq6$: fraction $|X|<10^{-6}$ (upper bound)")
    ax.set_xlabel("L"); ax.set_ylabel("weight of rank-deficient outputs")
    ax.set_xlim(0, 21); ax.set_ylim(1e-7, 1)
    ax.set_title("(c) node outputs vanish beyond L = 3")
    ax.legend(fontsize=6.8, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "small_z_mechanism.png"), dpi=200)
    fig.savefig(os.path.join(FIG, "small_z_mechanism.pdf"))
    plt.close(fig)

    for r in runs:
        r.pop("logx")
    json.dump(dict(exact_L3=ex, sampled=runs, closed_form_A=closed,
                   data_z0089={str(L): v for L, v in data.items()}),
              open(os.path.join(HERE, "small_z_mechanism.json"), "w"), indent=1, default=float)
    print("\nfigure ->", os.path.join(FIG, "small_z_mechanism.png"))


if __name__ == "__main__":
    main()
