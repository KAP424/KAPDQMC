r"""
Task 1: is S2(rho) = -ln Tr[rho^2] concave in rho?

The claim under review justifies a step "by concavity of S2 in rho".  This
script tests that justification and, crucially, tests the DIRECTION of the
inequality that concavity would actually give.

    Concavity of S2 would give   S2(sum_E p_E rho_E) >= sum_E p_E S2(rho_E).
    The claim asserts            S2(rho_bar)          <= mean_E S2(rho_E).

Those are opposite.  Below:

  (0) exact closed form on rho = diag(x, 1-x): f''(x) = -8x(1-x)/[x^2+(1-x)^2]^2 <= 0
      -> S2 IS concave for d = 2.
  (1) a diagonal-family scan for d = 2..6: concavity survives for d = 2 and
      fails for d >= 3, with an explicit rational witness at d = 3.
  (2) the Born-mixture structure: every |psi_E> is PURE, so S2(psi_E) = 0 and
      mean_E S2(rho_E) = 0 identically.  The claim's inequality then reads
      S2(rho_bar) <= 0, which forces rho_bar to be pure.

Run:  D:/app/conda/python.exe -I verify_concavity.py
"""

import numpy as np

LN2 = np.log(2.0)


def s2vec(v):
    """S2 of a probability vector (diagonal density matrix)."""
    v = np.asarray(v, dtype=float)
    return float(-np.log((v * v).sum()))


def ren2(rho):
    rho = 0.5 * (rho + rho.conj().T)
    return float(-np.log(max(np.trace(rho @ rho).real, 1e-300)))


def second_derivative_2x2(x):
    """f(x) = -ln[x^2+(1-x)^2];  f''(x) = -8x(1-x)/[x^2+(1-x)^2]^2."""
    g = x * x + (1 - x) ** 2
    return -8 * x * (1 - x) / (g * g)


def numeric_second_derivative_2x2(x, h=1e-4):
    f = lambda t: -np.log(t * t + (1 - t) ** 2)
    return (f(x + h) - 2 * f(x) + f(x - h)) / (h * h)


def diagonal_scan(d, lam_grid=None, q1_grid=None):
    """Worst concavity deficit for two DIAGONAL d x d density matrices.

    p = (1,0,...,0) and q = (q1, r, ..., r) with r = (1-q1)/(d-1).  Both live
    in the same basis, which is where the violation sits; random unitaries
    rotate the two states into different bases and hide it.
    """
    lam_grid = np.arange(0.1, 0.95, 0.05) if lam_grid is None else lam_grid
    q1_grid = np.arange(0.4, 1.0, 0.02) if q1_grid is None else q1_grid
    worst, arg = 0.0, None
    p = np.zeros(d)
    p[0] = 1.0
    for lam in lam_grid:
        for q1 in q1_grid:
            rest = (1 - q1) / (d - 1)
            q = np.array([q1] + [rest] * (d - 1))
            mix = lam * p + (1 - lam) * q
            diff = s2vec(mix) - (lam * s2vec(p) + (1 - lam) * s2vec(q))
            if diff < worst:
                worst, arg = float(diff), (float(lam), float(q1))
    return worst, arg


def main():
    print("=" * 78)
    print("TASK 1 -- concavity of S2(rho) = -ln Tr rho^2")
    print("=" * 78)
    print()
    print("(0) Exact closed form on rho = diag(x, 1-x):")
    print("      f(x)   = -ln[x^2 + (1-x)^2]")
    print("      f''(x) = -8x(1-x) / [x^2 + (1-x)^2]^2   <= 0 for all x in [0,1]")
    print()
    print(f"    {'x':>6} {'f(x)':>12} {'f\'\'(exact)':>14} {'f\'\'(numeric)':>14}")
    for x in (0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95):
        print(f"    {x:6.2f} {-np.log(x*x+(1-x)**2):12.6f} "
              f"{second_derivative_2x2(x):+14.6f} "
              f"{numeric_second_derivative_2x2(x):+14.6f}")
    print()
    print("    => S2 IS concave for qubits (d = 2).  So the concavity premise is")
    print("       not nonsense; the problem is the inequality direction.")
    print()

    print("(1) Diagonal-family scan, d = 2..6 (worst deficit; negative = violated):")
    print(f"    {'d':>3} {'worst diff':>14} {'at (lam, q1)':>18}")
    for d in (2, 3, 4, 5, 6):
        worst, arg = diagonal_scan(d)
        note = f"{arg[0]:.2f}, {arg[1]:.2f}" if arg else "-"
        print(f"    {d:3d} {worst:+14.6f} {note:>18}")
    print()
    print("    => concavity holds for d = 2 and FAILS for d >= 3.")
    print()

    print("(2) Explicit rational witness at d = 3 (diagonal, exact arithmetic):")
    p = [1.0, 0.0, 0.0]
    q = [0.71, 0.145, 0.145]
    lam = 0.7
    mix = [lam * a + (1 - lam) * b for a, b in zip(p, q)]
    lhs, rhs = s2vec(mix), lam * s2vec(p) + (1 - lam) * s2vec(q)
    print(f"      rho1 = diag(1.000, 0.000, 0.000)     S2 = {s2vec(p):.12f}")
    print(f"      rho2 = diag(0.710, 0.145, 0.145)     S2 = {s2vec(q):.12f}")
    print(f"      mix  = diag(0.913, 0.0435, 0.0435)   S2 = {lhs:.12f}")
    print(f"      lam S2(rho1) + (1-lam) S2(rho2)          = {rhs:.12f}")
    print(f"      deficit = {lhs - rhs:+.12f}  < 0  -> concavity VIOLATED")
    print()
    print("      The three weights are rationals 1, 71/100, 29/200, so this is a")
    print("      certificate, not a numerical accident.")
    print()

    print("(3) The Born-mixture structure -- what the physics actually needs.")
    print("    rho_bar = sum_E p_E |psi_E><psi_E| with EVERY |psi_E> pure, so")
    print("      S2(|psi_E>) = -ln 1 = 0     for every outcome E")
    print("      mean_E S2(rho_E) = 0        identically, not approximately.")
    print("    The claim's inequality   S2(rho_bar) <= mean_E S2(rho_E)   thus")
    print("    reduces to  S2(rho_bar) <= 0, i.e. rho_bar must be PURE.")
    print("    Any nontrivial Born mixture violates it.")
    print()
    gen = np.random.default_rng(20261030)
    print(f"    {'d':>4} {'n_out':>7} {'S2(rho_bar) min':>17} {'max':>10} {'mean':>10}")
    for d, nterms in ((2, 4), (3, 6), (8, 70), (16, 200)):
        vals = []
        for _ in range(2000):
            psi = []
            for _ in range(nterms):
                v = gen.standard_normal(d) + 1j * gen.standard_normal(d)
                psi.append(v / np.linalg.norm(v))
            p = gen.dirichlet(np.ones(nterms))
            rho = sum(w * np.outer(v, v.conj()) for w, v in zip(p, psi))
            vals.append(ren2(rho))
        vals = np.array(vals)
        print(f"    {d:4d} {nterms:7d} {vals.min():17.8f} {vals.max():10.8f} "
              f"{vals.mean():10.8f}")
    print("    Every value is strictly positive, so S2(rho_bar) <= 0 is false.")
    print()

    print("=" * 78)
    print("SUB-CHECK 1 RESULT")
    print("=" * 78)
    print("  * S2 IS concave for d = 2 and NOT concave for d >= 3 (explicit")
    print("    rational witness at d = 3, deficit -3.95e-3).  A blanket appeal to")
    print("    'concavity of S2' is therefore unsafe in this many-body setting,")
    print("    where the Hilbert space is 2^N-dimensional.")
    print("  * More importantly, the DIRECTION is wrong regardless of concavity:")
    print("    concavity yields  S2(rho_bar) >= mean_E S2(rho_E),  while the claim")
    print("    needs the reverse.  With pure trajectories the RHS is exactly 0,")
    print("    so the claim would force rho_bar to be pure -- false for every")
    print("    beta > 0.")
    print("  * Consequence: no upper (or lower) reference for the Born-averaged")
    print("    entropy follows from this concavity step.  Rigorous bounds must")
    print("    come from the per-trajectory argument instead, which the same")
    print("    repository derives independently and which does hold.")


if __name__ == "__main__":
    main()
