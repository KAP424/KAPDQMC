"""Check the 1D machinery against exact statements that must hold.

1. beta = 0 reproduces the ground-state entropies (normalisation).
2. Full enumeration reproduces p(E) normalisation and the Born mean.
3. The small-z theorem  lim <S_2(R)>/z^2 = 4 f (1-f) + 4 V_0(R)/a^2.
4. DPP sampling reproduces the exact Born distribution of E.
"""

import numpy as np
import onedim_born as ob

rng = np.random.default_rng(20261101)

print("== 1. beta = 0 normalisation ==")
for L in (3, 4, 5):
    Q, _ = ob.chain_state(L)
    N = Q.shape[0]
    regs = {"half": ob.half_ring_region(N)}
    mean, var, p = ob.exact_entropy_average(Q, 0.0, regs)
    gs = ob.ground_state_entropy(Q, regs["half"])
    print(f" L={L}  S2(beta=0)={mean['half']:.12f}  gs={gs:.12f}  diff={mean['half']-gs:+.2e}  sum p={p.sum():.12f}")

print()
print("== 2/3. small-z theorem, full enumeration ==")
for L in (3, 4, 5, 6):
    Q, _ = ob.chain_state(L)
    N, a = Q.shape
    f = 0.5
    reg = ob.half_ring_region(N)
    V0 = ob.ground_state_number_variance(Q, reg)
    pred = 4.0 * f * (1 - f) + 4.0 * V0 / a**2
    for beta in (3.0, 3.5, 4.0):
        mean, var, p = ob.exact_entropy_average(Q, beta, {"half": reg})
        z = a * np.exp(-2 * beta)
        print(f" L={L} beta={beta}  <S2>/z^2={mean['half']/z**2:.6f}   theorem={pred:.6f}"
              f"   (V0={V0:.4f}, nout={len(p)})")

print()
print("== 4. DPP sampler vs exact p(E), L=4 ==")
L = 4
Q, _ = ob.chain_state(L)
beta = 1.0
outcomes, logp = ob.born_probabilities(Q, beta)
p_exact = np.exp(logp)
index = {tuple(E): i for i, E in enumerate(outcomes)}
N, a = Q.shape
ns = 200000
counts = np.zeros(len(outcomes))
jw = np.exp(ob.log_jweights(a, beta) - ob.log_normalizer(a, beta))
jc = np.cumsum(jw)
for _ in range(ns):
    E = ob.sample_outcome(rng, Q, beta, jw, jc)
    counts[index[tuple(E)]] += 1
emp = counts / ns
chi2 = np.sum((emp - p_exact) ** 2 / p_exact)
dof = len(p_exact) - 1
print(f" chi2/dof = {chi2:.1f}/{dof} = {chi2/dof:.3f}   max|emp-p|/p = {np.max(np.abs(emp-p_exact)/p_exact):.3f}")
