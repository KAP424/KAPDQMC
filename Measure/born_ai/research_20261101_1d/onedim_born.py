"""1D free-fermion chain under a single-layer Born measurement.

This is the one-dimensional counterpart of the 2D honeycomb study summarised in
``../SUMMARY.md``.  The correspondence is

    2D honeycomb               1D chain (this module)
    ------------------------   --------------------------------------------
    M = 2 L^2 sites            N = 2 L sites, periodic
    n = L^2 particles          a = L particles, half filling
    |E| = a = L^2 sites        |E| = a = L sites
    z = a t = L^2 e^{-2 beta}  z = a t = L   e^{-2 beta}

so ``a`` is the *particle number* in both cases, exactly as the 2D notes say
("z = a t").  Everything else is copied verbatim: the Kraus operator is
``M_E = exp(-2 beta N_E)`` on a fixed-half output set E, the Born probability is
``p(E) = det(Q^T W_E Q) / Z_a(q)``, and the post-measurement state is the Slater
determinant with orbitals ``W_E^{1/2} Q``.

Observable: second Renyi entropy of an interval R,
``S_2(R) = -ln Tr rho_R^2 = -sum_k ln[nu_k^2 + (1-nu_k)^2]``,
where ``nu_k`` are the eigenvalues of the restricted correlation matrix.

Small-z theorem (same derivation as the 2D one, dimension independent):
``lim_{q->0} <S_2(R)> / q = r(M-r) + 4 V_0(R)`` for ``<N_R> = r/2``, i.e.
``<S_2>/z^2 -> 4 f (1-f) + 4 V_0(R)/a^2``.
For the half-ring (f = 1/2) this is ``1 + 4 V_0 / L^2 -> 1``.

Everything here is plain numpy; no scipy.
"""

import numpy as np

# --------------------------------------------------------------------------
# ground state
# --------------------------------------------------------------------------


def chain_state(L, mu=1e-8):
    """Half-filled periodic chain of N = 2L sites, L particles.

    Nearest-neighbour hopping -1 and a staggered potential +-mu that lifts the
    k = +-pi/2 zero-mode degeneracy (same device as the 2D study).
    Returns Q (N x a, orthonormal columns spanning the occupied space).
    """
    N = 2 * L
    a = L
    idx = np.arange(N)
    H = np.zeros((N, N))
    H[idx, (idx + 1) % N] += -1.0
    H[(idx + 1) % N, idx] += -1.0
    H[idx, idx] += mu * np.where(idx % 2 == 0, 1.0, -1.0)
    evals, evecs = np.linalg.eigh(H)
    Q = np.ascontiguousarray(evecs[:, :a])
    return Q, evals


def half_ring_region(N):
    return np.arange(N // 2)


def fraction_region(N, f):
    """Interval of the first round(f*N) sites (needs f*N integral)."""
    r = int(round(f * N))
    assert abs(r - f * N) < 1e-9
    return np.arange(r)


# --------------------------------------------------------------------------
# exact Born distribution
# --------------------------------------------------------------------------


def log_jweights(a, beta):
    """log C(a,j)^2 q^j, j = 0..a, with q = exp(-4 beta)."""
    lw = np.zeros(a + 1)
    for j in range(1, a + 1):
        lw[j] = lw[j - 1] + 2.0 * np.log((a - j + 1) / j) - 4.0 * beta
    return lw


def log_normalizer(a, beta):
    lw = log_jweights(a, beta)
    m = lw.max()
    return m + np.log(np.sum(np.exp(lw - m)))


def log_born_weight(Q, E, beta):
    """log det(Q^T W_E Q) via QR (stable for small t = exp(-2 beta))."""
    Qt = Q.copy()
    Qt[E, :] *= np.exp(-2.0 * beta)
    R = np.linalg.qr(Qt, mode="r")
    d = np.abs(np.diag(R))
    if np.any(d == 0.0):
        return -np.inf
    return 2.0 * np.sum(np.log(d))


def born_probabilities(Q, beta):
    """Exact p(E) for every fixed-half output set (small N only)."""
    from itertools import combinations

    N, a = Q.shape
    logZ = log_normalizer(a, beta)
    outcomes = np.array(list(combinations(range(N), a)), dtype=np.int64)
    logw = np.array([log_born_weight(Q, E, beta) for E in outcomes])
    logp = logw - logZ
    return outcomes, logp


# --------------------------------------------------------------------------
# exact sampling (stratified: DPP configuration F, error count j, output E)
# --------------------------------------------------------------------------


def _weighted_index(rng, w):
    total = w.sum()
    u = rng.random() * total
    c = np.cumsum(w)
    i = int(np.searchsorted(c, u, side="right"))
    return min(i, len(w) - 1)


def sample_dpp(rng, Q):
    """Projection DPP with P(F) = det(Q_F)^2 (Householder scheme)."""
    N, a = Q.shape
    V = Q.copy()
    chosen = np.empty(a, dtype=np.int64)
    for k in range(a, 0, -1):
        active = V[:, :k]
        w = np.einsum("ij,ij->i", active, active)
        if k < a:
            w[chosen[k:]] = 0.0
        w = np.maximum(w, 0.0)
        i = _weighted_index(rng, w)
        chosen[k - 1] = i
        if k > 1:
            v = active[i].copy()
            nv = np.linalg.norm(v)
            v[k - 1] += np.copysign(nv, v[k - 1])
            v /= np.linalg.norm(v)
            V[:, :k] -= 2.0 * np.outer(V[:, :k] @ v, v)
            V[i, :] = 0.0
    return chosen


def sample_outcome(rng, Q, beta, jweights=None, jcum=None):
    """One exact Born sample E (independent, no Markov chain)."""
    N, a = Q.shape
    F = sample_dpp(rng, Q)
    if jweights is None:
        jweights = np.exp(log_jweights(a, beta) - log_normalizer(a, beta))
        jcum = np.cumsum(jweights)
    j = int(np.searchsorted(jcum, rng.random() * jcum[-1], side="right"))
    j = min(j, a)
    inF = np.zeros(N, dtype=bool)
    inF[F] = True
    Fpos = F[rng.permutation(a)[:j]]
    Fc = np.flatnonzero(~inF)
    Fcpos = Fc[rng.permutation(len(Fc))[: a - j]]
    return np.sort(np.concatenate([Fpos, Fcpos]))


# --------------------------------------------------------------------------
# post-measurement observables
# --------------------------------------------------------------------------


def post_basis(Q, E, beta):
    Qt = Q.copy()
    Qt[E, :] *= np.exp(-2.0 * beta)
    Qo, _ = np.linalg.qr(Qt)
    return Qo


def renyi2(Qo, region):
    """S_2 of the region for the Slater state spanned by orthonormal Qo."""
    s = np.linalg.svd(Qo[region, :], compute_uv=False)
    s2 = np.clip(s * s, 0.0, 1.0)
    return float(-np.sum(np.log(s2 * s2 + (1.0 - s2) ** 2)))


def entropy_trajectory(Q, E, beta, regions):
    Qo = post_basis(Q, E, beta)
    return {name: renyi2(Qo, reg) for name, reg in regions.items()}


def exact_entropy_average(Q, beta, regions):
    """Full enumeration of all fixed-half outputs (small N only)."""
    outcomes, logp = born_probabilities(Q, beta)
    p = np.exp(logp)
    acc = {name: 0.0 for name in regions}
    sq = {name: 0.0 for name in regions}
    for E, pe in zip(outcomes, p):
        vals = entropy_trajectory(Q, E, beta, regions)
        for name in regions:
            acc[name] += pe * vals[name]
            sq[name] += pe * vals[name] ** 2
    mean = {k: v for k, v in acc.items()}
    var = {k: sq[k] - acc[k] ** 2 for k in regions}
    return mean, var, p


# --------------------------------------------------------------------------
# ground-state reference quantities
# --------------------------------------------------------------------------


def ground_state_entropy(Q, region):
    return renyi2(Q, region)


def ground_state_number_variance(Q, region):
    """V_0(R) = Var(N_R) for a Slater state: sum_{i in R, j not in R} |G_ij|^2."""
    G = Q @ Q.T
    mask = np.zeros(Q.shape[0], dtype=bool)
    mask[region] = True
    block = G[np.ix_(mask, ~mask)]
    return float(np.sum(block * block))
