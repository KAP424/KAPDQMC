'''Born measurement of the spin-1/2 XXZ chain (hard-core bosons), exact enumeration.

Question: is the collapse of the half-chain Renyi-2 entropy on z = n exp(-2 beta) a
free-fermion accident, or does it survive interactions and criticality?

Model: H = sum_i [ -1/2 (S+_i S-_{i+1} + h.c.) + Delta Sz_i Sz_{i+1} ] on a ring of
N = 2n sites, Sz = 0 sector.  Delta = 0 is the free chain, 0.5 a Luttinger liquid,
1 the Heisenberg point (KT critical), 2 the gapped Neel phase.  Every off-diagonal
element is <= 0, so the ground state psi_F is strictly positive in the occupation
basis (Perron-Frobenius): it has no nodes.  The sublattice rotation that maps H to the
usual antiferromagnetic XXZ chain is diagonal in the occupation basis, so it changes no
entropy.

Measurement: output set E of n sites, amplitude psi_F t^|E cap F| with t = exp(-2 beta),
probability p(E) = sum_F psi_F^2 q^|E cap F| / Z, q = t^2.  Only particle-number
conservation is used, not the Slater structure; sum_E q^|E cap F| = Z_n(q) for every F,
so P(j) = C(n,j)^2 q^j / Z_n(q) holds for any n-particle state.

Small-z prediction for any such state (no free-fermion input):
    S2bar / z^2 -> 1 + 4 V0 / n^2,   V0 = Var(N_A) in the ground state.

Even and odd n differ in the boundary condition of the free-fermion image, so compare
sizes of the same parity when reading finite-size trends.

Usage: D:/app/conda/python.exe xxz_collapse.py [Nmax]   (default 14, about 3 minutes)
'''

import json
import math
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, 'figures')
DELTAS = (0.0, 0.5, 1.0, 2.0)
ZS = (0.01, 0.03, 0.1, 0.3, 1.0, 2.0, 3.0)


def popcount_table(nbits):
    t = np.zeros(1 << nbits, dtype=np.int16)
    for b in range(nbits):
        t[1 << b:1 << (b + 1)] = t[:1 << b] + 1
    return t


class Chain:
    '''Sz = 0 sector of an N = 2n site ring, with region A = the first N/2 sites.'''

    def __init__(self, N):
        self.N, self.n = N, N // 2
        self.pc = popcount_table(N)
        self.states = np.flatnonzero(self.pc == self.n).astype(np.int64)
        self.D = len(self.states)
        self.NA = N // 2
        self.idxA = self.states & ((1 << self.NA) - 1)
        self.idxB = self.states >> self.NA
        self.DA, self.DB = 1 << self.NA, 1 << (N - self.NA)
        # |E cap F| for every pair (output E, configuration F)
        self.K = self.pc[self.states[:, None] & self.states[None, :]]
        self.nA = self.pc[self.idxA]          # particles in region A, per state

    def ground_state(self, delta):
        N, D = self.N, self.D
        index = -np.ones(1 << N, dtype=np.int64)
        index[self.states] = np.arange(D)
        H = np.zeros((D, D))
        ar = np.arange(D)
        for i in range(N):
            j = (i + 1) % N
            bi = (self.states >> i) & 1
            bj = (self.states >> j) & 1
            H[ar, ar] += delta * (bi - 0.5) * (bj - 0.5)
            src = np.flatnonzero(bi != bj)
            dst = index[self.states[src] ^ ((1 << i) | (1 << j))]
            H[dst, src] -= 0.5
        w, v = np.linalg.eigh(H)
        psi = v[:, 0] if v[:, 0].sum() > 0 else -v[:, 0]
        return psi, float(w[1] - w[0])

    def s2(self, amp):
        '''Renyi-2 entropy of region A for an unnormalised amplitude vector.'''
        Phi = np.zeros((self.DA, self.DB))
        Phi[self.idxA, self.idxB] = amp
        M = Phi @ Phi.T
        nrm = float(amp @ amp)
        return -math.log(float((M * M).sum()) / nrm ** 2)

    def born_average(self, psi, beta):
        t = math.exp(-2.0 * beta)
        A = psi[None, :] * t ** self.K
        nrm = (A * A).sum(axis=1)
        p = nrm / nrm.sum()
        s = np.array([self.s2(A[e]) for e in range(self.D)])
        return float((p * s).sum())


def main():
    Nmax = int(sys.argv[1]) if len(sys.argv) > 1 else 14
    sizes = list(range(8, Nmax + 1, 2))
    rec = {}
    for N in sizes:
        ch = Chain(N)
        for delta in DELTAS:
            t0 = time.time()
            psi, gap = ch.ground_state(delta)
            if psi.min() <= 0:
                print('WARNING: ground state not strictly positive, min = %.3e' % psi.min())
            p0 = psi ** 2
            m = float((p0 * ch.nA).sum())
            V0 = float((p0 * ch.nA ** 2).sum() - m * m)
            row = dict(delta=delta, N=N, n=ch.n, gap=gap, V0=V0, nA_mean=m,
                       S2_gs=ch.s2(psi), small_z=1.0 + 4.0 * V0 / ch.n ** 2, S2=[])
            for z in ZS:
                row['S2'].append(ch.born_average(psi, 0.5 * math.log(ch.n / z)))
            rec[(delta, N)] = row
            print('N=%2d Delta=%-3g D=%5d gap=%.2e V0=%.5f min psi=%.1e  [%.1fs]'
                  % (N, delta, ch.D, gap, V0, psi.min(), time.time() - t0), flush=True)
        # beta = 0 must reproduce the ground-state entropy (normalisation check)
        d0 = abs(ch.born_average(psi, 0.0) - rec[(DELTAS[-1], N)]['S2_gs'])
        print('N=%2d beta=0 check: |S2bar - S2_gs| = %.1e' % (N, d0), flush=True)

    ns = np.array([N // 2 for N in sizes], dtype=float)
    for delta in DELTAS:
        print('=' * 78)
        print('Delta = %g   (0: free chain, 0.5: Luttinger liquid, 1: Heisenberg KT point, 2: gapped Neel)' % delta)
        print('S2bar / z^2 for each n; spread = (max-min)/mean over n; g = dlnS2/dln n')
        print('%7s' % 'z' + ''.join('%11s' % ('n=%d' % (N // 2)) for N in sizes) + '%10s%10s' % ('spread', 'g'))
        for k, z in enumerate(ZS):
            vals = np.array([rec[(delta, N)]['S2'][k] for N in sizes])
            g = np.polyfit(np.log(ns), np.log(vals), 1)[0]
            spread = (vals.max() - vals.min()) / vals.mean()
            print('%7.3f' % z + ''.join('%11.5f' % (v / z ** 2) for v in vals) + '%10.3f%10.3f' % (spread, g))
        print('%7s' % 'thm' + ''.join('%11.5f' % rec[(delta, N)]['small_z'] for N in sizes) + '   (1 + 4 V0 / n^2)')
        print('%7s' % 'S2_gs' + ''.join('%11.5f' % rec[(delta, N)]['S2_gs'] for N in sizes) + '   (ground state, not divided by z^2)')

    fig, axes = plt.subplots(2, len(DELTAS), figsize=(14, 6.4), sharex='col')
    cm = plt.get_cmap('viridis')
    zz = np.array(ZS)
    for c, delta in enumerate(DELTAS):
        for i, N in enumerate(sizes):
            y = np.array(rec[(delta, N)]['S2'])
            col = cm(0.1 + 0.8 * i / max(len(sizes) - 1, 1))
            axes[0, c].plot(zz, y, 'o-', ms=3.5, lw=1, color=col, label='n=%d' % (N // 2))
            axes[1, c].plot(zz, y / zz ** 2, 'o-', ms=3.5, lw=1, color=col)
        axes[0, c].set_title('Delta=%g' % delta, fontsize=10)
        axes[1, c].axhline(1.0, color='k', ls='--', lw=0.8)
        axes[1, c].set_ylim(0, 1.3)
        axes[1, c].set_xlabel('z = n exp(-2 beta)')
        axes[0, c].grid(alpha=0.25)
        axes[1, c].grid(alpha=0.25)
    axes[0, 0].set_ylabel('S2bar (half chain)')
    axes[1, 0].set_ylabel('S2bar / z^2')
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, 'xxz_collapse.png'), dpi=180)
    plt.close(fig)

    with open(os.path.join(HERE, 'xxz_collapse.json'), 'w') as fh:
        json.dump([rec[k] for k in sorted(rec)], fh, indent=1)
    print('figure -> ' + os.path.join(FIG, 'xxz_collapse.png'))


if __name__ == '__main__':
    main()
