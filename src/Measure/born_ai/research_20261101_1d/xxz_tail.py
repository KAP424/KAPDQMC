'''Why the z collapse degrades with interaction and gap: the heavy tail of the one-hop ratio.

Near an output E whose complement S carries the dominant Fock component, the
post-measurement state is
    psi_S |S> + t * sum_{e in E, c in S} psi_{S-c+e} |S-c+e> + O(t^2),
so the one-hop amplitude is X_ec = psi_{S-c+e} / psi_S (for a Slater state this is
Q_E Q_S^-1).  With S drawn from psi_S^2 the moments are exact:
    w2 = sum_S psi_S^2 sum_ec X_ec^2 = n^2   (identity; used here as a check)
    w4 = sum_S psi_S^2 sum_ec X_ec^4 = sum_S sum_ec psi_{S-c+e}^4 / psi_S^2.
The second-order term of S2bar scales as t^4 w4 = z^4 rho_n with rho_n = w4 / n^4.
Collapse on z at this order needs rho_n to approach a constant as n grows.  If rho_n
keeps growing, the true expansion parameter is not z and S2bar / z^2 drifts down with n.

Usage: D:/app/conda/python.exe xxz_tail.py [Nmax]   (needs xxz_collapse.json)
'''

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from xxz_collapse import Chain, DELTAS, ZS

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, 'figures')


def one_hop_moments(ch, psi):
    states = ch.states
    index = -np.ones(1 << ch.N, dtype=np.int64)
    index[states] = np.arange(ch.D)
    bit = [((states >> i) & 1).astype(bool) for i in range(ch.N)]
    p2 = psi ** 2
    w2 = w4 = 0.0
    for c in range(ch.N):
        for e in range(ch.N):
            if c == e:
                continue
            sel = np.flatnonzero(bit[c] & ~bit[e])
            nb = index[states[sel] ^ ((1 << c) | (1 << e))]
            X = psi[nb] / psi[sel]
            w2 += float((p2[sel] * X ** 2).sum())
            w4 += float((p2[sel] * X ** 4).sum())
    return w2, w4


def main():
    Nmax = int(sys.argv[1]) if len(sys.argv) > 1 else 14
    sizes = list(range(8, Nmax + 1, 2))
    rho = {}
    for N in sizes:
        ch = Chain(N)
        n = ch.n
        for delta in DELTAS:
            psi, gap = ch.ground_state(delta)
            w2, w4 = one_hop_moments(ch, psi)
            rho[(delta, n)] = w4 / n ** 4
            print('n=%d Delta=%-3g  w2/n^2 = %.12f   rho = w4/n^4 = %.4e   log10(psi_max/psi_min) = %.1f'
                  % (n, delta, w2 / n ** 2, w4 / n ** 4, np.log10(psi.max() / psi.min())), flush=True)

    with open(os.path.join(HERE, 'xxz_collapse.json')) as fh:
        recs = {(r['delta'], r['n']): r for r in json.load(fh)}
    k1 = ZS.index(1.0)
    cm = plt.get_cmap('viridis')
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    for i, delta in enumerate(DELTAS):
        col = cm(0.05 + 0.85 * i / (len(DELTAS) - 1))
        for parity, ls, mk in ((0, '-', 'o'), (1, '--', 's')):
            ns = [N // 2 for N in sizes if (N // 2) % 2 == parity]
            if not ns:
                continue
            lab = 'Delta=%g' % delta if parity == 0 else None
            axes[0].semilogy(ns, [rho[(delta, n)] for n in ns], ls, marker=mk, color=col, label=lab)
            red = [recs[(delta, n)]['S2'][k1] / ZS[k1] ** 2 / recs[(delta, n)]['small_z'] for n in ns]
            axes[1].plot(ns, red, ls, marker=mk, color=col, label=lab)
    axes[0].set_xlabel('n (particles)')
    axes[0].set_ylabel('rho_n = w4 / n^4')
    axes[0].set_title('(a) heavy tail of the one-hop ratio X (solid: even n, dashed: odd n)', fontsize=9)
    axes[1].set_xlabel('n (particles)')
    axes[1].set_ylabel('S2bar(z=1) / z^2 / (1 + 4 V0 / n^2)')
    axes[1].set_title('(b) reduction of S2bar below the small-z law at z = 1', fontsize=9)
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.set_xticks(sorted({N // 2 for N in sizes}))
    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, 'xxz_tail.png'), dpi=180)
    plt.close(fig)
    print('figure -> ' + os.path.join(FIG, 'xxz_tail.png'))


if __name__ == '__main__':
    main()
