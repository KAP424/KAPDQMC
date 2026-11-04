"""Reproduce static theory figures; no Monte Carlo observations are generated.

Inputs: saved deterministic FFT results and exact finite-size channel formula.
The crossover curves concern the mean channel, not an entropy fit.
"""
from pathlib import Path
import csv
import hashlib
import json
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from compute_corner_baseline import attenuation

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'figures'
OUT.mkdir(exist_ok=True)
BLUE, ORANGE, GRAY = '#17639a', '#cf6d16', '#555555'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.alpha': .18, 'pdf.fonttype': 42})


def save(fig, name):
    fig.savefig(OUT / (name+'.png'), dpi=180, bbox_inches='tight')
    fig.savefig(OUT / (name+'.pdf'), bbox_inches='tight')
    plt.close(fig)


def main():
    path = ROOT/'baseline/ground_corners.csv'
    data = list(csv.DictReader(path.open()))
    fits = json.loads((ROOT/'baseline/summary.json').read_text())['fits']
    cv = (4+2*np.pi/(3*np.sqrt(3)))/(8*np.pi**2)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), layout='constrained')
    for p, color, marker in [(1/3, BLUE, 'o'), (1/2, ORANGE, 's')]:
        rows = [r for r in data if float(r['mu']) == 0 and float(r['p']) == p]
        sizes = np.array([int(r['L']) for r in rows])
        vals = np.array([float(r['DV']) for r in rows])
        label = 'p = 1/3' if p == 1/3 else 'p = 1/2'
        axes[0].plot(np.log(sizes), vals, marker, mfc='none', ms=5,
                     color=color, label=label+' (FFT)')
        fit = next(f for f in fits if f['mu'] == 0 and f['p'] == p
                   and f['L_min'] == 192 and f['correction_order'] == 2)
        xx = np.geomspace(192, 1536, 150)
        b = fit['coefficient']
        yy = b[0]+b[1]*np.log(xx)+b[2]/xx+b[3]/xx**2
        axes[0].plot(np.log(xx), yy, color=color, lw=1.4,
                     label=label+' (finite-window fit)')
        slopes = np.diff(vals)/np.diff(np.log(sizes))
        axes[1].plot(np.sqrt(sizes[1:]*sizes[:-1]), slopes,
                     marker+'-', mfc='none', color=color, ms=4, label=label)
    axes[0].set(xlabel=r'$\ln L$', ylabel=r'$D_{V,0}$',
                title='(a) Ground-state number-fluctuation corner')
    axes[0].legend(fontsize=9)
    axes[1].axhline(cv, color=GRAY, ls='--', lw=1.2,
                   label=f'Known Dirac value: {cv:.8f}')
    axes[1].set(xscale='log', xlabel=r'$\sqrt{L_iL_{i+1}}$ (log scale)',
                ylabel=r'$\Delta D_{V,0}/\Delta\ln L$',
                title='(b) Local slopes, without fitting')
    axes[1].legend(fontsize=9)
    fig.suptitle('Massless honeycomb reference: two valleys, one species; not Born entropy samples', fontsize=12)
    save(fig, 'ground_variance_corner')

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), layout='constrained')
    beta = np.linspace(0, 4, 161)
    channel_rows = []
    for a, color, style in [(36, BLUE, '-'), (324, ORANGE, '--')]:
        kap = np.array([attenuation(a, b)[0] for b in beta])
        axes[0].plot(beta, kap, style, color=color, label=f'a = {a}')
        channel_rows.extend(dict(a=a, beta=float(b), kappa=float(k)) for b,k in zip(beta,kap))
    axes[0].plot(beta, 1/np.cosh(beta)**2, ':', color='black', label=r'$a\to\infty:\ \mathrm{sech}^2\beta$')
    axes[0].plot(beta, 1/np.cosh(2*beta), '-.', color=GRAY, label=r'Lower bound: $\mathrm{sech}(2\beta)$')
    axes[0].set(yscale='log', xlabel=r'$\beta$', ylabel=r'$\kappa_a(\beta)$ (log scale)',
                title='(a) Exact mean-channel attenuation')
    axes[0].legend(fontsize=9)
    zs = np.geomspace(.01, 20, 180)
    limit = []
    for z in zs:
        terms = [1.]
        for j in range(1, 180):
            terms.append(terms[-1]*(z/j)**2)
        mean = np.dot(np.arange(len(terms)), terms)/sum(terms)
        limit.append(2*(1+mean/z))
    axes[1].plot(zs, limit, color='black', lw=1.5,
                 label=r'Fixed-$z$ limit: $2[1+I_1(2z)/I_0(2z)]$')
    for a,color,mark in [(144, BLUE, 'o'), (5184, ORANGE, 's')]:
        zsample = np.geomspace(.01, 20, 18)
        yy = [attenuation(a, .5*np.log(a/z))[0]/(z/a) for z in zsample]
        axes[1].plot(zsample, yy, mark, mfc='none', color=color, ms=4, label=f'a = {a}')
    axes[1].axhline(2, color=GRAY, lw=.8, ls=':')
    axes[1].axhline(4, color=GRAY, lw=.8, ls=':')
    axes[1].set(xscale='log', xlabel=r'$z=a e^{-2\beta}$ (log scale)',
                ylabel=r'$\kappa_a/e^{-2\beta}$', ylim=(1.9,4.12),
                title='(b) Size / strength crossover of the channel')
    axes[1].legend(fontsize=8.5, loc='lower right')
    fig.suptitle('Analytic channel results; no entropy coefficient is inferred from these curves', fontsize=12)
    save(fig, 'channel_crossover')
    with (ROOT/'baseline/channel_plot_values.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['a','beta','kappa']);w.writeheader();w.writerows(channel_rows)
    receipt = {'role':'Deterministic theory figures; no sampled or fitted entropy coefficient',
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'inputs':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [path, ROOT/'baseline/summary.json', ROOT/'compute_corner_baseline.py']},
        'analytic_Dirac_cV':cv, 'figures':['ground_variance_corner','channel_crossover']}
    (OUT/'theory_figure_receipt.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt,indent=2))


if __name__ == '__main__':
    main()
