"""Schematic of the two frozen boundary-test geometries (no simulation data)."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'figures/born_scaling/boundary_check'
OUT.mkdir(parents=True, exist_ok=True)
colors = ['#286A9F', '#A78126', '#C45D34']
fig, axes = plt.subplots(2, 3, figsize=(10.5, 7.0))
L = 6
for row, m in enumerate([3, 2]):
    w = L // m
    rectangles = [(w, L), (L, w), (w, L-w)]
    names = ['A', 'C', 'B']
    weights = [f'{m-1}/{m}', f'1/{m}', '-1']
    for col, ((width, height), name, weight, color) in enumerate(zip(rectangles, names, weights, colors)):
        ax = axes[row, col]
        ax.add_patch(Rectangle((0, 0), L, L, facecolor='#f5f5f3', edgecolor='#555555', linestyle='--'))
        ax.add_patch(Rectangle((0, 0), width, height, facecolor=color, alpha=.22, edgecolor='none'))
        for n in range(1, L):
            ax.plot([n, n], [0, L], color='#dddddd', lw=.55, zorder=0)
            ax.plot([0, L], [n, n], color='#dddddd', lw=.55, zorder=0)
        # Solid lines are entanglement cuts. Wrapped directions have no cut.
        if width < L:
            for x in [0, width]: ax.plot([x, x], [0, height], color=color, lw=2.8)
        if height < L:
            for y in [0, height]: ax.plot([0, width], [y, y], color=color, lw=2.8)
        ax.set(xlim=(-.35, L+.35), ylim=(-.35, L+.35), aspect='equal')
        ax.set_xticks([0, w, L]); ax.set_yticks([0, w, L])
        ax.tick_params(labelsize=9, length=0)
        for spine in ax.spines.values(): spine.set_visible(False)
        ax.set_title(f'{name}: {width}/6 × {height}/6 system; weight {weight}', fontsize=10)
        if col == 0:
            ax.set_ylabel(f'p = 1/{m}\n{m*m} translations', fontsize=12, labelpad=12)
        if row == 1: ax.set_xlabel('primitive-cell coordinate', fontsize=10)
fig.suptitle('Boundary-matched geometries on the periodic torus', fontsize=15, y=.985)
fig.text(.5, .027,
         'Solid: entanglement cut. Dashed: periodic seam. Translate all three regions together.\n'
         'Coordinates are schematic; physical B corners are 60° and 120°, not right angles.',
         ha='center', fontsize=10, color='#444444')
fig.subplots_adjust(top=.91, bottom=.13, left=.10, right=.98, hspace=.27, wspace=.18)
for suffix in ['png', 'pdf']:
    fig.savefig(OUT / f'geometry_schematic.{suffix}', dpi=180, bbox_inches='tight')
plt.close(fig)
print(OUT / 'geometry_schematic.png')
