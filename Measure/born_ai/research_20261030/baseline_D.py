"""Deterministic ground-state corner entropy D(L) for the honeycomb Dirac state.
Validated against the Julia beta=0 production baseline to 1e-8 (see module doc)."""
import numpy as np, json, os
import annealed_gaussian as ag

out = {}
for L in range(6, 63, 3):
    r = ag.annealed_corner(L, 1.0, mu=1e-8)
    out[L] = {k: v for k, v in r.items()}
    print(f"L={L:3d} SA={r['SA']:.6f} SB={r['SB']:.6f} SC={r['SC']:.6f} D={r['D']:.6f}")
json.dump(out, open('baseline_D.json','w'), indent=1)
