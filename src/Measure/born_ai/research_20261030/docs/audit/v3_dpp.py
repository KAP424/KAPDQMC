import sys, time
sys.dont_write_bytecode = True
sys.path.insert(0, "d:/JuliaDQMC/code/KAPDQMC/src/Measure/born_ai/research_20261030")
import numpy as np
import small_z_mechanism as sz
sz.RNG = np.random.default_rng(424242)
for L, ns in ((6, 8000), (9, 1500), (12, 600), (18, 120)):
    t0 = time.time()
    r = sz.sampled(L, ns)
    print("L=%2d n=%5d  tiny(|X|<1e-6) fraction = %.2e   E||X||^2/L^4 = %.3f (+-%.3f)  max single = %.0f   A_full = %.4f +- %.4f  closed = %.5f  [%.0fs]" % (
        L, ns, r["tiny_X_fraction"], r["X2_over_L4"], r["X2_over_L4_sem"], r["X2_max_over_L4"], r["A_full"], r["A_full_sem"], r["A_bright_formula"], time.time() - t0), flush=True)
