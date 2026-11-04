import sys
sys.dont_write_bytecode = True
sys.path.insert(0, "d:/JuliaDQMC/code/KAPDQMC/src/Measure/born_ai/research_20261030")
import numpy as np
import small_z_mechanism as sz
rng = np.random.default_rng(20261031)
w = sz.identity_random_states(rng)
print("identity check (random PH-symmetric N=8,10,12): max err = %.2e" % w)
# other seeds
for sd in (1, 2, 3):
    print("  seed", sd, "max err = %.2e" % sz.identity_random_states(np.random.default_rng(sd)))
ex = sz.exact_L3()
for k in "ABC":
    tot = ex["C_full"][k] + ex["C_node_adjugate"][k]
    print("L=3 region %s: full+node_adjugate = %.10f vs closed %.10f diff %.1e" % (k, tot, ex["C_bright_formula"][k], tot - ex["C_bright_formula"][k]))
print("n_node_outputs", ex["n_node_outputs"], "n_rank_le_n2", ex["n_rank_le_n2"], "node_weight", ex["node_weight"], "node_weight/81", ex["node_weight"] / 81)
print("CD_total", ex["CD_total"])
