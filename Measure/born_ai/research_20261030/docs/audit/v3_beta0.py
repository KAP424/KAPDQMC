import sys
sys.path.insert(0, "d:/JuliaDQMC/code/KAPDQMC/src/Measure/born_ai/research_20261030/docs/audit")
from v3_common import *
base = baseline()
print("L  n  Dmean(6dp)  Dmean-Dgs  max|D-Dgs|  min..max spread  SA-diff SB-diff SC-diff")
for L in (6, 12, 18, 24, 30, 36):
    hdr, a = read_csv(f"{R}/data_beta0/L{L}_beta0.0000.csv")
    col = {h: i for i, h in enumerate(hdr)}
    D = a[:, col["D"]]
    g = base[str(L)]
    print(L, D.size, f"{D.mean():.6f}", f"{abs(D.mean()-g['D']):.2e}", f"{np.abs(D-g['D']).max():.2e}",
          f"{D.max()-D.min():.2e}",
          f"{np.abs(a[:,col['SA']]-g['SA']).max():.2e}", f"{np.abs(a[:,col['SB']]-g['SB']).max():.2e}",
          f"{np.abs(a[:,col['SC']]-g['SC']).max():.2e}")
    # also what the table claim is
    print("    baseline D = %.10f ; D mean in file = %.10f" % (g["D"], D.mean()))
