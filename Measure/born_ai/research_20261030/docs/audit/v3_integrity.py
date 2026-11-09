import sys, glob
sys.path.insert(0, "d:/JuliaDQMC/code/KAPDQMC/src/Measure/born_ai/research_20261030/docs/audit")
from v3_common import *
worst = 0.0; nnan = 0; bad_idx = []
nfiles = 0
for d in DIRS + ("data_beta0",):
    for f in sorted(glob.glob(f"{R}/{d}/*.csv")):
        hdr, a = read_csv(f); nfiles += 1
        c = {h: i for i, h in enumerate(hdr)}
        D = a[:, c["D"]]; comb = (2 * a[:, c["SA"]] + a[:, c["SC"]]) / 3 - a[:, c["SB"]]
        worst = max(worst, np.abs(D - comb).max())
        nnan += int(np.isnan(a).sum())
        idx = a[:, c["sample"]].astype(int)
        if not (idx.min() == 1 and np.all(np.diff(idx) == 1)):
            bad_idx.append((d, f.split('/')[-1], idx.min(), idx.max(), idx.size))
print("files", nfiles, " max |D-(2SA+SC)/3+SB| = %.2e" % worst, " NaNs:", nnan)
print("non-contiguous sample index files:", bad_idx)
