import csv, math
import numpy as np
IID = "d:/JuliaDQMC/code/KAPDQMC/src/Measure/born_ai/research_20261029/iid_targeted"
data = {}
for L in (6, 12, 18):
    for b in (1, 3):
        ch = []
        for bt in (1, 2, 3, 4):
            with open(f"{IID}/L{L}_beta{b}_batch{bt}.csv", newline="") as fh:
                ch.append(np.array([float(r["D"]) for r in csv.DictReader(fh)]))
        data[(L, b)] = np.concatenate(ch)
for k, x in data.items(): print(k, x.size, end="; ")
print()
def top_share(x, q):
    k = max(1, int(math.ceil(x.size * q / 100)))
    s = np.sort(x)[::-1]
    return s[:k].sum() / x.sum()
def exkurt(x):
    d = x - x.mean(); m2 = (d**2).mean(); m4 = (d**4).mean()
    return m4 / m2**2 - 3
def exkurt_unbiased(x):
    n = x.size; d = x - x.mean(); m2 = (d**2).mean(); m4 = (d**4).mean(); g2 = m4/m2**2 - 3
    return (n-1)/((n-2)*(n-3))*((n+1)*g2 + 6)
for b in (1, 3):
    sh1 = [top_share(data[(L, b)], 1) for L in (6, 12, 18)]
    sh5 = [top_share(data[(L, b)], 5) for L in (6, 12, 18)]
    sl = np.polyfit(np.log([6, 12, 18]), np.log(sh1), 1)[0]
    print("beta=%d  top1%% share:" % b, ["%.4f" % v for v in sh1], " log-log slope vs L = %.3f" % sl, "  top5%%:", ["%.4f" % v for v in sh5])
    print("         excess kurtosis (population):", ["%.1f" % exkurt(data[(L, b)]) for L in (6, 12, 18)], " (bias-corrected):", ["%.1f" % exkurt_unbiased(data[(L, b)]) for L in (6, 12, 18)])
    print("         mean/median:", ["%.2f" % (data[(L, b)].mean()/np.median(data[(L, b)])) for L in (6, 12, 18)], " max/mean:", ["%.1f" % (data[(L, b)].max()/data[(L, b)].mean()) for L in (6, 12, 18)])
# CLT slopes with my own RNG, several seeds
ns = (8, 16, 32, 64, 128, 256, 512)
for seed in (1, 2, 3):
    rng = np.random.default_rng(seed)
    out = []
    for L in (6, 12, 18):
        for b in (1, 3):
            x = data[(L, b)]
            sds = []
            for n in ns:
                idx = rng.integers(0, x.size, size=(4000, n))
                sds.append(x[idx].mean(axis=1).std(ddof=1))
            out.append(np.polyfit(np.log(ns), np.log(sds), 1)[0])
    print("seed %d CLT slopes (L6b1,L6b3,L12b1,L12b3,L18b1,L18b3):" % seed, ["%.3f" % v for v in out])
