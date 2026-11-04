import sys
sys.path.insert(0, "d:/JuliaDQMC/code/KAPDQMC/src/Measure/born_ai/research_20261030/docs/audit")
from v3_common import *
pts = load_all()
e = pts[(36, 3.0)]
old, sold = 0.24213893125649893, 0.0028362794237245057
d = e["mean"] - old
print("L=36 beta=3: new %.5f +- %.5f ; old %.5f +- %.5f ; diff %.5f -> %.3f sigma" % (e["mean"], e["sem"], old, sold, d, d / np.hypot(e["sem"], sold)))
# more digits for table entries
for key in [(18, 2.0), (30, 2.5108), (36, 2.6931), (24, 2.0), (30, 2.2231), (36, 2.4055)]:
    e = pts[key]; print(key, "%.5f +- %.6f" % (e["mean"], e["sem"]))
# z for L=48, 60 at beta=3
for L in (48, 60):
    print("z(L=%d, beta=3) = %.4f" % (L, L * L * np.exp(-6)))
# pulls with more digits
ref = {5.9343: (18, 2.0), 10.5498: (24, 2.0)}
for zt, rk in ref.items():
    r = pts[rk]
    for L, b in [(30, {5.9343: 2.5108, 10.5498: 2.2231}[zt]), (36, {5.9343: 2.6931, 10.5498: 2.4055}[zt])]:
        e = pts[(L, b)]
        print("z=%.2f L=%d: pull vs ref %.4f sigma, h=%.4f" % (zt, L, (e["mean"] - r["mean"]) / np.hypot(e["sem"], r["sem"]), e["mean"] / baseline()[str(L)]["D"]))
# h from 0.76 to 0.50 claim: h at z=5.93: L=18 0.639, L=30 0.533, L=36 0.498; at z=10.55: L=24 0.761, L=30 0.689, L=36 0.631
