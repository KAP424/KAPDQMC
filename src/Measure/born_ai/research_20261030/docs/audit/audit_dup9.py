import os, json, glob
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DZ = os.path.join(ROOT, "data_matched_z")
DL = os.path.join(ROOT, "data_matched_largez")

a = np.genfromtxt(os.path.join(DZ, "L18_beta1.0000.csv"), delimiter=",", names=True)
b = np.genfromtxt(os.path.join(DL, "L18_beta1.0000.csv"), delimiter=",", names=True)
print("matched L18_beta1.0: n=%d  largez: n=%d" % (a.size, b.size))
print("first 6 D (matched):", np.round(a["D"][:6], 6))
print("first 6 D (largez ):", np.round(b["D"][:6], 6))
print("identical arrays?", a.size == b.size and np.allclose(np.sort(a["D"]), np.sort(b["D"])))
print("corr of sorted D:", np.corrcoef(np.sort(a["D"]), np.sort(b["D"]))[0, 1])
print("j first 6 matched:", a["j"][:6], " largez:", b["j"][:6])
print("sample ids matched:", a["sample"][:6], " largez:", b["sample"][:6])
print("overlap of sample sets:",
      len(set(a["sample"].astype(int)) & set(b["sample"].astype(int))))
for nm, x in (("matched", a), ("largez", b)):
    print("%s: D=%.6f +- %.6f  <j>=%.4f  jmax=%d  frac j=0: %.3f  z=%.4f"
          % (nm, x["D"].mean(), x["D"].std(ddof=1) / np.sqrt(x.size), x["j"].mean(),
             x["j"].max(), np.mean(x["j"] == 0), x["z"][0]))
print("\nt-test-ish on the two means:")
d = a["D"].mean() - b["D"].mean()
se = np.hypot(a["D"].std(ddof=1) / np.sqrt(a.size), b["D"].std(ddof=1) / np.sqrt(b.size))
print("  diff = %.6f  se = %.6f  = %.2f sigma" % (d, se, abs(d) / se))

print()
print("=" * 70)
print("ITEM 9: L=36 beta=3 two values")
print("=" * 70)
f = os.path.join(DZ, "L36_beta3.0000.csv")
c = np.genfromtxt(f, delimiter=",", names=True)
print("file %s: n=%d D=%.6f +- %.6f z=%.4f <j>=%.3f"
      % (os.path.basename(f), c.size, c["D"].mean(),
         c["D"].std(ddof=1) / np.sqrt(c.size), c["z"][0], c["j"].mean()))
print("SUMMARY: new 0.25993 +- 0.01164 ; old chain 0.24214 +- 0.00284 ; diff 1.48 sigma")
d = 0.25993 - 0.24214
se = np.hypot(0.01164, 0.00284)
print("  diff = %.5f, combined se = %.6f -> %.2f sigma" % (d, se, d / se))
# search for the old value anywhere
print()
print("grep for 0.24214 / 0.2421 in the tree:")
