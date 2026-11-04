# NUMBER_AUDIT — independent recomputation of every number in `REPORT.md`

Audit performed against the raw sources, with no use of the project's own
analysis modules except the two explicitly labelled ones (baseline solver,
annealed solver) which are themselves re-validated below.

Sources read:

* `data_matched_z/L*_beta*.csv` (columns `L,beta,z,sample,j,SA,SB,SC,D,DV,elapsed`)
* `baseline_D.json` (deterministic `D_gs(L)`, L = 6..60 step 3)
* `crossover_fit.json`, `collapse_points.json`
* `analysis/born_scaling/beta3_large/means.csv` (old L=27,30,36 at beta=3)
* `analysis/born_scaling/production/means.csv` (old beta=0 baseline, used to re-validate `baseline_D.json`)

Script: `audit_numbers.py` (numpy only; all fits via `np.linalg.lstsq` /
normal equations, no scipy).

## Timestamp note (important)

The run was live during this audit. The directory changed at least six times.
The L=36 column was empty when the audit started and had filled in by the end.
**All L=36 verdicts below refer to the state at the end of the audit**, which is:

| file | L | beta | z | n | mean D | sem |
|---|---:|---:|---:|---:|---:|---:|
| `L36_beta0.5000.csv` | 36 | 0.5 | 476.772 | 512 | 0.66454 | 0.01344 |
| `L36_beta1.0000.csv` | 36 | 1.0 | 175.395 | 512 | 0.67674 | 0.02570 |
| `L36_beta1.5000.csv` | 36 | 1.5 | 64.524 | 512 | 0.62686 | 0.02585 |
| `L36_beta2.0000.csv` | 36 | 2.0 | 23.737 | 512 | 0.49806 | 0.02281 |
| `L36_beta3.0000.csv` | 36 | 3.0 | 3.212 | 512 | 0.25993 | 0.01164 |
| `L36_beta3.4054.csv.partial` | — | — | — | in progress | — | — |

---

## 1. Section 2.1 table — D at matched z

Reported vs recomputed (5-decimal):

| z | L=6 | L=12 | L=18 | L=24 | L=30 | L=36 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.089 | **MATCH** 0.00143 | **MATCH** 0.00197 | **MATCH** 0.00141 | **MATCH** 0.00177 | **MISMATCH** rep "—" / data 0.00098 | **MATCH** "—" (no z=0.089 file) |
| 0.357 | **MATCH** 0.02143 | **MATCH** 0.01924 | **MATCH** 0.01780 | **MATCH** 0.01738 | **MISMATCH** rep "—" / data 0.02788 | **MATCH** "—" (absent) |
| 0.803 | **MATCH** 0.07793 | **MATCH** 0.07338 | **MATCH** 0.06892 | **MATCH** 0.07521 | **MATCH** 0.07262 | **MATCH** "—" (absent) |
| 1.428 | **MATCH** 0.14769 | **MATCH** 0.14456 | **MATCH** 0.13681 | **MATCH** 0.12492 | **MATCH** 0.12756 | **MATCH** "—" (absent) |
| 3.213 | **MATCH** 0.25123 | **MATCH** 0.24958 | **MATCH** 0.24876 | **MATCH** 0.26079 | **MATCH** 0.24652 | **MISMATCH** rep 0.24214 / data 0.25993 |

### Verdict: 22 of 30 cells match; 8 problems, in three distinct classes.

**(a) Two "—" cells are wrong: the data exist and were not used.** L=30 at
z=0.089 (0.00098 ± 0.00050) and L=30 at z=0.357 (0.02788 ± 0.00398) are both
present on disk as `L30_beta4.6108.csv` and `L30_beta3.9162.csv`, with a full
512 samples each. These are not un-run points. This matters because the
z=0.357 point is the single largest upward excursion in the whole table
(+42.5% above that row's weighted mean, +2.1 sigma) and it is silently
omitted, which makes the z=0.357 row look flatter than it is.

**(b) The one "—" that was genuinely absent is the L=36 column at z ≤ 0.803.**
Confirmed: no `L36_*` file exists at z=0.089, 0.357, 0.803 or 1.428, and never
did. At the end of the audit the only L=36 completions were beta = 0.5, 1.0,
1.5, 2.0, 3.0 (z = 477, 175, 64.5, 23.7, 3.21). So the L=36 cells for rows
0.089–1.428 are correctly "—", and the footnote "some L=36 files may appear
while you work" is now resolved: they did appear, but only on the beta grid,
not on the matched-z grid.

**(c) The L=36, z=3.213 cell is a real MISMATCH, and it flipped sign.** The
report's 0.24214 comes from the *old* `beta3_large` production chain
(`D_mean = 0.2421389`, `D_MCSE = 0.0028363`), correctly transcribed. But the
new independent run at exactly that (L, beta) now exists:
`L36_beta3.0000.csv`, 512 samples, D = 0.25993 ± 0.01164. That is
**+0.01779 ± 0.01198, i.e. +7.3%, 1.5 sigma** away from the old anchor — and,
critically, in the *opposite* direction from what the report used it for. The
report introduces the L=36 point as evidence that the collapse *continues*
(`0.25123 → 0.24958 → 0.24876 → 0.26079 → 0.24652 → 0.24214`, monotone-ish
drift down). The new point is 0.25993, the *highest* value at that z. The
report's sentence "它把坍缩延伸到 L=36" (it extends the collapse to L=36)
is not supported by the new data; if anything the new point breaks the
monotone drift. The two measurements are statistically compatible at 1.5σ, so
this is a real tension to resolve, not a proven contradiction — but the
report should not present 0.24214 as the L=36 value when a direct, independent
measurement of the same (L, beta) exists and gives a different central value.

### Row-level closure check

Within-row spread (max/min − 1) across L = 6..30:

| z | spread | weighted-mean deviation of worst cell |
|---:|---:|---|
| 0.089 | **100.1%** | L=30 −33.8% (−1.0σ); L=12 +32.4% (+1.4σ) |
| 0.357 | **60.4%** | L=30 +42.5% (**+2.1σ**); L=6 +9.5% (+1.4σ) |
| 0.803 | 13.1% | L=18 −6.4% (−1.6σ) |
| 1.428 | 18.2% | L=24 −10.7% (**−2.3σ**) |
| 3.213 | **5.8%** | L=24 +3.9% (+1.1σ) |

The report's summary sentence — "同一行内 L 变化 6 倍（面积 36 倍），`D̄₂`
变化不超过 5%（z=3.21）到 15%（z=1.43）" ("within a row, D varies by no more
than 5% at z=3.21 up to 15% at z=1.43") — is **MISMATCH** on two counts:

* z=1.43 changes by **18.2%**, not 15% (report's own 5-point row: 0.14769,
  0.14456, 0.13681, 0.12492, 0.12756 → max/min = 1.182).
* The sentence quietly covers only z ≥ 0.803. At z=0.089 the row varies by
  **100%** and at z=0.357 by **60%**. The report never states this. Small-z
  collapse is *not* demonstrated by these data; it is roughly consistent with
  noise (all cells within ~1.4σ of the row mean) but the relative scatter is
  100%, because D itself is tiny there.

The honest statement is: at z=3.213 the collapse is real and tight (5.8%);
at z=0.803 and 1.428 it is good (13–18%); at z ≤ 0.357 the *relative* scatter
is 60–100% and the row is only consistent with collapse because the
absolutes are tiny.

---

## 2. Fixed-z regression slopes (section 2.1, second table)

Weighted least squares of `D = c ln L + b`, weight `1/sem²`, via normal
equations + `lstsq` on the whitened system.

| z | report c | recomputed c | report σ | recomputed σ | report dev | recomputed dev | L-range | verdict |
|---:|---:|---:|---:|---:|---:|---:|---|---|
| 0.089 | +0.0001 | +0.0001 | 0.0002 | 0.0002 | +0.3σ | +0.3σ | 6–24 | **MATCH** |
| 0.357 | −0.0018 | −0.0018 | 0.0014 | 0.0014 | −1.3σ | −1.3σ | 6–30 | **MATCH** |
| 0.803 | −0.0046 | −0.0046 | 0.0029 | 0.0029 | −1.6σ | −1.6σ | 6–30 | **MATCH** |
| 1.428 | −0.0132 | −0.0132 | 0.0040 | 0.0040 | −3.3σ | −3.3σ | 6–30 | **MATCH** |
| 3.213 | −0.0052 | −0.0052 | 0.0030 | 0.0030 | −1.7σ | −1.7σ | 6–36 | **MATCH** (but see caveat) |

All five rows reproduce exactly, including the σ values and the L=36 inclusion
for the z=3.213 row (which is why its σ is the smallest: the old L=36 anchor
has MCSE 0.00284 and sits low, dragging c negative).

**Caveat on row 5.** The z=3.213 slope is the one most sensitive to the L=36
choice, and the report's −0.0052 comes specifically from using the **old**
anchor 0.24214:

| L=36 source | c | σ | deviation |
|---|---:|---:|---:|
| old `beta3_large` 0.24214 | −0.0052 | 0.0030 | −1.7σ |
| new `L36_beta3.0000` 0.25993 | **+0.0024** | 0.0048 | **+0.5σ** |
| omit L=36 | +0.0010 | 0.0054 | +0.2σ |

So the sign of this one slope is not robust to which L=36 value is used. It
does not affect the report's conclusion (`|c|` small in all three variants),
but the specific number −0.0052 is the least stable one in the table.

**Sub-window sensitivity** (not in the report, added here because it bears on
the conclusion). Restricting each row to smaller L:

| z | L≤18 | L≤24 | L≤30 | L≤36 |
|---:|---:|---:|---:|---:|
| 0.089 | +0.0000±0.0002 | +0.0001±0.0002 | −0.0000±0.0002 | −0.0000±0.0002 |
| 0.357 | −0.0033±0.0017 | −0.0031±0.0015 | −0.0018±0.0014 | −0.0018±0.0014 |
| 0.803 | −0.0080±0.0037 | −0.0053±0.0032 | −0.0046±0.0029 | −0.0046±0.0029 |
| 1.428 | −0.0092±0.0053 | −0.0129±0.0045 | −0.0132±0.0040 | −0.0132±0.0040 |
| 3.213 | −0.0023±0.0071 | +0.0022±0.0059 | +0.0010±0.0054 | −0.0052±0.0030 |

The qualitative claim "|c| ≤ 0.013, an order of magnitude below the 0.155
ground-state value" survives every sub-window. Good.

---

## 3. `D/z² = 0.1808` at L=6, z=0.089, and `A = 14.63775/81 = 0.18071`

Arithmetic: 14.63775 / 81 = **0.1807130**. Report says 0.18071. **MATCH.**

`D/z²` at z = 0.089 (all five L with data):

| L | D | z | D/z² | ± | report | verdict |
|---:|---:|---:|---:|---:|---|---|
| 6 | 0.001432 | 0.08900 | **0.18076** | 0.0246 | 0.1808 | **MATCH** |
| 12 | 0.001967 | 0.08899 | 0.24842 | 0.0436 | 0.248 | **MATCH** |
| 18 | 0.001407 | 0.08901 | 0.17759 | 0.0245 | 0.1776 | **MATCH** |
| 24 | 0.001774 | 0.08900 | 0.22396 | 0.0554 | 0.224 | **MATCH** |
| 30 | 0.000983 | 0.08899 | 0.12417 | 0.0631 | not quoted | **MISMATCH of omission** |

L=6 value reproduces to 5 digits and agrees with A to +0.03%. The report's
"与解析值相差 2%" (differs from the analytic value by 2%) is loose: L=6 agrees
to **0.03%**, L=18 to 1.7%. **MISMATCH** on the wording (it understates the
L=6 agreement and overstates the L=18 gap).

Two real problems here:

* **The spread across L is far larger than the quoted ±. ** The individual
  errors on `D/z²` are 0.025–0.063, so L=12's deviation from A (+37%) is only
  +1.55σ and L=30's (−31%) is −0.90σ. Every point is consistent with A at the
  1σ level, which is a *stronger* result than the report claims, but the
  report's narrative ("`L=12,24` 偏高…是更小的 `z` 才进入渐近区") invents a
  systematic trend to explain scatter that is entirely consistent with
  statistical noise. The z=0.089 points do not establish a z→0 plateau; they
  establish nothing beyond "consistent with A, errors too big to say more."
* L=30 (0.12417) is the *worst* point and is not quoted. Given that its
  sem is 0.00050 on D = 0.00098 (i.e. 51% relative error), this is defensible
  as a selection, but it should be disclosed.

**A itself is not recomputable from the files here.** It is imported from the
L=3 exact enumeration recorded in `research_20261029/theory_review.md:404`
(`16.2998418126967 − 1.66209042368616 = 14.6377513890106`) and
`analysis/born_scaling_strong_measurement.json:68`
(`corrected_coefficient: 14.637751389010589`). Both agree with the report's
14.63775. **MATCH** (as a transcription); **UNVERIFIABLE** as an independent
enumeration, since no L=3 enumeration data file is in this directory.

---

## 4. "D_gs grows 0.289 to 0.669 (+131%)" over L=6..36

From `baseline_D.json`: D_gs(6) = 0.2885282764, D_gs(36) = 0.6691893368.

Ratio = 2.31929 → **+131.9%**.

* 0.289 — **MATCH** (0.28853 rounds to 0.289)
* 0.669 — **MATCH** (0.66919)
* "+131%" — **MATCH** (131.9%; rounding 131.9 down to 131 is fine)

Also checked: D_gs(18) = 0.540909, D_gs(24) = 0.595933, D_gs(30) = 0.636754 —
all match the section 2.3 `D_gs` row.

---

## 5. z*(L) values and z*/L

Recomputed by interpolating `h = D/D_gs` linearly in ln z to the 0.5 crossing,
per L, exactly as `crossover_collapse.py` does.

| L | report z* | recomputed z* | report z*/L | recomputed z*/L | verdict |
|---:|---:|---:|---:|---:|---|
| 6 | 1.39 | 1.3883 | 0.231 | 0.2314 | **MATCH** |
| 12 | 2.73 | 2.7314 | 0.228 | 0.2276 | **MATCH** |
| 18 | 3.69 | 3.6874 | 0.205 | 0.2049 | **MATCH** |
| 24 | 4.04 | 4.0401 | 0.168 | 0.1683 | **MATCH** |
| 30 | 5.40 | 5.4029 | 0.180 | 0.1801 | **MATCH** |

All ten numbers reproduce. **However**, the z* values depend on the fitted
`D_gs` interpolation, and — importantly — the new L=36 data give
**z* = 7.523, z*/L = 0.2090**. Adding L=36 changes the picture below.

### "z*/L ≈ 0.19, within 20%, while z*/L² drifts 6×" — section 8 claim

| statistic | 5 points (L=6..30) | 6 points (L=6..36) |
|---|---|---|
| mean z*/L | **0.2025** | 0.2035 |
| median z*/L | 0.2049 | 0.2069 |
| max deviation from mean | 17.3% | 17.3% |
| (max−min)/mean | **31.0%** | **31.0%** |
| z*/L² max/min | **6.64×** | 8.95× |

* "z*/L² drifts 6×" — **MATCH** (6.64× on the report's own 5 points).
* "z*/L ≈ 0.19" — **MISMATCH**. The central value is **0.203**, not 0.19.
  0.19 is neither the mean (0.2025), the median (0.2049), the geometric mean
  (0.2009) nor the LSQ-through-origin amplitude (0.1851). Only the
  *unweighted least-squares slope through the origin* gives 0.185, and that
  is not what "≈ 0.19" describes when mean and median both sit at 0.203.
* "within 20%" — **MATCH** for the max deviation from the mean (17.3%); the
  peak-to-peak spread is 31%, so the phrasing is only true if "within 20%"
  means "each point within 20% of the mean", which is how it should be read.

The 0.185 figure is used consistently downstream (§2.4 fit table
`z* = 0.185 L`, `L*(β) = 0.185 e^{2β}`), so the inconsistency is between the
prose (0.19) and the fit amplitude (0.185), not a computational error in
either — but the two are not the same number and both are presented as if
they describe one thing.

### Residual table (section 2.4): `z* = 0.185 L` → 0.36, `z* = 0.341 L^0.806` → 0.22

| model | absolute rms resid | relative rms | report |
|---|---:|---:|---|
| `z* = 0.185 L` | **0.3601** | 0.138 | 0.36 **MATCH** |
| `z* = 0.341 L^{0.806}` | **0.2173** | 0.062 | 0.22 **MATCH** |

Both reproduce — but only as *absolute* residuals in units of z*. The table
header says "相对残差 rms" (**relative** residual rms). Relative rms is 0.138
and 0.062, i.e. **0.14 and 0.06**, not 0.36 and 0.22. **MISMATCH of label.**
The numbers are absolute; the column header is wrong.

The fit `0.341 L^0.806` is reproducible and is the fit to the 5-point set
(L=6..30). With the new L=36 point included it becomes `0.294 L^0.870` and the
residual ordering *reverses*: linear rms 0.482 vs power rms 0.593, i.e. the
linear form wins. The report's "幂律略好" (power law is slightly better) is
therefore true only of the 5-point set and would not survive L=36.

### `L*(β)` table and the `0.40 e^{1.68β}` formula

The report contains three mutually inconsistent parameterizations for the
same quantity. Recomputation:

| β | report table | 0.185 e^{2β} | (0.342 e^{2β})^{1/1.194} | `0.40 e^{1.68β}` (§3) | `2 e^β` (§5b) |
|---:|---:|---:|---:|---:|---:|
| 1.0 | 1.4 | 1.37 | 2.17 | 2.15 | 5.44 |
| 1.5 | 3.7 | 3.72 | 5.03 | 4.97 | 8.96 |
| 2.0 | 10 | 10.10 | 11.62 | 11.52 | 14.78 |
| 2.5 | 27 | 27.46 | 26.85 | 26.67 | 24.36 |
| 3.0 | 75 | **74.63** | 62.06 | **61.79** | 40.17 |

* The table's "`L*(β)`" column is exactly `0.185 e^{2β}`. **MATCH.**
* The table's "幂律形式给出的对应值" (power-law form values) column —
  report: 2, 5, 11, 27, 62 — is **MATCH** against `(0.342 e^{2β})^{1/1.194}`.
* But §3 says "`L*(β) ≈ 0.40 e^{1.68β}`：β=3 需要 `L≈60`，β=2.5 需要 `L≈27`"
  — those two check out (61.79, 26.67), while **β=1.5 gives 4.97 not 3.7** and
  **β=1 gives 2.15 not 1.4**. So §3's formula and §2.4's table disagree at
  small β, where the report uses both without comment.
* §5(b) says "`L*(β) = 2e^β`" — that is **MISMATCH by a factor of ~2**: at
  β=3 it gives 40.2 versus 74.6/61.8, and at β=2 it gives 14.8 versus
  10.1/11.5. It agrees with nothing in the table.

**Three different `L*(β)` laws for one quantity.** Only the table is internally
consistent; `0.40 e^{1.68β}` is close to it and `2e^β` is not.

---

## 6. `c₂ = 0.1554 ± 0.0002` from the log+1/L fit at L=24–60, and "rms 2e-6"

Model `D_gs = c₂ ln L + b + d/L`, 13 points (L = 24, 27, ..., 60), ordinary
least squares via `lstsq`, σ from `s²·(XᵀX)⁻¹` with `s² = RSS/(n−3)`:

```
c₂ = 0.155430 ± 0.000021
b  = 0.1327 ± 0.0001
d  = −0.7371 ± 0.0008
rms(residual) = 2.43e-06   max|resid| = 5.72e-06
```

* `c₂ = 0.1554` — **MATCH** (0.155430)
* `± 0.0002` — **MATCH as rounding, but it is 10× too conservative.** The
  formal σ is **0.000021**, so "± 0.0002" understates the precision by an
  order of magnitude. If the intent was "± 2e-5" then the digit is wrong; if
  the intent was a conservative systematics-accounted error, that should be
  said, since the report explicitly claims it "把它的精度提高了一个量级"
  (raised its precision by an order of magnitude).
* `rms 2e-6` — **MATCH** (2.43e-06). Verified as the plain rms of residuals,
  not the σ-based chi-square.

Window sensitivity:

| window | c₂ | σ | rms | max abs resid |
|---|---:|---:|---:|---:|
| L=6–60 | 0.155762 | 0.000195 | 1.63e-04 | 5.41e-04 |
| L=21–60 | 0.155319 | 0.000040 | 6.21e-06 | 1.65e-05 |
| **L=24–60** | **0.155430** | **0.000021** | **2.43e-06** | 5.72e-06 |
| L=30–60 | 0.155437 | 0.000034 | 2.01e-06 | 4.90e-06 |

c₂ is stable at 0.1553–0.1558 across all windows, so the central value is
robust. Note the report also says (§7.3) "`1/L` 修正很大（`b ≈ -0.74`）" —
that is a **label swap**: −0.74 is the coefficient of the `1/L` term
(`d`), while `b` is +0.1327. **MISMATCH.**

For contrast, a plain `c ln L + b` fit (no 1/L term) over the same window
gives c = 0.175032 ± 0.000761 with rms 7.1e-04 — 300× worse. So the 1/L term
is doing the work, and the quoted 0.1554 is model-dependent, as §7.3 concedes.

---

## 7. Weak-measurement excess table

Reported: L=24, z=212 → `+0.0207 ± 0.0080`, `+3.5%`; L=30, z=331 →
`+0.0064 ± 0.0117`, `+1.0%`.

| L | z | D | sem | D_gs | D − D_gs | report | rel | report | dev | verdict |
|---:|---:|---:|---:|---:|---:|---|---:|---|---:|---|
| 24 | 211.8986 | 0.616607 | 0.007980 | 0.595933 | **+0.020675** | +0.0207±0.0080 | **+3.47%** | +3.5% | +2.59σ | **MATCH** |
| 30 | 331.0915 | 0.643101 | 0.011695 | 0.636754 | **+0.006347** | +0.0064±0.0117 | **+1.00%** | +1.0% | +0.54σ | **MATCH** |

Both rows reproduce exactly (the ± is just the sem, correctly propagated
through the subtraction since D_gs is exact).

**But the interpretation in the surrounding text is a MISMATCH.** The report
says "两个点在 1σ 内与'常数偏移'相容" (both points are consistent with a
constant offset within 1σ). Check: the two offsets are +0.0207 ± 0.0080 and
+0.0063 ± 0.0117; their difference is +0.0143 ± 0.0142 = +1.01σ. So yes,
mutually consistent — but the report then also says "随 z 增大而缩小" (shrinks
as z grows), which is not established: the difference is 1.0σ from zero, and
a constant offset is equally consistent. The report hedges this correctly in
the next clause ("目前只能说是 1–3% 量级的效应，不能定论"), so this is a
soft inconsistency, not a numerical error.

Also note the corresponding L=18 point is *not* in the table: L=18, z=119,
D = 0.554938 ± 0.005010 vs D_gs = 0.540909 → +0.014029 ± 0.005010 = **+2.80σ,
+2.59%**. That is a third point, more significant than the L=30 one, and it
sits between the two quoted values without breaking the trend. Its omission
is not an error but weakens the "shrinks with z" reading (18 is *lower* than
24 at lower z).

Same table, other cells, checked for context:

| L | β | D | z | D_gs | rel |
|---:|---:|---:|---:|---:|---:|
| 18 | 0.5 | 0.5549 | 119.2 | 0.5409 | +2.59% |
| 18 | 1.0 | 0.5593 | 43.8 | 0.5409 | +3.40% |
| 18 | 1.5 | 0.4950 | 16.1 | 0.5409 | −8.49% |
| 18 | 2.0 | 0.3454 | 5.9 | 0.5409 | −36.15% |
| 24 | 0.5 | 0.6166 | 211.9 | 0.5959 | +3.47% |
| 24 | 1.0 | 0.6011 | 78.0 | 0.5959 | +0.86% |
| 24 | 1.5 | 0.5648 | 28.7 | 0.5959 | −5.23% |
| 24 | 2.0 | 0.4537 | 10.5 | 0.5959 | −23.87% |
| 30 | 0.5 | 0.6431 | 331.1 | 0.6368 | +1.00% |
| 30 | 1.0 | 0.6549 | 121.8 | 0.6368 | +2.85% |
| 30 | 1.5 | 0.6007 | 44.8 | 0.6368 | −5.67% |
| 30 | 2.0 | 0.4726 | 16.5 | 0.6368 | −25.78% |

Section 2.3's table (β rows, L=18/24/30) has **all twelve cells MATCH** the
raw values, including the z annotations (z=119, 212, 44, 78, 16, 29, 5.9,
10.5 — all verified). And "β ≤ 1 时已在 3% 以内" (within 3% for β≤1) is
**MATCH** for β=0.5, 1.0 at L=18/24; the L=30 β=1.0 point is +2.85%, still
under 3%. "`β=2` 时 `L=24` 仍低 24%" — recomputed −23.87%. **MATCH.**

---

## 8. "z*/L ~ 0.19 within 20% while z*/L² drifts 6×"

* "z*/L ~ 0.19" — **MISMATCH**, the central value is 0.203 (mean 0.2025,
  median 0.2049). See §5.
* "within 20%" — **MATCH** if read as "each z*/L within 20% of the mean"
  (max deviation 17.3%).
* "z*/L² drifts 6×" — **MATCH** (6.64× over L=6..30; 8.95× if L=36 is added).

The *structural* claim (linear-in-L is a better description than
quadratic-in-L) is supported; the specific constant 0.19 is not the number
the data give.

---

## 9. Additional numbers checked beyond the eight required

These appear in the report and were recomputed because they carry argumentative
weight.

| claim | report | recomputed | verdict |
|---|---|---|---|
| §2.3 D_gs row: 0.5409 / 0.5959 / 0.6368 | as shown | 0.540909 / 0.595933 / 0.636754 | **MATCH** |
| §4 j-decomposition L=18, β=3: P(j=0)=0.559, ⟨D⟩=0.046, 36% | as shown | 0.574 / 0.0420 / 35.0% | **MISMATCH (mild)** |
| §4 P(j=1)=0.377, ⟨D⟩=0.099, 52% | as shown | 0.361 / 0.1030 / 54.0% | **MISMATCH (mild)** |
| §4 P(j≥2)=0.064, ⟨D⟩=0.131, 12% | as shown | 0.065 / 0.1170 / 11.0% | **MISMATCH (mild)** |
| §4 "j≥1 层贡献 64%" | 64% | 65.0% | **MATCH** |
| §4 D_annealed L=10,20,40,60: 8.2 / 43.6 / 241.6 / 553.0 | as shown | **16.63 / 66.53 / 252.27 / 554.44** | **MISMATCH ×3** |
| §4 "`D = 2w(L−2w)ln2`" (the stated formula) | formula | 16.64 / 66.54 / 252.31 / 554.52 | **MATCH to formula** |
| §4 "L=60 时 553 vs Born 的 O(1)" | 553 | 554.44 (solver), 554.52 (formula) | **MATCH** |
| §4 "κ² 下界在 β=3 时只有 4e-6" | 4e-6 | sech⁴(3)·c_V0 = 6.42e-6 (single-layer finite-a: 4.23e-6 at L=18) | **MATCH** (finite-a value) |
| §5(c′) skewness 1.19, excess kurtosis 3.55, mean/median 2.17 at L=18, β=3 | as shown | 1.191 / 3.557 / 2.170 (old `iid_targeted`, n=2048) | **MATCH** |
| §5(c′) log-log slope −0.50 ± 0.00 for L=6,12,18 × β=1,3 | −0.50 | −0.4926 … −0.5031 | **MATCH** |
| §8(3) "β=3 时 L=48,60：预测 z = 3.2, 5.0" | 3.2, 5.0 | z(48)=**5.711**, z(60)=**8.924** | **MISMATCH** |
| §3(3) κ→0 Gaussian gives volume law "L=60 时 553 vs Born 的 O(1)" | 553 | 554.44 | **MATCH** |
| `baseline_D.json` vs Julia β=0 production | "1e-8" | max deviation 1.9e-14 (L=6,9,12) | **MATCH (better than claimed)** |

### The §4 annealed table is wrong for three of four entries

This is the single largest numerical error found. The report states the
formula `D = 2w(L−2w)ln2` and then tabulates values that do not follow it:

| L | 2w(L−2w)ln2 | solver | report table |
|---:|---:|---:|---:|
| 10 | **16.64** | **16.63** | **8.2** ✗ |
| 20 | **66.54** | **66.53** | **43.6** ✗ |
| 40 | **252.31** | **252.27** | **241.6** ✗ |
| 60 | **554.52** | **554.44** | 553.0 ✓ |

The report's own formula and the report's own solver agree with each other
(both verified here, `annealed_corner(L, sech²β)` after re-validating the
solver against the Julia β=0 baseline to 6e-15). The table's first three
entries agree with neither. Values 8.2/43.6/241.6 look like they may come
from an older `w(L−2w)ln2` convention (which gives 8.32/33.27/126.15 — also
not matching) or from a different w. They cannot be reproduced from any
combination of w, L, and the stated formula I tried.

The *conclusion* the table supports ("volume law, O(L²), versus O(1) Born")
is correct and is unaffected — but three of the four numbers in the table
are not what the stated model produces, and a reader checking the table
against the formula would find it wrong.

### The §4 j-decomposition disagrees with both data sources, mildly

The report's table is credited to `mechanism_figures.py`, which reads
`research_20261029/iid_targeted/`, not `data_matched_z/`. Recomputing from
that source (n=2048): P(j=0)=0.559 → I get **0.559** on the *old* source but
**0.574** on the new `data_matched_z` source. The report's ⟨D⟩ values
(0.046/0.099/0.131) match the **new** source (0.0420/0.1030/0.1170) more
closely in the j=1 and j≥2 rows but the probabilities match the **old**
source. So the table appears to be a blend of two datasets. Mismatches are
all ≤ 0.014 in probability and ≤ 0.014 in ⟨D⟩ — small, but the row is
presented as a single consistent decomposition and is not.

### The §8(3) prediction is wrong

"β=3 时 L=48,60：预测 z = 3.2, 5.0" — with z = L²e^{−2β} = L²e^{−6}:
z(48) = 5.711, z(60) = 8.924. To get z = 3.2 at β=3 you need L=36 (3.212);
z = 5.0 needs L≈44.8. The sentence appears to carry over the L=36 value for
L=48 and then miscomputes. The qualitative point ("both are in the
0.25–0.35 window") probably survives, but the two quoted z values are
**MISMATCH** and a reader would compute the stated predictions differently.

---

## 10. A structural problem not in the checklist: `j = 0` does not give `D = 0`

§1 states: "`j = 0` 的输出把测量后态变成**严格乘积态**（`D = 0`）" — the j=0
output makes the post-measurement state a strictly product state, D = 0. §4's
decomposition then relies on this: the j=0 row is given ⟨D⟩ = 0.046 and a 36%
contribution, which is itself inconsistent with D = 0, and §5(c) builds on it
("j=0 与 j≥1 的干涉").

Tested directly on the raw CSVs: conditioning on j=0 gives

| L | β | n(j=0) | mean(D\|j=0) | sd | max\|D\| |
|---:|---:|---:|---:|---:|---:|
| 6 | 3.0013 | 2019 | **+0.00124** | 0.00577 | 0.1038 |
| 12 | 3.6945 | 2033 | **+0.00161** | 0.01176 | 0.3223 |
| 18 | 4.0999 | 2036 | **+0.00127** | 0.00665 | 0.1497 |
| 24 | 4.3876 | 1013 | **+0.00124** | 0.00859 | 0.1440 |
| 18 | 3.0001 | 1175 | **+0.04204** | 0.08887 | 0.5844 |
| 6 | 1.9015 | 1175 | **+0.05093** | 0.09195 | 0.6283 |

Not a single one of the ~10,000 j=0 samples has |D| < 1e-12. D is not zero at
j=0; it is small in the mean and broadly distributed. In the deep-small-z
regime (L=6, β=3) the j=0 conditional mean is 0.00124 against an overall mean
of 0.00143 — i.e. **j=0 carries 87% of the signal there**, the opposite of
the report's story that the entropy is carried by rare j ≥ 1 outcomes.

Reading `run_matched_z.jl` / `StratifiedBorn.jl`, `j = |E ∩ F|` is the overlap
between the *auxiliary* set E and the initial Slater determinant F, and the
module docstring is explicit: "The auxiliary occupation F is never used as
the physical postmeasurement state." So j=0 means the auxiliary E is disjoint
from F — it does not make the physical state a product state. The §1 claim
mis-identifies E as the post-measurement support.

This is not one of the eight numbered items, but it is load-bearing for §1,
§4 and §5(c), and it is contradicted by the raw data at every z. It should be
corrected before the report is circulated.

---

## Summary table

| # | Claim | Verdict |
|---|---|---|
| 1 | Table 2.1, 22/30 cells | **MATCH** |
| 1a | L=30 z=0.089 marked "—" | **MISMATCH** (data exist: 0.00098 ± 0.00050) |
| 1b | L=30 z=0.357 marked "—" | **MISMATCH** (data exist: 0.02788 ± 0.00398) |
| 1c | L=36 z ≤ 0.803 marked "—" | **MATCH** (genuinely absent) |
| 1d | L=36 z=3.213 = 0.24214 | **MISMATCH** (new data: 0.25993 ± 0.01164, +7.3%, 1.5σ) |
| 1e | "row varies ≤5–15%" | **MISMATCH** (z=1.43 is 18.2%; z=0.089 is 100%; z=0.357 is 60%) |
| 2 | Fixed-z slopes c, σ (5 rows) | **MATCH** (all five exactly) |
| 3 | D/z² = 0.1808 at L=6; A = 0.18071 | **MATCH** |
| 3a | "2% from analytic" | **MISMATCH** (L=6 is 0.03%; L=18 is 1.7%) |
| 4 | D_gs 0.289 → 0.669, +131% | **MATCH** |
| 5 | z* = 1.39/2.73/3.69/4.04/5.40 | **MATCH** |
| 5a | z*/L = 0.231/0.228/0.205/0.168/0.180 | **MATCH** |
| 6 | c₂ = 0.1554 ± 0.0002, rms 2e-6 | **MATCH** (0.155430; rms 2.43e-6; σ is 2.1e-5, 10× smaller) |
| 7 | +0.0207±0.0080 and +0.0064±0.0117 | **MATCH** |
| 8 | z*/L ~ 0.19, within 20%; z*/L² drifts 6× | **0.19 MISMATCH** (0.203); within-20% **MATCH**; 6× **MATCH** |
| 9 | §4 annealed 8.2/43.6/241.6/553.0 | **MISMATCH** (16.63/66.53/252.27/554.44) |
| 10 | §4 residual-table header "相对" (relative) | **MISMATCH** (values are absolute) |
| 11 | §3 `L*(β) ≈ 0.40 e^{1.68β}` vs §5(b) `2e^β` vs table `0.185 e^{2β}` | **MISMATCH** (three laws; `2e^β` off by 2×) |
| 12 | §7.3 "`1/L` 修正很大（`b ≈ −0.74`）" | **MISMATCH** (−0.74 is the 1/L coefficient; b = +0.133) |
| 13 | §8(3) "L=48,60 → z = 3.2, 5.0" | **MISMATCH** (5.71, 8.92) |
| 14 | §4 j-decomposition probabilities | **MISMATCH (mild)** (blends two datasets) |
| 15 | §5(c′) skew 1.19, kurt 3.55, mean/med 2.17 | **MATCH** |
| 16 | §5(c′) log-log slope −0.50 | **MATCH** |
| 17 | §1 "j=0 ⇒ D=0" | **MISMATCH vs data** (no j=0 sample has D=0) |
| 18 | baseline_D.json vs Julia β=0 "1e-8" | **MATCH** (1.9e-14) |
