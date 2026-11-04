# Adversarial verification: is D non-self-averaging, and does z = L²e^(−2β) organise it?

**Verdict: REFUTED.** Both halves of the claim fail, and they fail for reasons that are
diagnostic rather than marginal. The "heavy tail" is real but is a small-`z`
(strong-measurement, small-`L`) finite-size effect that *disappears* as `L` grows, which is
the opposite of the claimed behaviour. And `z = L²e^(−2β)` is not a collapse variable at
any exponent — I scanned 99 `(α, γ)` pairs and not one reaches χ²/dof < 3.

Reproduce: `D:/app/conda/python.exe verify_tails.py`. Numbers in
`verify_tails_numbers.json`, tables in `tables/`, figures in `figures/`, stdout in `run.log`.

Data used, exactly as specified:
- `research_20261029/iid_targeted/L*_beta*_batch*.csv`, 6 `(L, β)` points × 4 batches × 512
  draws = 12,288 **iid** Born draws. Verified `|D − ((2S_A+S_C)/3 − S_B)| = 0` exactly.
- `analysis/born_scaling/production/means.csv`, 67 `(L, β)` points, `L ∈ {6…24}`.

---

## 1. The tail is real — but it shrinks with L, killing the claim's mechanism

Per-trajectory D at β=3 is genuinely violently non-Gaussian at L=6: skewness 27.3, excess
kurtosis 914, and the largest single draw is 278× the mean. So the claim is not
manufacturing a tail out of nothing. But the critical question is what happens as `L` grows
at fixed β, and there the trend is monotone in the wrong direction.

`tables/tail_share_vs_L.csv`, `figures/fig1_tail_share_vs_L.png`:

| β | quantity | L=6 | L=12 | L=18 | log-log slope vs L | last/first |
|---|---|---|---|---|---|---|
| 1 | top-1% share of mean | 0.0328 | 0.0292 | 0.0295 | **−0.10** | 0.90 |
| 1 | top-5% share of mean | 0.1353 | 0.1211 | 0.1237 | **−0.09** | — |
| 3 | top-1% share of mean | 0.4381 | 0.2357 | 0.0881 | **−1.40** | 0.20 |
| 3 | top-5% share of mean | 0.6976 | 0.6301 | 0.3228 | **−0.64** | — |

At β=3 the top 1% of trajectories supply 44% of the Born mean at L=6 but only 8.8% at L=18.
That is a factor of 5 collapse, not growth. The claim's stated kill condition
("if the tail fraction does not grow") is triggered in the strongest possible way: it
decreases, with slope −1.40.

Mean/median tells the same story (`fig1`, right panel): at β=3 it goes 4.45 → 4.10 → 2.17
as L goes 6 → 12 → 18. At β=1 it is 1.02 / 1.00 / 0.99 — the mean is already perfectly
representative of a typical trajectory there, and *always was*. So "the mean is NOT
representative of a typical trajectory" is a statement about β=3, L=6 only, not a property
of the ensemble.

Supporting shape data, `tables/d_distribution_shape.csv`:

| L | β | mean | median | sd | skew | ex-kurt | max/mean | Hill α |
|---|---|---|---|---|---|---|---|---|
| 6 | 3 | 0.0016 | 0.0004 | 0.0125 | 27.30 | 914.0 | 278.2 | 1.13 |
| 12 | 3 | 0.0199 | 0.0048 | 0.0673 | 3.70 | 23.4 | 29.2 | 1.30 |
| 18 | 3 | 0.0715 | 0.0329 | 0.1361 | 1.19 | 3.5 | 11.7 | 3.07 |
| 6 | 1 | 0.2940 | 0.2889 | 0.2397 | 0.06 | −0.0 | 3.9 | 6.62 |
| 18 | 1 | 0.5604 | 0.5637 | 0.4074 | −0.09 | 0.0 | 3.5 | 6.74 |

Excess kurtosis goes 914 → 23 → 3.5 at β=3. At β=1 it is **zero to two decimals** at every
size — a Gaussian. A distribution whose excess kurtosis is 0.0 is the textbook definition of
self-averaging.

### Why "heavy tailed" is the wrong label anyway

`tables/support_bound.csv`. `S₂` on a region of `a` sites is bounded by `ln a`, so D is
bounded: `|D| ≤ (2/3)ln(a_A) + ln(a_B)` at worst. Empirically `max|D| / bound` is 0.19–0.21
at β=1 and 0.068–0.082 at β=3 — never above 0.21 of the available range. The support is
bounded, so the tail cannot be a power law in the strict sense; the Hill index is only
meaningful as a local slope. `fig2` shows the survival function `P(D>x)` is visibly *curved*
on log-log axes and close to straight on semi-log axes — a stretched/truncated exponential,
not a Pareto tail. The "power-law-like" Hill index of ~1.1 at L=6, β=3 is an artefact of
fitting a local slope to a curved, truncated distribution over one decade.

---

## 2. The decisive self-averaging test: the mean obeys the CLT exactly

The sharpest refutation of "strongly non-self-averaging" does not depend on any tail
threshold. If D were strongly non-self-averaging, the dispersion of an `n`-sample mean would
fall slower than `n^(−1/2)`, or plateau. I measured the dispersion of the `n`-sample mean by
subsampling, 4000 replicates per `n`, `n` from 8 to 512.

`tables/mean_dispersion_vs_n.csv`:

| L | β | log-log slope of sd(mean_n) vs n | CLT value | sd(mean₂₀₄₈) / CLT prediction |
|---|---|---|---|---|
| 6 | 1 | −0.503 | −0.5 | 1.01 |
| 6 | 3 | −0.495 | −0.5 | 1.01 |
| 12 | 1 | −0.499 | −0.5 | 1.00 |
| 12 | 3 | −0.497 | −0.5 | 1.00 |
| 18 | 1 | −0.501 | −0.5 | 1.00 |
| 18 | 3 | −0.502 | −0.5 | 1.00 |

Every slope is −0.5 to within ±0.005, and the dispersion matches the CLT prediction to 1%.
This is the cleanest possible self-averaging signature. Note it holds at β=3, where the
distribution has excess kurtosis 914 — because the tail is bounded, the CLT still wins at
n=2048. The "rare high-D outcomes dominate" picture is not strong enough to break the CLT at
the n actually used.

An independent check on the production MCMC data, `tables/chain_rel_scatter_slopes.csv`,
`PART 7`: chain-to-chain relative scatter of D against L, 12 β values, 4 chains each.
**0 of 12** have a positive slope; the median log-log slope is −0.86. So the production
ensemble self-averages even faster than `n^(−1/2)` per chain — consistent with 2048 draws
being far past the point where the tail matters.

---

## 3. The z-story fails outright

### 3a. Equal z, different L ⇒ incompatible D

`tables/z_collapse_pairs.csv`, `tables/z_collapse_nodes.csv`, `fig3`, `fig6` (left).

I interpolated D in `ln z` within each L curve (so no binning error), then compared all L
pairs at matched z: **802 pairs**.

| statistic | value |
|---|---|
| fraction of pairs agreeing within 3σ | **0.444** |
| median \|pull\| | 3.49 |
| RMS \|pull\| | 17.6 |
| max \|pull\| | 99.3 |
| fraction within 3σ, restricted to z > 10 (asymptotic arm) | **0.112** |

At z ≈ 43.8 the five sizes present give D = 0.479 (L=12), 0.517 (L=15), 0.552 (L=18), 0.565
(L=21), 0.578 (L=24) — spread 0.099 on an MCSE of ~0.002, i.e. pulls of 8–59σ. At z ≈ 4.87:
0.294, 0.316, 0.326, 0.324, 0.322, 0.319, 0.317 for L = 6…24. If z collapsed, all these would
coincide. They do not, and they do not get closer at large z (agreement *worsens* to 11%).

### 3b. No exponent works — so it is not a wrong-exponent problem

`tables/collapse_exponent_scan.csv`, `PART 5`. Additive form `D = c ln L + A(z_α)`, scan
`α` from 0.5 to 3.5, with `c = 0.2205` from the β=0 series. Residual is reported as a
multiple of the typical MCSE; a genuine collapse needs ≤ ~2.

| α | 0.5 | 1.0 | 1.5 | 1.75 | **2.0** | 2.25 | 2.5 | 3.0 | 3.5 |
|---|---|---|---|---|---|---|---|---|---|
| excess scatter / MCSE | 6.9 | 22.3 | 27.9 | 33.7 | **42.4** | 39.8 | 35.5 | 30.5 | 49.5 |

Not a single `α` gets below 6.9× MCSE. The claimed α=2 sits near a broad maximum (42×).
Scanning the exponent does not rescue the idea.

`tables/mult_scaling_scan.csv`, `PART 10`. The same test for the multiplicative form
`D = D₀(L)·f(L^α e^(−γβ))`, 99 `(α, γ)` pairs:

- claimed `(α, γ) = (2, 2)`: χ²/dof = **121.5**
- best of all 99 pairs, `(α, γ) = (1.75, 2.5)`: χ²/dof = **23.6**
- pairs reaching χ²/dof < 3: **0 of 85 usable**

`fig6` shows this directly: the left panel (claimed z) has the seven L-curves cleanly
separated and ordered by L; the right panel (best scanned pair) overlaps them a bit better
at small `t` but χ²/dof = 24, so it is still not a collapse. One scaling variable, of any
power-law form, does not describe this surface.

### 3c. The mechanism is contradicted, not just unverified

`PART 6`, `tables/slope_overshoot.csv`, `fig4`. If the curvature really were "crossover in z
plus asymptotic corner coefficient", the local slope dD/dlnL would have to lie *between* the
two asymptotic slopes. It does not — it exceeds both:

- asymptote at β=0 (L=21–24): 0.1886
- asymptote at β=3 (L=21–24): 0.2381
- β=2.0 tail slope: **0.3238 ± 0.0233** (5.8σ above the β=0 asymptote)
- β=1.5: 0.2451 ± 0.0279 (2.0σ), β=2.5: 0.2925 ± 0.0284 (3.7σ), β=3.0: 0.2381 ± 0.0227 (2.2σ)

4 of 6 β values overshoot the β=0 asymptote by >2σ. An intermediate β producing a *larger*
slope than either limit is a genuine non-monotonicity, which no `c ln L + A(z)` crossover can
produce — that ansatz is monotone in the crossover variable. `fig4` makes it visible: at
fixed z, the slope takes different values depending on β, and it peaks near z ≈ 2–4 at β=2,
then falls.

---

## 4. What the data actually support (constructive part)

### 4a. The tail is controlled by z — in the decreasing direction

`tables/tail_shape_vs_z.csv`, `PART 9`. The tail statistics do organise against z, but as a
*decreasing* function:

| z | L | β | top-1% share | top-5% share | ex-kurtosis | mean/median |
|---|---|---|---|---|---|---|
| 0.089 | 6 | 3 | 0.438 | 0.698 | 914.0 | 4.45 |
| 0.357 | 12 | 3 | 0.236 | 0.630 | 23.4 | 4.10 |
| 0.803 | 18 | 3 | 0.088 | 0.323 | 3.5 | 2.17 |
| 4.87 | 6 | 1 | 0.033 | 0.135 | −0.0 | 1.02 |
| 43.8 | 18 | 1 | 0.030 | 0.124 | 0.0 | 0.99 |

Monotone decreasing in z, confirmed. So z *is* a meaningful control variable — for the
**shape** of the distribution, and with the sign opposite to the claim. Small z = strong
measurement = rare trajectories carry a lot of the mean. Since z grows with L at fixed β,
the strong-measurement regime is a *small-L* regime.

### 4b. What drives a high-D trajectory

`tables/tail_mechanism.csv`. A high-D trajectory is one where the B region happened to be
measured weakly, so S_B stayed high and the subtraction did not remove much:

- corr(D, S_A − S_B) = **+0.84 to +0.96** at every (L, β) — D is essentially an A-vs-B contrast
- corr(D, S_A+S_B+S_C) changes sign: −0.15 at (18, 1) to +0.22 at (18, 3), +0.55 at (6, 3)
- at β=3 top-1% trajectories have S_A ≫ ⟨S_A⟩ (0.910 vs 0.361 at L=18) while S_B is *below*
  its mean (0.230 vs 0.288)

So the tail is not a corner effect or a topological defect; it is the weak-measurement tail
of the measurement-strength distribution. That is a mundane origin, and it explains why the
tail dies with L: at fixed β, larger L means more measured sites and a more sharply
concentrated measurement strength, suppressing the weak-measurement tail. This is the
physics that makes `z = L²e^(−2β)` the right variable for the *shape* and the wrong one for
the *mean*.

### 4c. A better description: β-dependent corner coefficient, non-monotone in β

`tables/alt_model_c_of_beta.csv`, `PART 8`. Fitting `D = c(β) ln L + b(β)`:

| β | c(β) (all L) | χ²/dof | c(β) (L ≥ 12) | χ²/dof |
|---|---|---|---|---|
| 0 | 0.2205 | — | 0.2002 | — |
| 0.5 | 0.2030 | 8.2 | 0.1876 | 1.7 |
| 1.0 | 0.2248 | 12.7 | 0.1952 | 0.2 |
| **1.5** | **0.2766** | 7.8 | 0.2555 | 1.6 |
| 2.0 | 0.2764 | 26.5 | **0.3045** | 0.6 |
| 2.5 | 0.1634 | 317 | 0.2746 | 0.8 |
| 3.0 | 0.0435 | 566 | 0.1456 | 45.5 |

Two things to read off. First, restricted to L ≥ 12 the single-log form works well
(χ²/dof ≤ 1.7) for β = 0.5 … 2.5 — the whole-L fits have large χ² only because of the small-L
curvature. Second, c(β) rises from 0.20 (β=0) to a peak of ~0.30 near β=2, then falls. That
non-monotone c(β) is exactly the overshoot in §3c seen from the other side.

**This is my alternative proposal, and it is testable:** the moving curvature in the
`D` vs `ln L` fits is not a z-crossover mixing two asymptotes. It is what a *β-dependent*
corner coefficient looks like when read through a finite window `L ∈ [6, 24]`. The correct
statement is

```
D(L, β) = c(β) ln L + b(β),   c(β) non-monotone, peaked near β ≈ 2
```

with no claim about a universal z scaling of the mean. Note I am *not* claiming to have
proven the L → ∞ limit; at β=3 the L ≥ 12 fit still has χ²/dof = 45, so β=3 remains
unresolved and needs larger L, exactly as the earlier work concluded.

---

## 5. Read the production error bars with care

`PART 4D`, `tables/mcse_vs_chain_scatter.csv`. Comparing the reported `D_MCSE` to the actual
chain-to-chain scatter (4 chains): median ratio 0.79 overall, 0.71 for β ≤ 1.5, 0.99 for
β ≥ 2.5. Individual points are worse — L=6, β=2.25 has chainSE/reported = 2.39 and L=18,
β=1.5 has 1.60, L=21, β=3.0 has 1.52. So the quoted MCSE underestimates the true uncertainty
for several low-β points by up to 2.4×. This does not change any conclusion above (the z
failures are 40–100σ), but any future fit that leans on these error bars is at risk of
being over-confident at the ~2× level.

---

## 6. Where the claim is right

Being fair to it, three parts survive:

1. **At β=3, L=6 the ensemble really is wildly non-Gaussian** (ex-kurt 914, max draw 278×
   mean, top-1% carrying 44% of the mean). The observation that motivated the claim is real.
2. **z is a genuine control variable for the tail shape** (§4a) — just with the opposite sign.
3. **Mixing scales in a fit is a real hazard.** The `D` vs `ln L` fits *do* change when the
   window moves; the alternative in §4c is a restatement of that hazard, not a denial.

What fails is the inference from (1) to "the mean is unreliable and the fits are therefore
mixing z-crossover with asymptote." The mean *is* reliable: CLT exact to 1% at every point,
chain scatter falling with L at 12/12 β values, and the small-L tail gone by L=18 at β=3.

---

## 7. Verdict

**REFUTED**, on the claim's own stated kill conditions:

- *K1, tail fraction grows with L at fixed β*: fails. Slope −1.40 (β=3), −0.10 (β=1). Share
  drops 44% → 8.8% between L=6 and L=18.
- *K2, self-averaging*: fails the claim. sd of the n-sample mean scales as n^(−0.500 ± 0.005)
  and matches the CLT normalisation to 1%.
- *K3, z is a collapse variable*: fails. 44% of 802 matched-z pairs agree within 3σ (11% at
  z > 10); 0 of 85 scanned `(α, γ)` pairs reach χ²/dof < 3; α=2 gives 42× MCSE and
  (2,2) gives χ²/dof = 122.
- *K4, local slope organises around z*: fails, and worse than fails — the slope overshoots
  both asymptotes (0.324 at β=2 vs 0.189 at β=0 and 0.238 at β=3), which the crossover
  ansatz cannot produce at all.

The only defensible version of the original intuition is much weaker and points the other
way: *at small z (strong measurement, small L) the Born ensemble has a heavy, bounded tail
whose rare events matter; this tail is suppressed as L grows, and the ensemble is
self-averaging to CLT accuracy at the sample sizes used.* The curvature in `D` vs `ln L` is
better described by a non-monotone `c(β)` than by a z-crossover.

### Falsifier I would want next

The alternative in §4c predicts that at fixed β, `D` should be linear in `ln L` with the
*same* slope c(β) at L=36 and L=48 as at L=18–24. The refuted claim predicts continued
bending. One production run at L = 36, β = 2.0 (z = 0.66) and L = 36, β = 3.0 (z = 0.089)
settles it: the claim needs D(36, 2.0) to be far off the linear extrapolation from
L = 18–24; the alternative needs it on the line.

---

## Files

- `verify_tails.py` — all tests, deterministic (`RNG = 20261030`), no scipy dependency
- `verify_tails_numbers.json` — machine-readable summary
- `run.log` — full stdout
- `tables/` — 19 CSV tables listed inline above
- `figures/fig1_tail_share_vs_L.png` — tail share and mean/median vs L (the K1 kill)
- `figures/fig2_D_survival.png` — survival function, log-log vs semi-log (bounded tail)
- `figures/fig3_D_vs_z.png` — D vs z, all 7 sizes (the visual non-collapse)
- `figures/fig4_local_slope_vs_z.png` — local slope vs z, the overshoot
- `figures/fig5_quantiles_vs_L.png` — quantile fan vs L; the mean tracks the median
- `figures/fig6_mult_scaling_scan.png` — claimed z vs best scanned pair, both fail
