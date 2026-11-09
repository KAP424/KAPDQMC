# Adversarial verification: does `mean|G_E,ij|² ~ r^-α` hold, and does `α < 2` break the log corner?

**Verdict: REFUTED.** All three load-bearing pieces of the claim fail, and two of them fail
in a way that inverts the claim's conclusion.

1. **The exponent is not stable under binning.** α at β=1 spans 2.77–3.54 across six
   reasonable binnings at L=18, and β=3 spans 1.07–1.32 (L=18) and 0.65–0.88 (L=12). The
   claim quotes 3.4–3.6 and 1.0–1.5 with no error bar; the binning systematic is as large
   as the quoted band.
2. **The quoted α is not a converged number at all — it is an estimator artifact.**
   Calibrated on the *ground state*, whose continuum decay is exactly `|G|² ~ r^-4`, the
   identical pipeline returns α = 1.96–3.77 depending only on the binning. The residual
   scatter of a single power law about the ground-state profile is 1.9 dex per shell. Any
   α read off these 6 ≤ L ≤ 18 matrices carries an O(1) systematic that the claim does not
   acknowledge.
3. **The D_V claim is wrong twice.** The stated rule "`D_V` is ln L for α = 4 and saturates
   to a constant for α > 2" is false. Direct computation on the actual torus shows
   `D_V ~ L^max(0, 4−α)`; it is **not** constant anywhere in α ∈ (2, 4), and only
   approaches a constant above α ≈ 4.5. And the measured β=3 `D_V` does not follow from
   its own α anyway: its growth exponent is 3.1–3.5, which no α ≥ 0 can produce, because a
   strict r^-α correlator has `D_V` growing at most like L².

The strong-measurement corner is therefore **not** threatened by this route. What the β=3
data actually shows is a two-component correlator — a short-range part plus a
sum-rule-forced flat floor — whose apparent α drifts with L rather than converging.

Reproduce: `python -I verify_decay.py`. Tables in `docs/tables/`, figures in `figures/`.

---

## 0. Conventions are validated before use

Everything below rests on one identity, which I checked against the receipts' `DV0_full`
before using it for anything:

    D_V^full(G) = Σ_ij |G_ij|² h_x(dx) h_y(dy),      h(d) = min(d, L−d, w)/L,  w = L/3

| L | kernel − receipt (relative) |
|---|---|
| 6 | −4.3 × 10⁻¹⁶ |
| 12 | −3.2 × 10⁻¹⁵ |
| 18 | +1.0 × 10⁻¹⁵ |

The site labelling `cell = i//2, sub = i%2, x = cell % L, y = cell // L` with
`a₁=(1,0), a₂=(½,√3/2)` gives the correct honeycomb (nearest-neighbour degree exactly 3,
216/864/1944 NN pairs at L = 6/12/18). My synthetic displacement enumeration agrees with
the dense kernel sum to 1.1 × 10⁻¹⁶.

Also checked: `Σ_ij |G_ij|² = L²` holds exactly to 6 decimal places in every measured
matrix. This is the constraint that drives the whole story below.

---

## 1. Task 1 — α is not stable under binning, and the estimator is biased by O(1)

### 1.1 The binning systematic

Six binnings, GLS fit weighted by the four independent batches, over r ∈ [0.9, 0.62·L].
The GLS statistical error is tiny (≤ 0.02) — the batches are enormously self-averaging —
so **all** of the scatter below is systematic.

| L | β | shell_exact | geom_8 | geom_12 | linear_0.5 | linear_1.0 | wide_5 | span |
|---|---|---|---|---|---|---|---|---|
| 12 | 1 | 3.17 | 2.96 | 3.41 | 3.08 | 3.27 | 2.88 | **2.88 – 3.41** |
| 18 | 1 | 3.25 | 3.25 | 3.49 | 3.33 | 3.54 | 2.77 | **2.77 – 3.54** |
| 12 | 3 | 0.73 | 0.70 | 0.75 | 0.65 | 0.67 | 0.88 | **0.65 – 0.88** |
| 18 | 3 | 1.07 | 1.30 | 1.32 | 1.22 | 1.27 | 1.31 | **1.07 – 1.32** |

The claim's "α ≈ 3.4–3.6 at β=1" is reached only by `geom_12` / `linear_1.0` at L=18; the
shell-exact fit — the one that uses every distinct distance and no arbitrary bin edges —
gives 3.25. The "α ≈ 1.0–1.5 at β=3" band is consistent with the data but its width is
the *binning* width, not a physical error bar.

### 1.2 The window scan

`figures/verify_decay.png`, panel A-right and `docs/tables/alpha_window_scan.csv`:

| L, β | r_max=3 | 4.5 | 6 | 8 | 10 |
|---|---|---|---|---|---|
| 12, 1 | 3.50 | 3.42 | 3.46 | — | — |
| 12, 3 | 1.06 | 0.91 | 0.81 | — | — |
| 18, 1 | 2.92 | 3.19 | 3.46 | 3.23 | 3.25 |
| 18, 3 | 1.52 | 1.44 | 1.44 | 1.33 | 1.27 |

For β=1 at L=18, α moves 2.92 → 3.46 → 3.25 — non-monotone, and the L=12 and L=18 curves
sit on different tracks at every window. **Two lattice sizes of the same system do not
reproduce each other's exponent.** A quantity that has not converged in L cannot be quoted
as a physical exponent, and in particular cannot be fed into a statement about the L → ∞
behaviour of D_V.

### 1.3 The decisive control: the ground state

The μg = 10⁻⁸ ground state is a free Dirac state; its correlator decays as `|G| ~ r^-2`,
so `|G|² ~ r^-4` **exactly** in the continuum. Running the identical estimator on it:

| L | shell_exact | geom_8 | geom_12 | linear_0.5 | linear_1.0 | wide_5 |
|---|---|---|---|---|---|---|
| 12 | 2.62 | 3.41 | **1.96** | 3.27 | 3.55 | 3.33 |
| 18 | 2.59 | 3.77 | **2.04** | 3.74 | 4.01 | 3.62 |

True answer is 4. The estimator returns anything from **1.96 to 4.01**. That is a bias of
up to **2.0 in α**, purely from binning. The reason is visible in panel A-left: the
ground-state profile is not a clean power law on a torus at these sizes — shells alternate
between symmetry-forced zeros (r = 1.000, 1.732, 2.000 give `|G|² ≈ 5×10⁻⁶`, three orders
below their neighbours) and true power-law shells (r = 1.155 gives `2.3×10⁻²`). The
log-residual RMS of a straight-line fit is **1.9 dex per shell** for the ground state,
0.5 dex for β=1, 0.13 dex for β=3.

So the honest statement is: **α cannot be extracted to better than ±1 from these matrices.**
The claim's 3.4–3.6 vs 1.0–1.5 contrast survives as a qualitative statement that β=1
decays faster than β=3 — that direction is robust — but no quantitative *value* of α, and
therefore no inference that depends on whether α is above or below 2, is supported.

---

## 2. Task 2 — the D_V rule in the claim is simply false

The claim asserts: the ln L coefficient is "fixed by α = 4 (baseline) and **saturates to a
constant for α > 2**." I tested this directly by constructing
`|G_ij|² = c·r^-α` on the real honeycomb torus with the real region geometry, normalised
to the real sum rule `Σ|G|² = L²`, and computing `D_V(L)` for L = 6…192 by exact
displacement enumeration (`docs/tables/synthetic_DV_vs_alpha.csv`):

| α | D_V(6) | 12 | 18 | 48 | 192 | growth exp (128→192) |
|---|---|---|---|---|---|---|
| 1.0 | 1.422 | 5.283 | 11.62 | 80.39 | 1271 | **+1.98** |
| 1.5 | 1.202 | 4.096 | 8.661 | 55.85 | 833.0 | **+1.95** |
| 2.0 | 0.9859 | 2.904 | 5.670 | 30.62 | 366.1 | **+1.80** |
| 2.5 | 0.7946 | 1.907 | 3.283 | 12.91 | 95.99 | **+1.45** |
| 3.0 | 0.6425 | 1.216 | 1.790 | 4.660 | 18.44 | **+0.99** |
| 3.5 | 0.5321 | 0.8024 | 1.014 | 1.750 | 3.661 | **+0.52** |
| 4.0 | 0.4576 | 0.5781 | 0.6501 | 0.8259 | 1.075 | **+0.17** |
| 4.5 | 0.4098 | 0.4619 | 0.4855 | 0.5263 | 0.5586 | **+0.03** |
| 5.0 | 0.3801 | 0.4024 | 0.4099 | 0.4193 | 0.4235 | **+0.004** |

`D_V` follows `L^max(0, 4−α)` (panel B-right, dashed line). Consequences:

- **α = 2 does not mark a transition.** At α = 2 the exponent is +1.80, at α = 3 it is
  +0.99. There is no plateau at any α < 4.
- **Nothing is constant in α ∈ (2,4).** Constant behaviour requires α ≳ 4.5 — i.e. a
  correlator decaying *faster* than the free-Dirac baseline, which is the opposite of a
  "growth". The claim has the direction of the criterion backwards: it is α > 4, not
  α < 2, that removes the growth.
- **Even α = 4 is not truly ln L here.** The local exponent at α = 4 is +0.17 at L ≈ 100
  and still falling; the true ln L behaviour is the L → ∞ limit of the slow crossover, and
  the ground-state baseline confirms this (`DV/ln L` = 0.0722, 0.0737, 0.0732, 0.0717,
  0.0690 for L = 6, 12, 18, 48, 1536 — drifting slowly down, not constant).
- **An r^-α correlator cannot produce the growth the β=3 data shows.** The maximum
  possible exponent is +2 (α = 0). The measured β=3 `D_V` exponent is **+3.1 to +3.5**
  (§4). No power law of this form, at any α, fits. This alone refutes the claim's
  mechanism: whatever drives the β=3 `D_V` growth, it is not "α < 2".

---

## 3. Task 3 — the observed α is a finite-size crossover, not a converged exponent

Restricting to r < L/3 and subtracting the sum-rule-forced uniform floor c = Σ_off /[N(N−1)]
(`docs/tables/alpha_crossover.csv`):

| L, β | r<L/3 raw | r<L/3 after floor | all raw | all after floor | far(>L/3)/floor |
|---|---|---|---|---|---|
| 6, 1 | 2.41 | 2.00 | 1.97 | 2.00 | 0.34 |
| 6, 3 | 0.46 | 3.17 | 0.33 | 3.17 | 0.91 |
| 12, 1 | 2.94 | 2.65 | 2.76 | 2.65 | 0.09 |
| 12, 3 | 1.03 | 2.79 | 0.74 | 2.79 | 0.77 |
| 18, 1 | 3.27 | 3.67 | 3.11 | 3.67 | 0.04 |
| 18, 3 | 1.37 | 2.95 | 1.07 | 2.95 | 0.60 |

Two things happen at β=3:

1. **The α that the claim quotes is not scale-invariant.** At r < L/3 it is 0.46 → 1.03 →
   1.37 as L goes 6 → 12 → 18. It *increases* with L. A genuine r^-α with α < 1 would be
   stable; a near-field power law superposed on a flat floor necessarily drifts upward as
   the floor's share of each shell shrinks.
2. **The far field is close to flat and set by the sum rule.** The ratio far-field/floor
   is 0.91, 0.77, 0.60 — the far pairs are near, and slightly below, the uniform value
   forced by `Σ|G|² = L²`. The near/far contrast grows 1.31 → 3.69 → 9.47.

The two-regime picture is explicit in the amplitudes:

| L | β=3 near (r<1.3) | β=3 plateau (r>L/3) | β=1 near | β=1 plateau |
|---|---|---|---|---|
| 6 | 4.02 × 10⁻⁶ | 2.79 × 10⁻⁶ | 4.98 × 10⁻³ | 4.52 × 10⁻⁴ |
| 12 | 9.78 × 10⁻⁶ | 2.04 × 10⁻⁶ | 5.14 × 10⁻³ | 3.12 × 10⁻⁵ |
| 18 | 2.16 × 10⁻⁵ | 1.38 × 10⁻⁶ | 5.17 × 10⁻³ | 5.73 × 10⁻⁶ |

β=1: near-field is L-independent (L^{+0.04}) and the far field collapses as L^{−3.96} —
a clean, scaling correlator with no floor problem. β=3: the near field *grows* as L^{+1.51}
and the far field decays only as L^{−0.62}. Neither exponent is a physical α; both are the
trivial 1/N normalisation of a fixed total mass L² spread over N = 2L² sites.

**This also explains the L=12 vs L=18 disparity in §1.** At L=12 the floor sits at 0.77 of
the far-field shell; at L=18 it has dropped to 0.60. The crossover radius moves, so the
fitted α moves.

---

## 4. What actually carries D_V at β=3: trajectory variance, not a long-range correlator

Because `meanGsquare.bin` is the average of `|G_E,ij|²` over 512 draws, it mixes the
coherent part `<G>²` with the trajectory-to-trajectory variance. Splitting them
(`docs/tables/DV_split_coherent_variance.csv`):

| L, β | D_V total | coherent `<G>²` | variance | variance share |
|---|---|---|---|---|
| 6, 1 | 1.187 × 10⁻¹ | 2.20 × 10⁻² | 9.67 × 10⁻² | 0.815 |
| 12, 1 | 1.884 × 10⁻¹ | 3.20 × 10⁻² | 1.56 × 10⁻¹ | 0.830 |
| 18, 1 | 2.233 × 10⁻¹ | 3.73 × 10⁻² | 1.86 × 10⁻¹ | 0.833 |
| **6, 3** | **7.34 × 10⁻⁴** | **4.05 × 10⁻⁶** | **7.30 × 10⁻⁴** | **0.994** |
| **12, 3** | **8.34 × 10⁻³** | **1.18 × 10⁻⁵** | **8.33 × 10⁻³** | **0.999** |
| **18, 3** | **2.92 × 10⁻²** | **2.81 × 10⁻⁵** | **2.91 × 10⁻²** | **0.999** |

At β=3, **99.9 % of D_V is trajectory variance**; the coherent correlator contributes
0.1 %. Dividing by radius makes it starker — at r > 2.9 the variance fraction is 0.998–0.999.
So `mean|G_ij|²` at β=3 is emphatically *not* a measure of a long-range correlation: it is
a measure of how wildly `G_E,ij` fluctuates between outcomes. Calling it "the
correlation-square" and reading off a decay exponent misidentifies the object.

The coherent part's own scaling is unstable too: local exponent +1.54 then +2.14 for L =
6→12→18 — again drifting, not converging, and again the 1/N normalisation of a fixed mass.

---

## 5. The measured D_V versus the ln L baseline

`docs/tables/measured_DV.csv`, `figures/verify_decay_DV.png`:

| L | ground D_V (μ=0) | measured β=1 | measured β=3 |
|---|---|---|---|
| 6 | 0.1294 | 0.1187 | 7.34 × 10⁻⁴ |
| 12 | 0.1830 | 0.1884 | 8.34 × 10⁻³ |
| 18 | 0.2115 | 0.2233 | 2.92 × 10⁻² |
| 48 | 0.2775 | — | — |
| 1536 | 0.5064 | — | — |

β=1 tracks the ground state almost exactly (ratios 0.92, 1.03, 1.06) and its local
exponent 0.67 → 0.42 matches the ground state's 0.50 → 0.36. β=1 is consistent with a
healthy ln L corner, with the slight excess attributable to the 83 % variance share.

β=3 is far below the baseline and rising fast (local exponents 3.51 → 3.09). But — and this
is the key point — its absolute magnitude is ~7× below the ground state at L=18 despite
having "less" decay. So the correct reading is **not** "the corner grows faster than ln L
at strong measurement"; it is "at β=3 the corner has been *suppressed* by more than an order
of magnitude and is climbing back from a very small value as L grows." The β=3 curve would
have to rise by a factor ~7 just to reach the ground-state line, and there is no evidence in
these three sizes of where it saturates.

Both readings share a limitation worth stating: three sizes (6, 12, 18) cannot distinguish
ln L from any slow power, and β=3's own `κ²  D_V0` lower bound is ~2000× below the measured
value, so the bound carries no information here either.

---

## 6. What survives, and what I would do instead

**Survives:**
- `mean|G_ij|²` does decay faster at β=1 than at β=3. Direction is robust across all six
  binnings and both sizes.
- The non-negativity identity `D_V = Σ|G_ij|² h_x h_y ≥ 0` and the resulting lower bound
  `E[D_V] ≥ κ² D_V0` — I re-verified the kernel to 1e-15 against the receipts.
- The ground-state ln L corner coefficient (0.066–0.071 from the baseline scan) is a solid
  number, and β=1 is consistent with inheriting it.

**Refuted:**
- That α can be quoted as 3.4–3.6 / 1.0–1.5. The binning systematic exceeds the quoted
  width, and the ground-state control shows the estimator is biased by up to 2.0.
- That α > 2 makes D_V constant. It does not; the law is `L^max(0,4−α)` and constancy
  needs α ≳ 4.5.
- That α < 2 at β=3 drives a super-log corner. The measured β=3 exponent (+3.1…+3.5)
  exceeds the maximum any r^-α can produce (+2), so the mechanism is not operating.
- That `mean|G_ij|²` at β=3 measures a correlation. 99.9 % of it is trajectory variance.

**Concrete next steps, in priority order:**

1. **Use `<G_E,ij>²` and `Var(G_E,ij)` as separate observables, not their sum.** The
   variance is the object with physical content at strong measurement, and it is what the
   `D_V` bound actually constrains.
2. **Do not fit α at all at these sizes.** Fit the two-parameter form
   `|G|² = a·r^-α + c` with `c` fixed by the sum rule, or better, fit the coherent part
   alone. With ≤ 3 sizes, report the profile itself.
3. **Extend to L = 24, 36, 48 at β=3.** The drifting local exponent (0.46 → 1.03 → 1.37)
   must either stabilise or keep climbing; three points cannot tell which, and the entire
   claim hinges on it.
4. **Test the corner on the observable that is actually well-conditioned.** `D_V` at β=3 is
   7× below the ground state and 2000× above its own deterministic bound — there is a lot of
   room for it to be a variance-dominated finite-size transient. Measure `D_V` for a series
   of β at fixed L to see whether the suppression is monotone or has structure.

---

## Files

| path | contents |
|---|---|
| `verify_decay.py` | this analysis, self-contained |
| `docs/tables/alpha_by_binning.csv` | α for 6 binnings × {ground, β=1, β=3} × {L=12,18} |
| `docs/tables/alpha_window_scan.csv` | α vs fit window |
| `docs/tables/alpha_crossover.csv` | r < L/3 and floor-subtracted α |
| `docs/tables/synthetic_DV_vs_alpha.csv` | D_V(L) for α = 1.0…5.0, L = 6…192 |
| `docs/tables/measured_DV.csv` | measured D_V with batch SD and the κ²D_V0 bound |
| `docs/tables/DV_split_coherent_variance.csv` | D_V split into `<G>²` and variance |
| `figures/verify_decay.png` | 4-panel: profile, α-vs-window, synthetic D_V, growth law |
| `figures/verify_decay_DV.png` | measured D_V against the ground-state ln L baseline |
