# Referee report: the matched-`z` collapse of `D̄₂`

Review date 2026-10-30. Target: `REPORT.md` (claims C1–C5) plus
`BORN_SINGINGLE_LAYER_THEOREMS.md` and `LITERATURE.md` in this directory.
Every number below was recomputed from `data_matched_z/`, `baseline_D.json`
and `crossover_fit.json`; the script is `referee_checks.py`, the figure is
`figures/referee_checks.png`.

The short version: **the underlying sampling is clean and the fixed-`z` rescaling
is real, but the paper's causal story is a tautology, one headline number
flips sign on the newest data in the same directory, and the mechanism stated in
§1 of the report is contradicted by the report's own §4.**

---

## 0. What survives

I want to be clear about what I could not break, because it bounds how much
needs rewriting.

| Claim | Status after reanalysis |
|---|---|
| Sampler and entropy code | Sound. I recomputed all 31 `(L,β)` points from the raw per-sample CSVs; means and SEMs reproduce. |
| `z = L²e^{-2β}` governs the *position* of the crossover | **Correct and important.** At fixed `z`, `D̄₂` moves ≪ the ground-state value does. The old "per-β asymptote `c(β)`" reading is genuinely dead. |
| `D̄₂ < D_gs` for all finite `z`, `D̄₂ → D_gs` as `z → ∞` | Correct, and it is the single most robust statement in the paper. |
| Annealed Gaussian gives volume law | Correct in conclusion (§4 caveat below). |
| Exactly-sampled Born ensemble, no tail truncation | True, and it is the paper's real methodological asset. |

The rest of this report is about what does **not** survive.

---

## 1. The fatal one: C3 is a tautology, not a measurement

This is the objection I would lead with, and it is cheap to verify.

`C3` defines `z_*(L)` as the `h = D̄₂/D_gs = 1/2` crossing, then reports that
`z_*/L ≈ 0.19` and concludes the exponent is linear in `L`.

But if `C1` is even approximately true — `D̄₂ = f(z)` with no explicit `L` — then
the crossing point is *not an independent measurement at all*:

```
D̄₂(z_*) = D_gs(L)/2   ⇒   f(z_*(L)) = D_gs(L)/2
                      ⇒   z_*(L) = f^{-1}(D_gs(L)/2).
```

`z_*` is whatever number you must feed the already-assumed collapse curve to
reach half the ground-state value. It is a *derived* quantity. Fitting a power
law to it and reporting the exponent is reporting a property of `f` and
`D_gs`, both of which were assumed in advance.

I checked this numerically. Taking the empirical `f(z)` (mean of `D̄₂` over all
`L` at each `z`) and inverting it at `D_gs(L)/2` predicts the measured crossings
to within 9% at every `L`:

| `L` | 6 | 12 | 18 | 24 | 30 | 36 |
|---|---|---|---|---|---|---|
| `z_*` measured | 1.388 | 2.731 | 3.687 | 4.040 | 5.403 | 6.014 |
| `z_* = f⁻¹(D_gs/2)` | 1.509 | 2.713 | 3.612 | 4.334 | 4.962 | 5.525 |
| ratio | 1.09 | 0.99 | 0.98 | 1.07 | 0.92 | 0.92 |

There is nothing left over. Panel (a) of `figures/referee_checks.png`.

**Consequence for the paper.** `C3`'s exponent `2` in `L*(β) ~ e^{2β}` is *not*
supported by an additional measurement. It is a restatement of
`D_gs ~ 0.155 ln L` through the inverse of the assumed `h`. Any functional form
of `f` that saturates produces the same behaviour. So the phrase in `REPORT.md`
§2.4 — "the linear form has direct physical meaning" — is exactly backwards:
the linear form has *no* independent support, because the crossing it is fitted
to is not independent.

**The fix is available and cheap.** The non-tautological version of the test is
to *predict*, not to fit. Fit `f(z)` and `z_*(L) = A L^p` on `L ≤ 24`, then
predict the absolute value of `D̄₂` at `L = 30` and `L = 36` and compare. My
leave-one-out run of the report's own model gives relative rms errors of
33%, 23%, 24%, 11%, 33%, 12% for `L_out = 6,12,18,24,30,36` — i.e. **the
two-parameter collapse does not predict an unseen size to better than ~10–30%**,
which is the same order as the effects it claims. That is the number the paper
should report.

---

## 2. C4's dimensional argument fails its own consistency check

The report states, in §2.4:

> `z ~ L^d e^{-2β}`, crossover at `z ~ L^{d-1}`, so `d=1` has no crossover
> while `d=2` gives `L* ~ e^{2β}`.

Do the algebra:

```
L^d e^{-2β} = c L^{d-1}   ⇒   L e^{-2β} = c   ⇒   L* = c e^{2β}.
```

**The exponent of `L` cancels identically. `d` drops out entirely.** The
argument gives `L* ~ e^{2β}` for every `d`, including `d = 1`. It therefore
*cannot* explain why 1D is protected and 2D is not — which is the entire reason
`C4` exists in the paper. Only the dimensionless prefactor `c` differs between
dimensions, and a prefactor difference is not a protection theorem.

**Separately, the "defect spacing" language does not produce the stated
amplitude.** If `z` defects sit on `L²` sites, the density is `ρ = z/L²` and the
mean spacing is `d = L/√z`. The report's condition "spacing exceeds the boundary
length" is `L/√z > L`, i.e. **`z < 1` — independent of `L`.** That predicts
`z_* = O(1)`, giving `L* ~ e^β`, not `e^{2β}`. The report instead asserts
`z_* ≈ 0.19 L` with no derivation; those two statements are mutually
inconsistent, and the one that follows from the stated mechanism is the one that
is wrong.

So `C4` does not predict the observed amplitude `0.19`, and it does not survive
as a dimensional argument. Either drop it or replace it with something that
actually distinguishes dimensions.

### Is there a cleaner derivation? Partly, and it is not flattering

`z` has a mundane reading. With `|E| = a = L²` and `t = e^{-2β}`,

```
⟨j⟩ = z · I₁(2z)/I₀(2z)  ⇒  ⟨j⟩ ≃ z  for z ≪ 1,
```

so `z` is the mean number of filtered particles. And `z ∝ L²` simply because
the *number of measured sites* is `L²`. A protocol that measured a fixed number
`s` of sites would have `z = s e^{-2β}` and **no crossover in `L` at all**.

That is the honest content of `C1`: the crossover appears because the
measurement covers a finite *fraction* of the system, so the total filtered
particle number grows with system size. This is geometry, not a novel
measurement-strength variable. It is worth saying this plainly, because it
immediately raises the objection the paper never addresses: **is this
observable at all in a protocol where the measured region does not scale with
the sample?** If yes, `C1`–`C3` describe a lattice bookkeeping effect. If no,
the paper should say the effect is specific to half-system conditioning — which
is exactly question (e).

---

## 3. C2's "slope = 0" does not establish what it claims, and one slope is unstable

### 3a. The comparison is right; the *reporting* is not

Asking whether the fixed-`z` slope equals `0` or `0.155` is a fair question. And
the answer is decisive: `c = 0.155` is excluded at 21–466σ across the five `z`
lines. That part of `C2` I accept.

But "`D̄₂` saturates at fixed `z`" and "`dD̄₂/d ln L = 0` at fixed `z`" are the
same statement, and *any* saturating function gives the latter. So `C2`
as written rules out "`D̄₂` keeps growing like the ground state" and nothing
more. It does **not** rule out residual `L`-dependence, and there is some:

| `z` | χ²/dof vs constant | verdict |
|---|---|---|
| 0.089 | 0.91 | constant OK |
| 0.357 | 2.25 | **rejected** |
| 0.803 | 1.22 | constant OK |
| 1.428 | 3.25 | **rejected** |
| 3.213 | 0.43 | constant OK |

Mean χ²/dof = 1.61 over 5 dof. `f(z)` alone is a *good* description, not a
perfect one, and the report's "slope ≈ 0" framing hides two 2–3σ violations.

### 3b. The one slope that is 3.3σ is driven by the least-controlled point

At `z = 1.428` the reported slope is `−0.0132 ± 0.0040`. Its data:

```
L= 6  0.14769 ± 0.00402
L=12  0.14456 ± 0.00441
L=18  0.13681 ± 0.00431
L=24  0.12492 ± 0.00649
L=30  0.12756 ± 0.00811
```

`L=24` (n = 1024) and `L=30` (n = 512) carry the whole effect. Drop `L=24`:
slope becomes `−0.0110 ± 0.0044` (2.5σ). Drop `L=6`: `−0.0224 ± 0.0084` (2.7σ).
So the deviation is not carried by one point — but it *is* concentrated in the
two smallest samples, and the two largest `L` are already turning back up
(`0.1249 → 0.1276`). The report calls this "converges to zero with increasing
`z`", which is a post-hoc reading of five noisy slopes; a χ² test for a common
slope across all five `z` would be the honest summary and is not given.

### 3c. The reference value `0.155` is not what the ground state does at these `L`

`C2` compares to `0.155`, the extrapolated asymptote. But inside the window
where `D̄₂` is actually measured, the ground state's *local* slope is larger:

| window | `D_gs` local slope |
|---|---|
| L = 6–36 | 0.2087 |
| L = 6–24 | 0.2205 |
| L = 18–30 | 0.1876 |

The fair null hypothesis is "`D̄₂` has the same *local* slope as `D_gs` in the
same window", i.e. ~0.19–0.22, not 0.155. Using the correct local null makes the
exclusion even stronger, so this *helps* the paper — but it should be stated
correctly, because a referee who notices will wonder what else was compared to an
asymptote rather than a local value.

### 3d. The headline number at `z = 3.213` flips sign on the newest data

This is the most serious concrete error I found.

`REPORT.md` §2.1 tabulates `z = 3.213` through `L = 36` using the *old*
production value `D̄ = 0.24214`, giving slope `−0.0052 ± 0.0030`. But
`data_matched_z/L36_beta3.0000.csv` — written at 18:51, i.e. *after* the report
text — contains an independent 512-sample run at the same point giving
**`0.25993 ± 0.01164`**. The two differ by 1.4σ. Recompute the slope with the new
point:

| L=36 source | slope at `z=3.213` | χ²/dof vs constant |
|---|---|---|
| old production (report) | `−0.00515 ± 0.00305` | 1.27 |
| **new independent run** | **`+0.00237 ± 0.00481`** | 0.43 |

The sign flips. The report's claim "the only systematic deviation is `z=1.43`'s
−0.013" quietly depends on discarding the newer of two measurements of the same
point. Since both are in the same directory, this is a data-selection problem,
not a subtlety. It must be resolved — average them with the correct correlation,
or state which one is authoritative and why — before `C2` can be quoted.

Note also the two values are compatible with a *constant* only if the true value
is near `0.25`; a 1.4σ spread between independent runs at fixed `(L,β)` is
also a warning about how large the SEMs really are on this line.

---

## 4. C5 is right, but its supporting numbers are wrong

The conclusion — annealed Gaussian is volume-law and useless — is correct, and
worth keeping. The table in §4 is not.

| `L` | `w = L/3` | `2w(L−2w)ln2` | report | ratio |
|---|---|---|---|---|
| 10 | 3.33 | 15.40 | 8.2 | 1.88 |
| 20 | 6.67 | 61.61 | 43.6 | 1.41 |
| 40 | 13.33 | 246.45 | 241.6 | 1.02 |
| 60 | 20.0 | 554.52 | 553.0 | 1.00 |

`L = 10, 20` are off by 1.4–1.9×, and — as the report's own `mechanism_figures.py`
docstring notes — they are not multiples of 3, so `w = L/3` is not an integer and
the geometry is invalid. `L = 40, 60` agree. Fix the table to multiples of 3;
the qualitative point is unaffected, but leaving invalid sizes in a table that
carries a "this refutes the annealed route" claim is an avoidable referee target.

---

## 5. The mechanism in §1 contradicts §4, and §1 is the one that is wrong

`REPORT.md` §1 states:

> `j = 0` 的输出把测量后态变成**严格乘积态**（`D = 0`），所以整个角熵只由
> `j ≥ 1` 的稀有输出携带。

The data say otherwise, for every `L`:

| `L` | `z` | `⟨D \| j=0⟩` | share of `D̄₂` from `j=0` |
|---|---|---|---|
| 6 | 0.089 | 0.00124 | 85% |
| 12 | 0.089 | 0.00161 | 81% |
| 18 | 0.089 | 0.00127 | 90% |
| 24 | 0.089 | 0.00124 | 69% |
| 30 | 0.089 | 0.00092 | 92% |

At the smallest `z`, 69–92% of `D̄₂` comes from `j = 0` samples, whose mean `D`
is *not* zero. The report's own §4 table contradicts §1 in the same document:
it lists `⟨D | j=0⟩ = 0.046` with a 36% share at `L=18, β=3`.

The resolution is already written in `mechanism_figures.py`'s header — and the
report did not propagate it:

> `j = |E ∩ F|` is the overlap of the measurement set with the **auxiliary**
> Slater determinant, not with the physical post-measurement support. At `j=0`
> the filtered state is the near-product branch with all particles of the
> dominant Fock component on `Eᶜ`; it is a product state **only** as `β→∞`.

So `j = 0` gives a product state only in the strong-measurement limit, and at
finite `β` it carries `D ~ A z²`. This matters for the physics narrative: the
`z²` law is *not* "one filtered particle interfering with a product state", it
is the leading deviation of the dominant branch from a product state. That is a
different (and less interesting) statement, and §5(c) of the report gets it
wrong for the same reason.

**This is a one-paragraph fix, but as written §1 is a false statement about the
paper's own data, and it is the sentence that motivates the entire "rare
branches carry everything" story.** A referee who checks the CSV will find it in
one command.

---

## 6. Answers to the specific questions asked

### (a) Is `C2`'s slope-zero the right comparison?

Yes as a comparison, but it is a weak one, and the paper over-reads it. It rules
out "`D̄₂` grows like `D_gs`" at >20σ, which is real content. It does not
establish that `D̄₂` is `L`-independent: χ²/dof vs constant is 2.25 and 3.25 at
two of the five `z`. And the tautology in §1 above means the *crossing* extracted
from this is not an independent measurement. **Report the χ² table (§3a) and the
leave-one-out prediction error (§1), not the slope alone.**

### (b) How much evidence supports linear over `L^0.8`?

Less than the report implies, and the report already knows it — §2.4 gives
`rms = 0.36` for linear vs `0.22` for power law and then chooses linear anyway.
My refit on the current six crossings:

```
free power:  p = 0.799 ± 0.046   (dof = 4)
z*/L:        0.231, 0.228, 0.205, 0.168, 0.180, 0.167
```

`p = 1` sits **4.3σ** from the best fit with the new `L = 36` point included (and
3.1σ with only five points). The residuals are also not monotone — the sequence
dips at `L=24` and recovers at `L=30`, which no smooth `1/L` correction would do.

At `β = 3` the two forms give `L* = 61` (power law) vs `L* = 75` (linear) — a 20%
difference in the quantity the paper wants to quote. **Neither is measured; both
are extrapolations of a 3-parameter description fitted to 5–6 points.** The
honest statement is: `z_*(L)` grows *slower than linear* over `L = 6–36`, and
whether it is `L^0.8` or `L` cannot be decided at this precision. Quoting
`e^{2β}` rather than `e^{1.6β}` is therefore not supported.

### (c) Is there a cleaner derivation? Does it predict 0.19?

No, and no. See §2: the algebra cancels `d`, and the defect-spacing version
predicts `z_* = O(1)` rather than `0.19 L`. The amplitude `0.19` is not derived
anywhere — it is the mean of five `z_*/L` values with 14% scatter.

What *would* be a cleaner derivation: a statement about the entanglement
Hamiltonian. If the post-measurement state has a gapped EH (as Tang–Wen find in
1D), then `D` loses its `ln L` term when the gap exceeds the level spacing at the
corner, which is `~1/L`. That gives a crossover when `gap·L ~ 1`, a genuinely
different route to `z_*`. The paper cites Tang–Wen but does not use this.

### (d) Is there a known mechanism predicting a different `L*`?

Yes, and it is the one worth testing: **percolation of the `j = 0` patches.**
At fixed `z`, the sites in `E ∩ Fᶜ` are dilute; whether the corner `ln L` term
survives should depend on whether those sites percolate across the corner
region, which happens at a *finite density*, i.e. at `z/L² = O(1)` — giving
`z_* ~ L²` and `L* ~ e^β`, not `e^{2β}`. Note that `z/L² = e^{-2β}` is
`L`-independent, so this mechanism predicts **no crossover in `L` at fixed `β`**
— the opposite of `C3`.

This is directly testable with data already in hand: measure the size of the
largest connected `j = 0` cluster in the corner region as a function of `(L, β)`.
If it percolates below `z_*` and not above, `C3` is wrong and `C1`'s `L²`
bookkeeping is the whole story. **This is the measurement I would demand.**

### (e) Does fixed-half conditioning bias `D`?

Almost certainly yes, and the paper never checks. Two specific concerns:

1. **The measured set always covers exactly half the lattice.** `z = L²e^{-2β}`
   is therefore not a "measurement strength" in any protocol-independent sense —
   it is `(number of measured sites) × e^{-2β}`, and the first factor is forced
   by the conditioning. A protocol with `|E| = L²/2` or `2L²` would give a
   different `z` trajectory and hence a different `z_*(L)`. `C1` is a statement
   about *this* conditioning.

2. **`A` and `C` are symmetry-equivalent, and their difference is pure noise.**
   `⟨S_A − S_C⟩` is `±0.03` while `⟨S_A − S_B⟩ ~ 0.25` (I tabulated all 37
   points). Since `D = (2S_A + S_C)/3 − S_B = (S_A − S_B) + (S_A − S_C)/3`, the
   second term contributes a few percent of noise and no signal. So `D` is
   effectively `S_A − S_B` for two regions of *equal size* `L²/3` that differ only
   in shape and in where they sit relative to `E`. That is worth stating — and it
   means the `2:1` weighting in the definition is doing nothing.

   Note also that at `β=0`, `S_A = S_C` to `1e-14` by the 90° rotation symmetry
   of the Dirac ground state, so the `(S_A − S_C)/3` term is an *exactly zero*
   contribution at `β=0` and a finite-sample fluctuation otherwise. Any residual
   `L`-dependence in it is a sampling artifact, not physics.

### (f) Is `D_gs(L)` reliable at `L=60`?

The in-sample fit is fine; the *model dependence* is the problem, and it is
larger than the paper admits.

Extrapolation stability is good — training on `L ≤ 42` and testing on `45–60`
predicts to `3×10⁻⁴`:

| train | test | `c` | max pred error |
|---|---|---|---|
| 6–42 | 45–60 | 0.1561 | 3.2e-04 |
| 6–48 | 51–60 | 0.1560 | 2.0e-04 |
| 6–51 | 54–60 | 0.1559 | 1.6e-04 |

But change the correction exponent and `c` moves a lot:

| correction | `c₂` (L=24–60) |
|---|---|
| `+a/L` | 0.1554 |
| `+a/L + b/L²` | 0.1554 |
| `+a/L^1.5` | 0.1620 |
| `+a/L^0.5` | 0.1358 |
| pure `c ln L + b` | 0.1750 |

`c₂` spans **0.136–0.175**, i.e. ±13% around the quoted `0.1554`, and the data
inside `L ≤ 60` cannot distinguish these — every one of them fits to `<1e-4`.
The quoted `±0.0002` is the in-sample statistical error of one model, not an
uncertainty on the extrapolated coefficient.

Does this matter? For `C2` it does not — the exclusion of `0.155` at fixed `z` is
35–466σ, far beyond a 13% shift in the reference. For `C3` it does: `L*(β)` is
defined by matching `z` to `D_gs/2`, so a 13% error in `c₂` shifts `L*`
noticeably. **Report `c₂ = 0.155 ± 0.02` (model spread), not `0.1554 ± 0.0002`.**

---

## 7. Four mutually inconsistent formulas for `L*(β)`

`REPORT.md` gives, in one document:

| location | formula | `L*(β=3)` |
|---|---|---|
| §0 | `L*(β) ≈ 5e^β` | 100 |
| §2.4 | `L*(β) = 0.185 e^{2β}` | 75 |
| §3 | `L*(β) ≈ 0.40 e^{1.68β}` | 62 |
| §5(b) | `L*(β) = 2e^β` | 40 |

These range over a factor 2.5 at `β = 3`. All four cannot be right, and the
`e^β` versions contradict `C4` (which is supposed to give `e^{2β}`). This looks
like text written across several passes without a final reconciliation, but a
referee will treat it as evidence that the fitting was not settled. Pick one,
derive it once, and use it everywhere.

---

## 8. What I would accept as proof

In priority order, cheapest first.

1. **Kill the tautology.** Fit `f(z)` and `z_* = A L^p` on `L ≤ 24`; predict
   `D̄₂(L=30, 36)` at matched `z` blind. Success = prediction within the stated
   SEM. Current leave-one-out error is 11–33%, i.e. a failure.
2. **Add a `β = 0` anchor to `data_matched_z/`.** There is currently no
   `β = 0` run in this dataset, so the statement `D̄₂ → D_gs` is never validated
   *within the same code path*. Meanwhile `h` exceeds 1 by 1–3.5% at the largest
   `z` (`L=18,24,30`), which is physically impossible if `D_gs` is the true
   `z→∞` limit — see panel (d). Either the two `D` definitions differ or `D_gs`
   is biased; one 2048-sample `β = 0` run settles it. **Do this first — it costs
   one run and it can invalidate the normalisation used everywhere else.**
3. **Test the percolation alternative (§6d).** Largest `j=0` cluster size vs
   `(L, β)`. If it percolates at `z < z_*`, `C3` is wrong.
4. **Resolve the two `L=36, β=3` values** (`0.24214` old vs `0.25993` new) and
   rerun the `z=3.213` slope. Report the result whichever way it falls.
5. **Push `L` at fixed `z = 3.21` to `L = 42, 48`** with ≥ 2048 samples. This is
   the one measurement that separates `L^0.8` from `L^1`: the two forms differ by
   20% in `L*` at `β=3` but are indistinguishable on `L ≤ 36`.
6. **Measure the EH gap** at `z ≈ 4`. This is the only way to distinguish
   "coefficient returns" from "still a Dirac CFT", which the paper correctly
   flags as unresolved.

Until (1) and (2) are done, `C3` should be presented as a *parametrisation* of
the data rather than a measured crossover scale, and `C4` should be removed
rather than hedged — an argument that cancels `d` cannot be the explanation for
a `d`-dependent effect.

---

## 9. Writing-level issues that will draw fire

- **§1's `j=0 ⇒ D=0`** is false for this data (§5). Delete or correct.
- **§2.1's slope table** mixes an old `L=36` point with new sampling (§3d).
- **§4's annealed table** contains non-multiples of 3 (§4).
- **§2.4 chooses linear over power law** while printing a table in which the
  power law fits better (§6b). If linear is chosen on physical grounds, the
  physical grounds must be stated *and defended*, not asserted.
- **The four `L*(β)` formulas** (§7).
- **`c₂ = 0.1554 ± 0.0002`** should carry the model-spread error (§6f).

The data in `data_matched_z/` are good, the sampler is clean, and the central
rescaling observation — that `z`, not `β`, is the right axis — is correct and
worth publishing. The problems above are all in the layer above the data: one
tautological inference, one broken dimensional argument, one incorrect mechanism
sentence, and a headline slope that changes sign when the newest file in the
directory is included. None of that requires redoing the sampling, except for the
`β = 0` anchor, and all of it is fixable.
