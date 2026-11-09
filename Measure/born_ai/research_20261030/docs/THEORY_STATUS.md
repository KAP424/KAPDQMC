# Theory layer audit — Born single-layer theorems

Audit date: 2026-10-30. Read-only on originals.

Sources audited:

- `src/Measure/born_ai/BORN_SINGLE_LAYER_THEOREMS.md` (sections 1–5, 7)
- `src/Measure/born_ai/research_20261029/theory_review.md` (full)
- `src/Measure/born_ai/research_20261029/check_theory.py`
- Supporting: `research_20261029/theory_checks.json`, `baseline/summary.json`, `iid_targeted_analysis/{means,bounds}.csv`, `exact_corner_counterexample.{py,json}`

Everything below marked NUMERICALLY VERIFIED was re-run by me in the audit session; those runs are not archived (they are short deterministic checks, reproduced inline below as one-liners).

---

## 0. Summary judgement

The mathematical core is sound. I found **no false theorem**. The mean-channel identity, the κ bounds, the Jensen direction, the growth-order bound, the corner-kernel identity, the counterexample, and the Bessel crossover all survive independent recomputation, several of them to 1e-16.

What I flag is **scope inflation in the status labels**, in two places (items 1–2), plus two housekeeping hazards (items 3–4). Item 3 is the one place the documents get scope labelling exactly right and is the model the other two should follow.

1. `sech(2β) ≤ κ ≤ sech²β` is stated in the theorems file as a single boxed chain, but the *upper* end and the monotonicity are properties of the half-filled fixed-half-output family (M=2a, n=k=a) only. The general `λ ≥ sech(2β)` bound holds for any M,n,k (I verified 1510 combos, 0 violations); the upper bound does not — `theory_review.md:220–233` states it correctly, `BORN_SINGLE_LAYER_THEOREMS.md:181–187` drops the hypothesis.
2. The two "lower bound" statements (D_V ≥ sech⁴β·c_V ln L, and the S₂ growth-order bound) are labelled as bounds on the Born average, and they are — but the β=3 production points sit 3–4 orders of magnitude above the D_V bound (`mean/lower` up to 2150 at L=18, from `bounds.csv`). They are true and vacuous. Neither document claims otherwise in words, but the phrase "非扰动保护下界" (non-perturbative protective lower bound) in the paper-candidate thesis sentence at `BORN_SINGLE_LAYER_THEOREMS.md:509` invites a reader to think the bound controls the physics at the β of interest. It does not at β=3.
3. The counterexample section carries a "scope" caveat in the JSON and a matching sentence in both markdowns — this one is handled correctly and is the model the other two should follow.
4. `check_theory.py`'s `search_corner_inequality` writes a counterexample on the **12th random trial** and returns early (`check_theory.py:176–180`). The saved `.npz` and the rational-certificate example in `exact_corner_counterexample.json` are **different states** with different numbers (`D_S2 = 0.0156655` vs `0.0239703`). `theory_review.md:384` does warn not to mix them; the theorems file `:353–356` quotes only the rational one. No error, but two different "the" counterexamples circulating under one name is a citation hazard.

---

## 1. Claims table

Status vocabulary: PROVED = complete argument in the document, checked against hypothesis list; NUMERICALLY VERIFIED = I reproduced it; ASSUMED = used but not proved here; OPEN; REFUTED.

Status is assigned to **the statement as I restate it**, not as generously as the document words it. Where the document's label overstates, I say so in `does_not_imply`.

| id | statement | status | hypotheses | does_not imply | source |
|---|---|---|---|---|---|
| C1 | Fixed half-output Born average of the single-particle correlation matrix is diagonal-preserving with a uniform off-diagonal contraction: `E[G_E,ii] = G_0,ii`, `E[G_E,ij] = κ_a G_0,ij` (i≠j), `κ_a = 1 − (1−t)² Z_{a−1}(t²)/Z_a(t²)`, `t=e^{−2β}`, `Z_a(q)=Σ_j C(a,j)² q^j`. | PROVED + NUMERICALLY VERIFIED | M=2a lattice; fixed particle number n=a; initial pure Slater state, normalized, `Q†Q=I_a`; fixed output size `|E|=a`; single measurement layer. **No** condition on Fock-amplitude nodes. | Does **not** say individual trajectories have that correlation; does not apply to the *full local* (unconditional-k) protocol — that has `sech²β` exactly (`theory_review.md:449`); does not follow from translational invariance; says nothing about entropy or its corners. | `THEOREMS.md:153–179`; `theory_review.md:178–208`; checked to 4.4e-16 at β∈{0,0.3,1,3,6}, and to 4.4e-16 with a deliberately noded state (one exactly empty row, 35/70 Fock weights zero) |
| C1′ | General (M,n,k) version: `E[G_E,ij] = λ G_0,ij` with `λ = 1 − (1−t)² Z_{M−2,n−1,k−1}(q)/Z_{M,n,k}(q)`, `Z_{M,n,k} = Σ_j C(n,j)C(M−n,k−j)q^j`; and `λ ≥ sech(2β) = 2t/(1+t²)` for `0<k<M`, with `λ=1` at k=0,M. | PROVED + NUMERICALLY VERIFIED | Same, but arbitrary n, k; `0<k<M` for the strict bound. | The `≥ sech(2β)` bound is **not** tight and **not** the same object as κ_a; does not give an upper bound in general. | `theory_review.md:182–194`; my stress test: 1510 (M,n,k,β) combos over M∈{7,9,10,12}, 0 violations, worst margin 6.3e-14 at M=7,n=1,k=6,β=8 |
| C2 | `sech(2β) ≤ κ_a(β) ≤ sech²β` and `κ_a` is non-decreasing in `a`, for the **half-filled, fixed-half-output** family. | PROVED + NUMERICALLY VERIFIED | **This is the family-specific statement**: M=2a, n=k=a, i.e. exactly the protocol under study. The chain does not hold for general (M,n,k) — my test found 552/681 violations when n,k were varied off half-fill at fixed M, which is expected and is not an error in the document, but the theorems file omits the hypothesis. | Does not bound κ in the general (M,n,k) protocol; the lower end `sech(2β)` is `λ₁` only at `a=1`, and is not attained for a>1; monotonicity alone does not give the limit. | `THEOREMS.md:181–188` (hypothesis omitted there); `theory_review.md:220–233` (hypothesis correct); monotone for a=1..599 at β∈{0.05,0.3,1,3,8}, bounds hold at every a |
| C3 | `lim_{a→∞} κ_a(β) = sech²β` at fixed finite β, via `Z_a(t²) ~ (1+t)^{2a+1}/(2√(πat))`. | PROVED (Laplace/steepest-descent, stated not spelled out) + NUMERICALLY VERIFIED | Fixed `t>0`; limit in `a` first. | **Does not commute with the β→∞ limit.** The document says this correctly at `THEOREMS.md:471` but the asympotic itself is only valid for fixed t; at β=3,a=4096 the Laplace form is still off by 6.3e-3 in log (my check). It is an asymptotic, not a bound, and no error term is given. | `THEOREMS.md:189–195`; `theory_review.md:226–233`; verified log-difference 1.7e-2→5.4e-5 over a=16→4096 at β=1 |
| C4 | Jensen: `E[|G_E,ij|²] ≥ κ_a²|G_0,ij|²` for i≠j. | PROVED + NUMERICALLY VERIFIED (strict; see C4′) | Same as C1. Equality iff `G_E,ij` is deterministic across trajectories. | Does **not** mean the mean correlation has an algebraic tail. It bounds the **RMS** correlation, and rare trajectories carry the average — the document says so at `THEOREMS.md:203` and `theory_review.md:301`, correctly. Does not give any bound on `E|G_ij|` itself, nor on `|E G_ij|`. | `THEOREMS.md:197–203`; `theory_review.md:235–244` |
| C4′ | The Jensen gap is **exactly** the trajectory variance: `E|G_E,ij|² − κ²|G_0,ij|² = Var_E(G_E,ij)` for i≠j — because the mean is `κ G_0` exactly (C1), not approximately. | PROVED (one line from C1) + NUMERICALLY VERIFIED | C1. | This is the cleanest statement of "how strict" and it is **not in either document**. See upgradeable U1. | my check: max abs error 1.7e-17 / 8.0e-17 / 7.5e-20 across k∈{2,4,6}, β∈{1,3} |
| C4″ | Strictness quantified: at M=8,n=k=4 the ratio `E|G_ij|²/(κ²|G_0,ij|²)` is ≥1.02 already at β=0.3 and reaches 4.6 at β=3–4; the variance fraction of `E|G_ij|²` goes 0.086 → 0.54 → 0.78 over β=0.3,1,3. β=0 is the only case where Jensen is saturated. | NUMERICALLY VERIFIED (single random state) | — | Single random state, not a distributional statement about honeycomb Dirac. Do not quote as universal. | my check |
| C5 | Growth-order bound: `2κ_a²V_0(R) ≤ E[S₂(R)] ≤ 4 ln2 · V_0(R)`, hence `(κ_a²/2ln2)S_{2,0} ≤ E S₂ ≤ 2 ln2 S_{2,0}`. | PROVED + NUMERICALLY VERIFIED | Pure, particle-number-conserving **Gaussian** initial state; single region R; `2V ≤ S₂ ≤ 4ln2 V` for such states ([Kiefer-Emmanouilidis 2020] eq. 4). The upper half additionally needs the total-variance identity `E V_E = V_0 − Var(m_E)`, which needs only Fock-diagonal A, not Gaussianity (`theory_review.md:54`, correct). | Does **not** fix the leading coefficient; does **not** say anything about `D₂`, which subtracts B and so cannot use the single-region inequality. Does not survive β growing with L: then `κ_a²→0` and the lower constant vanishes. The document states all three caveats (`THEOREMS.md:244–247`). | `THEOREMS.md:226–245`; `theory_review.md:247–262`; all six production (L,β) points satisfy it (`bounds.csv`, `maximum_single_state_bound_violation = 0.0`) |
| C6 | Per-trajectory bound `e^{−4β} S_{2,0}(R) ≤ S_{2,E}(R) ≤ e^{4β} S_{2,0}(R)`, from `|ε_{E,k} − ε_{0,k}| ≤ 4β` and `|d ln s₂/dε| ≤ 1`. | PROVED + NUMERICALLY VERIFIED | Any diagonal single-body filter with weights in `[t,1]`; pure Gaussian state; single region R. **No Born weighting needed** — this is a per-output statement, and the document says so (`theory_review.md:293`). | Does not give a joint bound on the A/C/B combination; explicitly cannot be subtracted (`theory_review.md:293`: "不能将含负权重的角相减组合直接夹逼"). Does **not** say entropy decreases under filtering — the sup is `e^{4β}`, and my exact enumeration at M=10,n=k=5 found 16/252 outputs with `S_E > S_0` at β=0.3 and none at β≥1. The inequality is loose by a factor `e^{8β}`, so it does not constrain monotonicity either way. | `THEOREMS.md:249–284`; `theory_review.md:272–293`; `check_filter_spectrum` in `check_theory.py:139–160` (300 trials, excess 2.3e-12 / 3.1e-15); sup of `|dln s₂/dε|` = 0.999994 over ε∈[1e-6,12] |
| C7 | Corner kernel identity: `D_V^full(E) = Σ_ij |G_{E,ij}|² h_p(dx) h_p(dy) ≥ 0`, `h_p(d) = min(d,L−d,w)/L`, w=pL, after averaging over all L² primitive-cell translations. | PROVED + NUMERICALLY VERIFIED | L×L primitive-cell torus; each region contains **all sublattices** of the selected cells; p=w/L exactly (w=L//den, L divisible by den); region definitions A=w×L, C=L×w, B=w×(L−w) as in `theory_review.md:298–310`. | **Fails for regions that cut through a primitive cell** or select only one sublattice — stated at `theory_review.md:331`. The `h_p(dx)h_p(dy)` form uses `h_p` with the **same** w in both directions even though B's y-extent is L−w; this is nevertheless the exact form (see A1 below). Does not apply to the origin-only `D_V` without translation averaging; the document notes at `theory_review.md:331` that the sub-tile subgroup average is the one that is exactly non-negative per trajectory. | `THEOREMS.md:300–314`; `theory_review.md:298–340`; `check_theory.py:101–136` recomputed by me: weight max error 0 identity, and `direct_D_variance − kernel_D_variance` ≤ 4.4e-16 for (L,p) ∈ {(2,½),(3,⅓),(4,½),(6,½),(6,⅓)} |
| C8 | `E[D_V^full] ≥ κ_a² D_{V,0}^full` for the same geometry, hence `liminf E[D_V]/ln L ≥ sech⁴β · c_{V,0}` when `D_{V,0} = c_{V,0} ln L + O(1)`. | PROVED + NUMERICALLY VERIFIED | C1 (needs `κ` for the **off-diagonal** pairs; the kernel kills i=j since `h_p(0)=0`, which is what makes C1 usable here) + C7 (non-negative kernel, so Jensen can be applied **pairwise**, not to a subtracted combination) + the asymptotic `D_{V,0} ~ c_{V,0} ln L` **assumed as a hypothesis on the initial-state family**, plus the massless-limit caveat at `theory_review.md:358` (μ=10⁻⁸ fixed then L→∞ eventually hits a mass cutoff; the strict statement needs μ=0 or μL→0). | **Does not prove log scaling** — `liminf/ln L > 0` is compatible with `L`, `L ln L`, etc. (`theory_review.md:358`). And it is **not** a statement about `D₂`; the counterexample C9 is exactly the gap. Numerically the bound is 3 orders of magnitude below the data at β=3. | `THEOREMS.md:316–345`; `theory_review.md:342–358` |
| C8′ | Exact refinement (not in the documents): `E[D_V^full] = κ_a² D_{V,0}^full + Σ_ij K_ij Var_E(G_E,ij)`, since the kernel kills the diagonal and C1 makes the mean exact. | PROVED (one line) + NUMERICALLY VERIFIED | C1, C7. | The correction term is **not** negligible: `corr/(κ²D_{V,0})` = 0.23, 1.00, 2.76 at β=0.5,1,3 for L=2,p=½, and 0.28, 1.37, 8.22 for L=3,p=⅓. So at β=3 the Born-averaged D_V is dominated by trajectory fluctuations, and the κ² lower bound is conservative by construction, not accident. This is the quantitative reason the β=3 bound is loose. | my exact enumeration, L=2 (70 outputs) and L=3 (48,620 outputs), identity error ≤ 4.2e-16 |
| C8″ | `c_{V,0} = (4 + 2π/(3√3))/(8π²) = 0.06597528336...` | NUMERICALLY VERIFIED (arithmetic) / ASSUMED (as the correct physical coefficient) | Two Dirac cones, no spin, single-fermion normalization, the 60°/120° corner function of [Estienne–Stéphan–Witczak-Krempa 2021, eq. 9]. | The documents call it a literature baseline and do not claim it — correct. The L=1536 FFT fit gives 0.0659752612 / 0.0659752619 (`baseline/summary.json`), ~3e-8 below the analytic value, and the 1/L-only fit gives 0.0659704, a 7e-6 shift. So the numerical agreement is at the level of the finite-window correction, **not** a 9-digit confirmation. | `THEOREMS.md:330–337`; `theory_review.md:362–366`; my arithmetic check of the closed form |
| C9 | Counterexample to `D₂ ≥ 2D_V`: a pure Gaussian (real Slater) state at L=2, p=½, half-filling, full-translation-averaged, has `D_S2 = 0.0239703297221769`, `D_V = 0.0124652172426887`, `D_S2 − 2D_V = −0.000960104763200586 < 0`, certified in exact rational arithmetic. | PROVED (rational certificate) + NUMERICALLY VERIFIED | L=2, p=½, 2 sublattices, 8 sites, n=4; **full** primitive-cell translation average over all 4 translations; no Born weighting (it is an arbitrary Gaussian state, which is the point). | Refutes only the **universal** implication "S₂≥2V for each region ⇒ D₂≥2D_V". Does **not** refute a born-ensemble-specific inequality, does not refute `D₂ = c₂ ln L`, and says nothing about the user's conjecture. Both documents say this explicitly (`THEOREMS.md:378`, `theory_review.md:384` — "该反例使用非均匀 Gaussian 态；它推翻的是对任意 Gaussian 态直接相减得到的普遍不等式，不声称它已经推翻某个指定蜂窝 Born ensemble 的更特殊定理", plus the `scope` field in the JSON) — this one is correctly labelled. | `exact_corner_counterexample.py`; I recomputed from the stored 8×4 integer matrix: `D_S2 = 0.0239703297221738`, `D_V = 0.0124652172426873`, `gap = −9.601047632007e-4`, matching the JSON to 1e-14; projector idempotence 3.3e-15 |
| C9′ | Certificate logic: `D_S2 = (1/8)ln ℛ`, `ℛ = Π_t P_B(t)²/(P_A(t)P_C(t))`, and the proof is `0<x=ℛ−1<1` plus `ln ℛ ≤ x − x²/2 + x³/3 < 16 D_V`, all rational. | PROVED | The three-term alternating Taylor series bounds `ln(1+x)` from above for `0<x<1` — correct sign pattern (+,−,+, and the remainder after the 3rd term is `+x⁴/4−...≥0`... careful: the alternating series is an *upper* bound when truncated after a **positive** term and a *lower* bound after a negative one. The last retained term here is `+x³/3`, positive, so the truncation is an upper bound for `0<x<1`. Correct.) | Requires `ℛ` to be a rational **purity ratio**, which is why the state had to be real (complex would still be rational in principle but the script uses symmetric `X Xᵀ`). | `exact_corner_counterexample.py:82–90`; the JSON stores `ratio`, `DV`, `rational_positive_margin` as exact fractions |
| C10 | Crossover at `z = a t = L² e^{−2β}`: as `a→∞` with `at=z` fixed, `Z_a((z/a)²) → I₀(2z)`, `P(j=0) → 1/I₀(2z)`, `E j → z I₁(2z)/I₀(2z)`, and `κ_a/t → 2(1 + I₁(2z)/I₀(2z))`. | PROVED (termwise dominated convergence, stated) + NUMERICALLY VERIFIED | Half-filling a=L²; simultaneous limit `a→∞`, `t=z/a→0`; `I₀,I₁` modified Bessel. | **Does not** say the two orders of limit agree — the document correctly notes κ/t → 2 (strong-first) vs 4 (thermal-first) and that unscaled κ→0 either way (`THEOREMS.md:471`). Does **not** identify the crossover as the cause of the D₂ curvature; `THEOREMS.md:475` explicitly declines to claim that. Correctly scoped. | `THEOREMS.md:456–471`; I verified `κ/t` vs `2(1+I₁/I₀)` at z=0.089/0.357/0.803/3.213 for a=36,324,4000 — agreement improves monotonically in a, e.g. z=3.213: 3.2566 → 3.7651 → 3.8311 against 3.8370 |
| C11 | MMSE bound `E V_E ≤ B_lin = V_0 − d²V_0²/V_K`, and the fully explicit `B_count = b[R*/2 − 2V_0] + (4E[j²]/a²)V_0` for the half-filled protocol. | PROVED + NUMERICALLY VERIFIED | Only uses estimators built from `E` (no hidden j); hypergeometric moment formulas need `n,h>1` (stated at `theory_review.md:104`, degenerate cases handled). | Bounds the **number variance**, hence (via C5) the entropy — it is **not** an entropy bound and not a corner bound. `theory_review.md:417–420` draws the entropy consequence `0 ≤ C_R ≤ 2ln2 C_R^bright` correctly and warns at `:423` that it cannot be subtracted. | `theory_review.md:82–160`; `exact_output_check` asserts `hypergeom_variance`, `hypergeom_covariance`, `observable_count_estimator_mse` all <1e-10 across 24 (M,n,k,β) combos, and I re-ran it |
| C12 | Strong-measurement expansion constant `C_R^bright = r(M−r) + 4V_0(R)` and the rank-one node correction `ΔC_R(S) = W_S{f(x)+f(y) − 2[x+y−2xy]}`, with `−2 ≤ f(x)+f(y)−... ≤ 2ln2 − 1`. | ASSUMED HERE / already existing | Full-support outputs; region with `⟨N_R⟩ = r/2`. The document calls this pre-existing (`theory_review.md:388`, "旧 `analysis/born_scaling_strong_measurement.py` 已含完整的固定尺寸 full-support 展开"), and does not re-derive it. | I did **not** independently verify the bracket bounds `[−2, 2ln2−1]` or the numerical coincidence `C_D = 14.6378 vs 14.6346` at `theory_review.md:402–405`; they are auxiliary to the theory layer and labelled as non-novel. Do not count toward novelty. | `theory_review.md:388–423` |

### A1. Note on the kernel derivation (a place where I expected an error and found none)

`theory_review.md:322–331` gives the B-region crossing kernel as `K_B = p(1−p) − (p−h_x)(1−p−h_y)` with a **single** `h_p` in both slots, and then claims the weighted combination leaves `h_x h_y`. Since B is `w × (L−w)`, its y-extent is `L−w`, so a reader will expect `(1−p) − h_p^{(L−w)}(d_y)` in the second factor. I did, and I brute-forced it.

Result: **the document is right and the natural-looking alternative is wrong.** Expanding the document's expression gives `K_B = (1−p)h_x + p h_y − h_x h_y`, and then `(1−p)K_A + p K_C − K_B = (1−p)h_x + p h_y − [(1−p)h_x + p h_y − h_x h_y] = h_x h_y`. Exactly.

Numerics at (L,p) ∈ {(9,⅓),(6,½),(12,¼)}, all four sublattice pairings:

| quantity | max abs error |
|---|---|
| `K_A − h_x` | ≤ 1.1e-16 |
| `K_C − h_y` | ≤ 1.1e-16 |
| `K_B − [(1−p)h_x + p h_y − h_x h_y]` | ≤ 8.3e-17 |
| `(1−p)K_A + p K_C − K_B − h_x h_y` | ≤ 5.6e-17 |
| `K_B − (p−h_x)((1−p) − h_p^{(L−w)})` — the alternative | **fails, up to 2.5e-1** |

The reason the symmetric form works: `p − h_x` is not "fraction of translations with i in B" minus a coverage; it already encodes the one-dimensional complement count for a width-`w` window, and the y-factor `1−p−h_y` is the complement of a width-`w` window inside a width-`L` line, which is the same combinatorial object as B's y-extent because B's y-extent is `L−w` and `f_{L−w} = f_w + L − 2w` (`theory_review.md:324`). My first algebraic reaction was wrong; recording it because a referee will have the same reaction, and the document would benefit from carrying this one line of algebra explicitly rather than asserting "按权重组合后恰好只剩 h_x h_y" (`theory_review.md:331`).

---

## 2. Where status labels overstate

Adversarial pass, most serious first.

### O1. The `sech(2β) ≤ κ ≤ sech²β` box omits its half-fill hypothesis

`BORN_SINGLE_LAYER_THEOREMS.md:181–188` presents the chain under the lead-in "这个因子满足严格的尺寸界，并在固定有限 β 的热力极限趋于简单函数" (`:181`), with no mention of the half-fill condition. κ_a is of course only defined for the half-filled fixed-half-output protocol, so the label is defensible — but the box reads as a generic statement about a "contraction factor", and the sentence at `:197` then uses it to derive the Jensen bound without repeating the hypothesis. `theory_review.md:220–226` does this correctly: it states the chain, then explicitly separates `λ₁` (the a=1 value) from the generic single-exchange amplitude `λ₁` used at `:182`, with the warning at `:226`. The theorems file has no counterpart to that warning. Severity: low, but it is the kind of omission a referee catches.

### O2. "非扰动保护下界" is true and empty at the β the project actually uses

`bounds.csv` gives `mean/lower` for `DV_full`:

| β | L | mean | lower | ratio |
|---|---|---|---|---|
| 1 | 6 | 0.11867 | 0.02193 | 5.4 |
| 1 | 12 | 0.18841 | 0.03197 | 5.9 |
| 1 | 18 | 0.22329 | 0.03715 | 6.0 |
| 3 | 6 | 7.336e-4 | 3.751e-6 | 196 |
| 3 | 12 | 8.344e-3 | 7.976e-6 | 1046 |
| 3 | 18 | 2.916e-2 | 1.356e-5 | 2151 |

The bound is not wrong; it is 3 orders of magnitude away from binding exactly where the interesting physics (β=3 curvature) lives. The mechanism is C8′: at β=3 the variance correction is 8× the κ² term. Neither document claims the bound is tight — `THEOREMS.md:433` says "β=3 的下界很松" — so this is not mislabelling. But the phrase "非扰动保护下界" in the paper-candidate thesis sentence at `THEOREMS.md:509` presents the bound as the main deliverable, and at the β the project actually studies it carries no information. Recommend stating the ratio explicitly next to that sentence.

### O3. The D_V lower bound's `liminf` is stated with `sech⁴β`, while the tabulated finite-L bounds use `κ_{L²}²`

`THEOREMS.md:340–343` boxes `liminf_{L→∞} E[D_V(L,β)]/ln L ≥ sech⁴β · c_{V,0}`. Correct as a `liminf` over L, since `κ_{L²} → sech²β` by C3.

But the production lower-bound column in `bounds.csv` is `κ_a² · DV0` with `a = L²` finite (computed at `analyze_iid_targeted.py:103`, `values["DV_full_lowergap"] = values["DV_full"] - kappa**2 * receipt["DV0_full"]`). Two different numbers, related by `(κ_{L²}/sech²β)⁴`:

| β | L | κ_{L²} | sech²β | κ²/sech⁴ = (κ/sech²β)² |
|---|---|---|---|---|
| 1 | 18 | 0.4190759 | 0.4199743 | 0.99572 |
| 3 | 18 | 0.0080055 | 0.0098660 | 0.6584 |
| 3 | 1000 | 0.0098657 | 0.0098660 | 0.99994 |

Since `κ_a ≤ sech²β` (C2), the boxed `sech⁴β` constant is an **upper** bound on the finite-L coefficient — no contradiction, but a reader comparing §4's box with §6's table will find the numbers differ by up to 34% at β=3 and needs the sentence "the table uses κ_a, the box uses its limit" (this is upgradeable U6).

### O4. `c_{V,0}` agreement is quoted at more digits than the fitting controls

`THEOREMS.md:388` says the two fits give 0.0659752612 and 0.0659752619 "与解析值 0.0659752834 相近", and separately admits the 1/L-only window gives 0.0659704. Both statements are in the file and both are honest. Risk is downstream quoting of "0.06597526" as a 9-digit confirmation of the closed form when the finite-window systematics are 7e-6. Suggest quoting 0.0659753(7) or similar.

### O5. Two counterexamples under one name

`check_theory.py:163–182` saves `corner_inequality_counterexample.npz` (float search, `D_S2−2D_V = −7.1495e-4`); `exact_corner_counterexample.json` is the rational certificate (`−9.6010e-4`). `THEOREMS.md:353–356` quotes the rational numbers; `THEOREMS.md:378` links the rational files; `theory_review.md:384` warns about the `.npz`. All consistent, but the doc set would be safer with only the rational one cited in the main text and the `.npz` demoted to "search artifact".

### O6. `check_theory.py` early-returns on the 12th trial

`search_corner_inequality` returns as soon as `gap < −1e-7` (`check_theory.py:176–180`), so the JSON's `trials: 12` is a stopping time, not a search effort. Combined with `theory_review.md:384`'s remark that a random search finding nothing is not evidence, this is fine — but a reader might read `trials: 12` as "found on the 12th of many" when it is "stopped at 12 because it succeeded". Minor.

### O7. A claim I could not find support for

`THEOREMS.md:203` and `theory_review.md:301`: "若初态具有代数衰减的单体相关，则固定有限 β 下 Born 平均的相关模平方不可能在所有长距离都被纯指数截断." This is presented as a corollary of C4. It follows only if `|G_0,ij|²` has an algebraic tail with a coefficient that survives summation — true for Dirac, and the hypothesis is stated ("若...具有代数衰减的单体相关"), so it is conditional and sound. But it is not verified anywhere and no decay exponent is given. Status: ASSUMED-conditional, correctly hedged. Flagging only because it is the one physics-flavoured sentence in the theory layer and a referee will ask for the exponent.

---

## 3. Upgradeable items

Each is a statement that could be strengthened or made self-contained with modest effort, not new research.

| # | statement | sketch |
|---|---|---|
| U1 | Jensen gap identity: `E|G_E,ij|² − κ_a²|G_0,ij|² = Var_E(G_E,ij)` exactly, off-diagonal. | One line from C1: expand `E|G|² − |E G|²` and substitute `E G = κ G_0`. I verified it to 1e-17. This replaces the vague "and how strict the inequality is" with an exact object, and immediately explains O2 (C8′) and the β=3 looseness. |
| U2 | Exact `E[D_V] = κ_a² D_{V,0} + Σ_ij K_ij Var(G_E,ij)` (C8′), with the ratio `corr/(κ²D_{V,0})` as a computable "tightness diagnostic". | Combine U1 with C7. Verified exactly at L=2,3. Gives the documents a number to quote instead of "很松". |
| U3 | Upgrade the mean-channel theorem to arbitrary `|E|=k` **and** the full local protocol in one statement with the three amplitudes `G_0`, `diag G_0`, and `sech²β`. | `theory_review.md:449` already has the pieces; merges C1 and C1′ and removes O1. |
| U4 | Give the error term in `Z_a ~ (1+t)^{2a+1}/(2√(πat))`, hence a rate for `κ_a → sech²β`. | Standard Laplace with second-order term. Unblocks the "how fast does sech⁴β become valid" question in O3, and the finite-L bound comparison. |
| U5 | Prove the pairwise formulation of Jensen-applicability: since `h_p(0)=0`, the kernel has zero diagonal, so C1's off-diagonal form is sufficient — no diagonal statement needed. | Already implicit in `theory_review.md:342–345`; write it as its own lemma so C8 doesn't appear to need the diagonal-preservation half of C1. |
| U6 | State the `liminf` bound with the explicit finite-L form `E[D_V(L,β)]/ln L ≥ κ_{L²}(β)² · (c_{V,0} + o(1))` and note `κ_{L²}² ≤ sech⁴β` at fixed β — so the boxed `sech⁴β` constant is an **upper** bound on the finite-L coefficient. | Direct from C2 monotonicity. Resolves O3 with one sentence. |
| U7 | Quantify C6's looseness: the ratio `S_E/S_0` at β=1 is bounded by `[e^{−4},e^{4}] = [0.018,55]` while exact enumeration at M=10,n=k=5 gives a max of 1.0 at β≥1 and 1.13 at β=0.3. The bound is `e^{8β}`-loose and empirically the filter is entropy-non-increasing above β≈1. | One small enumeration, already done here; would let the document say "the per-trajectory bound is not the reason mean entropy decreases" with a number. |
| U8 | Make `c_{V,0}` a hypothesis-with-value rather than a quoted fit: state `D_{V,0}(L) = c_{V,0} ln L + O(1)` with `c_{V,0}` from [Estienne et al.], and report the L=1536 check as 0.0659753(7) rather than 9 digits. | Pure rewording plus the error bar, removes O4. |
| U9 | Fold the two counterexamples into one: cite only `exact_corner_counterexample.{py,json}` in the main text; move the `.npz` to a footnote as "initial floating-point search". | Edits only; removes O5. |
| U10 | Add the missing hypothesis to the boxed κ chain at `THEOREMS.md:181`, namely M=2a, n=k=a. | One clause; removes the only real (if minor) overstatement I found. |
| U11 | State C9's scope as a labelled hypothesis in the *theorem statement*, not only in the surrounding prose: "for arbitrary pure Gaussian states; nothing follows for a Born ensemble." | Currently the caveat is prose + JSON field; a formal "Remark (scope)" would survive quotation. |

---

## 4. What is genuinely missing (not upgradeable with modest effort — these are OPEN)

- Any statement about `D₂ = (2S₂(A)+S₂(C))/3 − S₂(B)` beyond "single-region bounds cannot be subtracted". C9 closes one route; no route is open.
- Whether the β=3 curvature in the production data is the `z = L²e^{−2β}` crossover. C10 supplies the mechanism's *existence*; `THEOREMS.md:475` correctly refuses to attribute.
- Any exponent for the algebraic-tail statement in O7.
- Whether the mean-channel structure constrains the **entropy** corner at all. Every tool in the theory layer acts on `V` (quadratic) or on single-region `S₂` (with an `e^{±4β}` slack); nothing acts on the combination.

---

## 5. Reproduction commands used in this audit

All read-only, all deterministic, none archived:

```text
python -c "..."   # C1/C1' channel identity at (M,n,k,beta) grid; 4.4e-16
python -c "..."   # C4 Jensen gap == Var, 1e-17
python -c "..."   # C8' E[DV] decomposition, L=2 (70 outputs), L=3 (48620 outputs)
python -c "..."   # C7 kernel identity re-derived from scratch, (L,p) grid
python -c "..."   # C9 recompute from stored integer matrix
python -I check_theory.py   # full project self-check, PASS, 24 exact checks + 5 geometry + 300 filter trials
```

`check_theory.py` re-run in an isolated copy: `status PASS`, geometry `variance_contraction_error` ≤ 4.4e-16 at all five (L,p), filter spectrum excess 3.1e-15, corner search stopping at trial 12 with `D_S2−2D_V = −7.15e-4`. Matches the committed `theory_checks.json` up to RNG-free reconstruction.
