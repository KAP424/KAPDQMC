# Literature positioning: single-layer Born-averaged entanglement of a 2D Dirac ground state

Audit date: 2026-10-30. Read-only audit of `src/Measure/born_ai/`. Scope: results published or publicly known **before January 2026** (the manuscript's target literature). Nothing newer is reported here.

Target question, stated precisely: *does the corner log coefficient of the second Rényi entropy of a 2D Dirac ground state survive one Born-averaged weak density-measurement layer?*

---

## 1. The five citations in the project, checked

| arXiv | Actual title / venue | Verdict |
|---|---|---|
| 2301.08268 | Weinstein, Sajith, Altman, Garratt, *Nonlocality and entanglement in measured critical quantum Ising chains*, **PRB 107, 245132 (2023)** | Accurate and relevant. |
| 2301.08255 | Yang, Mao, Jian, *Entanglement in one-dimensional critical state after measurements*, **PRB 108, 165120 (2023)** | Accurate and relevant. |
| 2305.20092 | Garratt, Altman, *Probing postmeasurement entanglement without postselection*, **PRX Quantum 5, 030311 (2024)** | Accurate. arXiv ID is 2305.20092, not 2301.x — the project cites only the journal ref, so no error. |
| 2102.06223 | Estienne, Stéphan, Witczak-Krempa, *Cornering the universal shape of fluctuations*, **Nat. Commun. 13, 134 (2022)** | Accurate and relevant. |
| 2003.03112 | Kiefer-Emmanouilidis, Unanyan, Sirker, Fleischhauer, *Bounds on the entanglement entropy by the number entropy in non-interacting fermionic systems*, **SciPost Phys. 8, 083 (2020)** | Accurate, but see the caveat in §6. |

Where the project uses them:

- `BORN_SCALING_RESULTS.md:68` — Weinstein and Yang-Mao-Jian as precedent that the Born-averaged log coefficient is measurement-independent in 1D. Correctly hedged in the same line ("它们支持研究这一假说，但不是…证明").
- `BORN_SINGLE_LAYER_THEOREMS.md:220, :502` — Kiefer-Emmanouilidis for the entropy/number-fluctuation bound.
- `BORN_SINGLE_LAYER_THEOREMS.md:337, :504` — Estienne et al. Eq. (9) for the Dirac number-fluctuation corner function `c_{V,0} = 0.0659753`.
- `BORN_SINGLE_LAYER_THEOREMS.md:505` — Garratt-Altman for the averaged-channel vs. nonlinear-trajectory distinction.
- `BORN_SCALING_RESULTS.md:44, :399` — Liao-Song-Zhao-Meng (2404.13876) for the smooth-minus-cornered subtraction method.

**Net: all five are real, correctly attributed, and used no more strongly than the source supports.** That is better than the usual baseline. The problems are omissions, not misquotations — see §5.

---

## 2. Q1 — one measurement layer on a critical ground state, Born-averaged entropy

The literature has essentially one system solved: 1D, and it splits cleanly by *what is being measured* and *how the outcomes are treated*.

**Weinstein et al. (2301.08268, PRB 107, 245132).** Critical transverse-field Ising, measurements at finite *spatial density*, restricted to parity-symmetric measurements. Measurements appear as (1+0)-dimensional defects in (1+1)D Euclidean spacetime. Core result, from the abstract: the replica defect "is an irrelevant perturbation to the Ising CFT. Strikingly, the asymptotic scalings of averaged correlations and entanglement entropy are therefore **unchanged** relative to the ground state." By contrast the defect from *postselecting the most likely outcomes* "is exactly marginal," and the effective central charge then varies continuously with measurement density.

**Yang, Mao, Jian (2301.08255, PRB 108, 165120).** Critical TFIM, transverse-spin weak measurements. Uniform-outcome post-measurement states keep log scaling with an effective `c_eff(β)` derived analytically. For the **Born-rule average over all outcomes**, "the numerically extracted effective central charge appears to be **independent of the measurement strength**" — and the authors flag this as contrary to the usual expectation that local measurements reduce entanglement.

Both are 1D, both are a *single round* of measurements on a ground state (not a time series), so both are structurally the right comparison class. Both say: **Born average protects the coefficient; postselection/forcing does not.** That is a real and reproducible 1D fact.

**2D free fermions / Dirac: no direct result was found.** Repeated arXiv queries (`abs:` combinations of *measurement*, *entanglement entropy*, *2+1*, *free fermions*; *corner* + *measurement*) returned no paper computing the Born-averaged corner coefficient of a 2D Dirac ground state after a measurement layer. The nearest neighbours are:

- **Tang & Wen, 2411.13705** (Nov 2024), massless free fermions, 1D. Power-law correlations and log-growing EE *do* persist after weak measurement, but "the conformal symmetry is lost and the entanglement Hamiltonian generally becomes gapped for arbitrary small measurement strength." The eigenfunctions of the entanglement Hamiltonian kernel are unchanged; the spectrum is not. Multi-interval EE shares the same effective central charge as the single interval.
- **Lee, Jian, Xu, 2301.05238** (PRX Quantum 4, 030317), (2+1)D O(N) Wilson-Fisher under decoherence/weak measurement. With post-selection: two new boundary/defect criticalities, possibly "extraordinary-log." Without post-selection: local correlators unchanged, but nonlocal operators (disorder operator) change qualitatively, and decoherence can confine. Note this is a *decohered density matrix*, not a Born-averaged ensemble — different object.
- **Khanna & Vasseur, 2508.02788** (arXiv Aug 2025; PRL 136, 160402). Tomonaga-Luttinger liquids, local charge measurement. Measurement-induced entanglement computed exactly by replica, and given "a natural interpretation in terms of Born averaging over conformally-invariant boundary conditions." They stress MIE for physical measurements "is fundamentally different from the entanglement induced by forcing measurement outcomes." This is the most rigorous existing statement of *why* Born averaging and forcing differ, and it is 1D.

**Answer to Q1.** For 1D CFT the Born-averaged post-measurement log coefficient is known, in specific symmetry classes, to equal the ground-state value. For 2D free fermions/Dirac, nothing is known — the question the project is asking is genuinely open in the literature. It is not, however, un-adjacent: Tang-Wen supplies the mechanism by which it could fail.

---

## 3. Q2 — is the log/corner coefficient protected, renormalized, or destroyed?

Three distinct statements must be kept apart, and the literature separates them more sharply than the project does.

1. **A log term survives.** Well supported. Tang-Wen: log EE persists at arbitrarily small measurement strength in 1D free fermions, even though the entanglement Hamiltonian is gapped. Weinstein/Yang-Mao-Jian: log persists with unchanged (Born) or renormalized (forced) coefficient.
2. **The coefficient is unchanged.** Supported only for 1D Born averages, and only in the measurement classes studied (parity-symmetric; transverse spin). Renormalized/continuous variation is the result for forced or postselected outcomes.
3. **The state is still the same CFT.** *This is where the project's hypothesis is weakest, and where the literature says no.* Tang-Wen's title is the claim: "A critical state under weak measurement is not critical." Conformal symmetry is lost; the entanglement Hamiltonian is gapped and long-ranged even for a single interval.

Statement 3 matters because a *corner* coefficient `a_α(θ)` is not like a 1D central charge. In 1D the coefficient of `ln L` is fixed by the algebra of the stress tensor and survives mild deformations. In 2D, `a_α(θ)` is defined as the coefficient of `ln L` in the CFT expansion of the entangling-surface term; it is a CFT data object (for `α→1`, `θ→π` it reduces to `C_T`). If the post-measurement state is not a CFT — which Tang-Wen says it is not, already in 1D free fermions — then there is no theorem forcing `a_2(θ)` to keep its Dirac value, even if *some* log term survives. The log and the coefficient are logically independent here.

**Exact free-fermion results.** None were found for the single-layer Born-averaged corner coefficient in 2D. The exact free-fermion results that do exist are for the *unmeasured* state: `a_α(θ)` for free Dirac and free boson in 2+1D, from **Helmes, Hayward Sierens, Chandran, Witczak-Krempa, Melko, 1606.03096 (PRB 94, 125142)** — lattice numerics plus field-theory estimates and new exact results for general `α, θ`. This is the canonical source for the number the project puts in its ansatz, and it is **not cited anywhere in the project** (see §6).

---

## 4. Q3 — Gaussian / Slater states under product / diagonal filters

Setup in the project matches the literature's tractable case exactly: `M_E = t^{N_E}` with `t = e^{-2β}` is a product of on-site diagonal filters, number-conserving, so every normalized outcome is a pure Slater determinant. Three things are true:

**(a) Each outcome is exactly computable.** `S_2` from the eigenvalues of the restricted correlation matrix `C_{E,R}`. No approximation needed per sample; the cost is the average, not the sample.

**(b) The Born-averaged entropy is NOT a function of the averaged correlation matrix.** `S_2` is concave in `C_R`, so `E[S_2(C_E)] ≠ S_2(E[C_E])` and in fact `E[S_2] ≤ S_2(E[C])`. The project states this correctly at `BORN_SINGLE_LAYER_THEOREMS.md:141` ("平均的是每个测量态的熵，**不使用平均相关矩阵的熵代替 Born 平均熵**") and again at `BORN_SCALING_RESULTS.md:111`. The literature agrees and makes it a design principle: **Garratt-Altman (PRX Quantum 5, 030311)** build cross-correlations precisely because the measurement-averaged entropy is not recoverable from averaged state data, and they give both upper and lower bounds on the measurement-averaged von Neumann entropy rather than an equality.

**(c) The averaged first moment *is* exactly computable, and the project already has it.** `research_20261029/theory_review.md:197` gives the exact single-particle decoherence factor `E_E G_{E,ij} = λ_1 G_{0,ij}` for `i≠j`, with the rigorous bound `sech(2β) ≤ λ_a ≤ sech²β` and `lim_{a→∞} λ_a = sech²β`. This is a genuine closed-form free-fermion result. It is, however, a statement about the *mean* correlation matrix — exactly the object (b) says is insufficient. Its value is as a rigorous *bound* generator (via concavity and Jensen), not as a route to the averaged entropy.

**Answer to Q3.** Averaged entropy is computable in principle and in practice by sampling; it is *not* computable from the averaged correlation matrix, and the averaged correlation matrix is *not* sufficient. No closed-form Born-averaged Rényi entropy for a 2D critical Gaussian state under diagonal filters was found in the literature. The project's `λ_a` result appears to be novel but, by construction, bounds-only.

---

## 5. Q4 — continuous monitoring of 2D free fermions

State of the art, all with dynamics (unitary evolution *plus* monitoring), which is a different problem from one measurement layer:

- **Chahine & Buchhold, 2309.12391 (PRB 110, 054313).** 2D, U(1), continuous monitoring. Exact replica quantum master equation → either an SU(R) NLSM of symmetry class AIII in (2+1)D, or third quantization to a non-Hermitian SU(2R) Hubbard model. Numerics: **weak monitoring gives `L log L` entanglement growth** with multifractal `D_q = 2` ("resembling a metallic Fermi liquid"); **strong monitoring gives area law**; and between them a high-symmetry point with both emergent conformal invariance and maximal multifractality.
- **Fava, Piroli, Swann, Bernard, Nahum, 2302.12820 (PRX 13, 041045).** Multi-flavor Majorana chain, monitored parity. Controlled NLSM derivation; the nontrivial phase has `S ~ (log L)²` — a different exponent from Chahine-Buchhold's `L log L`, and the authors note the contrast with previously studied models.
- **Fan, Yin, García-García, 2508.18468** (arXiv Aug 2025). GPU, `160×160` 2D, projective and homodyne. Finite-rate MIPT in 2D for both protocols, scale-invariant mutual information, `ν ≈ 1.3`; in 1D `L ~ 10000` is needed to see the absence of an MIPT.

**Answer to Q4.** Yes, there is a measurement-induced transition in 2D monitored free fermions, and there are competing claims about the subleading form (`L log L` vs `(log L)²`). But **none of this transfers to the project's problem**, and the project says so itself at `BORN_SCALING_RESULTS.md:15` and `BORN_SINGLE_LAYER_THEOREMS.md:247`. The `L log L` of Chahine-Buchhold is not the 2D CFT corner log and must not be conflated with `a_2(θ) ln L`. Continuous-monitoring scaling describes a steady state built from `∝ L` measurement layers; the project applies **one**.

---

## 6. Gaps and one substantive concern

**Gap 1 — the canonical corner-coefficient source is missing.** `BORN_SCALING_RESULTS.md:40-41` asserts `c_B^{(0)} = 2a_2(π/3) + 2a_2(2π/3) ≈ 0.155` and attributes the *method* to 2404.13876. But `a_α(θ)` for the free Dirac fermion — the actual numbers being combined — come from **Helmes et al., 1606.03096**, which is never cited. The project's own line 21 already downgrades 0.155 to "数值参考估计，不是精确解析值" (a finite-size fit, not exact), so the missing citation is the thing that would let a reader check the value against a published table.

**Gap 2 — the one paper that most directly bears on the hypothesis is not cited.** Tang & Wen, 2411.13705, "A critical state under weak measurement is not critical," is the sharpest existing challenge to the mechanism the project invokes. It should be cited and answered, not omitted. It does not refute the project's *data*; it undercuts the *reasoning* in lines 46-54.

**Gap 3 — the load-bearing power-counting step is unverified.** Lines 46-54 argue that the replica defect coupling has `y_g = 2 − 2Δ_O = −2` with `Δ_O = 2` for the Dirac density bilinear, hence irrelevant. Under the codimension-1 convention `y_g = (d−1) − 2Δ_O` with `d = 3`, this arithmetic is internally consistent. But the step is doing all the work for the hypothesis, and three things are nowhere checked in the files: (i) that `Σ_{r≠s} O_r O_s` is the leading defect operator rather than a single-replica term; (ii) that `O_r O_s` has dimension `2Δ_O` with no operator mixing; (iii) that the fixed-particle-number constraint does not change the counting — line 64 asserts the global mode cancels but does not show what the defect action then contains. Weinstein et al. had to do a genuine one-loop replica calculation to conclude "irrelevant" in 1D; a tree-level count is not the same standard of proof. This should be labelled a heuristic, as `BORN_SCALING_RESULTS.md:66` in fact already does — the issue is that lines 49-54 read more confidently than that.

**Minor — Kiefer-Emmanouilidis attribution.** `BORN_SINGLE_LAYER_THEOREMS.md:220` cites "式（4）" for `S_2 ≥ 2V`. What 2003.03112 actually establishes, per its abstract, is that `ln S^(2)` is tightly bounded by the Rényi *number* entropy `S^(2)_N = −ln Σ_n p²(n)`. That is a different inequality. The project also says the `S_2 ≥ 2V` bound "可由区域相关矩阵本征值逐项证明" (provable itemwise from the eigenvalues), i.e. it is a self-contained free-fermion fact. Recommendation: present it as such and demote the citation to "see also," rather than implying Eq. (4) is the proof.

**Minor — Estienne et al. use is correct but narrow.** `BORN_SINGLE_LAYER_THEOREMS.md:337` uses Eq. (9) for the number-fluctuation corner function. The abstract's "super-universal" claim (angular dependence independent of everything up to a prefactor, for uniform, isotropic systems with not-too-slow decay) is exactly the property being relied on, and it is correctly scoped there. No change needed.

---

## 7. Verdict on the assertions in "从物理上期待什么"

Section is `BORN_SCALING_RESULTS.md:30-71`. Each assertion typed below.

| # | Line | Assertion | Status |
|---|---|---|---|
| A1 | 32 | Area term can change strongly with β while the asymptotic corner log coefficient keeps its ground-state value | **Not supported.** No 2D result exists; the 1D evidence is for a different invariant (`c`, not `a_2(θ)`); Tang-Wen removes the CFT that `a_2(θ)` is defined relative to |
| A2 | 35-37 | `S̄_2 = α L − c ln L + γ + δ` | **Supported as a ground-state form** (standard CFT expansion, Helmes et al.). As a post-measurement ansatz it is an assumption |
| A3 | 40-41 | `D̄ = c_B^(0) ln L + γ_D + δ`, `c_B^(0) ≈ 0.155` | **Partly.** The combination and the `a_2(θ)` values are literature (Helmes 1606.03096 — uncited). 0.155 itself is a finite-size fit, correctly flagged as such at line 21 |
| A4 | 44 | Smooth-minus-cornered subtraction extracts the corner term; method does not prove the fixed point survives | **Supported.** Correctly scoped, correctly attributed to 2404.13876 |
| A5 | 46-54 | Defect coupling has `Δ_O = 2`, `y_g = −2`, irrelevant at tree level | **Not supported as a proof.** Internally consistent arithmetic, but unverified identification of the leading operator and no loop-level check; see Gap 3 |
| A6 | 56-64 | Exact POVM identity over the fixed-`N` sector; fixed output number correlates outcomes; global mode cancels at small β | **Supported as internal algebra**, verified by the project's own exact enumeration (`research_20261029/theory_review.md` §1, `check_theory.py`, 24 configuration sets). Not a literature claim |
| A7 | 66 | Not proved for arbitrary finite β; higher orders, replica limit, corner operator all unchecked; strong β may leave the basin | **Supported and commendably explicit.** This is the correct epistemic status |
| A8 | 68 | Weinstein and Yang-Mao-Jian as precedent | **Supported.** Both citations accurate, both 1D, both correctly hedged in the same sentence |
| A9 | 70 | Regime is `μL ≪ 1`; fixed nonzero μ makes the initial state gapped as `L→∞` | **Supported.** Straightforward Dirac-dispersion statement, correctly conservative |

**Bottom line.** The section is unusually well-hedged — A4, A7, A8, A9 are all sound and A7 pre-empts most of what an audit would say. The unsupported content is concentrated in A1 and A5, which are also the only places the hypothesis is actually *argued* rather than *stated*. The single highest-value literature addition is Tang & Wen (2411.13705), because it attacks exactly A1's implied mechanism while leaving the project's numerical programme untouched.

---

## 8. Bibliography

Primary, all pre-January-2026:

- Z. Weinstein, R. Sajith, E. Altman, S. J. Garratt, *Nonlocality and entanglement in measured critical quantum Ising chains*, [arXiv:2301.08268](https://arxiv.org/abs/2301.08268), Phys. Rev. B **107**, 245132 (2023).
- Z. Yang, D. Mao, C.-M. Jian, *Entanglement in one-dimensional critical state after measurements*, [arXiv:2301.08255](https://arxiv.org/abs/2301.08255), Phys. Rev. B **108**, 165120 (2023).
- S. J. Garratt, E. Altman, *Probing postmeasurement entanglement without postselection*, [arXiv:2305.20092](https://arxiv.org/abs/2305.20092), PRX Quantum **5**, 030311 (2024).
- B. Estienne, J.-M. Stéphan, W. Witczak-Krempa, *Cornering the universal shape of fluctuations*, [arXiv:2102.06223](https://arxiv.org/abs/2102.06223), Nat. Commun. **13**, 134 (2022).
- M. Kiefer-Emmanouilidis, R. Unanyan, J. Sirker, M. Fleischhauer, *Bounds on the entanglement entropy by the number entropy in non-interacting fermionic systems*, [arXiv:2003.03112](https://arxiv.org/abs/2003.03112), SciPost Phys. **8**, 083 (2020).

Additional, recommended by this audit:

- Q. Tang, X. Wen, *A critical state under weak measurement is not critical*, [arXiv:2411.13705](https://arxiv.org/abs/2411.13705) (2024). **Highest priority addition.**
- J. Helmes, L. E. Hayward Sierens, A. Chandran, W. Witczak-Krempa, R. G. Melko, *Universal corner entanglement of Dirac fermions and gapless bosons from the continuum to the lattice*, [arXiv:1606.03096](https://arxiv.org/abs/1606.03096), Phys. Rev. B **94**, 125142 (2016). Source of `a_α(θ)`.
- J. Y. Lee, C.-M. Jian, C. Xu, *Quantum criticality under decoherence or weak measurement*, [arXiv:2301.05238](https://arxiv.org/abs/2301.05238), PRX Quantum **4**, 030317 (2023).
- K. Chahine, M. Buchhold, *Entanglement phases, localization and multifractality of monitored free fermions in two dimensions*, [arXiv:2309.12391](https://arxiv.org/abs/2309.12391), Phys. Rev. B **110**, 054313 (2024).
- M. Fava, L. Piroli, T. Swann, D. Bernard, A. Nahum, *Nonlinear sigma models for monitored dynamics of free fermions*, [arXiv:2302.12820](https://arxiv.org/abs/2302.12820), Phys. Rev. X **13**, 041045 (2023).
- K. Khanna, R. Vasseur, *Measurement-induced entanglement in conformal field theory*, [arXiv:2508.02788](https://arxiv.org/abs/2508.02788), PRL **136**, 160402 (2026). Public from Aug 2025.
- Y. D. Liao, M. Song, J. Zhao, Z. Y. Meng, *Extracting universal corner entanglement entropy during the quantum Monte Carlo simulation*, [arXiv:2404.13876](https://arxiv.org/abs/2404.13876), Phys. Rev. B **110**, 235111 (2024).
- B. Fan, C. Yin, A. M. García-García, *Entanglement dynamics of monitored noninteracting fermions on graphics processing units*, [arXiv:2508.18468](https://arxiv.org/abs/2508.18468) (Aug 2025).
- P. Bueno, W. Witczak-Krempa, *Bounds on corner entanglement in quantum critical states*, [arXiv:1511.04077](https://arxiv.org/abs/1511.04077), Phys. Rev. B **93**, 045131 (2016).
