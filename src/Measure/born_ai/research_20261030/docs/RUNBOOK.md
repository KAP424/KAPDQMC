# RUN-BOOK — Born-trajectory Rényi-2 entropy scaling, `src/Measure/born_ai`

Audit date 2026-10-30. All original files were read only; nothing outside
`research_20261030/` was created or modified. Numbers marked **[measured]** were
produced during this audit on this machine (Julia 1.12.1, `BLAS.set_num_threads(1)`,
16 logical cores). Numbers marked **[recorded]** are quoted from receipts already
in the repository. Where a quantity is absent from the files, this document says so.

---

## 0. Environment

| item | value | evidence |
|---|---|---|
| Julia | 1.12.1 | `julia --version`; `julia_version = "1.12.1"` in every receipt |
| Python | `D:/app/conda/python.exe`, CPython 3.13.5, numpy 2.5.1, matplotlib 3.11.1 | `[-I -c ...]` |
| Julia project | `/d/JuliaDQMC/code/KAPDQMC/Project.toml` (`name = "KAPDQMC"`) | `Project.toml:1` |
| repo-root cwd | `D:/JuliaDQMC/code/KAPDQMC` | launchers use `cwd=ROOT.parents[2]` |

`--project=.` is **mandatory** for every Julia entry point here. `KAPDQMC` is not
in the default `@v1.12` environment: `julia -e 'using KAPDQMC'` fails with
`ArgumentError`, while `julia --project=. -e 'using KAPDQMC'` succeeds. **[measured]**

`--project=@v1.11` (the memory note) is **irrelevant here**. It resolves to
`C:\Users\admin\.julia\environments\v1.11\Project.toml` and exists only for
DataFrames; no script in `born_ai/` uses DataFrames, CSV.jl or DelimitedFiles from
that environment. `--project=.` and `--project=@.` both resolve to the same
`D:\JuliaDQMC\code\KAPDQMC\Project.toml` from the repo root. **[measured]**

---

## 1. The command that produces N independent Born-averaged S2(A), S2(B), S2(C)

### 1.1 There is no CLI knob for L or beta

`run_iid_targeted.jl` hard-codes the design in `prepare()`:

```julia
# research_20261029/run_iid_targeted.jl:29-38
for L in (18,12,6)
    m=ground_model(L;state=:staggered,mu=1e-8)
    ...
    for (bi,beta) in enumerate((1.,3.)),batch in 1:4
        seed=729_102_900+10000L+100bi+batch
        id="L$(L)_beta$(Int(beta))_batch$(batch)"
        push!(jobs,Dict("id"=>id,"L"=>L,"beta"=>beta,"batch"=>batch,"seed"=>seed,"records"=>512,...
```

The only CLI surface is `--prepare`, `--worker=N` and `--finalize`
(`run_iid_targeted.jl:192-196`). L ∈ {6,12,18}, beta ∈ {1,3}, 4 batches × 512
records = 2048 samples per parameter point. There is **no** `--L=` / `--beta=`
argument anywhere in the file.

### 1.2 Reproduce the frozen experiment (the canonical N-sample run)

```bash
cd /d/JuliaDQMC/code/KAPDQMC
julia --project=. src/Measure/born_ai/research_20261029/run_iid_targeted.jl --worker=1
julia --project=. src/Measure/born_ai/research_20261029/run_iid_targeted.jl --worker=2
julia --project=. src/Measure/born_ai/research_20261029/run_iid_targeted.jl --worker=3
julia --project=. src/Measure/born_ai/research_20261029/run_iid_targeted.jl --worker=4
julia --project=. src/Measure/born_ai/research_20261029/run_iid_targeted.jl --finalize
```

This is **idempotent and read-only when the results already exist**. I ran worker 1
and worker 2 during this audit; each printed `SKIP ...` for its six jobs and exited
0 **[measured]**. `--worker` must be in `1:4` (`run_iid_targeted.jl:123`); the
worker index *is* the batch index (`run_iid_targeted.jl:126`), so four processes
give four-fold parallelism. Note `--prepare` will refuse in place:

```julia
# research_20261029/run_iid_targeted.jl:27
ispath(OUT) && error("targeted directory exists; preserve existing evidence")
```

and `--finalize` will refuse because `iid_targeted/manifest.toml` already exists
(`run_iid_targeted.jl:185`). To re-run from scratch you must move
`research_20261029/iid_targeted/` aside first.

### 1.3 Arbitrary L and beta — the verified minimal driver

Because no CLI takes L/beta, the smallest correct path for a new (L, beta) is a
~15-line driver over the library. This one was executed during the audit at
L=6, beta=1, N=100 and produced the expected CSV and means **[measured]**:

```julia
# run from the repo root: julia --project=. thisfile.jl
using LinearAlgebra, Random
BLAS.set_num_threads(1)
const BA = joinpath(@__DIR__, "src", "Measure", "born_ai")
include(joinpath(BA, "research_20261029", "StratifiedBorn.jl"));  using .StratifiedBorn
include(joinpath(BA, "original_swap_scaling.jl"));                using .OriginalSwapScaling

L, beta, N = 6, 1.0, 100
m = ground_model(L; state=:staggered, mu=1e-8)      # original_swap_scaling.jl:26
s = StratifiedSampler(m.Q, beta)                    # StratifiedBorn.jl:22
rng = MersenneTwister(20261030)
w = EntropyWorkspace(m)                             # original_swap_scaling.jl:183
acc = zeros(3); acc2 = zeros(3)
println("L,beta,sample,j,SA,SB,SC,D")
for k in 1:N
    d = sample_stratum(s, 0, m.n; rng)              # StratifiedBorn.jl:52
    e = entropy!(w, m, d.E, beta)                   # original_swap_scaling.jl:192
    acc .+= (e.SA, e.SB, e.SC); acc2 .+= (e.SA, e.SB, e.SC).^2
    println("$L,$beta,$k,$(d.j),$(e.SA),$(e.SB),$(e.SC),$(e.D)")
end
mu = acc/N; se = sqrt.(max.(acc2/N .- mu.^2, 0)/N)
println(stderr, "mean A,B,C = $mu   SE = $se")
```

Observed at L=6, beta=1, N=100: mean A,B,C = `[2.0171, 1.7106, 2.0316]`,
SE = `[0.0353, 0.0370, 0.0372]`, region sizes `(24, 16, 24)` **[measured]**.

`jlo=0, jhi=m.n` is the *unstratified* call: it sums the full analytic j-weights
(`StratifiedBorn.jl:57`), so it is the unbiased i.i.d. sampler and matches
`run_iid_targeted`'s `sample_stratum(s,0,m.n;rng)` at `run_iid_targeted.jl:143`.

### 1.4 One-liner alternative (legacy path — see the warning)

```bash
cd /d/JuliaDQMC/code/KAPDQMC
julia --project=. -e 'using LinearAlgebra; BLAS.set_num_threads(1);
  include("src/Measure/born_ai/Free_M.jl");
  r = Free_Measure_EE(6, 1.0, 50; sampler=:iid, output_dir=joinpath(tempdir(),"run1"));
  println(r.entropy)'
```

Verified working **[measured]**. **But `Free_M.jl` gives only two regions, A and B
(`Free_M.jl:137-138`, `Free_M.jl:201-203`, header text at `Free_M.jl:217`), and it
builds the ground state with `LAPACK.syevd!` on the **massless** K
(`Free_M.jl:142,146-147`) — i.e. the legacy `state=:lapack`, `mu=0`, LAPACK-chosen
zero modes. It is *not* the `:staggered, mu=1e-8` state used by all production
data. Do not mix the two.

### 1.5 Exact enumeration at L=3

```bash
cd /d/JuliaDQMC/code/KAPDQMC
julia --project=. -e 'using LinearAlgebra; BLAS.set_num_threads(1);
  include("src/Measure/born_ai/original_swap_scaling.jl"); using .OriginalSwapScaling;
  include("src/Measure/born_ai/ExactBorn.jl"); using .ExactBorn;
  m = ground_model(3; state=:staggered, mu=1e-8);
  ex = born_trajectory_entropy(m.Q, 1.5, Tuple(values(region_indices(3)));
       ensemble=:fixed_half, keep_trajectories=true, max_configs=1_000_000);
  println(ex.nconfigurations, " ", ex.mean_entropy, " ", ex.normalization_error)'
```

Verified: `nconfigurations = 48620`, `mean_entropy = [0.149739, 0.119603, 0.149739]`,
`normalization_error = 8.88e-16`, 1.06 s wall **[measured]**.

### 1.6 Boundary / translated geometry (p=1/3 and p=1/2)

```bash
cd /d/JuliaDQMC/code/KAPDQMC
julia --project=. src/Measure/born_ai/run_born_boundary_check_fast.jl \
      --plan=src/Measure/born_ai/data/born_scaling/boundary_check/production/plan.toml \
      --run-id=production_boundary_L18_beta3_chain1
```

This is exactly what the launcher constructs:
`launch_born_boundary_production.py:26-27`,
`cmd=[shutil.which('julia'),'--project=.',str(ROOT/'run_born_boundary_check_fast.jl'),'--plan='+..., '--run-id='+...]`
with `cwd=ROOT.parents[2]` and `JULIA_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1,
MKL_NUM_THREADS=1` (`launch_born_boundary_production.py:17`). It refuses to
overwrite an existing chain (`run_born_boundary_check.jl:94`).

---

## 2. Signatures

### 2.1 Sampler

```julia
StratifiedSampler(P::AbstractMatrix{<:Real}, beta::Real)   # StratifiedBorn.jl:22
IIDSampler(P::AbstractMatrix{<:Real}, beta::Real)          # BornSampler.jl:37
```

Both require `Nsites == 2*Nparticles` (`BornSampler.jl:42`) and a full-column-rank
`P` (`BornSampler.jl:46-47`). `StratifiedSampler` wraps an `IIDSampler` and adds
`log_jweights`, `log_Z` (`StratifiedBorn.jl:22-26`).

```julia
sample_stratum(s, jlo::Integer, jhi::Integer=jlo; rng) -> (; F, E, j)  # StratifiedBorn.jl:52-62
sample_fixed_j(s, j::Integer; kwargs...)                              # StratifiedBorn.jl:64
sample_measurement(s::IIDSampler; rng) -> sorted E                    # BornSampler.jl:101
stratum_probability(s, jlo, jhi=jlo) -> Float64                       # StratifiedBorn.jl:34
record_digest(F, E, j) -> sha256 hex                                  # StratifiedBorn.jl:66
```

`E` (size `a = L²`) is the **suppressed** set; `S = setdiff(1:N, E)` is the
observed set. `j = |E ∩ F|` is the latent stratum index, `0 ≤ j ≤ a`;
`F` is the auxiliary projection-DPP set and **must not** be treated as the
post-measurement state (`BornSampler.jl:100`, `StratifiedBorn.jl:5`).

### 2.2 Entropy routine

```julia
EntropyWorkspace(model)                    # original_swap_scaling.jl:183-189
entropy!(work, model, E, beta)
  -> (; SA, SB, SC, A_minus_B, D)          # original_swap_scaling.jl:192-210
```

Returns natural-log Rényi-2 entropies of regions A, B, C and
`D = (2·SA + SC)/3 − SB` (`original_swap_scaling.jl:209`). One weighted orbital QR
(`LAPACK.geqrf!`/`orgqr!`, lines 196-197) then three regional purity Cholesky
factorizations (lines 199-208).

`run_iid_targeted.jl` adds its own reference and diagnostic layer:

```julia
reference_entropy(m, E, beta, regions) -> [SA, SB, SC, D]   # run_iid_targeted.jl:107-111
diagnostics!(G, Gsquare, work, kernel, kappa) -> 24-vector    # run_iid_targeted.jl:93-105
geometry(m) -> (; Kx, Ky, Kfull, Ksub, N0, V0, off, offnorm) # run_iid_targeted.jl:77-91
```

`reference_entropy` recomputes the same four numbers independently through
`ExactBorn.regional_entropy` (eigenvalue route) and the run aborts if they
disagree by more than 1e-9 (`run_iid_targeted.jl:150-153`).

### 2.3 One output CSV row

Header at `run_iid_targeted.jl:14`; 33 columns. Verified sample value from
`research_20261029/iid_targeted/L6_beta1_batch1.csv` row 1 **[measured]**:

| # | column | example | meaning |
|---|---|---|---|
| 1-5 | `L,beta,batch,sample,seed` | `6,1.0,1,1,729163001` | provenance |
| 6 | `j` | `4` | latent stratum `\|E∩F\|` |
| 7 | `F` | `2;3;5;8;…` | auxiliary DPP set, `;`-joined |
| 8 | `E` | `1;4;6;7;…` | suppressed set, `;`-joined |
| 9 | `record_sha256` | `d3080d7e…` | sha256 of `F:E:j` |
| 10-13 | `SA,SB,SC,D` | `2.1727,1.7367,2.3573,0.4975` | **the physics** — per-sample entropies and their corner combination (natural log) |
| 14-17 | `VA,VB,VC,DV` | `0.8893,0.7173,0.9224,0.1830` | number variances, origin geometry |
| 18-22 | `VA_full,VB_full,VC_full,DV_full,DV_subgroup` | | translation-averaged variants |
| 23-25 | `NA,NB,NC` | `12.442,8.018,12.844` | regional particle numbers |
| 26-30 | `channel_probe, channel_residual, Vlaw_{A,B,C}_residual` | `0.4190,0.0074,…` | exact-identity residuals |
| 31-33 | `sampling_seconds, entropy_seconds, diagnostic_seconds` | `7.94e-5,5.55e-5,2.33e-5` | per-record timings |

`SA`, `SB`, `SC`, `D` are computed **per trajectory before any averaging**
(`run_iid_targeted.jl:145`, `:156`). The mean over samples is the Born average
`Σ_E p_β(E) S₂(R;E)`, *not* `−log` of a mean purity — this distinction is stated
in `run_iid_targeted.jl:46` and implemented by averaging `entropies`, never
`exp(−entropy)`.

---

## 3. Regions A, B, C

### 3.1 Production definition — p = 1/3

```julia
# original_swap_scaling.jl:168-172
function region_indices(L)
    (A=area_index("HoneyComb120",[L,L],([1,1],[L÷3,L])),
     B=area_index("HoneyComb120",[L,L],([1,1],[L÷3,2L÷3])),
     C=area_index("HoneyComb120",[L,L],([1,1],[L,L÷3])))
end
```

`area_index` for `HoneyComb*` takes the `else` branch:

```julia
# src/public/Geometry.jl:387-399
counter = 1
index = zeros(Int64, 2 * prod(area[2] - area[1] + [1, 1]))
for lx in area[1][1]:area[2][1]
    for ly in area[1][2]:area[2][2]
        index[counter] = xy_i(Lattice, site, lx, ly) - 1
        index[counter+1] = index[counter] + 1
        counter += 2
    end
end
```

with

```julia
# src/public/Geometry.jl:11
return 2 * (x + (y - 1) * site[1])
```

So for cell `(x,y)`: site `2(x+(y−1)L)−1` is the **A sublattice**, site
`2(x+(y−1)L)` is the **B sublattice**. `x` runs over columns, `y` over rows.
**No mod is applied**: the rectangle is literally the corner
`1 ≤ x ≤ w, 1 ≤ y ≤ L` — regions are *not* wrapped. The lattice itself is
periodic (`nnK_Matrix` uses `mod1`); only the region mask is corner-anchored.

| region | cells | sites | size |
|---|---|---|---|
| A | `1 ≤ x ≤ L/3`, `1 ≤ y ≤ L` | `2·w·L` | `2L²/3` |
| B | `1 ≤ x ≤ L/3`, `1 ≤ y ≤ 2L/3` | `2·w·2w` | `4L²/9` |
| C | `1 ≤ x ≤ L`, `1 ≤ y ≤ L/3` | `2·L·w` | `2L²/3` |

with `w = L÷3`. Measured sizes **[measured, recomputed in Python from
`xy_i`]**:

| L | N=2L² | \|A\| | \|B\| | \|C\| |
|---|---|---|---|---|
| 6 | 72 | 24 | 16 | 24 |
| 12 | 288 | 96 | 64 | 96 |
| 18 | 648 | 216 | 144 | 216 |
| 36 | 2592 | 864 | 576 | 864 |

A and C have equal size but **are different sets** (`A ≠ C` for every L tested;
`A∩C = 2·w·w`, e.g. 32 at L=12). A is a strip running the full periodic length in
`y`; C is a strip running the full periodic length in `x`. B is the corner block
with two 60° and two 120° corners (`BORN_SCALING_RESULTS.md:411-413`).

### 3.2 Where p = 1/2 exists

Only in the translated-boundary diagnostics, never in the iid / production path:

```julia
# run_born_boundary_check.jl:40-51
for m in (3,2)
    w=L÷m
    shifts = full_translations ? collect(0:L-1) : collect(0:w:L-1)
    for ty in shifts, tx in shifts
        ids=(index_id(boundary_indices(L,w,L,tx,ty)),
             index_id(boundary_indices(L,w,L-w,tx,ty)),
             index_id(boundary_indices(L,L,w,tx,ty)))
```

with the translation wrap applied explicitly:

```julia
# run_born_boundary_check.jl:9-11
function boundary_indices(L,w,h,tx,ty)
    sort!([2(mod(x+tx,L)+mod(y+ty,L)*L)+s for y in 0:h-1 for x in 0:w-1 for s in 1:2])
end
```

Here `s ∈ 1:2` gives the same odd/even A/B pairing. At `tx=ty=0` these sets are
**identical to `area_index`'s** — verified for A, B and C at L = 6, 12, 18
**[measured, set comparison]**. The corner combination generalises to

```julia
# run_born_boundary_check.jl:77
D=((entry.m-1)*SA+SC)/entry.m-SB
```

i.e. `D = (2·SA + SC)/3 − SB` for p=1/3 and `D = (SA + SC)/2 − SB` for p=1/2.
Region sizes at p=1/2: `|A|=|C|=L²`, `|B|=L²/2` (L=6: 36/18/36; L=18: 324/162/324)
**[measured]**. Subgroup shift counts are `pd²` (9 for p=1/3, 4 for p=1/2) and full
translations `L²` (`check_born_boundary_geometry.jl:93-95`).

---

## 4. The Dirac ground state

Built once per L by `ground_model`, `original_swap_scaling.jl:26-73`. The
staggered branch is `original_swap_scaling.jl:32-50`:

```julia
factor=svd(K[1:2:N,2:2:N])                                    # :36  A–B hopping block
sigma=copy(factor.S)
null=findall(x->x<zero_singular_tolerance,sigma); nullcount=length(null)
nullcount==2 || error("expected exactly two Dirac singular values ...")   # :39
sigma[null].=0.
epsilon=hypot.(sigma,mu)                                      # :41
coefficient_A=sigma./sqrt.(2 .*epsilon.*(epsilon.+mu))        # :42
coefficient_B=-sqrt.((epsilon.+mu)./(2 .*epsilon))            # :43
Q=zeros(N,n)
Q[1:2:N,:].=factor.U.*coefficient_A'                          # :45
Q[2:2:N,:].=factor.V.*coefficient_B'                          # :46
@inbounds for i in 1:N; original[i,i]+=isodd(i) ? mu : -mu; end   # :47
```

So: **SVD of the A–B hopping block**, not a dense diagonalisation; the two Dirac
singular values are forced to exactly zero numerically; the interleaved staggered
chemical potential `+μ` on A, `−μ` on B with `μ/t = 1e-8` is then applied on the
diagonal of the *reference* Hamiltonian (line 47) and enters the state through
`ε = √(σ²+μ²)` (line 41). The projector is built as
`G = Q·inv(Q'Q)·Q'` (`original_swap_scaling.jl:71`) — deliberately matching what
`Free_M.jl:150` does, not its exact-arithmetic simplification.

**Measured consequence of `μ > 0`** **[measured]**: the two zero-mode-derived
orbitals sit *entirely on sublattice B*.

| L | null singular values | zero-mode orbitals: Σ\|Q_A\|² | Σ\|Q_B\|² | total Σ\|Q_A\|² | total Σ\|Q_B\|² | n |
|---|---|---|---|---|---|---|
| 6 | 2 | 0, 0 | 1, 1 | 17.000000 | 19.000000 | 36 |
| 12 | 2 | 0, 0 | 1, 1 | 71.000000 | 73.000000 | 144 |

Occupied energies run from `−3.0` (flat band) to `−1e-8` (the staggered zero
modes), i.e. the "gap to zero" is exactly `−μ`.

A LAPACK-only alternative exists (`state=:lapack`, `mu=0`,
`original_swap_scaling.jl:51-56`) and is what `Free_M.jl` uses. It chooses an
arbitrary zero-mode basis and is documented as *not* mixed into the new-size fits
(`BORN_SCALING_RESULTS.md:464`).

---

## 5. Per-sample cost

### 5.1 Dominant step

The dominant step in the **i.i.d. sampler** is `BornSampler.sample_projection`
(`BornSampler.jl:74-98`) — a sequential Householder elimination that removes one
orbital at each of the `a` selected sites. Its inner work per sweep is
`N·k` for the weight recomputation plus a `ger!` of an `N×k` block, summed over
`k = a…1`, i.e. **`O(N·a²) = O(L⁶)`** since `N = 2a = 2L²`. It is *not* a QR and
*not* an eigenproblem.

The entropy routine's dominant step is one `LAPACK.geqrf!` + `orgqr!` on the
`N×a` filtered orbital matrix (`original_swap_scaling.jl:196-197`), `O(N·a²)`, plus
three Cholesky factorizations of the restricted purity matrices (lines 206), each
`O(k³)` with `k = |R|`.

### 5.2 Measured cost per sample **[measured, this audit]**

Single BLAS thread, `julia --project=.`, warm-up discarded, minimum over 5–31 reps
of `sample_projection` and median of `entropy!`:

| L | a = L² | projection (ms) | entropy (ms) | total per sample (s) |
|---|---|---|---|---|
| 6 | 36 | 0.046 | 0.082 | 0.00013 |
| 12 | 144 | 2.01 | 1.84 | 0.0039 |
| 18 | 324 | 22.7 | 11.1 | 0.034 |
| 24 | 576 | 293 | 64.1 | 0.357 |
| 30 | 900 | 1589 | 248 | 1.84 |
| 36 | 1296 | 7256 | 644 | 7.90 |

Empirical log-log slopes in `a`: projection edges `8.65→8.14→5.83` (L = 12→18,
18→24, 24→30) and overall `12→36` = **3.73**; entropy overall `36→1296` = **2.50**.
So the projection is *steeper* than the `O(a³)` count predicts (cache/BLAS-1
overhead on skinny `ger!`), and the entropy is slightly *shallower*. Extrapolating
the projection at the measured 3.7 exponent from L=36 gives ≈ 2 min/sample at
L=48 and ≈ 40 min at L=60 — treat those extrapolations as order-of-magnitude only.

### 5.3 Recorded timings in the repository **[recorded]**

`research_20261029/iid_targeted/` receipts (512 records each, 4 batches/point):

| L | beta | sampling s | entropy s | diag s | elapsed s | sampling ms/record |
|---|---|---|---|---|---|---|
| 6 | 1 | 0.035–0.043 | 0.021–0.025 | 0.007–0.008 | 0.078–0.093 | ≈0.07 |
| 6 | 3 | 0.035–0.043 | 0.022–0.024 | 0.007–0.008 | 0.078–0.089 | ≈0.07 |
| 12 | 1 | 1.35–1.38 | 0.74–0.83 | 0.30 | 2.49–2.62 | ≈2.7 |
| 12 | 3 | 1.33–1.36 | 0.73–0.76 | 0.29 | 2.48–2.51 | ≈2.6 |
| 18 | 1 | 43.7–44.2 | 10.0–10.2 | 4.15–4.21 | 60.1–60.5 | ≈86 |
| 18 | 3 | 15.4–15.6 | 5.13–5.15 | 2.39–2.42 | 23.4–23.6 | ≈30 |

**Discrepancy to be aware of.** At L=18, beta=1 the recorded sampling cost is
≈86 ms/record (median 116 ms), but re-running the identical seed
`729_283_001` and even the full per-record loop in this audit gives
**23–24 ms/record** **[measured, 3 independent scripts, also with 4 concurrent
processes]**. The beta=3 figure (30 ms) *does* reproduce. No code path, seed, `j`
distribution or concurrency difference explains the beta=1 value; the L18 batches
were created within 0.25 s of each other (`created_at` 09:37:53.3–53.6), so they
did run as four concurrent workers. I could not confirm the cause and do not
invent one — treat the beta=1 L18 sampling cost as **contended / not
reproducible**, and use ≈25 ms as the planning figure with a 4× contingency.

Cross-check from other runners: `analysis/iid_benchmark.csv` gives
0.000159 / 0.00782 / 0.0861 s per sample at L = 6/12/18 (matches the receipts, not
my remeasurement); `data/born_scaling/beta3_large_throughput/` gives L=36,
20 timed sweeps in 24.9–27.8 s (1 worker) vs 57.4–59.3 s (4 workers), i.e. **4
concurrent processes each run ≈2.3× slower**, so total throughput rises ~1.7×
going 1→4 workers. Memory bandwidth, not core count, is the limit.

Boundary-check production receipts (fused *swap* sampler, not iid, 768 or 2304
sweeps/chain, 4 chains; `update_seconds` + `entropy_seconds` + `refresh_seconds`):

| L | elapsed s (per chain) | update s | entropy s | refresh s |
|---|---|---|---|---|
| 6 | 1.01–1.08 | 0.090–0.093 | 0.38–0.40 | 0.03–0.08 |
| 12 | 8.88–9.51 | 2.03–2.13 | 6.08–6.50 | 0.29–0.33 |
| 18 | 53.3–55.7 | 18.1–19.3 | 33.6–34.9 | 1.01–1.07 |
| 24 | 383–391 | 188–202 | 184–190 | 4.20–4.42 |
| 30 | 1505–1530 | 863–887 | 629–630 | 12.2–12.4 |
| 36 | 5240–5266 | 3444–3480 | 1754–1765 | 30.1–30.8 |

Note the crossover at L≈27: below it the entropy kernel dominates, above it the
swap updates do (L=30 update 866 s vs entropy 629 s; L=36 3460 s vs 1760 s).

---

## 6. Gotchas

1. **`L % 3 != 0` is rejected twice.** `original_swap_scaling.jl:28`
   (`L>=3 && L%3==0 || throw(...)`) and `Free_M.jl:129`
   (`entropy regions require L divisible by 3`). Verified: L = 7, 8, 10 all throw
   **[measured]**. The reason is structural, not cosmetic: the number of singular
   values below 1e-12 in the A–B hopping block is **2** for L = 9, 12, 15 and **0**
   for L = 7, 8, 10 **[measured]**; `ground_model` hard-requires exactly 2
   (`original_swap_scaling.jl:39`).

2. **Zero-mode convention is a physics choice, not a detail.** Exactly two
   singular values must fall below `zero_singular_tolerance` (default `1e-12`,
   `original_swap_scaling.jl:27`). With `μ = 1e-8 > 0` those modes land entirely on
   sublattice B **[measured]**. The repo's own audit warns this matters:
   *"The exponent d and specific target probability may depend on the chosen
   degenerate Slater ground state. Zero overlap with one selected ground state does
   not mean the target product state cannot project into any ground state."*
   (`analysis/zero_mode_sensitivity.toml:6`). `BORN_SCALING_RESULTS.md:464` states
   the old `μ = 0` LAPACK choice is kept only as a control.

3. **Memory is not a constraint, and the "2L²×L²" worry is half right.** Only `Q`
   is rectangular `N×a`: L=36 → 26.9 MB. `G`, `meanGsquare` are `N×N` = 53.7 MB
   each; `X` is `a×a` = 13.4 MB; `meanG` another 53.7 MB. At L=48 it is
   Q 85 MB / G 170 MB / X 43 MB. Peak working set at L=36 ≈ 150 MB
   **[measured arithmetic]**. `sample_projection` copies `Q` once per call
   (`BornSampler.jl:75`), so per-draw allocation is `N×a` — the source of the
   GC-driven timing spread.

4. **Float64 conditioning vs beta.** The filtered QR condition number grows
   roughly like `(1/e^{-2β})` (and with L):

   | L | β=1 | β=3 | β=6 | β=10 |
   |---|---|---|---|---|
   | 12 | 7.4 | 4.0e2 | 6.9e3 | 6.9e3 |
   | 18 | 7.4 | 4.0e2 | 1.6e5 | 5.8e5 |
   | 36 | 7.4 | 4.0e2 | 1.6e5 | 1.1e7 |

   **[measured]** At β=3 (the production value) cond ≈ 4e2, so ~12 of 16 digits
   survive and the entropy is safe. At β≥6 you are down to ~10 digits at L=18 and
   ~9 at L=36. The log-weights themselves are large negative (`−2245` at L=36,
   β=3) and `exp` of them underflows — which is fine because everything stays in
   log space (`BornSampler.jl:120`, `ExactBorn.jl:136`), but it means **you cannot
   importance-weight by exponentiating stored weights**. `attenuation = exp(-2β)`
   (`ExactBorn.jl:118`, `original_swap_scaling.jl:194`) only underflows past
   β ≈ 372.

5. **The swap sampler dies at β = 10; the iid sampler does not.** `λ = 1 − e^{-4β}`
   must be `< 1` (`original_swap_scaling.jl:125`, `Free_M.jl:156`). Measured:
   β=9 → λ = 0.9999999999999998 (OK); **β=10 → λ = 1.0 exactly in Float64 →
   `error("sampling inverse is numerically singular at this beta")`**
   **[measured]**. The i.i.d. path has no such ceiling: `StratifiedSampler` at
   β = 6, 9, 10, 12 all sample fine (with `log_Z = 0.0`, i.e. all mass in `j=0`)
   **[measured]**. So **β < 10 is a hard limit only for the Markov path**.

6. **Multithreading: none.** Every entry point calls `BLAS.set_num_threads(1)`
   (`run_iid_targeted.jl:4`, `run_born_scaling.jl:7`, `original_swap_scaling.jl`
   users, `check_born_sampler.jl:6`, …). Grepping `Threads.@threads`,
   `@spawn`, `@distributed` across all `.jl` outside `data/` returns **nothing**
   **[measured]**. Parallelism is entirely process-level: 4 workers in
   `run_iid_targeted` (batch index = worker index) and
   `execution_workers = 4` in the boundary plan. Julia here reports
   `Threads.nthreads() = 8`, but nothing uses them. Since 4 concurrent processes
   each slow by ≈2.3× **[recorded, L=36 throughput rounds]**, oversubscribing past
   4 workers is counter-productive.

7. **Refuse-to-overwrite is everywhere.** `run_iid_targeted.jl:27` (prepare),
   `:129` (job output), `:185` (manifest); `run_born_scaling.jl:33` (plan),
   `:129` via `valid_completed`; `run_born_boundary_check.jl:94`. `analyze_iid_targeted.py:65`
   raises `FileExistsError("Preserve existing successful analysis")`. A re-run
   always requires moving a directory aside.

8. **`Free_M.jl` is the legacy two-region, `μ=0` path.** It computes only A and B
   (`Free_M.jl:137-138`, `:201-203`) and uses `LAPACK.syevd!` on the massless `K`
   (`Free_M.jl:142-147`). Running it as a script loops `L ∈ {12,18}` ×
   `β ∈ 0.0:0.1:3.0` with 1000 samples each (`Free_M.jl:223-232`) — that is the
   long job to avoid launching casually (it was still running after 120 s when I
   stopped it, having reached L=12, β=0.6).

9. **Fixed half filling only.** `|E| = |S| = L²`. The sampler requires
   `N == 2a` (`BornSampler.jl:42`, `ExactBorn.jl:102`) and there is **no**
   reweighting for the event "exactly half were suppressed"
   (`BORN_SCALING_RESULTS.md:466-470`). `ensemble=:full_local` is a *different*
   ensemble and must not be compared with `:fixed_half`
   (`ExactBorn.jl:80-83`).

---

## 7. Exact enumeration at L = 3

**Yes.** `ExactBorn.born_trajectory_entropy` (`ExactBorn.jl:93-178`) enumerates
every outcome in the `:fixed_half` sector via `foreach_subset` (`ExactBorn.jl:20-31`).

* configuration count: `binomial(2a, a)` with `a = L²`. For L=3, `N=18`, `n=9`:
  **`C(18,9) = 48 620`** **[measured]**.
* the count is checked in `BigInt` **before** allocating, against `max_configs`
  (`ExactBorn.jl:106-108`). Default is 1e6, so L=4 (`C(32,16) = 601 080 390`) and
  L=6 (`C(72,36) = 4.4251e20`) are rejected, not attempted **[measured: L=6 threw
  `ArgumentError: exact enumeration requires 442512540276836779204 configurations`]**.
* **cost: ~1.5 s per (L=3, β) case.** Recorded receipts in
  `data/exact_born/exact_born_manifest.toml` list `elapsed_seconds` of
  1.471, 1.476, 1.553, 1.581, 1.175 for β = 1.0, 0.5, 2.0, 0.0, 3.0
  (`[recorded]`), summing to **7.53 s** for all five betas. My own cold run at
  β=1.5 took **1.055 s** with `keep_trajectories=true` and **0.767 s** with
  `false` **[measured]**. `keep_trajectories=false` is *not* constant memory — the
  implementation still retains all weights and entropies internally
  (`ExactBorn.jl:88-89`).
* per-configuration work is one `qr!` of the `N×a` filtered orbital matrix plus
  three regional `eigvals(Hermitian(V*V'))` (`ExactBorn.jl:130-143`), so the whole
  enumeration is `O(C(2a,a)·(N a²))`. Extrapolating that to L=4 is hopeless for a
  single core.
* output: `mean_entropy`, `mean_purity`, `annealed_entropy`, `entropy_covariance`,
  normalization checks, and optionally the full `configurations / log_weights /
  log_probabilities / probabilities / entropies` arrays
  (`ExactBorn.jl:166-177`). The first L=3 run in the repo wrote
  7.7–8.1 MB trajectory CSVs per beta (`data/born_scaling/exact_L3_mu1e-8_beta*_trajectories.csv`).
* the production driver is `analysis/born_scaling_exact_L3.jl`, which asserts
  `exact.nconfigurations == 48_620` (line 68) and uses `max_configs=50_000`
  (line 67).

L=3 is therefore the **only** size at which the Born average of S₂ is known
exactly. It is also the size where the zero-mode convention bites hardest: at L=3
A and C have different entropies *for a fixed Slater state*, and the exact file
records only one state (`analysis/born_scaling_exact_L3.jl:45`).

---

## 8. Minimum recipe, in order

```bash
cd /d/JuliaDQMC/code/KAPDQMC

# 1. Environment sanity (2 s)
julia --project=. -e 'using KAPDQMC, LinearAlgebra; println(VERSION, " BLAS=", BLAS.get_num_threads())'

# 2. Exact L=3 reference for the staggered state, all production betas (~10 s)
julia --project=. src/Measure/born_ai/analysis/born_scaling_exact_L3.jl

# 3. i.i.d. Born samples (frozen 24 jobs; SKIPs if already done)
for w in 1 2 3 4; do
  julia --project=. src/Measure/born_ai/research_20261029/run_iid_targeted.jl --worker=$w
done

# 4. Analysis (refuses to overwrite)
D:/app/conda/python.exe -I src/Measure/born_ai/research_20261029/analyze_iid_targeted.py
D:/app/conda/python.exe -I src/Measure/born_ai/research_20261029/plot_iid_targeted.py
```

Budget for step 3 with 4 workers: L=18 is the wall-clock driver at ≈60 s
(β=1) and ≈24 s (β=3) per 512-record batch, so the whole 24-job set fits in
well under 10 minutes; the recorded manifests were all produced inside ~90 s
(`created_at` 09:37:53 → 09:39:27). A new size (L=24, 30, 36) must be added by
editing `run_iid_targeted.jl:29-38` or by the driver in §1.3, and costs
≈0.36 s / 1.8 s / 7.9 s **per sample** respectively, i.e. 2048 samples at L=30 is
≈1 hour of single-core work and L=36 is ≈4.5 hours.

---

## 9. Unexplained / not in the files

* The L=18, β=1 sampling cost in the receipts (86–116 ms/record) does not
  reproduce (23–24 ms). No explanation is present in any log or receipt.
* No timing exists anywhere for the **iid** sampler at L = 24, 30, 36 — all iid
  data stops at L=18. The L=24/30/36 numbers in §5.2 are mine, not the project's.
* `data/born_scaling/exact_L3_mu1e-8.toml` records
  `region_sizes = [6, 4, 6]` for L=3, consistent with `(2·1·3, 2·1·2, 2·3·1)`, but
  L=3's A/B/C are test partitions, not a scaling geometry
  (`data/exact_born/exact_born_manifest.toml`, `geometry` key).
* No file records a total wall-clock for the full 24-Job iid production other than
  the per-receipt `elapsed_seconds`.
