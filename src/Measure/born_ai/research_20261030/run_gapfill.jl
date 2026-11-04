# Fill the crossover gap at the largest sizes, at z values already measured at
# smaller L, so the strong-side claim "D_bar depends on z alone" can be tested
# where it should start to fail.
#
#   z = 5.9343  (= 18^2 e^-4, the L=18 beta=2 point: D_bar = 0.3454 +- 0.0072)
#   z = 10.5498 (= 24^2 e^-4, the L=24 beta=2 point: D_bar = 0.4537 +- 0.0122)
#
# measured at L = 30, 36, i.e. beta = 2 + ln(L/18) and beta = 2 + ln(L/24).
# Each (L, beta) point is split over two jobs (256 samples each, independent
# seeds); every job writes its own file, so jobs never share an output path.
# Analysis scripts pool files per (L, beta).
#
# Usage (one process per job, 8 jobs):
#   julia --project=.. run_gapfill.jl --job=K        K = 1..8

using LinearAlgebra, Random, Printf
BLAS.set_num_threads(1)
include(joinpath(@__DIR__, "..", "research_20261029", "StratifiedBorn.jl"))
using .StratifiedBorn
include(joinpath(@__DIR__, "..", "original_swap_scaling.jl"))
using .OriginalSwapScaling

const HEADER = "L,beta,z,sample,j,SA,SB,SC,D,DV,elapsed"
const NHALF = 256

function jobs()
    out = Tuple{Int,Float64,Int}[]
    for L0 in (18, 24), L in (30, 36), half in (1, 2)
        push!(out, (L, round(2.0 + log(L / L0); digits=4), half))
    end
    out
end

function main()
    job = 0
    for a in ARGS
        startswith(a, "--job=") && (job = parse(Int, split(a, '=')[2]))
    end
    1 <= job <= length(jobs()) || error("--job must be 1..$(length(jobs()))")
    L, beta, half = jobs()[job]
    outdir = joinpath(@__DIR__, "data_gapfill")
    mkpath(outdir)
    fname = joinpath(outdir, @sprintf("L%d_beta%.4f_w%d.csv", L, beta, half))
    isfile(fname) && return
    m = ground_model(L; state=:staggered, mu=1e-8)
    work = EntropyWorkspace(m)
    s = StratifiedSampler(m.Q, beta)
    rng = MersenneTwister(2026_10_31 + 100_000 * half + 1000 * L + round(Int, 1e4 * beta))
    z = L * L * exp(-2beta)
    open(fname * ".partial", "w") do io
        println(io, HEADER)
        t0 = time()
        for k in 1:NHALF
            draw = sample_stratum(s, 0, m.n; rng)
            e = entropy!(work, m, draw.E, beta)
            V = [length(work.regions[t]) / 4 - sum(abs2, work.correlation[t]) for t in 1:3]
            DV = (2V[1] + V[3]) / 3 - V[2]
            @printf(io, "%d,%.4f,%.4f,%d,%d,%.10f,%.10f,%.10f,%.10f,%.10f,%.2f\n",
                    L, beta, z, (half - 1) * NHALF + k, draw.j, e.SA, e.SB, e.SC, e.D, DV,
                    time() - t0)
            k % 32 == 0 && (flush(io); println("job $job L=$L beta=$beta $k/$NHALF"); flush(stdout))
        end
    end
    mv(fname * ".partial", fname)
    println("DONE job $job L=$L beta=$beta z=$z")
end

main()
