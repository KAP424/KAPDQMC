# Matched-crossover study of the Born-averaged corner entropy.
#
# The mean-channel analysis says the single measurement layer has a SIZE-DEPENDENT
# strength: the auxiliary j distribution is governed by z = L^2 exp(-2 beta), the
# first strong-measurement correction entering at order z. Runs at fixed beta but
# different L therefore do NOT sit at the same point of the measurement-strength
# axis, and a fixed-beta fit of D against ln L mixes the thermodynamic corner
# coefficient with this crossover.
#
# This script samples the exact independent Born ensemble (StratifiedBorn, full
# analytic j distribution, no tail truncation) on a grid chosen so that constant-z
# lines can be compared directly.
#
# Usage: julia --project=. run_matched_z.jl --worker=K --out=DIR

using LinearAlgebra, Random, Dates, Printf
BLAS.set_num_threads(1)
include(joinpath(@__DIR__, "..", "research_20261029", "StratifiedBorn.jl"))
using .StratifiedBorn
include(joinpath(@__DIR__, "..", "original_swap_scaling.jl"))
using .OriginalSwapScaling

const HEADER = "L,beta,z,sample,j,SA,SB,SC,D,DV,elapsed"

# (L, beta) grid.  z = L^2 e^{-2 beta} is recorded per row so that equal-z
# pairs can be selected afterwards without re-deciding anything.
function grid()
    pts = Tuple{Int,Float64}[]
    # fixed-z families: z columns 0.089, 0.357, 0.803, 1.428, 3.213
    # (the L=6..36, beta=3 line of the earlier runs) reproduced at other L
    for L in (6, 12, 18, 24, 30, 36)
        for ztarget in (0.089, 0.357, 0.803, 1.428, 3.213)
            b = 0.5 * log(L * L / ztarget)
            b > 0 || continue
            push!(pts, (L, round(b; digits=4)))
        end
    end
    # thermodynamic-axis anchors at fixed beta, larger L than the old runs
    for L in (18, 24, 30, 36), b in (0.5, 1.0, 1.5, 2.0)
        push!(pts, (L, b))
    end
    sort!(unique(pts))
end

function main()
    worker = 1
    outdir = joinpath(@__DIR__, "data_matched_z")
    for a in ARGS
        startswith(a, "--worker=") && (worker = parse(Int, split(a, '=')[2]))
        startswith(a, "--out=") && (outdir = split(a, '=')[2])
    end
    mkpath(outdir)
    pts = grid()
    mine = [p for (i, p) in enumerate(pts) if mod1(i, 4) == worker]

    models = Dict{Int, Any}()
    for L in unique(first.(mine))
        models[L] = ground_model(L; state=:staggered, mu=1e-8)
    end

    for (L, beta) in mine
        m = models[L]
        work = EntropyWorkspace(m)
        s = StratifiedSampler(m.Q, beta)
        rng = MersenneTwister(2026_10_30 + 1000 * worker + L + round(Int, 10beta))
        n = L >= 30 ? 512 : (L >= 24 ? 1024 : 2048)
        z = L * L * exp(-2beta)
        fname = joinpath(outdir, @sprintf("L%d_beta%.4f.csv", L, beta))
        isfile(fname) && continue
        open(fname * ".partial", "w") do io
            println(io, HEADER)
            t0 = time()
            for k in 1:n
                draw = sample_stratum(s, 0, m.n; rng)
                e = entropy!(work, m, draw.E, beta)
                V = [length(work.regions[t]) / 4 - sum(abs2, work.correlation[t])
                     for t in 1:3]
                DV = (2V[1] + V[3]) / 3 - V[2]
                @printf(io, "%d,%.4f,%.4f,%d,%d,%.10f,%.10f,%.10f,%.10f,%.10f,%.2f\n",
                        L, beta, z, k, draw.j, e.SA, e.SB, e.SC, e.D, DV, time() - t0)
                k % 64 == 0 && (flush(io); println("PROGRESS L=$L beta=$beta $k/$n"); flush(stdout))
            end
        end
        mv(fname * ".partial", fname)
        println("DONE L=$L beta=$beta z=$z n=$n")
        flush(stdout)
    end
end

main()
