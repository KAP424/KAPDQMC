# Decisive matched-z test at a LARGE z, without interpolation.
#
# The heavy-tail verifier objected that at z ~ 43.8 the production curves (which are
# sampled on a beta grid and then interpolated in ln z) give
#     L=12: 0.479, L=15: 0.517, L=18: 0.552, L=21: 0.565, L=24: 0.578
# i.e. a spread far outside the reported MCSE, so z cannot be a collapse variable.
#
# But those numbers are interpolated across a region where D is strongly curved in
# ln z, and the two-parameter picture predicts D = D_gs(L) * h(z/z*(L)) which is NOT
# constant across L at fixed z: D_gs(L) grows.
#
# This script measures matched z = 43.85 directly at L = 12, 18, 24, 30 with the same
# independent sampler, so no interpolation enters.
#
# Usage: julia --project=. run_matched_largez.jl

using LinearAlgebra, Random, Printf
BLAS.set_num_threads(1)
include(joinpath(@__DIR__, "..", "research_20261029", "StratifiedBorn.jl"))
using .StratifiedBorn
include(joinpath(@__DIR__, "..", "original_swap_scaling.jl"))
using .OriginalSwapScaling

const HEADER = "L,beta,z,sample,j,SA,SB,SC,D,DV"
const ZTARGET = 43.85

function main()
    outdir = joinpath(@__DIR__, "data_matched_largez")
    mkpath(outdir)
    for L in (12, 18, 24, 30)
        beta = 0.5 * log(L * L / ZTARGET)
        z = L * L * exp(-2beta)
        m = ground_model(L; state=:staggered, mu=1e-8)
        work = EntropyWorkspace(m)
        s = StratifiedSampler(m.Q, beta)
        rng = MersenneTwister(2026_10_30 + 77L)
        n = L >= 30 ? 1024 : 2048
        fname = joinpath(outdir, @sprintf("L%d_beta%.4f.csv", L, beta))
        isfile(fname) && continue
        open(fname * ".partial", "w") do io
            println(io, HEADER)
            for k in 1:n
                draw = sample_stratum(s, 0, m.n; rng)
                e = entropy!(work, m, draw.E, beta)
                V = [length(work.regions[t]) / 4 - sum(abs2, work.correlation[t])
                     for t in 1:3]
                DV = (2V[1] + V[3]) / 3 - V[2]
                @printf(io, "%d,%.4f,%.4f,%d,%d,%.10f,%.10f,%.10f,%.10f,%.10f\n",
                        L, beta, z, k, draw.j, e.SA, e.SB, e.SC, e.D, DV)
                k % 128 == 0 && (flush(io); println("PROGRESS L=$L $k/$n"); flush(stdout))
            end
        end
        mv(fname * ".partial", fname)
        println("DONE L=$L beta=$beta z=$z n=$n")
        flush(stdout)
    end
end

main()
