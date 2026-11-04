# beta = 0 anchor through the *identical* Born code path.
#
# The referee objection: data_matched_z/ contains no beta = 0 point, so the
# normalisation D_bar -> D_gs has never been checked inside the same code path
# that produces the Born samples.  If D_gs (computed by baseline_D.py from a
# separate dense solver) is biased relative to the Born chain, every quoted
# ratio h = D_bar / D_gs inherits that bias, and the 1-3.5% positive excess at
# large z cannot be interpreted as physics.
#
# At beta = 0 the measurement operator is the identity, so every sample must
# give exactly the ground-state corner entropy.  Any spread across samples, or
# any offset from baseline_D.json, is a defect in the D path, not statistics.
#
# Usage: julia --project=. run_beta0_anchor.jl

using LinearAlgebra, Random, Printf
BLAS.set_num_threads(1)
include(joinpath(@__DIR__, "..", "research_20261029", "StratifiedBorn.jl"))
using .StratifiedBorn
include(joinpath(@__DIR__, "..", "original_swap_scaling.jl"))
using .OriginalSwapScaling

const HEADER = "L,beta,z,sample,j,SA,SB,SC,D,DV"

function main()
    outdir = joinpath(@__DIR__, "data_beta0")
    mkpath(outdir)
    for L in (6, 12, 18, 24, 30, 36)
        m = ground_model(L; state=:staggered, mu=1e-8)
        work = EntropyWorkspace(m)
        s = StratifiedSampler(m.Q, 0.0)
        rng = MersenneTwister(2026_11_01 + 13L)
        n = 32
        fname = joinpath(outdir, @sprintf("L%d_beta0.0000.csv", L))
        isfile(fname) && continue
        open(fname * ".partial", "w") do io
            println(io, HEADER)
            for k in 1:n
                draw = sample_stratum(s, 0, m.n; rng)
                e = entropy!(work, m, draw.E, 0.0)
                V = [length(work.regions[t]) / 4 - sum(abs2, work.correlation[t])
                     for t in 1:3]
                DV = (2V[1] + V[3]) / 3 - V[2]
                @printf(io, "%d,%.4f,%.4f,%d,%d,%.10f,%.10f,%.10f,%.10f,%.10f\n",
                        L, 0.0, 0.0, k, draw.j, e.SA, e.SB, e.SC, e.D, DV)
            end
        end
        mv(fname * ".partial", fname)
        println("DONE L=$L")
        flush(stdout)
    end
end

main()
