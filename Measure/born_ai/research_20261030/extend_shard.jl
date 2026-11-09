# Sharded version of extend_samples.jl.
#
# extend_samples.jl appends to one CSV, so two processes cannot work on the same
# point.  The expensive points (L=30 at small z needs ~34k samples = 23 CPU-h)
# would then take 23 h of wall clock.  This script writes each shard to its own
# file, so N shards run in parallel; merge_shards.py folds them back in.
#
# Usage:
#   julia --project=@v1.11 extend_shard.jl --file=data_matched_z/L30_beta4.6108.csv \
#         --shard=3 --nshards=8 --nextra=4256

using LinearAlgebra, Random, Dates, Printf
BLAS.set_num_threads(1)
include(joinpath(@__DIR__, "..", "research_20261029", "StratifiedBorn.jl"))
using .StratifiedBorn
include(joinpath(@__DIR__, "..", "original_swap_scaling.jl"))
using .OriginalSwapScaling

function main()
    path = nothing
    nextra = 512
    shard = 0
    nshards = 1
    seed0 = 2026_11_02
    for a in ARGS
        startswith(a, "--file=") && (path = split(a, '=')[2])
        startswith(a, "--nextra=") && (nextra = parse(Int, split(a, '=')[2]))
        startswith(a, "--shard=") && (shard = parse(Int, split(a, '=')[2]))
        startswith(a, "--nshards=") && (nshards = parse(Int, split(a, '=')[2]))
        startswith(a, "--seed=") && (seed0 = parse(Int, split(a, '=')[2]))
    end
    path === nothing && error("give --file")

    # Recover (L, beta) from the parent CSV's own rows.  Parsing the file name
    # is unsafe: "L12_beta3.6945.csv" would capture "3.6945." as the number.
    # Also count the parent's rows: the seed must depend on them, see below.
    isfile(path) || error("no such file: $path")
    L = beta = 0.0
    n0 = 0
    open(path) do io
        readline(io)                        # header
        line = readline(io)
        f = split(line, ',')
        L = parse(Float64, f[1])
        beta = parse(Float64, f[2])
        n0 = 1
        for l in eachline(io)
            isempty(strip(l)) || (n0 += 1)
        end
    end
    L = Int(L)
    z = L * L * exp(-2beta)

    m = ground_model(L; state=:staggered, mu=1e-8)
    work = EntropyWorkspace(m)
    s = StratifiedSampler(m.Q, beta)
    # Every (file, shard, run) gets its own stream.
    #  - the shard index keeps two shards of one run apart;
    #  - n0 (rows already in the parent) keeps a SECOND run on the same point
    #    from replaying the first run's shards draw for draw -- run_shards.py
    #    re-queues any point still above tolerance, and without n0 that second
    #    run would append exact duplicates and shrink the error bar for free;
    #  - the file tag keeps the two halves of a gapfill point (_w1/_w2, same
    #    L, beta and possibly same n0) apart, the bug that duplicated rows in
    #    extend_samples.jl.
    tag = let h = UInt64(0xcbf29ce484222325)            # FNV-1a of the base name
        for b in codeunits(basename(path))
            h = (h ⊻ b) * 0x100000001b3
        end
        Int(h % UInt64(1 << 31))
    end
    rng = MersenneTwister(seed0 + 104729 * shard + 7919 * L +
                          round(Int, 10_000 * beta) + 1_000_003 * n0 + tag)

    out = @sprintf("%s.shard%dof%d.csv", path, shard + 1, nshards)
    @printf("shard %d/%d of %s : L=%d beta=%.4f z=%.4f  %d samples -> %s\n",
            shard + 1, nshards, basename(path), L, beta, z, nextra,
            basename(out))

    open(out, "w") do io
        println(io, "L,beta,z,sample,j,SA,SB,SC,D,DV,elapsed")
        t0 = time()
        for k in 1:nextra
            draw = sample_stratum(s, 0, m.n; rng)
            e = entropy!(work, m, draw.E, beta)
            V = [length(work.regions[t]) / 4 - sum(abs2, work.correlation[t])
                 for t in 1:3]
            DV = (2V[1] + V[3]) / 3 - V[2]
            @printf(io, "%d,%.4f,%.4f,%d,%d,%.10f,%.10f,%.10f,%.10f,%.10f,%.2f\n",
                    L, beta, z, k, draw.j, e.SA, e.SB, e.SC, e.D, DV,
                    time() - t0)
            k % 64 == 0 && (flush(io); println("PROGRESS $k/$nextra"); flush(stdout))
        end
    end
    println("DONE $(basename(out)) : $nextra samples")
end

main()
