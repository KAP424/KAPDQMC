# Append more independent Born samples to existing (L, beta) CSV files.
#
# The production runner (run_matched_z.jl) skips files that already exist, so it
# cannot increase the sample count of a point.  This script opens the existing
# CSV, keeps every row, and appends `nextra` fresh samples drawn from a seed that
# is disjoint from the original run (different seed namespace, and the sample
# index continues from the existing count).
#
# Usage:
#   julia --project=@v1.11 extend_samples.jl --L=30 --beta=4.6110 --nextra=2560
#   julia --project=@v1.11 extend_samples.jl --file=data_gapfill/L36_beta2.6931_w1.csv --nextra=768

using LinearAlgebra, Random, Dates, Printf
BLAS.set_num_threads(1)
include(joinpath(@__DIR__, "..", "research_20261029", "StratifiedBorn.jl"))
using .StratifiedBorn
include(joinpath(@__DIR__, "..", "original_swap_scaling.jl"))
using .OriginalSwapScaling

const HEADER = "L,beta,z,sample,j,SA,SB,SC,D,DV,elapsed"

"""Read (L, beta, nrows, last_elapsed, ncols, max_sample) from an existing CSV.

The two data directories use slightly different column sets: the matched-z files
carry 10 columns (no elapsed), the gapfill files 11.  Parse positionally only for
the first two fields and take the elapsed column if present.

`ncols` is the width of the *header*.  Rows must be written with that same width:
a 10-column file that receives 11-column rows can no longer be read by
np.genfromtxt(names=True), which is what every figure script uses.

`max_sample` is the largest existing `sample` value.  New rows must continue from
it, NOT from the row count: the gapfill files number their second half 257..512,
so a row-count-based index restarts at 257 and collides.
"""
function existing(path)
    isfile(path) || return nothing
    L = beta = 0
    n = 0
    last_t = 0.0
    ncols = 11
    max_sample = 0
    open(path) do io
        eof(io) && return
        ncols = length(split(readline(io), ','))   # header
        for line in eachline(io)
            isempty(strip(line)) && continue
            f = split(line, ',')
            L = parse(Int, f[1])
            beta = parse(Float64, f[2])
            max_sample = max(max_sample, parse(Int, f[4]))
            n += 1
            length(f) >= 11 && (last_t = parse(Float64, f[11]))
        end
    end
    (L, beta, n, last_t, ncols, max_sample)
end

"""A stable integer tag for the file's identity.

The gapfill points are stored as two files, `..._w1.csv` and `..._w2.csv`, with
the same (L, beta, n0).  A seed built only from (L, beta, n0) therefore gives
both halves the SAME random stream, and the second half silently re-samples the
first.  Folding in the base name keeps the halves on disjoint streams while
staying reproducible run to run.
"""
function file_tag(path)
    h = UInt64(0xcbf29ce484222325)          # FNV-1a
    for b in codeunits(basename(path))
        h = (h ⊻ b) * 0x100000001b3
    end
    Int(h % UInt64(1 << 31))
end

function main()
    path = nothing
    L = beta = nothing
    nextra = 512
    seed0 = 2026_11_01
    for a in ARGS
        startswith(a, "--file=") && (path = split(a, '=')[2])
        startswith(a, "--L=") && (L = parse(Int, split(a, '=')[2]))
        startswith(a, "--beta=") && (beta = parse(Float64, split(a, '=')[2]))
        startswith(a, "--nextra=") && (nextra = parse(Int, split(a, '=')[2]))
        startswith(a, "--seed=") && (seed0 = parse(Int, split(a, '=')[2]))
    end

    if path === nothing
        (L === nothing || beta === nothing) && error("give --file or --L/--beta")
        L, beta = Int(L), Float64(beta)
        path = joinpath(@__DIR__, "data_matched_z",
                        @sprintf("L%d_beta%.4f.csv", L, beta))
    else
        path = isabspath(path) ? path : joinpath(@__DIR__, path)
    end

    info = existing(path)
    if info === nothing
        error("file does not exist: $path  (this script only appends)")
    end
    L, beta, n0, t_last, ncols, max_sample = info
    z = L * L * exp(-2beta)
    @printf("extending %s : L=%d beta=%.4f z=%.4f  have %d samples, adding %d  (%d-col header)\n",
            basename(path), L, beta, z, n0, nextra, ncols)

    m = ground_model(L; state=:staggered, mu=1e-8)
    work = EntropyWorkspace(m)
    s = StratifiedSampler(m.Q, beta)
    # Disjoint seed namespace from run_matched_z.jl (which used 2026_10_30 + ...).
    # The file tag separates the two halves of a gapfill point: `_w1` and `_w2`
    # share (L, beta, n0), so without it the second half re-draws the first.
    rng = MersenneTwister(seed0 + 7919 * L + round(Int, 10_000 * beta) + n0 +
                          file_tag(path))
    # Continue the sample index from the largest existing value, not from the row
    # count: gapfill w2 files are numbered 257..512, and n0 + k would restart at
    # 257 and duplicate those indices.
    next_index = max_sample + 1

    open(path, "a") do io
        t0 = time()
        for k in 1:nextra
            draw = sample_stratum(s, 0, m.n; rng)
            e = entropy!(work, m, draw.E, beta)
            V = [length(work.regions[t]) / 4 - sum(abs2, work.correlation[t])
                 for t in 1:3]
            DV = (2V[1] + V[3]) / 3 - V[2]
            if ncols >= 11
                @printf(io, "%d,%.4f,%.4f,%d,%d,%.10f,%.10f,%.10f,%.10f,%.10f,%.2f\n",
                        L, beta, z, next_index + k - 1, draw.j, e.SA, e.SB, e.SC, e.D, DV,
                        t_last + time() - t0)
            else   # 10-column file: no elapsed field
                @printf(io, "%d,%.4f,%.4f,%d,%d,%.10f,%.10f,%.10f,%.10f,%.10f\n",
                        L, beta, z, next_index + k - 1, draw.j, e.SA, e.SB, e.SC, e.D, DV)
            end
            k % 64 == 0 && (flush(io); println("PROGRESS $k/$nextra"); flush(stdout))
        end
    end
    println("DONE $(basename(path)) : $n0 -> $(n0 + nextra) samples")
end

main()
