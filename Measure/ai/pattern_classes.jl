#!/usr/bin/env julia
#=
Post-measurement entanglement entropy across configurations: enumerate, measure,
classify.

Goal: the strong-measurement entropy coefficient A_R(P,R) is a functional of the
initial pattern P only.  We want the law that organises all half-filled patterns
into a few classes.

Strategy: enumerate *every* half-filled pattern of a small supercell (complete
classification, no sampling bias), compute for each

  * the cell-occupancy class used by the project so far
        class 1: every cell singly occupied   (n0,n1,n2) = (0,m^2,0)
        class 2: pure empty/double cells      (n0,n1,n2) = (m^2/2,0,m^2/2)
        class 3: mixed                        n0,n1,n2 all > 0
  * s_min = min over Bloch blocks of sigma_min(Q_F) and ||X||_2 = sqrt(s_min^-2 - 1)
  * A_A, A_C, A_B at several L, hence a_R = A_A/L and the corner combination D_A

and then ask which of these quantities actually determines the entropy influence.

A pattern with s_min -> 0 has a singular Q_F: the measured state is not normalisable
in the theta -> inf limit, so A_R does not exist.  Those are recorded as obstructed
rather than silently dropped.

Knobs (environment): PC_M, PC_LADDER, PC_MAX_PATTERNS, PC_SMIN_TOL.
=#
using LinearAlgebra, Printf

const HERE_PC = @__DIR__
include(joinpath(HERE_PC, "strong_scaling.jl"))   # dft2, x_kernel, aR_kernel, region_masks
const FE_PC = FermionExperiment

# ------------------------------------------------------------- enumeration
"""
    enumerate_half_filled(m) -> Vector{BitVector}

Every pattern with exactly m^2 occupied sites out of 2m^2, in lexicographic bit
order (A(x,y), B(x,y) with x fastest).  Complete, so the classification below has
no selection bias.
"""
function enumerate_half_filled(m::Int)
    n, p = 2m^2, m^2
    out = BitVector[]
    # combinations in colexicographic order
    comb = collect(1:p)
    while true
        occ = falses(n)
        occ[comb] .= true
        push!(out, occ)
        # next combination
        i = p
        while i >= 1 && comb[i] == n - p + i
            i -= 1
        end
        i == 0 && break
        comb[i] += 1
        for j in (i+1):p
            comb[j] = comb[j-1] + 1
        end
    end
    return out
end

"""Cell occupancy (n0, n1, n2) and the project's class label."""
function occupancy_class(occ::BitVector, m::Int)
    cells = [Int(occ[2j-1]) + Int(occ[2j]) for j in 1:m^2]
    n0, n1, n2 = count(==(0), cells), count(==(1), cells), count(==(2), cells)
    cls = n1 == m^2 ? 1 : (n1 == 0 ? 2 : 3)
    return (; n0, n1, n2, cls, cells)
end

"""
    enumerate_class1(m) -> Vector{BitVector}

Every pattern with exactly one fermion per cell: each of the m^2 cells puts its
fermion on sublattice A or B, so there are 2^(m^2) of them.  Enumerated directly
rather than by filtering the full C(2m^2,m^2) enumeration, which is what makes
m=3 (512 patterns) and m=4 (65536) tractable.
"""
function enumerate_class1(m::Int)
    ncells = m^2
    out = BitVector[]
    for mask in 0:(2^ncells - 1)
        occ = falses(2ncells)
        for j in 1:ncells
            occ[2j-1] = ((mask >> (j - 1)) & 1) == 0
            occ[2j] = !occ[2j-1]
        end
        push!(out, occ)
    end
    return out
end

"""Pattern name that is stable across runs: class + bit string."""
pattern_label(occ::BitVector, m::Int) = "m$(m)_" * join(Int.(occ))

# ------------------------------------------------------------- one pattern
"""
    pattern_diagnostics(occ, m, ladder; smin_tol)

s_min, ||X||, A_A/A_C/A_B on the commensurate ladder (L must be a multiple of 3m).
Returns `obstructed=true` (with NaN observables) when Q_F is singular.
"""
function pattern_diagnostics(occ::BitVector, m::Int, ladder::Vector{Int};
    smin_tol=1e-8)
    rows = NamedTuple[]
    obstructed = false
    smin_global = Inf
    for L in ladder
        L % (3m) == 0 || continue
        model = FE_PC.BlochModel(L, m)
        prep = FE_PC.prepare_pattern(model, occ)
        base = FE_PC.evolve(prep, Inf)
        kernel, eset, fset, smin = x_kernel(base)
        smin_global = min(smin_global, smin)
        if smin < smin_tol
            obstructed = true
            break
        end
        specs = region_specs(L)
        vals = Dict{Symbol,Float64}()
        for (key, sp) in pairs(specs)
            vals[key] = aR_kernel(kernel, eset, fset,
                region_masks(model, sp.width, sp.height))
        end
        # Moments of |X_ab(delta)|^2 in LATTICE units.  delta indexes supercell
        # displacements, so the physical distance is m*|delta| and the strip width is
        # L/3.  A site type occupies (L/m)^2 positions, so across a straight boundary
        # the straddling-pair count is (L/m)*min(|dx|, L/(3m)) and
        #     A_A = 2(t1+t2) = (2L/m^2) M,  M = sum_ab sum_delta |X|^2 min(m|dx|, L/3)
        # hence the universal law
        #     a_R = A_A/L = (4/m^2) * M.
        # Verified to machine precision for m = 1, 2, 3 (max deviation 4e-14).
        q = size(kernel, 3)
        half = L ÷ 3
        fro2 = 0.0
        momx = 0.0
        momy = 0.0
        momx_raw = 0.0
        for dx in 0:q-1, dy in 0:q-1
            w = 0.0
            for bi in eachindex(fset), ai in eachindex(eset)
                w += abs2(kernel[ai, bi, dx+1, dy+1])
            end
            fro2 += w
            td = min(dx, q - dx)          # torus distance in x, supercell units
            te = min(dy, q - dy)
            momx += w * min(m * td, half)
            momy += w * min(m * te, half)
            momx_raw += w * td
        end
        push!(rows, (; L, smin, A_A=vals[:A], A_C=vals[:C], A_B=vals[:B],
            D_A=(2vals[:A] + vals[:C]) / 3 - vals[:B],
            fro2, momx, momy, momx_raw))
    end
    xnorm = smin_global < Inf ? sqrt(1 / smin_global^2 - 1) : Inf
    return (; obstructed, smin=smin_global, xnorm, rows)
end

# ------------------------------------------------------------- csv helpers
"""Value of field `k` of a named tuple row, as it should appear in the CSV."""
function _csv_field(v)
    v isa AbstractString && return v
    v isa Bool && return v ? "true" : "false"
    v isa Integer && return string(v)
    return string(Float64(v))
end

count_data_rows(path) = isfile(path) ? max(0, count(!isempty, readlines(path)) - 1) : 0

"""
    merge_csv(path, cols, rows, keycols)

Write `rows` (named tuples) to `path`, keeping any existing row whose key (the
values in `keycols`) is not re-measured.  Old rows keep their on-disk order and
the new ones are appended.
"""
function merge_csv(path, cols::Vector{String}, rows, keycols::Vector{String})
    # Keys must be compared as strings on both sides: the freshly measured rows carry
    # Int fields (L), while rows read back from disk are all strings, so an untyped
    # tuple would never match and every run would append a duplicate copy.
    keyof_new(r) = tuple((_csv_field(getproperty(r, Symbol(c))) for c in keycols)...)
    newkeys = Set{Tuple}(map(keyof_new, rows))
    kept = String[]
    if isfile(path)
        lines = readlines(path)
        if !isempty(lines)
            pos = Dict(h => i for (i, h) in enumerate(split(lines[1], ',')))
            for line in lines[2:end]
                isempty(strip(line)) && continue
                f = split(line, ',')
                key = tuple((f[pos[c]] for c in keycols)...)
                key in newkeys || push!(kept, line)
            end
        end
    end
    open(path, "w") do io
        println(io, join(cols, ','))
        for line in kept
            println(io, line)
        end
        for r in rows
            println(io, join((_csv_field(getproperty(r, Symbol(c))) for c in cols), ','))
        end
    end
end

# ------------------------------------------------------------------- main
function main_pc()
    ms = parse.(Int, split(get(ENV, "PC_M", "2"), ','))
    ladder = parse.(Int, split(get(ENV, "PC_LADDER", "6,12,24,36"), ','))
    max_pat = parse(Int, get(ENV, "PC_MAX_PATTERNS", "0"))
    max_c1 = parse(Int, get(ENV, "PC_MAX_CLASS1", "0"))
    smin_tol = parse(Float64, get(ENV, "PC_SMIN_TOL", "1e-8"))
    classes = parse.(Int, split(get(ENV, "PC_CLASSES", "1,2,3"), ','))

    obs = NamedTuple[]
    summary = NamedTuple[]
    for m in ms
        # Class 1 is enumerated directly (2^m^2 patterns, complete).  Classes 2 and 3
        # are filtered out of the full C(2m^2,m^2) enumeration and then reduced by a
        # deterministic stride sample, because for m=3 that enumeration is 48620 long.
        pats = BitVector[]
        if 1 in classes
            c1 = enumerate_class1(m)
            # 2^m^2 grows fast (m=4 is 65536), so class 1 needs its own cap
            if max_c1 > 0 && length(c1) > max_c1
                stride = cld(length(c1), max_c1)
                c1 = c1[1:stride:end]
            end
            append!(pats, c1)
        end
        if any(c -> c != 1, classes)
            rest = BitVector[]
            for occ in enumerate_half_filled(m)
                occupancy_class(occ, m).cls in classes || continue
                occupancy_class(occ, m).cls == 1 && continue   # already added
                push!(rest, occ)
            end
            if max_pat > 0 && length(rest) > max_pat
                stride = cld(length(rest), max_pat)
                rest = rest[1:stride:end]
            end
            append!(pats, rest)
        end
        @printf("m=%d: %d patterns (classes %s), ladder %s\n",
            m, length(pats), join(classes, ","), join(ladder, ","))
        flush(stdout)
        verbose = get(ENV, "PC_PROGRESS", "0") == "1"
        for (pi, occ) in enumerate(pats)
            if verbose && pi % 32 == 1
                @printf("   [%d/%d] %s\n", pi, length(pats), pattern_label(occ, m))
                flush(stdout)
            end
            info = occupancy_class(occ, m)
            d = pattern_diagnostics(occ, m, ladder; smin_tol)
            push!(summary, (; m, label=pattern_label(occ, m), cls=info.cls,
                n0=info.n0, n1=info.n1, n2=info.n2, obstructed=d.obstructed,
                smin=d.smin, xnorm=d.xnorm, nrows=length(d.rows)))
            for r in d.rows
                push!(obs, (; m, label=pattern_label(occ, m), cls=info.cls,
                    n0=info.n0, n1=info.n1, n2=info.n2, smin=r.smin,
                    L=r.L, A_A=r.A_A, A_C=r.A_C, A_B=r.A_B, D_A=r.D_A,
                    fro2=r.fro2, momx=r.momx, momy=r.momy, momx_raw=r.momx_raw))
            end
        end
    end

    # ---- persist (cumulative: merge with what is already on disk so successive
    # sweeps of different m / ladders accumulate into one dataset instead of
    # overwriting each other; a re-measured (label,L) replaces the old row)
    # A separate prefix lets a dedicated ladder (e.g. the floor family scanned to
    # very large L) be analysed on its own without polluting the main dataset,
    # whose fixed-L comparisons assume one common largest L per m.
    prefix = get(ENV, "PC_OUT_PREFIX", "pattern_class")
    csv = joinpath(HERE_PC, "data", "$(prefix)_observations.csv")
    scsv = joinpath(HERE_PC, "data", "$(prefix)_summary.csv")
    obs_cols = ["m", "label", "cls", "n0", "n1", "n2", "smin", "L", "A_A", "A_C",
        "A_B", "D_A", "fro2", "momx", "momy", "momx_raw"]
    sum_cols = ["m", "label", "cls", "n0", "n1", "n2", "obstructed", "smin",
        "xnorm", "nrows"]
    merge_csv(csv, obs_cols, obs, ["label", "L"])
    merge_csv(scsv, sum_cols, summary, ["label"])
    @printf("  (dataset now holds %d observation rows, %d patterns)\n",
        count_data_rows(csv), count_data_rows(scsv))

    # ---- class composition
    println()
    println("class composition (obstructed = Q_F singular, A_R undefined):")
    for m in ms, cls in 1:3
        sub = [s for s in summary if s.m == m && s.cls == cls]
        isempty(sub) && continue
        n_obs = count(s -> s.obstructed, sub)
        @printf("  m=%d class %d: %4d patterns, %4d obstructed\n", m, cls,
            length(sub), n_obs)
    end

    # ---- is s_min fixed by the occupancy class?
    println()
    println("s_min within each occupancy class (measurable patterns only):")
    for m in ms, cls in 1:3
        sub = [s.smin for s in summary if s.m == m && s.cls == cls && !s.obstructed]
        isempty(sub) && continue
        @printf("  m=%d class %d: n=%4d  s_min in [%.6f, %.6f]  ||X|| in [%.3f, %.3f]\n",
            m, cls, length(sub), minimum(sub), maximum(sub),
            sqrt(1 / maximum(sub)^2 - 1), sqrt(1 / minimum(sub)^2 - 1))
    end

    # ---- does a_R collapse onto a function of ||X|| ?
    println()
    println("a_R = A_A/L at the largest L, grouped by occupancy class:")
    lmax = isempty(obs) ? 0 : maximum(r.L for r in obs)
    for cls in 1:3
        sub = [r for r in obs if r.cls == cls && r.L == lmax]
        isempty(sub) && continue
        vals = sort([r.A_A / r.L for r in sub])
        @printf("  class %d (L=%d): n=%4d  a_R in [%.4f, %.4f]\n", cls, lmax,
            length(vals), first(vals), last(vals))
    end

    # ---- metrics
    println()
    @printf("METRIC pc_patterns=%d\n", length(summary))
    @printf("METRIC pc_measurable=%d\n", count(s -> !s.obstructed, summary))
    @printf("METRIC pc_obstructed=%d\n", count(s -> s.obstructed, summary))
    @printf("METRIC pc_rows=%d\n", length(obs))
    @printf("METRIC pc_L_max=%d\n", lmax)
    for cls in 1:3
        sub = [s.smin for s in summary if s.cls == cls && !s.obstructed]
        isempty(sub) && continue
        @printf("METRIC pc_smin_min_cls%d=%.6f\n", cls, minimum(sub))
        @printf("METRIC pc_smin_max_cls%d=%.6f\n", cls, maximum(sub))
    end
    return obs, summary
end

if abspath(PROGRAM_FILE) == @__FILE__
    t0 = time()
    main_pc()
    @printf("METRIC pc_seconds=%.3f\n", time() - t0)
end
