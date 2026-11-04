#!/usr/bin/env julia
"""
Finite-size census of every half-filled 1x1, 2x2 and 3x3 product pattern.

Run with `julia configuration_census.jl --benchmark=1000` to estimate runtime,
then `julia configuration_census.jl --mode=all` for exhaustive enumeration.
Use `--sizes-only` for all 2x2/L12 and 3x3/L18 patterns, followed by every
special 3x3 pattern at L27 and L36. `--validate-only` checks existing CSVs.
`--mode=sample` keeps every special pattern plus 2048 reproducibly sampled
general 3x3 patterns. All 1x1/2x2 patterns are retained in either mode.

The lattice is L=3m for each pattern period m. Counts are therefore finite-size
facts at L=3,6,9, NOT classifications proven for every larger lattice. Patterns
are not reduced by symmetry; all multiplicities are retained. Bit j of mask
is orbital j+1, with orbital order A(x,y),B(x,y), x fastest. `bitstring` lists
these occupations in ascending orbital order (least significant bit first).
Global-ground accessibility is defined by excess_energy, not negative_holes:
with zero modes, an excitation can replace a zero mode without leaving a
negative-energy hole. positive_particles and zero_occupation expose this case.
"""

using LinearAlgebra, Random, Printf
BLAS.set_num_threads(1)
include(joinpath(@__DIR__, "fermion_experiment.jl"))
using .FermionExperiment

const DATA_DIR = joinpath(@__DIR__, "data")
const CENSUS_SEED = 20261018

pattern_mask(mask::Integer, m::Int) = BitVector([!iszero(mask & (1 << j)) for j in 0:2m^2-1])
all_masks(m::Int) = [mask for mask in 0:(1 << (2m^2))-1 if count_ones(mask) == m^2]

function census_row(model, mask)
    occ = pattern_mask(mask, model.m)
    stats = pattern_statistics(occ)
    pre = prepare_pattern(model, occ)
    state = evolve(pre, Inf)
    diag = diagnostics(state)
    negatives = sum(count(<(-1e-9), e) for e in model.energies)
    occupied_negatives = sum(count(<(-1e-9), e) for e in pre.pivots)
    holes = negatives - occupied_negatives
    positives = sum(count(>(1e-9), e) for e in pre.pivots)
    zeros_occupied = sum(count(x -> abs(x) <= 1e-9, e) for e in pre.pivots)
    abs(diag.excess_energy-diag.accessible_excess) < 1e-8 || error("energy inconsistency")
    diag.residual < 1e-8 || error("nonstationary accessible limit")
    diag.min_initial_sv > 1e-9 || error("accessible limit has unreliable initial overlap")
    return (m=model.m, L=model.L, mask=mask,
        bitstring=join(Int.(occ)), special=stats.special, doubles=stats.double_cells,
        energy=diag.energy, excess_energy=diag.excess_energy,
        min_initial_sv=diag.min_initial_sv, negative_holes=holes,
        positive_particles=positives, zero_occupation=zeros_occupied,
        sublattice_imbalance=stats.sublattice_imbalance,
        residual=diag.residual, reconstruction_error=diag.reconstruction_error)
end

function write_row(io, row)
    println(io, join(values(row), ','))
end

function benchmark(n::Int)
    model = BlochModel(9, 3)
    masks = all_masks(3)
    shuffle!(MersenneTwister(CENSUS_SEED), masks)
    census_row(model, masks[1]) # compilation is excluded from throughput
    n = min(n, length(masks))
    elapsed = @elapsed for mask in masks[1:n]
        census_row(model, mask)
    end
    @printf("census benchmark: %d patterns, %.3f seconds, exhaustive m3 estimate %.1f seconds\n",
        n, elapsed, elapsed*length(masks)/n)
    mkpath(DATA_DIR)
    open(joinpath(DATA_DIR, "configuration_census_benchmark.csv"), "w") do io
        println(io, "count,seconds,estimated_full_m3_seconds,seed,blas_threads")
        println(io, "$n,$elapsed,$(elapsed*length(masks)/n),$CENSUS_SEED,1")
    end
end

function chosen_masks(m::Int, mode::AbstractString)
    masks = all_masks(m)
    (m < 3 || mode == "all") && return masks
    special = filter(mask -> pattern_statistics(pattern_mask(mask,m)).special, masks)
    general = filter(mask -> !pattern_statistics(pattern_mask(mask,m)).special, masks)
    shuffle!(MersenneTwister(CENSUS_SEED), general)
    return sort!(vcat(special, general[1:min(2048,length(general))]))
end

function run_census(mode::AbstractString)
    mode in ("all", "sample") || error("mode must be all or sample")
    mkpath(DATA_DIR)
    rows = NamedTuple[]
    started = time()
    open(joinpath(DATA_DIR, "configuration_census.csv"), "w") do io
        firstrow = true
        for m in 1:3
            model = BlochModel(3m, m)
            masks = chosen_masks(m,mode)
            for (i,mask) in enumerate(masks)
                row = census_row(model,mask)
                if firstrow
                    println(io,join(keys(row),','))
                    firstrow = false
                end
                write_row(io,row)
                push!(rows,row)
                if i % 1000 == 0
                    flush(io)
                    @printf("m=%d progress=%d/%d total_seconds=%.1f\n",m,i,length(masks),time()-started)
                    flush(stdout)
                end
            end
            @printf("m=%d complete: %d patterns, L=%d\n",m,length(masks),3m)
            flush(stdout)
        end
    end
    open(joinpath(DATA_DIR,"configuration_census_summary.csv"),"w") do io
        println(io,"m,L,special,doubles,negative_holes,positive_particles,zero_occupation,count,ground_count,min_excess,max_excess,min_initial_sv,max_initial_sv")
        groups=sort!(unique([(r.m,r.special,r.doubles,r.negative_holes,r.positive_particles,r.zero_occupation) for r in rows]))
        for key in groups
            group=filter(r -> (r.m,r.special,r.doubles,r.negative_holes,r.positive_particles,r.zero_occupation)==key,rows)
            m,special,doubles,holes,positives,zeros_occupied=key
            println(io,join((m,3m,special,doubles,holes,positives,zeros_occupied,length(group),
                count(r->abs(r.excess_energy)<1e-8,group),
                minimum(r.excess_energy for r in group),maximum(r.excess_energy for r in group),
                minimum(r.min_initial_sv for r in group),maximum(r.min_initial_sv for r in group)),','))
        end
    end
    open(joinpath(DATA_DIR,"configuration_census_manifest.txt"),"w") do io
        println(io,"mode=$mode\nseed=$CENSUS_SEED\nblas_threads=1\njulia_version=$VERSION")
        println(io,"elapsed_seconds=$(time()-started)\ntol=1e-10\nenergy_zero_threshold=1e-9")
        println(io,"orbitals=A(x,y),B(x,y); x fastest; bitstring lists low bit first")
        println(io,"L=3m; finite-size census; no symmetry reduction")
        for m in 1:3
            n=count(r->r.m==m,rows)
            println(io,"m=$m sampled=$n total=$(binomial(2m^2,m^2))")
        end
    end
    @printf("census complete: %d rows, %.1f seconds\n",length(rows),time()-started)
    validate_census(rows,"configuration_census")
end

function translate_mask(mask,m,dx,dy)
    moved=0
    for y in 0:m-1, x in 0:m-1, s in 0:1
        source=s+2*(x+m*y)
        dest=s+2*(mod(x+dx,m)+m*mod(y+dy,m))
        if !iszero(mask & (1 << source))
            moved |= 1 << dest
        end
    end
    return moved
end

function validate_census(rows,prefix)
    lookup=Dict((r.m,r.L,r.mask)=>r for r in rows)
    length(lookup)==length(rows) || error("duplicate census rows")
    complement_energy_error=0.0
    complement_sv_error=0.0
    translation_energy_error=0.0
    translation_sv_error=0.0
    complement_checks=0
    translation_checks=0
    for r in rows
        c=get(lookup,(r.m,r.L,((1 << (2r.m^2))-1) ⊻ r.mask),nothing)
        if !isnothing(c)
            complement_checks+=1
            complement_energy_error=max(complement_energy_error,abs(r.energy-c.energy))
            complement_sv_error=max(complement_sv_error,abs(r.min_initial_sv-c.min_initial_sv))
            @assert r.negative_holes == c.positive_particles
            @assert r.positive_particles == c.negative_holes
            @assert r.zero_occupation+c.zero_occupation == 4
        end
        for (dx,dy) in ((1,0),(0,1))
            t=get(lookup,(r.m,r.L,translate_mask(r.mask,r.m,dx,dy)),nothing)
            if !isnothing(t)
                translation_checks+=1
                translation_energy_error=max(translation_energy_error,abs(r.energy-t.energy))
                translation_sv_error=max(translation_sv_error,abs(r.min_initial_sv-t.min_initial_sv))
            end
        end
    end
    max_residual=maximum(r.residual for r in rows)
    max_reconstruction_error=maximum(r.reconstruction_error for r in rows)
    for err in (complement_energy_error,complement_sv_error,
            translation_energy_error,translation_sv_error,max_residual,max_reconstruction_error)
        @assert err < 1e-8
    end
    open(joinpath(DATA_DIR,prefix*"_validation.csv"),"w") do io
        println(io,"row_count,unique_count,complement_checks,translation_checks,complement_energy_error,complement_sv_error,translation_energy_error,translation_sv_error,max_residual,max_reconstruction_error")
        println(io,join((length(rows),length(lookup),complement_checks,translation_checks,complement_energy_error,complement_sv_error,
            translation_energy_error,translation_sv_error,max_residual,max_reconstruction_error),','))
    end
    open(joinpath(DATA_DIR,prefix*"_overview.csv"),"w") do io
        println(io,"m,L,special,count,global_ground_count,obstructed_count,ground_fraction,min_initial_sv,min_excess,max_excess")
        groups=sort!(unique([(r.m,r.L,r.special) for r in rows]))
        for key in groups
            group=filter(r -> (r.m,r.L,r.special)==key,rows)
            ground=count(r->abs(r.excess_energy)<1e-8,group)
            println(io,join((key...,length(group),ground,length(group)-ground,ground/length(group),
                minimum(r.min_initial_sv for r in group),
                minimum(r.excess_energy for r in group),maximum(r.excess_energy for r in group)),','))
        end
    end
end

function load_census(path)
    lines=readlines(path)
    headers=Symbol.(split(lines[1],','))
    integer_fields=(:m,:L,:mask,:doubles,:negative_holes,:positive_particles,:zero_occupation,:sublattice_imbalance)
    rows=NamedTuple[]
    for line in lines[2:end]
        cells=split(line,',')
        vals=map(zip(headers,cells)) do (name,value)
            name==:bitstring ? String(value) : name==:special ? parse(Bool,value) :
                name in integer_fields ? parse(Int,value) : parse(Float64,value)
        end
        push!(rows,NamedTuple{Tuple(headers)}(Tuple(vals)))
    end
    return rows
end

function size_census()
    mkpath(DATA_DIR)
    rows=NamedTuple[]
    started=time()
    open(joinpath(DATA_DIR,"configuration_census_sizes.csv"),"w") do io
        firstrow=true
        for (m,L,only_special) in ((2,12,false),(3,18,false),(3,27,true),(3,36,true))
            model=BlochModel(L,m)
            masks=all_masks(m)
            if only_special
                filter!(mask -> pattern_statistics(pattern_mask(mask,m)).special,masks)
            end
            for (i,mask) in enumerate(masks)
                row=census_row(model,mask)
                if firstrow
                    println(io,join(keys(row),','))
                    firstrow=false
                end
                write_row(io,row)
                push!(rows,row)
                if i % 2000 == 0
                    flush(io)
                    @printf("size census m=%d L=%d progress=%d/%d total_seconds=%.1f\n",m,L,i,length(masks),time()-started)
                    flush(stdout)
                end
            end
            @printf("size census m=%d L=%d complete: %d patterns\n",m,L,length(masks))
            flush(stdout)
        end
    end
    validate_census(rows,"configuration_census_sizes")
    open(joinpath(DATA_DIR,"configuration_census_sizes_manifest.txt"),"w") do io
        println(io,"mode=exhaustive_by_selected_class\nblas_threads=1\njulia_version=$VERSION")
        println(io,"elapsed_seconds=$(time()-started)\ntol=1e-10\nenergy_zero_threshold=1e-9")
        println(io,"m2 L12: all70; m3 L18: all48620; m3 L27 and36: all512special each")
        println(io,"No symmetry reduction. All rows are distinct full masks at specified L.")
    end
    @printf("size census complete: %d rows, %.1f seconds\n",length(rows),time()-started)
end

function state_counterexamples()
    mkpath(DATA_DIR)
    open(joinpath(DATA_DIR,"state_counterexamples.csv"),"w") do io
        println(io,"L,m,pattern_a,pattern_b,energy_a,energy_b,ground_energy,projector_distance_absolute,projector_distance_normalized,overlap_min_sv,negative_holes_a,negative_holes_b")
        for L in (6,12)
            model=BlochModel(L,1)
            a=evolve(prepare_pattern(model,Bool[1,0]),Inf)
            b=evolve(prepare_pattern(model,Bool[0,1]),Inf)
            da,db=diagnostics(a),diagnostics(b)
            d=projector_distance(a,b;normalized=false)
            overlap=minimum(minimum(svdvals(x'*y)) for (x,y) in zip(a.orbitals,b.orbitals))
            @assert isapprox(d,2;atol=1e-10)
            @assert abs(da.excess_energy)<1e-9 && abs(db.excess_energy)<1e-9
            @assert overlap<1e-10
            println(io,join((L,1,"all_A","all_B",da.energy,db.energy,model.ground_energy,d,d/L,overlap,0,0),','))
        end
    end
end

function fixed_measurement_controls()
    mkpath(DATA_DIR)
    patterns=["all_A"=>Bool[1,0,1,0,1,0,1,0],
        "all_B"=>Bool[0,1,0,1,0,1,0,1],
        "single_B"=>Bool[0,1,1,0,1,0,1,0],
        "stripe"=>Bool[1,0,0,1,1,0,0,1],
        "general_full_empty"=>Bool[1,1,0,0,1,1,0,0]]
    open(joinpath(DATA_DIR,"fixed_measurement_controls.csv"),"w") do io
        println(io,"L,m,initial_pattern,measurement_target,beta,initial_excess_energy,S_y,S_x,S_B,delta_original,delta_balanced,target_empty_occupancy")
        for L in (6,12)
            model=BlochModel(L,2)
            target_pre=prepare_pattern(model,patterns[1].second)
            for (name,occ) in patterns
                base=evolve(prepare_pattern(model,occ),Inf)
                initial_excess=diagnostics(base).excess_energy
                # Measurement uses fixed all-A target for every initial pattern.
                # Rebinding metadata does not alter the state's occupied subspace.
                fixed_state=FermionState(target_pre,Inf,0.0,base.orbitals)
                for beta in (0.0,0.5,1.0,2.0,4.0)
                    state=measure(fixed_state,beta)
                    sy=entropy_strip(state;orientation=:x)
                    sx=entropy_strip(state;orientation=:y)
                    sb=entropy_rectangle(state)
                    empty=diagnostics(state).empty_occupancy
                    println(io,join((L,2,name,"all_A",beta,initial_excess,sy,sx,sb,sy-sb,(2sy+sx)/3-sb,empty),','))
                end
            end
        end
    end
end

if abspath(PROGRAM_FILE)==@__FILE__
    bench=findfirst(startswith("--benchmark="),ARGS)
    if "--validate-only" in ARGS
        for prefix in ("configuration_census","configuration_census_sizes")
            path=joinpath(DATA_DIR,prefix*".csv")
            isfile(path) && validate_census(load_census(path),prefix)
        end
    elseif "--sizes-only" in ARGS
        size_census()
    elseif !isnothing(bench)
        benchmark(parse(Int,split(ARGS[bench],'=')[2]))
    else
        modearg=findfirst(startswith("--mode="),ARGS)
        mode=isnothing(modearg) ? "all" : split(ARGS[modearg],'=')[2]
        run_census(mode)
        state_counterexamples()
        fixed_measurement_controls()
    end
end
