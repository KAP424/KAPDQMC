#!/usr/bin/env julia
"""
Small-lattice controls for spatial arrangement at fixed occupation statistics.
Only L=9 and L=18 are used. Select two special m=3 patterns (imbalance 5,
no double cells) and two general patterns (imbalance 1, one double/empty cell),
all certified ground-energy accessible at BOTH L=9 and L=18 by the census.

Selection compares the beta=1 minus beta=0 change in the translation-averaged
rectangle entropy at L=9. Every special translation orbit is screened; at most
64 general orbits are sampled with fixed seed. The selected pair is an example,
not an exhaustive entropy classification or a thermodynamic-limit claim.
"""

include(joinpath(@__DIR__,"configuration_census.jl"))

function canonical_translation(mask,m=3)
    minimum(translate_mask(mask,m,dx,dy) for dx in 0:m-1,dy in 0:m-1)
end

function arrangement_entropies(state)
    sa=entropy_strip(state;orientation=:x)
    sc=entropy_strip(state;orientation=:y)
    sb=entropy_rectangle(state)
    return (;S_A=sa,S_C=sc,S_B=sb,D=(2sa+sc)/3-sb)
end

function phase_average(model,mask,beta)
    out=zeros(4)
    for dy in 0:model.m-1,dx in 0:model.m-1
        pre=prepare_pattern(model,pattern_mask(translate_mask(mask,model.m,dx,dy),model.m))
        state=evolve(pre,Inf;beta=beta)
        out .+= collect(values(arrangement_entropies(state)))
    end
    return out/model.m^2
end

function select_arrangements()
    c9=filter(r->r.m==3,load_census(joinpath(DATA_DIR,"configuration_census.csv")))
    c18=Dict(r.mask=>r for r in load_census(joinpath(DATA_DIR,"configuration_census_sizes.csv")) if r.m==3 && r.L==18)
    eligible=filter(r->abs(r.excess_energy)<1e-8 && abs(c18[r.mask].excess_energy)<1e-8,c9)
    model=BlochModel(9,3)
    selected=NamedTuple[]
    open(joinpath(DATA_DIR,"arrangement_screening.csv"),"w") do io
        println(io,"family,mask,occupied_bits,double_cells,sublattice_imbalance,beta0_mean_SB,beta1_mean_SB,beta1_change_mean_SB,beta0_mean_D,beta1_mean_D")
        for (family,doubles,imbalance) in (("special",0,5),("general",1,1))
            pool=sort!(unique(canonical_translation(r.mask) for r in eligible if r.doubles==doubles && r.sublattice_imbalance==imbalance))
            total_orbits=length(pool)
            if length(pool)>64
                shuffle!(MersenneTwister(CENSUS_SEED),pool)
                resize!(pool,64)
                sort!(pool)
            end
            candidates=NamedTuple[]
            for mask in pool
                s0,s1=phase_average(model,mask,0.0),phase_average(model,mask,1.0)
                score=s1[3]-s0[3]
                push!(candidates,(;mask,score))
                println(io,join((family,mask,join(Int.(pattern_mask(mask,3))),doubles,imbalance,s0[3],s1[3],score,s0[4],s1[4]),','))
            end
            sort!(candidates;by=r->r.score)
            for (suffix,entry) in (("a",first(candidates)),("b",last(candidates)))
                push!(selected,(;name=family*"_matched_"*suffix,family,mask=entry.mask,
                    occupied=pattern_mask(entry.mask,3),screened_orbits=length(pool),eligible_orbits=total_orbits))
            end
        end
    end
    open(joinpath(DATA_DIR,"arrangement_selection.csv"),"w") do io
        println(io,"pattern,family,m,mask,occupied_bits,double_cells,empty_cells,sublattice_imbalance,screened_orbits,eligible_orbits")
        for p in selected
            stats=pattern_statistics(p.occupied)
            println(io,join((p.name,p.family,3,p.mask,join(Int.(p.occupied)),stats.double_cells,
                stats.empty_cells,stats.sublattice_imbalance,p.screened_orbits,p.eligible_orbits),','))
        end
    end
    return selected
end

function shifted_indices(L,indices,dx,dy)
    [begin
        cell=(i-1)÷2;s=mod1(i,2)
        x,y=mod(cell,L),cell÷L
        s+2*(mod(x+dx,L)+L*mod(y+dy,L))
    end for i in indices]
end

function run_arrangement_controls()
    mkpath(DATA_DIR)
    selected=select_arrangements()
    open(joinpath(DATA_DIR,"arrangement_controls.csv"),"w") do io
        println(io,"pattern,family,m,L,mask,complement,shift_x,shift_y,beta,excess_energy_before_measurement,S_A,S_C,S_B,D,empty_occupancy")
        for L in (9,18), p in selected
            model=BlochModel(L,3)
            for complement in (false,true),dy in 0:2,dx in 0:2
                mask=translate_mask(p.mask,3,dx,dy)
                complement && (mask=((1<<18)-1) ⊻ mask)
                pre=prepare_pattern(model,pattern_mask(mask,3))
                base=evolve(pre,Inf)
                excess=diagnostics(base).excess_energy
                @assert abs(excess)<1e-8
                for beta in (0.0,0.5,1.0,2.0)
                    state=measure(base,beta)
                    ss=arrangement_entropies(state)
                    println(io,join((p.name,p.family,3,L,mask,complement,dx,dy,beta,excess,
                        values(ss)...,diagnostics(state).empty_occupancy),','))
                end
            end
            println("arrangement controls complete $(p.name) L=$L")
            flush(stdout)
        end
    end
    # Explicitly translate the measurement region WITH the state for one shift.
    # Fixed-region translated-pattern entropies need not agree; this must agree.
    open(joinpath(DATA_DIR,"arrangement_covariance.csv"),"w") do io
        println(io,"pattern,L,beta,base_SB,translated_state_shifted_region_SB,absolute_error")
        for p in selected,L in (9,18),beta in (0.0,1.0,2.0)
            model=BlochModel(L,3)
            base=evolve(prepare_pattern(model,p.occupied),Inf;beta=beta)
            moved=evolve(prepare_pattern(model,pattern_mask(translate_mask(p.mask,3,1,1),3)),Inf;beta=beta)
            indices=rectangle_indices(L)
            s0=entropy_rectangle(base)
            s1=entropy2(correlation_matrix(moved,shifted_indices(L,indices,1,1)))
            @assert abs(s0-s1)<1e-9
            println(io,join((p.name,L,beta,s0,s1,abs(s0-s1)),','))
        end
    end
end

function summarize_arrangement_controls()
    lines=readlines(joinpath(DATA_DIR,"arrangement_controls.csv"))
    header=split(lines[1],',')
    pos=Dict(name=>i for (i,name) in enumerate(header))
    groups=Dict{Tuple{String,Int,Float64,Bool},Vector{Vector{Float64}}}()
    lookup=Dict{Tuple{String,Int,Float64,Bool,Int,Int},Vector{Float64}}()
    for line in lines[2:end]
        row=split(line,',')
        name=String(row[pos["pattern"]]);L=parse(Int,row[pos["L"]])
        beta=parse(Float64,row[pos["beta"]]);comp=parse(Bool,row[pos["complement"]])
        dx,dy=parse(Int,row[pos["shift_x"]]),parse(Int,row[pos["shift_y"]])
        values=[parse(Float64,row[pos[field]]) for field in ("S_A","S_C","S_B","D")]
        push!(get!(groups,(name,L,beta,comp),Vector{Float64}[]),values)
        lookup[(name,L,beta,comp,dx,dy)]=values
    end
    open(joinpath(DATA_DIR,"arrangement_phase_averages.csv"),"w") do io
        println(io,"pattern,L,beta,complement,phase_count,mean_S_A,mean_S_C,mean_S_B,mean_D,phase_range_S_A,phase_range_S_C,phase_range_S_B,phase_range_D")
        for key in sort!(collect(keys(groups)))
            values=groups[key]
            averages=sum(values)/length(values)
            spans=[maximum(v[j] for v in values)-minimum(v[j] for v in values) for j in 1:4]
            println(io,join((key...,length(values),averages...,spans...),','))
        end
    end
    complement_error=0.0
    for (key,value) in lookup
        name,L,beta,comp,dx,dy=key
        comp && continue
        complement_error=max(complement_error,maximum(abs,value-lookup[(name,L,beta,true,dx,dy)]))
    end
    @assert complement_error<1e-9
    covlines=readlines(joinpath(DATA_DIR,"arrangement_covariance.csv"))[2:end]
    covariance_error=maximum(parse(Float64,last(split(line,','))) for line in covlines)
    open(joinpath(DATA_DIR,"arrangement_validation.csv"),"w") do io
        println(io,"raw_rows,phase_average_rows,max_complement_entropy_error,max_translation_covariance_error")
        println(io,join((length(lines)-1,length(groups),complement_error,covariance_error),','))
    end
end

if abspath(PROGRAM_FILE)==@__FILE__
    "--summarize-only" in ARGS || run_arrangement_controls()
    summarize_arrangement_controls()
end
