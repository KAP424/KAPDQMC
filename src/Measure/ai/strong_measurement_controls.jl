#!/usr/bin/env julia
#=
Fixed-finite-L strong-measurement test for the initial-state protocol.
For graph orbitals [I;tX], t=exp(-2beta), the leading single-species Renyi-2
coefficient for region R is
2(||X[E in R,F outside R]||_F^2+||X[E outside R,F in R]||_F^2).
Small test lattices only: L=6 for m<=2, L=9 for m=3.
=#
using LinearAlgebra
BLAS.set_num_threads(1)
include(joinpath(@__DIR__,"fermion_experiment.jl"))
const FE=FermionExperiment
include(joinpath(@__DIR__,"experiment_patterns.jl"))
include(joinpath(@__DIR__,"arrangement_patterns.jl"))

function full_orbitals(state)
    model=state.prepared.model
    p=model.m^2
    out=zeros(ComplexF64,2model.L^2,model.L^2)
    for i in axes(out,1)
        a,rx,ry=FE.site_coordinates(model,i)
        for ky in 0:model.q-1,kx in 0:model.q-1
            k=1+kx+model.q*ky
            phase=cis(2pi*(rx*kx+ry*ky)/model.q)/model.q
            out[i,(k-1)*p+1:k*p]=phase .* state.orbitals[k][a,:]
        end
    end
    return out
end

function graph_coefficient(state,indices)
    model=state.prepared.model
    q=full_orbitals(state)
    f=FE.occupied_indices(state.prepared)
    e=setdiff(1:2model.L^2,f)
    x=q[e,:]/q[f,:]
    region=Set(indices)
    fr=map(in(region),f)
    er=map(in(region),e)
    coefficient=2(sum(abs2,x[er,.!fr])+sum(abs2,x[.!er,fr]))
    return coefficient,opnorm(x),minimum(svdvals(q[f,:]))
end

function small_strong_controls()
    dir=joinpath(@__DIR__,"data")
    mkpath(dir)
    open(joinpath(dir,"strong_measurement_controls.csv"),"w") do io
        println(io,"pattern,family,m,L,beta,base_excess_energy,min_F_singular,X_operator_norm,t_Xnorm,leading_coefficient,SB_cross,SB_eigen,SB_scaled_by_exp4beta,relative_error_to_leading,empty_occupancy")
        for pattern in vcat(experiment_patterns(),matched_arrangement_patterns())
            L=pattern.m==3 ? 9 : 6
            model=FE.BlochModel(L,pattern.m)
            base=FE.evolve(FE.prepare_pattern(model,pattern.occupied),Inf)
            base_excess=FE.diagnostics(base).excess_energy
            indices=FE.rectangle_indices(L)
            outside=setdiff(1:2L^2,indices)
            coefficient,xnorm,smin=graph_coefficient(base,indices)
            for beta in (2.0,3.0,4.0,6.0,8.0)
                state=FE.measure(base,beta)
                c=FE.correlation_matrix(state)
                singular=svdvals(c[indices,outside])
                sb=-sum(log1p.(-2 .* singular.^2))
                sb_eigen=FE.entropy2(c[indices,indices])
                scaled=sb*exp(4beta)
                relative=scaled/coefficient-1
                println(io,join((pattern.name,pattern.family,pattern.m,L,beta,base_excess,smin,
                    xnorm,exp(-2beta)*xnorm,coefficient,sb,sb_eigen,scaled,relative,
                    FE.diagnostics(state).empty_occupancy),','))
            end
        end
    end
end

function previous_strong_pairs()
    dir=joinpath(@__DIR__,"data")
    lines=readlines(joinpath(dir,"observations.csv"))
    names=split(lines[1],',')
    pos=Dict(name=>i for (i,name) in enumerate(names))
    rows=Dict{Tuple{String,Int,Float64},Vector{SubString{String}}}()
    for line in lines[2:end]
        row=split(line,',')
        row[pos["theta"]]=="Inf" || continue
        beta=parse(Float64,row[pos["beta"]])
        beta in (4.0,8.0) || continue
        rows[(String(row[pos["pattern"]]),parse(Int,row[pos["L"]]),beta)]=row
    end
    open(joinpath(dir,"strong_measurement_previous_pairs.csv"),"w") do io
        println(io,"pattern,L,region,S_beta4,S_beta8,beta_decay_slope,scaled_ratio_beta8_to_beta4")
        for (name,L,beta) in sort!(collect(keys(rows)))
            beta==8 || continue
            r4=rows[(name,L,4.0)];r8=rows[(name,L,8.0)]
            for region in ("S_A","S_C","S_B")
                s4=parse(Float64,r4[pos[region]])
                s8=parse(Float64,r8[pos[region]])
                println(io,join((name,L,region,s4,s8,-log(s8/s4)/4,s8/s4*exp(16)),','))
            end
        end
    end
end

function uniform_A_kernel_check()
    dir=joinpath(@__DIR__,"data")
    open(joinpath(dir,"uniform_A_measurement_kernel.csv"),"w") do io
        println(io,"L,beta,nonzero_k_checked,max_covariance_error,empty_occupancy,predicted_empty_occupancy")
        for L in (6,9,12),beta in (0.0,0.5,1.0,2.0,4.0)
            model=FE.BlochModel(L,1)
            state=FE.evolve(FE.prepare_pattern(model,Bool[1,0]),Inf;beta=beta)
            err=0.0;checked=0
            for (h,q) in zip(model.hamiltonians,state.orbitals)
                f=h[1,2]
                abs(f)>1e-10 || continue # exact zero modes retain initial A occupation
                expected=[(1+tanh(2beta))/2 -f/abs(f)/(2cosh(2beta));
                    -conj(f)/abs(f)/(2cosh(2beta)) (1-tanh(2beta))/2]
                err=max(err,maximum(abs,q*q'-expected))
                checked+=1
            end
            @assert err<1e-12
            empty=FE.diagnostics(state).empty_occupancy
            predicted=(L^2-2)*exp(-4beta)/(1+exp(-4beta))
            @assert isapprox(empty,predicted;atol=1e-11,rtol=1e-12)
            println(io,join((L,beta,checked,err,empty,predicted),','))
        end
    end
end

if abspath(PROGRAM_FILE)==@__FILE__
    small_strong_controls()
    previous_strong_pairs()
    uniform_A_kernel_check()
end
