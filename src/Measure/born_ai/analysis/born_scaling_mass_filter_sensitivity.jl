# Deterministic paired-E regulator sensitivity after finite-beta filtering.
# The eight random E sets are fixed test inputs, not Monte Carlo samples.
using LinearAlgebra, Random, SHA, TOML
BLAS.set_num_threads(1)
include(joinpath(@__DIR__,"..","original_swap_scaling.jl"))
using .OriginalSwapScaling

const MASS_FILTER_OUT=@__DIR__
filter_sha(p)=bytes2hex(sha256(read(p)))

function stable_log_weight(Q,E,beta)
    filtered=copy(Q); filtered[E,:].*=exp(-2beta)
    2sum(log,abs.(diag(qr(filtered).R)))-2sum(log,abs.(diag(qr(Q).R)))
end

function main_mass_filter()
    sources=[@__FILE__,joinpath(@__DIR__,"..","original_swap_scaling.jl")]
    source_hashes=Dict(abspath(p)=>filter_sha(p) for p in sources)
    rows=NamedTuple[]
    for L in (6,12,18)
        first=ground_model(L;state=:staggered,mu=1e-8)
        second=ground_model(L;state=:staggered,mu=1e-9)
        work=EntropyWorkspace(first)
        n=L^2; N=2n; seed=20261027+L; rng=MersenneTwister(seed)
        cases=Pair{String,Vector{Int}}[
            "uniform_A_target"=>collect(2:2:N),
            "uniform_B_target"=>collect(1:2:N),
            "pure02_stripe_x_suppressed_even_x"=>[2*(x+L*y)+s for y in 0:L-1 for x in 0:2:L-1 for s in 1:2]]
        append!(cases,["random_half_$(j)"=>sort(randperm(rng,N)[1:n]) for j in 1:8])
        for (name,E) in cases,beta in (0.5,1.5,3.0)
            @assert length(E)==length(unique(E))==n
            s1=entropy!(work,first,E,beta)
            s2=entropy!(work,second,E,beta)
            lw1=stable_log_weight(first.Q,E,beta)
            lw2=stable_log_weight(second.Q,E,beta)
            deltaD=s1.D-s2.D; deltaSA=s1.SA-s2.SA
            push!(rows,(L=L,beta=beta,case=name,seed=seed,E_indices=join(E,';'),mu_primary=1e-8,mu_secondary=1e-9,
                primary_S_A=s1.SA,secondary_S_A=s2.SA,delta_S_A=deltaSA,
                primary_S_B=s1.SB,secondary_S_B=s2.SB,delta_S_B=s1.SB-s2.SB,
                primary_S_C=s1.SC,secondary_S_C=s2.SC,delta_S_C=s1.SC-s2.SC,
                primary_D=s1.D,secondary_D=s2.D,delta_D=deltaD,
                relative_D_difference=abs(deltaD)/max(abs(s1.D),abs(s2.D),1e-12),
                primary_log_weight=lw1,secondary_log_weight=lw2,delta_log_weight=lw1-lw2,
                relative_probability_difference=expm1(lw1-lw2)))
        end
        println("L=$L fixed-E mass sensitivity complete")
        flush(stdout)
    end
    csvpath=joinpath(MASS_FILTER_OUT,"born_scaling_mass_filter_sensitivity.csv")
    open(csvpath,"w") do io
        println(io,join(string.(keys(first(rows))),','))
        for row in rows;println(io,join(values(row),','));end
    end
    @assert length(rows)==99
    @assert all(filter_sha(p)==v for (p,v) in source_hashes)
    maxima=Dict(string(field)=>maximum(abs(getproperty(r,field)) for r in rows)
        for field in (:delta_S_A,:delta_S_B,:delta_S_C,:delta_D,:delta_log_weight,:relative_probability_difference))
    receipt=Dict("status"=>"PASS","sizes"=>[6,12,18],"betas"=>[0.5,1.5,3.0],"mus"=>[1e-8,1e-9],
        "comparisons"=>length(rows),"configurations_per_L"=>11,"maxima"=>maxima,
        "sampling"=>"No Monte Carlo. Eight seeded random half sets per L are fixed deterministic test inputs, alongside uniform-A, uniform-B and pure-0/2 alternating-x sets.",
        "entropy_method"=>"The same production EntropyWorkspace is reused sequentially for both masses with identical E. Every call reloads the model orbitals before filtering and QR.",
        "probability_comparison"=>"Both Slaters have the same fixed-half analytic normalizer, hence exp(delta_log_weight)-1 is their relative probability difference for that one E.",
        "scope"=>"Only these 99 fixed-E checks. They are not Born averages and do not bound regulator sensitivity of all outcomes or establish a Monte Carlo convergence criterion.",
        "source_hashes"=>source_hashes,"files"=>Dict(basename(csvpath)=>filter_sha(csvpath),basename(@__FILE__)=>filter_sha(@__FILE__)))
    open(joinpath(MASS_FILTER_OUT,"born_scaling_mass_filter_sensitivity.toml"),"w") do io
        TOML.print(io,receipt;sorted=true)
    end
    println("PASS: finite-beta fixed-E regulator checks; maxima=",maxima)
end

main_mass_filter()
