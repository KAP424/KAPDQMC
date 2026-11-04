using LinearAlgebra, Random, SHA, TOML, Dates, Test, Statistics
BLAS.set_num_threads(1)
include("StratifiedBorn.jl")
using .StratifiedBorn
include("../ExactBorn.jl")
using .ExactBorn

const HERE=@__DIR__
const OUT=joinpath(HERE,"validation")
isdir(OUT) && !isempty(readdir(OUT)) && error("Refuse to overwrite existing validation")
mkpath(OUT)
sha(p)=bytes2hex(sha256(read(p)))
Qpath=normpath(joinpath(HERE,"../data/local_orbitals_L2.bin"))
sha(Qpath)=="92f3c634d56097f0e7333664c8410fa8fd939312ecd5333483a57e61df995b21" || error("L2 reference orbitals changed")
Q=Matrix{Float64}(undef,8,4);open(io->read!(io,Q),Qpath,"r")
regions=(A=[1,2,3,4],B=[1,2],C=[1,2,5,6])
beta=3.;s=StratifiedSampler(Q,beta)
exact=born_trajectory_entropy(Q,beta,regions)
sets=exact.configurations;lookup=Dict(Tuple(E)=>i for (i,E) in enumerate(sets))
pf=[abs2(det(s.iid.Q[F,:])) for F in sets]
conditional=zeros(length(sets),5)
for (fi,F) in enumerate(sets),(ei,E) in enumerate(sets)
    j=length(intersect(E,F));conditional[ei,j+1]+=pf[fi]/binomial(4,j)^2
end
D=(2exact.entropies[:,1]+exact.entropies[:,3])/3-exact.entropies[:,2]
observations=hcat(exact.entropies,D)
weights=[stratum_probability(s,j) for j in 0:4]
max_marginal=maximum(abs.(conditional*weights-exact.probabilities))
max_mass=maximum(abs.(vec(sum(conditional;dims=1)).-1))
results=Dict{String,Any}[]
frequencies=Dict{String,Any}[]
rows_path=joinpath(OUT,"draws.csv")
open(rows_path,"w") do io
    println(io,"method,stratum,sample,seed,j,F,E,record_sha256,SA,SB,SC,D")
    @testset "Deterministic mixture, endpoints, tail support, empirical conditional samples" begin
        @test max_marginal<2e-12
        @test max_mass<2e-12
        @test isapprox(sum(weights),1.;atol=2e-14)
        @test_throws ArgumentError sample_fixed_j(s,-1)
        @test_throws ArgumentError sample_stratum(s,3,5)
        for (si,(jlo,jhi)) in enumerate(((0,0),(1,1),(2,2),(3,3),(4,4),(2,4),(0,4)))
            seed=291029100+si;rng=MersenneTwister(seed);n=6000
            counts=zeros(Int,70);jcounts=zeros(Int,5)
            values=zeros(n,4)
            for k in 1:n
                draw=sample_stratum(s,jlo,jhi;rng);e=lookup[Tuple(draw.E)]
                @test jlo<=draw.j<=jhi
                jlo==jhi==0 && @test draw.E==setdiff(1:8,draw.F)
                jlo==jhi==4 && @test draw.E==draw.F
                counts[e]+=1;jcounts[draw.j+1]+=1;values[k,:].=observations[e,:]
                println(io,join(("stratum","$(jlo):$(jhi)",k,seed,draw.j,join(draw.F,';'),join(draw.E,';'),record_digest(draw.F,draw.E,draw.j),values[k,:]...),','))
            end
            pw=weights[jlo+1:jhi+1];pw=pw/sum(pw)
            expected=conditional[:,jlo+1:jhi+1]*pw
            mu=vec(expected'*observations)
            variance=vec(expected'*((observations.-mu').^2))
            z=[abs(counts[k]/n-expected[k])/sqrt(expected[k]*(1-expected[k])/n) for k in 1:70 if 1e-10<expected[k]<1-1e-10]
            @test maximum(z)<7
            push!(frequencies,Dict("stratum"=>"$(jlo):$(jhi)","n"=>n,"seed"=>seed,"max_marginal_frequency_z"=>maximum(z),"j_counts"=>jcounts))
            push!(results,Dict("method"=>"stratum","jlo"=>jlo,"jhi"=>jhi,"n"=>n,"exact_mean"=>mu,"sample_mean"=>vec(mean(values;dims=1)),"true_mean_sd"=>sqrt.(variance/n),"empirical_SE"=>vec(std(values;dims=1))/sqrt(n)))
        end
        # Call the actual existing IID API, not just the new full-range wrapper.
        n=30000;seed=291029199;rng=MersenneTwister(seed);values=zeros(n,4)
        for k in 1:n
            E=BornSampler.sample_measurement(s.iid;rng);e=lookup[Tuple(E)]
            values[k,:].=observations[e,:]
            println(io,join(("original_iid","all",k,seed,-1,"",join(E,';'),record_digest(Int[],E,-1),values[k,:]...),','))
        end
        mu=vec(exact.probabilities'*observations)
        variance=vec(exact.probabilities'*((observations.-mu').^2))
        push!(results,Dict("method"=>"original_iid","n"=>n,"seed"=>seed,"exact_mean"=>mu,"sample_mean"=>vec(mean(values;dims=1)),"true_mean_sd"=>sqrt.(variance/n),"empirical_SE"=>vec(std(values;dims=1))/sqrt(n)))
        # Degenerate projector and beta=0 remain valid for every j stratum.
        product=StratifiedSampler(Matrix{Float64}(I,8,8)[:,1:4],0.)
        for j in 0:4
            x=sample_fixed_j(product,j;rng)
            @test x.F==collect(1:4)
            @test length(intersect(x.E,x.F))==j
        end
    end
end
receipt=Dict("status"=>"PASS","created_at"=>string(now()),"arithmetic"=>"Float64; exact refers to the distribution identity, not arbitrary precision",
    "beta"=>beta,"N"=>8,"a"=>4,"regions"=>Dict(string(k)=>v for (k,v) in pairs(regions)),
    "observables"=>["SA","SB","SC","D"],"maximum_mixture_probability_error"=>max_marginal,
    "maximum_conditional_mass_error"=>max_mass,"j_probabilities"=>weights,"frequency_checks"=>frequencies,"mean_checks"=>results,
    "empirical_check_limit"=>"A finite frequency check is not proof of rare-tail coverage; true finite-enumeration variance is separately reported.",
    "files"=>Dict(replace(relpath(p,HERE),'\\'=>'/')=>sha(p) for p in (Qpath,@__FILE__,joinpath(HERE,"StratifiedBorn.jl"),joinpath(HERE,"../BornSampler.jl"),joinpath(HERE,"../ExactBorn.jl"),rows_path)))
open(io->TOML.print(io,receipt),joinpath(OUT,"receipt.toml"),"w")
println("PASS validation mixture_error=$max_marginal conditional_mass_error=$max_mass")
