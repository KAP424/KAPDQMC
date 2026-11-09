# Supplemental check of the COMPLETE local POVM, distinct from fixed_half.
# Run: julia --project=. src/Measure/born_ai/compare_full_local_born.jl
include("compare_exact_born.jl")

function run_full_local_control()
    outdir=joinpath(@__DIR__,"data","exact_born_full")
    mkpath(outdir)
    nsamples=30_000  # fixed before inspecting the Monte Carlo output
    case=setup_case(2); P=case.P; N,a=size(P)
    Q=Matrix(qr(P).Q)[:,1:a]
    sources=[@__FILE__,joinpath(@__DIR__,"compare_exact_born.jl"),
             joinpath(@__DIR__,"ExactBorn.jl"),joinpath(@__DIR__,"BornSampler.jl"),
             joinpath(@__DIR__,"Free_M.jl"),joinpath(@__DIR__,"..","..","public","Geometry.jl")]
    source_hashes=Dict(abspath(p)=>sha(p) for p in sources)
    runs=Dict{String,Any}[]
    # Independent occupation-basis expression for checking the sampler's law.
    occupied=Vector{Vector{Int}}(); occupation_p=Float64[]
    ExactBorn.foreach_subset(N,a) do F
        push!(occupied,copy(F)); push!(occupation_p,abs2(det(Q[F,:])))
    end
    @assert abs(sum(occupation_p)-1)<1e-12
    open(joinpath(outdir,"summary.csv"),"w") do summary
        csv(summary,("L","beta","ensemble","observable","exact_mean","iid_mean",
                     "iid_SE","exact_distribution_SE","difference","z_exact_distribution","samples"))
        for beta in BETAS
            exact=ExactBorn.born_trajectory_entropy(P,beta,case.regions;ensemble=:full_local)
            lookup=Dict(configuration_key(E)=>i for (i,E) in enumerate(exact.configurations))
            exact_values=reduce(vcat,[permutedims(observables(view(exact.entropies,i,:)))
                                     for i in eachindex(exact.probabilities)])
            truth=vec(exact_values'*exact.probabilities)
            variance=vec(sum(exact.probabilities.*(exact_values.-permutedims(truth)).^2;dims=1))
            r=exp(-2beta)/(1+exp(-2beta))
            mixture_error=0.
            for (i,E) in enumerate(exact.configurations)
                p=0.
                for (F,pF) in zip(occupied,occupation_p)
                    joint=1.
                    for site in 1:N
                        pminus=site in F ? r : 1-r
                        joint*=site in E ? pminus : 1-pminus
                    end
                    p+=pF*joint
                end
                mixture_error=max(mixture_error,abs(p-exact.probabilities[i]))
            end
            @assert mixture_error<1e-12
            seed=31_000_000+round(Int,10_000beta)
            rng=MersenneTwister(seed); counts=zeros(Int,exact.nconfigurations)
            values=zeros(nsamples,5); entropy_error=0.
            open(joinpath(outdir,"L2_beta$(tag(beta))_samples.csv"),"w") do samples
                csv(samples,("sample","E_indices",OBSERVABLES...))
                for sample in 1:nsamples
                    # F is auxiliary; the physical entropy below always uses P.
                    F=BornSampler.sample_projection(rng,Q)
                    E=[site for site in 1:N if rand(rng)<(site in F ? r : 1-r)]
                    i=lookup[configuration_key(E)]; counts[i]+=1
                    s=entropy_reference(P,E,beta,case.regions)
                    values[sample,:].=s
                    entropy_error=max(entropy_error,maximum(abs.(s-exact_values[i,:])))
                    csv(samples,(sample,configuration_key(E),s...))
                end
            end
            @assert entropy_error<1e-10
            open(joinpath(outdir,"L2_beta$(tag(beta))_trajectories.csv"),"w") do trajectories
                csv(trajectories,("E_indices","cardinality","probability",OBSERVABLES...,"iid_count"))
                for (i,E) in enumerate(exact.configurations)
                    csv(trajectories,(configuration_key(E),length(E),exact.probabilities[i],exact_values[i,:]...,counts[i]))
                end
            end
            for (j,name) in enumerate(OBSERVABLES)
                mu=mean(values[:,j]); empirical=std(values[:,j])/sqrt(nsamples)
                oracle=sqrt(variance[j]/nsamples)
                z=oracle>1e-12 ? (mu-truth[j])/oracle : ""
                csv(summary,(2,beta,"full_local",name,truth[j],mu,empirical,oracle,mu-truth[j],z,nsamples))
            end
            push!(runs,Dict("beta"=>beta,"seed"=>seed,"n"=>nsamples,
                "nconfigurations"=>exact.nconfigurations,"normalization_error"=>exact.normalization_error,
                "maximum_mixture_probability_error"=>mixture_error,
                "maximum_entropy_reference_error"=>entropy_error))
            println("full_local L=2 beta=$beta complete: $(exact.nconfigurations) outcomes, $nsamples independent samples")
        end
    end
    @assert all(sha(path)==hash for (path,hash) in source_hashes)
    outputs=Dict(basename(p)=>sha(p) for p in readdir(outdir;join=true) if endswith(p,".csv"))
    open(joinpath(outdir,"manifest.toml"),"w") do io
        TOML.print(io,Dict("status"=>"complete","numerical_audit"=>"PASS",
            "ensemble"=>"full_local","L"=>2,"samples_per_beta"=>nsamples,
            "region_A"=>case.regions.A,"region_B"=>case.regions.B,"region_C"=>case.regions.C,
            "ground_orbital_sha256"=>bytes2hex(sha256(reinterpret(UInt8,vec(P)))),
            "sources"=>source_hashes,"outputs"=>outputs,"runs"=>runs);sorted=true)
    end
end

if abspath(PROGRAM_FILE)==@__FILE__; run_full_local_control(); end
