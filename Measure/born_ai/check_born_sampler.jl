using LinearAlgebra, Random, Test, DelimitedFiles, Printf
include("BornSampler.jl")
using .BornSampler
push!(LOAD_PATH, normpath(joinpath(@__DIR__, "..", "..")))
using KAPDQMC
BLAS.set_num_threads(1)

function subsets(n, k)
    result = Vector{Vector{Int}}()
    function add(prefix, first)
        length(prefix) == k && return push!(result, copy(prefix))
        for i in first:n-(k-length(prefix))+1
            push!(prefix, i)
            add(prefix, i+1)
            pop!(prefix)
        end
        nothing
    end
    add(Int[],1)
    result
end

outdir = joinpath(@__DIR__, "analysis")
mkpath(outdir)
rng = MersenneTwister(20261027)
K = nnK_Matrix("HoneyComb120", [2,2])
P = eigen(Symmetric(K)).vectors[:,1:4]
sets = subsets(8,4)
max_latent_error = 0.0
max_normalization_error = 0.0
max_empirical_z = 0.0
@testset "normalized probabilities and independent sampler" begin
    for original in (P, randn(rng, 8,4), Matrix{Float64}(I,8,8)[:,1:4])
        for beta in (0.0,0.5,1.0,3.0,6.0)
            sampler = IIDSampler(original,beta)
            pf = [abs2(det(sampler.Q[F,:])) for F in sets]
            pe = exp.([log_probability(sampler,E) for E in sets])
            global max_normalization_error = max(max_normalization_error,abs(sum(pe)-1))
            @test isapprox(sum(pe), 1; atol=2e-12)
            for (i,E) in enumerate(sets)
                latent = sum(pf[f]*exp(-4beta*length(intersect(E,F))) for (f,F) in enumerate(sets)) /
                    exp(log_normalizer(4,beta))
                global max_latent_error = max(max_latent_error,abs(latent-pe[i]))
                @test isapprox(latent,pe[i]; atol=2e-12)
            end
        end
    end
    rows = Vector{Any}[]
    samples = 100000
    lookup = Dict(Tuple(E)=>i for (i,E) in enumerate(sets))
    for beta in (0.0,0.5,1.0,3.0)
        sampler = IIDSampler(P,beta)
        pe = exp.([log_probability(sampler,E) for E in sets])
        counts = zeros(Int,length(sets))
        for _ in 1:samples
            E = sample_measurement(sampler; rng)
            counts[lookup[Tuple(E)]] += 1
        end
        for i in eachindex(sets)
            observed = counts[i]/samples
            z = (observed-pe[i])/sqrt(pe[i]*(1-pe[i])/samples)
            global max_empirical_z = max(max_empirical_z,abs(z))
            push!(rows, [beta,i,join(sets[i],";"),pe[i],counts[i],samples,z])
        end
    end
    @test max_empirical_z < 6.0
    writedlm(joinpath(outdir,"iid_frequency_check.csv"),
        vcat(permutedims(["beta","state_id","E","exact_probability","count","samples","z"]),
             permutedims.(rows)...), ',')
end

benchmark = Vector{Any}[]
for L in (6,12,18)
    H = nnK_Matrix("HoneyComb120", [L,L])
    Q = eigen(Symmetric(H)).vectors[:,1:L^2]
    sampler = IIDSampler(Q,3.0)
    sample_measurement(sampler;rng)
    t = @elapsed for _ in 1:20
        sample_measurement(sampler;rng)
    end
    push!(benchmark,[L,2L^2,20,t,t/20])
end
writedlm(joinpath(outdir,"iid_benchmark.csv"),
    vcat(permutedims(["L","Nsites","samples","seconds","seconds_per_sample"]),
         permutedims.(benchmark)...), ',')
open(joinpath(outdir,"iid_validation.txt"),"w") do io
    println(io,"PASS: deterministic exact latent-mixture identity, normalization, and empirical samples")
    println(io,"max_latent_probability_error=",max_latent_error)
    println(io,"max_normalization_error=",max_normalization_error)
    println(io,"max_single_state_empirical_z=",max_empirical_z)
    println(io,"empirical_samples=400000; seed=20261027; Julia=",VERSION)
    println(io,"BLAS_threads=",BLAS.get_num_threads())
end
println(read(joinpath(outdir,"iid_validation.txt"),String))
