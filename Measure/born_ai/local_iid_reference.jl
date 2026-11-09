# Independent outcome reference for the fixed projectors saved by the local run.
using LinearAlgebra,Random,SHA,TOML
include(joinpath(@__DIR__,"BornSampler.jl"))
using .BornSampler
BLAS.set_num_threads(1)
const DATA=joinpath(@__DIR__,"data")
const NREF=512
sha(path)=bytes2hex(sha256(read(path)))
function metrics(E,N,starts)
    occ=falses(N);occ[E].=true;a=N÷2
    imbalance=(sum(occ[1:2:N])-sum(occ[2:2:N]))/a
    n02=count(i->occ[2i-1]==occ[2i],1:a)/a
    memory=[2sum(occ[S])/a-1 for S in starts]
    imbalance,n02,memory
end
function logweight(G,E,beta)
    beta==0 && return 0.
    B=Matrix{Float64}(I,length(E),length(E))+expm1(-4beta)*G[E,E]
    2sum(log,diag(cholesky(Symmetric(B)).L))
end
metadata=Dict{String,Any}("status"=>"running","n_per_setting"=>NREF,"seed"=>823441,
    "method"=>"Independent latent projection-DPP exact BornSampler; same saved LAPACK ground orbitals as local experiments, no burn-in and no entropy.",
    "sampler_sha256"=>sha(joinpath(@__DIR__,"BornSampler.jl")),"script_sha256"=>sha(@__FILE__),"settings"=>Any[])
open(joinpath(DATA,"local_iid_reference.csv"),"w") do io
    println(io,"L,beta,draw,seed,imbalance,n02_fraction,class_id,log_weight,log_probability,memory_uniform_A,memory_02_stripe,memory_random,seconds")
    for L in (6,12,18)
        a=L^2;N=2a
        P=reshape(reinterpret(Float64,read(joinpath(DATA,"local_orbitals_L$(L).bin"))),N,a)
        G=reshape(reinterpret(Float64,read(joinpath(DATA,"local_projector_L$(L).bin"))),N,N)
        for beta in (0.,.5,1.,2.,3.)
            sampler=IIDSampler(P,beta);seed=823441+10000L+round(Int,100beta);rng=MersenneTwister(seed)
            stripe=[i for i in 1:N if isodd(((i-1)÷2)%L+1)]
            startseed=729031+10000L+round(Int,beta*100)*10+3
            starts=[collect(1:2:N),stripe,sort(randperm(MersenneTwister(startseed),N)[1:a])]
            elapsed=@elapsed for draw in 1:NREF
                start=time();E=sample_measurement(sampler;rng);imb,n02,memory=metrics(E,N,starts);lw=logweight(G,E,beta)
                println(io,join((L,beta,draw,seed,imb,n02,n02==0 ? 1 : n02==1 ? 2 : 3,lw,lw-log_normalizer(a,beta),memory...,time()-start),','))
            end
            flush(io);println("iid done L=$L beta=$beta N=$NREF seconds=$elapsed")
            push!(metadata["settings"],Dict("L"=>L,"beta"=>beta,"samples"=>NREF,"seconds"=>elapsed,
                "P_sha256"=>sha(joinpath(DATA,"local_orbitals_L$(L).bin")),"G_sha256"=>sha(joinpath(DATA,"local_projector_L$(L).bin"))))
        end
    end
end
metadata["status"]="complete";metadata["output_sha256"]=sha(joinpath(DATA,"local_iid_reference.csv"))
open(io->TOML.print(io,metadata),joinpath(DATA,"local_iid_manifest.toml"),"w")
