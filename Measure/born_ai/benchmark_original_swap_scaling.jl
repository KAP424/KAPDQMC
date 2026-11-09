# Benchmark only; no production grid or inference from short chains.
using LinearAlgebra, Random, SHA, TOML, Dates
BLAS.set_num_threads(1)
include("original_swap_scaling.jl")
using .OriginalSwapScaling
module HistoricalReference
include("Free_M.jl")
end
include("ExactBorn.jl")
const ROOT=@__DIR__
const DATA=joinpath(ROOT,"data","born_scaling")
sha(path)=bytes2hex(sha256(read(path)))
csv(io,row)=println(io,join(string.(row),','))

function validate_kernel()
    results=Dict{String,Any}[]
    for L in (3,6), beta in (0.,1.,3.)
        model=ground_model(L); seed=61_000+100L+Int(10beta)
        ref=HistoricalReference.Measure_EE(beta,model.N,model.n;rng=MersenneTwister(seed))
        fast=swap_state(model.G,ref.E_idx,beta)
        ref.X.=fast.X
        r1=MersenneTwister(seed+1); r2=MersenneTwister(seed+1)
        tmp=zeros(model.n); tmp2=zeros(model.n,model.n)
        maximum_error=0.; accepted=0
        for sweep in 1:30
            if (sweep-1)%5==0
                rebuild!(fast,model.G)
                ref.X=inv(cholesky(Symmetric(I(model.n)-ref.λ*model.G[ref.E_idx,ref.E_idx])))
            end
            a=HistoricalReference.UpdateMeasure!(ref,model.G,tmp,tmp2,model.Q;rng=r1)
            b=sweep!(fast,model.G,r2)
            @assert a.accepted==b.accepted && ref.E_idx==fast.E && ref.S_idx==fast.S
            maximum_error=max(maximum_error,norm(ref.X-fast.X)/norm(ref.X))
            accepted+=a.accepted
        end
        @assert maximum_error<1e-8
        push!(results,Dict("L"=>L,"beta"=>beta,"sweeps"=>30,"seed"=>seed,"accepted"=>accepted,
                          "matching_E_S_and_acceptance_every_sweep"=>true,"max_inverse_relative_difference"=>maximum_error))
    end
    results
end

function reference_entropy(model,E,beta)
    weights=ones(model.N); weights[E].=exp(-2beta)
    Q=Matrix(qr(weights.*model.Q).Q)[:,1:model.n]
    [ExactBorn.regional_entropy(Q,r,model.N) for r in values(region_indices(model.L))]
end

function main()
    mkpath(DATA)
    sources=[@__FILE__,joinpath(ROOT,"original_swap_scaling.jl"),joinpath(ROOT,"Free_M.jl"),
             joinpath(ROOT,"ExactBorn.jl"),normpath(joinpath(ROOT,"../../public/Geometry.jl"))]
    hashes=Dict(abspath(path)=>sha(path) for path in sources)
    started=time(); validation=validate_kernel(); results=Dict{String,Any}[]; saved=String[]
    timing_path=joinpath(DATA,"original_swap_benchmark.csv")
    open(timing_path,"w") do io
        csv(io,("L","beta","update_sweeps","proposals_per_sweep","update_seconds","update_seconds_per_sweep",
                "rebuild_seconds","rebuild_count","entropy_samples","entropy_seconds","entropy_seconds_per_sample",
                "acceptance","max_inverse_drift","max_logweight_drift","max_entropy_reference_error"))
        for L in (6,9,12,15,18)
            model=ground_model(L); append!(saved,save_model(model,DATA;prefix="benchmark_original_swap"))
            work=EntropyWorkspace(model)
            for beta in (1.,3.)
                seed=70_000+100L+Int(10beta); rng=MersenneTwister(seed)
                state=swap_state(model.G,randperm(rng,model.N)[1:model.n],beta)
                # Compile and warm up before starting the measured intervals.
                for step in 1:5; sweep!(state,model.G,rng); end
                rebuild!(state,model.G); entropy!(work,model,state.E,beta)
                updates=0.; rebuilds=0.; maxdrift=0.; maxlog=0.; accepted=0
                for step in 1:100
                    stats=nothing
                    updates+=@elapsed stats=sweep!(state,model.G,rng)
                    accepted+=stats.accepted
                    if step%5==0
                        drift=nothing
                        rebuilds+=@elapsed drift=rebuild!(state,model.G)
                        maxdrift=max(maxdrift,drift.inverse_drift); maxlog=max(maxlog,drift.logweight_drift)
                    end
                end
                maxdrift<1e-5 || error("benchmark inverse drift exceeded tolerance")
                enttime=0.; maxerror=0.
                for sample in 1:10
                    sweep!(state,model.G,rng)
                    sample%5==0 && rebuild!(state,model.G)
                    observed=nothing
                    enttime+=@elapsed observed=entropy!(work,model,state.E,beta)
                    if sample in (1,10)
                        reference=reference_entropy(model,state.E,beta)
                        maxerror=max(maxerror,maximum(abs.([observed.SA,observed.SB,observed.SC]-reference)))
                    end
                end
                maxerror<1e-9 || error("purity Cholesky entropy differs from ExactBorn reference")
                row=Dict("L"=>L,"beta"=>beta,"seed"=>seed,"update_sweeps"=>100,"proposals_per_sweep"=>model.N,
                    "update_seconds"=>updates,"update_seconds_per_sweep"=>updates/100,
                    "rebuild_seconds"=>rebuilds,"rebuild_count"=>20,"entropy_samples"=>10,"entropy_seconds"=>enttime,
                    "entropy_seconds_per_sample"=>enttime/10,"acceptance"=>accepted/(100model.N),
                    "max_inverse_drift"=>maxdrift,"max_logweight_drift"=>maxlog,"max_entropy_reference_error"=>maxerror)
                push!(results,row)
                csv(io,(L,beta,100,model.N,updates,updates/100,rebuilds,20,10,enttime,enttime/10,
                        row["acceptance"],maxdrift,maxlog,maxerror))
                flush(io)
                println("L=$L beta=$beta update=$(round(updates/100,digits=6))s rebuild5=$(round(rebuilds/100,digits=6))s entropy3=$(round(enttime/10,digits=6))s acceptance=$(round(row["acceptance"],digits=4))")
                flush(stdout)
            end
        end
    end
    @assert all(sha(path)==hash for (path,hash) in hashes)
    push!(saved,timing_path)
    receipt=Dict("status"=>"complete","audit_status"=>"PASS","created_at"=>string(now()),"elapsed_seconds"=>time()-started,
        "blas_threads"=>BLAS.get_num_threads(),"julia_version"=>string(VERSION),"zero_mode_rule"=>"historical LAPACK first n orbitals",
        "protocol"=>"Original uniform E/S swap plus Metropolis, 2L^2 attempts per sweep; no Gibbs or importance sampling",
        "timing_scope"=>"Warmup excluded. update/rebuild/three-region entropy timings separate; short-run acceptance is not a mixing certificate",
        "entropy_method"=>"Weighted QR, .5 I + 2(C-.5 I)(C-.5 I)' purity, Cholesky; selected samples independently checked against ExactBorn regional eigenvalue routine",
        "source_sha256"=>hashes,"files"=>Dict(replace(relpath(p,ROOT),'\\'=>'/')=>sha(p) for p in saved),
        "kernel_validation"=>validation,"timing_results"=>results)
    open(io->TOML.print(io,receipt),joinpath(DATA,"original_swap_benchmark.toml"),"w")
    println("PASS elapsed=$(time()-started)s")
end
if abspath(PROGRAM_FILE)==@__FILE__; main(); end
