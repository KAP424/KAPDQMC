# Timing only: no production or pilot samples are created by this script.
using LinearAlgebra, Random, Statistics, SHA, TOML, Dates
BLAS.set_num_threads(1)
include("original_swap_scaling.jl")
using .OriginalSwapScaling
const ROOT=@__DIR__
const DATA=joinpath(ROOT,"data","born_scaling")
sha(path)=bytes2hex(sha256(read(path)))
function main()
    started=time(); results=Dict{String,Any}[]
    path=joinpath(DATA,"benchmark_large.csv")
    sources=Dict(abspath(p)=>sha(p) for p in [@__FILE__,joinpath(ROOT,"original_swap_scaling.jl")])
    open(path,"w") do io
        println(io,"L,beta,mu,model_seconds,warmup_sweeps,timed_sweeps,wall_seconds,seconds_per_sweep,update_seconds,entropy_seconds,rebuild_seconds,acceptance,max_inverse_drift,max_logweight_drift")
        for L in (21,24)
            model=nothing
            modeltime=@elapsed model=ground_model(L;state=:staggered,mu=1e-8)
            rng=MersenneTwister(871_000+L)
            state=swap_state(model.G,randperm(rng,model.N)[1:model.n],1.5)
            work=EntropyWorkspace(model)
            for step in 1:5
                sweep!(state,model.G,rng); entropy!(work,model,state.E,1.5)
            end
            rebuild!(state,model.G)
            updates=0.; entropies=0.; rebuilds=0.; accepted=0; maxdrift=0.; maxlog=0.
            wall=@elapsed for step in 1:20
                stats=nothing
                updates+=@elapsed stats=sweep!(state,model.G,rng)
                accepted+=stats.accepted
                entropies+=@elapsed entropy!(work,model,state.E,1.5)
                if step%5==0
                    drift=nothing
                    rebuilds+=@elapsed drift=rebuild!(state,model.G)
                    maxdrift=max(maxdrift,drift.inverse_drift); maxlog=max(maxlog,drift.logweight_drift)
                end
            end
            row=Dict("L"=>L,"beta"=>1.5,"mu"=>1e-8,"model_seconds"=>modeltime,"warmup_sweeps"=>5,
                "timed_sweeps"=>20,"wall_seconds"=>wall,"seconds_per_sweep"=>wall/20,
                "update_seconds"=>updates,"entropy_seconds"=>entropies,"rebuild_seconds"=>rebuilds,
                "acceptance"=>accepted/(20model.N),"max_inverse_drift"=>maxdrift,"max_logweight_drift"=>maxlog,
                "Q_sha256"=>bytes2hex(sha256(reinterpret(UInt8,vec(model.Q)))),
                "G_sha256"=>bytes2hex(sha256(reinterpret(UInt8,vec(model.G)))))
            push!(results,row)
            println(io,join((L,1.5,1e-8,modeltime,5,20,wall,wall/20,updates,entropies,rebuilds,
                row["acceptance"],maxdrift,maxlog),',')); flush(io)
            println("L=$L beta=1.5 per_sweep=$(wall/20)s update=$(updates/20)s entropy=$(entropies/20)s rebuild=$(rebuilds/20)s")
            flush(stdout)
        end
    end
    @assert all(sha(p)==hash for (p,hash) in sources)
    receipt=Dict("status"=>"complete","created_at"=>string(now()),"source_sha256"=>sources,
        "csv_sha256"=>sha(path),"elapsed_seconds"=>time()-started,"blas_threads"=>1,
        "scope"=>"Single-process warm benchmark only, 20 sweeps including three-region entropy each sweep and rebuild every 5; no retained study samples",
        "results"=>results)
    open(io->TOML.print(io,receipt),joinpath(DATA,"benchmark_large.toml"),"w")
end
if abspath(PROGRAM_FILE)==@__FILE__; main(); end
