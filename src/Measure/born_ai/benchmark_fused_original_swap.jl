# Independent implementation experiment only. This does not alter the frozen sampler.
using LinearAlgebra, Random, Statistics, SHA, TOML, Dates
BLAS.set_num_threads(1)
include("original_swap_scaling.jl")
using .OriginalSwapScaling
const ROOT=@__DIR__
const DATA=joinpath(ROOT,"data","born_scaling")
sha(path)=bytes2hex(sha256(read(path)))

"""Original uniform E/S proposal and decisions, with the two accepted rank updates fused."""
function sweep_fused!(state,G,rng=Random.default_rng())
    n=length(state.E); lambda=state.lambda; accepted=0
    for attempt in 1:2n
        i=rand(rng,1:n); j=rand(rng,1:n); s=state.S[i]
        @inbounds for k in 1:n
            state.g[k]=G[state.E[k],s]
            state.grow[k]=G[s,state.E[k]]
        end
        mul!(state.xg,state.X,state.g)
        xjj=state.X[j,j]; f=state.xg[j]/xjj
        @inbounds for k in 1:n; state.z[k]=state.xg[k]-f*state.X[k,j]; end
        state.z[j]=0.
        delta=1-lambda*G[s,s]-lambda^2*dot(state.grow,state.z)
        ratio=delta*xjj
        isfinite(ratio) && ratio>0 && delta>0 && xjj>0 || error("invalid original-swap ratio")
        if rand(rng)<min(1.,ratio)
            @inbounds for k in 1:n
                state.column[k]=state.X[k,j]; state.row[k]=state.X[j,k]
            end
            # One pass over X, with cached pre-update column/row/z; no fastmath.
            alpha=-1/xjj; gamma=lambda^2/delta
            @inbounds for k in 1:n
                row_k=state.row[k]; z_k=state.z[k]
                @simd for ell in 1:n
                    state.X[ell,k]+=alpha*state.column[ell]*row_k+gamma*state.z[ell]*z_k
                end
            end
            f=lambda/delta
            @inbounds for k in 1:n
                state.X[k,j]=f*state.z[k]; state.X[j,k]=f*state.z[k]
            end
            state.X[j,j]=1/delta
            state.E[j],state.S[i]=state.S[i],state.E[j]
            state.log_weight+=log(ratio)
            accepted+=1
        end
    end
    (;accepted,proposed=2n)
end

function compare_trajectory(model,beta;sweeps=30)
    seed=872_000+100model.L+round(Int,10beta)
    init=MersenneTwister(seed); E=randperm(init,model.N)[1:model.n]
    reference=swap_state(model.G,E,beta); fused=swap_state(model.G,E,beta)
    rng_ref=MersenneTwister(seed+1); rng_fused=MersenneTwister(seed+1)
    maxmatrix=0.; maxlog=0.; maxrefdrift=0.; maxfuseddrift=0.; accepted=0
    for step in 1:sweeps
        a=sweep!(reference,model.G,rng_ref); b=sweep_fused!(fused,model.G,rng_fused)
        @assert a==b && reference.E==fused.E && reference.S==fused.S "same-seed trajectory differs"
        accepted+=a.accepted
        maxmatrix=max(maxmatrix,norm(reference.X-fused.X)/norm(reference.X))
        maxlog=max(maxlog,abs(reference.log_weight-fused.log_weight))
        if step%5==0
            dr=rebuild!(reference,model.G); df=rebuild!(fused,model.G)
            maxrefdrift=max(maxrefdrift,dr.inverse_drift)
            maxfuseddrift=max(maxfuseddrift,df.inverse_drift)
        end
    end
    @assert maxmatrix<1e-5 && maxlog<1e-5 && maxrefdrift<1e-5 && maxfuseddrift<1e-5
    Dict("L"=>model.L,"beta"=>beta,"mu"=>model.mu,"seed"=>seed,"sweeps"=>sweeps,
        "trajectory_equal"=>true,"accepted"=>accepted,"proposed"=>model.N*sweeps,
        "max_relative_inverse_difference"=>maxmatrix,"max_logweight_difference"=>maxlog,
        "max_original_rebuild_drift"=>maxrefdrift,"max_fused_rebuild_drift"=>maxfuseddrift)
end

function time_kernel(model,kernel,beta,seed;steps=20)
    rng=MersenneTwister(seed); E=randperm(rng,model.N)[1:model.n]
    state=swap_state(model.G,E,beta); work=EntropyWorkspace(model)
    for step in 1:5
        kernel(state,model.G,rng); entropy!(work,model,state.E,beta)
    end
    rebuild!(state,model.G)
    update=0.; entropy=0.; rebuild=0.; accepted=0; maxdrift=0.
    wall=@elapsed for step in 1:steps
        stats=nothing
        update+=@elapsed stats=kernel(state,model.G,rng)
        accepted+=stats.accepted
        if step%5==0
            drift=nothing
            rebuild+=@elapsed drift=rebuild!(state,model.G)
            maxdrift=max(maxdrift,drift.inverse_drift)
        end
        entropy+=@elapsed entropy!(work,model,state.E,beta)
    end
    (;wall,update,entropy,rebuild,accepted,maxdrift,final_E=copy(state.E),logweight=state.log_weight)
end

function main()
    mkpath(DATA); start=time()
    sources=Dict(abspath(p)=>sha(p) for p in [@__FILE__,joinpath(ROOT,"original_swap_scaling.jl")])
    validations=Dict{String,Any}[]; timings=Dict{String,Any}[]
    models=Dict{Int,GroundModel}()
    for L in (3,6,18,21,24)
        models[L]=ground_model(L;state=:staggered,mu=1e-8)
        for beta in (L<=6 ? (0.,.5,1.,2.,3.) : (1.5,3.))
            row=compare_trajectory(models[L],beta)
            push!(validations,row)
            println("VALIDATED L=$L beta=$beta matrixdiff=",row["max_relative_inverse_difference"]," drift=",row["max_fused_rebuild_drift"]); flush(stdout)
        end
    end
    path=joinpath(DATA,"benchmark_fused_original_swap.csv")
    open(path,"w") do io
        println(io,"L,beta,mu,repeat,kernel,warmup_sweeps,timed_sweeps,wall_seconds,seconds_per_sweep,update_seconds,entropy_seconds,rebuild_seconds,accepted,max_inverse_drift")
        for L in (18,21,24), repeat in 1:3
            model=models[L]; beta=1.5; seed=873_000+100L+repeat
            # Alternate ordering to reduce a systematic ordering/thermal effect.
            order=isodd(repeat) ? (("original",sweep!),("fused",sweep_fused!)) : (("fused",sweep_fused!),("original",sweep!))
            pair=Dict{String,Any}()
            for (label,kernel) in order
                result=time_kernel(model,kernel,beta,seed)
                pair[label]=result
                row=Dict("L"=>L,"beta"=>beta,"mu"=>model.mu,"repeat"=>repeat,"kernel"=>label,
                    "seed"=>seed,"timed_sweeps"=>20,"wall_seconds"=>result.wall,"seconds_per_sweep"=>result.wall/20,
                    "update_seconds"=>result.update,"entropy_seconds"=>result.entropy,"rebuild_seconds"=>result.rebuild,
                    "accepted"=>result.accepted,"max_inverse_drift"=>result.maxdrift)
                push!(timings,row)
                println(io,join((L,beta,model.mu,repeat,label,5,20,result.wall,result.wall/20,result.update,result.entropy,result.rebuild,result.accepted,result.maxdrift),',')); flush(io)
            end
            @assert pair["original"].accepted==pair["fused"].accepted && pair["original"].final_E==pair["fused"].final_E
            println("TIMED L=$L repeat=$repeat original=",pair["original"].wall/20," fused=",pair["fused"].wall/20," speedup=",pair["original"].wall/pair["fused"].wall); flush(stdout)
        end
    end
    @assert all(sha(p)==hash for (p,hash) in sources)
    summary=Dict{String,Any}[]
    for L in (18,21,24)
        ref=[r for r in timings if r["L"]==L && r["kernel"]=="original"]
        fused=[r for r in timings if r["L"]==L && r["kernel"]=="fused"]
        push!(summary,Dict("L"=>L,"original_seconds_per_sweep"=>median([r["seconds_per_sweep"] for r in ref]),
            "fused_seconds_per_sweep"=>median([r["seconds_per_sweep"] for r in fused]),
            "total_speedup"=>median([r["wall_seconds"] for r in ref])/median([r["wall_seconds"] for r in fused]),
            "update_speedup"=>median([r["update_seconds"] for r in ref])/median([r["update_seconds"] for r in fused])))
    end
    receipt=Dict("status"=>"PASS","created_at"=>string(now()),"source_sha256"=>sources,
        "csv_sha256"=>sha(path),"elapsed_seconds"=>time()-start,"blas_threads"=>1,
        "scope"=>"Independent fused rank-update benchmark only; frozen sampler and study datasets unchanged. Every timed sweep measures all three entropies; inverse refresh every five sweeps.",
        "validations"=>validations,"timings"=>timings,"summary"=>summary)
    open(io->TOML.print(io,receipt),joinpath(DATA,"benchmark_fused_original_swap.toml"),"w")
end
if abspath(PROGRAM_FILE)==@__FILE__; main(); end
