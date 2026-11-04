# Timing only: no observations from this script enter any physical estimate.
include("run_born_scaling.jl")
include("fused_original_swap_kernel.jl")

function throughput_main()
    stage = "pilot_beta3_large"
    planpath = joinpath(ROOT,"data","born_scaling",stage,"$(stage)_plan.toml")
    plan = validate_plan(planpath)
    worker = parse(Int,arg("worker","1"))
    benchmark_dir = abspath(arg("output",""))
    isempty(arg("output","")) && error("need --output")
    iterations = parse(Int,arg("timed-sweeps","20"))
    job = only(filter(j->j["L"]==36 && j["chain"]==worker,plan["jobs"]))
    previous = TOML.parsefile(paths_for(job).receipt)
    previous["status"]=="complete" || error("pilot must finish before throughput benchmark")
    model = load_model(job)
    seed = 2_100_000_000+worker
    rng = MersenneTwister(seed)
    state = swap_state(model.G,Int.(previous["final_E"]),3.)
    work = EntropyWorkspace(model)
    for step in 1:5
        FusedOriginalSwapKernel.sweep_fused!(state,model.G,rng)
        entropy!(work,model,state.E,3.)
    end
    rebuild!(state,model.G)
    open(joinpath(benchmark_dir,"worker_$(worker)_ready"),"w") do io
        println(io,"ready")
    end
    started_wait = time()
    while !isfile(joinpath(benchmark_dir,"go"))
        time()-started_wait<300 || error("benchmark barrier timeout")
        sleep(.02)
    end
    update_seconds=0.; entropy_seconds=0.; rebuild_seconds=0.
    accepted=0; maximum_inverse_drift=0.; maximum_logweight_drift=0.
    elapsed=@elapsed for step in 1:iterations
        result=nothing
        update_seconds+=@elapsed result=FusedOriginalSwapKernel.sweep_fused!(state,model.G,rng)
        accepted+=result.accepted
        if step%5==0
            drift=nothing
            rebuild_seconds+=@elapsed drift=rebuild!(state,model.G)
            maximum_inverse_drift=max(maximum_inverse_drift,drift.inverse_drift)
            maximum_logweight_drift=max(maximum_logweight_drift,drift.logweight_drift)
        end
        entropy_seconds+=@elapsed entropy!(work,model,state.E,3.)
    end
    BLAS.get_num_threads()==1 || error("BLAS thread count changed")
    result=Dict("status"=>"complete","purpose"=>"Timing only, excluded from all entropy inference",
        "L"=>36,"beta"=>3.,"mu"=>1e-8,"seed"=>seed,"worker"=>worker,
        "blas_threads"=>BLAS.get_num_threads(),"warmup_sweeps"=>5,"timed_sweeps"=>iterations,
        "elapsed_seconds"=>elapsed,"update_seconds"=>update_seconds,"entropy_seconds"=>entropy_seconds,
        "rebuild_seconds"=>rebuild_seconds,"accepted"=>accepted,"proposed"=>iterations*model.N,
        "maximum_inverse_drift"=>maximum_inverse_drift,"maximum_logweight_drift"=>maximum_logweight_drift,
        "final_E_sha256"=>bytes2hex(sha256(join(state.E,","))),"final_log_weight"=>state.log_weight,
        "pilot_receipt"=>rel(paths_for(job).receipt),"pilot_receipt_sha256"=>sha(paths_for(job).receipt),
        "source_sha256"=>Dict(rel(@__FILE__)=>sha(@__FILE__),
            "fused_original_swap_kernel.jl"=>sha(joinpath(ROOT,"fused_original_swap_kernel.jl")),
            "original_swap_scaling.jl"=>sha(joinpath(ROOT,"original_swap_scaling.jl"))))
    open(joinpath(benchmark_dir,"worker_$(worker).toml"),"w") do io
        TOML.print(io,result)
    end
    println("complete worker=$worker seconds=$elapsed")
end

throughput_main()
