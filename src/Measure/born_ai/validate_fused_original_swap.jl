# Validate the standalone production wrapper without starting production.
include("run_born_scaling_fused.jl")
const VALIDATION_DIR=joinpath(ROOT,"data","born_scaling","fused_wrapper_validation")
const EXPECTED_ORIGINAL_SHA="1af51fff868fbe63025323618506295ce5622179c1381de5ea64549e8c0d40cb"
const EXPECTED_RUNNER_SHA="3d9004c1d112a29ee1a7b440a945dc89516f7cabba8e279e0e5dd145e5a65c76"

function rejected_plan(path)
    try
        validate_fused_plan(path)
        return false
    catch
        return true
    end
end

function validate_wrapper()
    started=time(); mkpath(VALIDATION_DIR)
    @assert sha(joinpath(ROOT,"original_swap_scaling.jl"))==EXPECTED_ORIGINAL_SHA
    @assert sha(joinpath(ROOT,"run_born_scaling.jl"))==EXPECTED_RUNNER_SHA
    sources=source_hashes()
    for p in (@__FILE__,joinpath(ROOT,"run_born_scaling_fused.jl"),joinpath(ROOT,"fused_original_swap_kernel.jl"))
        sources[rel(p)]=sha(p)
    end
    wrapped=which(OriginalSwapScaling.sweep!,Tuple{SwapState,Matrix{Float64},MersenneTwister})
    generic=which(OriginalSwapScaling.sweep!,Tuple{Any,Any,Any})
    @assert normpath(String(wrapped.file))==normpath(joinpath(ROOT,"run_born_scaling_fused.jl"))
    @assert normpath(String(generic.file))==normpath(joinpath(ROOT,"original_swap_scaling.jl"))
    tests=Dict{String,Any}[]
    for L in (3,6,18,24)
        model=ground_model(L;state=:staggered,mu=1e-8)
        betas=L<=6 ? (0.,.5,1.5,3.) : (1.5,3.)
        starts=L<=6 ? ("random","uniform_A","uniform_B","02_stripe") : ("random","uniform_B")
        steps=L<=6 ? 20 : 10
        for beta in betas,(startindex,start) in enumerate(starts)
            seed=874_000+100L+round(Int,10beta)+startindex
            ra=MersenneTwister(seed); rb=MersenneTwister(seed)
            ea=initial_set(L,start,ra); eb=initial_set(L,start,rb)
            a=swap_state(model.G,ea,beta); b=swap_state(model.G,eb,beta)
            matrixdiff=0.; logdiff=0.; drift_a=0.; drift_b=0.
            for step in 1:steps
                sa=invoke(OriginalSwapScaling.sweep!,Tuple{Any,Any,Any},a,model.G,ra)
                sb=OriginalSwapScaling.sweep!(b,model.G,rb)
                @assert sa==sb && a.E==b.E && a.S==b.S
                matrixdiff=max(matrixdiff,norm(a.X-b.X)/norm(a.X))
                logdiff=max(logdiff,abs(a.log_weight-b.log_weight))
                if step%5==0
                    da=rebuild!(a,model.G); db=rebuild!(b,model.G)
                    drift_a=max(drift_a,da.inverse_drift); drift_b=max(drift_b,db.inverse_drift)
                end
            end
            @assert max(matrixdiff,logdiff,drift_a,drift_b)<1e-5
            push!(tests,Dict("L"=>L,"beta"=>beta,"start"=>start,"seed"=>seed,"sweeps"=>steps,
                "same_seed_trajectory_equal"=>true,"max_inverse_difference"=>matrixdiff,"max_logweight_difference"=>logdiff,
                "max_original_drift"=>drift_a,"max_fused_drift"=>drift_b))
        end
        println("WRAPPER VALIDATED L=$L"); flush(stdout)
    end
    # Actual base-runner pipeline, with the typed wrapper method active.
    basepath=joinpath(ROOT,"data","born_scaling","smallcheck","smallcheck_plan.toml")
    plan=TOML.parsefile(basepath)
    @assert rejected_plan(basepath)
    plan["stage"]="kernel_validation"; plan["source_sha256"]=sources
    plan["update_kernel"]="fused_rank_updates"
    job=deepcopy(first(filter(j->j["beta"]==1.5 && j["start"]=="random",plan["jobs"])))
    job["run_id"]="fused_wrapper_validation_L3_beta1p5"; job["stage"]="kernel_validation"
    job["sweeps"]=20; job["seed"]=875_003; job["output_dir"]=rel(VALIDATION_DIR)
    plan["jobs"]=[job]
    planpath=joinpath(VALIDATION_DIR,"kernel_validation_plan.toml")
    write_toml(planpath,plan); validate_fused_plan(planpath)
    # Reject plans that omit the implementation binding or mislabel the kernel.
    for kind in ("missing_source","wrong_kernel","pilot_job")
        bad=deepcopy(plan)
        if kind=="missing_source"; delete!(bad["source_sha256"],"fused_original_swap_kernel.jl")
        elseif kind=="wrong_kernel"; bad["update_kernel"]="original"
        else; bad["jobs"][1]["stage"]="pilot"
        end
        path=joinpath(VALIDATION_DIR,"rejected_$(kind).toml")
        write_toml(path,bad); @assert rejected_plan(path)
    end
    run_job(job,plan,planpath)
    model=load_model(job); rng=MersenneTwister(job["seed"])
    state=swap_state(model.G,initial_set(3,"random",rng),1.5); work=EntropyWorkspace(model)
    lines=readlines(paths_for(job).samples); max_entropy=0.; max_log=0.
    for step in 1:20
        stats=invoke(OriginalSwapScaling.sweep!,Tuple{Any,Any,Any},state,model.G,rng)
        step%5==0 && rebuild!(state,model.G)
        ent=entropy!(work,model,state.E,1.5); cells=split(lines[step+1],',')
        @assert parse(Int,cells[14])==stats.accepted
        observed=parse.(Float64,cells[10:13]); expected=[ent.SA,ent.SB,ent.SC,ent.D]
        max_entropy=max(max_entropy,maximum(abs.(observed-expected)))
        max_log=max(max_log,abs(parse(Float64,cells[16])-state.log_weight))
    end
    @assert max_entropy<1e-12 && max_log<1e-8
    finalize(planpath)
    @assert all(sha(joinpath(ROOT,p))==h for (p,h) in sources)
    @assert sha(joinpath(ROOT,"original_swap_scaling.jl"))==EXPECTED_ORIGINAL_SHA
    @assert sha(joinpath(ROOT,"run_born_scaling.jl"))==EXPECTED_RUNNER_SHA
    receipt=Dict("status"=>"PASS","created_at"=>string(now()),"elapsed_seconds"=>time()-started,
        "source_sha256"=>sources,"original_sources_unchanged"=>true,"wrapper_method_file"=>String(wrapped.file),
        "original_generic_method_file"=>String(generic.file),"case_count"=>length(tests),"tests"=>tests,
        "guard_checks"=>["old_pilot_rejected","missing_kernel_source_rejected","wrong_kernel_rejected","pilot_job_rejected"],
        "pipeline_entropy_difference"=>max_entropy,"pipeline_logweight_difference"=>max_log,
        "pipeline_manifest"=>rel(joinpath(VALIDATION_DIR,"kernel_validation_manifest.toml")),
        "pipeline_manifest_sha256"=>sha(joinpath(VALIDATION_DIR,"kernel_validation_manifest.toml")))
    write_toml(joinpath(VALIDATION_DIR,"validation.toml"),receipt)
    println("FUSED WRAPPER PASS cases=$(length(tests)) elapsed=$(time()-started)")
end
if abspath(PROGRAM_FILE)==@__FILE__; validate_wrapper(); end
