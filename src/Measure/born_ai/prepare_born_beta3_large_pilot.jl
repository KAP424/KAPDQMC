# Independently authorized, short allocation pilot. Old sources/plans are read-only.
include("run_born_scaling.jl")

function prepare_beta3_large_pilot()
    stage="pilot_beta3_large"
    directory=joinpath(ROOT,"data","born_scaling",stage)
    plan_path=joinpath(directory,"$(stage)_plan.toml")
    isfile(plan_path) && error("pilot plan is already frozen; resume rather than replace")
    launcher=joinpath(ROOT,"launch_born_beta3_large.py")
    isfile(launcher) || error("write the immutable launcher before freezing the plan")
    models=Dict{String,Any}[]; jobs=Dict{String,Any}[]
    mkpath(directory)
    for L in (27,30,36)
        println("BUILD MODEL L=$L"); flush(stdout)
        model=ground_model(L;state=:staggered,mu=1e-8)
        paths=save_model(model,joinpath(directory,"models");prefix="state")
        receipt=only(filter(p->endswith(p,".toml"),paths))
        push!(models,Dict("L"=>L,"mu"=>1e-8,"model_manifest"=>rel(receipt),
            "model_manifest_sha256"=>sha(receipt),"files"=>Dict(rel(p)=>sha(p) for p in paths)))
        for (chain,start) in enumerate(("random","uniform_A","uniform_B","02_stripe"))
            seed=1_900_000_000+1_000_000L+300_000+chain
            id="$(stage)_L$(L)_beta$(tag(3.))_mu$(tag(1e-8))_chain$(chain)_$(start)"
            push!(jobs,Dict("run_id"=>id,"stage"=>stage,"L"=>L,"beta"=>3.,"mu"=>1e-8,
                "state"=>"staggered","chain"=>chain,"seed"=>seed,"start"=>start,"sweeps"=>512,
                "measure_every"=>1,"refresh_every"=>5,"output_dir"=>rel(directory),
                "model_manifest"=>rel(receipt),"model_manifest_sha256"=>sha(receipt)))
        end
        println("MODEL READY L=$L"); flush(stdout)
    end
    prior_seeds=Set{Int}()
    for previous in ("pilot","pilot_extra","smallcheck","production")
        path=joinpath(ROOT,"data","born_scaling",previous,"$(previous)_plan.toml")
        p=validate_plan(path)
        union!(prior_seeds,job["seed"] for job in p["jobs"])
    end
    any(job["seed"] in prior_seeds for job in jobs) && error("pilot seed reuses a previous stage")
    sources=source_hashes()
    for path in (@__FILE__,launcher,joinpath(ROOT,"run_born_scaling_fused.jl"),joinpath(ROOT,"fused_original_swap_kernel.jl"))
        sources[rel(path)]=sha(path)
    end
    evidence_path=joinpath(ROOT,"data","born_scaling","fused_wrapper_validation","validation.toml")
    validation=TOML.parsefile(evidence_path)
    validation["status"]=="PASS" || error("fused arithmetic equivalence has not passed")
    plan=Dict("version"=>1,"stage"=>stage,"created_at"=>string(now()),"state"=>"staggered","mu"=>1e-8,
        "purpose"=>"Independent 512-sweep pilot for beta=3 at L27/30/36; timing, thermalization and variance allocation only. Never pooled with production.",
        "source_sha256"=>sources,"blas_threads"=>1,"models"=>models,"jobs"=>jobs,
        "columns"=>collect(COLUMNS),"phase"=>PHASE,"burnin_discarded"=>0,"update_kernel"=>"fused_rank_updates",
        "seed_formula"=>"1900000000 + 1000000*L + 300000 + chain; distinct from old stages and forthcoming production",
        "pilot_burnin_candidates"=>[32,64,128,256],"planned_production_burnin_if_supported"=>256,
        "planned_production_minimum_retained_per_chain"=>2048,
        "precision_target_D_absolute_MCSE"=>.003,"precision_target_SA_absolute_MCSE"=>.01,
        "production_rule"=>"No production or L42 is scheduled by this pilot plan. Report measured timing/variance to root and freeze a separate fixed-count plan and fresh seeds before production.",
        "kernel_validation_path"=>rel(evidence_path),"kernel_validation_sha256"=>sha(evidence_path),
        "sweep_definition"=>"2L^2 uniform E/S pairs with original Metropolis; rejected states remain in the trace",
        "initial_02_stripe"=>"E is the suppressed set; E=A/B for uniform_A/B. Even L: both sublattices on odd x columns. Odd L27: alternate full/empty through L-1, then only A on the final column; not pure 0/2.",
        "acceptance_columns"=>"Actual accepted/proposed attempts since previous sample; measure_every=1",
        "max_inverse_drift_tolerance"=>1e-5,"max_logweight_drift_tolerance"=>1e-5)
    write_toml(plan_path,plan)
    println("PREPARED $plan_path jobs=$(length(jobs)) rows=$(sum(j["sweeps"] for j in jobs))")
end

if abspath(PROGRAM_FILE)==@__FILE__; prepare_beta3_large_pilot(); end
