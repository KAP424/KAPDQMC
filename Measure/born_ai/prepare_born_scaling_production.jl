# Freeze explicit production jobs after the independent pilot. No MC here.
# julia --project=. thisfile --allocation=... --stage=production
include("run_born_scaling.jl")

function prepare_production()
    allocation=abspath(arg("allocation","")); isfile(allocation) || error("provide --allocation=CSV")
    stage=arg("stage","production"); startswith(stage,"production") || error("production stage required")
    directory=joinpath(ROOT,"data","born_scaling",stage); mkpath(directory)
    planpath=joinpath(directory,"$(stage)_plan.toml")
    isfile(planpath) && error("frozen plan already exists")
    lines=readlines(allocation); lines[1]=="L,beta,retained_sweeps,burnin,grid" || error("invalid allocation header")
    allocations=map(lines[2:end]) do line
        x=split(line,','); (L=parse(Int,x[1]),beta=parse(Float64,x[2]),keep=parse(Int,x[3]),burnin=parse(Int,x[4]),grid=x[5])
    end
    length(unique((x.L,x.beta) for x in allocations))==length(allocations) || error("duplicate L/beta")
    all(x->x.beta>0 && x.keep>=2048 && x.burnin==256,allocations) || error("unexpected production design")
    models=Dict{String,Any}[]; jobs=Dict{String,Any}[]
    for L in sort(unique(x.L for x in allocations))
        model=ground_model(L;state=:staggered,mu=1e-8)
        paths=save_model(model,joinpath(directory,"models");prefix="state")
        receipt=only(filter(p->endswith(p,".toml"),paths))
        push!(models,Dict("L"=>L,"mu"=>1e-8,"model_manifest"=>rel(receipt),
             "model_manifest_sha256"=>sha(receipt),"files"=>Dict(rel(p)=>sha(p) for p in paths)))
        for x in filter(x->x.L==L,allocations), (chain,start) in enumerate(("random","uniform_A","uniform_B","02_stripe"))
            seed=1_600_000_000+1_000_000L+100round(Int,1000x.beta)+chain
            id="$(stage)_L$(L)_beta$(tag(x.beta))_mu$(tag(1e-8))_chain$(chain)_$(start)"
            push!(jobs,Dict("run_id"=>id,"stage"=>stage,"L"=>L,"beta"=>x.beta,"mu"=>1e-8,
                "state"=>"staggered","chain"=>chain,"seed"=>seed,"start"=>start,
                "sweeps"=>x.keep+x.burnin,"retained_sweeps"=>x.keep,"analysis_burnin"=>x.burnin,
                "grid"=>x.grid,"measure_every"=>1,"refresh_every"=>5,"output_dir"=>rel(directory),
                "model_manifest"=>rel(receipt),"model_manifest_sha256"=>sha(receipt)))
        end
    end
    evidence=Dict{String,String}()
    for name in ("pilot","pilot_extra")
        path=joinpath(ROOT,"data","born_scaling",name,"$(name)_manifest.toml")
        isfile(path) && (evidence[rel(path)]=sha(path))
    end
    kernel_validation=joinpath(ROOT,"data","born_scaling","fused_wrapper_validation","validation.toml")
    isfile(kernel_validation) || error("fused wrapper validation receipt is required")
    evidence[rel(kernel_validation)]=sha(kernel_validation)
    sources=source_hashes(); sources[rel(@__FILE__)]=sha(@__FILE__)
    for file in ("fused_original_swap_kernel.jl","run_born_scaling_fused.jl")
        path=joinpath(ROOT,file); isfile(path) || error("validated fused source required: $file")
        sources[rel(path)]=sha(path)
    end
    plan=Dict("version"=>1,"stage"=>stage,"created_at"=>string(now()),"state"=>"staggered","mu"=>1e-8,
        "source_sha256"=>sources,"blas_threads"=>1,"models"=>models,"jobs"=>jobs,"columns"=>collect(COLUMNS),
        "phase"=>PHASE,"burnin_discarded"=>0,"analysis_burnin"=>256,"update_kernel"=>"fused_rank_updates",
        "seed_formula"=>"1600000000 + 1000000*L + 100*round(1000*beta) + chain; unique across all production cases",
        "allocation_path"=>rel(allocation),"allocation_sha256"=>sha(allocation),"pilot_evidence"=>evidence,
        "precision_target_D_absolute_MCSE"=>.003,"precision_target_SA_absolute_MCSE"=>.01,
        "allocation_note"=>"Independent pilot variance determines fixed lengths before production. Pilot draws are excluded. Targets are planned absolute MCSE, not guaranteed relative precision or model errors.",
        "sweep_definition"=>"2L^2 uniform E/S index pairs with the original Metropolis rule, including rejected states",
        "initial_02_stripe"=>"E is the suppressed set. Even L: both sublattices on odd x columns; odd L: alternate full/empty through L-1 and only A on final column. uniform_A/B mean E=A/B.",
        "acceptance_columns"=>"accepted/proposed count all attempts since previous sample",
        "max_inverse_drift_tolerance"=>1e-5,"max_logweight_drift_tolerance"=>1e-5)
    write_toml(planpath,plan)
    total_retained=sum(j["retained_sweeps"] for j in jobs)
    println("PREPARED $planpath cases=$(length(allocations)) chains=$(length(jobs)) retained=$total_retained")
end

prepare_production()
