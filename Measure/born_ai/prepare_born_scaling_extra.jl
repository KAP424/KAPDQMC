# Authorized additional pilot for variance allocation; never changes old plans.
include("run_born_scaling.jl")

function prepare_extra()
    parent_path=joinpath(ROOT,"data","born_scaling","pilot","pilot_plan.toml")
    parent=validate_plan(parent_path)
    directory=joinpath(ROOT,"data","born_scaling","pilot_extra"); mkpath(directory)
    path=joinpath(directory,"pilot_extra_plan.toml")
    isfile(path) && error("extra pilot plan is already frozen; resume it instead")
    model=only(filter(m->m["L"]==18 && m["mu"]==1e-8,parent["models"]))
    plan=deepcopy(parent)
    plan["stage"]="pilot_extra"; plan["created_at"]=string(now())
    plan["parent_plan"]=rel(parent_path); plan["parent_plan_sha256"]=sha(parent_path)
    plan["source_sha256"][rel(@__FILE__)]=sha(@__FILE__)
    plan["models"]=[deepcopy(model)]
    plan["purpose"]="Independent additional pilot for beta-specific variance allocation, not production inference"
    plan["seed_formula"]="930000000 + 100000000*3 + 1000000*L + 100*round(1000*beta) + chain"
    jobs=Dict{String,Any}[]
    for beta in (1.,2.,2.5), (chain,start) in enumerate(("random","uniform_A","uniform_B","02_stripe"))
        seed=930_000_000+300_000_000+18_000_000+100round(Int,1000beta)+chain
        id="pilot_extra_L18_beta$(tag(beta))_mu$(tag(1e-8))_chain$(chain)_$(start)"
        push!(jobs,Dict("run_id"=>id,"stage"=>"pilot_extra","L"=>18,"beta"=>beta,"mu"=>1e-8,
            "state"=>"staggered","chain"=>chain,"seed"=>seed,"start"=>start,"sweeps"=>2048,
            "measure_every"=>1,"refresh_every"=>5,"output_dir"=>rel(directory),
            "model_manifest"=>model["model_manifest"],"model_manifest_sha256"=>model["model_manifest_sha256"]))
    end
    plan["jobs"]=jobs
    write_toml(path,plan)
    println("PREPARED $path jobs=$(length(jobs))")
end

if abspath(PROGRAM_FILE)==@__FILE__; prepare_extra(); end
