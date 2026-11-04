# Freeze a separately approved extension allocation. Does not run Monte Carlo.
# Usage: julia --project=. thisfile --allocation=CSV --workers=BENCHMARK_CHOICE
include("run_born_scaling.jl")

function prepare_beta3_large_production()
    allocation=abspath(arg("allocation","")); isfile(allocation) || error("provide approved --allocation=CSV")
    workers=parse(Int,arg("workers","0")); workers in (1,2,4) || error("provide benchmark-selected --workers=1,2,4")
    stage=arg("stage","production_beta3_large"); startswith(stage,"production_beta3_large") || error("unexpected production stage")
    directory=joinpath(ROOT,"data","born_scaling",stage)
    path=joinpath(directory,"$(stage)_plan.toml")
    isfile(path) && error("production plan is already frozen")
    lines=readlines(allocation); first(lines)=="L,beta,retained_sweeps,burnin,grid" || error("allocation schema mismatch")
    allocations=map(lines[2:end]) do line
        x=split(line,','); (L=parse(Int,x[1]),beta=parse(Float64,x[2]),keep=parse(Int,x[3]),burn=parse(Int,x[4]),grid=x[5])
    end
    sort([x.L for x in allocations])==[27,30,36] || error("exactly one beta3 case at each of L27/30/36 is authorized")
    all(x->x.beta==3. && x.keep>=2048 && x.keep%256==0 && x.burn>=256 && x.grid=="main",allocations) || error("invalid allocation")
    length(unique(x.burn for x in allocations))==1 || error("freeze one common thermalization length")
    pilot_manifest=joinpath(ROOT,"data","born_scaling","pilot_beta3_large","pilot_beta3_large_manifest.toml")
    p=TOML.parsefile(pilot_manifest); p["status"]=="complete" || error("pilot incomplete")
    pilot_plan=joinpath(ROOT,p["plan_path"]); sha(pilot_plan)==p["plan_sha256"] || error("pilot plan changed")
    pp=validate_plan(pilot_plan)
    for (file,hash) in p["files"]; sha(joinpath(ROOT,file))==hash || error("pilot evidence changed"); end
    pilot_analysis=joinpath(ROOT,"analysis","born_scaling","pilot_beta3_large","summary.json")
    benchmark=joinpath(ROOT,"data","born_scaling","beta3_large_throughput","summary.json")
    protocol=joinpath(ROOT,"analysis","born_scaling","beta3_large_protocol.json")
    holdout=joinpath(ROOT,"analysis","born_scaling","beta3_large_holdout","summary.json")
    launcher=joinpath(ROOT,"launch_born_beta3_large_production.py")
    fresh_entry=joinpath(ROOT,"run_born_beta3_fresh_chain.jl")
    allocation_design=joinpath(dirname(allocation),"production_allocation_design.json")
    for file in (pilot_analysis,benchmark,protocol,holdout,launcher,fresh_entry,allocation_design); isfile(file) || error("required evidence unavailable: $file"); end
    jobs=Dict{String,Any}[]
    models=deepcopy(pp["models"])
    for x in allocations
        model=only(filter(m->m["L"]==x.L,models))
        for (chain,start) in enumerate(("random","uniform_A","uniform_B","02_stripe"))
            seed=2_000_000_000+1_000_000x.L+300_000+chain
            id="$(stage)_L$(x.L)_beta$(tag(x.beta))_mu$(tag(1e-8))_chain$(chain)_$(start)"
            push!(jobs,Dict("run_id"=>id,"stage"=>stage,"L"=>x.L,"beta"=>3.,"mu"=>1e-8,"state"=>"staggered",
                "chain"=>chain,"seed"=>seed,"start"=>start,"sweeps"=>x.keep+x.burn,"retained_sweeps"=>x.keep,
                "analysis_burnin"=>x.burn,"grid"=>"main","measure_every"=>1,"refresh_every"=>5,
                "output_dir"=>rel(directory),"model_manifest"=>model["model_manifest"],
                "model_manifest_sha256"=>model["model_manifest_sha256"]))
        end
    end
    oldseeds=Set{Int}()
    for folder in readdir(joinpath(ROOT,"data","born_scaling");join=true)
        isdir(folder) || continue
        oldpath=joinpath(folder,basename(folder)*"_plan.toml")
        isfile(oldpath) || continue
        old=TOML.parsefile(oldpath)
        union!(oldseeds,j["seed"] for j in old["jobs"])
    end
    any(j["seed"] in oldseeds for j in jobs) && error("production seed overlaps another frozen plan")
    sources=source_hashes()
    for file in (@__FILE__,launcher,fresh_entry,joinpath(ROOT,"run_born_scaling_fused.jl"),joinpath(ROOT,"fused_original_swap_kernel.jl"),protocol)
        sources[rel(file)]=sha(file)
    end
    kernel=joinpath(ROOT,"data","born_scaling","fused_wrapper_validation","validation.toml")
    evidence=Dict(rel(f)=>sha(f) for f in (pilot_manifest,pilot_analysis,kernel,benchmark,protocol,holdout,allocation_design))
    plan=Dict("version"=>1,"stage"=>stage,"created_at"=>string(now()),"state"=>"staggered","mu"=>1e-8,
        "source_sha256"=>sources,"blas_threads"=>1,"execution_workers"=>workers,"execution_mode"=>"fresh_process_per_chain","models"=>models,"jobs"=>jobs,
        "columns"=>collect(COLUMNS),"phase"=>PHASE,"burnin_discarded"=>0,"analysis_burnin"=>only(unique(x.burn for x in allocations)),
        "update_kernel"=>"fused_rank_updates","allocation_path"=>rel(allocation),"allocation_sha256"=>sha(allocation),
        "pilot_evidence"=>evidence,"analysis_protocol_path"=>rel(protocol),"analysis_protocol_sha256"=>sha(protocol),
        "throughput_benchmark_path"=>rel(benchmark),"throughput_benchmark_sha256"=>sha(benchmark),
        "seed_formula"=>"2000000000 + 1000000*L + 300000 + chain; fresh seeds distinct from all frozen previous plans",
        "state_reuse"=>"Exact mu1e-8 projector binaries from this pilot are reused; no pilot observation/initial state enters production means.",
        "precision_target_D_absolute_MCSE"=>.003,"precision_target_SA_absolute_MCSE"=>.01,
        "allocation_note"=>"Fixed lengths approved after independent pilot timing/variance and before any production samples; absolute targets are estimates, not guarantees.",
        "allocation_rule"=>"max(2048, ceil_to_multiple_of_256(1.20 * max needed sample counts across pilot burnin128/256 and observables D/SA))); 20% variance budget, frozen before production",
        "allocation_design_path"=>rel(allocation_design),"allocation_design_sha256"=>sha(allocation_design),
        "sweep_definition"=>pp["sweep_definition"],"initial_02_stripe"=>pp["initial_02_stripe"],
        "acceptance_columns"=>pp["acceptance_columns"],"max_inverse_drift_tolerance"=>1e-5,"max_logweight_drift_tolerance"=>1e-5)
    mkpath(directory);write_toml(path,plan)
    println("PREPARED $path chains=$(length(jobs)) retained=$(sum(j["retained_sweeps"] for j in jobs)) workers=$workers")
end

if abspath(PROGRAM_FILE)==@__FILE__; prepare_beta3_large_production(); end
