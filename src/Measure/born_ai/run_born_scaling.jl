# Fixed-plan original-swap pilot/production runner. No importance or Gibbs moves.
# Prepare authorized presets with --prepare-only --stage=pilot (or smallcheck).
# Execute --plan=path --worker=1 --workers=4; completed chains are hash-validated.
# --plans=path1,path2 executes stages sequentially in each worker.
# --plan=path --finalize validates all chains and publishes the complete manifest.
using LinearAlgebra, Random, SHA, TOML, Dates
BLAS.set_num_threads(1)
include("original_swap_scaling.jl")
using .OriginalSwapScaling
const ROOT=@__DIR__
const PHASE="all_recorded_no_burnin_discard"
const COLUMNS=("L","beta","mu","chain","seed","start","sweep","record_index","measure_every",
               "SA","SB","SC","D","accepted","proposed","log_weight","stage","phase","run_id",
               "refreshed","inverse_drift","logweight_drift")
const HEADER=join(COLUMNS,',')
arg(name,default)=let a=filter(x->startswith(x,"--$name="),ARGS); isempty(a) ? default : split(last(a),'=';limit=2)[2] end
flag(name)="--$name" in ARGS
sha(path)=bytes2hex(sha256(read(path)))
rel(path)=replace(relpath(path,ROOT),'\\'=>'/')
csv(io,row)=println(io,join(string.(row),','))
tag(x)=replace(string(x),'.'=>'p','-'=>'m')
write_toml(path,value)=open(io->TOML.print(io,value),path,"w")

function source_hashes()
    Dict(rel(path)=>sha(path) for path in [@__FILE__,joinpath(ROOT,"original_swap_scaling.jl"),
         normpath(joinpath(ROOT,"../../public/Geometry.jl"))])
end

function prepare(stage)
    stage in ("pilot","smallcheck") || error("production requires a separately authorized explicit plan; no default production scan")
    directory=joinpath(ROOT,"data","born_scaling",stage); mkpath(directory)
    plan_path=joinpath(directory,"$(stage)_plan.toml")
    isfile(plan_path) && error("plan already exists; resume its frozen jobs instead of replacing it")
    sizes=stage=="pilot" ? (6,12,18) : (3,)
    sweeps=stage=="pilot" ? 2048 : 16384
    stagecode=stage=="pilot" ? 1 : 2
    models=Dict{String,Any}[]; jobs=Dict{String,Any}[]
    for L in sizes
        model=ground_model(L;state=:staggered,mu=1e-8)
        paths=save_model(model,joinpath(directory,"models");prefix="state")
        model_manifest=only(filter(p->endswith(p,".toml"),paths))
        push!(models,Dict("L"=>L,"mu"=>1e-8,"model_manifest"=>rel(model_manifest),
            "model_manifest_sha256"=>sha(model_manifest),"files"=>Dict(rel(p)=>sha(p) for p in paths)))
        for beta in (.5,1.5,3.), (chain,start) in enumerate(("random","uniform_A","uniform_B","02_stripe"))
            seed=930_000_000+100_000_000stagecode+1_000_000L+100round(Int,1000beta)+chain
            jobid="$(stage)_L$(L)_beta$(tag(beta))_mu$(tag(1e-8))_chain$(chain)_$(start)"
            push!(jobs,Dict("run_id"=>jobid,"stage"=>stage,"L"=>L,"beta"=>beta,"mu"=>1e-8,
                 "state"=>"staggered","chain"=>chain,"seed"=>seed,"start"=>start,"sweeps"=>sweeps,
                 "measure_every"=>1,"refresh_every"=>5,"output_dir"=>rel(directory),
                 "model_manifest"=>rel(model_manifest),"model_manifest_sha256"=>sha(model_manifest)))
        end
    end
    plan=Dict("version"=>1,"stage"=>stage,"created_at"=>string(now()),"state"=>"staggered","mu"=>1e-8,
        "source_sha256"=>source_hashes(),"blas_threads"=>1,"models"=>models,"jobs"=>jobs,
        "columns"=>collect(COLUMNS),"phase"=>PHASE,"burnin_discarded"=>0,
        "seed_formula"=>"930000000 + 100000000*stage_code + 1000000*L + 100*round(1000*beta) + chain; pilot=1,smallcheck=2",
        "sweep_definition"=>"2L^2 uniform E/S index pairs, with the original Metropolis acceptance rule; rejected states remain in the trace",
        "initial_02_stripe"=>"Even L: E contains both A/B on odd x columns. Odd L: alternate full/empty for x<=L-1, and place only A in the final column; this is half-filled but not pure 0/2.",
        "acceptance_columns"=>"accepted/proposed count all attempts since the previous recorded sample; measure_every=1 in these presets",
        "max_inverse_drift_tolerance"=>1e-5,"max_logweight_drift_tolerance"=>1e-5)
    write_toml(plan_path,plan)
    println("PREPARED $plan_path jobs=$(length(jobs))")
end

function validate_plan(path)
    plan=TOML.parsefile(path)
    plan["version"]==1 || error("unsupported plan version")
    plan["state"]=="staggered" || error("these runs require an explicit staggered state")
    plan["blas_threads"]==1 || error("plan requires one BLAS thread")
    for (file,hash) in plan["source_sha256"]
        sha(joinpath(ROOT,file))==hash || error("source changed after plan freeze: $file")
    end
    length(unique(job["run_id"] for job in plan["jobs"]))==length(plan["jobs"]) || error("duplicate job IDs")
    plan
end

function load_model(job)
    path=joinpath(ROOT,job["model_manifest"])
    sha(path)==job["model_manifest_sha256"] || error("model receipt hash mismatch")
    metadata=TOML.parsefile(path)
    metadata["L"]==job["L"] && metadata["mu"]==job["mu"] && metadata["zero_mode_rule"]=="staggered" || error("model parameter mismatch")
    for (file,hash) in metadata["files"]
        sha(joinpath(dirname(path),file))==hash || error("model binary hash mismatch")
    end
    N=metadata["Nsites"]; n=metadata["Nparticles"]
    matrices=Dict{String,Any}()
    for (label,shape) in (("Q",(N,n)),("G",(N,N)),("energies",(N,)))
        file=only(filter(p->endswith(p,"_$(label).bin"),collect(keys(metadata["files"]))))
        matrix=Array{Float64}(undef,shape...)
        open(io->read!(io,matrix),joinpath(dirname(path),file),"r")
        matrices[label]=matrix
    end
    GroundModel(job["L"],N,n,matrices["Q"],matrices["G"],matrices["energies"],:staggered,
        job["mu"],metadata["null_singular_count"],metadata["zero_singular_tolerance"],
        metadata["orthogonality_error"],metadata["ground_energy_error"],metadata["eigenvector_residual"])
end

function initial_set(L,start,rng)
    N=2L^2
    start=="random" && return randperm(rng,N)[1:L^2]
    start=="uniform_A" && return collect(1:2:N)
    start=="uniform_B" && return collect(2:2:N)
    start=="02_stripe" || error("unknown initial state")
    E=Int[]
    for y in 1:L,x in 1:L
        A=2(x+(y-1)*L)-1
        if isodd(L) && x==L
            push!(E,A)
        elseif isodd(x)
            append!(E,(A,A+1))
        end
    end
    length(E)==L^2 || error("stripe does not have half filling")
    E
end

function paths_for(job)
    dir=joinpath(ROOT,job["output_dir"]); stem=job["run_id"]
    (;samples=joinpath(dir,stem*"_samples.csv"),initial=joinpath(dir,stem*"_initial.csv"),
      receipt=joinpath(dir,stem*"_receipt.toml"))
end

function validate_completed(job,plan_path)
    paths=paths_for(job)
    isfile(paths.receipt) || return false
    receipt=TOML.parsefile(paths.receipt)
    receipt["status"]=="complete" || return false
    receipt["job"]==job || error("completed job parameters do not match frozen plan")
    receipt["plan_sha256"]==sha(plan_path) || error("completed chain belongs to another plan")
    for (file,hash) in receipt["files"]
        sha(joinpath(ROOT,file))==hash || error("completed chain file hash mismatch")
    end
    open(paths.samples,"r") do io
        readline(io)==HEADER || error("CSV schema mismatch")
        count=0
        for line in eachline(io)
            cells=split(line,','); count+=1
            length(cells)==length(COLUMNS) || error("CSV column mismatch")
            parse(Int,cells[1])==job["L"] && parse(Float64,cells[2])==job["beta"] &&
            parse(Float64,cells[3])==job["mu"] && parse(Int,cells[4])==job["chain"] &&
            parse(Int,cells[5])==job["seed"] && cells[6]==job["start"] || error("CSV parameter mismatch")
            parse(Int,cells[7])==count*job["measure_every"] && parse(Int,cells[8])==count || error("CSV sequence mismatch")
            cells[17]==job["stage"] && cells[18]==PHASE && cells[19]==job["run_id"] || error("CSV stage mismatch")
        end
        count==job["sweeps"]÷job["measure_every"] || error("incomplete record count")
    end
    true
end

function row(job,sweep,index,entropy,accepted,proposed,logweight;refreshed=false,inverse_drift=0.,logweight_drift=0.)
    (job["L"],job["beta"],job["mu"],job["chain"],job["seed"],job["start"],sweep,index,job["measure_every"],
     entropy.SA,entropy.SB,entropy.SC,entropy.D,accepted,proposed,logweight,job["stage"],PHASE,job["run_id"],
     Int(refreshed),inverse_drift,logweight_drift)
end

function run_job(job,plan,plan_path)
    validate_completed(job,plan_path) && (println("SKIP verified $(job["run_id"])"); return)
    job["sweeps"]>0 && job["measure_every"]>0 && job["refresh_every"]>0 || error("invalid run lengths")
    job["sweeps"]%job["measure_every"]==0 || error("measurement interval must divide sweeps")
    paths=paths_for(job); mkpath(dirname(paths.samples))
    model=load_model(job); rng=MersenneTwister(job["seed"])
    E=initial_set(job["L"],job["start"],rng); state=swap_state(model.G,E,job["beta"])
    work=EntropyWorkspace(model); start_time=time()
    initial=entropy!(work,model,state.E,job["beta"])
    open(paths.initial,"w") do io; println(io,HEADER); csv(io,row(job,0,0,initial,0,0,state.log_weight)); end
    receipt=Dict{String,Any}("status"=>"running","job"=>job,"plan_sha256"=>sha(plan_path),
        "source_sha256"=>plan["source_sha256"],"started_at"=>string(now()),"initial_E"=>copy(E),
        "initial_state_definition"=>plan["initial_02_stripe"],"model_manifest"=>job["model_manifest"],
        "model_manifest_sha256"=>job["model_manifest_sha256"],"blas_threads"=>1,"julia_version"=>string(VERSION))
    write_toml(paths.receipt,receipt)
    accepted=0; proposed=0; window_accepted=0; window_proposed=0; maxdrift=0.; maxlog=0.; maxasym=0.
    refreshes=0; recorded=0; partial=paths.samples*".partial"
    try
        open(partial,"w") do io
            println(io,HEADER)
            for sweep in 1:job["sweeps"]
                stats=sweep!(state,model.G,rng)
                accepted+=stats.accepted; proposed+=stats.proposed
                window_accepted+=stats.accepted; window_proposed+=stats.proposed
                refreshed=sweep%job["refresh_every"]==0
                drift=(;inverse_drift=0.,logweight_drift=0.)
                if refreshed
                    maxasym=max(maxasym,norm(state.X-state.X',Inf))
                    drift=rebuild!(state,model.G); refreshes+=1
                    maxdrift=max(maxdrift,drift.inverse_drift); maxlog=max(maxlog,drift.logweight_drift)
                    drift.inverse_drift<=plan["max_inverse_drift_tolerance"] || error("inverse drift tolerance exceeded")
                    drift.logweight_drift<=plan["max_logweight_drift_tolerance"] || error("logweight drift tolerance exceeded")
                end
                if sweep%job["measure_every"]==0
                    entropy=entropy!(work,model,state.E,job["beta"]); recorded+=1
                    csv(io,row(job,sweep,recorded,entropy,window_accepted,window_proposed,state.log_weight;
                        refreshed,inverse_drift=drift.inverse_drift,logweight_drift=drift.logweight_drift))
                    window_accepted=0; window_proposed=0
                end
                sweep%256==0 && flush(io)
            end
        end
        # Last partial interval gets a final numerical audit without creating a sample.
        drift=rebuild!(state,model.G)
        maxdrift=max(maxdrift,drift.inverse_drift); maxlog=max(maxlog,drift.logweight_drift)
        maxdrift<=plan["max_inverse_drift_tolerance"] && maxlog<=plan["max_logweight_drift_tolerance"] || error("final numerical drift tolerance exceeded")
        mv(partial,paths.samples;force=true)
        receipt["status"]="complete"; receipt["completed_at"]=string(now()); receipt["elapsed_seconds"]=time()-start_time
        receipt["accepted"]=accepted; receipt["proposed"]=proposed; receipt["acceptance"]=accepted/proposed
        receipt["records"]=recorded; receipt["refreshes"]=refreshes; receipt["maximum_inverse_drift"]=maxdrift
        receipt["maximum_logweight_drift"]=maxlog; receipt["maximum_inverse_asymmetry"]=maxasym
        receipt["final_E"]=copy(state.E)
        metadata=TOML.parsefile(joinpath(ROOT,job["model_manifest"]))
        receipt["model_files"]=metadata["files"]
        receipt["files"]=Dict(rel(p)=>sha(p) for p in (paths.samples,paths.initial))
        write_toml(paths.receipt,receipt)
        validate_completed(job,plan_path) || error("post-write validation failed")
        println("DONE $(job["run_id"]) seconds=$(round(receipt["elapsed_seconds"],digits=2)) acceptance=$(round(accepted/proposed,digits=4)) drift=$maxdrift")
        flush(stdout)
    catch exception
        receipt["status"]="failed"; receipt["elapsed_seconds"]=time()-start_time
        receipt["error"]=sprint(showerror,exception); write_toml(paths.receipt,receipt)
        rethrow()
    end
end

function finalize(plan_path)
    plan=validate_plan(plan_path); files=Dict{String,String}(); chain_receipts=Dict{String,Any}[]
    for job in plan["jobs"]
        validate_completed(job,plan_path) || error("chain is not complete: $(job["run_id"])")
        path=paths_for(job).receipt; receipt=TOML.parsefile(path)
        files[rel(path)]=sha(path); merge!(files,receipt["files"])
        push!(chain_receipts,Dict("run_id"=>job["run_id"],"path"=>rel(path),"sha256"=>sha(path),
            "maximum_inverse_drift"=>receipt["maximum_inverse_drift"],"maximum_logweight_drift"=>receipt["maximum_logweight_drift"],
            "elapsed_seconds"=>receipt["elapsed_seconds"],"records"=>receipt["records"]))
    end
    for model in plan["models"]; merge!(files,model["files"]); end
    files[rel(plan_path)]=sha(plan_path)
    for (file,hash) in files; sha(joinpath(ROOT,file))==hash || error("file changed before finalization"); end
    result=Dict("status"=>"complete","stage"=>plan["stage"],"phase"=>PHASE,"created_at"=>string(now()),
        "plan_path"=>rel(plan_path),"plan_sha256"=>sha(plan_path),"source_sha256"=>plan["source_sha256"],
        "state"=>plan["state"],"mu"=>plan["mu"],"files"=>files,"jobs"=>plan["jobs"],"chains"=>chain_receipts,
        "total_records"=>sum(job["sweeps"]÷job["measure_every"] for job in plan["jobs"]),
        "burnin_discarded"=>0,"columns"=>collect(COLUMNS),"julia_version"=>string(VERSION),"blas_threads_per_worker"=>1)
    path=joinpath(dirname(plan_path),"$(plan["stage"])_manifest.toml")
    write_toml(path,result)
    println("FINALIZED $path chains=$(length(chain_receipts)) records=$(result["total_records"])")
end

function main()
    if flag("prepare-only")
        prepare(arg("stage","pilot")); return
    end
    value=arg("plans",arg("plan","")); isempty(value) && error("provide --plan or --plans")
    paths=abspath.(split(value,','))
    worker=parse(Int,arg("worker","1")); workers=parse(Int,arg("workers","1"))
    1<=worker<=workers<=4 || error("worker count must be from one to four")
    for path in paths
        if flag("finalize"); finalize(path); continue; end
        plan=validate_plan(path)
        for (index,job) in enumerate(plan["jobs"])
            mod(index-1,workers)==worker-1 || continue
            run_job(job,plan,path)
        end
    end
end
if abspath(PROGRAM_FILE)==@__FILE__; main(); end
