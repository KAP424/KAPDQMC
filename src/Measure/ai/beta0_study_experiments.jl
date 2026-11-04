# Beta-zero follow-up. Existing sources, numerical kernels and datasets are read-only.
# Exact support is retained at finite theta and in the accessible infinite-time
# limit. A blocked product state is never relabeled a ground state.
include(joinpath(@__DIR__, "beta_patterns_experiments.jl"))
include(joinpath(@__DIR__, "occupancy_classes_patterns.jl"))
include(joinpath(@__DIR__, "irregular_m3_patterns.jl"))
include(joinpath(@__DIR__, "second_class_extra_patterns.jl"))

const B0_COLUMNS = vcat(BP_COLUMNS, ["class_id", "ground_reachable", "limit_state",
    "projection_method", "job_file_sha256"])
const B0_CODE_FILES = ["beta0_study_experiments.jl", "beta_patterns_experiments.jl",
    "fermion_experiment.jl", "real_entropy_extension.jl", "beta_patterns_strip_extension.jl",
    "experiment_patterns.jl", "arrangement_patterns.jl", "occupancy_classes_patterns.jl",
    "irregular_m3_patterns.jl", "second_class_extra_patterns.jl"]

"""All existing catalogs; repeated names must have identical definitions."""
function beta0_study_patterns()
    catalog = Dict{String,Any}()
    for p in vcat(experiment_patterns(), matched_arrangement_patterns(),
                  occupancy_class_patterns(), irregular_m3_patterns(), second_class_extra_patterns())
        n = [Int(p.occupied[2i-1])+Int(p.occupied[2i]) for i in 1:p.m^2]
        cls = all(==(1),n) ? 1 : all(!=(1),n) ? 2 : 3
        hasproperty(p,:class_id) && p.class_id != cls && error("cell class disagrees: $(p.name)")
        item = (; name=p.name, family=p.family, m=p.m, occupied=p.occupied, class_id=cls,
            expected_ground=hasproperty(p,:expected_ground) ? p.expected_ground : nothing)
        count(p.occupied)==p.m^2 && length(p.occupied)==2p.m^2 || error("invalid filling")
        if haskey(catalog,p.name)
            old=catalog[p.name]
            old.m==p.m && old.family==p.family && old.occupied==p.occupied ||
                error("conflicting catalog definition: $(p.name)")
            isnothing(item.expected_ground) && continue
        end
        catalog[p.name]=item
    end
    sort(collect(values(catalog)); by=p->p.name)
end

function b0_jobs(catalog; jobfile="", names="uniform_A", sizes="72,108", thetas="1:0.5:10")
    byname=Dict(p.name=>p for p in catalog)
    jobs=Tuple{Any,Int,Float64}[]
    if !isempty(jobfile)
        rows,_=bp_rows(jobfile)
        for r in rows
            all(haskey(r,k) for k in ("pattern","L","theta")) || error("job file requires pattern,L,theta")
            haskey(r,"beta") && parse(Float64,r["beta"])!=0 && error("beta-zero jobs only")
            haskey(byname,r["pattern"]) || error("unknown pattern $(r["pattern"])")
            push!(jobs,(byname[r["pattern"]],parse(Int,r["L"]),parse(Float64,r["theta"])))
        end
    else
        selected=String.(split(names,','))
        length(unique(selected))==length(selected) || error("duplicate selected pattern")
        all(haskey(byname,n) for n in selected) || error("unknown selected pattern")
        ls=bp_grid(sizes); all(isinteger,ls) || error("L must be integral")
        for L in Int.(ls), name in selected, theta in bp_grid(thetas)
            push!(jobs,(byname[name],L,theta))
        end
    end
    for (p,L,t) in jobs
        L>=3 && L%(3p.m)==0 || error("incompatible L=$L m=$(p.m): require L % (3m) == 0")
        t>=0 && !isnan(t) || error("theta must be nonnegative or Inf")
        isfinite(t) && abs(t-round(t;digits=10))>1e-12 && error("theta precision exceeds 10 decimal places")
    end
    keys=[(p.name,L,t,0.) for (p,L,t) in jobs]
    length(unique(keys))==length(keys) || error("duplicate exact job key")
    sort(jobs;by=j->(j[2],j[1].m,j[1].name,j[3]))
end

"""Two dense Float64 matrices dominate rectangle memory, not the Bloch state."""
b0_dense_bytes(L) = 16 * (4Int128(L)^2÷9)^2

function b0_entropies(state; rectangle_threads=8, strip_threads=1)
    if state.theta==0
        # At beta=0 the diagonal initial projector restricts to a projector on
        # every region. This shortcut is independently checked in validation.
        return (;sa=0.,sc=0.,sb=0.,stripseconds=0.,rectseconds=0.,
            imag_error=0.,real_error_bound=0.,method="exact_product_state")
    end
    start=time(); sa=0.; sc=0.
    BLAS.set_num_threads(strip_threads)
    try
        sa=BetaPatternsStripExtension.entropy_strip_folded(state)
        sc=BetaPatternsStripExtension.entropy_strip_folded(state;orientation=:y)
    finally
        BLAS.set_num_threads(rectangle_threads)
    end
    stripseconds=time()-start; start=time()
    sb,check=RealEntropyExtension.entropy_rectangle_real(state;with_checks=true)
    (;sa,sc,sb,stripseconds,rectseconds=time()-start,imag_error=check.max_imaginary,
        real_error_bound=check.real_part_entropy_bound,method="real_cholesky")
end

function beta0_study_main()
    mode=bp_arg("mode","plan"); mode in ("plan","run") || error("mode must be plan or run")
    all(==(0),bp_grid(bp_arg("betas","0"))) || error("this runner only accepts beta=0")
    catalog=beta0_study_patterns(); byname=Dict(p.name=>p for p in catalog)
    jobfile=bp_arg("job-file",""); isempty(jobfile) || (jobfile=abspath(jobfile))
    planfile=bp_arg("plan",""); isempty(planfile) || (planfile=abspath(planfile))
    jobs=b0_jobs(catalog;jobfile,names=bp_arg("patterns","uniform_A"),
        sizes=bp_arg("Ls","72,108"),thetas=bp_arg("thetas","1:0.5:10"))
    threads=parse(Int,bp_arg("threads","8")); stripthreads=parse(Int,bp_arg("strip-threads","1"))
    threads>0 && stripthreads>0 || error("threads must be positive")
    BLAS.set_num_threads(threads)
    backend=sprint(show,BLAS.get_config())
    occursin(BP_REQUESTED_BACKEND=="mkl" ? "mkl" : "openblas",lowercase(backend)) || error("BLAS backend mismatch")
    maxgib=parse(Float64,bp_arg("max-dense-gib","24"))
    all(j->b0_dense_bytes(j[2])/2.0^30<=maxgib,jobs) || error("two-matrix rectangle estimate exceeds --max-dense-gib")
    datadir=joinpath(@__DIR__,"data")
    output=abspath(bp_arg("output",joinpath(datadir,"beta0_study_observations.csv")))
    startswith(basename(output),"beta0_study") && endswith(output,".csv") || error("output must use beta0_study*.csv")
    ledger=replace(output,r"\.csv$"=>"_provenance.tsv")
    old,lines=bp_rows(output)
    if isfile(output)
        String.(split(first(split(read(output,String),'\n')),','))==B0_COLUMNS || error("output schema mismatch")
        for r in old
            p=get(byname,r["pattern"],nothing); isnothing(p) && error("unknown saved pattern")
            r["occupied_bits"]==join(Int.(p.occupied)) && parse(Int,r["m"])==p.m || error("saved catalog mismatch")
            parse(Float64,r["beta"])==0 && parse(Float64,r["rank_tol"])==1e-10 || error("saved protocol mismatch")
        end
    end
    known=Set(bp_key(r) for r in old); length(known)==length(old) || error("duplicate saved key")
    pending=[j for j in jobs if !((j[1].name,j[2],j[3],0.) in known)]
    hashes=Dict(f=>bp_hash(joinpath(@__DIR__,f)) for f in B0_CODE_FILES)
    jobhash=isempty(jobfile) ? "" : bp_hash(jobfile)
    runid=Dates.format(now(Dates.UTC),"yyyymmddTHHMMSSsss")
    manifest=Dict{String,Any}("run_id"=>runid,"mode"=>mode,"status"=>(mode=="plan" ? "plan" : "running"),
        "started_utc"=>string(now(Dates.UTC)),"output"=>output,"provenance"=>ledger,
        "requested_rows"=>length(jobs),"new_rows"=>length(pending),"already_present"=>length(jobs)-length(pending),
        "jobs"=>[Dict("pattern"=>p.name,"L"=>L,"theta"=>string(t),"beta"=>0.) for (p,L,t) in pending],
        "job_file"=>jobfile,"job_file_sha256"=>jobhash,"plan_file"=>planfile,
        "plan_sha256"=>(isempty(planfile) ? "" : bp_hash(planfile)),"code_sha256"=>hashes,
        "rank_tol"=>1e-10,"beta"=>[0.],"blas_threads"=>threads,"strip_threads"=>stripthreads,
        "blas_backend"=>BP_REQUESTED_BACKEND,"blas_config"=>backend,"julia"=>string(VERSION),
        "active_project"=>something(Base.active_project(),""),"ground_checks"=>Any[],
        "memory_estimates"=>[Dict("L"=>L,"rectangle_orbitals"=>4L^2÷9,
            "two_dense_bytes"=>Int(b0_dense_bytes(L)),"two_dense_GiB"=>Float64(b0_dense_bytes(L)/2.0^30))
            for L in sort(unique(j[2] for j in jobs))],
        "optimizations"=>["reuse identical L,m Bloch eigensystems across initial patterns",
            "theta=0 exact product-state entropy", "validated conjugate momentum folding for strips"],
        "protocol"=>"beta=0 only; exact spectral-support finite theta or accessible Inf; preserve initial-selected zero modes; blocked Inf is an excited state, never ground equilibrium; premeasurement_ground and ground_reachable mean asymptotic reachability, not finite-time convergence; one species natural-log S2; fixed origin (1,1); D=(2SA+SC)/3-SB; independently evaluate both strips; no copying symmetry-related entropy rows")
    mkpath(dirname(output))
    manifestpath=replace(output,r"\.csv$"=>"_$(mode)_$runid.toml")
    println("Beta-zero $mode: $(length(jobs)) requested, $(length(pending)) missing, $(length(jobs)-length(pending)) already present")
    open(io->TOML.print(io,manifest),manifestpath,"w")
    mode=="plan" && return manifestpath
    snapshots=Dict{String,String}()
    for f in B0_CODE_FILES
        snapshot=joinpath(datadir,"provenance","code",hashes[f],f)
        mkpath(dirname(snapshot)); isfile(snapshot) || cp(joinpath(@__DIR__,f),snapshot)
        bp_hash(snapshot)==hashes[f] || error("code snapshot mismatch")
        snapshots[f]=snapshot
    end
    for (kind,path,hash) in (("jobs",jobfile,jobhash),("plan",planfile,manifest["plan_sha256"]))
        isempty(path) && continue
        snapshot=joinpath(datadir,"provenance",kind,hash,basename(path))
        mkpath(dirname(snapshot)); isfile(snapshot) || cp(path,snapshot)
        bp_hash(snapshot)==hash || error("request snapshot mismatch")
        manifest[kind*"_snapshot"]=snapshot
    end
    manifest["code_snapshots"]=snapshots
    isfile(output) || write(output,join(B0_COLUMNS,',')*"\n")
    bp_restore_ledger(output,ledger,old,lines)
    active_model=nothing; model=nothing; active_pattern=nothing; prepared=nothing; limit=nothing
    try
        open(output,"a") do io
            open(ledger,"a") do journal
                for (i,(p,L,theta)) in enumerate(pending)
                    beginpoint=time(); prepseconds=0.
                    if active_pattern!=(p.name,L)
                        prepared=nothing; GC.gc(true); start=time()
                        if active_model!=(L,p.m)
                            model=nothing; GC.gc(true)
                            model=FE.BlochModel(L,p.m); active_model=(L,p.m)
                        end
                        prepared=FE.prepare_pattern(model,p.occupied;tol=1e-10)
                        limit=FE.diagnostics(FE.evolve(prepared,Inf))
                        ground=abs(limit.excess_energy)<1e-8
                        limit.residual<1e-9 || error("nonstationary supported limit")
                        abs(limit.excess_energy-limit.accessible_excess)<1e-8 || error("accessible energy mismatch")
                        isnothing(p.expected_ground) || p.expected_ground==ground || error("catalog reachability mismatch")
                        bp_arg("require-ground","false")=="true" && !ground && error("ground-unreachable requested pattern")
                        prepseconds=time()-start; active_pattern=(p.name,L)
                        push!(manifest["ground_checks"],Dict("pattern"=>p.name,"m"=>p.m,"L"=>L,
                            "occupied_bits"=>join(Int.(p.occupied)),"ground_reachable"=>ground,
                            "limit_state"=>(ground ? "ground_state" : "accessible_excited_state"),
                            "excess"=>limit.excess_energy,"residual"=>limit.residual,
                            "min_pivot_sv"=>limit.min_pivot_sv))
                    end
                    start=time(); state=FE.evolve(prepared,theta); evseconds=time()-start
                    e=b0_entropies(state;rectangle_threads=threads,strip_threads=stripthreads)
                    start=time(); d=FE.diagnostics(state); diagseconds=time()-start
                    d.projector_error<1e-10 && d.reconstruction_error<1e-9 || error("orbital checks failed")
                    ground=abs(limit.excess_energy)<1e-8
                    row=Dict(c=>"" for c in B0_COLUMNS)
                    vals=(p.name,p.family,p.m,L,theta,0.,e.sa,e.sc,e.sb,e.sa-e.sb,(2e.sa+e.sc)/3-e.sb,
                        d.energy,d.excess_energy,d.accessible_excess,d.residual,d.min_initial_sv,
                        d.log_initial_overlap2,d.empty_occupancy,d.projector_error,d.reconstruction_error,d.min_pivot_sv,
                        ground,limit.excess_energy,limit.residual,join(Int.(p.occupied)),e.imag_error,e.real_error_bound,time()-beginpoint)
                    for (c,v) in zip(BP_COLUMNS[1:28],vals); row[c]=string(v); end
                    for (c,v) in zip(BP_COLUMNS[35:39],(prepseconds,evseconds,e.stripseconds,e.rectseconds,diagseconds))
                        row[c]=string(v)
                    end
                    row["run_id"]=runid; row["code_sha256"]=hashes["beta0_study_experiments.jl"]
                    row["source_kind"]="computed"; row["entropy_method"]=e.method; row["rank_tol"]="1.0e-10"
                    row["class_id"]=string(p.class_id); row["ground_reachable"]=string(ground)
                    row["limit_state"]=ground ? "ground_state" : "accessible_excited_state"
                    row["projection_method"]="exact_spectral_support"; row["job_file_sha256"]=jobhash
                    line=join((row[c] for c in B0_COLUMNS),',')
                    println(io,line); flush(io)
                    println(journal,bp_ledger_line(row,line,length(old)+i+1)); flush(journal)
                    @printf("%d/%d %s L=%d theta=%g D=%.12g seconds=%.2f ground=%s\n",
                        i,length(pending),p.name,L,theta,parse(Float64,row["D"]),time()-beginpoint,string(ground))
                    flush(stdout); state=nothing; GC.gc(false)
                end
            end
        end
        all(bp_hash(joinpath(@__DIR__,f))==hashes[f] for f in B0_CODE_FILES) || error("source changed during run")
        isempty(jobfile) || bp_hash(jobfile)==jobhash || error("job file changed during run")
        isempty(planfile) || bp_hash(planfile)==manifest["plan_sha256"] || error("plan changed during run")
        manifest["status"]="complete"
    catch err
        manifest["status"]="failed"; manifest["error"]=sprint(showerror,err); rethrow()
    finally
        manifest["finished_utc"]=string(now(Dates.UTC)); manifest["output_sha256"]=bp_hash(output)
        manifest["ledger_sha256"]=bp_hash(ledger)
        open(io->TOML.print(io,manifest),manifestpath,"w")
    end
    manifestpath
end

abspath(PROGRAM_FILE)==abspath(@__FILE__) && beta0_study_main()
