#=
Follow-up of the irregular 0/2 pattern, using the validated exact Bloch evolution,
folded strip entropy and real Cholesky rectangle entropy.
Existing experiment files are read-only. Every appended row has a hash ledger.
=#
include(joinpath(@__DIR__, "beta_patterns_experiments.jl"))
include(joinpath(@__DIR__, "occupancy_classes_patterns.jl"))
include(joinpath(@__DIR__, "irregular_m3_patterns.jl"))

function irregular_followup_main()
    names = split(bp_arg("patterns", "near02_irregular3"), ',')
    patterns = [only(filter(p -> p.name == name, vcat(occupancy_class_patterns(), irregular_m3_patterns()))) for name in names]
    sizes = Int.(bp_grid(bp_arg("Ls", "18,27,36,45,54,63,72,81,90,108,126,144")))
    thetas = bp_grid(bp_arg("thetas", "Inf"))
    betas = bp_grid(bp_arg("betas", "0:0.25:2"))
    all(L -> all(p -> L % (3p.m) == 0, patterns), sizes) || error("incompatible cut/supercell")
    all(b -> isfinite(b) && b >= 0, betas) || error("invalid beta")
    all(t -> !isnan(t) && t >= 0, thetas) || error("invalid theta")
    threads = parse(Int, bp_arg("threads", "8"))
    BLAS.set_num_threads(threads)
    datadir = joinpath(@__DIR__, "data")
    output = abspath(bp_arg("output", joinpath(datadir, "irregular_followup_observations.csv")))
    ledger = replace(output, r"\.csv$" => "_provenance.tsv")
    mkpath(dirname(output))
    isfile(output) || write(output, join(BP_COLUMNS, ',') * "\n")
    old, lines = bp_rows(output)
    bp_restore_ledger(output, ledger, old, lines)
    known = Set(bp_key(r) for r in old)
    length(known) == length(old) || error("duplicate output key")
    jobs = [(p,L,t,b) for L in sizes for p in patterns for t in thetas for b in betas
            if !((p.name,L,t,b) in known)]
    runid = Dates.format(now(Dates.UTC), "yyyymmddTHHMMSSsss")
    files = ["irregular_followup_experiments.jl", "irregular_m3_patterns.jl", "occupancy_classes_patterns.jl",
             "beta_patterns_experiments.jl", "fermion_experiment.jl",
             "real_entropy_extension.jl", "beta_patterns_strip_extension.jl"]
    hashes = Dict(f => bp_hash(joinpath(@__DIR__, f)) for f in files)
    for f in files
        snapshot = joinpath(datadir, "provenance", "code", hashes[f], f)
        mkpath(dirname(snapshot))
        isfile(snapshot) || cp(joinpath(@__DIR__, f), snapshot)
        bp_hash(snapshot) == hashes[f] || error("snapshot hash mismatch")
    end
    manifest = Dict{String,Any}("run_id"=>runid, "status"=>"running",
        "started_utc"=>string(now(Dates.UTC)), "patterns"=>names, "L"=>sizes,
        "theta"=>string.(thetas), "beta"=>betas, "new_rows"=>length(jobs),
        "output"=>output, "code_sha256"=>hashes, "rank_tol"=>1e-10,
        "julia"=>string(VERSION), "blas_threads"=>threads,
        "blas_config"=>sprint(show,BLAS.get_config()), "ground_checks"=>Any[],
        "protocol"=>"Exact finite theta -> own-empty exp(-2beta) -> QR; one species natural-log S2; fixed origin; D=(2SA+SC)/3-SB; no SA=SC assumption")
    manifestpath = joinpath(datadir, "irregular_followup_run_$runid.toml")
    open(io -> TOML.print(io,manifest),manifestpath,"w")
    println("Irregular follow-up: $(length(jobs)) new rows")
    active = nothing; active_theta = nothing; prepared = nothing; state = nothing; limit = nothing
    try
        open(output,"a") do io
            open(ledger,"a") do journal
                for (i,(p,L,theta,beta)) in enumerate(jobs)
                    beginpoint = time(); prepseconds = 0.; evseconds = 0.
                    if active != (p.name,L)
                        prepared = nothing; state = nothing; GC.gc(true); start = time()
                        prepared = FE.prepare_pattern(FE.BlochModel(L,p.m),p.occupied;tol=1e-10)
                        limit = FE.diagnostics(FE.evolve(prepared,Inf))
                        abs(limit.excess_energy)<1e-8 && limit.residual<1e-9 || error("GS access failed: $(p.name) L=$L")
                        abs(limit.accessible_excess)<1e-8 || error("GS energy mismatch")
                        prepseconds = time()-start; active = (p.name,L); active_theta = nothing
                        push!(manifest["ground_checks"],Dict("pattern"=>p.name,"L"=>L,
                            "excess"=>limit.excess_energy,"residual"=>limit.residual,
                            "min_pivot_sv"=>limit.min_pivot_sv))
                        @printf("GS %s L=%d excess=%.3g residual=%.3g\n",p.name,L,limit.excess_energy,limit.residual)
                    end
                    if active_theta != theta
                        start=time(); state=FE.evolve(prepared,theta); evseconds=time()-start; active_theta=theta
                    end
                    measured=FE.measure(state,beta)
                    start=time(); BLAS.set_num_threads(1)
                    sa=0.; sc=0.
                    try
                        sa=BetaPatternsStripExtension.entropy_strip_folded(measured)
                        sc=BetaPatternsStripExtension.entropy_strip_folded(measured;orientation=:y)
                    finally
                        BLAS.set_num_threads(threads)
                    end
                    stripseconds=time()-start
                    start=time()
                    sb,check=RealEntropyExtension.entropy_rectangle_real(measured;with_checks=true)
                    rectseconds=time()-start
                    start=time(); d=FE.diagnostics(measured); diagseconds=time()-start
                    d.projector_error<1e-10 && d.reconstruction_error<1e-9 || error("orbital checks failed")
                    row=Dict(c=>"" for c in BP_COLUMNS)
                    values=(p.name,p.family,p.m,L,theta,beta,sa,sc,sb,sa-sb,(2sa+sc)/3-sb,
                        d.energy,d.excess_energy,d.accessible_excess,d.residual,d.min_initial_sv,
                        d.log_initial_overlap2,d.empty_occupancy,d.projector_error,d.reconstruction_error,
                        d.min_pivot_sv,true,limit.excess_energy,limit.residual,join(Int.(p.occupied)),
                        check.max_imaginary,check.real_part_entropy_bound,time()-beginpoint)
                    for (c,v) in zip(BP_COLUMNS[1:28],values); row[c]=string(v); end
                    for (c,v) in zip(BP_COLUMNS[35:39],(prepseconds,evseconds,stripseconds,rectseconds,diagseconds))
                        row[c]=string(v)
                    end
                    row["run_id"]=runid; row["code_sha256"]=hashes["irregular_followup_experiments.jl"]
                    row["source_kind"]="computed"; row["entropy_method"]="real_cholesky"; row["rank_tol"]="1.0e-10"
                    line=join((row[c] for c in BP_COLUMNS),',')
                    println(io,line); flush(io)
                    println(journal,bp_ledger_line(row,line,length(old)+i+1)); flush(journal)
                    @printf("%d/%d %s L=%d theta=%g beta=%g D=%.12g seconds=%.2f\n",
                        i,length(jobs),p.name,L,theta,beta,parse(Float64,row["D"]),time()-beginpoint)
                    flush(stdout); GC.gc(false)
                end
            end
        end
        manifest["status"]="complete"
    catch err
        manifest["status"]="failed"; manifest["error"]=sprint(showerror,err); rethrow()
    finally
        manifest["finished_utc"]=string(now(Dates.UTC))
        manifest["output_sha256"]=bp_hash(output)
        open(io->TOML.print(io,manifest),manifestpath,"w")
    end
end
abspath(PROGRAM_FILE)==abspath(@__FILE__) && irregular_followup_main()
