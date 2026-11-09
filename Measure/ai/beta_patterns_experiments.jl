# Dense-beta calculations for additional configurations. Existing kernels and data are read-only.
# Finite theta always means exact FE.evolve at that theta followed by FE.measure.
using LinearAlgebra, Dates, SHA, Printf, TOML
const BP_REQUESTED_BACKEND = let
    prefix="--backend="; i=findfirst(a->startswith(a,prefix),ARGS)
    isnothing(i) ? "openblas" : ARGS[i][length(prefix)+1:end]
end
BP_REQUESTED_BACKEND in ("openblas","mkl") || error("unknown BLAS backend")
if BP_REQUESTED_BACKEND=="mkl"
    @eval using MKL
end
include(joinpath(@__DIR__, "fermion_experiment.jl"))
import .FermionExperiment as FE
include(joinpath(@__DIR__, "real_entropy_extension.jl"))
include(joinpath(@__DIR__, "beta_patterns_strip_extension.jl"))
include(joinpath(@__DIR__, "experiment_patterns.jl"))
include(joinpath(@__DIR__, "arrangement_patterns.jl"))

const BP_COLUMNS = ["pattern","family","m","L","theta","beta","S_A","S_C","S_B",
 "delta","D","energy","excess_energy","accessible_excess","residual","min_initial_sv",
 "log_initial_overlap2","empty_occupancy","projector_error","reconstruction_error","min_pivot_sv",
 "premeasurement_ground","premeasurement_excess","premeasurement_residual","occupied_bits",
 "rectangle_imag_error","rectangle_real_error_bound","seconds","run_id","code_sha256",
 "source_kind","source_path","source_row","source_sha256","preparation_seconds",
 "evolution_seconds","strip_seconds","rectangle_seconds","diagnostics_seconds","entropy_method",
 "rank_tol"]
const BP_LEDGER_COLUMNS = ["output_row","row_sha256","run_id","source_kind","source_path","source_row","source_sha256","csv_row"]
const BP_SOURCE_NAMES = ["ground_observations_baseline.csv", "ref_calibration_observations.csv",
 "ref_calibration_equilibrium.csv", "ref_calibration_late.csv", "equilibrium_extended.csv",
 "equilibrium_arrangements.csv", "random_half3_beta1_equilibrium.csv", "random_half3_beta1_observations.csv"]
bp_hash(path) = bytes2hex(sha256(read(path)))
bp_key(r) = (r["pattern"], parse(Int,r["L"]), parse(Float64,r["theta"]), round(parse(Float64,r["beta"]);digits=10))
function bp_arg(name,default)
    prefix="--$name="; i=findfirst(a->startswith(a,prefix),ARGS)
    isnothing(i) ? default : ARGS[i][length(prefix)+1:end]
end
function bp_grid(raw)
    values=Float64[]
    for piece in split(raw,',')
        q=split(piece,':')
        if length(q)==1
            push!(values,parse(Float64,piece))
        elseif length(q)==3
            a,d,b=parse.(Float64,q); d>0 || error("grid step must be positive")
            append!(values,collect(a:d:b))
        else
            error("use comma values or start:step:end")
        end
    end
    sort(unique([isfinite(v) ? round(v;digits=10) : v for v in values]))
end
function bp_rows(path)
    isfile(path) || return Dict{String,String}[],String[]
    raw=read(path,String); endswith(raw,"\n") || error("incomplete CSV tail: $path")
    lines=split(chomp(raw),'\n'); cols=String.(split(strip(lines[1]),',')); rows=Dict{String,String}[]
    for line in lines[2:end]
        values=String.(split(strip(line),',')); length(values)==length(cols) || error("invalid CSV row: $path")
        push!(rows,Dict(zip(cols,values)))
    end
    rows,String.(strip.(lines[2:end]))
end
function bp_ledger_line(row,line,index)
    values=[string(index),bytes2hex(sha256(line)),row["run_id"],row["source_kind"],row["source_path"],row["source_row"],row["source_sha256"],line]
    all(v->!occursin('\t',v)&&!occursin('\n',v),values) || error("unsafe ledger field")
    join(values,'\t')
end
function bp_restore_ledger(output,ledger,rows,lines)
    expected=[bp_ledger_line(r,l,i+1) for (i,(r,l)) in enumerate(zip(rows,lines))]
    if isfile(ledger)
        raw=read(ledger,String); endswith(raw,"\n") || error("incomplete ledger tail")
        old=split(chomp(raw),'\n')[2:end]
        length(old)<=length(expected) || error("ledger ahead of output")
        all(strip(old[i])==expected[i] for i in eachindex(old)) || error("ledger differs from output")
        if length(old)<length(expected)
            open(ledger,"a") do io
                for line in expected[length(old)+1:end]; println(io,line); end
            end
        end
    else
        open(ledger,"w") do io
            println(io,join(BP_LEDGER_COLUMNS,'\t')); foreach(l->println(io,l),expected)
        end
    end
end
function bp_source_valid(r,p,path)
    r["family"]==p.family && parse(Int,r["m"])==p.m || error("source catalogue differs: $path")
    parse(Int,r["L"])%(3p.m)==0 || return false
    ground=get(r,"premeasurement_ground",get(r,"ground","false"))
    lowercase(ground)=="true" || return false
    excess=get(r,"premeasurement_excess",get(r,"premeasurement_limit_excess",get(r,"accessible_excess","NaN")))
    abs(parse(Float64,excess))<1e-8 || error("source GS excess: $path")
    if haskey(r,"occupied_bits")
        r["occupied_bits"]==join(Int.(p.occupied)) || error("source pattern bits differ: $path")
    end
    haskey(r,"rank_tol") && parse(Float64,r["rank_tol"])!=1e-10 && error("source rank tolerance differs")
    haskey(r,"grid_version") && !startswith(r["grid_version"],"equilibrium_") && error("unexpected equilibrium protocol")
    a,c,b,d=parse.(Float64,[r[f] for f in ("S_A","S_C","S_B","D")])
    abs((2a+c)/3-b-d)<1e-9 || error("source D convention differs: $path")
    true
end
function main()
    mode=bp_arg("mode","plan"); mode in ("plan","run") || error("mode must be plan or run")
    names=String.(split(bp_arg("patterns","one_B_2,one_B_3,random_half_3"),','))
    length(unique(names))==length(names) || error("duplicate pattern name")
    catalogue=vcat(experiment_patterns(),matched_arrangement_patterns())
    patterns=[only(filter(p->p.name==name,catalogue)) for name in names]
    sizes_raw=bp_grid(bp_arg("Ls","144,180")); all(isinteger,sizes_raw) || error("L must be integer")
    sizes=Int.(sizes_raw)
    thetas=bp_grid(bp_arg("thetas","4:0.5:7")); betas=bp_grid(bp_arg("betas","0:0.1:2"))
    all(L->L>=3&&all(p->L%(3p.m)==0,patterns),sizes) || error("L must preserve every supercell cut phase: L % (3m) = 0")
    all(t->t>=0&&!isnan(t),thetas) || error("invalid theta")
    all(b->b>=0&&isfinite(b),betas) || error("invalid beta")
    threads=parse(Int,bp_arg("threads","8")); threads>0 || error("invalid thread count")
    BLAS.set_num_threads(threads)
    backend_config=sprint(show,BLAS.get_config())
    occursin(BP_REQUESTED_BACKEND=="mkl" ? "mkl" : "openblas",lowercase(backend_config)) || error("requested BLAS backend is not active")
    method=bp_arg("method","real_cholesky"); method=="real_cholesky" || error("only validated general real_cholesky is currently enabled")
    strip_method=bp_arg("strip-method","folded")
    strip_method in ("original","folded") || error("invalid strip method")
    strip_threads=parse(Int,bp_arg("strip-threads","1")); strip_threads>0 || error("invalid strip thread count")
    datadir=joinpath(@__DIR__,"data")
    output=abspath(bp_arg("output",joinpath(datadir,"beta_patterns_observations.csv")))
    startswith(basename(output),"beta_patterns") && endswith(output,".csv") || error("output must use beta_patterns*.csv")
    ledger=replace(output,r"\.csv$"=>"_provenance.tsv")
    force=bp_arg("recompute","false")=="true"
    sources=Dict{Tuple{String,Int,Float64,Float64},Any}(); receipts=Dict{String,Any}[]
    for source_name in BP_SOURCE_NAMES
        path=joinpath(datadir,source_name); isfile(path) || continue
        abspath(path)==output && continue
        rows,_=bp_rows(path); hash=bp_hash(path)
        push!(receipts,Dict("path"=>abspath(path),"sha256"=>hash,"rows"=>length(rows),
            "protocol"=>"validated same FE Hamiltonian, own-empty measurement, single species natural-log S2, origin (1,1), weighted D"))
        for (i,r) in enumerate(rows)
            r["pattern"] in names || continue
            p=only(filter(p->p.name==r["pattern"],patterns))
            bp_source_valid(r,p,path) || continue
            k=bp_key(r)
            if haskey(sources,k)
                prior=sources[k][1]
                maximum(abs(parse(Float64,r[f])-parse(Float64,prior[f])) for f in ("S_A","S_C","S_B","D"))<1e-8 || error("conflicting reuse key $k")
            end
            sources[k]=(r,abspath(path),i+1,hash)
        end
    end
    old,lines=bp_rows(output)
    if isfile(output)
        String.(split(first(split(read(output,String),'\n')),','))==BP_COLUMNS || error("output schema mismatch")
        all(r->parse(Float64,r["rank_tol"])==1e-10,old) || error("saved rank tolerance differs")
    end
    known=Set(bp_key(r) for r in old); length(known)==length(old) || error("duplicate output key")
    jobs=[(p,L,t,b) for L in sizes for p in patterns for t in thetas for b in betas if !((p.name,L,t,b) in known)]
    reuse=count(j->!force&&haskey(sources,(j[1].name,j[2:4]...)),jobs)
    runid=Dates.format(now(Dates.UTC),"yyyymmddTHHMMSSsss")
    codefiles=["beta_patterns_experiments.jl","fermion_experiment.jl","real_entropy_extension.jl","beta_patterns_strip_extension.jl","experiment_patterns.jl","arrangement_patterns.jl"]
    hashes=Dict(f=>bp_hash(joinpath(@__DIR__,f)) for f in codefiles)
    snapshots=Dict{String,String}()
    for f in codefiles
        snapshot=joinpath(datadir,"provenance","code",hashes[f],f)
        mkpath(dirname(snapshot)); isfile(snapshot)||cp(joinpath(@__DIR__,f),snapshot)
        bp_hash(snapshot)==hashes[f] || error("source snapshot mismatch")
        snapshots[f]=snapshot
    end
    manifest=Dict{String,Any}("run_id"=>runid,"mode"=>mode,"status"=>mode=="plan" ? "plan" : "running",
        "started_utc"=>string(now(Dates.UTC)),"patterns"=>names,"L"=>sizes,"theta"=>string.(thetas),"beta"=>betas,
        "new_output_rows"=>length(jobs),"reuse_rows"=>reuse,"new_compute_rows"=>length(jobs)-reuse,
        "already_in_output"=>length(old),"output"=>output,"provenance"=>ledger,"sources"=>receipts,
        "code_sha256"=>hashes,"code_snapshots"=>snapshots,"blas_threads"=>BLAS.get_num_threads(),"julia"=>string(VERSION),
        "blas_backend"=>BP_REQUESTED_BACKEND,"blas_config"=>backend_config,"active_project"=>something(Base.active_project(),""),
        "strip_method"=>strip_method,"strip_threads"=>strip_threads,"rectangle_threads"=>threads,
        "protocol"=>"catalogued half-filled supercells; exact finite theta FE.evolve then own-empty FE.measure exp(-2beta); single species natural-log Renyi2; origin=(1,1); SA and SC independently evaluated; D=(2SA+SC)/3-SB",
        "recompute_existing_sources"=>force,"entropy_method"=>method,"rank_tol"=>1e-10,
        "pattern_catalogue"=>[Dict("name"=>p.name,"family"=>p.family,"m"=>p.m,"occupied_bits"=>join(Int.(p.occupied))) for p in patterns],
        "ground_checks"=>Dict{String,Any}[],"entropy_checks"=>Dict{String,Any}[],"strip_checks"=>Dict{String,Any}[])
    manifestpath=joinpath(datadir,"beta_patterns_$(mode)_$runid.toml")
    mkpath(dirname(output)); open(io->TOML.print(io,manifest),manifestpath,"w")
    println("beta patterns: $(length(jobs)) output rows, $reuse reused, $(length(jobs)-reuse) computed, mode=$mode")
    mode=="plan" && return
    isfile(output) || write(output,join(BP_COLUMNS,',')*"\n")
    bp_restore_ledger(output,ledger,old,lines)
    active=nothing; prepared=nothing; limit=nothing; state=nothing; active_theta=nothing; started=time()
    try
        open(output,"a") do io
            open(ledger,"a") do journal
                for (i,(p,L,theta,beta)) in enumerate(jobs)
                    beginpoint=time(); row=Dict(c=>"" for c in BP_COLUMNS); prepseconds=0.; evseconds=0.
                    if active!=(p.name,L)
                        prepared=nothing; state=nothing; GC.gc(true); start=time()
                        prepared=FE.prepare_pattern(FE.BlochModel(L,p.m),p.occupied;tol=1e-10)
                        limit=FE.diagnostics(FE.evolve(prepared,Inf))
                        abs(limit.excess_energy)<1e-8&&limit.residual<1e-9 || error("GS access failed: $(p.name) L=$L")
                        abs(limit.accessible_excess)<1e-8 || error("accessible-energy mismatch")
                        prepseconds=time()-start; active=(p.name,L); active_theta=nothing
                        push!(manifest["ground_checks"],Dict("pattern"=>p.name,"L"=>L,"excess_energy"=>limit.excess_energy,
                            "accessible_excess"=>limit.accessible_excess,"residual"=>limit.residual,"min_pivot_sv"=>limit.min_pivot_sv))
                        @printf("GS %s L=%d excess=%.4g residual=%.4g\n",p.name,L,limit.excess_energy,limit.residual)
                    end
                    if active_theta!=theta
                        start=time(); state=FE.evolve(prepared,theta); evseconds=time()-start; active_theta=theta
                    end
                    measured=FE.measure(state,beta)
                    k=(p.name,L,theta,beta); stripseconds=0.; rectseconds=0.
                    if !force&&haskey(sources,k)
                        r,path,sourcerow,sourcehash=sources[k]
                        sa,sc,sb=parse.(Float64,[r[f] for f in ("S_A","S_C","S_B")])
                        imagerr=get(r,"rectangle_imag_error",""); realbound=get(r,"rectangle_real_error_bound","")
                        row["source_kind"]="reused"; row["source_path"]=path; row["source_row"]=string(sourcerow); row["source_sha256"]=sourcehash
                        row["entropy_method"]="source_recorded"
                    else
                        start=time()
                        BLAS.set_num_threads(strip_threads)
                        try
                            if strip_method=="folded"
                                sa,check_a=BetaPatternsStripExtension.entropy_strip_folded(measured;with_checks=true)
                                sc,check_c=BetaPatternsStripExtension.entropy_strip_folded(measured;orientation=:y,with_checks=true)
                                push!(manifest["strip_checks"],Dict("output_row"=>length(old)+i+1,"pattern"=>p.name,"L"=>L,
                                    "theta"=>string(theta),"beta"=>beta,"orientation_A"=>Dict(string(k)=>v for (k,v) in pairs(check_a)),
                                    "orientation_C"=>Dict(string(k)=>v for (k,v) in pairs(check_c))))
                            else
                                sa=FE.entropy_strip(measured); sc=FE.entropy_strip(measured;orientation=:y)
                            end
                        finally
                            BLAS.set_num_threads(threads)
                        end
                        stripseconds=time()-start
                        start=time(); sb,check=RealEntropyExtension.entropy_rectangle_real(measured;with_checks=true); rectseconds=time()-start
                        imagerr=string(check.max_imaginary); realbound=string(check.real_part_entropy_bound)
                        push!(manifest["entropy_checks"],Dict("output_row"=>length(old)+i+1,"pattern"=>p.name,"L"=>L,
                            "theta"=>string(theta),"beta"=>beta,"max_imaginary"=>check.max_imaginary,
                            "real_part_entropy_bound"=>check.real_part_entropy_bound,"orbital_projector_error"=>check.orbital_projector_error))
                        row["entropy_method"]=method; row["source_kind"]="computed"
                    end
                    start=time(); d=FE.diagnostics(measured); diagseconds=time()-start
                    d.projector_error<1e-10&&d.reconstruction_error<1e-9 || error("projector/reconstruction check failed")
                    base=(p.name,p.family,p.m,L,theta,beta,sa,sc,sb,sa-sb,(2sa+sc)/3-sb,d.energy,d.excess_energy,d.accessible_excess,
                        d.residual,d.min_initial_sv,d.log_initial_overlap2,d.empty_occupancy,d.projector_error,d.reconstruction_error,d.min_pivot_sv,
                        true,limit.excess_energy,limit.residual,join(Int.(p.occupied)),imagerr,realbound,time()-beginpoint)
                    for (c,v) in zip(BP_COLUMNS[1:28],base); row[c]=string(v); end
                    # Keep source entropy fields byte-identical, including the source D rounding.
                    if row["source_kind"]=="reused"
                        r=sources[k][1]
                        for c in ("S_A","S_C","S_B","D"); row[c]=r[c]; end
                    end
                    for (c,v) in zip(BP_COLUMNS[35:39],(prepseconds,evseconds,stripseconds,rectseconds,diagseconds)); row[c]=string(v); end
                    row["run_id"]=runid; row["code_sha256"]=hashes["beta_patterns_experiments.jl"]; row["rank_tol"]="1.0e-10"
                    line=join((row[c] for c in BP_COLUMNS),','); println(io,line); flush(io)
                    println(journal,bp_ledger_line(row,line,length(old)+i+1)); flush(journal)
                    @printf("%d/%d %s L=%d theta=%g beta=%g %s D=%.12g seconds=%.2f\n",i,length(jobs),p.name,L,theta,beta,row["source_kind"],parse(Float64,row["D"]),time()-beginpoint)
                    flush(stdout); GC.gc(false)
                end
            end
        end
        manifest["status"]="complete"
    catch err
        manifest["status"]="failed"; manifest["error"]=sprint(showerror,err); rethrow()
    finally
        manifest["finished_utc"]=string(now(Dates.UTC)); manifest["elapsed_seconds"]=time()-started
        manifest["output_sha256_at_completion"]=bp_hash(output); manifest["provenance_sha256_at_completion"]=bp_hash(ledger)
        manifest["total_rows_at_completion"]=length(first(bp_rows(output)))
        open(io->TOML.print(io,manifest),manifestpath,"w")
    end
end
abspath(PROGRAM_FILE)==abspath(@__FILE__)&&main()
