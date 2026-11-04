# Independent dense-beta runner. Only the validated FE and real-entropy kernels are used.
using LinearAlgebra, Dates, SHA, Printf, TOML
include(joinpath(@__DIR__, "fermion_experiment.jl"))
import .FermionExperiment as FE
include(joinpath(@__DIR__, "real_entropy_extension.jl"))
include(joinpath(@__DIR__, "uniformA_entropy_extension.jl"))

const UA_COLUMNS = ["pattern","family","m","L","theta","beta","S_A","S_C","S_B",
 "delta","D","energy","excess_energy","accessible_excess","residual","min_initial_sv",
 "log_initial_overlap2","empty_occupancy","projector_error","reconstruction_error","min_pivot_sv",
 "premeasurement_ground","premeasurement_excess","premeasurement_residual","occupied_bits",
 "rectangle_imag_error","rectangle_real_error_bound","seconds","run_id","code_sha256",
 "source_kind","source_path","source_row","source_sha256","preparation_seconds",
 "evolution_seconds","strip_seconds","rectangle_seconds","diagnostics_seconds","entropy_method"]
const UA_LEDGER_COLUMNS = ["output_row","row_sha256","run_id","source_kind","source_path","source_row","source_sha256","csv_row"]
ua_hash(path)=bytes2hex(sha256(read(path)))
ua_key(r)=(parse(Int,r["L"]),parse(Float64,r["theta"]),parse(Float64,r["beta"]))
function ua_arg(name,default)
    prefix="--$name=";i=findfirst(a->startswith(a,prefix),ARGS)
    isnothing(i) ? default : ARGS[i][length(prefix)+1:end]
end
function ua_grid(raw)
    values=Float64[]
    for piece in split(raw,',')
        q=split(piece,':')
        if length(q)==1
            push!(values,parse(Float64,piece))
        elseif length(q)==3
            a,d,b=parse.(Float64,q);d>0 || error("grid step must be positive")
            append!(values,collect(a:d:b))
        else
            error("use comma values or start:step:end")
        end
    end
    return sort(unique([isfinite(v) ? round(v;digits=10) : v for v in values]))
end
function ua_rows(path)
    isfile(path) || return Dict{String,String}[],String[]
    raw=read(path,String);endswith(raw,"\n") || error("incomplete CSV tail: $path")
    lines=split(chomp(raw),'\n');cols=String.(split(strip(lines[1]),','));rows=Dict{String,String}[]
    for line in lines[2:end]
        values=String.(split(strip(line),','));length(values)==length(cols) || error("invalid CSV row")
        push!(rows,Dict(zip(cols,values)))
    end
    return rows,String.(strip.(lines[2:end]))
end
function ua_ledger_line(row,line,index)
    values=[string(index),bytes2hex(sha256(line)),row["run_id"],row["source_kind"],row["source_path"],row["source_row"],row["source_sha256"],line]
    all(v->!occursin('\t',v)&&!occursin('\n',v),values) || error("unsafe ledger field")
    return join(values,'\t')
end
function ua_restore_ledger(output,ledger,rows,lines)
    expected=[ua_ledger_line(r,l,i+1) for (i,(r,l)) in enumerate(zip(rows,lines))]
    if isfile(ledger)
        raw=read(ledger,String);endswith(raw,"\n") || error("incomplete ledger tail")
        old=split(chomp(raw),'\n')[2:end]
        length(old)<=length(expected) || error("ledger ahead of output")
        all(strip(old[i])==expected[i] for i in eachindex(old)) || error("ledger differs from output")
        if length(old)<length(expected)
            open(ledger,"a") do io
                for line in expected[length(old)+1:end];println(io,line);end
            end
        end
    else
        open(ledger,"w") do io
            println(io,join(UA_LEDGER_COLUMNS,'\t'));foreach(l->println(io,l),expected)
        end
    end
end
function main()
    mode=ua_arg("mode","plan");mode in ("plan","run") || error("mode must be plan or run")
    sizes=Int.(ua_grid(ua_arg("Ls","108,144")))
    thetas=ua_grid(ua_arg("thetas","3:0.5:8"));betas=ua_grid(ua_arg("betas","0:0.1:2"))
    all(L->L>=3&&L%3==0,sizes) || error("L must be multiple of three")
    all(t->t>=0&&!isnan(t),thetas) || error("invalid theta")
    all(b->b>=0&&isfinite(b),betas) || error("invalid beta")
    BLAS.set_num_threads(parse(Int,ua_arg("threads","8")))
    method=ua_arg("method","real_cholesky")
    method in ("real_cholesky","halfcomplex") || error("unknown entropy method")
    datadir=joinpath(@__DIR__,"data")
    output=abspath(ua_arg("output",joinpath(datadir,"uniformA_dense_observations.csv")))
    startswith(basename(output),"uniformA_dense") || error("output must use uniformA_dense prefix")
    ledger=replace(output,r"\.csv$"=>"_provenance.tsv")
    force=ua_arg("recompute","false")=="true"
    originals=[joinpath(datadir,n) for n in ("ground_observations_baseline.csv","ref_calibration_observations.csv",
        "ref_calibration_equilibrium.csv","ref_calibration_late.csv","uniformA_dense_benchmark.csv") if isfile(joinpath(datadir,n))]
    sources=Dict{Tuple{Int,Float64,Float64},Any}();receipts=Dict{String,Any}[]
    for path in originals
        abspath(path)==output && continue
        rows,_=ua_rows(path);hash=ua_hash(path)
        push!(receipts,Dict("path"=>abspath(path),"sha256"=>hash,"rows"=>length(rows)))
        for (i,r) in enumerate(rows)
            r["pattern"]=="uniform_A" || continue
            k=ua_key(r)
            ground=get(r,"premeasurement_ground",get(r,"ground","true"))
            lowercase(ground)=="true" || error("source is not GS-accessible")
            abs(parse(Float64,get(r,"premeasurement_excess",get(r,"accessible_excess","0"))))<1e-8 || error("source excess")
            if haskey(sources,k)
                prior=sources[k][1]
                maximum(abs(parse(Float64,r[f])-parse(Float64,prior[f])) for f in ("S_A","S_C","S_B","D"))<1e-9 || error("conflicting reuse key $k")
            end
            sources[k]=(r,abspath(path),i+1,hash)
        end
    end
    old,lines=ua_rows(output);known=Set(ua_key(r) for r in old)
    length(known)==length(old) || error("duplicate output key")
    jobs=[(L,t,b) for L in sizes for t in thetas for b in betas if !((L,t,b) in known)]
    reuse=count(k->!force&&haskey(sources,k),jobs)
    runid=Dates.format(now(Dates.UTC),"yyyymmddTHHMMSSsss")
    codefiles=["uniformA_dense_experiments.jl","fermion_experiment.jl","real_entropy_extension.jl","uniformA_entropy_extension.jl"]
    hashes=Dict(f=>ua_hash(joinpath(@__DIR__,f)) for f in codefiles)
    snapshot=joinpath(datadir,"provenance","code",hashes["uniformA_dense_experiments.jl"],"uniformA_dense_experiments.jl")
    mkpath(dirname(snapshot));isfile(snapshot)||cp(@__FILE__,snapshot)
    ua_hash(snapshot)==hashes["uniformA_dense_experiments.jl"]||error("source snapshot mismatch")
    manifest=Dict{String,Any}("run_id"=>runid,"mode"=>mode,"status"=>mode=="plan" ? "plan" : "running",
        "started_utc"=>string(now(Dates.UTC)),"L"=>sizes,"theta"=>string.(thetas),"beta"=>betas,
        "new_output_rows"=>length(jobs),"reuse_rows"=>reuse,"new_compute_rows"=>length(jobs)-reuse,
        "already_in_output"=>length(old),"output"=>output,"provenance"=>ledger,"sources"=>receipts,
        "code_sha256"=>hashes,"blas_threads"=>BLAS.get_num_threads(),"julia"=>string(VERSION),
        "protocol"=>"uniform_A bits10; own empty B rows exp(-2beta); exact finite theta; single species natural-log Renyi2; origin1,1; D=(2SA+SC)/3-SB",
        "recompute_existing_sources"=>force,"runner_snapshot"=>snapshot,"entropy_method"=>method,"entropy_checks"=>Dict{String,Any}[])
    manifestpath=joinpath(datadir,"uniformA_dense_$(mode)_$runid.toml")
    open(io->TOML.print(io,manifest),manifestpath,"w")
    println("uniformA: $(length(jobs)) output rows, $reuse reused, $(length(jobs)-reuse) computed, mode=$mode")
    mode=="plan" && return
    isfile(output) || write(output,join(UA_COLUMNS,',')*"\n")
    ua_restore_ledger(output,ledger,old,lines)
    active=nothing;prepared=nothing;limit=nothing;state=nothing;active_theta=nothing;started=time()
    open(output,"a") do io
        open(ledger,"a") do journal
            for (i,k) in enumerate(jobs)
                L,theta,beta=k;beginpoint=time();row=Dict(c=>"" for c in UA_COLUMNS)
                if !force&&haskey(sources,k)
                    r,path,sourcerow,sourcehash=sources[k]
                    for c in UA_COLUMNS;haskey(r,c)&&(row[c]=r[c]);end
                    row["source_kind"]="reused";row["source_path"]=path;row["source_row"]=string(sourcerow);row["source_sha256"]=sourcehash
                    row["entropy_method"]="source_recorded";row["seconds"]="0.0";row["occupied_bits"]="10";row["premeasurement_ground"]="true"
                    row["delta"]=string(parse(Float64,r["S_A"])-parse(Float64,r["S_B"]))
                else
                    prepseconds=0.;evseconds=0.
                    if active!=L
                        prepared=nothing;state=nothing;GC.gc(true);start=time()
                        prepared=FE.prepare_pattern(FE.BlochModel(L,1),Bool[1,0]);limit=FE.diagnostics(FE.evolve(prepared,Inf))
                        abs(limit.excess_energy)<1e-8&&limit.residual<1e-9 || error("GS access failed")
                        prepseconds=time()-start;active=L;active_theta=nothing
                    end
                    if active_theta!=theta
                        start=time();state=FE.evolve(prepared,theta);evseconds=time()-start;active_theta=theta
                    end
                    measured=FE.measure(state,beta)
                    start=time();sa=FE.entropy_strip(measured);sc=FE.entropy_strip(measured;orientation=:y);stripseconds=time()-start
                    abs(sa-sc)<1e-8 || error("uniform_A strip orientation symmetry failed")
                    start=time()
                    sb,check=method=="halfcomplex" ? UniformAEntropyExtension.entropy_rectangle_half(measured;with_checks=true) : RealEntropyExtension.entropy_rectangle_real(measured;with_checks=true)
                    rectseconds=time()-start
                    if method=="halfcomplex"
                        push!(manifest["entropy_checks"],Dict("output_row"=>length(old)+i+1,"L"=>L,"theta"=>string(theta),"beta"=>beta,"max_structure_error"=>check.max_structure_error,"orbital_projector_error"=>check.orbital_projector_error,"workspace_matrix_bytes"=>check.workspace_matrix_bytes))
                    end
                    start=time();d=FE.diagnostics(measured);diagseconds=time()-start
                    d.projector_error<1e-10&&d.reconstruction_error<1e-9 || error("projector check failed")
                    base=("uniform_A","special",1,L,theta,beta,sa,sc,sb,sa-sb,(2sa+sc)/3-sb,d.energy,d.excess_energy,d.accessible_excess,
                        d.residual,d.min_initial_sv,d.log_initial_overlap2,d.empty_occupancy,d.projector_error,d.reconstruction_error,d.min_pivot_sv,
                        true,limit.excess_energy,limit.residual,"10",check.max_imaginary,check.real_part_entropy_bound,time()-beginpoint)
                    for (c,v) in zip(UA_COLUMNS[1:28],base);row[c]=string(v);end
                    row["entropy_method"]=method;row["source_kind"]="computed";row["source_path"]="";row["source_row"]="";row["source_sha256"]=""
                    for (c,v) in zip(UA_COLUMNS[35:39],(prepseconds,evseconds,stripseconds,rectseconds,diagseconds));row[c]=string(v);end
                end
                row["run_id"]=runid;row["code_sha256"]=hashes["uniformA_dense_experiments.jl"]
                line=join((row[c] for c in UA_COLUMNS),',');println(io,line);flush(io)
                println(journal,ua_ledger_line(row,line,length(old)+i+1));flush(journal)
                @printf("%d/%d L=%d theta=%g beta=%g %s D=%.12g seconds=%.2f\n",i,length(jobs),L,theta,beta,row["source_kind"],parse(Float64,row["D"]),time()-beginpoint)
                flush(stdout);GC.gc(false)
            end
        end
    end
    manifest["status"]="complete";manifest["finished_utc"]=string(now(Dates.UTC));manifest["elapsed_seconds"]=time()-started
    manifest["output_sha256_at_completion"]=ua_hash(output);manifest["provenance_sha256_at_completion"]=ua_hash(ledger)
    open(io->TOML.print(io,manifest),manifestpath,"w")
end
abspath(PROGRAM_FILE)==abspath(@__FILE__)&&main()
