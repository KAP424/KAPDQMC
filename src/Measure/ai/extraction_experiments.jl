using LinearAlgebra, Dates, SHA, Printf
include(joinpath(@__DIR__, "fermion_experiment.jl"))
import .FermionExperiment as FE
include(joinpath(@__DIR__, "experiment_patterns.jl"))
include(joinpath(@__DIR__, "arrangement_patterns.jl"))

const EX_DATA = joinpath(@__DIR__, "data")
const EX_COLUMNS = ["pattern", "family", "m", "L", "theta", "beta", "S_A", "S_C", "S_B",
    "delta", "D", "energy", "excess_energy", "accessible_excess", "residual", "min_initial_sv",
    "log_initial_overlap2", "empty_occupancy", "projector_error", "reconstruction_error", "min_pivot_sv"]
const EX_PROVENANCE = ["pattern", "L", "theta", "beta", "row_sha256", "source", "source_file",
    "source_file_sha256", "source_line", "source_row_sha256", "run_id", "code_sha256", "pattern_bits",
    "premeasurement_ground", "premeasurement_limit_excess", "premeasurement_limit_residual",
    "calculation_seconds", "observation_csv"]
const EX_GS = Set(["uniform_A", "uniform_B", "one_B_2", "one_B_3", "two_B_3",
    "diagonal_B_3", "full_empty_cluster3", "random_half_3"])

function ex_arg(name, default)
    prefix = "--$name="
    i = findfirst(a -> startswith(a,prefix),ARGS)
    return isnothing(i) ? default : ARGS[i][length(prefix)+1:end]
end

ex_hash(s::AbstractString) = bytes2hex(sha256(s))
ex_filehash(path) = bytes2hex(sha256(read(path)))
ex_key(r) = (r[1],parse(Int,r[4]),parse(Float64,r[5]),parse(Float64,r[6]))

"""Read a simple unquoted table; only our own interrupted last line may be repaired."""
function ex_table(path, columns; delimiter=',', repair=false)
    isfile(path) || return Tuple{Vector{String},Int,String}[]
    contents = read(path,String)
    lines = split(chomp(contents),'\n')
    String.(split(strip(lines[1]),delimiter)) == columns || error("schema mismatch: $path")
    result = Tuple{Vector{String},Int,String}[]
    for (i,line) in enumerate(lines[2:end])
        clean = String(strip(line))
        fields = String.(split(clean,delimiter))
        if length(fields) != length(columns)
            if repair && i == length(lines)-1 && !endswith(contents,"\n")
                backup = path*".partial-"*Dates.format(now(Dates.UTC),"yyyymmddTHHMMSSsss")
                write(backup,line)
                write(path,join(lines[1:end-1],'\n')*"\n")
                println("Recovered incomplete final line in $path; preserved tail in $backup")
                break
            end
            error("malformed row $(i+1) in $path")
        end
        push!(result,(fields,i+1,clean))
    end
    return result
end

function ex_source_rows(paths)
    source = Dict()
    hashes = Dict{String,String}()
    for path in paths
        isfile(path) || continue
        hashes[path] = ex_filehash(path)
        for (row,line,raw) in ex_table(path,EX_COLUMNS)
            key = ex_key(row)
            if haskey(source,key)
                previous = source[key].row
                previous[1:3] == row[1:3] || error("inconsistent source identity at $key")
                for j in 4:length(row)
                    a,b = parse(Float64,previous[j]),parse(Float64,row[j])
                    isapprox(a,b;atol=1e-8,rtol=1e-10) || error("conflicting old observations at $key column $(EX_COLUMNS[j])")
                end
            else
                source[key] = (;row,raw,path,line,filehash=hashes[path])
            end
        end
    end
    return source,hashes
end

"""
The provenance TSV is a write-ahead journal containing the exact observation
CSV line and its SHA256. On restart, committed journal rows missing from the
observation CSV are restored before calculating anything. No old inputs change.
"""
function ex_existing(output, provenance; repair=false)
    saved = Dict()
    for (entry,line,_) in ex_table(provenance,EX_PROVENANCE;delimiter='\t',repair=repair)
        raw = entry[end]
        ex_hash(raw) == entry[5] || error("observation hash mismatch at provenance line $line")
        row = String.(split(raw,','))
        length(row) == length(EX_COLUMNS) || error("bad observation in provenance line $line")
        key = ex_key(row)
        (entry[1],parse(Int,entry[2]),parse(Float64,entry[3]),parse(Float64,entry[4])) == key ||
            error("provenance identity mismatch")
        haskey(saved,key) && error("duplicate provenance key $key")
        saved[key] = (;raw,entry)
    end
    observed = Dict()
    for (row,_,raw) in ex_table(output,EX_COLUMNS;repair=repair)
        key = ex_key(row)
        haskey(observed,key) && error("duplicate observation key $key")
        haskey(saved,key) || error("observation $key has no provenance journal entry")
        saved[key].raw == raw || error("observation differs from journal at $key")
        observed[key] = raw
    end
    restore = [key for key in keys(saved) if !haskey(observed,key)]
    if repair && !isempty(restore)
        isfile(output) || write(output,join(EX_COLUMNS,',')*"\n")
        open(output,"a") do io
            for key in sort(restore;by=string)
                println(io,saved[key].raw)
            end
        end
        println("Restored $(length(restore)) committed journal rows to observations")
    end
    return saved,length(restore)
end

function ex_main()
    mode = ex_arg("mode","plan")
    mode in ("plan","run") || error("--mode must be plan or run")
    threads = parse(Int,ex_arg("threads","4"))
    threads > 0 || error("threads must be positive")
    BLAS.set_num_threads(threads)
    catalogue = vcat(filter(p -> p.name in EX_GS,experiment_patterns()),matched_arrangement_patterns())
    requested = split(ex_arg("patterns",""),',')
    names = Set(p.name for p in catalogue)
    requested == [""] || all(in(names),requested) || error("unknown or non-GS candidate pattern")
    patterns = filter(p -> requested == [""] || p.name in requested,catalogue)
    sizes = sort(unique(parse.(Int,split(ex_arg("Ls","54,72"),','))))
    thetas = sort(unique(parse.(Float64,split(ex_arg("thetas","2,3,4,5,6,7,8,10,12,14,16"),','))))
    betas = sort(unique(parse.(Float64,split(ex_arg("betas","0,0.5,1,2"),','))))
    all(L -> L>0 && all(p -> L%(3p.m)==0,patterns),sizes) || error("every L must be divisible by 3m for every selected pattern")
    all(t -> isfinite(t) && t>=0,thetas) || error("theta must be finite and nonnegative")
    all(b -> isfinite(b) && b>=0,betas) || error("beta must be finite and nonnegative")
    output = abspath(ex_arg("output",joinpath(EX_DATA,"extraction_observations.csv")))
    stem = replace(splitext(output)[1],r"_observations$"=>"")
    provenance,manifest = stem*"_provenance.tsv",stem*"_manifest.txt"
    default_inputs = join([joinpath(EX_DATA,f) for f in
        ("observations.csv","extended_observations.csv","late_time_observations.csv")],';')
    paths = abspath.(split(ex_arg("inputs",default_inputs),';'))
    source,sourcehashes = ex_source_rows(paths)
    existing,recovery = ex_existing(output,provenance;repair=mode=="run")
    # A changed named configuration must never silently reuse a previous run.
    bits = Dict(p.name=>join(Int.(p.occupied)) for p in catalogue)
    for (key,value) in existing
        haskey(bits,key[1]) && value.entry[13] == bits[key[1]] || error("configuration catalogue changed at $key")
    end
    jobs = [(;pattern=p,L,theta,beta) for L in sizes for p in patterns for theta in thetas for beta in betas]
    remaining = filter(j -> !haskey(existing,(j.pattern.name,j.L,j.theta,j.beta)),jobs)
    reusable = count(j -> haskey(source,(j.pattern.name,j.L,j.theta,j.beta)),remaining)
    newjobs = filter(j -> !haskey(source,(j.pattern.name,j.L,j.theta,j.beta)),remaining)
    estimate = sum(1.8*(j.L/90)^6 + .04*(j.L/54)^3 for j in newjobs;init=0.0)
    println("Extraction mode=$mode requested=$(length(jobs)) complete=$(length(jobs)-length(remaining)) restore=$recovery reusable=$reusable new=$(length(newjobs)) estimated_compute_seconds=$(round(estimate;digits=1))")
    for p in patterns
        pending = filter(j->j.pattern.name==p.name,remaining)
        reused = count(j->haskey(source,(j.pattern.name,j.L,j.theta,j.beta)),pending)
        println("$(p.name): rows=$(length(sizes)*length(thetas)*length(betas)) pending=$(length(pending)) reusable=$reused new=$(length(pending)-reused)")
    end
    println("output=$output\nprovenance=$provenance\nmanifest=$manifest")
    mode == "plan" && return
    mkpath(dirname(output))
    isfile(output) || write(output,join(EX_COLUMNS,',')*"\n")
    isfile(provenance) || write(provenance,join(EX_PROVENANCE,'\t')*"\n")
    runid = Dates.format(now(Dates.UTC),"yyyymmddTHHMMSSsss")
    codehash = ex_filehash(@__FILE__)
    open(manifest,"a") do io
        println(io,"START UTC=$(now(Dates.UTC)); run_id=$runid; Julia=$VERSION; BLAS_threads=$threads; args=$(join(ARGS,' '))")
        println(io,"patterns=$(join((p.name for p in patterns),':')); L=$(join(sizes,':')); theta=$(join(thetas,':')); beta=$(join(betas,':')); requested=$(length(jobs)); reusable=$reusable; new=$(length(newjobs))")
        println(io,"Single-species natural-log Renyi2. A=L/3 by L; C=L by L/3; B=L/3 by2L/3; D=(2SA+SC)/3-SB. Initial limit classification is measured before applying beta; finite theta evolution is never replaced by that limit.")
        println(io,"Provenance is the write-ahead source of truth; observation_csv and row_sha256 verify each exact CSV line. A process interruption between journal and CSV writes is restored on restart. Source precedence follows inputs order; duplicate source keys must agree numerically.")
        for (path,hash) in sort(collect(sourcehashes);by=first)
            println(io,"SOURCE_SHA256 $path $hash")
        end
        for file in ("fermion_experiment.jl","experiment_patterns.jl","arrangement_patterns.jl","extraction_experiments.jl")
            println(io,"CODE_SHA256 $file $(ex_filehash(joinpath(@__DIR__,file)))")
        end
    end
    active_group,active_theta = nothing,nothing
    prepared,state = nothing,nothing
    limit_diagnostic = nothing
    started = time()
    open(output,"a") do dataio
        open(provenance,"a") do journal
            for (i,j) in enumerate(remaining)
                p,L,theta,beta = j.pattern,j.L,j.theta,j.beta
                key = (p.name,L,theta,beta)
                if active_group != (p.name,L)
                    prepared = FE.prepare_pattern(FE.BlochModel(L,p.m),p.occupied)
                    limit_diagnostic = FE.diagnostics(FE.evolve(prepared,Inf))
                    limit_diagnostic.residual < 1e-9 || error("infinite-time limit failed stationarity check")
                    active_group,active_theta,state = (p.name,L),nothing,nothing
                    GC.gc(false)
                end
                ground = abs(limit_diagnostic.excess_energy)<1e-8
                if haskey(source,key)
                    old = source[key]
                    old.row[2] == p.family && parse(Int,old.row[3]) == p.m || error("source pattern metadata mismatch")
                    abs(parse(Float64,old.row[14])-prepared.accessible_energy+prepared.model.ground_energy)<1e-7 ||
                        error("source accessible energy mismatch at $key")
                    raw = old.raw
                    sourcekind,sourcefile,sourcehash,sourceline = "reused",old.path,old.filehash,old.line
                    seconds = 0.0
                else
                    calcstart = time()
                    if active_theta != theta
                        state = FE.evolve(prepared,theta)
                        active_theta = theta
                    end
                    measured = FE.measure(state,beta)
                    sa = FE.entropy_strip(measured)
                    sc = FE.entropy_strip(measured;orientation=:y)
                    sb = FE.entropy_rectangle(measured)
                    d = FE.diagnostics(measured)
                    raw = join((p.name,p.family,p.m,L,theta,beta,sa,sc,sb,sa-sb,(2sa+sc)/3-sb,
                        d.energy,d.excess_energy,d.accessible_excess,d.residual,d.min_initial_sv,
                        d.log_initial_overlap2,d.empty_occupancy,d.projector_error,d.reconstruction_error,d.min_pivot_sv),',')
                    sourcekind,sourcefile,sourcehash,sourceline = "computed",abspath(@__FILE__),codehash,0
                    seconds = time()-calcstart
                end
                rowhash = ex_hash(raw)
                entry = (p.name,L,theta,beta,rowhash,sourcekind,sourcefile,sourcehash,sourceline,
                    rowhash,runid,codehash,bits[p.name],ground,limit_diagnostic.excess_energy,
                    limit_diagnostic.residual,seconds,raw)
                println(journal,join(entry,'\t')); flush(journal)
                println(dataio,raw); flush(dataio)
                @printf("%d/%d %s L=%d theta=%g beta=%g ground=%s source=%s seconds=%.3f\n",
                    i,length(remaining),p.name,L,theta,beta,string(ground),sourcekind,seconds)
                flush(stdout)
            end
        end
    end
    final,unrestored = ex_existing(output,provenance)
    unrestored == 0 || error("journal contains observations not yet materialized")
    all(j->haskey(final,(j.pattern.name,j.L,j.theta,j.beta)),jobs) || error("scan coverage incomplete")
    open(manifest,"a") do io
        println(io,"FINISH UTC=$(now(Dates.UTC)); run_id=$runid; total_rows=$(length(final)); requested=$(length(jobs)); written=$(length(remaining)); elapsed_seconds=$(time()-started)")
        println(io,"OBSERVATIONS_SHA256=$(ex_filehash(output))")
        println(io,"PROVENANCE_SHA256=$(ex_filehash(provenance))")
    end
    println("Completed extraction scan: $(length(jobs)) requested keys present; $(length(final)) total rows")
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && ex_main()
