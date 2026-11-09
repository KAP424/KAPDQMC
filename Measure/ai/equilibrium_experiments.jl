using LinearAlgebra, Dates, SHA, Printf
include(joinpath(@__DIR__, "fermion_experiment.jl"))
import .FermionExperiment as FE
include(joinpath(@__DIR__, "experiment_patterns.jl"))

const EQ_DATA = joinpath(@__DIR__, "data")
const EQ_BASE_COLUMNS = ["pattern", "family", "m", "L", "theta", "beta", "S_A", "S_C", "S_B",
    "delta", "D", "energy", "excess_energy", "accessible_excess", "residual", "min_initial_sv",
    "log_initial_overlap2", "empty_occupancy", "projector_error", "reconstruction_error", "min_pivot_sv"]
const EQ_COLUMNS = vcat(EQ_BASE_COLUMNS, ["premeasurement_ground", "premeasurement_energy",
    "premeasurement_excess", "premeasurement_residual", "ground_energy", "limit_fingerprint",
    "source", "source_file", "source_sha256", "source_row", "calculation_seconds", "scan_stage", "grid_version"])
const EQ_GS_PATTERNS = Set(["uniform_A", "uniform_B", "one_B_2", "one_B_3", "two_B_3",
    "diagonal_B_3", "full_empty_cluster3", "random_half_3"])
const EQ_COMMON_L = [18, 36, 54, 72, 90]
const EQ_BETAS = [0., .25, .5, .75, 1., 1.5, 2., 3., 4.]
const EQ_VERSION = "equilibrium_v2_2026-10-18"

function eq_argument(name, default)
    prefix = "--$name="
    index = findfirst(x -> startswith(x, prefix), ARGS)
    return isnothing(index) ? default : ARGS[index][length(prefix)+1:end]
end

eq_key(row) = (row[1], parse(Int,row[4]), parse(Float64,row[6]))

function eq_read_rows(path, columns; repair_tail=false)
    rows = Dict{Tuple{String,Int,Float64},Tuple{Vector{String},Int}}()
    isfile(path) || return rows
    contents = read(path,String)
    lines = split(chomp(contents),'\n')
    split(strip(lines[1]),',') == columns || error("schema mismatch: $path")
    for (number,line) in enumerate(lines[2:end])
        fields = String.(split(strip(line),','))
        if length(fields) != length(columns)
            if repair_tail && number == length(lines)-1 && !endswith(contents,"\n")
                backup = path * ".partial-" * Dates.format(now(Dates.UTC),"yyyymmddTHHMMSS")
                write(backup,line)
                write(path,join(lines[1:end-1],'\n') * "\n")
                println("Recovered incomplete final output row; tail preserved in $backup")
                break
            end
            error("malformed row $(number+1): $path")
        end
        parse(Float64,fields[5]) == Inf || continue
        key = eq_key(fields)
        haskey(rows,key) && error("duplicate equilibrium key $key in $path")
        rows[key] = (fields,number+1)
    end
    return rows
end

function eq_sizes(pattern)
    if pattern.name in EQ_GS_PATTERNS
        return pattern.m == 1 ? collect(12:6:90) : collect(3pattern.m:3pattern.m:90)
    end
    return sort(unique(vcat(collect(6:6:54),EQ_COMMON_L)))
end

function eq_jobs(patterns, historical; stage="all", maxL=90, include_audits=true)
    jobs = NamedTuple[]
    for pattern in patterns, L in filter(<=(maxL),eq_sizes(pattern)), beta in EQ_BETAS
        current = L in EQ_COMMON_L ? "core" : pattern.name in EQ_GS_PATTERNS ? "gs_dense" : "blocked_legacy"
        stage in ("all",current) || continue
        L % (3pattern.m) == 0 || error("cuts are not aligned to the configuration period")
        push!(jobs,(;pattern,L,beta,stage=current))
    end
    if include_audits && stage in ("all","audit")
        names = Dict(p.name=>p for p in patterns)
        for ((name,L,beta),(row,_)) in historical
            haskey(names,name) && L <= maxL && beta == 8. || continue
            push!(jobs,(pattern=names[name],L=L,beta=beta,stage="audit"))
        end
    end
    priority = Dict("core"=>1,"gs_dense"=>2,"blocked_legacy"=>3,"audit"=>4)
    sort!(jobs;by=j->(priority[j.stage],j.L,j.pattern.name,j.beta))
    return jobs
end

function eq_fingerprint(state)
    buffer = IOBuffer()
    for q in state.orbitals
        write(buffer,q)
    end
    return bytes2hex(sha256(take!(buffer)))
end

function eq_main()
    mode = eq_argument("mode","plan")
    mode in ("plan","run") || error("mode must be plan or run")
    stage = eq_argument("stage","all")
    stage in ("all","core","gs_dense","blocked_legacy","audit") || error("unknown stage")
    threads = parse(Int,eq_argument("threads","4"))
    BLAS.set_num_threads(threads)
    selected = split(eq_argument("patterns",""),',')
    patterns = filter(p -> selected == [""] || p.name in selected,experiment_patterns())
    isempty(patterns) && error("no patterns selected")
    maxL = parse(Int,eq_argument("max-L","90"))
    input = abspath(eq_argument("input",joinpath(EQ_DATA,"observations.csv")))
    output = abspath(eq_argument("output",joinpath(EQ_DATA,"equilibrium_extended.csv")))
    historical = eq_read_rows(input,EQ_BASE_COLUMNS)
    complete = eq_read_rows(output,EQ_COLUMNS;repair_tail=mode=="run")
    jobs = eq_jobs(patterns,historical;stage=stage,maxL=maxL)
    remaining = filter(j -> !haskey(complete,(j.pattern.name,j.L,j.beta)),jobs)
    reused = count(j -> haskey(historical,(j.pattern.name,j.L,j.beta)),remaining)
    println("Grid=$EQ_VERSION mode=$mode stage=$stage selected_rows=$(length(jobs)) already_done=$(length(jobs)-length(remaining)) reusable=$reused new=$(length(remaining)-reused)")
    for pattern in patterns
        localjobs = filter(j->j.pattern.name==pattern.name,jobs)
        println("$(pattern.name): sizes=$(join(sort(unique(j.L for j in localjobs)),':')); rows=$(length(localjobs)); expected_group=$(pattern.name in EQ_GS_PATTERNS ? "GS_candidate" : "blocked_control")")
    end
    mode == "plan" && return
    mkpath(dirname(output))
    isfile(output) || write(output,join(EQ_COLUMNS,',')*"\n")
    input_hash = bytes2hex(sha256(read(input)))
    code_hash = bytes2hex(sha256(read(@__FILE__)))
    manifest = joinpath(dirname(output),"equilibrium_extended_manifest.txt")
    open(manifest,"a") do io
        println(io,"START UTC=$(now(Dates.UTC)); Julia=$VERSION; BLAS_threads=$threads; grid=$EQ_VERSION; args=$(join(ARGS,' '))")
        println(io,"Source=$input; SHA256=$input_hash; requested=$(length(jobs)); remaining=$(length(remaining)); reusable=$reused; new=$(length(remaining)-reused)")
        println(io,"All theta=Inf; each (pattern,L) limit is evolved once, then each beta is applied independently to that immutable limit. Classification uses its premeasurement energy, abs(excess)<1e-8. Measured energy is not used to decide whether the unmeasured limit is a global ground state.")
        println(io,"Common L=18,36,54,72,90; GS candidates: m1=12:6:90,m2=6:6:90,m3=9:9:90; blocked controls=6:6:54,72,90. beta=0,.25,.5,.75,1,1.5,2,3,4. Historical beta8 preserved as audit only.")
        println(io,"Columns=$(join(EQ_COLUMNS,','))")
        for file in ("fermion_experiment.jl","experiment_patterns.jl","equilibrium_experiments.jl")
            println(io,"SHA256 $file $(bytes2hex(sha256(read(joinpath(@__DIR__,file)))))")
        end
    end
    active = nothing
    state = nothing
    base = nothing
    fingerprint = ""
    ground = false
    rowcount = 0
    started = time()
    open(output,"a") do io
        for job in remaining
            pattern,L,beta = job.pattern,job.L,job.beta
            group = (pattern.name,L)
            if group != active
                if state !== nothing
                    eq_fingerprint(state) == fingerprint || error("measurement mutated the initial limit")
                end
                model = FE.BlochModel(L,pattern.m)
                prepared = FE.prepare_pattern(model,pattern.occupied)
                state = FE.evolve(prepared,Inf)
                base = FE.diagnostics(state)
                base.residual < 1e-9 || error("theta=Inf state is not an eigenstate")
                base.excess_energy >= -1e-8 || error("energy is below computed ground energy")
                ground = abs(base.excess_energy) < 1e-8
                fingerprint = eq_fingerprint(state)
                active = group
                GC.gc(false)
            end
            key = (pattern.name,L,beta)
            if haskey(historical,key)
                oldrow,sourcerow = historical[key]
                abs(parse(Float64,oldrow[14])-base.accessible_excess) < 1e-7 ||
                    error("historical accessible energy disagrees at $key")
                row = oldrow
                source,sourcefile,sourcehash,seconds = "reused",basename(input),input_hash,0.0
            else
                calculation_start = time()
                measured = FE.measure(state,beta)
                sa = FE.entropy_strip(measured)
                sc = FE.entropy_strip(measured;orientation=:y)
                sb = FE.entropy_rectangle(measured)
                d = FE.diagnostics(measured)
                row = string.((pattern.name,pattern.family,pattern.m,L,Inf,beta,sa,sc,sb,
                    sa-sb,(2sa+sc)/3-sb,d.energy,d.excess_energy,d.accessible_excess,d.residual,
                    d.min_initial_sv,d.log_initial_overlap2,d.empty_occupancy,d.projector_error,
                    d.reconstruction_error,d.min_pivot_sv))
                source,sourcefile,sourcehash,sourcerow = "computed",basename(@__FILE__),code_hash,0
                seconds = time()-calculation_start
            end
            extra = string.((ground,base.energy,base.excess_energy,base.residual,
                state.prepared.model.ground_energy,fingerprint,source,sourcefile,sourcehash,
                sourcerow,seconds,job.stage,EQ_VERSION))
            println(io,join(vcat(collect(row),collect(extra)),',')); flush(io)
            rowcount += 1
            @printf("%d/%d %s L=%d beta=%g ground=%s source=%s D=%s seconds=%.3f\n",
                rowcount,length(remaining),pattern.name,L,beta,string(ground),source,row[11],seconds)
            flush(stdout)
        end
    end
    state === nothing || eq_fingerprint(state) == fingerprint || error("limit mutated")
    final = eq_read_rows(output,EQ_COLUMNS)
    all(j->haskey(final,(j.pattern.name,j.L,j.beta)),jobs) || error("scan coverage incomplete")
    open(manifest,"a") do io
        println(io,"FINISH UTC=$(now(Dates.UTC)); rows_written=$rowcount; total_rows=$(length(final)); elapsed_seconds=$(time()-started); OUTPUT_SHA256=$(bytes2hex(sha256(read(output))))")
    end
    println("Complete: $(length(jobs)) requested rows covered; $(length(final)) total rows in $output")
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && eq_main()
