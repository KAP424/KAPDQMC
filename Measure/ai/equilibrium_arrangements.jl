#= Size-scaling controls at identical local occupation counts.

The four patterns were selected at L=9 and checked independently at L=18.
This script tests their size dependence up to L=90, at a fixed cut origin.
Translation-averaged small-size controls are in arrangement_controls.csv.
=#
include(joinpath(@__DIR__, "equilibrium_experiments.jl"))
include(joinpath(@__DIR__, "arrangement_patterns.jl"))

function arrangement_scaling_main()
    BLAS.set_num_threads(parse(Int, eq_argument("threads", "4")))
    patterns = matched_arrangement_patterns()
    sizes = collect(9:9:parse(Int, eq_argument("max-L", "90")))
    betas = [0., .5, 1., 2.]
    path = joinpath(EQ_DATA, "equilibrium_arrangements.csv")
    completed = eq_read_rows(path, EQ_COLUMNS; repair_tail=true)
    jobs = [(p, L, beta) for L in sizes for p in patterns for beta in betas]
    remaining = filter(j -> !haskey(completed, (j[1].name, j[2], j[3])), jobs)
    println("Arrangement grid: $(length(jobs)) rows, $(length(remaining)) remaining")
    isempty(remaining) && return
    mkpath(EQ_DATA)
    isfile(path) || write(path, join(EQ_COLUMNS, ',') * "\n")
    selection_hash = bytes2hex(sha256(read(joinpath(@__DIR__, "arrangement_patterns.jl"))))
    manifest = joinpath(EQ_DATA, "equilibrium_arrangements_manifest.txt")
    if isfile(manifest)
        for line in eachline(manifest)
            startswith(line, "PATTERNS_SHA256=") &&
                line != "PATTERNS_SHA256=$selection_hash" &&
                error("Pattern catalogue changed; do not combine different scans")
        end
    end
    code_hash = bytes2hex(sha256(read(@__FILE__)))
    open(manifest, "a") do io
        println(io, "START UTC=$(now(Dates.UTC)); Julia=$VERSION; BLAS_threads=$(BLAS.get_num_threads()); args=$(join(ARGS,' '))")
        println(io, "PATTERNS_SHA256=$selection_hash")
        println(io, "L=$(join(sizes,':')); beta=$(join(betas,':')); theta=Inf; cut origin=1,1; requested=$(length(jobs))")
        for file in ("fermion_experiment.jl", "equilibrium_experiments.jl",
                "equilibrium_arrangements.jl", "arrangement_patterns.jl")
            println(io, "SHA256 $file $(bytes2hex(sha256(read(joinpath(@__DIR__,file)))))")
        end
    end
    active = nothing
    state = nothing
    base = nothing
    fingerprint = ""
    began = time()
    open(path, "a") do io
        for (i, (pattern, L, beta)) in enumerate(remaining)
            group = (pattern.name, L)
            if group != active
                state === nothing || eq_fingerprint(state) == fingerprint ||
                    error("Measurement changed the unmeasured state")
                model = FE.BlochModel(L, pattern.m)
                state = FE.evolve(FE.prepare_pattern(model, pattern.occupied), Inf)
                base = FE.diagnostics(state)
                base.residual < 1e-9 || error("Limit is not stationary at $group")
                fingerprint = eq_fingerprint(state)
                active = group
                GC.gc(false)
            end
            start = time()
            measured = FE.measure(state, beta)
            sa = FE.entropy_strip(measured)
            sc = FE.entropy_strip(measured; orientation=:y)
            sb = FE.entropy_rectangle(measured)
            d = FE.diagnostics(measured)
            seconds = time() - start
            row = (pattern.name, pattern.family, pattern.m, L, Inf, beta,
                sa, sc, sb, sa-sb, (2sa+sc)/3-sb, d.energy, d.excess_energy,
                d.accessible_excess, d.residual, d.min_initial_sv,
                d.log_initial_overlap2, d.empty_occupancy, d.projector_error,
                d.reconstruction_error, d.min_pivot_sv,
                abs(base.excess_energy)<1e-8, base.energy, base.excess_energy,
                base.residual, state.prepared.model.ground_energy, fingerprint,
                "computed", basename(@__FILE__), code_hash, 0, seconds,
                "arrangement_control", "equilibrium_arrangements_v1")
            println(io, join(row, ',')); flush(io)
            @printf("%d/%d %s L=%d beta=%g ground=%s SB=%.10g D=%.10g seconds=%.3f\n",
                i, length(remaining), pattern.name, L, beta,
                string(abs(base.excess_energy)<1e-8), sb, (2sa+sc)/3-sb, seconds)
            flush(stdout)
        end
    end
    state === nothing || eq_fingerprint(state) == fingerprint || error("Limit changed")
    final = eq_read_rows(path, EQ_COLUMNS)
    all(j -> haskey(final, (j[1].name, j[2], j[3])), jobs) || error("Incomplete grid")
    open(manifest, "a") do io
        println(io, "FINISH UTC=$(now(Dates.UTC)); rows=$(length(final)); elapsed_seconds=$(time()-began); OUTPUT_SHA256=$(bytes2hex(sha256(read(path))))")
    end
    println("Completed arrangement size scan: $(length(final)) rows")
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && arrangement_scaling_main()
