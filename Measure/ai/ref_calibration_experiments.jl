# Additional independent size-convergence data for the Ref-based reanalysis.
# Existing source files and the validated evolution / entropy kernels are read-only.
using LinearAlgebra, Dates, SHA, Printf, TOML
include(joinpath(@__DIR__, "fermion_experiment.jl"))
import .FermionExperiment as FE
include(joinpath(@__DIR__, "real_entropy_extension.jl"))
include(joinpath(@__DIR__, "experiment_patterns.jl"))
include(joinpath(@__DIR__, "arrangement_patterns.jl"))

const RC_NAMES = ["uniform_A", "uniform_B", "one_B_2", "one_B_3", "two_B_3",
    "diagonal_B_3", "special_matched_a", "special_matched_b", "full_empty_cluster3",
    "random_half_3", "general_matched_a", "general_matched_b"]
const RC_COLUMNS = ["pattern", "family", "m", "L", "theta", "beta", "S_A", "S_C", "S_B",
    "delta", "D", "energy", "excess_energy", "accessible_excess", "residual", "min_initial_sv",
    "log_initial_overlap2", "empty_occupancy", "projector_error", "reconstruction_error", "min_pivot_sv",
    "premeasurement_ground", "premeasurement_excess", "premeasurement_residual", "occupied_bits",
    "rectangle_imag_error", "rectangle_real_error_bound", "seconds", "run_id", "code_sha256"]

function rc_arg(name, default)
    prefix = "--$name="
    i = findfirst(a -> startswith(a, prefix), ARGS)
    return isnothing(i) ? default : ARGS[i][length(prefix)+1:end]
end
rc_hash(p) = bytes2hex(sha256(read(p)))

function rc_rows(path)
    isfile(path) || return Dict[]
    raw = read(path, String)
    endswith(raw, "\n") || error("incomplete file; inspect before resuming: $path")
    lines = split(chomp(raw), '\n')
    cols = String.(split(strip(lines[1]), ','))
    result = Dict[]
    for line in lines[2:end]
        values = String.(split(strip(line), ','))
        length(values) == length(cols) || error("invalid row in $path")
        push!(result, Dict(zip(cols, values)))
    end
    return result
end
rc_key(r) = (r["pattern"], parse(Int,r["L"]), parse(Float64,r["theta"]), parse(Float64,r["beta"]))

function main()
    names = split(rc_arg("patterns", join(RC_NAMES, ',')), ',')
    catalogue = vcat(experiment_patterns(), matched_arrangement_patterns())
    all(p -> p in RC_NAMES, names) || error("unknown ground-accessible candidate")
    patterns = [only(filter(p -> p.name == name, catalogue)) for name in names]
    sizes = sort(unique(parse.(Int, split(rc_arg("Ls", "90,108"), ','))))
    thetas = sort(unique(parse.(Float64, split(rc_arg("thetas", "2,2.5,3,3.5,4,4.5,5,5.5,6"), ','))))
    betas = sort(unique(parse.(Float64, split(rc_arg("betas", "0"), ','))))
    all(L -> all(p -> L % (3p.m) == 0, patterns), sizes) || error("sizes must preserve cut phase")
    all(t -> t >= 0 && !isnan(t), thetas) || error("invalid theta")
    all(b -> isfinite(b) && b >= 0, betas) || error("invalid beta")
    BLAS.set_num_threads(parse(Int, rc_arg("threads", "8")))
    output = abspath(rc_arg("output", joinpath(@__DIR__, "data", "ref_calibration_observations.csv")))
    baselinepath = joinpath(@__DIR__, "data", "ground_observations_baseline.csv")
    skipold = rc_arg("recompute", "false") != "true"
    known = Set(rc_key(r) for r in rc_rows(output))
    if skipold
        union!(known, Set(rc_key(r) for r in rc_rows(baselinepath)))
    end
    jobs = [(p, L, t, b) for L in sizes for p in patterns for t in thetas for b in betas
        if !((p.name, L, t, b) in known)]
    println("Ref calibration: $(length(jobs)) new jobs, output=$output")
    rc_arg("mode", "run") == "plan" && return
    mkpath(dirname(output))
    isfile(output) || write(output, join(RC_COLUMNS, ',') * "\n")
    runid = Dates.format(now(Dates.UTC), "yyyymmddTHHMMSSsss")
    files = ["ref_calibration_experiments.jl", "fermion_experiment.jl", "real_entropy_extension.jl",
        "experiment_patterns.jl", "arrangement_patterns.jl"]
    hashes = Dict(f => rc_hash(joinpath(@__DIR__, f)) for f in files)
    manifest = Dict{String,Any}("run_id"=>runid, "status"=>"running", "started_utc"=>string(now(Dates.UTC)),
        "julia"=>string(VERSION), "blas_threads"=>BLAS.get_num_threads(), "code_sha256"=>hashes,
        "patterns"=>names, "L"=>sizes, "theta"=>string.(thetas), "beta"=>betas,
        "new_jobs"=>length(jobs), "baseline_path"=>baselinepath,
        "baseline_sha256"=>rc_hash(baselinepath), "output"=>output,
        "protocol"=>"one species; natural-log Renyi2; same cut origin and D; exact finite theta; own empty-site measurement",
        "selection"=>"size convergence and window sensitivity, never closeness to equilibrium target")
    manifestpath = joinpath(dirname(output), "ref_calibration_run_$runid.toml")
    open(io -> TOML.print(io, manifest), manifestpath, "w")
    active = nothing
    prepared = nothing
    limitdiag = nothing
    state = nothing
    active_theta = nothing
    started = time()
    open(output, "a") do io
        for (i, (p, L, theta, beta)) in enumerate(jobs)
            if active != (p.name, L)
                prepared = nothing; state = nothing; GC.gc(true)
                prepared = FE.prepare_pattern(FE.BlochModel(L, p.m), p.occupied)
                limitdiag = FE.diagnostics(FE.evolve(prepared, Inf))
                abs(limitdiag.excess_energy) < 1e-8 || error("ground-state access failed")
                limitdiag.residual < 1e-9 || error("limit is not stationary")
                active = (p.name, L); active_theta = nothing
            end
            start = time()
            if active_theta != theta
                state = FE.evolve(prepared, theta)
                active_theta = theta
            end
            measured = FE.measure(state, beta)
            sa = FE.entropy_strip(measured)
            sc = FE.entropy_strip(measured; orientation=:y)
            sb, checks = RealEntropyExtension.entropy_rectangle_real(measured; with_checks=true)
            d = FE.diagnostics(measured)
            d.projector_error < 1e-10 || error("lost orbital orthogonality")
            d.reconstruction_error < 1e-9 || error("initial state reconstruction failed")
            row = (p.name, p.family, p.m, L, theta, beta, sa, sc, sb, sa-sb, (2sa+sc)/3-sb,
                d.energy, d.excess_energy, d.accessible_excess, d.residual, d.min_initial_sv,
                d.log_initial_overlap2, d.empty_occupancy, d.projector_error, d.reconstruction_error,
                d.min_pivot_sv, true, limitdiag.excess_energy, limitdiag.residual, join(Int.(p.occupied)),
                checks.max_imaginary, checks.real_part_entropy_bound, time()-start, runid,
                hashes["ref_calibration_experiments.jl"])
            println(io, join(row, ',')); flush(io)
            @printf("%d/%d %s L=%d theta=%g beta=%g D=%.12g seconds=%.2f\n",
                i, length(jobs), p.name, L, theta, beta, row[11], time()-start)
            flush(stdout)
            GC.gc(false)
        end
    end
    manifest["status"] = "complete"
    manifest["finished_utc"] = string(now(Dates.UTC))
    manifest["elapsed_seconds"] = time()-started
    manifest["output_sha256_at_completion"] = rc_hash(output)
    open(io -> TOML.print(io, manifest), manifestpath, "w")
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
