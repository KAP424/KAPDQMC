using LinearAlgebra, Dates, SHA, Printf
include(joinpath(@__DIR__, "fermion_experiment.jl"))
import .FermionExperiment as FE
include(joinpath(@__DIR__, "experiment_patterns.jl"))

function argument(name, default)
    prefix = "--$name="
    index = findfirst(x -> startswith(x, prefix), ARGS)
    return isnothing(index) ? default : ARGS[index][length(prefix)+1:end]
end

const DATA_DIR = joinpath(@__DIR__, "data")
const OBS_COLUMNS = ["pattern", "family", "m", "L", "theta", "beta", "S_A", "S_C", "S_B",
    "delta", "D", "energy", "excess_energy", "accessible_excess", "residual", "min_initial_sv",
    "log_initial_overlap2", "empty_occupancy", "projector_error", "reconstruction_error", "min_pivot_sv"]

function read_completed(path)
    complete = Set{Tuple{String,Int,Float64,Float64}}()
    if isfile(path)
        lines = readlines(path)
        split(first(lines), ',') == OBS_COLUMNS || error("existing output has a different schema")
        for line in lines[2:end]
            row = split(line, ',')
            length(row) == length(OBS_COLUMNS) || error("incomplete CSV row in $path")
            push!(complete, (row[1], parse(Int, row[4]), parse(Float64, row[5]), parse(Float64, row[6])))
        end
    end
    return complete
end

function observe(io, complete, pattern, state, beta)
    L = state.prepared.model.L
    key = (pattern.name, L, state.theta, Float64(beta))
    key in complete && return
    measured = FE.measure(state, beta)
    started = time()
    if state.theta == 0
        SA = SC = SB = 0.0
    else
        SA = FE.entropy_strip(measured; orientation=:x)
        SC = FE.entropy_strip(measured; orientation=:y)
        SB = FE.entropy_rectangle(measured)
    end
    d = FE.diagnostics(measured)
    row = (pattern.name, pattern.family, pattern.m, L, state.theta, beta,
        SA, SC, SB, SA-SB, (2SA+SC)/3-SB, d.energy, d.excess_energy,
        d.accessible_excess, d.residual, d.min_initial_sv, d.log_initial_overlap2,
        d.empty_occupancy, d.projector_error, d.reconstruction_error, d.min_pivot_sv)
    println(io, join(row, ',')); flush(io)
    push!(complete, key)
    @printf("%s L=%d theta=%g beta=%g D=%.8g S_A=%.8g elapsed=%.2fs\n",
        pattern.name, L, state.theta, beta, (2SA+SC)/3-SB, SA, time()-started)
    flush(stdout)
end

function sizes(pattern, stage)
    if stage == "equilibrium"
        pattern.name == "uniform_A" && return [12,18,24,30,36,42,48,54,60,72,90]
        pattern.name == "uniform_B" && return [12,24,36]
        return pattern.m == 2 ? collect(6:6:54) : collect(9:9:54)
    elseif stage == "nonequilibrium"
        pattern.name == "uniform_A" && return [18,30,42,54,72]
        return [18,36,54]
    else
        return [6pattern.m]
    end
end

function main()
    stage = argument("stage", "all")
    stage in ("all", "pilot", "equilibrium", "nonequilibrium", "convergence") || error("unknown stage")
    threads = parse(Int, argument("threads", "2"))
    BLAS.set_num_threads(threads)
    maxL = parse(Int, argument("max-L", "1000"))
    custom_sizes = argument("Ls", "")
    custom_thetas = argument("thetas", "")
    custom_betas = argument("betas", "")
    requested = split(argument("patterns", ""), ',')
    patterns = filter(p -> requested == [""] || p.name in requested, experiment_patterns())
    isempty(patterns) && error("no matching patterns")
    mkpath(DATA_DIR)
    write_pattern_catalogue(joinpath(DATA_DIR, "patterns.csv"))
    output = argument("output", joinpath(DATA_DIR, "observations.csv"))
    complete = read_completed(output)
    isfile(output) || open(io -> println(io, join(OBS_COLUMNS, ',')), output, "w")
    models = Dict{Tuple{Int,Int},FE.BlochModel}()
    model(L,m) = get!(models, (L,m)) do
        FE.BlochModel(L,m)
    end
    stages = stage == "all" ? ["equilibrium", "nonequilibrium", "convergence"] : [stage]
    noneq_names = ["uniform_A", "stripe_AB_x", "full_empty_checker", "two_B_3", "random_half_3"]
    theta_grid = [0.,0.25,0.5,0.75,1.,1.5,2.,2.5,3.,3.5,4.,4.5,5.,5.5,6.,8.,12.,16.]
    open(output, "a") do io
        for current_stage in stages
            current_stage == "convergence" && continue
            for pattern in patterns
                current_stage == "nonequilibrium" && !(pattern.name in noneq_names) && continue
                selected_sizes = isempty(custom_sizes) ? sizes(pattern,current_stage) : parse.(Int,split(custom_sizes,','))
                selected_sizes = filter(<=(maxL), selected_sizes)
                all(L -> L % (3pattern.m) == 0, selected_sizes) || error("L must be a multiple of 3m for matching cut phases")
                for L in selected_sizes
                    theta_values = current_stage == "equilibrium" ? [Inf] :
                        current_stage == "pilot" ? [0.,1.,Inf] : theta_grid
                    betas = current_stage == "equilibrium" ? [0.,0.25,0.5,1.,2.,4.] :
                        current_stage == "pilot" ? [0.,1.,8.] : [0.,0.5,1.,2.]
                    current_stage == "equilibrium" && L == maximum(selected_sizes) &&
                        (betas = vcat(betas,8.))
                    isempty(custom_thetas) || (theta_values = parse.(Float64,split(custom_thetas,',')))
                    isempty(custom_betas) || (betas = parse.(Float64,split(custom_betas,',')))
                    all((pattern.name,L,theta,beta) in complete for theta in theta_values for beta in betas) && continue
                    prepared = FE.prepare_pattern(model(L,pattern.m), pattern.occupied)
                    for theta in theta_values
                        all((pattern.name,L,theta,beta) in complete for beta in betas) && continue
                        state = FE.evolve(prepared, theta)
                        for beta in betas
                            observe(io,complete,pattern,state,beta)
                        end
                    end
                    GC.gc()
                end
            end
        end
    end

    if "convergence" in stages
        path = joinpath(DATA_DIR, "convergence.csv")
        open(path,"w") do io
            println(io,"pattern,m,L,theta,energy,excess_energy,residual,distance_to_limit,min_initial_sv,reconstruction_error")
            for pattern in patterns, L in filter(<=(maxL), [18,36,54])
                prepared = FE.prepare_pattern(model(L,pattern.m),pattern.occupied)
                limit = FE.evolve(prepared,Inf)
                for theta in [0.,0.25,0.5,1.,2.,4.,8.,16.,32.,64.,128.,256.,Inf]
                    state = FE.evolve(prepared,theta)
                    d = FE.diagnostics(state)
                    println(io,join((pattern.name,pattern.m,L,theta,d.energy,d.excess_energy,
                        d.residual,FE.projector_distance(state,limit;normalized=false),
                        d.min_initial_sv,d.reconstruction_error),','))
                end
            end
        end
    end

    open(joinpath(DATA_DIR,"run_manifest.txt"),"a") do io
        println(io,"UTC=$(Dates.now(Dates.UTC)); Julia=$(VERSION); BLAS_threads=$threads; args=$(join(ARGS,' ')); observations=$(length(complete))")
        for filename in ("fermion_experiment.jl","experiment_patterns.jl","run_experiments.jl","neq.md")
            println(io,"SHA256 $filename $(bytes2hex(sha256(read(joinpath(@__DIR__,filename)))))")
        end
    end
    println("Completed stage=$stage; total observations in $output: $(length(complete))")
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
