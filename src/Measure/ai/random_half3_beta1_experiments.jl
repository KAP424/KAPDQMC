#=
Targeted, resumable random_half_3 / beta=1 scan. Old data and kernels are read-only.
Run --Ls=90,108,144 --thetas=2,3,4,5,6,7,8,10,12,14,16,18,20,24,28,32
or --Ls=99,108,117,126,135,144,153,162,171,180 --thetas=Inf.
=#
using LinearAlgebra, Dates, SHA, Printf, TOML
include(joinpath(@__DIR__, "fermion_experiment.jl"))
import .FermionExperiment as FE
include(joinpath(@__DIR__, "real_entropy_extension.jl"))

const RH_COLUMNS = ["pattern", "family", "m", "L", "theta", "beta", "S_A", "S_C", "S_B",
    "delta", "D", "energy", "excess_energy", "accessible_excess", "residual", "min_initial_sv",
    "log_initial_overlap2", "empty_occupancy", "projector_error", "reconstruction_error", "min_pivot_sv",
    "premeasurement_ground", "premeasurement_limit_excess", "premeasurement_limit_residual",
    "rectangle_imag_error", "calculation_seconds", "rectangle_seconds", "rank_tol", "run_id", "code_sha256",
    "entropy_method", "rectangle_real_error_bound"]
const RH_BITS = "100110001011010110"

function rh_arg(name, default)
    prefix = "--$name="
    index = findfirst(a -> startswith(a,prefix),ARGS)
    return isnothing(index) ? default : ARGS[index][length(prefix)+1:end]
end
rh_hash(p) = bytes2hex(sha256(read(p)))

function rh_rectangle(state; method=:cholesky)
    entropy,checks = RealEntropyExtension.entropy_rectangle_real(state;method,with_checks=true)
    return (entropy=entropy,imag_error=checks.max_imaginary,error_bound=checks.real_part_entropy_bound)
end

function rh_existing(path)
    rows = Dict{Tuple{Int,Float64},Vector{String}}()
    isfile(path) || return rows
    raw = read(path,String)
    endswith(raw,"\n") || error("Unterminated output row; preserve and inspect before resuming: $path")
    lines = split(chomp(raw),'\n')
    split(strip(lines[1]),',') == RH_COLUMNS || error("output schema mismatch")
    for line in lines[2:end]
        fields = String.(split(strip(line),','))
        length(fields) == length(RH_COLUMNS) || error("malformed output row")
        fields[1] == "random_half_3" && parse(Float64,fields[6]) == 1 || error("wrong protocol in output")
        key = (parse(Int,fields[4]),parse(Float64,fields[5]))
        haskey(rows,key) && error("duplicate saved key $key")
        rows[key] = fields
    end
    return rows
end

function main()
    sizes = sort(unique(parse.(Int,split(rh_arg("Ls","90,108,144"),','))))
    thetas = sort(unique(parse.(Float64,split(rh_arg("thetas","2,3,4,5,6,7,8,10,12,14,16,18,20,24,28,32"),','))))
    all(L -> L>0 && L%9==0,sizes) || error("L must be positive and divisible by 9")
    all(t -> t>=0 && !isnan(t),thetas) || error("invalid theta")
    threads = parse(Int,rh_arg("threads","4"))
    threads > 0 || error("invalid BLAS thread count")
    BLAS.set_num_threads(threads)
    tol = parse(Float64,rh_arg("tol","1e-10"))
    entropy_method = Symbol(rh_arg("entropy-method","cholesky"))
    entropy_method in (:cholesky,:eigen) || error("invalid entropy method")
    output = abspath(rh_arg("output",joinpath(@__DIR__,"data","random_half3_beta1_observations.csv")))
    mode = rh_arg("mode","run")
    mode in ("run","plan") || error("invalid mode")
    saved = rh_existing(output)
    all(parse(Float64,row[28])==tol for row in values(saved)) || error("rank tolerance differs from saved rows")
    jobs = [(L,t) for L in sizes for t in thetas if !haskey(saved,(L,t))]
    println("random_half_3 beta=1: $(length(jobs)) new rows; $(length(saved)) already present")
    mode == "plan" && return
    mkpath(dirname(output))
    isfile(output) || write(output,join(RH_COLUMNS,',')*"\n")
    runid = Dates.format(now(Dates.UTC),"yyyymmddTHHMMSSsss")
    manifestpath = joinpath(dirname(output),"random_half3_beta1_run_$runid.toml")
    sources = ["fermion_experiment.jl","real_entropy_extension.jl","random_half3_beta1_experiments.jl"]
    hashes = Dict(f=>rh_hash(joinpath(@__DIR__,f)) for f in sources)
    manifest = Dict{String,Any}("run_id"=>runid,"julia"=>string(VERSION),"blas_threads"=>threads,
        "started_utc"=>string(now(Dates.UTC)),"pattern_bits"=>RH_BITS,"beta"=>1.0,"rank_tol"=>tol,
        "L"=>sizes,"theta"=>string.(thetas),"new_rows"=>length(jobs),"output"=>output,
        "code_sha256"=>hashes,"status"=>"running","entropy_method"=>string(entropy_method),
        "protocol"=>"single-species natural-log Renyi2; finite theta is never replaced by the ground limit; A=L/3*L, C=L*L/3, B=L/3*2L/3; origin=(1,1)")
    open(io->TOML.print(io,manifest),manifestpath,"w")
    occupied = BitVector([c=='1' for c in RH_BITS])
    previousL = 0
    prepared = nothing
    limitdiag = nothing
    started = time()
    open(output,"a") do io
        for (i,(L,theta)) in enumerate(jobs)
            if L != previousL
                prepared = nothing
                GC.gc(true)
                prepared = FE.prepare_pattern(FE.BlochModel(L,3),occupied;tol=tol)
                limitdiag = FE.diagnostics(FE.evolve(prepared,Inf))
                abs(limitdiag.excess_energy)<1e-8 || error("ground-state projection failed at L=$L")
                limitdiag.residual<1e-9 || error("nonstationary limit at L=$L")
                previousL = L
                @printf("L=%d ground_excess=%.4g residual=%.4g pivot_min=%.4g\n",
                    L,limitdiag.excess_energy,limitdiag.residual,prepared.min_pivot_sv)
                flush(stdout)
            end
            rowstart = time()
            state = FE.evolve(prepared,theta;beta=1.0)
            sa = FE.entropy_strip(state)
            sc = FE.entropy_strip(state;orientation=:y)
            rectstart = time()
            rect = rh_rectangle(state;method=entropy_method)
            sb = rect.entropy
            rectseconds = time()-rectstart
            d = FE.diagnostics(state)
            d.projector_error<1e-10 || error("nonorthogonal measured orbitals")
            d.reconstruction_error<1e-9 || error("initial-state reconstruction failed")
            row = ("random_half_3","general",3,L,theta,1.0,sa,sc,sb,sa-sb,(2sa+sc)/3-sb,
                d.energy,d.excess_energy,d.accessible_excess,d.residual,d.min_initial_sv,
                d.log_initial_overlap2,d.empty_occupancy,d.projector_error,d.reconstruction_error,d.min_pivot_sv,
                true,limitdiag.excess_energy,limitdiag.residual,rect.imag_error,time()-rowstart,rectseconds,
                tol,runid,hashes["random_half3_beta1_experiments.jl"],string(entropy_method),rect.error_bound)
            println(io,join(row,',')); flush(io)
            @printf("%d/%d L=%d theta=%g D=%.12g seconds=%.3f rectangle=%.3f\n",
                i,length(jobs),L,theta,(2sa+sc)/3-sb,time()-rowstart,rectseconds)
            flush(stdout)
            GC.gc(true)
        end
    end
    final = rh_existing(output)
    all(haskey(final,(L,t)) for L in sizes for t in thetas) || error("requested grid incomplete")
    manifest["status"] = "complete"
    manifest["finished_utc"] = string(now(Dates.UTC))
    manifest["elapsed_seconds"] = time()-started
    manifest["output_sha256_at_completion"] = rh_hash(output)
    manifest["total_rows_at_completion"] = length(final)
    open(io->TOML.print(io,manifest),manifestpath,"w")
    println("Completed targeted scan; total saved rows=$(length(final))")
end

abspath(PROGRAM_FILE)==abspath(@__FILE__) && main()
