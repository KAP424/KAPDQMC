using LinearAlgebra, Printf, Dates, SHA
include(joinpath(@__DIR__,"fermion_experiment.jl"))
import .FermionExperiment as FE
include(joinpath(@__DIR__,"experiment_patterns.jl"))

BLAS.set_num_threads(1)
outdir = joinpath(@__DIR__,"data")
mkpath(outdir)
csvpath = joinpath(outdir,"numerical_sensitivity.csv")
logpath = joinpath(outdir,"numerical_sensitivity.log")

function entropies(state)
    sa = FE.entropy_strip(state)
    sc = FE.entropy_strip(state;orientation=:y)
    sb = FE.entropy_rectangle(state)
    return (sa,sc,sb,(2sa+sc)/3-sb)
end

function run_sensitivity()
    max_tol_distance = 0.0
    max_tol_entropy = 0.0
    max_reconstruction = 0.0
    min_pivot = Inf
    max_late_distance = 0.0
    max_late_entropy = 0.0
    worst_late = ""
    rows = 0
    started = time()
    open(csvpath,"w") do io
        println(io,"pattern,m,L,theta,beta,tol,projector_distance_to_tol1e10,SA,SC,SB,D,max_entropy_difference_to_tol1e10,projector_distance_to_limit,max_entropy_difference_to_limit,reconstruction_error,min_pivot_sv,min_initial_sv,excess_energy,residual,original_occupied_rows_preserved")
        for pattern in experiment_patterns(), L in (18,36)
            model = FE.BlochModel(L,pattern.m)
            pref = FE.prepare_pattern(model,pattern.occupied;tol=1e-10)
            reference_states = Dict()
            reference_entropies = Dict()
            for theta in (6.,256.,Inf), beta in (0.,1.,8.)
                state = FE.evolve(pref,theta;beta=beta)
                reference_states[(theta,beta)] = state
                reference_entropies[(theta,beta)] = entropies(state)
            end
            for tol in (1e-8,1e-10,1e-12)
                prepared = FE.prepare_pattern(model,pattern.occupied;tol=tol)
                rowset_preserved = prepared.F == findall(pattern.occupied) &&
                    prepared.E == findall(.!pattern.occupied)
                @assert rowset_preserved
                for theta in (6.,256.,Inf), beta in (0.,1.,8.)
                    state = FE.evolve(prepared,theta;beta=beta)
                    e = entropies(state)
                    dr = FE.projector_distance(state,reference_states[(theta,beta)])
                    de = maximum(abs.(e .- reference_entropies[(theta,beta)]))
                    dl = FE.projector_distance(state,reference_states[(Inf,beta)])
                    el = maximum(abs.(e .- reference_entropies[(Inf,beta)]))
                    d = FE.diagnostics(state)
                    println(io,join((pattern.name,pattern.m,L,theta,beta,tol,dr,e...,
                        de,dl,el,d.reconstruction_error,d.min_pivot_sv,d.min_initial_sv,
                        d.excess_energy,d.residual,rowset_preserved),','))
                    rows += 1
                    max_tol_distance = max(max_tol_distance,dr)
                    max_tol_entropy = max(max_tol_entropy,de)
                    max_reconstruction = max(max_reconstruction,d.reconstruction_error)
                    min_pivot = min(min_pivot,d.min_pivot_sv)
                    if theta == 256.
                        if dl > max_late_distance
                            max_late_distance = dl
                            worst_late = "$(pattern.name), L=$L, beta=$beta, tol=$tol"
                        end
                        max_late_entropy = max(max_late_entropy,el)
                    end
                end
            end
            flush(io)
            println("sensitivity complete: $(pattern.name) L=$L")
            flush(stdout)
        end
    end
    open(logpath,"w") do io
        println(io,"UTC=$(Dates.now(Dates.UTC)); Julia=$VERSION; BLAS_threads=$(BLAS.get_num_threads())")
        println(io,"Patterns: all 13 actual experiment patterns. L=[18,36]; theta=[6,256,Inf]; beta=[0,1,8]; tol=[1e-8,1e-10,1e-12].")
        println(io,"Reference tolerance=1e-10. Projector distances use Frobenius norm divided by sqrt(Npart). All entropies are single-species S2 with natural logarithm.")
        println(io,"Rows=$rows; runtime_seconds=$(time()-started)")
        println(io,"maximum_projector_change_with_tolerance=$max_tol_distance")
        println(io,"maximum_entropy_or_D_change_with_tolerance=$max_tol_entropy")
        println(io,"maximum_initial_orbital_reconstruction_error=$max_reconstruction")
        println(io,"minimum_retained_pivot_singular_value=$min_pivot")
        println(io,"maximum_theta256_vs_infinite_projector_distance=$max_late_distance")
        println(io,"maximum_theta256_vs_infinite_entropy_or_D_difference=$max_late_entropy")
        println(io,"worst_theta256_projector_case=$worst_late")
        println(io,"Original occupied row sets F and complementary E verified unchanged in every preparation.")
        println(io,"Boundary cancellation: A=(L/3)xL, C=Lx(L/3), B=(L/3)x(2L/3). For local boundary coefficients ax,ay, SA=2axL, SC=2ayL, SB=4axL/3+2ayL/3, hence (2SA+SC)/3-SB cancels them. Origin x0=y0=1 and L divisible by3m ensure matching microscopic phases and integer periods along each side. Corners and nonlocal shape contributions remain. For LlogL states this cancellation alone does not establish a pure corner-log form.")
        for file in ("fermion_experiment.jl","experiment_patterns.jl","sensitivity_checks.jl")
            println(io,"SHA256 $file $(bytes2hex(sha256(read(joinpath(@__DIR__,file)))))")
        end
    end
    println(read(logpath,String))
    @assert max_tol_distance < 1e-9
    @assert max_tol_entropy < 1e-8
    @assert max_reconstruction < 1e-10
end

run_sensitivity()
