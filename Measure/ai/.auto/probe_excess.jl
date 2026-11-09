# Scratch probe: is the theta=inf accessible state the ground state at each L?
using LinearAlgebra
include(joinpath(@__DIR__, "..", "fermion_experiment.jl"))
const FE = FermionExperiment
include(joinpath(@__DIR__, "..", "experiment_patterns.jl"))

catalogue = Dict(p.name => p for p in experiment_patterns())
for name in ("one_B_2", "one_B_3", "uniform_A")
    p = catalogue[name]
    println("== ", name, "  (m=", p.m, ")")
    for L in (12, 18, 24, 36, 48, 60, 72, 90, 108, 126, 144, 180, 216)
        L % p.m == 0 || continue
        model = FE.BlochModel(L, p.m)
        prep = FE.prepare_pattern(model, p.occupied)
        d = FE.diagnostics(FE.evolve(prep, Inf))
        println("  L=", L, "  excess=", d.excess_energy, "  min_pivot_sv=", d.min_pivot_sv,
            "  min_initial_sv=", d.min_initial_sv)
    end
end
