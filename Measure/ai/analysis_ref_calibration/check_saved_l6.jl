# Small independent comparison against printed notebook values, no large matrices.
using LinearAlgebra, Printf
include(joinpath(@__DIR__, "..", "fermion_experiment.jl"))
using .FermionExperiment
BLAS.set_num_threads(1)
pre = prepare_pattern(BlochModel(6, 1), Bool[true, false])
open(joinpath(@__DIR__, "current_l6_time_values.csv"), "w") do io
    println(io, "L,theta,S_A,S_C,S_B,D,twice_D")
    for j in 1:60
        theta = j / 10
        state = evolve(pre, theta)
        sa = entropy_strip(state)
        sc = entropy_strip(state; orientation=:y)
        sb = entropy_rectangle(state)
        d = (2sa + sc)/3 - sb
        @printf(io, "6,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n", theta,sa,sc,sb,d,2d)
    end
end
