push!(LOAD_PATH, "D:\\JuliaDQMC\\code\\KAPDQMC\\src\\")
using KAPDQMC
using Test
using Random
using LinearAlgebra

@testset "KAPDQMC.jl" begin
    path = "test/tU_Measure/"

    rng = MersenneTwister(1234)

    model = tU_Hubbard_Para(Ht=1.0, Hu1=3.8, Hu2=3.8, Θrelax=3.1, Θquench=0., Lattice="HoneyComb120", site=[3, 3], Δt=0.03, BatchSize=10, Initial="V")
    s = Initial_s(model, rng)

    beta = 0.
    Eidx = collect(1:2:model.Ns)

    s = phy_update_M(path, model, Eidx, beta, s, 10, true)
end