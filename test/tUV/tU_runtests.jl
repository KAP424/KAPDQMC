push!(LOAD_PATH, "C:/Users/admin/Desktop/JuliaDQMC/code/KAPDQMC/src/")
using KAPDQMC
using Test
using Random
using LinearAlgebra

@testset "KAPDQMC.jl" begin
    path = "test/tUV/"

    rng = MersenneTwister(1234)

    model = tUV_Hubbard_Para(Ht=1.0, Hu=4.0, Hv=0.0, Θ=1.0, Lattice="HoneyComb120", site=[3, 3], Δt=0.1, BatchSize=10, Initial="H0")

    s = Initial_s(model, rng)

    s = phy_update(path, model, s, 10, true)


    L = model.site[2]
    indexA = area_index(model.Lattice, model.site, ([1, 1], [div(L, 3), L]))
    # # HalfHalf
    indexB = area_index(model.Lattice, model.site, ([1, 1], [div(L, 3), div(2 * L, 3)]))
    # println(indexB)

    λ = 0.5
    Nλ = 2

    ss = [copy(s), copy(s)]

    # println(@btime ctrl_SCEEicr($path,$model,$indexA,$indexB,$Sweeps,$λ,$Nλ,$ss,$true) )
    ss = ctrl_SCEEicr(path, model, indexA, indexB, 2, λ, Nλ, ss, true)

end
