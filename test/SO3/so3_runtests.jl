push!(LOAD_PATH, "D:\\JuliaDQMC\\code\\KAPDQMC\\src\\")
using KAPDQMC
using Test
using Random
using LinearAlgebra

@testset "KAPDQMC.jl" begin
    path = "D:\\JuliaDQMC\\code\\KAPDQMC\\test\\SO3\\"
    rng = MersenneTwister(time_ns())

    Lattice = "HoneyComb120"
    site = [6, 6]

    model = SO3_Hubbard_Para(Ht=1.0, HJ1=1.0, HJ2=1.0,
        Θrelax=1.5, Θquench=0.0, Lattice=Lattice, site=site, Δt=0.1, BatchSize=5, Initial="H0")
    # @assert norm(model.α) < 1e-10 "α should be zero when HJ1 and HJ2 are zero!"

    s = Initial_s(model, rng)

    s = phy_update(path, model, s, 4, true)


    L = model.site[2]
    indexA = area_index(model.Lattice, model.site, ([1, 1], [div(L, 3), L]))
    # # HalfHalf
    indexB = area_index(model.Lattice, model.site, ([1, 1], [div(L, 3), div(2 * L, 3)]))
    # println(indexB)
    ss = [copy(s), copy(s)]
    λ = 0.5
    Nλ = 2
    ss = ctrl_SCEEicr(path, model, indexA, indexB, 2, λ, Nλ, ss, true)

    # # s = ctrl_SCDOPicr(path, model, π / 2, indexA, indexB, 20, λ, Nλ, s, true)

    # # println(@btime ctrl_SCEEicr($path,$model,$indexA,$indexB,$Sweeps,$λ,$Nλ,$ss,$true) )

    # ------------------------------------------------------------------------------------------------------------------------------------------------------


end
