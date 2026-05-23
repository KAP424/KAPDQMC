push!(LOAD_PATH, "D:\\JuliaDQMC\\code\\KAPDQMC\\src\\")
using KAPDQMC
using Test
using Random
using LinearAlgebra

KAPDQMC.nn2idx("triangular90", [4, 4], 5)

@testset "KAPDQMC.jl" begin
    path = "test/VBS/"

    rng = MersenneTwister(time_ns())

    model = VBS_Hubbard_Para(SUN=3, Ht=1.0, HJ1=3.0, HJ2=3.0,
        Θrelax=3.6, Θquench=0., Lattice="HoneyComb120",
        site=[6, 6], Δt=0.05, BatchSize=5, Initial="H0")

    s = Initial_s(model, rng)
    # s = phy_update(path, model, s, 10, false)
    # s = phy_update(path, model, s, 300, true)


    Phy = KAPDQMC.VBSDQMC.PhyBuffer(model.Ns, 0)
    Phy.G = I(model.Ns) - model.Pt * inv(model.Pt' * model.Pt) * model.Pt'
    Ek, Ev, R0, R1 = KAPDQMC.VBSDQMC.phy_measure(model, Phy, div(model.Nt, 2), s)
    println("Ek = $Ek, Ev = $Ev, R0 = $R0, R1 = $R1")


    # L = model.site[2]
    # indexA = area_index(model.Lattice, model.site, ([1, 1], [div(L, 3), L]))
    # # # HalfHalf
    # indexB = area_index(model.Lattice, model.site, ([1, 1], [div(L, 3), div(2 * L, 3)]))

    # # println(model.Ns)
    # # println((indexA))
    # # println((indexB))

    # λ = 0.5
    # Nλ = 2

    # ss = [copy(s), copy(s)]
    # ss = ctrl_SCEEicr(path, model, indexA, indexB, 2, λ, Nλ, ss, true)


    # # # println(@btime ctrl_SCEEicr($path,$model,$indexA,$indexB,$Sweeps,$λ,$Nλ,$ss,$true) )


end
