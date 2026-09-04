push!(LOAD_PATH, "D:\\JuliaDQMC\\code\\KAPDQMC\\src\\")
using KAPDQMC
using Test
using Random
using LinearAlgebra


@testset "KAPDQMC.jl" begin
    path = "test/VBS/"

    rng = MersenneTwister(time_ns())

    # model = VBS_Hubbard_Para(SUN=3, Ht=1.0, HJ1=1.0, HJ2=1.0,
    #     Θrelax=3.6, Θquench=0., Lattice="HoneyComb120",
    #     site=[9, 9], Δt=0.05, BatchSize=5, Initial="VBS")

    model = VBS_Hubbard_Para(SUN=3, Ht=1.0, HJ1=3.5, HJ2=2.9,
        Θrelax=0.0, Θquench=3.6, Lattice="HoneyComb120",
        site=[9, 9], Δt=0.05, BatchSize=5, Initial="VBS", relax=true)

    s = Initial_s(model, rng)
    # s = phy_update(path, model, s, 10, false)
    # s = phy_update(path, model, s, 300, true)


    # Phy = KAPDQMC.VBSDQMC.PhyBuffer(model.Ns, 0)
    # Phy.G = I(model.Ns) - model.Pt * inv(model.Pt' * model.Pt) * model.Pt'
    # Ek, Ev, R0, R1 = KAPDQMC.VBSDQMC.phy_measure(model, Phy, div(model.Nt, 2), s)
    # println("Ek = $Ek, Ev = $Ev, R0 = $R0, R1 = $R1")


    # L = model.site[2]
    # indexA = area_index(model.Lattice, model.site, ([1, 1], [div(L, 3), L]))
    # # # HalfHalf
    # indexB = area_index(model.Lattice, model.site, ([1, 1], [div(L, 3), div(2 * L, 3)]))

    # G0 = I(model.Ns) - model.Pt * inv(model.Pt' * model.Pt) * model.Pt'
    # gm_F = GroverMatrix(G0, G0)
    # gm_A = GroverMatrix(G0[indexA, indexA], G0[indexA, indexA])
    # gm_B = GroverMatrix(G0[indexB, indexB], G0[indexB, indexB])
    # println((det(gm_A)))
    # println((det(gm_B)))
    # println((det(gm_F)))


    # println(model.Ns)
    # println((indexA))
    # println((indexB))

    # λ = 0.
    # Nλ = 1

    # ss = [copy(s), copy(s)]
    # ss = ctrl_SCEEicr(path, model, indexA, indexB, 2, λ, Nλ, ss, true)


    # # # println(@btime ctrl_SCEEicr($path,$model,$indexA,$indexB,$Sweeps,$λ,$Nλ,$ss,$true) )


end
