push!(LOAD_PATH, "D:\\JuliaDQMC\\code\\KAPDQMC\\src\\")

using KAPDQMC

using DelimitedFiles
using ProgressMeter
using LinearAlgebra
using Random

path = "D:\\JuliaDQMC\\data\\U=0\\flux-square\\"

flux = π
Lattice = "SQUARE90"
filling_num = 0.5
Δt = 0.1
Initial = "H0"

L = [4, 6, 8, 10]
# L = collect(4:2:20)
# L = collect(22:2:40)
# L = collect(42:2:50)


println("L = ", L)
for i in eachindex(L)

    site = [L[i], L[i]]

    # Half
    indexA = area_index(Lattice, site, ([1, 1], [div(L[i], 2), L[i]]))

    # HalfHalf
    indexB = area_index(Lattice, site, ([1, 1], [div(L[i], 2), div(L[i], 2)]))

    K = nnK_Matrix(Lattice, site, flux=flux)

    E, V = LAPACK.syevd!('V', 'L', copy(K))
    eK = V * Diagonal(exp.(-Δt .* E)) * V'

    Ns = size(K, 1)
    ns = round(Int, Ns * filling_num)
    tmpNn = Matrix{ComplexF64}(undef, Ns, ns)
    tmpnN = Matrix{ComplexF64}(undef, ns, Ns)
    tmpnn = Matrix{ComplexF64}(undef, ns, ns)
    tau = Vector{ComplexF64}(undef, ns)
    ipiv = Vector{LAPACK.BlasInt}(undef, ns)
    G0 = Array{ComplexF64}(undef, Ns, Ns)

    Pt = zeros(ComplexF64, Ns, ns)  # 预分配 Pt
    if Initial == "H0"
        KK = Matrix{ComplexF64}(K)

        # KK[KK.!=0] .+= (rand(size(KK)...)*1e-2)[KK.!=0]
        # KK = (KK + KK') ./ 2

        μ = 0.01
        KK += μ * Diagonal(repeat([-1, 1], div(Ns, 2)))

        E, V = LAPACK.syevd!('V', 'L', KK)
        Pt = V[:, 1:ns]
    elseif Initial == "V"
        for i in 1:ns
            Pt[i*2, i] = 1
        end
    end

    EA = EB = 0.0
    EA1 = EB1 = 1.0

    while abs(EA - EA1) + abs(EB - EB1) > 1e-2
        EA = EA1
        EB = EB1

        for _ in 1:5
            mul!(tmpNn, eK, Pt)

            mul!(Pt, eK, tmpNn)

            LAPACK.geqrf!(Pt, tau)
            LAPACK.orgqr!(Pt, tau, ns)
        end
        mul!(tmpnn, Pt', Pt)
        LAPACK.getrf!(tmpnn, ipiv)
        LAPACK.getri!(tmpnn, ipiv)
        mul!(tmpNn, Pt, tmpnn)
        mul!(G0, tmpNn, Pt')
        lmul!(-1.0, G0)
        for iii in diagind(G0)
            G0[iii] += 1
        end

        EA1 = -log(abs2(det(GroverMatrix(G0[indexA[:], indexA[:]], G0[indexA[:], indexA[:]]))))
        EB1 = -log(abs2(det(GroverMatrix(G0[indexB[:], indexB[:]], G0[indexB[:], indexB[:]]))))
    end


    # open("$(path)eq_flux$(flux).csv", "a") do io
    #     lock(io)
    #     writedlm(io, [L[i], EA, EB, EA - EB]', ',')
    #     unlock(io)
    # end
    println("L=$(L[i]) EA=$(EA) EB=$(EB) ΔE=$(EA - EB)")

end


# println("$(path)eq_flux$(flux).csv Done!")

