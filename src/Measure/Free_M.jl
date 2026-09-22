push!(LOAD_PATH, "D:\\JuliaDQMC\\code\\KAPDQMC\\src\\")
using KAPDQMC
using Test
using Random
using LinearAlgebra

@testset "Measure_EE.jl" begin
    Sweeps = 1000
    Result = zeros(Float64, Sweeps, 2)
    #############-----preset-----####################
    Lattice = "HoneyComb120"
    Initial = "H0"
    L = 6
    beta = 4.0
    site = [L, L]
    idxA = area_index(Lattice, site, ([1, 1], [L ÷ 3, L]))
    idxB = area_index(Lattice, site, ([1, 1], [L ÷ 3, L * 2 ÷ 3]))
    #############-----preset-----####################


    λ = 1 - exp(-4 * beta)
    K = nnK_Matrix(Lattice, site)

    Ns = size(K)[1]
    ns = round(Int, Ns / 2)
    Pt = zeros(Float64, Ns, ns)  # 预分配 Pt
    E, V = LAPACK.syevd!('V', 'L', K)
    Pt = V[:, 1:ns]
    G = Pt * inv(Pt' * Pt) * Pt'

    # 初始化
    E_idx = randperm(Ns)[1:ns]
    S_idx = setdiff(1:Ns, E_idx)
    X = inv(I(ns) - λ * G[E_idx, E_idx])
    g = zeros(Float64, ns)
    zbar = zeros(Float64, ns)
    tmpNf = zeros(Float64, ns)     # work for z
    tmpNfNf = zeros(Float64, ns, ns)  # work for Xnew

    MO = ones(Ns)
    M_Pt = zeros(Float64, Ns, ns)
    GM_A = zeros(Float64, length(idxA), length(idxA))
    GM_B = zeros(Float64, length(idxB), length(idxB))

    function Poss_M(E_idx, Pt)
        """
        测试函数
        """
        Ns = size(Pt, 1)
        n = ones(Ns)
        n[E_idx] .= exp(-2 * beta)
        MO = Diagonal(n)
        M_Pt = MO * Pt
        Poss = det(M_Pt' * M_Pt)
        return Poss
    end

    for loop in 1:Sweeps
        if loop % 100 == 0
            println("loop = $loop")
            X = inv(I(ns) - λ * G[E_idx, E_idx])
        end
        i = rand(1:ns)  #E增加i
        j = rand(1:ns)   #E减少j

        g .= G[E_idx, S_idx[i]]
        mul!(tmpNf, X, g)     # work for z
        zbar .= tmpNf .- (tmpNf[j] / X[j, j]) .* X[:, j]
        zbar[j] = zero(eltype(zbar))

        Δ = 1 - λ * G[S_idx[i], S_idx[i]] - λ^2 * dot(G[S_idx[i], E_idx], zbar)
        ratio = Δ * X[j, j]

        ##############################################################################
        E_idx2 = copy(E_idx)
        S_idx2 = copy(S_idx)
        tmp = E_idx2[j]
        E_idx2[j] = S_idx[i]         # E[j] <- S[i]
        S_idx2[i] = tmp         # S[i] <- E[j]
        Poss1 = Poss_M(E_idx, Pt)
        Poss2 = Poss_M(E_idx2, Pt)
        @assert abs(ratio - Poss2 / Poss1) / abs(ratio) < 1e-5 abs(ratio - Poss2 / Poss1) / abs(ratio)
        ##############################################################################

        if rand() < ratio
            mul!(tmpNfNf, X[:, j], X[j, :]')

            X .-= tmpNfNf ./ X[j, j]
            mul!(tmpNfNf, zbar, zbar')
            X .+= (λ^2 / Δ) .* tmpNfNf
            X[:, j] .= (λ / Δ) .* zbar
            X[j, :] .= (λ / Δ) .* conj.(zbar)
            X[j, j] = 1 / Δ

            tmp = E_idx[j]
            E_idx[j] = S_idx[i]         # E[j] <- S[i]
            S_idx[i] = tmp         # S[i] <- E[j]
            ##############################################################################
            X_new_true = inv(I(ns) - λ * G[E_idx, E_idx])
            X_update_abs_error = norm(X_new_true - X)
            X_update_rel_error = X_update_abs_error / norm(X_new_true)
            @assert X_update_rel_error < 1e-5 X_update_rel_error
            ##############################################################################
        end
        MO .= 1
        MO[E_idx] .= exp(-2 * beta)
        mul!(M_Pt, Diagonal(MO), Pt)

        M_G = M_Pt * inv(M_Pt' * M_Pt) * M_Pt'

        GroverMatrix!(GM_A, view(M_G, idxA, idxA), view(M_G, idxA, idxA))
        GroverMatrix!(GM_B, view(M_G, idxB, idxB), view(M_G, idxB, idxB))

        Result[loop, 1] = det(GM_A)
        Result[loop, 2] = det(GM_B)
    end

    file = "./Free_HC.csv"
    writedlm(file, Result, ',')

end