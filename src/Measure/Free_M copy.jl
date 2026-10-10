push!(LOAD_PATH, "D:\\JuliaDQMC\\code\\KAPDQMC\\src\\")
using KAPDQMC
using Test
using Random
using LinearAlgebra
using DelimitedFiles

mutable struct Measure_EE_
    beta::Float64
    λ::Float64
    E_idx::Vector{Int}
    S_idx::Vector{Int}
    X::Matrix{Float64}
    g::Vector{Float64}
    zbar::Vector{Float64}
    MO::Vector{Float64}
    M_BR::Matrix{Float64}
    M_BL::Matrix{Float64}
    M_G::Matrix{Float64}
end


function Measure_EE(beta, Ns, ns)
    idx = randperm(Ns)[1:ns]
    return Measure_EE_(
        beta,
        1 - exp(-4 * beta),
        idx,
        setdiff(1:Ns, idx),
        zeros(Float64, ns, ns),
        zeros(Float64, ns),
        zeros(Float64, ns),
        ones(Float64, Ns),
        zeros(Float64, Ns, ns),
        zeros(Float64, ns, Ns),
        zeros(Float64, Ns, Ns)
    )
end

function Poss_M(beta, E_idx, Pt)
    """
    测试函数
    """
    Ns = size(Pt, 1)
    n = ones(Ns)
    n[E_idx] .= exp(-2 * beta)
    MO_test = Diagonal(n)
    M_Pt_test = MO_test * Pt
    Poss = det(M_Pt_test' * M_Pt_test)
    return Poss
end

function UpdateMeasure!(M::Measure_EE_, G::Matrix{Float64}, tmpn, tmpnn, Pt)
    ns = length(M.E_idx)
    for _ in 1:2*ns
        i = rand(1:ns)  #E增加i
        j = rand(1:ns)   #E减少j

        M.g .= G[M.E_idx, M.S_idx[i]]
        mul!(tmpn, M.X, M.g)
        M.zbar .= tmpn .- (tmpn[j] / M.X[j, j]) .* M.X[:, j]

        Δ = 1 - M.λ * G[M.S_idx[i], M.S_idx[i]] - M.λ^2 * dot(G[M.S_idx[i], M.E_idx], M.zbar)
        ratio = Δ * M.X[j, j]

        #############################################################################
        # E_idx2 = copy(M.E_idx)
        # S_idx2 = copy(M.S_idx)
        # tmp = E_idx2[j]
        # E_idx2[j] = S_idx2[i]         # E[j] <- S[i]
        # S_idx2[i] = tmp         # S[i] <- E[j]
        # Poss1 = Poss_M(M.beta, M.E_idx, Pt)
        # Poss2 = Poss_M(M.beta, E_idx2, Pt)
        # @assert abs(ratio - Poss2 / Poss1) / abs(ratio) < 1e-5 abs(ratio - Poss2 / Poss1) / abs(ratio)
        ##############################################################################

        if rand() < ratio
            mul!(tmpnn, M.X[:, j], M.X[j, :]')

            M.X .-= tmpnn ./ M.X[j, j]
            mul!(tmpnn, M.zbar, M.zbar')
            M.X .+= (M.λ^2 / Δ) .* tmpnn
            M.X[:, j] .= (M.λ / Δ) .* M.zbar
            M.X[j, :] .= (M.λ / Δ) .* conj.(M.zbar)
            M.X[j, j] = 1 / Δ

            tmp = M.E_idx[j]
            M.E_idx[j] = M.S_idx[i]         # E[j] <- S[i]
            M.S_idx[i] = tmp         # S[i] <- E[j]
            ##############################################################################
            # X_new_true = inv(I(ns) - M.λ * G[M.E_idx, M.E_idx])
            # X_update_abs_error = norm(X_new_true - M.X)
            # X_update_rel_error = X_update_abs_error / norm(X_new_true)
            # @assert X_update_rel_error < 1e-5 X_update_rel_error
            ##############################################################################
        end
    end
    M.MO .= 1
    M.MO[M.E_idx] .= exp(-2 * M.beta)
end

struct tmpME_
    tmpn::Vector{Float64}
    tmpnn::Matrix{Float64}
end


function Free_Measure_EE(beta)
    Sweeps = 10000
    Result = zeros(Float64, Sweeps, 2)
    #############-----preset-----####################
    Lattice = "HoneyComb120"
    L = 6
    site = [L, L]
    idxA = area_index(Lattice, site, ([1, 1], [L ÷ 3, L]))
    idxB = area_index(Lattice, site, ([1, 1], [L ÷ 3, L * 2 ÷ 3]))
    #############-----preset-----####################


    K = nnK_Matrix(Lattice, site)
    Ns = size(K)[1]
    ns = round(Int, Ns / 2)
    Pt = zeros(Float64, Ns, ns)  # 预分配 Pt
    E, V = LAPACK.syevd!('V', 'L', K)
    Pt .= V[:, 1:ns]
    BR = copy(Pt)
    BL = copy(Pt')
    G = BR * inv(BL * BR) * BL

    # 初始化
    M1 = Measure_EE(beta, Ns, ns)
    M2 = Measure_EE(beta, Ns, ns)
    M1.X = inv(I(ns) - M1.λ * G[M1.E_idx, M1.E_idx])
    M2.X = inv(I(ns) - M2.λ * G[M2.E_idx, M2.E_idx])

    tmpn = zeros(Float64, ns)     # work for z
    tmpnn = zeros(Float64, ns, ns)  # work for Xnew
    tmpNn = zeros(Float64, Ns, ns)
    tmpnN = zeros(Float64, ns, Ns)
    tau = Vector{Float64}(undef, ns)
    ipiv = Vector{LAPACK.BlasInt}(undef, ns)

    GM_A = zeros(Float64, length(idxA), length(idxA))
    GM_B = zeros(Float64, length(idxB), length(idxB))

    for loop in 1:Sweeps
        if loop % 100 == 0
            # println("loop = $loop")
            M1.X = inv(I(ns) - M1.λ * G[M1.E_idx, M1.E_idx])
            M2.X = inv(I(ns) - M2.λ * G[M2.E_idx, M2.E_idx])
        end

        UpdateMeasure!(M1, G, tmpn, tmpnn, Pt)
        UpdateMeasure!(M2, G, tmpn, tmpnn, Pt)

        # 因为是自由费米子，所以这里special，两个replica的BR，BL是一样的
        mul!(M1.M_BR, Diagonal(M1.MO), BR)
        LAPACK.geqrf!(M1.M_BR, tau)
        LAPACK.orgqr!(M1.M_BR, tau, ns)

        mul!(M1.M_BL, BL, Diagonal(M1.MO))
        LAPACK.gerqf!(M1.M_BL, tau)
        LAPACK.orgrq!(M1.M_BL, tau, ns)

        KAPDQMC.tUDQMC.get_G!(tmpnn, tmpnN, ipiv, M1.M_BL, M1.M_BR, M1.M_G)

        mul!(M2.M_BR, Diagonal(M2.MO), BR)
        LAPACK.geqrf!(M2.M_BR, tau)
        LAPACK.orgqr!(M2.M_BR, tau, ns)
        mul!(M2.M_BL, BL, Diagonal(M2.MO))
        LAPACK.gerqf!(M2.M_BL, tau)
        LAPACK.orgrq!(M2.M_BL, tau, ns)

        KAPDQMC.tUDQMC.get_G!(tmpnn, tmpnN, ipiv, M2.M_BL, M2.M_BR, M2.M_G)

        GroverMatrix!(GM_A, view(M1.M_G, idxA, idxA), view(M2.M_G, idxA, idxA))
        GroverMatrix!(GM_B, view(M1.M_G, idxB, idxB), view(M2.M_G, idxB, idxB))

        Result[loop, 1] = det(GM_A)
        Result[loop, 2] = det(GM_B)
    end

    file = "./src/Measure/FreeHC_$(beta).csv"
    writedlm(file, Result, ',')
end


# beta_set = [0.0, 1.0, 2.0, 3.0, 4.0]
beta_set = collect(3.0:1.0:10.0)
for beta in beta_set
    Free_Measure_EE(beta)
end

