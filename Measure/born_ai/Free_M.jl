push!(LOAD_PATH, normpath(joinpath(@__DIR__, "..", "..")))
using KAPDQMC
using Test
using Random
using LinearAlgebra
using DelimitedFiles
include("BornSampler.jl")
using .BornSampler

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


function Measure_EE(beta, Ns, ns; rng=Random.default_rng())
    Ns == 2ns || throw(ArgumentError("the swap sampler requires half filling"))
    BornSampler.check_beta(beta)
    idx = randperm(rng, Ns)[1:ns]
    return Measure_EE_(
        beta,
        -expm1(-4 * beta),
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

function UpdateMeasure!(M::Measure_EE_, G::Matrix{Float64}, tmpn, tmpnn, Pt;
                        rng=Random.default_rng())
    ns = length(M.E_idx)
    accepted = 0
    for _ in 1:2*ns
        i = rand(rng, 1:ns)  #E增加i
        j = rand(rng, 1:ns)   #E减少j

        M.g .= G[M.E_idx, M.S_idx[i]]
        mul!(tmpn, M.X, M.g)
        M.zbar .= tmpn .- (tmpn[j] / M.X[j, j]) .* M.X[:, j]
        M.zbar[j] = 0.0 # excluded site; exactly zero in the derivation

        Δ = 1 - M.λ * G[M.S_idx[i], M.S_idx[i]] - M.λ^2 * dot(G[M.S_idx[i], M.E_idx], M.zbar)
        ratio = Δ * M.X[j, j]
        isfinite(ratio) && ratio > 0 && Δ > 0 && M.X[j,j] > 0 ||
            error("invalid swap ratio: rebuild X, shorten refresh_every, or use sampler=:iid")

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

        if rand(rng) < min(1.0,ratio)
            accepted += 1
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
    return (; accepted, proposed=2ns)
end

struct tmpME_
    tmpn::Vector{Float64}
    tmpnn::Matrix{Float64}
end


"""
Fixed-half-outcome ensemble. `:local` keeps the swap kernel; `:iid` draws
independent outcomes of the SAME distribution. Average the saved per-sample
entropy; the auxiliary F in the iid sampler never replaces the physical state.
"""
function Free_Measure_EE(L, beta, Sweeps; sampler=:local,
                         burnin=(sampler == :local ? 100 : 0), refresh_every=100,
                         rng=Random.default_rng(),
                         output_dir=joinpath(@__DIR__, "samples", "L=$(L)"))
    sampler in (:local,:iid) || throw(ArgumentError("sampler must be :local or :iid"))
    L >= 3 && L % 3 == 0 || throw(ArgumentError("entropy regions require L divisible by 3"))
    Sweeps > 0 && burnin >= 0 && refresh_every > 0 || throw(ArgumentError("invalid sampling lengths"))
    Result = zeros(Float64, Sweeps, 2)
    Entropy = similar(Result)
    Diagnostics = zeros(Float64, Sweeps, 3)
    #############-----preset-----####################
    Lattice = "HoneyComb120"
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
    M1 = Measure_EE(beta, Ns, ns; rng)
    iid = sampler == :iid ? IIDSampler(Pt,beta) : nothing
    if sampler == :local
        M1.λ < 1 || error("I-lambda G is numerically singular; use sampler=:iid")
        M1.X = inv(cholesky(Symmetric(I(ns) - M1.λ * G[M1.E_idx, M1.E_idx])))
    end

    tmpn = zeros(Float64, ns)     # work for z
    tmpnn = zeros(Float64, ns, ns)  # work for Xnew
    tmpNn = zeros(Float64, Ns, ns)
    tmpnN = zeros(Float64, ns, Ns)
    tau = Vector{Float64}(undef, ns)
    ipiv = Vector{LAPACK.BlasInt}(undef, ns)

    GM_A = zeros(Float64, length(idxA), length(idxA))
    GM_B = zeros(Float64, length(idxB), length(idxB))

    for step in 1:burnin+Sweeps
        if sampler == :local
            if (step-1) % refresh_every == 0
                M1.X = inv(cholesky(Symmetric(I(ns) - M1.λ * G[M1.E_idx, M1.E_idx])))
            end
            stats = UpdateMeasure!(M1, G, tmpn, tmpnn, Pt; rng)
        else
            M1.E_idx .= sample_measurement(iid; rng)
            M1.S_idx .= setdiff(1:Ns,M1.E_idx)
            M1.MO .= 1.0
            M1.MO[M1.E_idx] .= exp(-2beta)
            stats = (; accepted=0, proposed=0)
        end
        step <= burnin && continue
        loop = step-burnin
        Diagnostics[loop,:] .= (loop,stats.accepted,stats.proposed)

        # 因为是自由费米子，所以这里special，两个replica的BR，BL是一样的
        mul!(M1.M_BR, Diagonal(M1.MO), BR)
        LAPACK.geqrf!(M1.M_BR, tau)
        LAPACK.orgqr!(M1.M_BR, tau, ns)

        mul!(M1.M_BL, BL, Diagonal(M1.MO))
        LAPACK.gerqf!(M1.M_BL, tau)
        LAPACK.orgrq!(M1.M_BL, tau, ns)

        KAPDQMC.tUDQMC.get_G!(tmpnn, tmpnN, ipiv, M1.M_BL, M1.M_BR, M1.M_G)

        GroverMatrix!(GM_A, view(M1.M_G, idxA, idxA), view(M1.M_G, idxA, idxA))
        GroverMatrix!(GM_B, view(M1.M_G, idxB, idxB), view(M1.M_G, idxB, idxB))

        Entropy[loop, 1] = -logdet(cholesky(Symmetric(GM_A)))
        Entropy[loop, 2] = -logdet(cholesky(Symmetric(GM_B)))
        Result[loop, :] .= exp.(-Entropy[loop, :])
    end

    mkpath(output_dir)
    stem = "FreeHCL$(L)_β$(beta)_$(sampler)"
    file = joinpath(output_dir,stem*".csv")
    writedlm(file, Result, ',')
    writedlm(joinpath(output_dir,stem*"_entropy.tsv"),Entropy,'\t')
    writedlm(joinpath(output_dir,stem*"_diagnostics.tsv"),
        vcat(permutedims(["sample","accepted","proposed"]),Diagnostics),'\t')
    open(joinpath(output_dir,stem*"_metadata.txt"),"w") do io
        println(io,"sampler=$sampler; beta=$beta; L=$L; retained=$Sweeps; burnin=$burnin; refresh_every=$refresh_every")
        println(io,"one local sweep = $Ns attempted swaps; iid has no acceptance ratio")
        println(io,"ensemble: |E|=Nsites/2; fixed ground-state Slater; real orbitals")
        println(io,"purity columns: A,B; entropy columns: S2(A),S2(B); average per-sample entropy")
    end
    return (; purity=Result, entropy=Entropy, diagnostics=Diagnostics, file)
end


if abspath(PROGRAM_FILE) == @__FILE__
    L_set = [12, 18]
    beta_set = collect(0.0:0.1:3.0)
    for L in L_set
        for beta in beta_set
            println("L = $(L), beta = $(beta)")
            Free_Measure_EE(L, beta, 1000)
        end
    end
end

