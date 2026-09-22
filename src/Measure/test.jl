push!(LOAD_PATH, "D:\\JuliaDQMC\\code\\KAPDQMC\\src\\")
using KAPDQMC
using Test
using Random
using LinearAlgebra

Lattice = "HoneyComb120"
Initial = "H0"

model = tU_Hubbard_Para(Ht=1.0, Hu1=2.5, Hu2=2.5, Θrelax=1.0, Θquench=0.0,
    Lattice=Lattice, site=[3, 3], Δt=0.05, BatchSize=5, Initial=Initial, flux=0, opt="y")


##############################################################################
# beta = 10
# MO = Diagonal(exp.(beta .* n))


# nidx = randperm(model.Ns)[1:(model.Ns÷2)]

# n = ones(model.Ns)
# n[nidx] .= exp(-beta)
# MO = Diagonal(n)

# # MO = I(model.Ns)
# M_Pt = MO * model.Pt
# # M_Pt = Matrix(qr(M_Pt).Q)


# G = M_Pt * inv(M_Pt' * M_Pt) * M_Pt'

# GM = GroverMatrix(G, G)

# EE = -log(abs2(det(GM)))

# det(M_Pt' * M_Pt) / det(model.Pt' * model.Pt)




##############################################################################
function Poss_M(E_idx, Pt)
    Ns = size(Pt, 1)
    n = ones(Ns)
    n[E_idx] .= exp(-2 * beta)
    MO = Diagonal(n)
    M_Pt = MO * Pt
    Poss = det(M_Pt' * M_Pt)
    return Poss
end
# 更新测试
beta = 4

for _ in 1:10
    E_idx = randperm(model.Ns)[1:(model.Ns÷2)]
    S_idx = setdiff(1:model.Ns, E_idx)
    @assert sort(vcat(E_idx, S_idx)) == collect(1:model.Ns)

    for __ in 1:10
        i = rand(1:model.Ns÷2)  #E增加i
        j = rand(1:model.Ns÷2)   #E减少j

        Poss1 = Poss_M(E_idx, model.Pt)

        E_idx2 = copy(E_idx)
        S_idx2 = copy(S_idx)
        E_idx2[j] = S_idx[i]         # E[j] <- S[i]
        S_idx2[i] = E_idx[j]      # S[i] <- E[j]
        @assert sort(vcat(E_idx2, S_idx2)) == collect(1:model.Ns)

        Poss2 = Poss_M(E_idx2, model.Pt)
        Poss2 / Poss1

        ##############################################################################

        G = model.Pt * inv(model.Pt' * model.Pt) * model.Pt'

        λ = 1 - exp(-4 * beta)
        X = inv(I(model.Ns ÷ 2) - λ * G[E_idx, E_idx])

        g = G[E_idx, S_idx[i]]
        z = X * g

        zbar = z .- X[:, j] .* (z[j] / X[j, j])
        # zbar[j] = zero(eltype(zbar))

        Δ = 1 - λ * G[S_idx[i], S_idx[i]] - λ^2 * dot(G[S_idx[i], E_idx], zbar)

        ratio = Δ * X[j, j]

        if abs(ratio - Poss2 / Poss1) / abs(ratio) > 1e-5
            println(abs(ratio - Poss2 / Poss1))
            println(ratio)
            println(Poss2 / Poss1)
        end

        if rand() < abs(ratio)
            # 这里直接接受这次提议，测试 X 的增量更新。
            # 先保存旧 X，因为下面的三条更新必须全部使用旧值。
            keep = [k for k in axes(X, 1) if k != j]
            keep = collect(1:model.Ns ÷ 2)

            X_new = zeros(eltype(X), size(X))
            # 删除第 j 行、第 j 列后的 Schur 补逆矩阵：
            X_new[keep, keep] .=
                X[keep, keep] .-
                (1 / X[j, j]) .* (X[keep, j] * X[j, keep]')

            # 先更新 \hat E × \hat E 子块，再更新第 j 行和第 j 列。
            X_new[keep, keep] .+=
                (λ^2 / Δ) .* (zbar[keep] * zbar[keep]')
            X_new[keep, j] .= (λ / Δ) .* zbar[keep]
            X_new[j, keep] .= (λ / Δ) .* conj.(zbar[keep])
            X_new[j, j] = 1 / Δ

            # 数组构型也做同一个交换。
            E_idx = copy(E_idx2)
            S_idx = copy(S_idx2)

            X_new_true = inv(I(model.Ns ÷ 2) - λ * G[E_idx, E_idx])
            X_update_abs_error = norm(X_new_true - X_new)
            X_update_rel_error = X_update_abs_error / norm(X_new_true)
            if X_update_abs_error > 1e-5
                println("X update absolute error = ", X_update_abs_error)
                println("X update relative error = ", X_update_rel_error)
                if X_update_rel_error > 1e-5
                    error("!!!!!!!!!!!!!!!!!!!!")
                end
            end
            X = copy(X_new)
        end

    end
end

