# push!(LOAD_PATH, "D:\\JuliaDQMC\\code\\KAPDQMC\\src\\")
# using Random
# using LinearAlgebra, LinearAlgebra.BLAS, LinearAlgebra.LAPACK
# using KAPDQMC

# model = tV_Hubbard_Para(Ht=1.0, Hv1=1.0, Hv2=1.0, Θrelax=0.1, Θquench=0.0, Lattice="SQUARE90",
#     site=[4, 4], Δt=0.05, BatchSize=5, Initial="H0", flux=0)
# println("$(model.Lattice) Hubbard model initialized.")

# rng = MersenneTwister(1234)
# s = Initial_s(model, rng)

# nnidx = nnidx_F(model.Lattice, model.site)

# println(size(nnidx))

# tmpV = zeros(Float64, model.Ns, model.Ns, size(nnidx, 2))
# UV = zeros(Float64, model.Ns, model.Ns, size(nnidx, 2))

# for bond in 1:size(nnidx, 2)
#     for i in 1:size(nnidx, 1)
#         x, y = nnidx[i, bond]
#         tmpV[x, y, bond] = model.η[s[i, bond, 1]]
#         tmpV[y, x, bond] = model.η[s[i, bond, 1]]
#     end
#     E, V = eigen(tmpV[:, :, bond])
#     UV[:, :, bond] .= V
# end

# UV[:, :, 1]

# for i in 1:size(nnidx, 2)
#     for j in 1:size(nnidx, 2)
#         if i != j
#             iscomute = UV[:, :, i]' * tmpV[:, :, j] * UV[:, :, i]
#             iscomute = iscomute - diagm(diag(iscomute))
#             @assert norm(iscomute) > 1e-10
#         else
#             isdiag = UV[:, :, i]' * tmpV[:, :, j] * UV[:, :, i]
#             isdiag = isdiag - diagm(diag(isdiag))
#             @assert norm(isdiag) < 1e-10
#         end
#     end
#     @assert norm(UV[:, :, i]' * UV[:, :, i] - I(model.Ns)) < 1e-10
# end

# ######################################################################
# # divide bond-interation into 2 parts: up + right and down + left
# # each part has no public site
# ######################################################################

# nnidx2 = Matrix{Tuple{Int64,Int64}}(undef, size(nnidx, 1) * 2, div(size(nnidx, 2), 2))
# nnidx2[:, 1] = vcat(nnidx[:, 1], nnidx[:, 3])
# nnidx2[:, 2] = vcat(nnidx[:, 2], nnidx[:, 4])

# tmpV = zeros(Float64, model.Ns, model.Ns, size(nnidx2, 2))
# UV = zeros(Float64, model.Ns, model.Ns, size(nnidx2, 2))

# a, b = size(nnidx2)
# s = zeros(UInt8, a, b, model.Nt)
# sp = Random.Sampler(rng, [1, 2, 3, 4])
# for i in eachindex(s)
#     s[i] = rand(rng, sp)
# end

# for bond in 1:size(nnidx2, 2)
#     for i in 1:size(nnidx2, 1)
#         x, y = nnidx2[i, bond]
#         tmpV[x, y, bond] = model.η[s[i, bond, 1]]
#         tmpV[y, x, bond] = model.η[s[i, bond, 1]]
#     end
#     # E, V = eigen(tmpV[:, :, bond])
#     # UV[:, :, bond] .= V
# end

# for i in 1:size(nnidx2, 2)
#     for j in 1:size(nnidx2, 2)
#         if i != j
#             iscomute = UV[:, :, i]' * tmpV[:, :, j] * UV[:, :, i]
#             iscomute = iscomute - diagm(diag(iscomute))
#             @assert norm(iscomute) > 1e-10
#             @assert norm(tmpV[:, :, i] * tmpV[:, :, j] - tmpV[:, :, j] * tmpV[:, :, i]) > 1e-10
#         else
#             isdiag = UV[:, :, i]' * tmpV[:, :, j] * UV[:, :, i]
#             isdiag = isdiag - diagm(diag(isdiag))
#             @assert norm(isdiag) < 1e-10
#         end
#     end
#     @assert norm(UV[:, :, i]' * UV[:, :, i] - I(model.Ns)) < 1e-10
# end


# ss = copy(s)
# lt = 1
# idx_bond = 1
# x_idx = 2
# ss[x_idx, idx_bond, lt] = rand(rng, model.samplers_dict[s[x_idx, idx_bond, lt]])
# tmpVV = zeros(Float64, model.Ns, model.Ns, size(nnidx2, 2))
# for bond in 1:size(nnidx2, 2)
#     for i in 1:size(nnidx2, 1)
#         x, y = nnidx2[i, bond]
#         tmpVV[x, y, bond] = model.η[ss[i, bond, 1]]
#         tmpVV[y, x, bond] = model.η[ss[i, bond, 1]]
#     end
# end
# x, y = nnidx2[x_idx, idx_bond]
# subidx = [x, y]
# (tmpV-tmpVV)[subidx, subidx, idx_bond]
# size(tmpV)

# tmpV - tmpVV

# findall(s .!= ss)
# findall(tmpVV .!= tmpV)


if abspath(PROGRAM_FILE) == @__FILE__
    push!(LOAD_PATH, "D:\\JuliaDQMC\\code\\KAPDQMC\\src\\")
    using KAPDQMC
    using LinearAlgebra, Random
    rng = MersenneTwister(1234)

    model = tV_Hubbard_Para(Ht=1.0, Hv1=1.0, Hv2=1.0, Θrelax=3.0, Θquench=0.0,
        Lattice="SQUARE90", site=[3, 3], Δt=0.05, BatchSize=5, Initial="H0", flux=π)
    println("$(model.Lattice) Hubbard model initialized.")

    lt = 1
    bond_num = size(model.nnidx, 2)
    tmpVV = zeros(Float64, bond_num, model.Ns, model.Ns)
    println("TEST For diag transformation of nn interaction UV*Diagonal(s)*UV' = V)")
    for _ in 1:10
        s = Initial_s(model, rng)
        for j in bond_num:-1:1
            tmpN = zeros(Float64, model.Ns)
            tmpV = zeros(Float64, model.Ns, model.Ns)
            for i in 1:div(model.Ns, 2)
                # println(i," ",j,": ",model.nnidx[i,j])
                x, y = model.nnidx[i, j]
                tmpN[x] = model.η[s[i, j, lt]]
                tmpN[y] = -model.η[s[i, j, lt]]

                tmpV[x, y] = model.η[s[i, j, lt]]
                tmpV[y, x] = model.η[s[i, j, lt]]
            end
            tmpVV[j, :, :] = tmpV[:, :]
            @assert norm(model.UV[:, :, j] * Diagonal(tmpN) * model.UV[:, :, j]' - tmpV) < 1e-5
            @assert norm(model.UV[:, :, j]' * model.UV[:, :, j] - I(model.Ns)) < 1e-5
            # specially: UV'=UV
            @assert norm(model.UV[:, :, j] - model.UV[:, :, j]') < 1e-5
        end
    end

    println("TEST for not comute for V_i,V_j (i≠j)")
    for i in 1:bond_num
        for j in 1:bond_num
            if i != j
                @assert norm(tmpVV[i, :, :] * tmpVV[j, :, :] - tmpVV[j, :, :] * tmpVV[i, :, :]) > 1e-5
            end
        end
    end

    # TEST for comute for V_i,V_i^′
    for j in bond_num:-1:1
        tmp = tmpVV[j, :, :]
        for i in 1:div(model.Ns, 2)
            x, y = model.nnidx[i, j]
            tmp[x, y] = -tmp[x, y]
            tmp[y, x] = -tmp[y, x]
            @assert norm(tmp * tmpVV[j, :, :] - tmpVV[j, :, :] * tmp) < 1e-5
        end
    end

    println("TEST for diag transformation of update uv*Diagonal(Δs)*UV' = ΔV")
    uv = [-2^0.5/2 -2^0.5/2; -2^0.5/2 2^0.5/2]
    for sx_f in 1:4
        for sx_i in 1:4
            tmp22 = (model.η[sx_f] - model.η[sx_i]) * [0 1; 1.0 0]
            tmp2 = (model.η[sx_f] - model.η[sx_i]) * [1, -1]
            @assert norm(uv * Diagonal(tmp2) * uv' .- tmp22) < 1e-5
        end
    end
end


