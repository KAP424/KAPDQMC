# turn off symmetric HS decomposition when debuging

# 将全局锁定义为模块级别的常量，避免每次函数调用都重新创建
const PHY_UPDATE_LOCK = ReentrantLock()

function phy_update(path::String, model::SO3_Hubbard_Para_, s::Array{UInt8,3}, Sweeps::Int64, record::Bool)
    TTT = time_ns()
    ERROR = 1e-6

    UPD = UpdateBuffer()
    NN = length(model.nodes)
    Phy = PhyBuffer(model.Ns, NN)
    Θidx = div(NN, 2) + 1

    name = name_Lattice(model.Lattice)

    if model.HJ1 == model.HJ2
        file = "$(path)/SO3phy$(name)_t$(model.Ht)V$(model.HJ1)size$(model.site)Δt$(model.Δt)Θ$(model.Θrelax)BS$(model.BatchSize).csv"
    else
        file = "$(path)/SO3phy$(name)_t$(model.Ht)V$(model.HJ1)_$(model.HJ2)size$(model.site)Δt$(model.Δt)Θ$(model.Θrelax)_$(model.Θquench)BS$(model.BatchSize).csv"
    end

    # 使用更类型稳定的随机数生成器初始化方式
    rng = MersenneTwister(Threads.threadid() + UInt64(time_ns()))

    PHY_RECORD = zeros(5)
    counter = 0

    G, BLs, BRs, tmpN, tmpNN, tmpnn, tmpnN, tmpNn, tau, ipiv, BM =
        Phy.G, Phy.BLs, Phy.BRs, Phy.N, Phy.NN, Phy.nn, Phy.nN, Phy.Nn, Phy.tau, Phy.ipiv, Phy.BM
    exp_αη_pos, exp_αη_neg = model.exp_αη_pos, model.exp_αη_neg

    BRs[:, :, 1] .= model.Pt
    BLs[:, :, NN] .= model.Pt'
    # BRs[:, :, 1] .= model.HalfeKinv * model.Pt
    # BLs[:, :, NN] .= model.Pt' * model.HalfeK
    for idx in NN-1:-1:1
        BM_F!(tmpN, tmpNN, BM, model, s, idx)
        mul!(tmpnN, view(BLs, :, :, idx + 1), BM)
        LAPACK.gerqf!(tmpnN, tau)
        LAPACK.orgrq!(tmpnN, tau)
        copyto!(view(BLs, :, :, idx), tmpnN)
        # view(BLs,:,:,idx) .= Matrix(qr!(tmpNn).Q)'
    end

    idx = 1
    get_G!(tmpnn, tmpnN, ipiv, view(BLs, :, :, 1), view(BRs, :, :, 1), G)
    for _ in 1:Sweeps
        # println("\n Sweep: $loop ")
        for lt in axes(s, 3)
            #####################################################################
            # println(lt)
            @assert norm(G - Gτ(model, s, lt - 1)) < ERROR "Initial G does not match Gτ for lt=$(lt): $(norm(G-Gτ(model,s,lt-1))) , $(norm(G)) , $(norm(Gτ(model,s,lt-1))) "
            #####################################################################

            mul!(tmpNN, G, model.eKinv)
            mul!(G, model.eK, tmpNN)
            # @assert norm(model.eKinv * G * model.eK - Gτ(model, s, lt - 1)) < 1e-8 "12321312G does not match Gτ for lt=$(lt): $(norm(model.eKinv * G * model.eK - Gτ(model, s, lt - 1)))"

            for j in reverse(axes(s, 2))
                fill!(tmpN, 1.0)
                for i in axes(s, 1)
                    x, y = model.bondidx[i, j]
                    tmpN[x] = exp_αη_neg[lt, s[i, j, lt]]  # 预计算的 exp(-α*η)
                    tmpN[y] = exp_αη_pos[lt, s[i, j, lt]]  # 预计算的 exp(α*η)
                end
                WrapV!(tmpNN, G, tmpN, view(model.UV, :, :, j), "B")

                UpdatePhyLayer!(rng, j, view(s, :, j, lt), lt, model, UPD, Phy)
                ####################################################################
                print("*")
                GG = model.eK * Gτ(model, s, lt - 1) * model.eKinv
                for jj in size(model.bondidx, 2):-1:j
                    # println("jj=$(jj)")
                    E = zeros(model.Ns)
                    for ii in 1:size(s)[1]
                        x, y = model.bondidx[ii, jj]
                        E[x] = exp_αη_neg[lt, s[ii, jj, lt]]  # 预计算的 exp(-α*η)
                        E[y] = exp_αη_pos[lt, s[ii, jj, lt]]  # 预计算的 exp(α*η)
                    end
                    GG = model.UV[:, :, jj] * Diagonal(E) * model.UV[:, :, jj]' * GG * model.UV[:, :, jj] * Diagonal(1 ./ E) * model.UV[:, :, jj]'
                end
                if (norm(G - GG) > ERROR)
                    println("lt=$(lt) j=$(j)")
                    error(j, " update error: ", norm(G - GG), "  lt=", lt)
                end
                ####################################################################
            end

            if record && abs(idx - Θidx) <= 1
                PHY_RECORD .+= phy_measure(model, Phy, lt, s)
                counter += 1
            end

            if any(model.nodes .== lt)
                idx += 1
                BM_F!(tmpN, tmpNN, BM, model, s, idx - 1)
                mul!(tmpNn, BM, view(BRs, :, :, idx - 1))
                LAPACK.geqrf!(tmpNn, tau)
                LAPACK.orgqr!(tmpNn, tau)
                copyto!(view(BRs, :, :, idx), tmpNn)

                # copyto!(tmpNN, G)

                get_G!(tmpnn, tmpnN, ipiv, view(BLs, :, :, idx), view(BRs, :, :, idx), G)

                #------------------------------------------------------------------#
                # axpy!(-1.0, G, tmpNN)
                # if norm(tmpNN) > 1e-7
                #     println("Warning for Batchsize Wrap Error : $(norm(tmpNN))")
                # end
                #-------------------------------------------------------------------
            end

        end
        for lt in reverse(axes(s, 3))
            #####################################################################
            if norm(G - Gτ(model, s, lt)) > ERROR
                error("Wrap-$(lt)   :   $(norm(G-Gτ(model,s,lt-1))) , $(norm(G)) , $(norm(Gτ(model,s,lt-1))) ")
            end
            ######################################################################
            for j in axes(s, 2)
                UpdatePhyLayer!(rng, j, view(s, :, j, lt), lt, model, UPD, Phy)
                fill!(tmpN, 1.0)
                for i in axes(s, 1)
                    x, y = model.bondidx[i, j]
                    tmpN[x] = exp_αη_pos[lt, s[i, j, lt]]  # 预计算的 exp(α*η)
                    tmpN[y] = exp_αη_neg[lt, s[i, j, lt]]  # 预计算的 exp(-α*η)
                end
                WrapV!(tmpNN, G, tmpN, view(model.UV, :, :, j), "B")
            end
            mul!(tmpNN, model.eKinv, G)
            mul!(G, tmpNN, model.eK)

            if record && abs(idx - Θidx) <= 1
                PHY_RECORD .+= phy_measure(model, Phy, lt - 1, s)
                counter += 1
            end

            if any(model.nodes .== (lt - 1))
                idx -= 1
                BM_F!(tmpN, tmpNN, BM, model, s, idx)
                mul!(tmpnN, view(BLs, :, :, idx + 1), BM)
                LAPACK.gerqf!(tmpnN, tau)
                LAPACK.orgrq!(tmpnN, tau)
                copyto!(view(BLs, :, :, idx), tmpnN)

                # copyto!(tmpNN , G)

                get_G!(tmpnn, tmpnN, ipiv, view(BLs, :, :, idx), view(BRs, :, :, idx), G)

                # #------------------------------------------------------------------#
                # axpy!(-1.0, G, tmpNN)  
                # if norm(tmpNN)>1e-7
                #     println("Warning for Batchsize Wrap Error : $(norm(tmpNN))")
                # end
                # #------------------------------------------------------------------#
            end
        end

        if record
            lock(PHY_UPDATE_LOCK) do
                open(file, "a") do io
                    writedlm(io, PHY_RECORD' ./ counter, ',')
                end
            end
            PHY_RECORD = zeros(5)
            counter = 0
        end
    end
    if record
        TTT = round(Int, (time_ns() - TTT) / 1e9)
        hour = TTT ÷ 3600
        minite = (TTT % 3600) ÷ 60
        second = TTT % 60
        println("      acc = ", round(100 * UPD.acc / prod(size(s)) / Sweeps / 2, digits=2), "%", "  $(Sweeps) Sweep finished in ", string(lpad(string(hour), 2, '0'), ":", lpad(string(minite), 2, '0'), ":", lpad(string(second), 2, '0')))
    end
    return s
end

function UpdatePhyLayer!(rng, j, s, lt, model::SO3_Hubbard_Para_, UPD::UpdateBuffer_, Phy::PhyBuffer_)
    for i in axes(s, 1)
        x, y = model.bondidx[i, j]
        UPD.subidx .= [x, y]
        sx = rand(rng, model.samplers_dict[s[i]])
        p = get_r!(UPD, model.αη[lt, sx] - model.αη[lt, s[i]], Phy.G)
        p *= model.γ[sx] / model.γ[s[i]]
        if rand(rng) < p
            UPD.acc += 1
            Gupdate!(Phy, UPD)
            s[i] = sx
        end
    end
end

function Correlation_Cal(G, i, j, k, l)
    """
    calculate the correlation <c†_i c_j c†_k c_l> = <c†_i c_j><c†_k c_l> + <c†_i c_l><c_j c†_k>
    G_ij = c_i c†_j = δ_ij - c†_j c_i
    """
    ans = (Int(i == j) - G[j, i]) * (Int(k == l) - G[l, k]) + (Int(i == l) - G[l, i]) * G[j, k]
    return 2 * real(ans)
end


function phy_measure(model::SO3_Hubbard_Para_, Phy::PhyBuffer_, lt, s)
    """
    (Ek,Ev,R0,R1)
    """
    G0 = Phy.G[:, :]
    tmpN = Phy.N
    tmpNN = Phy.NN

    # 使用来自 phy_update 的预计算指数值
    exp_αη_neg, exp_αη_pos = model.exp_αη_neg, model.exp_αη_pos

    if lt > model.Nt / 2
        for t in lt:-1:div(model.Nt, 2)+1
            for j in axes(s, 2)
                fill!(tmpN, 1.0)
                for i in axes(s, 1)
                    x, y = model.bondidx[i, j]
                    tmpN[x] = exp_αη_pos[t, s[i, j, t]]  # 预计算的 exp(α*η)
                    tmpN[y] = exp_αη_neg[t, s[i, j, t]]  # 预计算的 exp(-α*η)
                end

                WrapV!(tmpNN, G0, tmpN, view(model.UV, :, :, j), "B")
            end
            mul!(tmpNN, model.eKinv, G0)
            mul!(G0, tmpNN, model.eK)
            # G0= model.eKinv*G0*model.eK
        end
    else
        for t in lt+1:div(model.Nt, 2)
            mul!(tmpNN, G0, model.eKinv)
            mul!(G0, model.eK, tmpNN)
            # G0=model.eK*G0*model.eKinv

            for j in reverse(axes(s, 2))
                fill!(tmpN, 1.0)
                for i in axes(s, 1)
                    x, y = model.bondidx[i, j]
                    tmpN[x] = exp_αη_neg[t, s[i, j, t]]  # 预计算的 exp(-α*η)
                    tmpN[y] = exp_αη_pos[t, s[i, j, t]]  # 预计算的 exp(α*η)
                end
                WrapV!(tmpNN, G0, tmpN, view(model.UV, :, :, j), "B")
                # G0=model.UV[j,:,:]'*diagm(exp.(E))*model.UV[j,:,:] *G0* model.UV[j,:,:]'*diagm(exp.(-E))*model.UV[j,:,:]
            end
        end
    end
    #####################################################################
    # if norm(G0 - Gτ(model, s, div(model.Nt, 2))) > 1e-7
    #     error("record error lt=$(lt) : $(norm(G0-Gτ(model,s,div(model.Nt,2))))")
    # end
    #####################################################################
    # mul!(tmpNN, model.HalfeK, G0)
    # mul!(G0, tmpNN, model.HalfeKinv)
    # G0=model.HalfeK* G0 *model.HalfeKinv

    Ek = 2 * model.Ht * real(sum(model.K .* G0))
    EJ = 0.0
    # for k in 1:length(model.bondidx)
    #     x, y = model.bondidx[k]
    #     Ev += (1 - G0[x, x]) * (1 - G0[y, y]) - G0[x, y] * G0[y, x]
    # end
    # Ek = real(Ek)
    # Ev = real(Ev)

    R0so3 = R1so3 = R0u1 = R1u1 = 0.0

    if occursin("HoneyComb", model.Lattice) || model.Lattice == "SQUARE90"
        for rx in 1:model.site[1]
            for ry in 1:model.site[2]
                tmp1 = tmp2 = 0
                for ix in 1:model.site[1]
                    for iy in 1:model.site[2]
                        # SO3 order parameter
                        for zi in 1:2
                            for zj in 1:2
                                # <sx ⋅ sx>
                                # i -> iy, j -> iz, k -> jy, l -> jz 
                                i = xyzσTidx(model.Lattice, model.site, ix, iy, zi, 2)
                                j = xyzσTidx(model.Lattice, model.site, ix, iy, zi, 3)
                                k = xyzσTidx(model.Lattice, model.site, mod1(ix + rx, model.site[1]), mod1(iy + ry, model.site[2]), zj, 2)
                                l = xyzσTidx(model.Lattice, model.site, mod1(ix + rx, model.site[1]), mod1(iy + ry, model.site[2]), zj, 3)
                                tmp1 -= (-1)^(zi + zj) * Correlation_Cal(G0, i, j, k, l)
                                tmp1 -= (-1)^(zi + zj) * Correlation_Cal(G0, j, i, l, k)
                                tmp1 += (-1)^(zi + zj) * Correlation_Cal(G0, i, j, l, k)
                                tmp1 += (-1)^(zi + zj) * Correlation_Cal(G0, j, i, k, l)

                                # <sy ⋅ sy>
                                i = xyzσTidx(model.Lattice, model.site, ix, iy, zi, 1)
                                j = xyzσTidx(model.Lattice, model.site, ix, iy, zi, 3)
                                k = xyzσTidx(model.Lattice, model.site, mod1(ix + rx, model.site[1]), mod1(iy + ry, model.site[2]), zj, 1)
                                l = xyzσTidx(model.Lattice, model.site, mod1(ix + rx, model.site[1]), mod1(iy + ry, model.site[2]), zj, 3)
                                tmp1 -= (-1)^(zi + zj) * Correlation_Cal(G0, i, j, k, l)
                                tmp1 -= (-1)^(zi + zj) * Correlation_Cal(G0, j, i, l, k)
                                tmp1 += (-1)^(zi + zj) * Correlation_Cal(G0, i, j, l, k)
                                tmp1 += (-1)^(zi + zj) * Correlation_Cal(G0, j, i, k, l)

                                # <sz ⋅ sz>
                                i = xyzσTidx(model.Lattice, model.site, ix, iy, zi, 1)
                                j = xyzσTidx(model.Lattice, model.site, ix, iy, zi, 2)
                                k = xyzσTidx(model.Lattice, model.site, mod1(ix + rx, model.site[1]), mod1(iy + ry, model.site[2]), zj, 1)
                                l = xyzσTidx(model.Lattice, model.site, mod1(ix + rx, model.site[1]), mod1(iy + ry, model.site[2]), zj, 2)
                                tmp1 -= (-1)^(zi + zj) * Correlation_Cal(G0, i, j, k, l)
                                tmp1 -= (-1)^(zi + zj) * Correlation_Cal(G0, j, i, l, k)
                                tmp1 += (-1)^(zi + zj) * Correlation_Cal(G0, i, j, l, k)
                                tmp1 += (-1)^(zi + zj) * Correlation_Cal(G0, j, i, k, l)

                                # U(1) order parameter
                                for σ in 1:3
                                    i = xyzσTidx(model.Lattice, model.site, ix, iy, zi, σ)
                                    j = xyzσTidx(model.Lattice, model.site, mod1(ix + rx, model.site[1]), mod1(iy + ry, model.site[2]), zj, σ)
                                    tmp2 += (-1)^(zi + zj) * adjoint(G0[i, j]) * (Int(i == j) - G0[j, i])
                                end
                            end
                        end
                    end
                end
                R0so3 += tmp1
                R1so3 += cos(2 * π / model.site[1] * rx + 2 * π / model.site[2] * ry) * tmp1
                R0u1 += tmp2
                R1u1 += cos(2 * π / model.site[1] * rx + 2 * π / model.site[2] * ry) * tmp2
            end
        end
        R0so3 /= 4 * prod(model.site)^2
        R1so3 /= 4 * prod(model.site)^2
        R0u1 /= 4 * prod(model.site)^2
        R1u1 /= 4 * prod(model.site)^2
        @assert abs(imag(R0u1)) < 1e-10 "R0u1 should be real, but got $(R0u1)"
        @assert abs(imag(R1u1)) < 1e-10 "R1u1 should be real, but got $(R1u1)"
    else
        error("Measurement for Lattice $(model.Lattice) not implemented yet!")
    end
    return Ek, R0so3, R1so3, real(R0u1), real(R1u1)
end