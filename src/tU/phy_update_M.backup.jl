# 后选择的测量

# 基本没改，只需要添加一下测量的初始化
function phy_update_M(path::String, model::tU_Hubbard_Para_, Eidx::Vector{Int64}, beta::Float64, s::Array{UInt8,2}, Sweeps::Int64, record::Bool=false)
    global LOCK = ReentrantLock()
    TTT = time_ns()
    ERROR = 1e-6

    NN = length(model.nodes)
    Θidx = div(NN, 2) + 1
    Phy = PhyBuffer(model.Ns, NN)
    UPD = UpdateBuffer()

    name = name_Lattice(model.Lattice)

    if model.Θquench == 0.0
        file = "$(path)/tUMphy$(name)_t$(model.Ht)U$(model.Hu1)size$(model.site)Δt$(model.Δt)Θ$(model.Θrelax)BS$(model.BatchSize).csv"
    else
        file = "$(path)/tUMphy$(name)_t$(model.Ht)U$(model.Hu1)_$(model.Hu2)size$(model.site)Δt$(model.Δt)Θ$(model.Θrelax)_$(model.Θquench)BS$(model.BatchSize).csv"
    end

    Ns = model.Ns
    ns = div(Ns, 2)
    rng = MersenneTwister(Threads.threadid() + time_ns())

    G, tau, ipiv, BLs, BRs, tmpN, tmpNN, BM, tmpNn, tmpnn, tmpnN =
        Phy.G, Phy.tau, Phy.ipiv, Phy.BLs, Phy.BRs, Phy.N, Phy.NN, Phy.BM, Phy.Nn, Phy.nn, Phy.nN

    Ek = Eu = SDW0 = SDW1 = 0.0
    counter = 0

    BRs[:, :, 1] .= model.HalfeKinv * model.Pt
    BLs[:, :, NN] .= model.Pt' * model.HalfeK
    # BRs[:, :, 1] .= model.Pt
    # BLs[:, :, NN] .= model.Pt'

    Mdiag = ones(Ns)
    Mdiag[Eidx] .= exp(-2 * beta)
    M = MeasureG(Mdiag, eltype(G))

    for idx in NN-1:-1:1
        BM_F!(tmpN, tmpNN, BM, model, s, idx)
        mul!(tmpnN, view(BLs, :, :, idx + 1), BM)
        LAPACK.gerqf!(tmpnN, tau)
        LAPACK.orgrq!(tmpnN, tau, ns)
        copyto!(view(BLs, :, :, idx), tmpnN)
    end

    idx = 1
    get_G!(tmpnn, tmpnN, ipiv, view(BLs, :, :, idx), view(BRs, :, :, idx), G)
    for _ in 1:Sweeps
        # println("\n Sweep: $loop ")
        for lt in 1:model.Nt
            #####################################################################
            # # println("lt=",lt-1)
            # if norm(G - Gτ(model, s, lt - 1)) > ERROR
            #     error(lt - 1, "Wrap error:  ", norm(G - Gτ(model, s, lt - 1)))
            # end
            #####################################################################

            @inbounds @simd for iii in 1:Ns
                tmpN[iii] = @fastmath model.exp_αη_pos[lt, s[iii, lt]]
            end
            WrapKV!(tmpNN, model.eK, model.eKinv, tmpN, G, "Forward", "B")

            UpdatePhyLayer!(rng, view(s, :, lt), lt, model, UPD, Phy)
            # ---------------------------------------------------------------------------------------------------------
            # record physical quantities
            if record && 0 <= (Θidx - idx) <= 1
                tmp = phy_measure_M(model, M, lt, s, G, tmpNN, tmpN)
                counter += 1
                Ek += tmp[1]
                Eu += tmp[2]
                SDW0 += tmp[3]
                SDW1 += tmp[4]
            end
            # ---------------------------------------------------------------------------------------------------------

            if any(model.nodes .== lt)
                idx += 1
                BM_F!(tmpN, tmpNN, BM, model, s, idx - 1)
                mul!(tmpNn, BM, view(BRs, :, :, idx - 1))
                LAPACK.geqrf!(tmpNn, tau)
                LAPACK.orgqr!(tmpNn, tau, ns)
                copyto!(view(BRs, :, :, idx), tmpNn)

                copyto!(tmpNN, G)

                get_G!(tmpnn, tmpnN, ipiv, view(BLs, :, :, idx), view(BRs, :, :, idx), G)
                #####################################################################
                # axpy!(-1.0, G, tmpNN)
                # if norm(tmpNN) > ERROR
                #     println("Warning for Batchsize Wrap Error : $(norm(tmpNN))")
                # end
                #####################################################################
            end
        end

        for lt in model.Nt:-1:1
            #####################################################################
            # # print("-")
            # if norm(G - Gτ(model, s, lt)) > ERROR
            #     error(lt, " Wrap error:  ", norm(G - Gτ(model, s, lt)))
            # end
            #####################################################################

            UpdatePhyLayer!(rng, view(s, :, lt), lt, model, UPD, Phy)
            # ---------------------------------------------------------------------------------------------------------
            # record physical quantities
            if record && 0 <= (idx - Θidx) <= 1
                tmp = phy_measure_M(model, M, lt, s, G, tmpNN, tmpN)
                counter += 1
                Ek += tmp[1]
                Eu += tmp[2]
                SDW0 += tmp[3]
                SDW1 += tmp[4]
            end
            # ---------------------------------------------------------------------------------------------------------
            @inbounds @simd for iii in 1:Ns
                tmpN[iii] = @fastmath model.exp_αη_neg[lt, s[iii, lt]]
            end
            WrapKV!(tmpNN, model.eK, model.eKinv, tmpN, G, "Backward", "B")

            if any(model.nodes .== (lt - 1))
                # println("idx=",idx," lt=",lt-1)
                idx -= 1
                BM_F!(tmpN, tmpNN, BM, model, s, idx)
                mul!(tmpnN, view(BLs, :, :, idx + 1), BM)
                LAPACK.gerqf!(tmpnN, tau)
                LAPACK.orgrq!(tmpnN, tau, ns)
                copyto!(view(BLs, :, :, idx), tmpnN)

                get_G!(tmpnn, tmpnN, ipiv, view(BLs, :, :, idx), view(BRs, :, :, idx), G)
            end
        end

        if record
            lock(LOCK) do
                open(file, "a") do io
                    writedlm(io, [Ek Eu SDW0 SDW1] ./ counter, ',')
                end
            end
            Ek = Eu = SDW0 = SDW1 = 0.0
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

# ================================================================
# 输入 G，令 G' = I - G，计算
#     MG' = M G' [I + (M² - I) G']⁻¹ M ,  M 为固定对角阵
# 并返回 I - MG'（存放在 W.MG 中）。G 只读，不会被修改。
#
# 推导 (D = M² - I 对角)：由 push-through 恒等式 G'(I + DG') = (I + G'D)G'，
#     G' (I + DG')⁻¹ = (I + G'D)⁻¹ G'
# 故  MG' = M · B⁻¹ · (G' M),   B = I + G' D
# 实现：不单独构造 G'。一次遍历 G，按列缩放（列主序，连续访存），
#       非对角部分直接取 -G[i,j]，对角线再补上 G' 里的 1，同时得到
#         B = I + G' D    （工作区；对角线补 m_j² = 1 + d_j）
#         MG ← G' M       （直接写进输出；对角线补 m_j）
#       B 做 LU (getrf!)，解 B Z = G' M (getrs!，原地覆盖 MG)
#       最后一次遍历：MG ← I - M Z（按行缩放、取负、对角线加 1），O(N²)
# 总计 1 次 getrf + 1 次 getrs ≈ 8/3 N³ flops，不显式求逆；
# 额外内存只有 B、MG (各 N×N) 和 ipiv (N)；M、D 只在构造时算一次。
# 注意：B、MG 都是工作区，返回的 W.MG 会在下一次调用时被覆盖，需要保留请先拷贝；
#       多线程时每个线程各自构造一个 MeasureG。
# ================================================================
struct MeasureG{T<:Union{Float64,ComplexF64}}
    m::Vector{T}                              # diag(M)
    d::Vector{T}                              # diag(M² - I)
    B::Matrix{T}                              # 工作区：I + G' D，被 LU 原地覆盖
    MG::Matrix{T}                             # 输出：I - M G' [I + (M² - I) G']⁻¹ M，每次调用都会覆盖
    ipiv::Vector{LinearAlgebra.BLAS.BlasInt}
end

function MeasureG(M::AbstractVector, ::Type{T}) where {T<:Union{Float64,ComplexF64}}
    N = length(M)
    m = T.(M)                                 # 广播总会新分配，不与调用方的数组共用内存
    return MeasureG{T}(m, m .^ 2 .- one(T), Matrix{T}(undef, N, N), Matrix{T}(undef, N, N), Vector{LinearAlgebra.BLAS.BlasInt}(undef, N))
end
# MeasureG(M::AbstractVector) = MeasureG(M, float(eltype(M)))
# MeasureG(M::AbstractMatrix, T::Type...) = MeasureG(diag(M), T...)

function Measure_G!(G::AbstractMatrix{T}, W::MeasureG{T}) where {T}
    m, d, B, MG = W.m, W.d, W.B, W.MG
    N = length(m)
    size(G) == (N, N) || throw(DimensionMismatch("G must be N×N with N = length(M)"))

    # 一次遍历 G（G' = I - G 不单独构造）：B = I + G' D,  MG = G' M
    @inbounds for j in 1:N
        dj = d[j]
        mj = m[j]
        @simd for i in 1:N
            g = -G[i, j]                          # G' = I - G：这里先取 -G[i,j]，对角线的 1 在循环外补，G 本身不改
            B[i, j] = g * dj
            MG[i, j] = g * mj
        end
        B[j, j] += m[j]^2                         # 对角：I 的 1 加上 G' 的 1·d_j，合计 1 + d_j = m_j²
        MG[j, j] += mj                            # 对角：G' 的 1·m_j
    end

    LAPACK.getrf!(B, W.ipiv)
    LAPACK.getrs!('N', B, W.ipiv, MG)           # MG ← B⁻¹ (G' M)

    @inbounds for j in 1:N
        @simd for i in 1:N
            MG[i, j] = -MG[i, j] * m[i]                     # 按行缩放并取负：MG ← -M B⁻¹ (G' M)
        end
        MG[j, j] += one(T)                        # 对角线加 1：MG ← I - M B⁻¹ (G' M) = I - MG'
    end
    return MG
end

# 只改了一行
function phy_measure_M(model::tU_Hubbard_Para_, M, lt, s, G, tmpNN, tmpN)
    """
    (Ek,Ev,R0,R1)    
    """
    G0 = G[:, :]
    SDW0 = SDW1 = 0.0

    if lt > model.Nt / 2
        for t in lt:-1:div(model.Nt, 2)+1
            for i in axes(s, 1)
                tmpN[i] = model.exp_αη_neg[t, s[i, t]]
            end
            WrapKV!(tmpNN, model.eK, model.eKinv, tmpN, G0, "Backward", "B")
        end
    else
        for t in lt+1:div(model.Nt, 2)
            for i in axes(s, 1)
                tmpN[i] = model.exp_αη_pos[t, s[i, t]]
            end
            WrapKV!(tmpNN, model.eK, model.eKinv, tmpN, G0, "Forward", "B")
        end
    end
    copyto!(G0, Measure_G!(G0, M))
    ####################################################################
    # if norm(G0 - Gτ(model, s, div(model.Nt, 2))) > 1e-7
    #     error("record error lt=$(lt) : $(norm(G0-Gτ(model,s,div(model.Nt,2))))")
    # end
    #####################################################################
    mul!(tmpNN, model.HalfeK, G0)
    mul!(G0, tmpNN, model.HalfeKinv)

    Ek = -2 * model.Ht * real(sum(model.K .* G0))
    Eu = 0
    for i in 1:model.Ns
        Eu += (1 - G0[i, i]) * adjoint(G0[i, i])
    end
    @assert imag(Eu) < 1e-8 "Eu get imaginary part! $(Eu)"
    Eu = real(Eu)

    # @assert sum(abs.(imag(diag(G0))))<1e-8 "G0 diag get imaginary part! $(norm(imag(diag(G0))))"

    if occursin("HoneyComb", model.Lattice)
        for rx in 1:model.site[1]
            for ry in 1:model.site[2]
                tmp = ComplexF64(0.0)
                for ix in 1:model.site[1]
                    for iy in 1:model.site[2]
                        idx1 = xy_i(model.Lattice, model.site, ix, iy) - 1
                        idx2 = xy_i(model.Lattice, model.site, mod1(ix + rx, model.site[1]), mod1(iy + ry, model.site[2])) - 1

                        tmp += Correlation_Cal(G0, idx1, idx1, idx2, idx2) + Correlation_Cal_dagger(G0, idx1, idx1, idx2, idx2)
                        tmp += Correlation_Cal(G0, idx1 + 1, idx1 + 1, idx2 + 1, idx2 + 1) + Correlation_Cal_dagger(G0, idx1 + 1, idx1 + 1, idx2 + 1, idx2 + 1)
                        tmp -= Correlation_Cal(G0, idx1 + 1, idx1 + 1, idx2, idx2) + Correlation_Cal_dagger(G0, idx1 + 1, idx1 + 1, idx2, idx2)
                        tmp -= Correlation_Cal(G0, idx1, idx1, idx2 + 1, idx2 + 1) + Correlation_Cal_dagger(G0, idx1, idx1, idx2 + 1, idx2 + 1)
                        tmp -= (1 - G0[idx1, idx1]) * adjoint(G0[idx2, idx2]) + (1 - G0[idx2, idx2]) * adjoint(G0[idx1, idx1])
                        tmp -= (1 - G0[idx1+1, idx1+1]) * adjoint(G0[idx2+1, idx2+1]) + (1 - G0[idx2+1, idx2+1]) * adjoint(G0[idx1+1, idx1+1])
                        tmp += (1 - G0[idx1+1, idx1+1]) * adjoint(G0[idx2, idx2]) + (1 - G0[idx2, idx2]) * adjoint(G0[idx1+1, idx1+1])
                        tmp += (1 - G0[idx1, idx1]) * adjoint(G0[idx2+1, idx2+1]) + (1 - G0[idx2+1, idx2+1]) * adjoint(G0[idx1, idx1])

                    end
                end
                @assert abs(imag(tmp)) < 1e-6 "record error for imag part of order parameter: $tmp"
                SDW0 += real(tmp)
                # SDW1 += cos(2 * π / model.site[1] * rx + 2 * π / model.site[2] * ry) * real(tmp)
                q1 = 2π * rx / model.site[1]
                q2 = 2π * ry / model.site[2]
                SDW1 += (cos(q1) + cos(q2) + cos(q1 - q2)) / 3 * real(tmp)
                # 最短的 6 个动量是 $\pm b_1/L$、$\pm b_2/L$、$\pm(b_1-b_2)/L$
                # 三个方向由晶格对称性等价，取平均不改变期望值，只降低噪声。
            end
        end

        SDW0 = SDW0 / model.Ns^2
        SDW1 = SDW1 / model.Ns^2
        return Ek, Eu, SDW0, SDW1
    elseif model.Lattice == "SQUARE"
        for rx in 1:model.site[1]
            for ry in 1:model.site[2]
                tmp = 0
                for ix in 1:model.site[1]
                    for iy in 1:model.site[2]
                        idx1 = ix + (iy - 1) * model.site[1]
                        idx2 = mod1(rx + ix, model.site[1]) + mod((ry + iy - 1), model.site[2]) * model.site[1]
                        tmp += (1 - G0[idx1, idx1]) * (1 - G0[idx2, idx2]) - G0[idx1, idx2] * G0[idx2, idx1]
                    end
                end
                tmp /= prod(model.site)
                R0 += tmp * cos(π * (rx + ry))
                R1 += cos(π * (rx + ry) + 2 * π / model.site[1] * rx + 2 * π / model.site[2] * ry) * tmp
            end
        end
    elseif model.Lattice == "triangular90"
        R0cdw = R1cdw = R0sc = R1sc = 0.0
        for rx in 1:model.site[1]
            for ry in 1:model.site[2]
                tmp1 = tmp2 = 0
                for ix in 1:model.site[1]
                    for iy in 1:model.site[2]
                        for zi in 0:1
                            for zj in 0:1
                                idx1 = ix + (iy - 1) * model.site[1] + zi
                                idx2 = mod1(rx + ix, model.site[1]) + mod((ry + iy - 1), model.site[2]) * model.site[1] + zj
                                tmp1 += (-1)^(zi + zj) * 2 * real(Correlation_Cal(G0, idx1, idx1, idx2, idx2))    # <n_i n_j>
                                tmp1 += (-1)^(zi + zj) * 2 * real((1 - G0[idx1, idx1]) * (1 - adjoint(G0[idx2, idx2])))    #<n_i><n_j>
                                delta = idx1 == idx2 ? 1 : 0
                                tmp2 += abs2(delta - G0[idx1, idx2]) + abs2(G0[idx1, idx2])     # <c_i c†_j>
                            end
                        end
                    end
                end
                tmp1 /= prod(model.site)
                tmp2 /= prod(model.site)
                R0cdw += tmp1
                R1cdw += cos(2 * π / model.site[1] * rx + 2 * π / model.site[2] * ry) * tmp1
                R0sc += tmp2
                R1sc += cos(2 * π / model.site[1] * rx + 2 * π / model.site[2] * ry) * tmp2
            end
        end
        return Ek, Eu, R0cdw, R1cdw, R0sc, R1sc
    elseif model.Lattice == "triangular45"
        R0cdw = R1cdw = R0sc = R1sc = 0.0
        for rx in 1:model.site[1]
            for ry in 1:model.site[2]
                tmp1 = tmp2 = 0
                for ix in 1:model.site[1]
                    for iy in 1:model.site[2]
                        idx1 = ix + (iy - 1) * model.site[1]
                        idx2 = mod1(rx + ix, model.site[1]) + mod((ry + iy - 1), model.site[2]) * model.site[1]
                        tmp1 += (-1)^(rx + ry) * 2 * real(Correlation_Cal(G0, idx1, idx1, idx2, idx2))    # <n_i n_j>
                        tmp1 += (-1)^(rx + ry) * 2 * real((1 - G0[idx1, idx1]) * (1 - adjoint(G0[idx2, idx2])))    #<n_i><n_j>
                        delta = idx1 == idx2 ? 1 : 0
                        tmp2 += abs2(delta - G0[idx1, idx2]) + abs2(G0[idx1, idx2])     # <c_i c†_j>
                    end
                end
                tmp1 /= prod(model.site)
                tmp2 /= prod(model.site)
                R0cdw += tmp1
                R1cdw += cos(2 * π / model.site[1] * rx + 2 * π / model.site[2] * ry) * tmp1
                R0sc += tmp2
                R1sc += cos(2 * π / model.site[1] * rx + 2 * π / model.site[2] * ry) * tmp2
            end
        end
        return Ek, Eu, R0cdw, R1cdw, R0sc, R1sc
    end
end

