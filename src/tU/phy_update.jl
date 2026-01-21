

function phy_update(path::String, model::tU_Hubbard_Para_, s::Array{UInt8,2}, Sweeps::Int64, record::Bool=false)
    global LOCK = ReentrantLock()
    ERROR = 1e-6

    NN = length(model.nodes)
    Θidx = div(NN, 2) + 1
    Phy = PhyBuffer(model.Ns, NN)
    UPD = UpdateBuffer()

    name = if model.Lattice == "SQUARE90"
        "□90"
    elseif model.Lattice == "SQUARE45"
        "□45"
    elseif model.Lattice == "HoneyComb60"
        "HC"
    elseif model.Lattice == "HoneyComb120"
        "HC120"
    else
        error("Lattice: $(model.Lattice) is not allowed !")
    end
    if model.Θquench == 0.0
        file = "$(path)/tUphy$(name)_t$(model.Ht)U$(model.Hu1)size$(model.site)Δt$(model.Δt)Θ$(model.Θrelax)BS$(model.BatchSize).csv"
    else
        file = "$(path)/tUphy$(name)_t$(model.Ht)U$(model.Hu1)_$(model.Hu2)size$(model.site)Δt$(model.Δt)Θ$(model.Θrelax)_$(model.Θquench)BS$(model.BatchSize).csv"
    end

    Ns = model.Ns
    ns = div(Ns, 2)
    rng = MersenneTwister(Threads.threadid() + time_ns())

    G, tau, ipiv, BLs, BRs, tmpN, tmpNN, BM, tmpNn, tmpnn, tmpnN =
        Phy.G, Phy.tau, Phy.ipiv, Phy.BLs, Phy.BRs, Phy.N, Phy.NN, Phy.BM, Phy.Nn, Phy.nn, Phy.nN

    Ek = Eu = CDW0 = CDW1 = SDW0 = SDW1 = 0.0
    counter = 0

    BRs[:, :, 1] .= model.Pt
    BLs[:, :, NN] .= model.Pt'
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
                tmpN[iii] = @fastmath cis(model.α[lt] * model.η[s[iii, lt]])
            end
            WrapKV!(tmpNN, model.eK, model.eKinv, tmpN, G, "Forward", "B")

            UpdatePhyLayer!(rng, view(s, :, lt), lt, model, UPD, Phy)
            # ---------------------------------------------------------------------------------------------------------
            # record physical quantities
            if record && 0 <= (Θidx - idx) <= 1
                tmp = phy_measure(model, lt, s, G, tmpNN, tmpN)
                counter += 1
                Ek += tmp[1]
                Eu += tmp[2]
                CDW0 += tmp[3]
                CDW1 += tmp[4]
                SDW0 += tmp[5]
                SDW1 += tmp[6]
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
                tmp = phy_measure(model, lt, s, G, tmpNN, tmpN)
                counter += 1
                Ek += tmp[1]
                Eu += tmp[2]
                CDW0 += tmp[3]
                CDW1 += tmp[4]
                SDW0 += tmp[5]
                SDW1 += tmp[6]
            end
            # ---------------------------------------------------------------------------------------------------------
            @inbounds @simd for iii in 1:Ns
                tmpN[iii] = @fastmath cis(-model.α[lt] * model.η[s[iii, lt]])
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
                    writedlm(io, [Ek Eu CDW0 CDW1 SDW0 SDW1] ./ counter, ',')
                end
            end
            Ek = Eu = CDW0 = CDW1 = SDW0 = SDW1 = 0.0
            counter = 0
        end
    end
    return s
end

function phy_measure(model::tU_Hubbard_Para_, lt, s, G, tmpNN, tmpN)
    """
    (Ek,Ev,R0,R1)    
    """
    G0 = G[:, :]
    CDW0 = CDW1 = SDW0 = SDW1 = 0.0

    if lt > model.Nt / 2
        for t in lt:-1:div(model.Nt, 2)+1
            for i in axes(s, 1)
                tmpN[i] = cis(-model.α[t] * model.η[s[i, t]])
            end
            WrapKV!(tmpNN, model.eK, model.eKinv, tmpN, G0, "Backward", "B")
        end
    else
        for t in lt+1:div(model.Nt, 2)
            for i in axes(s, 1)
                tmpN[i] = cis(model.α[t] * model.η[s[i, t]])
            end
            WrapKV!(tmpNN, model.eK, model.eKinv, tmpN, G0, "Forward", "B")
        end
    end
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
                tmp1 = 0.0     #<up up> <down down>  
                tmp2 = 0.0     #<up down> <down up>
                for ix in 1:model.site[1]
                    for iy in 1:model.site[2]
                        idx1 = xy_i(model.site, ix, iy) - 1
                        idx2 = xy_i(model.site, mod1(ix + rx, model.site[1]), mod1(iy + ry, model.site[2])) - 1

                        delta = idx1 == idx2 ? 1 : 0

                        tmp1 += ((1 - G0[idx1, idx1]) * (1 - G0[idx2, idx2]) + (delta - G0[idx1, idx2]) * G0[idx2, idx1])
                        +adjoint(G0[idx1, idx1] * G0[idx2, idx2] + G0[idx1, idx2] * (delta - G0[idx2, idx1]))  # <i_A j_A>
                        tmp1 += ((1 - G0[idx1+1, idx1+1]) * (1 - G0[idx2+1, idx2+1]) + (delta - G0[idx1+1, idx2+1]) * G0[idx2+1, idx1+1])
                        +adjoint(G0[idx1+1, idx1+1] * G0[idx2+1, idx2+1] + G0[idx1+1, idx2+1] * (delta - G0[idx2+1, idx1+1]))  # <i_B j_B>
                        tmp1 -= ((1 - G0[idx1+1, idx1+1]) * (1 - G0[idx2, idx2]) - G0[idx1+1, idx2] * G0[idx2, idx1+1])
                        +adjoint(G0[idx1+1, idx1+1] * G0[idx2, idx2] - G0[idx1+1, idx2] * G0[idx2, idx1+1])  # <i_B j_A>
                        tmp1 -= ((1 - G0[idx1, idx1]) * (1 - G0[idx2+1, idx2+1]) - G0[idx1, idx2+1] * G0[idx2+1, idx1])
                        +adjoint(G0[idx1, idx1] * G0[idx2+1, idx2+1] - G0[idx1, idx2+1] * G0[idx2+1, idx1])  # <i_A j_B>

                        tmp2 += (1 - G0[idx1, idx1]) * adjoint(G0[idx2, idx2])
                        +adjoint(G0[idx1, idx1]) * (1 - G0[idx2, idx2])    # <i_A j_A>
                        tmp2 += (1 - G0[idx1+1, idx1+1]) * adjoint(G0[idx2+1, idx2+1])
                        +adjoint(G0[idx1+1, idx1+1]) * (1 - G0[idx2+1, idx2+1])    # <i_B j_B>
                        tmp2 -= (1 - G0[idx1, idx1]) * adjoint(G0[idx2+1, idx2+1])
                        +adjoint(G0[idx1, idx1]) * (1 - G0[idx2+1, idx2+1])    # <i_A j_B>
                        tmp2 -= (1 - G0[idx1+1, idx1+1]) * adjoint(G0[idx2, idx2])
                        +adjoint(G0[idx1+1, idx1+1]) * (1 - G0[idx2, idx2])    # <i_B j_A>
                    end
                end
                CDW0 += tmp1 + tmp2
                CDW1 += cos(2 * π / model.site[1] * rx + 2 * π / model.site[2] * ry) * (tmp1 + tmp2)
                SDW0 += tmp1 - tmp2
                SDW1 += cos(2 * π / model.site[1] * rx + 2 * π / model.site[2] * ry) * (tmp1 - tmp2)
            end
        end

        CDW0 = real(CDW0) / model.Ns^2
        CDW1 = real(CDW1) / model.Ns^2
        SDW0 = real(SDW0) / model.Ns^2
        SDW1 = real(SDW1) / model.Ns^2
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
    end
    return Ek, Eu, CDW0, CDW1, SDW0, SDW1
end

function UpdatePhyLayer!(rng, s, lt, model::tU_Hubbard_Para_, UPD::UpdateBuffer_, Phy::PhyBuffer_)
    for i in eachindex(s)
        UPD.subidx = [i]
        sx = rand(rng, model.samplers_dict[s[i]])
        p = get_r!(UPD, model.α[lt] * (model.η[sx] - model.η[s[i]]), Phy.G)
        p *= model.γ[sx] / model.γ[s[i]]
        if rand(rng) < p
            Gupdate!(Phy, UPD)
            s[i] = sx
        end
    end
end