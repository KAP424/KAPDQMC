# turn off symmetric HS decomposition when debuging

function phy_update(path::String, model::M_VBS_Hubbard_Para_, s::Array{UInt8,3}, Sweeps::Int64, record::Bool)
    TTT = time_ns()
    global LOCK = ReentrantLock()
    ERROR = 1e-6

    UPD = UpdateBuffer()
    NN = length(model.nodes)
    Phy = PhyBuffer(model.Ns, NN)
    Θidx = div(NN, 2) + 1

    name = name_Lattice(model.Lattice)

    if model.HJ2 == model.HJ1
        file = "$(path)/VBS$(model.SUN)phy$(name)_t$(model.Ht)V$(model.HJ1)size$(model.site)Δt$(model.Δt)Θ$(model.Θrelax)BS$(model.BatchSize).csv"
    else
        file = "$(path)/VBS$(model.SUN)phy$(name)_t$(model.Ht)V$(model.HJ1)_$(model.HJ2)size$(model.site)Δt$(model.Δt)Θ$(model.Θrelax)_$(model.Θquench)BS$(model.BatchSize).csv"
    end

    rng = MersenneTwister(Threads.threadid() + time_ns())

    Ek = Ev = 0.0
    R0 = 0.0
    R1 = 0.0
    counter = 0

    G, BLs, BRs, tmpN, tmpNN, tmpnn, tmpnN, tmpNn, tau, ipiv, BM =
        Phy.G, Phy.BLs, Phy.BRs, Phy.N, Phy.NN, Phy.nn, Phy.nN, Phy.Nn, Phy.tau, Phy.ipiv, Phy.BM

    BRs[:, :, 1] .= model.HalfeKinv * model.Pt
    BLs[:, :, NN] .= model.Pt' * model.HalfeK

    # BRs[:, :, 1] .= model.Pt
    # BLs[:, :, NN] .= model.Pt'

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
            # # println(lt)
            # if norm(G - Gτ(model, s, lt - 1)) / norm(G) > ERROR
            #     error("Wrap-$(lt)   :   $(norm(G-Gτ(model,s,lt-1))) , $(norm(G)) , $(norm(Gτ(model,s,lt-1))) ")
            # end
            #####################################################################

            mul!(tmpNN, G, model.eKinv)
            mul!(G, model.eK, tmpNN)
            # G=model.eK*G*model.eKinv

            for j in reverse(axes(s, 2))
                for i in axes(s, 1)
                    x, y = model.nnidx[i, j]
                    tmpN[x] = model.exp_αη_pos[lt, s[i, j, lt]]
                    tmpN[y] = model.exp_αη_neg[lt, s[i, j, lt]]
                end
                WrapV!(tmpNN, G, tmpN, view(model.UV, :, :, j), "B")

                UpdatePhyLayer!(rng, j, view(s, :, j, lt), lt, model, UPD, Phy)
                ####################################################################
                # # print("*")
                # GG = model.eK * Gτ(model, s, lt - 1) * model.eKinv
                # for jj in size(model.nnidx, 2):-1:j
                #     E = zeros(model.Ns)
                #     for ii in 1:size(s)[1]
                #         x, y = model.nnidx[ii, jj]
                #         E[x] = model.exp_αη_pos[lt, s[ii, jj, lt]]
                #         E[y] = model.exp_αη_neg[lt, s[ii, jj, lt]]
                #     end
                #     GG = model.UV[:, :, jj] * Diagonal(E) * model.UV[:, :, jj]' * GG * model.UV[:, :, jj] * Diagonal(1.0 ./ E) * model.UV[:, :, jj]'
                # end
                # if (norm(G - GG) > ERROR)
                #     println("lt=$(lt) j=$(j)")
                #     error(j, " update error: ", norm(G - GG), "  lt=", lt)
                # end
                ####################################################################
            end

            if record && abs(idx - Θidx) <= 1
                tmp = phy_measure(model, Phy, lt, s)
                Ek += tmp[1]
                Ev += tmp[2]
                R0 += tmp[3]
                R1 += tmp[4]
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
            # if norm(G - Gτ(model, s, lt)) > ERROR
            #     error("Wrap-$(lt)   :   $(norm(G-Gτ(model,s,lt-1))) , $(norm(G)) , $(norm(Gτ(model,s,lt-1))) ")
            # end
            ######################################################################
            for j in axes(s, 2)
                UpdatePhyLayer!(rng, j, view(s, :, j, lt), lt, model, UPD, Phy)
                for i in axes(s, 1)
                    x, y = model.nnidx[i, j]
                    tmpN[x] = model.exp_αη_neg[lt, s[i, j, lt]]
                    tmpN[y] = model.exp_αη_pos[lt, s[i, j, lt]]
                end
                WrapV!(tmpNN, G, tmpN, view(model.UV, :, :, j), "B")
            end
            mul!(tmpNN, model.eKinv, G)
            mul!(G, tmpNN, model.eK)

            if record && abs(idx - Θidx) <= 1
                tmp = phy_measure(model, Phy, lt - 1, s)
                Ek += tmp[1]
                Ev += tmp[2]
                R0 += tmp[3]
                R1 += tmp[4]
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
            lock(LOCK) do
                open(file, "a") do io
                    writedlm(io, [Ek, Ev, R0, R1]' ./ counter, ',')
                end
            end
            Ek = Ev = 0.0
            R0 = R1 = 0.0
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

function UpdatePhyLayer!(rng, j, s, lt, model::M_VBS_Hubbard_Para_, UPD::UpdateBuffer_, Phy::PhyBuffer_)
    for i in axes(s, 1)
        x, y = model.nnidx[i, j]
        UPD.subidx .= [x, y]
        sx = rand(rng, model.samplers_dict[s[i]])
        p = get_r!(UPD, model.αη[lt, sx] - model.αη[lt, s[i]], Phy.G)^model.SUN
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
    return 2 * real((Int(i == j) - G[j, i]) * (Int(k == l) - G[l, k]) + (Int(i == l) - G[l, i]) * G[j, k])
end

function phy_measure(model::M_VBS_Hubbard_Para_, Phy::PhyBuffer_, lt, s)
    """
    (Ek,Ev,R0,R1)    
    """
    G0 = Phy.G[:, :]
    tmpN = Phy.N
    tmpNN = Phy.NN
    tmp = zeros(ComplexF64, 4)

    if lt > model.Nt / 2
        for t in lt:-1:div(model.Nt, 2)+1
            for j in axes(s, 2)
                for i in axes(s, 1)
                    x, y = model.nnidx[i, j]
                    tmpN[x] = model.exp_αη_neg[t, s[i, j, t]]
                    tmpN[y] = model.exp_αη_pos[t, s[i, j, t]]
                end

                WrapV!(tmpNN, G0, tmpN, view(model.UV, :, :, j), "B")
                # G0=model.UV[j,:,:]'*diagm(exp.(-E))*model.UV[j,:,:] *G0* model.UV[j,:,:]'*diagm(exp.(E))*model.UV[j,:,:]
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
                for i in axes(s, 1)
                    x, y = model.nnidx[i, j]
                    tmpN[x] = model.exp_αη_pos[t, s[i, j, t]]
                    tmpN[y] = model.exp_αη_neg[t, s[i, j, t]]
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
    mul!(tmpNN, model.HalfeK, G0)
    mul!(G0, tmpNN, model.HalfeKinv)
    # G0=model.HalfeK* G0 *model.HalfeKinv

    Ek = -imag(model.Ht * sum(model.K .* G0))
    Ev = 0.0

    for i in 1:length(model.nnidx)
        x1, y1 = model.nnidx[i]
        for j in 1:length(model.nnidx)
            x2, y2 = model.nnidx[j]
            Ev -= model.SUN * Correlation_Cal(G0, x1, y1, x2, y2)
            Ev += model.SUN * Correlation_Cal(G0, x1, y1, y2, x2)
            Ev += model.SUN * Correlation_Cal(G0, y1, x1, x2, y2)
            Ev -= model.SUN * Correlation_Cal(G0, y1, x1, y2, x2)

            Ev += model.SUN * 2 * real(G0[x1, y1] * adjoint(G0[x2, y2]))
            Ev -= model.SUN * 2 * real(G0[y1, x1] * adjoint(G0[x2, y2]))
            Ev -= model.SUN * 2 * real(G0[x1, y1] * adjoint(G0[y2, x2]))
            Ev += model.SUN * 2 * real(G0[y1, x1] * adjoint(G0[y2, x2]))

            Ev += model.SUN * (model.SUN - 1) * 2 * imag(G0[x1, y1]) * 2 * imag(G0[x2, y2])
            Ev -= model.SUN * (model.SUN - 1) * 2 * imag(G0[y1, x1]) * 2 * imag(G0[x2, y2])
            Ev -= model.SUN * (model.SUN - 1) * 2 * imag(G0[x1, y1]) * 2 * imag(G0[y2, x2])
            Ev += model.SUN * (model.SUN - 1) * 2 * imag(G0[y1, x1]) * 2 * imag(G0[y2, x2])
        end
    end

    Ev *= -model.HJ2 / 8 / model.SUN

    R0 = R1 = 0.0
    if occursin("HoneyComb", model.Lattice)
        for rx in 0:model.site[1]-1
            for ry in 0:model.site[2]-1
                tmp = 0.0
                for ix in 1:model.site[1]
                    for iy in 1:model.site[2]
                        idx1 = xy_i(model.Lattice, model.site, ix, iy) - 1
                        idx2 = xy_i(model.Lattice, model.site, mod1(ix + rx, model.site[1]), mod1(iy + ry, model.site[2])) - 1

                        nn1 = nn2idx(model.Lattice, model.site, idx1)
                        nn2 = nn2idx(model.Lattice, model.site, idx2)

                        for iδ in eachindex(nn1)
                            tmp -= model.SUN * Correlation_Cal(G0, idx1, nn1[iδ], idx2, nn2[iδ])
                            tmp += model.SUN * Correlation_Cal(G0, nn1[iδ], idx1, idx2, nn2[iδ])
                            tmp += model.SUN * Correlation_Cal(G0, idx1, nn1[iδ], nn2[iδ], idx2)
                            tmp -= model.SUN * Correlation_Cal(G0, nn1[iδ], idx1, nn2[iδ], idx2)

                            tmp += model.SUN * 2 * real(G0[idx1, nn1[iδ]] * adjoint(G0[idx2, nn2[iδ]]))
                            tmp -= model.SUN * 2 * real(G0[nn1[iδ], idx1] * adjoint(G0[idx2, nn2[iδ]]))
                            tmp -= model.SUN * 2 * real(G0[idx1, nn1[iδ]] * adjoint(G0[nn2[iδ], idx2]))
                            tmp += model.SUN * 2 * real(G0[nn1[iδ], idx1] * adjoint(G0[nn2[iδ], idx2]))

                            tmp += model.SUN * (model.SUN - 1) * 2 * imag(G0[idx1, nn1[iδ]]) * 2 * imag(G0[idx2, nn2[iδ]])
                            tmp -= model.SUN * (model.SUN - 1) * 2 * imag(G0[nn1[iδ], idx1]) * 2 * imag(G0[idx2, nn2[iδ]])
                            tmp -= model.SUN * (model.SUN - 1) * 2 * imag(G0[idx1, nn1[iδ]]) * 2 * imag(G0[nn2[iδ], idx2])
                            tmp += model.SUN * (model.SUN - 1) * 2 * imag(G0[nn1[iδ], idx1]) * 2 * imag(G0[nn2[iδ], idx2])

                        end
                    end
                end

                R0 += cos(2π * (rx / 3 + ry / 3)) * tmp / 2
                R1 += cos(2π * (rx / 3 + ry / 3 + rx / model.site[1] + ry / model.site[2])) * tmp / 2
            end
        end
        R0 /= model.Ns
        R1 /= model.Ns
    elseif model.Lattice == "SQUARE"
        error("Correlation measurement not implemented for SQUARE lattice yet!")
        # for rx in 1:model.site[1]
        #     for ry in 1:model.site[2]
        #         tmp = 0
        #         for ix in 1:model.site[1]
        #             for iy in 1:model.site[2]
        #                 idx1 = ix + (iy - 1) * model.site[1]
        #                 idx2 = mod1(rx + ix, model.site[1]) + mod((ry + iy - 1), model.site[2]) * model.site[1]
        #                 tmp += (1 - G0[idx1, idx1]) * (1 - G0[idx2, idx2]) - G0[idx1, idx2] * G0[idx2, idx1]
        #             end
        #         end
        #         tmp /= prod(model.site)
        #         R0 += tmp * cos(π * (rx + ry))
        #         R1 += cos(π * (rx + ry) + 2 * π / model.site[1] * rx + 2 * π / model.site[2] * ry) * tmp
        #     end
        # end
    end
    # 1-R1/R0
    return Ek, Ev, R0, R1
end
