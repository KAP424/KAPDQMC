# Not developed !!!
# Not developed !!!
# Not developed !!!
# Not developed !!!

function ctrl_SCDOPicr(path::String, model::tV_Hubbard_Para_, alpha::Float64, indexA::Vector{Int64}, indexB::Vector{Int64}, Sweeps::Int64, λ::Float64, Nλ::Int64, s::Array{UInt8,3}, record)
    ERROR = 1e-6
    global LOCK = ReentrantLock()
    Ns = model.Ns
    ns = div(Ns, 2)
    NN = length(model.nodes)
    Θidx = div(NN, 2) + 1

    UPD = UpdateBuffer()
    SCEE = SCEEBuffer(model.Ns)
    A = DOPBuffer(alpha, indexA)
    B = DOPBuffer(alpha, indexB)
    G = G4Buffer(model.Ns, NN)

    name = if model.Lattice == "SQUARE"
        "□"
    elseif model.Lattice == "HoneyComb60"
        "HC"
    elseif model.Lattice == "HoneyComb120"
        "HC120"
    else
        error("Lattice: $(model.Lattice) is not allowed !")
    end
    file = "$(path)/tVSCDOP$(name)_t$(model.Ht)V$(model.Hv1)_$(model.Hv2)size$(model.site)Δt$(model.Δt)Θ$(model.Θrelax)_$(model.Θquench)N$(Nλ)BS$(model.BatchSize).csv"
    rng = MersenneTwister(Threads.threadid() + time_ns())

    atexit() do
        if record
            lock(LOCK) do
                open(file, "a") do io
                    writedlm(io, O', ',')
                end
            end
        end
        # writedlm("$(path)ss/SS$(name)_t$(model.Ht)U$(model.Hu)size$(model.site)Δt$(model.Δt)Θ$(model.Θ)λ$(Int(round(Nλ*λ))).csv", [ss[1] ss[2]],",")
    end

    Gt, Gt0, G0t, BLMs, BRMs, BMs, BMsinv = G.Gt, G.Gt0, G.G0t, G.BLMs, G.BRMs, G.BMs, G.BMinvs
    tmpN, tmpNN, tmpNn, tmpnN, tau = SCEE.N, SCEE.NN, SCEE.Nn, SCEE.nN, SCEE.tau


    tmpO = 0.0
    counter = 0
    O = zeros(Float64, Sweeps + 1)
    O[1] = λ

    for idx in 1:NN-1
        BM_F!(tmpN, tmpNN, view(BMs, :, :, idx), model, s, idx)
        BMinv_F!(tmpN, tmpNN, view(BMsinv, :, :, idx), model, s, idx)
    end

    BLMs[:, :, NN] .= model.Pt'
    BRMs[:, :, 1] .= model.Pt

    for i in 1:NN-1
        mul!(tmpnN, view(BLMs, :, :, NN - i + 1), view(BMs, :, :, NN - i))
        LAPACK.gerqf!(tmpnN, tau)
        LAPACK.orgrq!(tmpnN, tau, ns)
        copyto!(view(BLMs, :, :, NN - i), tmpnN)

        mul!(tmpNn, view(BMs, :, :, i), view(BRMs, :, :, i))
        LAPACK.geqrf!(tmpNn, tau)
        LAPACK.orgqr!(tmpNn, tau, ns)
        copyto!(view(BRMs, :, :, i + 1), tmpNn)
    end

    idx = 1
    get_ABX!(G, A, B, SCEE, model.nodes, idx, "Forward")
    for loop in 1:Sweeps
        # println("\n ====== Sweep $loop / $Sweeps ======")
        for lt in 1:model.Nt
            @inbounds @simd for iii in 1:Ns
                @fastmath tmpN[iii] = cis(model.α[lt] * model.η[s[iii, lt]])
            end

            WrapKV!(tmpNN, model.eK, model.eKinv, tmpN, Gt0, "Forward", "L")
            WrapKV!(tmpNN, model.eK, model.eKinv, tmpN, Gt, "Forward", "B")
            WrapKV!(tmpNN, model.eK, model.eKinv, tmpN, G0t, "Forward", "R")

            #####################################################################
            # Gt_,G0_,Gt0_,G0t_=G4(model,s,lt,div(model.Nt,2))
            # if norm(Gt-Gt_)+norm(Gt0-Gt0_)+norm(G0t-G0t_)>ERROR
            #     println( norm(Gt-Gt_),'\n',norm(Gt0-Gt0_),'\n',norm(G0t-G0t_),'\n',norm(G0-G0_) )
            #     error("WrapTime=$lt ")
            # end
            # XAinv_=cis(A.alpha)*I(length(A.index)).+(1 - cis(A.alpha)) * view(G0_,indexA[:],indexA[:])
            # detXA_=abs2(det(XAinv_))
            # XAinv_=inv(XAinv_)

            # XBinv_=cis(B.alpha)*I(length(B.index)).+(1 - cis(B.alpha)) * view(G0_,indexB[:],indexB[:])
            # detXB_=abs2(det(XBinv_))
            # XBinv_=inv(XBinv_)
            # if norm(XAinv_-A.Xinv)+norm(B.Xinv-XBinv_)+abs(A.detX-detXA_)+abs(B.detX-detXB_)>ERROR
            #     println(norm(XAinv_-A.Xinv)," ",norm(B.Xinv-XBinv_)," ",abs(A.detX-detXA_)," ",abs(B.detX-detXB_))
            #     error("s:  $lt : WrapTime")
            # end
            #####################################################################

            WrapK!(tmpNN, G, model.eK, model.eKinv)

            for j in reverse(axes(ss[1], 2))
                for i in axes(ss[1], 1)
                    x, y = model.nnidx[i, j]
                    tmpN[x] = model.α[lt] * model.η[s[i, j, lt]]
                    tmpN[y] = -model.α[lt] * model.η[s[i, j, lt]]
                end
                tmpN .= exp.(tmpN)

                WrapV!(tmpNN, Gt0, tmpN, view(model.UV, :, :, j), "L")

                WrapV!(tmpNN, Gt, tmpN, view(model.UV, :, :, j), "B")

                WrapV!(tmpNN, G0t, tmpN, view(model.UV, :, :, j), "R")

                # update
                UpdateDOPLayer!(rng, j, view(s, :, j, lt), lt, G, A, B, model, UPD, SCEE, λ)
            end

            ##------------------------------------------------------------------------
            tmpO += (A.detX / B.detX)^(1 / Nλ)
            counter += 1
            ##------------------------------------------------------------------------

            if any(model.nodes .== lt)
                idx += 1
                BM_F!(tmpN, tmpNN, view(BMs, :, :, idx - 1), model, s, idx - 1)
                BMinv_F!(tmpN, tmpNN, view(BMsinv, :, :, idx - 1), model, s, idx - 1)
                for i in idx:max(Θidx, idx)
                    # println("update BR i=",i)
                    mul!(tmpNn, view(BMs, :, :, i - 1), view(BRMs, :, :, i - 1))
                    LAPACK.geqrf!(tmpNn, tau)
                    LAPACK.orgqr!(tmpNn, tau, ns)
                    copyto!(view(BRMs, :, :, i), tmpNn)
                end

                for i in idx-1:-1:min(Θidx, idx)
                    # println("update BL i=",i)
                    mul!(tmpnN, view(BLMs, :, :, i + 1), view(BMs, :, :, i))
                    LAPACK.gerqf!(tmpnN, tau)
                    LAPACK.orgrq!(tmpnN, tau, ns)
                    copyto!(view(BLMs, :, :, i), tmpnN)
                end
                get_ABX!(G, A, B, SCEE, model.nodes, idx, "Forward")
            end

        end

        # println("\n ----------------reverse update ----------------")

        for lt in model.Nt:-1:1

            #####################################################################
            Gt_, G0_, Gt0_, G0t_ = G4(model, s, lt, div(model.Nt, 2))
            if norm(Gt - Gt_) + norm(Gt0 - Gt0_) + norm(G0t - G0t_) > ERROR
                println(norm(Gt - Gt_), '\n', norm(Gt0 - Gt0_), '\n', norm(G0t - G0t_), '\n', norm(G0 - G0_))
                error("WrapTime=$lt ")
            end
            XAinv_ = cis(A.alpha) * I(length(A.index)) .+ (1 - cis(A.alpha)) * view(G0_, indexA[:], indexA[:])
            detXA_ = abs2(det(XAinv_))
            XAinv_ = inv(XAinv_)

            XBinv_ = cis(B.alpha) * I(length(B.index)) .+ (1 - cis(B.alpha)) * view(G0_, indexB[:], indexB[:])
            detXB_ = abs2(det(XBinv_))
            XBinv_ = inv(XBinv_)
            if norm(XAinv_ - A.Xinv) + norm(B.Xinv - XBinv_) + abs(A.detX - detXA_) + abs(B.detX - detXB_) > ERROR
                println(norm(XAinv_ - A.Xinv), " ", norm(B.Xinv - XBinv_), " ", abs(A.detX - detXA_), " ", abs(B.detX - detXB_))
                error("s:  $lt : WrapTime")
            end
            #####################################################################

            for j in axes(ss[1], 2)
                UpdateDOPLayer!(rng, j, view(s, :, j, lt), lt, G, A, B, model, UPD, SCEE, λ)
                for i in axes(ss[1], 1)
                    x, y = model.nnidx[i, j]
                    tmpN[x] = model.α[lt] * model.η[ss[1][i, j, lt]]
                    tmpN[y] = -model.α[lt] * model.η[ss[1][i, j, lt]]
                end
                tmpN .= exp.(.-tmpN)

                WrapV!(tmpNN, Gt01, tmpN, view(model.UV, :, :, j), "L")

                WrapV!(tmpNN, Gt1, tmpN, view(model.UV, :, :, j), "B")

                WrapV!(tmpNN, G0t1, tmpN, view(model.UV, :, :, j), "R")
            end
            WrapK!(tmpNN, G1, model.eKinv, model.eK)
            WrapK!(tmpNN, G2, model.eKinv, model.eK)

            ##------------------------------------------------------------------------
            tmpO += (A.detX / B.detX)^(1 / Nλ)
            counter += 1
            ##------------------------------------------------------------------------

            if any(model.nodes .== (lt - 1))
                idx -= 1
                BM_F!(tmpN, tmpNN, view(BMs, :, :, idx), model, s, idx)
                BMinv_F!(tmpN, tmpNN, view(BMsinv, :, :, idx), model, s, idx)
                for i in idx:-1:min(Θidx, idx)
                    # println("update BL i=",i)
                    mul!(tmpnN, view(BLMs, :, :, i + 1), view(BMs, :, :, i))
                    LAPACK.gerqf!(tmpnN, tau)
                    LAPACK.orgrq!(tmpnN, tau, ns)
                    copyto!(view(BLMs, :, :, i), tmpnN)
                end
                for i in idx+1:max(Θidx, idx)
                    # println("update BR i=",i)
                    mul!(tmpNn, view(BMs, :, :, i - 1), view(BRMs, :, :, i - 1))
                    LAPACK.geqrf!(tmpNn, tau)
                    LAPACK.orgqr!(tmpNn, tau, ns)
                    copyto!(view(BRMs, :, :, i), tmpNn)
                end
                get_ABX!(G, A, B, SCEE, model.nodes, idx, "Backward")
            else
                @inbounds @simd for iii in 1:Ns
                    @fastmath tmpN[iii] = cis(-model.α[lt] * model.η[s[iii, lt]])
                end

                WrapKV!(tmpNN, model.eK, model.eKinv, tmpN, Gt0, "Backward", "L")
                WrapKV!(tmpNN, model.eK, model.eKinv, tmpN, Gt, "Backward", "B")
                WrapKV!(tmpNN, model.eK, model.eKinv, tmpN, G0t, "Backward", "R")
            end

        end

        O[loop+1] = tmpO / counter
        tmpO = 0.0
        counter = 0
    end

    return s
end

function get_ABX!(G::G4Buffer_, A::DOPBuffer_, B::DOPBuffer_, SCEE::SCEEBuffer_, nodes, idx, direction::String="Backward")
    G4!(SCEE, G, nodes, idx, direction)
    A.Xinv .= view(G.G0, A.index, A.index)
    tmp = cis(A.alpha)
    lmul!(1 - tmp, A.Xinv)
    for i in diagind(A.Xinv)
        A.Xinv[i] += tmp
    end
    A.detX = abs2(det(A.Xinv))
    LAPACK.getrf!(A.Xinv, A.ipiv)
    LAPACK.getri!(A.Xinv, A.ipiv)

    B.Xinv .= view(G.G0, B.index, B.index)
    tmp = cis(B.alpha)
    lmul!(1 - tmp, B.Xinv)
    for i in diagind(B.Xinv)
        B.Xinv[i] += tmp
    end
    tmp = det(B.Xinv)
    @assert imag(tmp) < 1e-8 tmp
    B.detX = abs(tmp)
    LAPACK.getrf!(B.Xinv, B.ipiv)
    LAPACK.getri!(B.Xinv, B.ipiv)
end

function UpdateSCDOPLayer!(rng, s, lt, G::G4Buffer_, A::DOPBuffer_, B::DOPBuffer_, model::tV_Hubbard_Para_, UPD::UpdateBuffer_, SCEE::SCEEBuffer_, λ)
    for i in axes(s, 1)
        x, y = model.nnidx[i, j]
        UPD.subidx = [x, y]
        UPD.subidx = [i]

        sx = rand(rng, model.samplers_dict[s[i]])
        p = get_r!(UPD, model.α[lt] * (model.η[sx] - model.η[s[i]]), G.Gt)
        p *= model.γ[sx] / model.γ[s[i]]

        detTau_A = abs(get_abTau!(A, UPD, G))
        detTau_B = abs(get_abTau!(B, UPD, G))

        @fastmath p *= (detTau_A)^λ * (detTau_B)^(1 - λ)
        if rand(rng) < p
            A.detX *= detTau_A
            B.detX *= detTau_B

            GMupdate!(A)
            GMupdate!(B)
            G4update!(SCEE, UPD, G)
            s[i] = sx
        end
    end
end
