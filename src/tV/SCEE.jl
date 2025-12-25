function ctrl_SCEEicr(path::String, model::tV_Hubbard_Para_, indexA::Vector{Int64}, indexB::Vector{Int64}, Sweeps::Int64, λ::Float64, Nλ::Int64, ss::Vector{Array{UInt8,3}}, record)
    global LOCK = ReentrantLock()
    ERROR = 1e-6
    # WrapErr = Matrix{Float64}(undef, model.Ns, model.Ns)

    NN = length(model.nodes)
    Θidx = div(NN, 2) + 1

    UPD = UpdateBuffer()
    SCEE = SCEEBuffer(model.Ns)
    A = AreaBuffer(indexA)
    B = AreaBuffer(indexB)
    G1 = G4Buffer(model.Ns, NN)
    G2 = G4Buffer(model.Ns, NN)

    name = if model.Lattice == "SQUARE"
        "□"
    elseif model.Lattice == "HoneyComb60"
        "HC"
    elseif model.Lattice == "HoneyComb120"
        "HC120"
    else
        error("Lattice: $(model.Lattice) is not allowed !")
    end
    if length(unique(model.α)) == 1
        file = "$(path)/tVSCEE$(name)_t$(model.Ht)V$(model.Hv1)size$(model.site)Δt$(model.Δt)Θ$(model.Θrelax)N$(Nλ)BS$(model.BatchSize).csv"
    else
        file = "$(path)/tVSCEE$(name)_t$(model.Ht)V$(model.Hv1)_$(model.Hv2)size$(model.site)Δt$(model.Δt)Θ$(model.Θrelax)_$(model.Θquench)N$(Nλ)BS$(model.BatchSize).csv"
    end
    rng = MersenneTwister(Threads.threadid() + time_ns())

    atexit() do
        if record
            lock(LOCK) do
                open(file, "a") do io
                    writedlm(io, O', ',')
                end
            end
        end
    end

    Gt1, G01, Gt01, G0t1, BLMs1, BRMs1, BMs1, BMsinv1 =
        G1.Gt, G1.G0, G1.Gt0, G1.G0t, G1.BLMs, G1.BRMs, G1.BMs, G1.BMinvs
    Gt2, G02, Gt02, G0t2, BLMs2, BRMs2, BMs2, BMsinv2 =
        G2.Gt, G2.G0, G2.Gt0, G2.G0t, G2.BLMs, G2.BRMs, G2.BMs, G2.BMinvs

    # 预分配临时数组
    tmpN, tmpN_, tmpNN, tmpNn, tmpnN, tau = SCEE.N, SCEE.N_, SCEE.NN, SCEE.Nn, SCEE.nN, SCEE.tau

    counter = 0
    O = zeros(Float64, Sweeps + 1)
    O[1] = λ
    tmpO = 0.0

    for idx in 1:NN-1
        BM_F!(tmpN, tmpNN, view(BMs1, :, :, idx), model, ss[1], idx)
        BM_F!(tmpN, tmpNN, view(BMs2, :, :, idx), model, ss[2], idx)
        BMinv_F!(tmpN, tmpNN, view(BMsinv1, :, :, idx), model, ss[1], idx)
        BMinv_F!(tmpN, tmpNN, view(BMsinv2, :, :, idx), model, ss[2], idx)
        # @assert norm(view(BMs1,:,:,idx)*view(BMsinv1,:,:,idx)-I(Ns))<1e-8 "BM1 inv error at idx=$idx"
    end

    transpose!(view(BLMs1, :, :, NN), model.Pt)
    copyto!(view(BRMs1, :, :, 1), model.Pt)

    transpose!(view(BLMs2, :, :, NN), model.Pt)
    copyto!(view(BRMs2, :, :, 1), model.Pt)


    # 没办法优化BL和BR的初始化，只能先全部算出来
    for i in 1:NN-1
        mul!(tmpnN, view(BLMs1, :, :, NN - i + 1), view(BMs1, :, :, NN - i))
        LAPACK.gerqf!(tmpnN, tau)
        LAPACK.orgrq!(tmpnN, tau)
        copyto!(view(BLMs1, :, :, NN - i), tmpnN)

        mul!(tmpNn, view(BMs1, :, :, i), view(BRMs1, :, :, i))
        LAPACK.geqrf!(tmpNn, tau)
        LAPACK.orgqr!(tmpNn, tau)
        copyto!(view(BRMs1, :, :, i + 1), tmpNn)
        # ---------------------------------------------------------------
        mul!(tmpnN, view(BLMs2, :, :, NN - i + 1), view(BMs2, :, :, NN - i))
        LAPACK.gerqf!(tmpnN, tau)
        LAPACK.orgrq!(tmpnN, tau)
        copyto!(view(BLMs2, :, :, NN - i), tmpnN)

        mul!(tmpNn, view(BMs2, :, :, i), view(BRMs2, :, :, i))
        LAPACK.geqrf!(tmpNn, tau)
        LAPACK.orgqr!(tmpNn, tau)
        copyto!(view(BRMs2, :, :, i + 1), tmpNn)

    end


    idx = 1
    get_ABGM!(G1, G2, A, B, SCEE, model.nodes, idx, "Forward")
    for loop in 1:Sweeps
        # println("\n ====== Sweep $loop / $Sweeps ======")
        for lt in 1:model.Nt
            # #####################################################################
            #     # # println("\n WrapTime check at lt=$lt")
            Gt1_, G01_, Gt01_, G0t1_ = G4(model, ss[1], lt - 1, div(model.Nt, 2), "Forward")
            Gt2_, G02_, Gt02_, G0t2_ = G4(model, ss[2], lt - 1, div(model.Nt, 2), "Forward")
            if norm(Gt1 - Gt1_) + norm(Gt2 - Gt2_) + norm(Gt01 - Gt01_) + norm(Gt02 - Gt02_) + norm(G0t1 - G0t1_) + norm(G0t2 - G0t2_) > ERROR
                println(norm(Gt1 - Gt1_), ' ', norm(Gt2 - Gt2_), '\n', norm(G01 - G01_), ' ', norm(G02 - G02_), '\n', norm(Gt01 - Gt01_), ' ', norm(Gt02 - Gt02_), '\n', norm(G0t1 - G0t1_), ' ', norm(G0t2 - G0t2_))
                error("$lt : WrapTime")
            end
            GM_A_ = GroverMatrix(G01_[indexA[:], indexA[:]], G02_[indexA[:], indexA[:]])
            gmInv_A_ = inv(GM_A_)
            GM_B_ = GroverMatrix(G01_[indexB[:], indexB[:]], G02_[indexB[:], indexB[:]])
            gmInv_B_ = inv(GM_B_)
            detg_A_ = det(GM_A_)
            detg_B_ = det(GM_B_)
            if norm(gmInv_A_ - A.gmInv) + norm(B.gmInv - gmInv_B_) + abs(A.detg - detg_A_) + abs(B.detg - detg_B_) > ERROR
                println(norm(gmInv_A_ - A.gmInv), " ", norm(B.gmInv - gmInv_B_), " ", abs(A.detg - detg_A_), " ", abs(B.detg - detg_B_))
                error("s2:  $lt : WrapTime")
            end
            # #####################################################################

            WrapK!(tmpNN, G1, model.eK, model.eKinv)
            WrapK!(tmpNN, G2, model.eK, model.eKinv)

            for j in reverse(axes(ss[1], 2))
                for i in axes(ss[1], 1)
                    x, y = model.nnidx[i, j]
                    tmpN[x] = model.α[lt] * model.η[ss[1][i, j, lt]]
                    tmpN[y] = -model.α[lt] * model.η[ss[1][i, j, lt]]
                    tmpN_[x] = model.α[lt] * model.η[ss[2][i, j, lt]]
                    tmpN_[y] = -model.α[lt] * model.η[ss[2][i, j, lt]]
                end
                tmpN .= exp.(tmpN)
                tmpN_ .= exp.(tmpN_)

                WrapV!(tmpNN, Gt01, tmpN, view(model.UV, :, :, j), "L")
                WrapV!(tmpNN, Gt02, tmpN_, view(model.UV, :, :, j), "L")

                WrapV!(tmpNN, Gt1, tmpN, view(model.UV, :, :, j), "B")
                WrapV!(tmpNN, Gt2, tmpN_, view(model.UV, :, :, j), "B")

                WrapV!(tmpNN, G0t1, tmpN, view(model.UV, :, :, j), "R")
                WrapV!(tmpNN, G0t2, tmpN_, view(model.UV, :, :, j), "R")

                # update
                UpdateSCEELayer!(rng, j, view(ss[1], :, j, lt), view(ss[2], :, j, lt), lt, G1, G2, A, B, model, UPD, SCEE, λ)
                # # #####################################################################
                #     print('-')
                #     Gt1_,G01_,Gt01_,G0t1_=G4(model,ss[1],lt-1,div(model.Nt,2),"Forward")
                #     Gt2_,G02_,Gt02_,G0t2_=G4(model,ss[2],lt-1,div(model.Nt,2),"Forward")
                #     Gt1_=model.eK*Gt1_*model.eKinv
                #     Gt01_=model.eK*Gt01_
                #     G0t1_=G0t1_*model.eKinv
                #     Gt2_=model.eK*Gt2_*model.eKinv
                #     Gt02_=model.eK*Gt02_
                #     G0t2_=G0t2_*model.eKinv

                #     GM_A_=GroverMatrix(G01_[indexA[:],indexA[:]],G02_[indexA[:],indexA[:]])
                #     gmInv_A_=inv(GM_A_)
                #     GM_B_=GroverMatrix(G01_[indexB[:],indexB[:]],G02_[indexB[:],indexB[:]])
                #     gmInv_B_=inv(GM_B_)
                #     detg_A_=det(GM_A_)
                #     detg_B_=det(GM_B_)

                #     for jj in 3:-1:j
                #         E=zeros(model.Ns)
                #         E_=zeros(model.Ns)
                #         for ii in 1:size(ss[1])[1]
                #             x,y=model.nnidx[ii,jj]
                #             E[x]= model.α[lt] * model.η[ss[1][ii,jj,lt]]
                #             E[y]=-model.α[lt] * model.η[ss[1][ii,jj,lt]]
                #             E_[x]= model.α[lt] * model.η[ss[2][ii,jj,lt]]
                #             E_[y]=-model.α[lt] * model.η[ss[2][ii,jj,lt]]
                #         end
                #         Gt1_=model.UV[:,:,jj]*Diagonal(exp.(E))*model.UV[:,:,jj] *Gt1_* model.UV[:,:,jj]*Diagonal(exp.(-E))*model.UV[:,:,jj]
                #         Gt01_=model.UV[:,:,jj]*Diagonal(exp.(E))*model.UV[:,:,jj]*Gt01_
                #         G0t1_=G0t1_*model.UV[:,:,jj]*Diagonal(exp.(-E))*model.UV[:,:,jj]
                #         Gt2_=model.UV[:,:,jj]*Diagonal(exp.(E_))*model.UV[:,:,jj] *Gt2_* model.UV[:,:,jj]*Diagonal(exp.(-E_))*model.UV[:,:,jj]
                #         Gt02_=model.UV[:,:,jj]*Diagonal(exp.(E_))*model.UV[:,:,jj]*Gt02_
                #         G0t2_=G0t2_*model.UV[:,:,jj]*Diagonal(exp.(-E_))*model.UV[:,:,jj]
                #     end

                #     if norm(Gt1-Gt1_)+norm(G01-G01_)+norm(Gt01-Gt01_)+norm(G0t1-G0t1_)+
                #         norm(Gt2-Gt2_)+norm(G02-G02_)+norm(Gt02-Gt02_)+norm(G0t2-G0t2_)+
                #     norm(gmInv_A_-A.gmInv)+norm(B.gmInv-gmInv_B_)+abs(A.detg-detg_A_)+abs(B.detg-detg_B_)>ERROR

                #         println('\n',norm(Gt1-Gt1_),'\n',norm(G01-G01_),'\n',norm(Gt01-Gt01_),'\n',norm(G0t1-G0t1_))
                #         println('\n',norm(Gt2-Gt2_),'\n',norm(G02-G02_),'\n',norm(Gt02-Gt02_),'\n',norm(G0t2-G0t2_))
                #         println(norm(gmInv_A_-A.gmInv)," ",norm(B.gmInv-gmInv_B_)," ",abs(A.detg-detg_A_)," ",abs(B.detg-detg_B_))
                #         error("s1:  $lt  $j:,,,asdasdasd")
                #     end
                # # ######################################################################
            end

            ##------------------------------------------------------------------------
            tmpO += (A.detg / B.detg)^(1 / Nλ)
            counter += 1
            ##------------------------------------------------------------------------

            if any(model.nodes .== lt)
                idx += 1
                BM_F!(tmpN, tmpNN, view(BMs1, :, :, idx - 1), model, ss[1], idx - 1)
                BMinv_F!(tmpN, tmpNN, view(BMsinv1, :, :, idx - 1), model, ss[1], idx - 1)
                BM_F!(tmpN, tmpNN, view(BMs2, :, :, idx - 1), model, ss[2], idx - 1)
                BMinv_F!(tmpN, tmpNN, view(BMsinv2, :, :, idx - 1), model, ss[2], idx - 1)
                for i in idx:max(Θidx, idx)
                    # println("update BR i=",i)
                    mul!(tmpNn, view(BMs1, :, :, i - 1), view(BRMs1, :, :, i - 1))
                    LAPACK.geqrf!(tmpNn, tau)
                    LAPACK.orgqr!(tmpNn, tau)
                    copyto!(view(BRMs1, :, :, i), tmpNn)
                    # ---------------------------------------------------------------
                    mul!(tmpNn, view(BMs2, :, :, i - 1), view(BRMs2, :, :, i - 1))
                    LAPACK.geqrf!(tmpNn, tau)
                    LAPACK.orgqr!(tmpNn, tau)
                    copyto!(view(BRMs2, :, :, i), tmpNn)
                end

                for i in idx-1:-1:min(Θidx, idx)
                    # println("update BL i=",i)
                    mul!(tmpnN, view(BLMs1, :, :, i + 1), view(BMs1, :, :, i))
                    LAPACK.gerqf!(tmpnN, tau)
                    LAPACK.orgrq!(tmpnN, tau)
                    copyto!(view(BLMs1, :, :, i), tmpnN)
                    # ---------------------------------------------------------------
                    mul!(tmpnN, view(BLMs2, :, :, i + 1), view(BMs2, :, :, i))
                    LAPACK.gerqf!(tmpnN, tau)
                    LAPACK.orgrq!(tmpnN, tau)
                    copyto!(view(BLMs2, :, :, i), tmpnN)
                end

                #####################################################################
                # if lt != div(model.Nt,2)
                #     copyto!(WrapErr, Gt1)
                #     axpy!(1.0, Gt2, WrapErr)
                #     axpy!(1.0, G01, WrapErr)
                #     axpy!(1.0, G02, WrapErr)
                #     axpy!(1.0, Gt01, WrapErr)
                #     axpy!(1.0, Gt02, WrapErr)
                #     axpy!(1.0, G0t1, WrapErr)
                #     axpy!(1.0, G0t2, WrapErr)
                # end
                #####################################################################
                get_ABGM!(G1, G2, A, B, SCEE, model.nodes, idx, "Forward")
                #####################################################################
                # if lt != div(model.Nt,2)
                #     axpy!(-1.0, Gt1, WrapErr)
                #     axpy!(-1.0, Gt2, WrapErr)
                #     axpy!(-1.0, G01, WrapErr)
                #     axpy!(-1.0, G02, WrapErr)
                #     axpy!(-1.0, Gt01, WrapErr)
                #     axpy!(-1.0, Gt02, WrapErr)
                #     axpy!(-1.0, G0t1, WrapErr)
                #     axpy!(-1.0, G0t2, WrapErr)
                #     tmp=norm(WrapErr)
                #     if tmp>ERROR
                #         println("Forward WrapTime error for at lt=$lt : $tmp")
                #     end
                # end
                #####################################################################
            end
        end


        for lt in model.Nt:-1:1

            #####################################################################
            Gt1_, G01_, Gt01_, G0t1_ = G4(model, ss[1], lt, div(model.Nt, 2), "Backward")
            Gt2_, G02_, Gt02_, G0t2_ = G4(model, ss[2], lt, div(model.Nt, 2), "Backward")
            if norm(Gt1 - Gt1_) + norm(Gt2 - Gt2_) + norm(Gt01 - Gt01_) + norm(Gt02 - Gt02_) + norm(G0t1 - G0t1_) + norm(G0t2 - G0t2_) > ERROR
                println(norm(Gt1 - Gt1_), '\n', norm(Gt2 - Gt2_), '\n', norm(Gt01 - Gt01_), '\n', norm(Gt02 - Gt02_), '\n', norm(G0t1 - G0t1_), '\n', norm(G0t2 - G0t2_))
                error("$lt : WrapTime")
            end
            GM_A_ = GroverMatrix(G01_[indexA[:], indexA[:]], G02_[indexA[:], indexA[:]])
            gmInv_A_ = inv(GM_A_)
            GM_B_ = GroverMatrix(G01_[indexB[:], indexB[:]], G02_[indexB[:], indexB[:]])
            gmInv_B_ = inv(GM_B_)
            detg_A_ = det(GM_A_)
            detg_B_ = det(GM_B_)
            if norm(gmInv_A_ - A.gmInv) + norm(B.gmInv - gmInv_B_) + abs(A.detg - detg_A_) + abs(B.detg - detg_B_) > ERROR
                println(norm(gmInv_A_ - A.gmInv), " ", norm(B.gmInv - gmInv_B_), " ", abs(A.detg - detg_A_), " ", abs(B.detg - detg_B_))
                error("s2:  $lt : WrapTime")
            end
            #####################################################################

            for j in axes(ss[1], 2)
                # update
                UpdateSCEELayer!(rng, j, view(ss[1], :, j, lt), view(ss[2], :, j, lt), lt, G1, G2, A, B, model, UPD, SCEE, λ)
                # #####################################################################
                #     print('*')
                #     Gt1_,G01_,Gt01_,G0t1_=G4(model,ss[1],lt-1,div(model.Nt,2),"Forward")
                #     Gt2_,G02_,Gt02_,G0t2_=G4(model,ss[2],lt-1,div(model.Nt,2),"Forward")
                #     Gt1_=model.eK*Gt1_*model.eKinv
                #     Gt01_=model.eK*Gt01_
                #     G0t1_=G0t1_*model.eKinv
                #     Gt2_=model.eK*Gt2_*model.eKinv
                #     Gt02_=model.eK*Gt02_
                #     G0t2_=G0t2_*model.eKinv

                #     GM_A_=GroverMatrix(G01_[indexA[:],indexA[:]],G02_[indexA[:],indexA[:]])
                #     gmInv_A_=inv(GM_A_)
                #     GM_B_=GroverMatrix(G01_[indexB[:],indexB[:]],G02_[indexB[:],indexB[:]])
                #     gmInv_B_=inv(GM_B_)
                #     detg_A_=det(GM_A_)
                #     detg_B_=det(GM_B_)

                #     for jj in 3:-1:j
                #         E=zeros(model.Ns)
                #         E_=zeros(model.Ns)
                #         for ii in 1:size(ss[1])[1]
                #             x,y=model.nnidx[ii,jj]
                #             E[x]= model.α[lt] * model.η[ss[1][ii,jj,lt]]
                #             E[y]=-model.α[lt] * model.η[ss[1][ii,jj,lt]]
                #             E_[x]= model.α[lt] * model.η[ss[2][ii,jj,lt]]
                #             E_[y]=-model.α[lt] * model.η[ss[2][ii,jj,lt]]
                #         end
                #         Gt1_=model.UV[:,:,jj]*Diagonal(exp.(E))*model.UV[:,:,jj] *Gt1_* model.UV[:,:,jj]*Diagonal(exp.(-E))*model.UV[:,:,jj]
                #         Gt01_=model.UV[:,:,jj]*Diagonal(exp.(E))*model.UV[:,:,jj]*Gt01_
                #         G0t1_=G0t1_*model.UV[:,:,jj]*Diagonal(exp.(-E))*model.UV[:,:,jj]
                #         Gt2_=model.UV[:,:,jj]*Diagonal(exp.(E_))*model.UV[:,:,jj] *Gt2_* model.UV[:,:,jj]*Diagonal(exp.(-E_))*model.UV[:,:,jj]
                #         Gt02_=model.UV[:,:,jj]*Diagonal(exp.(E_))*model.UV[:,:,jj]*Gt02_
                #         G0t2_=G0t2_*model.UV[:,:,jj]*Diagonal(exp.(-E_))*model.UV[:,:,jj]
                #     end

                #     if norm(Gt1-Gt1_)+norm(G01-G01_)+norm(Gt01-Gt01_)+norm(G0t1-G0t1_)+
                #         norm(Gt2-Gt2_)+norm(G02-G02_)+norm(Gt02-Gt02_)+norm(G0t2-G0t2_)+
                #     norm(gmInv_A_-A.gmInv)+norm(B.gmInv-gmInv_B_)+abs(A.detg-detg_A_)+abs(B.detg-detg_B_)>ERROR

                #         println('\n',norm(Gt1-Gt1_),'\n',norm(G01-G01_),'\n',norm(Gt01-Gt01_),'\n',norm(G0t1-G0t1_))
                #         println('\n',norm(Gt2-Gt2_),'\n',norm(G02-G02_),'\n',norm(Gt02-Gt02_),'\n',norm(G0t2-G0t2_))
                #         println(norm(gmInv_A_-A.gmInv)," ",norm(B.gmInv-gmInv_B_)," ",abs(A.detg-detg_A_)," ",abs(B.detg-detg_B_))
                #         error("s1:  $lt  $j:,,,asdasdasd")
                #     end
                # ######################################################################

                for i in axes(ss[1], 1)
                    x, y = model.nnidx[i, j]
                    tmpN[x] = model.α[lt] * model.η[ss[1][i, j, lt]]
                    tmpN[y] = -model.α[lt] * model.η[ss[1][i, j, lt]]
                    tmpN_[x] = model.α[lt] * model.η[ss[2][i, j, lt]]
                    tmpN_[y] = -model.α[lt] * model.η[ss[2][i, j, lt]]
                end
                tmpN .= exp.(.-tmpN)
                tmpN_ .= exp.(.-tmpN_)

                WrapV!(tmpNN, Gt01, tmpN, view(model.UV, :, :, j), "L")
                WrapV!(tmpNN, Gt02, tmpN_, view(model.UV, :, :, j), "L")

                WrapV!(tmpNN, Gt1, tmpN, view(model.UV, :, :, j), "B")
                WrapV!(tmpNN, Gt2, tmpN_, view(model.UV, :, :, j), "B")

                WrapV!(tmpNN, G0t1, tmpN, view(model.UV, :, :, j), "R")
                WrapV!(tmpNN, G0t2, tmpN_, view(model.UV, :, :, j), "R")

            end

            WrapK!(tmpNN, G1, model.eKinv, model.eK)
            WrapK!(tmpNN, G2, model.eKinv, model.eK)

            ##------------------------------------------------------------------------
            tmpO += (A.detg / B.detg)^(1 / Nλ)
            counter += 1
            ##------------------------------------------------------------------------
            if any(model.nodes .== (lt - 1))
                idx -= 1
                BM_F!(tmpN, tmpNN, view(BMs1, :, :, idx), model, ss[1], idx)
                BM_F!(tmpN, tmpNN, view(BMs2, :, :, idx), model, ss[2], idx)
                BMinv_F!(tmpN, tmpNN, view(BMsinv1, :, :, idx), model, ss[1], idx)
                BMinv_F!(tmpN, tmpNN, view(BMsinv2, :, :, idx), model, ss[2], idx)
                for i in idx:-1:min(Θidx, idx)
                    # println("update BL i=",i)
                    mul!(tmpnN, view(BLMs1, :, :, i + 1), view(BMs1, :, :, i))
                    LAPACK.gerqf!(tmpnN, tau)
                    LAPACK.orgrq!(tmpnN, tau)
                    copyto!(view(BLMs1, :, :, i), tmpnN)

                    mul!(tmpnN, view(BLMs2, :, :, i + 1), view(BMs2, :, :, i))
                    LAPACK.gerqf!(tmpnN, tau)
                    LAPACK.orgrq!(tmpnN, tau)
                    copyto!(view(BLMs2, :, :, i), tmpnN)
                end
                for i in idx+1:max(Θidx, idx)
                    # println("update BR i=",i)
                    mul!(tmpNn, view(BMs1, :, :, i - 1), view(BRMs1, :, :, i - 1))
                    LAPACK.geqrf!(tmpNn, tau)
                    LAPACK.orgqr!(tmpNn, tau)
                    copyto!(view(BRMs1, :, :, i), tmpNn)

                    mul!(tmpNn, view(BMs2, :, :, i - 1), view(BRMs2, :, :, i - 1))
                    LAPACK.geqrf!(tmpNn, tau)
                    LAPACK.orgqr!(tmpNn, tau)
                    copyto!(view(BRMs2, :, :, i), tmpNn)
                end

                # #####################################################################
                #     if lt-1 != div(model.Nt,2)
                #         copyto!(WrapErr, Gt1)
                #         axpy!(1.0, Gt2, WrapErr)
                #         axpy!(1.0, G01, WrapErr)
                #         axpy!(1.0, G02, WrapErr)
                #         axpy!(1.0, Gt01, WrapErr)
                #         axpy!(1.0, Gt02, WrapErr)
                #         axpy!(1.0, G0t1, WrapErr)
                #         axpy!(1.0, G0t2, WrapErr)
                #     end
                # #####################################################################

                get_ABGM!(G1, G2, A, B, SCEE, model.nodes, idx, "Backward")

                # #####################################################################
                #     if lt-1 != div(model.Nt,2)
                #         axpy!(-1.0, Gt1, WrapErr)
                #         axpy!(-1.0, Gt2, WrapErr)
                #         axpy!(-1.0, G01, WrapErr)
                #         axpy!(-1.0, G02, WrapErr)
                #         axpy!(-1.0, Gt01, WrapErr)
                #         axpy!(-1.0, Gt02, WrapErr)
                #         axpy!(-1.0, G0t1, WrapErr)
                #         axpy!(-1.0, G0t2, WrapErr)
                #         tmp=norm(WrapErr)
                #         if tmp>ERROR
                #             println("Backward WrapTime error for at lt=$(lt-1) : $tmp")
                #         end
                #     end
                # #####################################################################
            end
        end


        O[loop+1] = tmpO / counter
        tmpO = counter = 0
    end
    return ss
end

function get_ABGM!(G1::G4Buffer_, G2::G4Buffer_, A::AreaBuffer_, B::AreaBuffer_, SCEE::SCEEBuffer_, nodes, idx, direction::String="Backward")
    G4!(SCEE, G1, nodes, idx, direction)
    G4!(SCEE, G2, nodes, idx, direction)
    GroverMatrix!(A.gmInv, view(G1.G0, A.index, A.index), view(G2.G0, A.index, A.index))
    A.detg = det(A.gmInv)
    LAPACK.getrf!(A.gmInv, A.ipiv)
    LAPACK.getri!(A.gmInv, A.ipiv)

    GroverMatrix!(B.gmInv, view(G1.G0, B.index, B.index), view(G2.G0, B.index, B.index))
    B.detg = det(B.gmInv)
    LAPACK.getrf!(B.gmInv, B.ipiv)
    LAPACK.getri!(B.gmInv, B.ipiv)
end

function UpdateSCEELayer!(rng, j, s1, s2, lt, G1::G4Buffer_, G2::G4Buffer_, A::AreaBuffer_, B::AreaBuffer_, model::tV_Hubbard_Para_, UPD::UpdateBuffer_, SCEE::SCEEBuffer_, λ)
    for i in axes(s1, 1)
        x, y = model.nnidx[i, j]
        UPD.subidx = [x, y]

        # update s1
        begin
            sx = rand(rng, model.samplers_dict[s1[i]])
            p = get_r!(UPD, model.α[lt] * (model.η[sx] - model.η[s1[i]]), G1.Gt)
            p *= model.γ[sx] / model.γ[s1[i]]

            detTau_A = get_abTau1!(A, UPD, G2.G0, G1.Gt0, G1.G0t)
            detTau_B = get_abTau1!(B, UPD, G2.G0, G1.Gt0, G1.G0t)

            @fastmath p *= (detTau_A)^λ * (detTau_B)^(1 - λ)
            if rand(rng) < p
                A.detg *= detTau_A
                B.detg *= detTau_B

                GMupdate!(A)
                GMupdate!(B)
                G4update!(SCEE, UPD, G1)
                s1[i] = sx
            end
        end

        # update ss[2]
        begin
            sx = rand(rng, model.samplers_dict[s2[i]])
            p = get_r!(UPD, model.α[lt] * (model.η[sx] - model.η[s2[i]]), G2.Gt)
            p *= model.γ[sx] / model.γ[s2[i]]

            detTau_A = get_abTau2!(A, UPD, G1.G0, G2.Gt0, G2.G0t)
            detTau_B = get_abTau2!(B, UPD, G1.G0, G2.Gt0, G2.G0t)

            @fastmath p *= (detTau_A)^λ * (detTau_B)^(1 - λ)
            if rand(rng) < p
                A.detg *= detTau_A
                B.detg *= detTau_B

                GMupdate!(A)
                GMupdate!(B)
                G4update!(SCEE, UPD, G2)
                s2[i] = sx
            end
        end
    end
end

"""
    Overwrite G according to eK and eKinv , with option mid
        Forward Wrap :      WrapK!(tmpNN,Gt,Gt0,G0t,eK,eKinv)
            Gt = eK ⋅ Gt ⋅ eKinv
            Gt0 = eK ⋅ Gt0
            G0t = G0t ⋅ eKinv
        Inverse Wrap:       WrapK!(tmpNN,Gt,Gt0,G0t,eKinv,eK)    
            Gt = eKinv ⋅ Gt ⋅ eK
            Gt0 = eKinv ⋅ Gt0
            G0t = G0t ⋅ eK
    Only wrap Kinetic part forward direction  
    ------------------------------------------------------------------------------
"""
function WrapK!(tmpNN::Matrix{Float64}, G::G4Buffer_, eK::Matrix{Float64}, eKinv::Matrix{Float64})
    mul!(tmpNN, G.Gt, eKinv)
    mul!(G.Gt, eK, tmpNN)

    mul!(tmpNN, eK, G.Gt0)
    copyto!(G.Gt0, tmpNN)
    mul!(tmpNN, G.G0t, eKinv)
    copyto!(G.G0t, tmpNN)
end
