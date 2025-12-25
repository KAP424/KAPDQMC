function ctrl_SCDOPicr(path::String, model::tU_Hubbard_Para_, alpha::Float64, indexA::Vector{Int64}, indexB::Vector{Int64}, Sweeps::Int64, λ::Float64, Nλ::Int64, s::Matrix{UInt8}, record)
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
    if length(unique(model.α)) == 1
        file = "$(path)/tUSCDOP$(name)_t$(model.Ht)U$(model.Hu1)size$(model.site)Δt$(model.Δt)Θ$(model.Θrelax)N$(Nλ)BS$(model.BatchSize).csv"
    else
        file = "$(path)/tUSCDOP$(name)_t$(model.Ht)U$(model.Hu1)_$(model.Hu2)size$(model.site)Δt$(model.Δt)Θ$(model.Θrelax)_$(model.Θquench)N$(Nλ)BS$(model.BatchSize).csv"
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

    transpose!(view(BLMs, :, :, NN), model.Pt)
    copyto!(view(BRMs, :, :, 1), model.Pt)

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

            UpdateSCDOPLayer!(rng, view(s, :, lt), lt, G, A, B, model, UPD, SCEE, λ)

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

            UpdateSCDOPLayer!(rng, view(s, :, lt), lt, G, A, B, model, UPD, SCEE, λ)

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
    B.detX = abs2(det(B.Xinv))
    LAPACK.getrf!(B.Xinv, B.ipiv)
    LAPACK.getri!(B.Xinv, B.ipiv)
end

function UpdateSCDOPLayer!(rng, s, lt, G::G4Buffer_, A::DOPBuffer_, B::DOPBuffer_, model::tU_Hubbard_Para_, UPD::UpdateBuffer_, SCEE::SCEEBuffer_, λ)
    for i in axes(s, 1)
        UPD.subidx = [i]

        sx = rand(rng, model.samplers_dict[s[i]])
        p = get_r!(UPD, model.α[lt] * (model.η[sx] - model.η[s[i]]), G.Gt)
        p *= model.γ[sx] / model.γ[s[i]]

        detTau_A = abs2(get_abTau!(A, UPD, G))
        detTau_B = abs2(get_abTau!(B, UPD, G))

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

# function DOP_icr(path::String,model::Hubbard_Para_,ω,index::Vector{Int64},Sweeps::Int64,λ::Float64,Nλ::Int64,s::Matrix{UInt8},record)::Matrix{UInt8}
#     if model.Lattice=="SQUARE"
#         name="□"
#     elseif model.Lattice=="HoneyComb60"
#         name="HC60"
#     elseif model.Lattice=="HoneyComb120"
#         name="HC120"
#     end
#     file="$(path)DOP$(name)_t$(model.t)U$(model.U)size$(model.site)Δt$(model.Δt)Θ$(model.Θ)N$(Nλ)BS$(model.BatchSize)ω$(round(ω,digits=2))N$(length(index)).csv"

#     atexit() do
#         if record
#             open(file, "a") do io
#                 lock(io)
#                 writedlm(io, O', ',')
#                 unlock(io)
#             end
#         end
#         writedlm("$(path)s/S$(name)_t$(model.t)U$(model.U)size$(model.site)Δt$(model.Δt)Θ$(model.Θ)λ$(Int(round(Nλ*λ)))ω$(round(ω,digits=2)).csv", s,",")
#     end

#     rng=MersenneTwister(Threads.threadid()+round(Int,time()*1000))
#     elements=(1, 2, 3, 4)

#     Gt=zeros(ComplexF64,model.Ns,model.Ns)
#     G0=zeros(ComplexF64,model.Ns,model.Ns)
#     Gt0=zeros(ComplexF64,model.Ns,model.Ns)
#     G0t=zeros(ComplexF64,model.Ns,model.Ns)
#     Xinv=zeros(ComplexF64,length(index),length(index))
#     detX=0

#     tmpO=0
#     counter=0
#     O=zeros(Sweeps+1)
#     O[1]=λ

#     I1=I(model.Ns)
#     I2=I(length(index))


#     for loop in 1:Sweeps
#         for lt in 1:model.Nt
#             if mod(lt,model.WrapTime)==1 || lt==div(model.Nt,2)+1
#                 Gt,G0,Gt0,G0t=G4(model,s,lt,div(model.Nt,2))
#                 Xinv=inv( G0[index[:],index[:]]+exp(1im*ω)*(I2-G0[index[:],index[:]]) )
#                 detX=1/abs2(det(Xinv))
#             else
#                 D=[model.η[x] for x in s[:,lt]]
#                 Gt=diagm(exp.(1im*model.α.*D))*model.eK *Gt* model.eKinv*diagm(exp.(-1im*model.α.*D))
#                 G0t=G0t*model.eKinv*diagm(exp.(-1im*model.α.*D))
#                 Gt0=diagm(exp.(1im*model.α.*D))*model.eK*Gt0

#                 #####################################################################
#                 # Gt_,G0_,Gt0_,G0t_=G4(model,s,lt,div(model.Nt,2))

#                 # if norm(Gt-Gt_)+norm(Gt0-Gt0_)+norm(G0t-G0t_)>1e-3
#                 #     println( norm(Gt-Gt_),'\n',norm(Gt0-Gt0_),'\n',norm(G0t-G0t_),'\n')
#                 #     error("$lt : WrapTime")
#                 # end
#                 #####################################################################


#             end

#             for x in 1:model.Ns
#                 b=transpose(Gt0[x,index[:]]) *Xinv
#                 a=G0t[index[:],x]
#                 Tau=b*a

#                 sp=Random.Sampler(rng,[i for i in elements if i != s[x,lt]])
#                 sx=rand(rng,sp)

#                 Δ=exp(1im*model.α*(model.η[sx]-model.η[s[x,lt]]))-1
#                 r=1+Δ*(1-Gt[x,x])

#                 c=1+(1-exp(1im*ω))*Δ/r*Tau
#                 p=model.γ[sx]/model.γ[s[x,lt]]*abs2(r)*abs2(c)^(λ)

#                 if rand(rng)<p
#                     rho=(1-exp(1im*ω))*Δ/r/c
#                     Xinv-=rho* ( Xinv*a .* b)
#                     detX*=abs2(c)

#                     G0+=Δ/r* (G0t[:,x] .* transpose(Gt0[x,:]))
#                     Gt0+=Δ/r* (Gt[:,x] .* transpose(Gt0[x,:]))
#                     G0t-=Δ/r* (G0t[:,x] .* transpose( (I1-Gt)[x,:] ) )
#                     Gt-=Δ/r* (Gt[:,x] .* transpose( (I1-Gt)[x,:]) )         
#                     s[x,lt]=sx

#                     #####################################################################
#                     # print('-')
#                     # Gt_,G0_,Gt0_,G0t_=G4(model,s,lt,div(model.Nt,2))

#                     # if norm(Gt-Gt_)+norm(G0-G0_)+norm(Gt0-Gt0_)+norm(G0t-G0t_)>1e-3
#                     #     println('\n',norm(Gt-Gt_),'\n',norm(G0-G0_),'\n',norm(Gt0-Gt0_),'\n',norm(G0t-G0t_))
#                     #     error("$lt  $x:,,,asdasdasd")
#                     # end
#                     #####################################################################
#                 end

#             end

#             ##------------------------------------------------------------------------
#             tmpO+=detX^(1/Nλ)
#             counter+=1
#             ##------------------------------------------------------------------------
#         end

#         for lt in model.Nt-1:-1:1
#             if mod(lt,model.WrapTime)==0 || lt==div(model.Nt,2)
#                 Gt,G0,Gt0,G0t=G4(model,s,lt,div(model.Nt,2))
#                 Xinv=inv( G0[index[:],index[:]]+exp(1im*ω)*(I2-G0[index[:],index[:]]) )
#                 detX=1/abs2(det(Xinv))
#             else
#                 D=[model.η[x] for x in s[:,lt+1]]
#                 Gt=model.eKinv*diagm(exp.(-1im*model.α.*D)) *Gt* diagm(exp.(1im*model.α.*D))*model.eK
#                 G0t=G0t*diagm(exp.(1im*model.α.*D))*model.eK 
#                 Gt0=model.eKinv*diagm(exp.(-1im*model.α.*D)) *Gt0

#                 #####################################################################
#                 # Gt_,G0_,Gt0_,G0t_=G4(model,s,lt,div(model.Nt,2))

#                 # if norm(Gt-Gt_)+norm(Gt0-Gt0_)+norm(G0t-G0t_)>1e-3
#                 #     println( norm(Gt-Gt_),'\n',norm(Gt0-Gt0_),'\n',norm(G0t-G0t_),'\n')
#                 #     error("$lt : WrapTime")
#                 # end
#                 #####################################################################
#                 end

#                 for x in 1:model.Ns
#                     b=transpose(Gt0[x,index[:]]) *Xinv
#                     a=G0t[index[:],x]
#                     Tau=b*a

#                     sp=Random.Sampler(rng,[i for i in elements if i != s[x,lt]])
#                     sx=rand(rng,sp)

#                     Δ=exp(1im*model.α*(model.η[sx]-model.η[s[x,lt]]))-1
#                     r=1+Δ*(1-Gt[x,x])

#                     c=1+(1-exp(1im*ω))*Δ/r*Tau
#                     p=model.γ[sx]/model.γ[s[x,lt]]*abs2(r)*abs2(c)^(λ)

#                     if rand(rng)<p
#                         rho=(1-exp(1im*ω))*Δ/r/c
#                         Xinv-=rho* ( Xinv*a .* b)
#                         detX*=abs2(c)

#                         G0+=Δ/r* (G0t[:,x] .* transpose(Gt0[x,:]))
#                         Gt0+=Δ/r* (Gt[:,x] .* transpose(Gt0[x,:]))
#                         G0t-=Δ/r* (G0t[:,x] .* transpose( (I1-Gt)[x,:] ) )
#                         Gt-=Δ/r* (Gt[:,x] .* transpose( (I1-Gt)[x,:]) )         
#                         s[x,lt]=sx

#                         #####################################################################
#                         # print('-')
#                         # Gt_,G0_,Gt0_,G0t_=G4(model,s,lt,div(model.Nt,2))

#                         # if norm(Gt-Gt_)+norm(G0-G0_)+norm(Gt0-Gt0_)+norm(G0t-G0t_)>1e-3
#                         #     println('\n',norm(Gt-Gt_),'\n',norm(G0-G0_),'\n',norm(Gt0-Gt0_),'\n',norm(G0t-G0t_))
#                         #     error("$lt  $x:,,,asdasdasd")
#                         # end
#                         #####################################################################
#                     end

#                 end

#             ##------------------------------------------------------------------------
#             tmpO+=detX^(1/Nλ)
#             counter+=1
#             ##------------------------------------------------------------------------
#         end

#         O[loop+1]=tmpO/counter
#         tmpO=counter=0

#     end

#     return s
# end