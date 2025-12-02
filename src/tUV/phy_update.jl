function UpdatePhyLayer!(rng,s,model::tUV_Hubbard_Para_,UPD::UpdateBuffer_,Phy::PhyBuffer_)
    for i in axes(s,1)
        x,y=model.nnidx[i]
        UPD.subidx.=[x,y]
        sx = rand(rng, model.samplers_dict[s[i]])
        p=get_r!(UPD,model.η[sx]- model.η[s[i]],Phy.G)
        p*=model.γ[sx]/model.γ[s[i]]
        if model.Type==Float64
            p*=exp( ( model.η[s[i]] - model.η[sx] ) * (model.a+1) )
        end
        if rand(rng)<p
            Gupdate!(Phy,UPD)
            s[i]=sx
        end
    end
end

function phy_update(path::String,model::tUV_Hubbard_Para_,s::Array{UInt8,2},Sweeps::Int64,record::Bool)
    @assert size(s,1)==length(model.nnidx) "size of s $(size(s)) and nnidx $(length(model.nnidx)) not match!"
    @assert size(s,2)==model.Nt "size of s and nnidx not match!"
    global LOCK=ReentrantLock()
    ERROR=1e-6

    UPD = UpdateBuffer(model.Type)
    UPD.a = model.a
    NN=length(model.nodes)
    Phy = PhyBuffer(model.Type,model.Ns, NN) 
    Θidx=div(NN,2)+1

    name = if model.Lattice=="SQUARE" "□" 
    elseif model.Lattice=="HoneyComb60" "HC" 
    elseif model.Lattice=="HoneyComb120" "HC120" 
    else error("Lattice: $(model.Lattice) is not allowed !") end  
    file="$(path)/Phy$(name)_t$(model.t)U$(model.U)V$(model.V)size$(model.site)Δt$(model.Δt)Θ$(model.Θ)BS$(model.BatchSize).csv"

    rng=MersenneTwister(Threads.threadid()+time_ns())

    tau = Phy.tau
    ipiv = Phy.ipiv
    Ek=Eu=Ev=CDW0=CDW1=SDW0=SDW1=0.0
    counter=0

    G = Phy.G

    # 预分配 BL 和 BR
    BLs = Phy.BLs
    BRs = Phy.BRs
    # 预分配临时数组
    tmpN = Phy.N
    tmpNN = Phy.NN
    BM = Phy.BM
    tmpNn = Phy.Nn
    tmpnn = Phy.nn
    tmpnN = Phy.nN

    copyto!(view(BRs,:,:,1) , model.Pt)
    transpose!(view(BLs,:,:,NN) , model.Pt)

    for idx in NN-1:-1:1
        BM_F!(tmpN,tmpNN,BM,model,s,idx)
        mul!(tmpnN,view(BLs,:,:,idx+1), BM)
        LAPACK.gerqf!(tmpnN, tau)
        LAPACK.orgrq!(tmpnN, tau)
        copyto!(view(BLs,:,:,idx), tmpnN)
        # view(BLs,:,:,idx) .= Matrix(qr!(tmpNn).Q)'
    end

    idx=1
    get_G!(tmpnn,tmpnN,ipiv,view(BLs,:,:,1), view(BRs,:,:,1),G)
    for loop in 1:Sweeps
        # println("\n Sweep: $loop ")
        for lt in axes(s,2)
            #####################################################################
                # print("*")
                # if norm(G-Gτ(model,s,lt-1))>ERROR
                #     error("Wrap-$(lt)   :   $(norm(G-Gτ(model,s,lt-1))) , $(norm(G)) , $(norm(Gτ(model,s,lt-1))) ")
                # end
            #####################################################################

            fill!(tmpN,0)
            for i in axes(s,1)
                x,y=model.nnidx[i]
                tmpN[x]+=model.η[s[i,lt]] 
                tmpN[y]+=model.a*model.η[s[i,lt]]
            end
            if model.Type==Float64
                tmpN.= exp.(tmpN)
            else
                tmpN.= exp.(1im.*tmpN)
            end
            WrapKV!(tmpNN,model.eK,model.eKinv,tmpN,G,"Forward","B")

            UpdatePhyLayer!(rng,view(s,:,lt),model,UPD,Phy)

            if record && abs(idx-Θidx)<=1
                tmp=phy_measure(model,Phy,lt,s)
                Ek+=tmp[1]
                Eu+=tmp[2]
                Ev+=tmp[3]
                CDW0+=tmp[4]
                CDW1+=tmp[5]
                SDW0+=tmp[6]
                SDW1+=tmp[7]
                counter+=1
            end

            if any(model.nodes.== lt)
                idx+=1
                BM_F!(tmpN,tmpNN,BM,model, s, idx - 1)
                mul!(tmpNn, BM, view(BRs,:,:,idx-1))
                LAPACK.geqrf!(tmpNn, tau)
                LAPACK.orgqr!(tmpNn, tau)
                copyto!(view(BRs,:,:,idx), tmpNn)
                
                # copyto!(tmpNN , G)

                get_G!(tmpnn,tmpnN,ipiv,view(BLs,:,:,idx), view(BRs,:,:,idx),G)

                #------------------------------------------------------------------#
                # axpy!(-1.0, G, tmpNN)  
                # if norm(tmpNN)>1e-7
                #     println("Warning for Batchsize Wrap Error : $(norm(tmpNN))")
                # end
                #------------------------------------------------------------------#

            end

        end

        for lt in reverse(axes(s,2))
            #####################################################################
                # print("-")
                # if norm(G-Gτ(model,s,lt))>ERROR
                #     error("Wrap-$(lt)   :   $(norm(G-Gτ(model,s,lt)))")
                # end
            ######################################################################
            UpdatePhyLayer!(rng,view(s,:,lt),model,UPD,Phy)

            if record && abs(idx-Θidx)<=1
                tmp=phy_measure(model,Phy,lt,s)
                Ek+=tmp[1]
                Eu+=tmp[2]
                Ev+=tmp[3]
                CDW0+=tmp[4]
                CDW1+=tmp[5]
                SDW0+=tmp[6]
                SDW1+=tmp[7]
                counter+=1
            end

            fill!(tmpN,0)
            for i in axes(s,1)
                x,y=model.nnidx[i]
                tmpN[x]+=model.η[s[i,lt]]
                tmpN[y]+=model.a*model.η[s[i,lt]]
            end
            if model.Type==Float64
                tmpN.= exp.(.-tmpN)
            else
                tmpN.= exp.(-1im.*tmpN)
            end
            WrapKV!(tmpNN,model.eK,model.eKinv,tmpN,G,"Backward","B")

            if any(model.nodes.== (lt-1))
                idx-=1
                BM_F!(tmpN,tmpNN,BM,model, s, idx)
                mul!(tmpnN,view(BLs,:,:,idx+1),BM)
                LAPACK.gerqf!(tmpnN, tau)
                LAPACK.orgrq!(tmpnN, tau)
                copyto!(view(BLs,:,:,idx) , tmpnN)
                get_G!(tmpnn,tmpnN,ipiv,view(BLs,:,:,idx), view(BRs,:,:,idx),G)
                # #------------------------------------------------------------------#
                # axpy!(-1.0, G, tmpNN)  
                # if norm(tmpNN)>1e-7
                #     println("Warning for Batchsize Wrap Error : $(norm(tmpNN))")
                # end
                # #------------------------------------------------------------------#
            end
        end

        if record 
            # && mod1(loop,5)==5
            # @assert imag(CDW0)+imag(CDW1)+imag(SDW0)+imag(SDW1)<1e-8 "struct factor get imaginary part! $(imag(CDW0)),$(imag(CDW1)),$(imag(SDW0)),$(imag(SDW1))"
            CDW0=real(CDW0)
            CDW1=real(CDW1)
            SDW0=real(SDW0)
            SDW1=real(SDW1)
            lock(LOCK) do
                open(file, "a") do io
                    writedlm(io,[Ek,Eu,Ev,CDW0, CDW1, SDW0, SDW1]' ./ counter, ',')
                end
            end
            Ek=Eu=Ev=CDW0=CDW1=SDW0=SDW1=0.0
            counter=0
        end
    end
    return s
end 



function phy_measure(model::tUV_Hubbard_Para_,Phy::PhyBuffer_,lt,s)
    """
    (Ek,Ev,R0,R1)    
    """
    G0=Phy.G[:,:]
    tmpN=Phy.N
    tmpNN=Phy.NN
    CDW0=CDW1=SDW0=SDW1=0.0
    
    if lt>model.Nt/2
        for t in lt:-1:div(model.Nt,2)+1
            fill!(tmpN,0)
            for i in axes(s,1)
                x,y=model.nnidx[i]
                tmpN[x]+=model.η[s[i,t]]
                tmpN[y]+=model.a*model.η[s[i,t]]
            end
            if model.Type==Float64
                tmpN.= exp.(.-tmpN)
            else
                tmpN.= exp.(-1im.*tmpN)
            end
            WrapKV!(tmpNN,model.eK,model.eKinv,tmpN,G0,"Backward","B")
        end
    else
        for t in lt+1:div(model.Nt,2)
            fill!(tmpN,0)
            for i in axes(s,1)
                x,y=model.nnidx[i]
                tmpN[x]+=model.η[s[i,t]]
                tmpN[y]+=model.a*model.η[s[i,t]]
            end
            if model.Type==Float64
                tmpN.= exp.(tmpN)
            else
                tmpN.= exp.(1im.*tmpN)
            end
            WrapKV!(tmpNN,model.eK,model.eKinv,tmpN,G0,"Forward","B")
        end
    end
    #####################################################################
    # if norm(G0-Gτ(model,s,div(model.Nt,2)))>1e-7
    #     error("record error lt=$(lt) : $(norm(G0-Gτ(model,s,div(model.Nt,2))))")
    # end
    #####################################################################
    mul!(tmpNN,model.HalfeK,G0)
    mul!(G0,tmpNN,model.HalfeKinv)
    # G0=model.HalfeK* G0 *model.HalfeKinv

    Ek=2*model.t*real(sum(model.K.*G0))
    Eu=0
    if model.Type==ComplexF64
        for i in 1:model.Ns
            Eu+=(1-G0[i,i])*adjoint(G0[i,i])
        end
    else
        for i in 1:model.Ns
            Eu+=(1-G0[i,i])* (1-G0[i,i])
        end
    end
    @assert imag(Eu)<1e-8 "Eu get imaginary part! $(Eu)" 
    Eu=model.U*real(Eu)
    Ev=0.0
    if model.Type==ComplexF64
        for k in eachindex(model.nnidx)
            x,y=model.nnidx[k]
            Ev+=(1-G0[x,x])*(1-G0[y,y])-G0[x,y]*G0[y,x]         #<up up>
            Ev+= adjoint( G0[x,x]*G0[y,y]-G0[x,y]*G0[y,x] )     #<down down>
            Ev+= (1-G0[y,y]) * adjoint(G0[x,x])+ (1-G0[x,x]) * adjoint(G0[y,y])  #<up down> + <down up>
        end
    else
        for k in eachindex(model.nnidx)
            x,y=model.nnidx[k]
            Ev+=2*( (1-G0[x,x])*(1-G0[y,y])-G0[x,y]*G0[y,x] )      #<up up> + <down down>
            Ev+=2*(1-G0[y,y]) * (1-G0[x,x])   #<up down> + <down up>
        end
    end
    @assert imag(Ev)<1e-8 "Ev get imaginary part! $(Ev)"
    Ev=model.V*real(Ev)
    # @assert abs(sum(imag(diag(G0))))<1e-8 "G0 diag get imaginary part! $(norm(imag(diag(G0))))"
    if occursin("HoneyComb", model.Lattice)
        for rx in 1:model.site[1]
            for ry in 1:model.site[2]
                tmp1=0.0im     #<up up> <down down>  
                tmp2=0.0im     #<up down> <down up>
                for ix in 1:model.site[1]
                    for iy in 1:model.site[2]
                        idx1=xy_i(model.Lattice,model.site,ix,iy)-1
                        idx2=xy_i(model.Lattice,model.site,mod1(ix+rx,model.site[1]),mod1(iy+ry,model.site[2]))-1
                        
                        delta = idx1==idx2 ? 1 : 0
                        if model.Type==Float64
                            tmp1+=2*( (1-G0[idx1,idx1]) * (1-G0[idx2,idx2]) + (delta-G0[idx1,idx2])*G0[idx2,idx1] ) # <i_A j_A>
                            tmp1+=2*( (1-G0[idx1+1,idx1+1]) * (1-G0[idx2+1,idx2+1]) + (delta-G0[idx1+1,idx2+1])*G0[idx2+1,idx1+1] ) # <i_B j_B>
                            tmp1-=2*( (1-G0[idx1+1,idx1+1])*(1-G0[idx2,idx2])-G0[idx1+1,idx2]*G0[idx2,idx1+1] ) # <i_B j_A>
                            tmp1-=2*( (1-G0[idx1,idx1])*(1-G0[idx2+1,idx2+1])-G0[idx1,idx2+1]*G0[idx2+1,idx1] ) # <i_A j_B>
                        
                            tmp2+=2*(1-G0[idx1,idx1]) * (1-G0[idx2,idx2])   # <i_A j_A>
                            tmp2+=2*(1-G0[idx1+1,idx1+1]) * (1-G0[idx2+1,idx2+1])   # <i_B j_B>
                            tmp2-=2*(1-G0[idx1+1,idx1+1]) * (1-G0[idx2,idx2])   # <i_B j_A>
                            tmp2-=2*(1-G0[idx1,idx1])   * (1-G0[idx2+1,idx2+1])   # <i_A j_B>
                        else
                            tmp1+=( (1-G0[idx1,idx1]) * (1-G0[idx2,idx2]) + (delta-G0[idx1,idx2])*G0[idx2,idx1] ) 
                                + adjoint( G0[idx1,idx1]*G0[idx2,idx2] + G0[idx1,idx2]*(delta-G0[idx2,idx1]) )  # <i_A j_A>
                            tmp1+=( (1-G0[idx1+1,idx1+1]) * (1-G0[idx2+1,idx2+1]) + (delta-G0[idx1+1,idx2+1])*G0[idx2+1,idx1+1] )
                                + adjoint( G0[idx1+1,idx1+1]*G0[idx2+1,idx2+1] + G0[idx1+1,idx2+1]*(delta-G0[idx2+1,idx1+1]) )  # <i_B j_B>
                            tmp1-=( (1-G0[idx1+1,idx1+1])*(1-G0[idx2,idx2])-G0[idx1+1,idx2]*G0[idx2,idx1+1] )
                                + adjoint( G0[idx1+1,idx1+1]*G0[idx2,idx2]-G0[idx1+1,idx2]*G0[idx2,idx1+1] )  # <i_B j_A>
                            tmp1-=( (1-G0[idx1,idx1])*(1-G0[idx2+1,idx2+1])-G0[idx1,idx2+1]*G0[idx2+1,idx1] )
                                + adjoint( G0[idx1,idx1]*G0[idx2+1,idx2+1]-G0[idx1,idx2+1]*G0[idx2+1,idx1] )  # <i_A j_B>

                            tmp2+= (1-G0[idx1,idx1])*adjoint(G0[idx2,idx2]) 
                                + adjoint(G0[idx1,idx1])*(1-G0[idx2,idx2])    # <i_A j_A>
                            tmp2+= (1-G0[idx1+1,idx1+1])*adjoint(G0[idx2+1,idx2+1])
                                + adjoint(G0[idx1+1,idx1+1])*(1-G0[idx2+1,idx2+1])    # <i_B j_B>
                            tmp2-= (1-G0[idx1,idx1])*adjoint(G0[idx2+1,idx2+1]) 
                                + adjoint(G0[idx1,idx1])*(1-G0[idx2+1,idx2+1])    # <i_A j_B>
                            tmp2-= (1-G0[idx1+1,idx1+1])*adjoint(G0[idx2,idx2]) 
                                + adjoint(G0[idx1+1,idx1+1])*(1-G0[idx2,idx2])    # <i_B j_A>
                        end
                    end
                end
                CDW0+=tmp1+tmp2
                CDW1+=cos(2*π/model.site[1]*rx+2*π/model.site[2]*ry )* (tmp1+tmp2)
                SDW0+=tmp1-tmp2
                SDW1+=cos(2*π/model.site[1]*rx+2*π/model.site[2]*ry )* (tmp1-tmp2)
            end
        end
        CDW0=CDW0/model.Ns^2
        CDW1=CDW1/model.Ns^2
        SDW0=SDW0/model.Ns^2
        SDW1=SDW1/model.Ns^2
    elseif model.Lattice=="SQUARE"
        for rx in 1:model.site[1]
            for ry in 1:model.site[2]
                tmp=0
                for ix in 1:model.site[1]
                    for iy in 1:model.site[2]
                        idx1=ix+(iy-1)*model.site[1]
                        idx2=mod1(rx+ix,model.site[1])+mod((ry+iy-1),model.site[2])*model.site[1]
                        tmp+=(1-G0[idx1,idx1])*(1-G0[idx2,idx2])-G0[idx1,idx2]*G0[idx2,idx1]
                    end
                end
                tmp/=prod(model.site)
                R0+=tmp*cos(π*(rx+ry))
                R1+=cos(π*(rx+ry)+2*π/model.site[1]*rx+2*π/model.site[2]*ry )*tmp
            end
        end
    end
    return Ek,Eu,Ev,CDW0,CDW1,SDW0,SDW1
end