function phy_update(path::String,model::tV_Hubbard_Para_,s::Array{UInt8,3},Sweeps::Int64,record::Bool)
    global LOCK=ReentrantLock()
    ERROR=1e-6

    UPD = UpdateBuffer()
    NN=length(model.nodes)
    Phy = PhyBuffer(model.Ns, NN) 
    Θidx=div(NN,2)+1

    name = if model.Lattice=="SQUARE" "□" 
    elseif model.Lattice=="HoneyComb60" "HC" 
    elseif model.Lattice=="HoneyComb120" "HC120" 
    else error("Lattice: $(model.Lattice) is not allowed !") end  
    file="$(path)/tVphy$(name)_t$(model.Ht)V$(model.Hv)size$(model.site)Δt$(model.Δt)Θ$(model.Θ)BS$(model.BatchSize).csv"

    rng=MersenneTwister(Threads.threadid()+time_ns())

    Ek=Ev=0.0
    R0=zeros(Float64,4)
    R1=zeros(Float64,4)
    counter=0

    G ,BLs,BRs,tmpN,tmpNN,tmpnn,tmpnN,tmpNn,tau,ipiv,BM= 
            Phy.G,Phy.BLs,Phy.BRs,Phy.N,Phy.NN,Phy.nn,Phy.nN,Phy.Nn,Phy.tau,Phy.ipiv,Phy.BM


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
    for _ in 1:Sweeps
        # println("\n Sweep: $loop ")
        for lt in axes(s,3)
            #####################################################################
                # println(lt)
                if norm(G-Gτ(model,s,lt-1))>ERROR
                    error("Wrap-$(lt)   :   $(norm(G-Gτ(model,s,lt-1))) , $(norm(G)) , $(norm(Gτ(model,s,lt-1))) ")
                end
            #####################################################################

            mul!(tmpNN,G,model.eKinv)
            mul!(G,model.eK,tmpNN)
            # G=model.eK*G*model.eKinv
            
            for j in reverse(axes(s,2))
                for i in axes(s,1)
                    x,y=model.nnidx[i,j]
                    tmpN[x]=model.η[s[i,j,lt]]
                    tmpN[y]=-model.η[s[i,j,lt]]
                end
                tmpN.= exp.(tmpN)
                WrapV!(tmpNN,G,tmpN,view(model.UV,:,:,j),"B")

                UpdatePhyLayer!(rng,j,view(s,:,j,lt),model,UPD,Phy)
                ####################################################################
                    # print("*")
                    # GG=model.eK*Gτ(model,s,lt-1)*model.eKinv
                    # for jj in 3:-1:j
                    #     E=zeros(model.Ns)
                    #     for ii in 1:size(s)[1]
                    #         x,y=model.nnidx[ii,jj]
                    #         E[x]=model.η[s[ii,jj,lt]]
                    #         E[y]=-model.η[s[ii,jj,lt]]
                    #     end
                    #     GG=model.UV[:,:,jj]*Diagonal(exp.(E))*model.UV[:,:,jj]' *GG* model.UV[:,:,jj]*Diagonal(exp.(-E))*model.UV[:,:,jj]'
                    # end
                    # if(norm(G-GG)>ERROR)
                    #     println("lt=$(lt) j=$(j)")
                    #     error(j," update error: ",norm(G-GG),"  lt=",lt)
                    # end
                ####################################################################
            end

            if record && abs(idx-Θidx)<=1
                tmp=phy_measure(model,Phy,lt,s)
                Ek+=tmp[1]
                Ev+=tmp[2]
                axpy!(1,tmp[3],R0)
                axpy!(1,tmp[4],R1)
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

        for lt in reverse(axes(s,3))
            #####################################################################
                # if norm(G-Gτ(model,s,lt))>ERROR
                #     error("Wrap-$(lt)   :   $(norm(G-Gτ(model,s,lt+1)))")
                # end
            ######################################################################
            for j in axes(s,2)
                UpdatePhyLayer!(rng,j,view(s,:,j,lt),model,UPD,Phy)
                for i in axes(s,1)
                    x,y=model.nnidx[i,j]
                    tmpN[x]=model.η[s[i,j,lt]]
                    tmpN[y]=-model.η[s[i,j,lt]]
                end
                tmpN.=exp.(.-tmpN)
                WrapV!(tmpNN,G,tmpN,view(model.UV,:,:,j),"B")
            end
            mul!(tmpNN,model.eKinv,G)
            mul!(G,tmpNN,model.eK)
            
            if record && abs(idx-Θidx)<=1
                tmp=phy_measure(model,Phy,lt-1,s)
                Ek+=tmp[1]
                Ev+=tmp[2]
                axpy!(1,tmp[3],R0)
                axpy!(1,tmp[4],R1)
                counter+=1
            end

            if any(model.nodes.== (lt-1))
                idx-=1
                BM_F!(tmpN,tmpNN,BM,model, s, idx)
                mul!(tmpnN,view(BLs,:,:,idx+1),BM)
                LAPACK.gerqf!(tmpnN, tau)
                LAPACK.orgrq!(tmpnN, tau)
                copyto!(view(BLs,:,:,idx) , tmpnN)
                # BL .= Matrix(qr(( BL * BM )').Q)'

                # copyto!(tmpNN , G)

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
            lock(LOCK) do
                open(file, "a") do io
                    writedlm(io,vcat([Ek, Ev], R0, R1)' ./ counter, ',')
                end
            end
            Ek=Ev=0.0
            fill!(R0,0.0)
            fill!(R1,0.0)
            counter=0
        end
    end
    return s
end 

function UpdatePhyLayer!(rng,j,s,model::tV_Hubbard_Para_,UPD::UpdateBuffer_,Phy::PhyBuffer_)
    for i in axes(s,1)
        x,y=model.nnidx[i,j]
        UPD.subidx.=[x,y]
        sx = rand(rng, model.samplers_dict[s[i]])
        p=get_r!(UPD,model.η[sx]- model.η[s[i]],Phy.G)
        p*=model.γ[sx]/model.γ[s[i]]
        if p<-1e-3
            println("Negative Sign: $(p)")
        end
        if rand(rng)<p
            Gupdate!(Phy,UPD)
            s[i]=sx
        end
    end
end

function phy_measure(model::tV_Hubbard_Para_,Phy::PhyBuffer_,lt,s)
    """
    (Ek,Ev,R0,R1)    
    """
    G0=Phy.G[:,:]
    tmpN=Phy.N
    tmpNN=Phy.NN
    tmp=zeros(Float64,4)
    R0=zeros(Float64,4)
    R1=zeros(Float64,4)
    
    if lt>model.Nt/2
        for t in lt:-1:div(model.Nt,2)+1
            for j in axes(s,2)
                for i in axes(s,1)
                    x,y=model.nnidx[i,j]
                    tmpN[x]=model.η[s[i,j,t]]
                    tmpN[y]=-model.η[s[i,j,t]]
                end
                tmpN.=exp.(.-tmpN)

                WrapV!(tmpNN,G0,tmpN,view(model.UV,:,:,j),"B")
                # G0=model.UV[j,:,:]'*diagm(exp.(-E))*model.UV[j,:,:] *G0* model.UV[j,:,:]'*diagm(exp.(E))*model.UV[j,:,:]
            end
            mul!(tmpNN,model.eKinv,G0)
            mul!(G0,tmpNN,model.eK)
            # G0= model.eKinv*G0*model.eK
        end
    else
        for t in lt+1:div(model.Nt,2)
            mul!(tmpNN,G0,model.eKinv)
            mul!(G0,model.eK,tmpNN)
            # G0=model.eK*G0*model.eKinv
            for j in reverse(axes(s,2))
                for i in axes(s,1)
                    x,y=model.nnidx[i,j]
                    tmpN[x]=model.η[s[i,j,t]]
                    tmpN[y]=-model.η[s[i,j,t]]
                end
                tmpN.= exp.(tmpN)
                WrapV!(tmpNN,G0,tmpN,view(model.UV,:,:,j),"B")
                # G0=model.UV[j,:,:]'*diagm(exp.(E))*model.UV[j,:,:] *G0* model.UV[j,:,:]'*diagm(exp.(-E))*model.UV[j,:,:]
            end
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

    Ek=model.Ht*sum(model.K.*G0)
    Ev=0.0
    for k in 1:length(model.nnidx)
        x,y=model.nnidx[k]
        Ev+=(1-G0[x,x])*(1-G0[y,y])-G0[x,y]*G0[y,x]
    end
    Ev*=model.Hv

    if occursin("HoneyComb", model.Lattice)
        for rx in 1:model.site[1]
            for ry in 1:model.site[2]
                fill!(tmp,0.0)
                for ix in 1:model.site[1]
                    for iy in 1:model.site[2]
                        idx1=xy_i(model.Lattice,model.site,ix,iy)-1
                        idx2=xy_i(model.Lattice,model.site,mod1(ix+rx,model.site[1]),mod1(iy+ry,model.site[2]))-1
                        if idx1==idx2
                            tmp[1]+=(1-G0[idx1,idx1])
                            tmp[2]+=(1-G0[idx1+1,idx1+1])
                            # tmp[1]+=(1-G0[idx1,idx1]) * (1-G0[idx2,idx2]) - G0[idx1,idx2]*G0[idx2,idx1]
                            # tmp[2]+=(1-G0[idx1+1,idx1+1]) * (1-G0[idx2+1,idx2+1]) - G0[idx1+1,idx2+1]*G0[idx2+1,idx1+1]
                        else
                            tmp[1]+=(1-G0[idx1,idx1]) * (1-G0[idx2,idx2]) - G0[idx1,idx2]*G0[idx2,idx1]
                            tmp[2]+=(1-G0[idx1+1,idx1+1]) * (1-G0[idx2+1,idx2+1]) - G0[idx1+1,idx2+1]*G0[idx2+1,idx1+1]
                        end
                        tmp[3]+=(1-G0[idx1+1,idx1+1])*(1-G0[idx2,idx2])-G0[idx1+1,idx2]*G0[idx2,idx1+1]
                        tmp[4]+=(1-G0[idx1,idx1])*(1-G0[idx2+1,idx2+1])-G0[idx1,idx2+1]*G0[idx2+1,idx1]
                    end
                end
                axpy!(1,tmp,R0)
                axpy!(cos(2*π/model.site[1]*rx+2*π/model.site[2]*ry),tmp,R1)
            end
        end
        lmul!(4/model.Ns^2,R0)
        lmul!(4/model.Ns^2,R1)
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
    # 1-R1/R0
    return Ek,Ev,R0,R1
end