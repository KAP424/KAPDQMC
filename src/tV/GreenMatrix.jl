function Initial_s(model::tV_Hubbard_Para_,rng::MersenneTwister)::Array{UInt8,3}
    sp=Random.Sampler(rng,[1,2,3,4])
    a,b=size(model.nnidx)
    s=zeros(UInt8,a,b,model.Nt)

    for i in eachindex(s)
        s[i] =rand(rng,sp)
    end  
    return s
end

function BM_F!(tmpN,tmpNN,BM,model::tV_Hubbard_Para_, s::Array{UInt8, 3}, idx::Int64)
    """
    不包头包尾
    """
    @assert 0< idx <=length(model.nodes)

    fill!(tmpNN,0)
    @inbounds for i in diagind(tmpNN)
        tmpNN[i] = 1
    end

    for lt in model.nodes[idx] + 1:model.nodes[idx + 1]
        mul!(BM,model.eK,tmpNN)
        for j in reverse(axes(s,2))
            for i in axes(s,1)
                x,y=model.nnidx[i,j]
                tmpN[x]=model.α[lt] * model.η[s[i,j,lt]] 
                tmpN[y]=-model.α[lt] * model.η[s[i,j,lt]]
            end
            tmpN.= exp.(tmpN)
            mul!(tmpNN,view(model.UV,:,:,j),BM)
            mul!(BM,Diagonal(tmpN),tmpNN)
            mul!(tmpNN,view(model.UV,:,:,j),BM)
            copyto!(BM,tmpNN)
        end
    end
end

function BMinv_F!(tmpN,tmpNN,BM,model::tV_Hubbard_Para_, s::Array{UInt8, 3}, idx::Int64)
    """
    不包头包尾
    """
    @assert 0< idx <=length(model.nodes)

    fill!(tmpNN,0)
    @inbounds for i in diagind(tmpNN)
        tmpNN[i] = 1
    end

    for lt in model.nodes[idx] + 1:model.nodes[idx + 1]
        mul!(BM,tmpNN,model.eKinv)
        for j in  reverse(axes(s,2))
            for i in axes(s,1)
                x,y=model.nnidx[i,j]
                tmpN[x]=model.α[lt] * model.η[s[i,j,lt]]
                tmpN[y]=-model.α[lt] * model.η[s[i,j,lt]]
            end
            tmpN.= exp.(.-tmpN)

            mul!(tmpNN,BM,view(model.UV,:,:,j))
            mul!(BM,tmpNN,Diagonal(tmpN))
            mul!(tmpNN,BM,view(model.UV,:,:,j))
            copyto!(BM,tmpNN)
        end
    end
end

"""
    Return p=det(r). Overwrite 
        Δ = uv ⋅ diag( exp(α⋅[-2s,2s]) - I ) ⋅ uvᵀ 
        r = I + Δ ⋅ (I - Gt[subidx,subidx])
        r ≡ inv(r) ⋅ ̇Δ .
    ------------------------------------------------------------------------------
"""
function get_r!(UPD::UpdateBuffer_,Δs::Float64,Gt::Matrix{Float64})
    UPD.tmp2.= Δs.*[1.0, -1.0]
    UPD.tmp2 .= exp.(UPD.tmp2).-1
    mul!(UPD.r,UPD.uv,Diagonal(UPD.tmp2))
    mul!(UPD.Δ,UPD.r,UPD.uv)
    mul!(UPD.r,UPD.Δ,view(Gt,UPD.subidx,UPD.subidx))
    axpby!(1.0,UPD.Δ, -1.0, UPD.r)   # r = I + Δ ⋅ (I - Gt1[subidx,subidx])
    UPD.r[1,1]+=1; UPD.r[2,2]+=1;
    p=det(UPD.r)
    # redefine r=inv(r) ⋅ ̇Δ 
    inv22!(UPD.tmp22,UPD.r)
    mul!(UPD.r,UPD.tmp22,UPD.Δ)
    return p
end

"""
    Overwrite G according to UV , D and option LR
        LR=1 : Only Left:   G = (UV * D * UV') * G
        LR=2 : Only Right   G = G * (UV * D * UV')'
        LR=3 : Both Side and D will be changed to 1/D !!!   G = (UV * D * UV') * G * (UV * inv(D) * UV')'
    Only wrap interaction part 
    ------------------------------------------------------------------------------
"""
function WrapV!(tmpNN::Matrix{Float64},G::Matrix{Float64},D::Vector{Float64},UV::SubArray{Float64, 2, Array{Float64, 3}},LR::String)
    if LR=="L"
        mul!(tmpNN,UV,G)
        mul!(G,Diagonal(D),tmpNN)
        mul!(tmpNN,UV,G)
        copyto!(G, tmpNN)
    elseif LR=="R"
        mul!(tmpNN, G , UV)
        mul!(G, tmpNN , Diagonal(D))
        mul!(tmpNN, G , UV)
        copyto!(G, tmpNN)
    elseif LR=="B"
        mul!(tmpNN,UV,G)
        mul!(G,tmpNN,UV)
        mul!(tmpNN,Diagonal(D),G)
        D.= 1 ./D
        mul!(G,tmpNN,Diagonal(D))
        mul!(tmpNN,UV,G)
        mul!(G,tmpNN,UV)
    end
end

# Below is just used for debug

"equal time Green function"
function Gτ(model::tV_Hubbard_Para_,s::Array{UInt8,3},τ::Int64)::Array{Float64,2}
    BL::Array{Float64,2}=model.Pt'[:,:]
    BR::Array{Float64,2}=model.Pt[:,:]

    E=zeros(model.Ns)
    counter=0
    for lt in model.Nt:-1:τ+1
        for j in axes(s,2)
            fill!(E,0.0)
            for i in axes(s,1)
                x,y=model.nnidx[i,j]
                E[x]=model.α[lt] * model.η[s[i,j,lt]]
                E[y]=-model.α[lt] * model.η[s[i,j,lt]]
            end
            BL=BL*model.UV[:,:,j]*Diagonal(exp.(E))*model.UV[:,:,j]

            #####################################################################
            # V=zeros(Float64,model.Ns,model.Ns)
            # for i in 1:size(s)[2]
            #     x,y=model.nnidx[i,j]
            #     V[x,y]=V[y,x]=s[lt,i,j]
            # end
            # tmp=model.UV[:,:,j]'*diagm(E)*model.UV[:,:,j]
            # if norm(tmp-V)>1e-6
            #     println("diagnose error")
            # end
            #####################################################################
        end
        BL=BL*model.eK
        counter+=1
        if counter==model.BatchSize
            counter=0
            BL=Matrix(qr(BL').Q)'
        end
    end
    counter=0
    for lt in 1:τ
        BR=model.eK*BR
        for j in reverse(axes(s,2))
            fill!(E,0.0)
            for i in axes(s,1)
                x,y=model.nnidx[i,j]
                E[x]=model.α[lt] * model.η[s[i,j,lt]]
                E[y]=-model.α[lt] * model.η[s[i,j,lt]]
            end
            BR=model.UV[:,:,j]*Diagonal(exp.(E))*model.UV[:,:,j] *BR
            #####################################################################
            # V=zeros(Float64,model.Ns,model.Ns)
            # for i in 1:size(s)[2]
            #     x,y=model.nnidx[i,j]
            #     V[x,y]=V[y,x]=s[lt,i,j]
            # end
            # tmp=model.UV[:,:,j]'*diagm(E)*model.UV[:,:,j]
            # if norm(tmp-V)>1e-6
            #     println("diagnose error")
            # end
            #####################################################################
        end
        counter+=1
        if counter==model.BatchSize
            counter=0
            BR=Matrix(qr(BR).Q)
        end
    end

    BL=Matrix(qr(BL').Q)'
    BR=Matrix(qr(BR).Q)

    return I(model.Ns)-BR*inv(BL*BR)*BL
    # return BL,BR
end


"displaced Green function G(τ₁,τ₂)"
function G4(model::tV_Hubbard_Para_,s::Array{UInt8,3},τ1::Int64,τ2::Int64,direction="Forward")
    if τ1>τ2
        BBs=zeros(Float64,cld(τ1-τ2,model.BatchSize),model.Ns,model.Ns)
        BBsInv=zeros(Float64,size(BBs))
        
        UL=zeros(Float64,1+size(BBs)[1],div(model.Ns,2),model.Ns)
        UR=zeros(Float64,size(UL)[1],model.Ns,div(model.Ns,2))
        G=zeros(Float64,size(UL)[1],model.Ns,model.Ns)

        UL[end,:,:]=model.Pt'[:,:]
        UR[1,:,:]=model.Pt[:,:]
    
        counter=0
        for lt in 1:τ2
            UR[1,:,:]=model.eK*UR[1,:,:]
            for j in reverse(axes(s,2))
                E=zeros(model.Ns)
                for i in axes(s,1)
                    x,y=model.nnidx[i,j]
                    E[x]=model.α[lt] * model.η[s[i,j,lt]]
                    E[y]=-model.α[lt] * model.η[s[i,j,lt]]
                end
                UR[1,:,:]=model.UV[:,:,j]*Diagonal(exp.(E))*model.UV[:,:,j]'*UR[1,:,:]
            end

            counter+=1
            if counter==model.BatchSize
                counter=0
                UR[1,:,:]=Matrix(qr(UR[1,:,:]).Q)
            end
        end
        UR[1,:,:]=Matrix(qr(UR[1,:,:]).Q) 
    
        counter=0
        for lt in model.Nt:-1:τ1+1
            for j in axes(s,2)
                E=zeros(model.Ns)
                for i in axes(s,1)
                    x,y=model.nnidx[i,j]
                    E[x]=model.α[lt] * model.η[s[i,j,lt]]
                    E[y]=-model.α[lt] * model.η[s[i,j,lt]]
                end
                UL[end,:,:]=UL[end,:,:]*model.UV[:,:,j]*Diagonal(exp.(E))*model.UV[:,:,j]'
            end
            UL[end,:,:]=UL[end,:,:]*model.eK

            counter+=1
            if counter==model.BatchSize
                counter=0
                UL[end,:,:]=Matrix(qr(UL[end,:,:]').Q)'
            end
        end
        UL[end,:,:]=Matrix(qr(UL[end,:,:]').Q)'
    
        for lt in 1:size(BBs)[1]-1
            BBs[lt,:,:]=I(model.Ns)
            BBsInv[lt,:,:]=I(model.Ns)
            for lt2 in 1:model.BatchSize
                BBs[lt,:,:]=model.eK*BBs[lt,:,:]
                BBsInv[lt,:,:]=BBsInv[lt,:,:]*model.eKinv
                for j in reverse(axes(s,2))
                    E=zeros(model.Ns)
                    for i in axes(s,1)
                        x,y=model.nnidx[i,j]
                        E[x]=model.α[τ2+(lt-1)*model.BatchSize+lt2] * model.η[s[i,j,τ2+(lt-1)*model.BatchSize+lt2]]
                        E[y]=-model.α[τ2+(lt-1)*model.BatchSize+lt2] * model.η[s[i,j,τ2+(lt-1)*model.BatchSize+lt2]]
                    end
                    BBs[lt,:,:]=model.UV[:,:,j]*Diagonal(exp.(E))*model.UV[:,:,j]'*BBs[lt,:,:]
                    BBsInv[lt,:,:]=BBsInv[lt,:,:]*model.UV[:,:,j]*Diagonal(exp.(-E))*model.UV[:,:,j]'

                end
            end
        end
    
        BBs[end,:,:]=I(model.Ns)
        BBsInv[end,:,:]=I(model.Ns)
        for lt in τ2+(size(BBs)[1]-1)*model.BatchSize+1:τ1
            BBs[end,:,:]=model.eK*BBs[end,:,:]
            BBsInv[end,:,:]=BBsInv[end,:,:]*model.eKinv
            for j in reverse(axes(s,2))
                E=zeros(model.Ns)
                for i in axes(s,1)
                    x,y=model.nnidx[i,j]
                    E[x]=model.α[lt] * model.η[s[i,j,lt]]
                    E[y]=-model.α[lt] * model.η[s[i,j,lt]]
                end
                BBs[end,:,:]=model.UV[:,:,j]*Diagonal(exp.(E))*model.UV[:,:,j]'*BBs[end,:,:]
                BBsInv[end,:,:]=BBsInv[end,:,:]*model.UV[:,:,j]*Diagonal(exp.(-E))*model.UV[:,:,j]' 
            end
        end
    
        for i in 1:size(BBs)[1]
            UL[end-i,:,:]=Matrix(qr( (UL[end-i+1,:,:]*BBs[end-i+1,:,:])' ).Q)'
            UR[i+1,:,:]=Matrix(qr(BBs[i,:,:]*UR[i,:,:]).Q)
        end

        for i in 1:size(G)[1]
            G[i,:,:]=I(model.Ns)-UR[i,:,:]*inv(UL[i,:,:]*UR[i,:,:])*UL[i,:,:]

            #####################################################################
            # if i <size(G)[1]
            #     if norm(Gτ(model,s,τ2+(i-1)*model.BatchSize)-G[i,:,:])>1e-3
            #         error("$i Gt:  $(norm(Gτ(model,s,τ2+(i-1)*model.BatchSize)-G[i,:,:]))")
            #     end
            # else
            #     if norm(Gτ(model,s,τ1)-G[i,:,:])>1e-3
            #         error("$i Gt:  $(norm(Gτ(model,s,τ1)-G[i,:,:]))")
            #     end
            # end
            #####################################################################
        end

        G12=I(model.Ns)
        G21=-I(model.Ns)
        for i in 1:size(BBs)[1]
            G12=G12*BBs[end-i+1,:,:]*G[end-i,:,:]
            G21=G21*( I(model.Ns)-G[i,:,:] )*BBsInv[i,:,:]
        end
        
        return G[end,:,:],G[1,:,:],G12,G21
    
    elseif τ1<τ2
        G2,G1,G21,G12=G4(model,s,τ2,τ1)
        return G1,G2,G12,G21
    else
        G=Gτ(model,s,τ1)
        if direction=="Forward"
            return G,G,G,-(I(model.Ns)-G)
        elseif direction=="Backward"
            return G,G,-(I(model.Ns)-G),G
        end
    end
end
