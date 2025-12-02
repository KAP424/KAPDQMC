function Initial_s(model::tU_Hubbard_Para_,rng::MersenneTwister)::Array{UInt8,2}
    sp=Random.Sampler(rng,[1,2,3,4])
    s=zeros(model.Ns,model.Nt)
    for i in eachindex(s)
        s[i] =rand(rng,sp)
    end    
    return s
end

function BM_F!(tmpN,tmpNN,BM,model::tU_Hubbard_Para_, s::Array{UInt8, 2}, idx::Int64)
    """
    不包头包尾
    """
    @assert 0< idx <=length(model.nodes)

    fill!(BM,0)
    @inbounds for i in diagind(BM)
        BM[i] = 1
    end
    for lt in model.nodes[idx] + 1:model.nodes[idx + 1]
        @inbounds @simd for i in 1:model.Ns
            tmpN[i] =  cis( model.α *model.η[s[i, lt]])
        end
        mul!(tmpNN,model.eK, BM)
        mul!(BM,Diagonal(tmpN), tmpNN)
    end
end

function BMinv_F!(tmpN,tmpNN,BM,model::tU_Hubbard_Para_, s::Array{UInt8, 2}, idx::Int64)
    """
    不包头包尾
    """
    @assert 0< idx <=length(model.nodes)

    fill!(BM,0)
    @inbounds for i in diagind(BM)
        BM[i] = 1
    end

    for lt in model.nodes[idx] + 1:model.nodes[idx + 1]
        @inbounds for i in 1:model.Ns
            tmpN[i] =  cis( -model.α *model.η[s[i, lt]])
        end
        mul!(tmpNN,BM, model.eKinv)
        mul!(BM,tmpNN,Diagonal(tmpN))
    end
end

function get_r!(UPD::UpdateBuffer_,Δs::Float64,Gt)
    @fastmath Δ = cis(Δs) - 1
    @fastmath p = 1 + Δ * (1 - Gt[UPD.subidx[1], UPD.subidx[1]])
    UPD.r[1,1] = Δ / p
    return abs2(p)
end

function WrapKV!(tmpNN,eK,eKinv,D,G,direction,LR)
    if direction=="Forward"
        if LR=="L"
            mul!(tmpNN, eK, G)
            mul!(G,Diagonal(D),tmpNN)
        elseif LR=="R"
            mul!(tmpNN, G,eKinv)
            mul!(G,tmpNN , Diagonal(D))
        elseif LR=="B"
            mul!(tmpNN, eK, G)
            mul!(G,tmpNN,eKinv)
            mul!(tmpNN,Diagonal(D),G)
            D.=1 ./D
            mul!(G,tmpNN,Diagonal(D))
        end
    elseif direction=="Backward"
        if LR=="L"
            mul!(tmpNN,Diagonal(D),G)
            mul!(G, eKinv, tmpNN)
        elseif LR=="R"
            mul!(tmpNN,G,Diagonal(D))
            mul!(G, tmpNN,eK)
        elseif LR=="B"
            mul!(tmpNN,Diagonal(D),G)
            D.=1 ./D
            mul!(G,tmpNN,Diagonal(D))
            mul!(tmpNN, eKinv, G)
            mul!(G,tmpNN,eK)
        end
    end
end

# Below is just used for debug

"equal time Green function"
function Gτ(model::tU_Hubbard_Para_,s::Array{UInt8,2},τ::Int64)::Array{ComplexF64,2}
    """
    equal time Green function
    """
    BL::Array{ComplexF64,2}=model.Pt'[:,:]
    BR::Array{ComplexF64,2}=model.Pt[:,:]

    counter=0
    for i in model.Nt:-1:τ+1
        D=[model.η[x] for x in s[:,i]]
        BL=BL*diagm(exp.(1im*model.α.*D))*model.eK
        counter+=1
        if counter==model.BatchSize
            counter=0
            BL=Matrix(qr(BL').Q)'
        end
    end
    counter=0
    for i in 1:1:τ
        D=[model.η[x] for x in s[:,i]]
        BR=diagm(exp.(1im*model.α.*D))*model.eK*BR
        counter+=1
        if counter==model.BatchSize
            counter=0
            BR=Matrix(qr(BR).Q)
        end
    end

    BL=Matrix(qr(BL').Q)'
    BR=Matrix(qr(BR).Q)

    return I(model.Ns)-BR*inv(BL*BR)*BL
end

"displaced Green function G(τ₁,τ₂)"
function G4(model::tU_Hubbard_Para_,s::Array{UInt8,2},τ1::Int64,τ2::Int64,direction="Forward")
    if τ1>τ2
        BBs=zeros(model.Type,cld(τ1-τ2,model.BatchSize),model.Ns,model.Ns)
        BBsInv=zeros(model.Type,size(BBs))
        
        UL=zeros(model.Type,1+size(BBs)[1],div(model.Ns,2),model.Ns)
        UR=zeros(model.Type,size(UL)[1],model.Ns,div(model.Ns,2))
        G=zeros(model.Type,size(UL)[1],model.Ns,model.Ns)

        UL[end,:,:]=model.Pt'[:,:]
        UR[1,:,:]=model.Pt[:,:]
    
        counter=0
        for lt in 1:τ2
            E=zeros(model.Type,model.Ns)
            for i in axes(s,1)
                x,y=model.nnidx[i]
                E[x]+=model.η[s[i,lt]]
                E[y]+=model.a*model.η[s[i,lt]]
            end
            if model.Type==Float64
                E .= exp.(E)
            else
                E .= exp.(1im.*E)
            end
            UR[1,:,:]=Diagonal(E)*model.eK*UR[1,:,:]

            counter+=1
            if counter==model.BatchSize
                counter=0
                UR[1,:,:]=Matrix(qr(UR[1,:,:]).Q)
            end
        end
        UR[1,:,:]=Matrix(qr(UR[1,:,:]).Q) 
    
        counter=0
        for lt in model.Nt:-1:τ1+1
            E=zeros(model.Type,model.Ns)
            for i in axes(s,1)
                x,y=model.nnidx[i]
                E[x]+=model.η[s[i,lt]]
                E[y]+=model.a*model.η[s[i,lt]]
            end
            if model.Type==Float64
                E .= exp.(E)
            else
                E .= exp.(1im.*E)
            end
            UL[end,:,:]=UL[end,:,:]*Diagonal(E)*model.eK

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
                E=zeros(model.Type,model.Ns)
                for i in axes(s,1)
                    x,y=model.nnidx[i]
                    E[x]+=model.η[s[i,τ2+(lt-1)*model.BatchSize+lt2]]
                    E[y]+=model.a*model.η[s[i,τ2+(lt-1)*model.BatchSize+lt2]]
                end
                if model.Type==Float64
                    E .= exp.(E)
                else
                    E .= exp.(1im.*E)
                end
                BBs[lt,:,:]=Diagonal(E)*model.eK*BBs[lt,:,:]
                BBsInv[lt,:,:]=BBsInv[lt,:,:]*model.eKinv*Diagonal(1 ./E)
            end
        end
    
        BBs[end,:,:]=I(model.Ns)
        BBsInv[end,:,:]=I(model.Ns)
        for lt in τ2+(size(BBs)[1]-1)*model.BatchSize+1:τ1
            E=zeros(model.Type,model.Ns)
            for i in axes(s,1)
                x,y=model.nnidx[i]
                E[x]+=model.η[s[i,lt]]
                E[y]+=model.a*model.η[s[i,lt]]
            end
            if model.Type==Float64
                E .= exp.(E)
            else
                E .= exp.(1im.*E)
            end
            BBs[end,:,:]=Diagonal(E)*model.eK*BBs[end,:,:]
            BBsInv[end,:,:]=BBsInv[end,:,:]*model.eKinv*Diagonal(1 ./E)
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
