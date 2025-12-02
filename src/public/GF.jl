function Free_G!(G,Lattice,site,Θ,Initial,ns)
    """
    input:
        Lattice: "HoneyComb" or "SQUARE"
        site: [Int64,Int64]
        Θ: ComplexF64
        Initial: "H0" or "V"
    return Green function with Free Hubbard Hamiltonian from H0 initial state or SDW initial state 
    对于得到H0初态(平衡态结果)
    必须要对H0加一个极其微弱的交错化学势,以去除基态简并,从而得到正确的结果
    """
    K=K_Matrix(Lattice,site)
    Ns=size(K)[1]
    # ns=div(Ns, 2)

    Δt=0.1
    Nt=Int(round(Θ/Δt))

    Pt = zeros(Float64, Ns, ns)  # 预分配 Pt
    if Initial=="H0"
        KK=Matrix{Float64}(K)

        KK[KK .!= 0] .+=( rand(size(KK)...) * 1e-1)[KK.!= 0]
        KK=(KK+KK')./2
        
        # μ=1e-4
        # if occursin("HoneyComb", Lattice)
        #     KK+=μ*Diagonal(repeat([-1, 1], div(Ns, 2)))
        # elseif Lattice=="SQUARE"
        #     for i in 1:Ns
        #         x,y=i_xy(Lattice,site,i)
        #         KK[i,i]+=μ*(-1)^(x+y)
        #     end
        # end

        E,V=LAPACK.syevd!('V', 'L',KK)
        Pt=V[:,1:ns]
    elseif Initial=="V" 
        if occursin("HoneyComb", Lattice)
            for i in 1:ns
                Pt[i*2,i]=1
            end
        else
            count=1
            for i in 1:Ns
                x,y=i_xy(Lattice,site,i)
                if (x+y)%2==1
                    Pt[i,count]=1
                    count+=1
                    if count>ns
                        break
                    end
                end
            end
        end
    end

    E,V=LAPACK.syevd!('V', 'L',K[:,:])
    eK=V*Diagonal(exp.(-Δt.*E))*V'

    BL = Array{Float64}(undef, ns, Ns)
    BR = Array{Float64}(undef, Ns, ns)
    tmpNn = Matrix{Float64}(undef, Ns, ns)
    tmpnN = Matrix{Float64}(undef, ns, Ns)
    tmpnn= Matrix{Float64}(undef, ns, ns)
    tau = Vector{Float64}(undef, ns)
    ipiv = Vector{LAPACK.BlasInt}(undef, ns)

    BL.=Pt'
    BR.=Pt

    count=0
    for i in 1:Nt
        mul!(tmpnN,BL,eK)
        BL.=tmpnN

        mul!(tmpNn,eK,BR)
        BR.=tmpNn

        count+=1
        if count ==10
            LAPACK.gerqf!(BL, tau)
            LAPACK.orgrq!(BL, tau, ns)

            LAPACK.geqrf!(BR, tau)
            LAPACK.orgqr!(BR, tau, ns)
            count=0
        end

    end

    mul!(tmpnn,BL,BR)
    LAPACK.getrf!(tmpnn,ipiv)
    LAPACK.getri!(tmpnn, ipiv)
    mul!(tmpNn,BR,tmpnn)
    mul!(G, tmpNn,BL)
    lmul!(-1.0,G)
    for i in diagind(G)
        G[i]+=1
    end

    return BL,BR

end

"""
    No Return. Overwrite G = G - G · inv(r) ⋅ Δ · (I-G)
    ------------------------------------------------------------------------------
"""
function Gupdate!(Phy::PhyBuffer_,UPD::UpdateBuffer_)
    mul!(Phy.zN,UPD.r,view(Phy.G,UPD.subidx,:))
    lmul!(-1.0,Phy.zN)
    axpy!(1.0,UPD.r,view(Phy.zN,:,UPD.subidx))   # useful for GΘτ,Gτ
    mul!(Phy.NN, view(Phy.G,:,UPD.subidx),Phy.zN)
    axpy!(-1.0, Phy.NN, Phy.G)
end


"""
    Update G4 by overwriting Gt,G0,Gt0,G0t with r and subidx.
        G0 = G0 + G0t(:,subidx) ⋅ r ⋅ Gt0[subidx,:]
        Gt0 = Gt0 + r ⋅ Gt[subidx,:] ⋅ G0t(:,subidx)

        G0t = G0t - G0t(:,subidx) ⋅ r ⋅ (I-Gt[subidx,:])
        Gt = Gt - r ⋅ Gt[subidx,:] ⋅ r ⋅ (I-Gt[subidx,:])
    with r ≡ inv(r) ⋅ Δ
    ------------------------------------------------------------------------------
"""
function G4update!(SCEE::SCEEBuffer_,UPD::UpdateBuffer_,G::G4Buffer_)
    mul!(SCEE.zN,UPD.r,view(G.Gt0,UPD.subidx,:))   # useful for GΘ,GτΘ
    mul!(SCEE.NN, view(G.G0t,:,UPD.subidx),SCEE.zN)
    axpy!(1.0, SCEE.NN, G.G0)
    mul!(SCEE.NN, view(G.Gt,:,UPD.subidx),SCEE.zN)
    axpy!(1.0, SCEE.NN, G.Gt0)

    mul!(SCEE.zN,UPD.r,view(G.Gt,UPD.subidx,:))
    lmul!(-1.0,SCEE.zN)
    axpy!(1.0,UPD.r,view(SCEE.zN,:,UPD.subidx))   # useful for GΘτ,Gτ
    mul!(SCEE.NN, view(G.G0t,:,UPD.subidx),SCEE.zN)
    axpy!(-1.0, SCEE.NN, G.G0t)
    mul!(SCEE.NN, view(G.Gt,:,UPD.subidx),SCEE.zN)
    axpy!(-1.0, SCEE.NN, G.Gt)
end

"""
    Update gmInv with a,b,Tau.
        gmInv = gmInv - gmInv ⋅ a ⋅ inv(Tau) ⋅ b
    Universal for s1 and s2.
    ------------------------------------------------------------------------------
"""
function GMupdate!(A::AreaBuffer_)
    mul!(A.Nz, A.gmInv,A.a )
    inv22!(A.Tau)
    mul!(A.zN,A.Tau,A.b)
    mul!(A.NN, A.Nz, A.zN)
    axpy!(-1.0, A.NN, A.gmInv)
end


"""
    Return det(Tau)
    Update s1 and overwrite a , b , Tau.
        a = G0t1(:,subidx) ⋅ r
        b = Gt01[subidx,:] ⋅ (2G02-I) ⋅ gmInv
    with r ≡ inv(r) ⋅ Δ
    Warning : G02 here !!!  Gt01,G0t1
    ------------------------------------------------------------------------------
"""
function get_abTau1!(A::AreaBuffer_,UPD::UpdateBuffer_,G0,Gt0,G0t)
    copyto!(A.NN, view(G0,A.index,A.index))
    lmul!(2.0, A.NN)
    for i in diagind(A.NN)
        A.NN[i] -= 1
    end
    mul!(A.zN,view(Gt0,UPD.subidx,A.index),A.NN)
    mul!(A.b, A.zN, A.gmInv)
    mul!(A.a,view(G0t,A.index,UPD.subidx),UPD.r)
    mul!(A.Tau, A.b, A.a)
    for i in diagind(A.Tau)
        A.Tau[i] += 1
    end
    return det(A.Tau)
end


"""
    Return det(Tau)
    Update s2 and overwrite a , b , Tau.
        a = (2G01-I) ⋅ G0t2(:,subidx)
        b = r ⋅ Gt02[subidx,:] ⋅ gmInv
    with r ≡ inv(r) ⋅ Δ
    Warning : G01 here !!!  Gt02,G0t2
    ------------------------------------------------------------------------------
"""
function get_abTau2!(A::AreaBuffer_,UPD::UpdateBuffer_,G0,Gt0,G0t)
    copyto!(A.NN , view(G0,A.index,A.index))
    lmul!(2.0, A.NN)
    for i in diagind(A.NN)
        A.NN[i] -= 1
    end
    mul!(A.a,A.NN,view(G0t,A.index,UPD.subidx))
    mul!(A.zN,UPD.r,view(Gt0,UPD.subidx,A.index))
    mul!(A.b,A.zN,A.gmInv)
    mul!(A.Tau, A.b, A.a)
    for i in diagind(A.Tau)
        A.Tau[i] += 1
    end
    return det(A.Tau)
end

"""
    No Return. Overwrite G 
        G = I - BR ⋅ inv(BL ⋅ BR) ⋅ BL 
    ------------------------------------------------------------------------------
"""
function get_G!(tmpnn,tmpnN,ipiv,BL,BR,G)
    # 目标: 计算 G = I - BR * inv(BL * BR) * BL，避免显式求逆 (getri!)
    # 步骤:
    # 1. tmpnn ← M = BL * BR
    # 2. LU 分解 tmpnn 得到 pivot ipiv
    # 3. tmpnN ← BL (右端)，求解 M * X = BL 得 X = inv(M)*BL  (使用 getrs!)
    # 4. G ← BR * X
    # 5. G ← I - G
    # 数值优势: 避免显式逆，提升稳定性与性能，减少 FLOPs。
    mul!(tmpnn, BL, BR)                 # tmpnn = M = BL*BR
    LAPACK.getrf!(tmpnn, ipiv)          # LU 分解 (in-place)
    tmpnN .= BL                         # 右端初始化: RHS = BL
    LAPACK.getrs!('N', tmpnn, ipiv, tmpnN) # 解 M * X = BL, 结果写回 tmpNn
    mul!(G, BR, tmpnN)                  # G = BR * inv(M) * BL
    lmul!(-1.0, G)                      # G = - G
    @inbounds for i in diagind(G)       # G = I - BR * inv(M) * BL
        G[i] += 1.0
    end
end

function G4!(SCEE::SCEEBuffer_,G::G4Buffer_,nodes::Vector{Int64},idx::Int64,direction="Forward")
    Θidx=div(length(nodes),2)+1
    BLMs, BRMs, BMs, BMinvs, Gt, G0, Gt0, G0t =
        G.BLMs, G.BRMs, G.BMs, G.BMinvs, G.Gt, G.G0, G.Gt0, G.G0t
    II, tmpnn, tmpnN, tmpNN, tmpNN_, ipiv =
        SCEE.II, SCEE.nn, SCEE.nN, SCEE.NN, SCEE.NN_, SCEE.ipiv

    get_G!(tmpnn,tmpnN,ipiv,view(BLMs,:,:,idx),view(BRMs,:,:,idx),Gt)
    if idx==Θidx
        G0 .= Gt
        if direction=="Forward"
            Gt0.= Gt
            G0t.= Gt .- II 
        elseif direction=="Backward"
            Gt0.= Gt .- II
            G0t.= Gt
        end
    else
        get_G!(tmpnn,tmpnN,ipiv,view(BLMs,:,:,Θidx),view(BRMs,:,:,Θidx),G0)
    
        Gt0 .= II
        G0t .= II
        if idx<Θidx
            for j in idx:Θidx-1
                if j==idx
                    tmpNN_ .= Gt
                else
                    get_G!(tmpnn,tmpnN,ipiv,view(BLMs,:,:,j),view(BRMs,:,:,j),tmpNN_)
                end
                mul!(tmpNN,tmpNN_, G0t)
                mul!(G0t, view(BMs,:,:,j), tmpNN)
                tmpNN .= II .- tmpNN_
                mul!(tmpNN_,Gt0, tmpNN)
                mul!(Gt0, tmpNN_, view(BMinvs,:,:,j))
                
            end
            lmul!(-1.0, Gt0)
        else
            for j in Θidx:idx-1
                if j==Θidx
                    tmpNN_ .= G0
                else
                    get_G!(tmpnn,tmpnN,ipiv,view(BLMs,:,:,j),view(BRMs,:,:,j),tmpNN_)
                end
                mul!(tmpNN, tmpNN_, Gt0)
                mul!(Gt0, view(BMs,:,:,j), tmpNN)
                tmpNN .= II .- tmpNN_
                mul!(tmpNN_, G0t, tmpNN)
                mul!(G0t, tmpNN_,view(BMinvs,:,:,j))
            end
            lmul!(-1.0, G0t)
        end        
    end
end

"""
    No Return. Overwrite A with inv(B)
    ------------------------------------------------------------------------------
"""
function inv22!(A,B)
    detB=det(B)
    A[1,1]=B[2,2]/detB
    A[1,2]=-B[1,2]/detB
    A[2,1]=-B[2,1]/detB
    A[2,2]=B[1,1]/detB
end

function inv22!(A)
    if size(A)==(2,2)
        A./=det(A)
        tmp=A[1,1]
        A[1,1]=A[2,2]
        A[2,2]=tmp
        A[1,2]=-A[1,2]
        A[2,1]=-A[2,1]
    elseif  size(A)==(1,1)
        A[1,1]=1.0/A[1,1]
    end
end

function GroverMatrix(G1,G2)
    II=I(size(G1)[1])
    return G1*G2+(II-G1)*(II-G2)
end

function GroverMatrix!(GM,G1,G2)
    mul!(GM,G1,G2)
    lmul!(2.0, GM)
    axpy!(-1.0, G1, GM)
    axpy!(-1.0, G2, GM)
    for i in diagind(GM)
        GM[i] += 1.0
    end
end


