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


