# 8/9 filling of particles, U=0

push!(LOAD_PATH,"C:/Users/admin/Desktop/JuliaDQMC/code/SU2PQMC/")
using DelimitedFiles
using ProgressMeter
using KAPDQMC_tU  
using LinearAlgebra
using Random

function GroverMatrix!(GM,G1,G2)
    mul!(GM,G1,G2)
    lmul!(2.0, GM)
    axpy!(-1.0, G1, GM)
    axpy!(-1.0, G2, GM)
    for i in diagind(GM)
        GM[i] += 1.0
    end
end

Lattice="HoneyComb120"    


Θ=collect(6:0.1:9.0)
dΘ=Θ[2]-Θ[1]

# L=collect(42:3:48)
# L=collect(51:3:60)
# L=collect(63:3:63)
L=[72]

path="./"

Initial="V"

for i in eachindex(L)

    site=[L[i],L[i]]
    # Half
    indexA=area_index(Lattice,site,([1,1],[div(L[i],3),L[i]]))

    # HalfHalf
    indexB=area_index(Lattice,site,([1,1],[div(L[i],3),div(2*L[i],3)]))

    EEHalfdata=zeros(length(Θ)+1)
    EEHalfHalfdata=zeros(length(Θ)+1)
    EEHalfdata[1]=L[i]
    EEHalfHalfdata[1]=L[i]

    K=K_Matrix(Lattice,site)
    dt=0.1
    E,V=LAPACK.syevd!('V', 'L',K[:,:])
    eK=V*Diagonal(exp.(-dt.*E))*V'

    Ns=size(K)[1]

    @assert (Ns*8)%(9*2)==0 "ns must be integer"
    ns=div(Ns,2)

    G = Array{Float64}(undef, Ns, Ns)
    GMA = Array{Float64}(undef, length(indexA), length(indexA))
    GMB = Array{Float64}(undef, length(indexB), length(indexB))
    BL = Array{Float64}(undef, ns, Ns)
    BR = Array{Float64}(undef, Ns, ns)
    tmpNn = Matrix{Float64}(undef, Ns, ns)
    tmpnN = Matrix{Float64}(undef, ns, Ns)
    tmpnn= Matrix{Float64}(undef, ns, ns)
    tau = Vector{Float64}(undef, ns)
    ipiv = Vector{LAPACK.BlasInt}(undef, ns)

    BL,BR = Free_G!(G,Lattice,site,Θ[1],Initial)

    p = Progress(length(Θ),desc="ns=$(ns/Ns), L=$(L[i])")  # 创建进度条
    next!(p)       # 更新进度
    for j in 2:length(Θ)
        next!(p)       # 更新进度
        for iii in 1:div(dΘ,dt)
            BL.=BL*eK
            BR.=eK*BR
            
            LAPACK.gerqf!(BL, tau)
            LAPACK.orgrq!(BL, tau, ns)

            LAPACK.geqrf!(BR, tau)
            LAPACK.orgqr!(BR, tau, ns)
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
        GroverMatrix!(GMA,view(G,indexA,indexA),view(G,indexA,indexA))
        GroverMatrix!(GMB,view(G,indexB,indexB),view(G,indexB,indexB))
        EEHalfdata[j+1]=-log(abs2(det(GMA)))
        EEHalfHalfdata[j+1]=-log(abs2(det(GMB)))
    end

    open("$(path)neqU0EE_Half.csv", "a") do io
        lock(io)
        writedlm(io, EEHalfdata', ',')
        unlock(io)
    end

    open("$(path)neqU0EE_HalfHalf.csv", "a") do io
        lock(io)
        writedlm(io, EEHalfHalfdata', ',')
        unlock(io)
    end

end


