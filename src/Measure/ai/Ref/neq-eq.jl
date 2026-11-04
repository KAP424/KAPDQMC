
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

dΘ=0.5
dt=0.1

# L=collect(3:3:42)
# L=collect(45:3:54)
# L=collect(57:3:66)
L=collect(69:3:72)

path="./"
Initial="H0"



for i in eachindex(L)
    println("\n L=",L[i])
    EA=EB=EC=0.0
    EA2=EB2=EC2=2.0

    site=[L[i],L[i]]
    # Half
    indexA=area_index(Lattice,site,([1,1],[div(L[i],3),L[i]]))

    # HalfHalf
    indexB=area_index(Lattice,site,([1,1],[div(L[i],3),div(2*L[i],3)]))

    Θ=[0.0]
    EEHalfdata=[Float64(L[i])]
    EEHalfHalfdata=[Float64(L[i])]
    EESCdata=[Float64(L[i])]

    K=K_Matrix(Lattice,site)
    E,V=LAPACK.syevd!('V', 'L',K[:,:])
    eK=V*Diagonal(exp.(-dt.*E))*V'

    Ns=size(K)[1]

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

    BL,BR = Free_G!(G,Lattice,site,Θ[1],Initial,ns)
    push!(EEHalfdata, EA)
    push!(EEHalfHalfdata, EB)
    push!(EESCdata, EC)
    count=0
    while abs(EA-EA2)+abs(EB-EB2)+abs(EC-EC2)>1e-6
        print(count+=1,"-")
        EA2=EA
        EB2=EB
        EC2=EC
        for iii in 1:round(Int,dΘ/dt)
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
        for iii in diagind(G)
            G[iii]+=1
        end
        GroverMatrix!(GMA,view(G,indexA,indexA),view(G,indexA,indexA))
        GroverMatrix!(GMB,view(G,indexB,indexB),view(G,indexB,indexB))

        EA=-log(abs2(det(GMA)))
        EB=-log(abs2(det(GMB)))
        EC=EA-EB
        push!(Θ, Θ[end]+dΘ)
        push!(EEHalfdata, EA)
        push!(EEHalfHalfdata, EB)
        push!(EESCdata, EC)

        
    end

    open("$(path)neq-eqA_H0.csv", "a") do io
        lock(io)
        writedlm(io, [L[i] , EA]', ',')
        unlock(io)
    end

    open("$(path)neq-eqB_H0.csv", "a") do io
        lock(io)
        writedlm(io, [L[i] , EB]'', ',')
        unlock(io)
    end

    open("$(path)neq-eqC_H0.csv", "a") do io
        lock(io)
        writedlm(io, [L[i] , EC]', ',')
        unlock(io)
    end
end
