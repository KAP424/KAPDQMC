# using density channel ±1,±2 HS transformation

struct tV_Hubbard_Para_
    Lattice::String
    Ht::Float64
    Hv1::Float64
    Hv2::Float64
    site::Vector{Int64}
    Θrelax::Float64
    Θquench::Float64
    Ns::Int64
    Nt::Int64
    K::Array{Float64,2}
    BatchSize::Int64
    Δt::Float64
    α::Vector{Float64}
    γ::Vector{Float64}
    η::Vector{Float64}
    Pt::Array{Float64,2}
    HalfeK::Array{Float64,2}
    eK::Array{Float64,2}
    HalfeKinv::Array{Float64,2}
    eKinv::Array{Float64,2}
    nnidx::Matrix{Tuple{Int64, Int64}}
    nodes::Vector{Int64}
    UV::Array{Float64, 3}
    samplers_dict::Dict{UInt8, Random.Sampler}
end

function tV_Hubbard_Para(;Ht,Hv1,Hv2,Δt,Θrelax,Θquench,Lattice::String,site,BatchSize,Initial::String)
    K=K_Matrix(Lattice,site)
    Ns=size(K,1)

    E,V=LAPACK.syevd!('V', 'L',-Ht.*K[:,:])
    HalfeK=V*Diagonal(exp.(-Δt.*E./2))*V'
    eK=V*Diagonal(exp.(-Δt.*E))*V'
    HalfeKinv=V*Diagonal(exp.(Δt.*E./2))*V'
    eKinv=V*Diagonal(exp.(Δt.*E))*V'

    Pt=zeros(Float64,Ns,div(Ns,2))
    if Initial=="H0"
        KK=K[:,:]
        # 交错化学势，打开gap，去兼并
        μ=1e-3
        if occursin("HoneyComb", Lattice)
            KK+=μ*Diagonal(repeat([-1, 1], div(Ns, 2)))
        elseif Lattice=="SQUARE"
            for i in 1:Ns
                x,y=i_xy(Lattice,site,i)
                KK[i,i]+=μ*(-1)^(x+y)
            end
        end

        # hopping 扰动，避免能级简并
        # KK[KK .!= 0] .+=( rand(size(KK)...) * 1e-3)[KK.!= 0]
        # KK=(KK+KK')./2
        
        E,V=LAPACK.syevd!('V', 'L',KK[:,:])
        Pt.=V[:,div(Ns,2)+1:end]
    elseif Initial=="V" 
        if occursin("HoneyComb", Lattice)
            for i in 1:div(Ns,2)
                Pt[i*2-1,i]=1
            end
        else
            count=1
            for i in 1:Ns
                x,y=i_xy(Lattice,site,i)
                if (x+y)%2==1
                    Pt[i,count]=1
                    count+=1
                end
            end
        end
    end
    Pt=HalfeKinv*Pt

    Nt=2*cld(Θrelax+Θquench,Δt)
    ΔU = (Hv1 - Hv2) / Θquench*Δt
    Hv = vcat(fill(Hv1, round(Int,Θrelax/Δt)), reverse( collect(Hv2:ΔU:Hv1) ) , collect(Hv2:ΔU:Hv1) , fill(Hv1, round(Int,Θrelax/Δt)) )

    α = sqrt.(Δt .* Hv ./ 2)
    γ = [1 + sqrt(6) / 3, 1 + sqrt(6) / 3, 1 - sqrt(6) / 3, 1 - sqrt(6) / 3]
    η = [sqrt(2 * (3 - sqrt(6))), -sqrt(2 * (3 - sqrt(6))), sqrt(2 * (3 + sqrt(6))), -sqrt(2 * (3 + sqrt(6)))]

    nnidx=nnidx_F(Lattice,site)
    if div(Nt, 2) % BatchSize == 0
        nodes = collect(0:BatchSize:Nt)
    else
        nodes = vcat(0, reverse(collect(div(Nt, 2) - BatchSize:-BatchSize:1)), collect(div(Nt, 2):BatchSize:Nt), Nt)
    end

    UV=zeros(Float64,Ns,Ns,size(nnidx,2))
    for j in axes(nnidx,2)
        for i in axes(nnidx,1)
            x,y=nnidx[i,j]
            UV[x,x,j]=UV[x,y,j]=UV[y,x,j]=-2^0.5/2
            UV[y,y,j]=2^0.5/2
        end
    end

    rng=MersenneTwister(Threads.threadid()+time_ns())
    elements = (1, 2, 3, 4)
    samplers_dict = Dict{UInt8, Random.Sampler}()
    for excluded in elements
        allowed = [i for i in elements if i != excluded]
        samplers_dict[excluded] = Random.Sampler(rng, allowed)
    end

    return tV_Hubbard_Para_(Lattice,Ht,Hv1,Hv2,site,Θrelax,Θquench,Ns,Nt,K,BatchSize,Δt,α,γ,η,Pt,HalfeK,eK,HalfeKinv,eKinv,nnidx,nodes,UV,samplers_dict)

end

mutable struct UpdateBuffer_
	uv::Matrix{Float64}      # 2 x 2
	tmp22::Matrix{Float64}   # 2 x 2
	tmp2::Vector{Float64}    # length 2
	r::Matrix{Float64}       # 2 x 2
	Δ::Matrix{Float64}       # 2 x 2
    subidx::Vector{Int64}  # length 2
end

function UpdateBuffer()
    uv=[-2^0.5/2 -2^0.5/2;-2^0.5/2 2^0.5/2]
    return UpdateBuffer_(
        uv,
        Matrix{Float64}(undef, 2, 2),
        Vector{Float64}(undef, 2),
        Matrix{Float64}(undef, 2, 2),
        Matrix{Float64}(undef, 2, 2),
        Vector{Int64}(undef, 2),
    )
end


# ---------------------------------------------------------------------------------------

function PhyBuffer(Ns,NN)
    ns = div(Ns,2)
    return PhyBuffer_(
        Vector{Float64}(undef, ns),
        Vector{LAPACK.BlasInt}(undef, ns),

        Matrix{Float64}(undef, Ns, Ns),
        Matrix{Float64}(undef, Ns, Ns),
        Array{Float64}(undef, ns, Ns, NN),
        Array{Float64}(undef, Ns, ns, NN),

        Vector{Float64}(undef, Ns),
        Matrix{Float64}(undef, Ns, Ns),
        Matrix{Float64}(undef, Ns, ns),
        Matrix{Float64}(undef, ns, ns),
        Matrix{Float64}(undef, ns, Ns),
        Matrix{Float64}(undef, 2, Ns),
    )
end
# ---------------------------------------------------------------------------------------

function SCEEBuffer(Ns)
    ns=div(Ns, 2)
    return SCEEBuffer_(
        Matrix{Float64}(I, Ns, Ns),
        Vector{Float64}(undef, Ns),
        Vector{Float64}(undef, Ns),
        Matrix{Float64}(undef, 2, Ns),
        Matrix{Float64}(undef, ns, ns),
        Matrix{Float64}(undef, Ns, Ns),
        Matrix{Float64}(undef, Ns, Ns),
        Matrix{Float64}(undef, Ns, ns),
        Matrix{Float64}(undef, ns, Ns),
        Vector{LAPACK.BlasInt}(undef, ns),
        Vector{Float64}(undef, ns),
    )
end

function G4Buffer(Ns,NN)
    ns=div(Ns, 2)
    return G4Buffer_(
        Matrix{Float64}(undef, Ns, Ns),
        Matrix{Float64}(undef, Ns, Ns),
        Matrix{Float64}(undef, Ns, Ns),
        Matrix{Float64}(undef, Ns, Ns),

        Array{Float64,3}(undef, ns, Ns, NN),
        Array{Float64,3}(undef, Ns, ns, NN),
        Array{Float64,3}(undef, Ns, Ns, NN),
        Array{Float64,3}(undef, Ns, Ns, NN),
    )
end

function AreaBuffer(index)
    nA = length(index)
    return AreaBuffer_(
        index,
        0.0,
        Matrix{Float64}(undef, nA, nA),
        Matrix{Float64}(undef, nA, nA),
        Matrix{Float64}(undef, nA, 2),
        Matrix{Float64}(undef, 2, nA),
        Matrix{Float64}(undef, nA, 2),
        Matrix{Float64}(undef, 2, nA),
        Matrix{Float64}(undef, 2, 2),
        Vector{LAPACK.BlasInt}(undef, nA),
    )
end



if abspath(PROGRAM_FILE) == @__FILE__
    push!(LOAD_PATH,"C:/Users/admin/Desktop/JuliaDQMC/code/spinlessPQMC/Hopping/")
    using KAPDQMC_tV
    using LinearAlgebra,Random

    model=KAPDQMC_tV.Hubbard_Para(1.0,4.0,"HoneyComb120",[3,3],0.1,2.0,10,"H0")
    println("Hubbard model initialized.")
    s=Initial_s(model,MersenneTwister(1234))
    
    # TEST For diag transformation of nn interaction UV*Diagonal(s)*UV' = V
    lt=1
    tmpVV=zeros(Float64,3,model.Ns,model.Ns)
    for j in 3:-1:1
        tmpN=zeros(Float64,model.Ns)
        tmpV=zeros(Float64,model.Ns,model.Ns)
        println("---------------")
        for i in 1:div(model.Ns,2)
            # println(i," ",j,": ",model.nnidx[i,j])
            x,y=model.nnidx[i,j]
            tmpN[x]=model.η[s[lt,i,j]]
            tmpN[y]=-model.η[s[lt,i,j]]
            
            tmpV[x,y]=model.η[s[lt,i,j]]
            tmpV[y,x]=model.η[s[lt,i,j]]
        end
        tmpVV[j,:,:]=tmpV[:,:]
        @assert norm(model.UV[:,:,j]*Diagonal(tmpN)*model.UV[:,:,j]'-tmpV)<1e-5
        @assert norm(model.UV[:,:,j]'*model.UV[:,:,j]-I(model.Ns))<1e-5
        @assert norm(model.UV[:,:,j]-model.UV[:,:,j]')<1e-5
    end

    # TEST for not comute for V1,V2,V3
    @assert norm(tmpVV[1,:,:]*tmpVV[2,:,:]-tmpVV[2,:,:]*tmpVV[1,:,:])>1e-5
    @assert norm(tmpVV[1,:,:]*tmpVV[3,:,:]-tmpVV[3,:,:]*tmpVV[1,:,:])>1e-5
    @assert norm(tmpVV[2,:,:]*tmpVV[3,:,:]-tmpVV[3,:,:]*tmpVV[2,:,:])>1e-5

    # TEST for comute for V_i,V_i^′
    for j in 3:-1:1
        tmp=tmpVV[j,:,:]
        for i in 1:div(model.Ns,2)
            x,y=model.nnidx[i,j]
            tmp[x,y]=-tmp[x,y]
            tmp[y,x]=-tmp[y,x]
            @assert norm(tmp * tmpVV[j,:,:] - tmpVV[j,:,:]*tmp) <1e-5
        end
    end

    # TEST for diag transformation of update UV*Diagonal(s)*UV' = V
    uv=[-2^0.5/2 -2^0.5/2;-2^0.5/2 2^0.5/2]
    for sx_f in 1:4
        for sx_i in 1:4
            tmp22= (model.η[sx_f]-model.η[sx_i])*[0 1 ; 1.0 0]
            tmp2=(model.η[sx_f]-model.η[sx_i])*[1,-1]
            println(norm(uv*Diagonal(tmp2)*uv' .- tmp22))
        end
    end
end



