module OriginalSwapScaling
using LinearAlgebra, Random, SHA, TOML
push!(LOAD_PATH,normpath(joinpath(@__DIR__,"..","..")))
using KAPDQMC

export GroundModel, SwapState, EntropyWorkspace, ground_model, save_model,
       swap_state, sweep!, rebuild!, entropy!, region_indices

struct GroundModel
    L::Int
    N::Int
    n::Int
    Q::Matrix{Float64}
    G::Matrix{Float64}
    energies::Vector{Float64}
    zero_mode_rule::Symbol
    mu::Float64
    null_singular_count::Int
    zero_singular_tolerance::Float64
    orthogonality_error::Float64
    ground_energy_error::Float64
    eigenvector_residual::Float64
end

"""Historical syevd, or a stable negative-energy basis of K+mu*diag(+A,-B)."""
function ground_model(L;state=:lapack,mu=0.,zero_mode_rule=state,zero_mode_selector=nothing,
                      zero_singular_tolerance=1e-12)
    L>=3 && L%3==0 || throw(ArgumentError("L must be a positive multiple of 3"))
    K=nnK_Matrix("HoneyComb120",[L,L]); original=copy(K)
    N=size(K,1); n=N÷2
    nullcount=0
    if zero_mode_rule in (:staggered,:staggered_svd)
        isfinite(mu) && mu>0 || throw(ArgumentError("staggered state requires finite mu>0"))
        zero_mode_selector===nothing || throw(ArgumentError("custom selector is incompatible with staggered SVD"))
        zero_singular_tolerance>0 || throw(ArgumentError("invalid singular-value threshold"))
        factor=svd(K[1:2:N,2:2:N])
        sigma=copy(factor.S)
        null=findall(x->x<zero_singular_tolerance,sigma); nullcount=length(null)
        nullcount==2 || error("expected exactly two Dirac singular values on the L multiple-of-3 grid")
        sigma[null].=0.
        epsilon=hypot.(sigma,mu)
        coefficient_A=sigma./sqrt.(2 .*epsilon.*(epsilon.+mu))
        coefficient_B=-sqrt.((epsilon.+mu)./(2 .*epsilon))
        Q=zeros(N,n)
        Q[1:2:N,:].=factor.U.*coefficient_A'
        Q[2:2:N,:].=factor.V.*coefficient_B'
        @inbounds for i in 1:N; original[i,i]+=isodd(i) ? mu : -mu; end
        occupied_energies=-epsilon
        energies=sort(vcat(-epsilon,epsilon))
        zero_mode_rule=:staggered
    elseif zero_mode_rule==:lapack
        mu==0 || throw(ArgumentError("nonzero mu requires state=:staggered"))
        zero_mode_selector===nothing || throw(ArgumentError("do not provide a selector with :lapack"))
        energies,V=LAPACK.syevd!('V','L',K)
        Q=copy(V[:,1:n])
        occupied_energies=energies[1:n]
    else
        mu==0 || throw(ArgumentError("custom zero-mode selector is only for mu=0"))
        zero_mode_selector===nothing && throw(ArgumentError("explicit zero_mode_selector is required for this rule"))
        energies,V=LAPACK.syevd!('V','L',K)
        Q=Matrix{Float64}(zero_mode_selector(energies,V,n))
        occupied_energies=energies[1:n]
    end
    size(Q)==(N,n) || error("invalid occupied basis dimensions")
    orthogonality=norm(Q'*Q-I)
    energyerror=abs(sum(Q.*(original*Q))-sum(energies[1:n]))
    residual=norm(original*Q-Q.*occupied_energies')
    orthogonality<1e-10 && energyerror<1e-8 || error("selected orbitals are not an orthonormal ground state")
    residual<1e-9 || error("occupied-basis eigenvector residual too large")
    # Match Free_M.jl's projector construction, not just its exact-arithmetic simplification.
    G=Q*inv(Q'*Q)*Q'
    GroundModel(L,N,n,Q,G,energies,zero_mode_rule,Float64(mu),nullcount,Float64(zero_singular_tolerance),orthogonality,energyerror,residual)
end

function save_model(model,directory;prefix="original_swap")
    mkpath(directory)
    stem="$(prefix)_L$(model.L)_$(model.zero_mode_rule)"*(model.mu==0 ? "" : "_mu$(model.mu)")
    paths=String[]
    for (name,matrix) in (("Q",model.Q),("G",model.G),("energies",model.energies))
        path=joinpath(directory,"$(stem)_$(name).bin")
        open(io->write(io,matrix),path,"w"); push!(paths,path)
    end
    metadata=Dict("L"=>model.L,"Nsites"=>model.N,"Nparticles"=>model.n,
        "zero_mode_rule"=>String(model.zero_mode_rule),"zero_mode_count"=>count(x->abs(x)<1e-10,model.energies),
        "mu"=>model.mu,"null_singular_count"=>model.null_singular_count,"zero_singular_tolerance"=>model.zero_singular_tolerance,
        "orthogonality_error"=>model.orthogonality_error,"ground_energy_error"=>model.ground_energy_error,
        "eigenvector_residual"=>model.eigenvector_residual,
        "format"=>"Float64 native-endian column-major; Q=Nsites x Nparticles, G=Nsites x Nsites, energies=Nsites",
        "files"=>Dict(basename(p)=>bytes2hex(sha256(read(p))) for p in paths))
    path=joinpath(directory,"$(stem)_model.toml")
    open(io->TOML.print(io,metadata),path,"w"); push!(paths,path)
    paths
end

mutable struct SwapState
    beta::Float64
    lambda::Float64
    E::Vector{Int}
    S::Vector{Int}
    X::Matrix{Float64}
    g::Vector{Float64}
    grow::Vector{Float64}
    xg::Vector{Float64}
    z::Vector{Float64}
    column::Vector{Float64}
    row::Vector{Float64}
    log_weight::Float64
end

function rebuild!(state,G)
    n=length(state.E)
    A=Matrix{Float64}(I,n,n)-state.lambda*G[state.E,state.E]
    factor=cholesky(Symmetric(A))
    truth=Matrix(inv(factor)); logweight=2sum(log,diag(factor.U))
    inverse_drift=norm(state.X-truth)/norm(truth)
    logweight_drift=abs(state.log_weight-logweight)
    state.X.=truth; state.log_weight=logweight
    (;inverse_drift,logweight_drift)
end

function swap_state(G,E,beta)
    N=size(G,1); n=length(E)
    N==2n && length(unique(E))==n && all(i->1<=i<=N,E) || throw(ArgumentError("invalid half-filled suppressed set"))
    isfinite(beta) && beta>=0 || throw(ArgumentError("beta must be finite and nonnegative"))
    lambda=-expm1(-4Float64(beta)); lambda<1 || error("sampling inverse is numerically singular at this beta")
    state=SwapState(beta,lambda,copy(E),setdiff(1:N,E),Matrix{Float64}(I,n,n),
        zeros(n),zeros(n),zeros(n),zeros(n),zeros(n),zeros(n),0.)
    rebuild!(state,G)
    state
end

"""Exactly 2n uniform E/S index pairs and Metropolis decisions, including rejected states."""
function sweep!(state,G,rng=Random.default_rng())
    n=length(state.E); lambda=state.lambda; accepted=0
    for attempt in 1:2n
        i=rand(rng,1:n); j=rand(rng,1:n); s=state.S[i]
        @inbounds for k in 1:n
            state.g[k]=G[state.E[k],s]
            state.grow[k]=G[s,state.E[k]]
        end
        mul!(state.xg,state.X,state.g)
        xjj=state.X[j,j]; f=state.xg[j]/xjj
        @inbounds for k in 1:n; state.z[k]=state.xg[k]-f*state.X[k,j]; end
        state.z[j]=0.
        delta=1-lambda*G[s,s]-lambda^2*dot(state.grow,state.z)
        ratio=delta*xjj
        isfinite(ratio) && ratio>0 && delta>0 && xjj>0 || error("invalid original-swap ratio; no proposal was silently altered")
        # Consume the acceptance draw even at beta=0, exactly as Free_M.jl.
        if rand(rng)<min(1.,ratio)
            @inbounds for k in 1:n
                state.column[k]=state.X[k,j]; state.row[k]=state.X[j,k]
            end
            BLAS.ger!(-1/xjj,state.column,state.row,state.X)
            BLAS.ger!(lambda^2/delta,state.z,state.z,state.X)
            f=lambda/delta
            @inbounds for k in 1:n
                state.X[k,j]=f*state.z[k]; state.X[j,k]=f*state.z[k]
            end
            state.X[j,j]=1/delta
            state.E[j],state.S[i]=state.S[i],state.E[j]
            state.log_weight+=log(ratio)
            accepted+=1
        end
    end
    (;accepted,proposed=2n)
end

function region_indices(L)
    (A=area_index("HoneyComb120",[L,L],([1,1],[L÷3,L])),
     B=area_index("HoneyComb120",[L,L],([1,1],[L÷3,2L÷3])),
     C=area_index("HoneyComb120",[L,L],([1,1],[L,L÷3])))
end

mutable struct EntropyWorkspace
    Q::Matrix{Float64}
    tau::Vector{Float64}
    regions::NTuple{3,Vector{Int}}
    restricted::Vector{Matrix{Float64}}
    correlation::Vector{Matrix{Float64}}
    purity::Vector{Matrix{Float64}}
end

function EntropyWorkspace(model)
    regions=Tuple(values(region_indices(model.L)))
    EntropyWorkspace(similar(model.Q),zeros(model.n),regions,
        [zeros(length(r),model.n) for r in regions],
        [zeros(length(r),length(r)) for r in regions],
        [zeros(length(r),length(r)) for r in regions])
end

"""One weighted orbital QR plus three regional purity Cholesky factorizations."""
function entropy!(work,model,E,beta)
    work.Q.=model.Q
    attenuation=exp(-2beta)
    @inbounds for j in 1:model.n,i in E; work.Q[i,j]*=attenuation; end
    LAPACK.geqrf!(work.Q,work.tau)
    LAPACK.orgqr!(work.Q,work.tau,model.n)
    ent=zeros(3)
    for index in 1:3
        idx=work.regions[index]; V=work.restricted[index]; C=work.correlation[index]; B=work.purity[index]
        @inbounds for j in 1:model.n,i in eachindex(idx); V[i,j]=work.Q[idx[i],j]; end
        mul!(C,V,V')
        @inbounds for i in axes(C,1); C[i,i]-=.5; end
        BLAS.syrk!('U','N',2.,C,0.,B)
        @inbounds for i in axes(B,1); B[i,i]+=.5; end
        factor=cholesky!(Symmetric(B,:U))
        ent[index]=-2sum(log,diag(factor.U))
    end
    (;SA=ent[1],SB=ent[2],SC=ent[3],A_minus_B=ent[1]-ent[2],D=(2ent[1]+ent[3])/3-ent[2])
end
end # module
