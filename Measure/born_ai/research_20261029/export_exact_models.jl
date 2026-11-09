# Model export for the independent exhaustive Born benchmark. Existing data are read only.
using LinearAlgebra, SHA, TOML
BLAS.set_num_threads(1)
include("../original_swap_scaling.jl")
using .OriginalSwapScaling
using KAPDQMC
out=joinpath(@__DIR__,"exact_strata","models"); mkpath(out)
mu=1e-8
for L in (2,3)
    if L==3
        model=ground_model(L;state=:staggered,mu=mu)
        Q=model.Q
        description="OriginalSwapScaling.ground_model, same production state"
    else
        K=nnK_Matrix("HoneyComb120",[L,L]); N=2L^2
        F=svd(K[1:2:N,2:2:N]); sigma=copy(F.S)
        epsilon=hypot.(sigma,mu)
        Q=zeros(N,L^2)
        Q[1:2:N,:].=F.U.*(sigma./sqrt.(2 .*epsilon.*(epsilon.+mu)))'
        Q[2:2:N,:].=F.V.*(-sqrt.((epsilon.+mu)./(2 .*epsilon)))'
        for i in 1:N; K[i,i]+=isodd(i) ? mu : -mu; end
        @assert norm(Q'*Q-I)<1e-12
        @assert norm(K*Q+Q.*epsilon')<1e-12
        description="Same honeycomb Hamiltonian and staggered SVD formula, generalized to L2 (no Dirac zero modes)"
    end
    path=joinpath(out,"Q_L$(L)_mu1e-8.bin")
    open(io->write(io,Q),path,"w")
    data=Dict("L"=>L,"N"=>2L^2,"n"=>L^2,"mu"=>mu,"description"=>description,
        "Q_file"=>basename(path),"Q_sha256"=>bytes2hex(sha256(read(path))),
        "orthogonality_error"=>norm(Q'*Q-I),"julia_version"=>string(VERSION),
        "format"=>"Float64 little-endian column-major",
        "export_script_sha256"=>bytes2hex(sha256(read(@__FILE__))),
        "model_source_sha256"=>bytes2hex(sha256(read(joinpath(@__DIR__,"../original_swap_scaling.jl")))))
    open(io->TOML.print(io,data),joinpath(out,"model_L$(L).toml"),"w")
    println("L=",L," Q_sha256=",data["Q_sha256"])
end
