# Independent real-space cross-check of the revised production entropy pipeline.
# Run: julia --project=. src/Measure/born_ai/analysis/audit_entropy_pipeline.jl
using LinearAlgebra, Random, Statistics, SHA, TOML, Dates, DelimitedFiles
BLAS.set_num_threads(1)
const HERE=@__DIR__
const SOURCE=joinpath(HERE,"..","Free_M.jl")
const SOURCE_SHA=bytes2hex(sha256(read(SOURCE)))
const SAMPLER_SHA=bytes2hex(sha256(read(joinpath(HERE,"..","BornSampler.jl"))))
include(SOURCE)

function trace_statistics(x)
    n=length(x); mu=mean(x); variance=var(x)
    # beta=0 entropy is deterministic: rounding noise is not MC variation.
    if maximum(x)-minimum(x)<1e-12
        return Dict("mean"=>mu,"naive_se"=>0.,"block_se"=>0.,"iat"=>1.,"ess"=>Float64(n),"lag1"=>0.)
    end
    y=x.-mu; denom=dot(y,y)
    acf=[dot(view(y,1:n-k),view(y,1+k:n))/denom for k in 1:min(300,n÷2)]
    rho=vcat(1.0,acf)
    pairsum=0.; previous=Inf
    # Geyer initial positive, monotone sequence: (rho0+rho1), (rho2+rho3), ... .
    for k in 1:2:length(rho)-1
        pair=rho[k]+rho[k+1]
        pair<=0 && break
        pair=min(pair,previous)
        pairsum+=pair; previous=pair
    end
    tau=max(1.,-1+2pairsum) # conservative: do not report ESS > n
    b=max(10,round(Int,sqrt(n))); nb=n÷b
    means=[mean(view(x,1+(i-1)*b:i*b)) for i in 1:nb]
    Dict("mean"=>mu,"naive_se"=>sqrt(variance/n),"block_se"=>std(means)/sqrt(nb),
        "iat"=>tau,"ess"=>n/tau,"lag1"=>acf[1],"block_size"=>b,"blocks"=>nb)
end

function reference_trajectory(L,beta,sweeps,sampler,seed,burnin)
    rng=MersenneTwister(seed)
    K=nnK_Matrix("HoneyComb120",[L,L]); E,V=LAPACK.syevd!('V','L',K)
    N=size(K,1); n=N÷2; P=V[:,1:n]; G=P*inv(P'*P)*P'
    M=Measure_EE(beta,N,n;rng)
    iid=sampler==:iid ? IIDSampler(P,beta) : nothing
    if sampler==:local; M.X=inv(cholesky(Symmetric(I(n)-M.λ*G[M.E_idx,M.E_idx]))); end
    idxA=area_index("HoneyComb120",[L,L],([1,1],[L÷3,L]))
    idxB=area_index("HoneyComb120",[L,L],([1,1],[L÷3,2L÷3]))
    ent=zeros(sweeps,2); v=zeros(n); vv=zeros(n,n)
    for step in 1:burnin+sweeps
        if sampler==:local
            if (step-1)%100==0; M.X=inv(cholesky(Symmetric(I(n)-M.λ*G[M.E_idx,M.E_idx]))); end
            UpdateMeasure!(M,G,v,vv,P;rng)
        else
            M.E_idx.=sample_measurement(iid;rng)
        end
        step<=burnin && continue
        weights=ones(N); weights[M.E_idx].=exp(-2beta)
        Q=Matrix(qr(weights.*P).Q)[:,1:n]
        for (col,idx) in enumerate((idxA,idxB))
            nu=eigvals(Symmetric(Q[idx,:]*Q[idx,:]'))
            ent[step-burnin,col]=-sum(log.(nu.^2+(1 .-nu).^2))
        end
    end
    ent
end

function main()
    started=time(); results=Dict{String,Any}[]
    root=joinpath(HERE,"entropy_smoke"); mkpath(root)
    for beta in (0.,1.,3.), sampler in (:local,:iid)
        L=6; n=1000; seed=3701+round(Int,10beta)+(sampler==:iid ? 100000 : 0)
        burnin=sampler==:local ? 100 : 0
        t=time()
        r=Free_Measure_EE(L,beta,n;sampler,burnin,rng=MersenneTwister(seed),output_dir=root)
        ref=reference_trajectory(L,beta,n,sampler,seed,burnin)
        err=maximum(abs.(r.entropy-ref))
        @assert err<1e-10
        @assert maximum(abs.(r.entropy+log.(r.purity)))<1e-12
        diag=r.diagnostics
        row=Dict("L"=>L,"beta"=>beta,"sampler"=>String(sampler),"samples"=>n,"burnin"=>burnin,
            "seed"=>seed,"max_S2_reference_error"=>err,"elapsed_seconds"=>time()-t,
            "S2_A"=>trace_statistics(r.entropy[:,1]),"S2_B"=>trace_statistics(r.entropy[:,2]),
            "paired_A_minus_B"=>trace_statistics(r.entropy[:,1]-r.entropy[:,2]),
            "proposed"=>sum(diag[:,3]),"accepted"=>sum(diag[:,2]))
        if sampler==:local; row["acceptance"]=sum(diag[:,2])/sum(diag[:,3]); end
        push!(results,row)
        println("sampler=$sampler beta=$beta S2_error=$err elapsed=$(row["elapsed_seconds"])s")
        flush(stdout)
    end
    @assert bytes2hex(sha256(read(SOURCE)))==SOURCE_SHA "source changed during audit"
    @assert bytes2hex(sha256(read(joinpath(HERE,"..","BornSampler.jl"))))==SAMPLER_SHA
    out=Dict("status"=>"PASS","source_sha256"=>SOURCE_SHA,"sampler_sha256"=>SAMPLER_SHA,
        "script_sha256"=>bytes2hex(sha256(read(@__FILE__))),"elapsed_seconds"=>time()-started,
        "blas_threads"=>BLAS.get_num_threads(),"created_at"=>string(now()),"results"=>results,
        "random_streams"=>"local seed=3701+10beta; iid seed=103701+10beta; each reference replays its own sampler seed",
        "iat_convention"=>"Geyer initial positive monotone pairs (rho0+rho1),(rho2+rho3),...; g=-1+2sum(pairs), conservatively clamped g>=1")
    open(joinpath(HERE,"entropy_pipeline_audit.toml"),"w") do io; TOML.print(io,out); end
end
if abspath(PROGRAM_FILE)==@__FILE__; main(); end
