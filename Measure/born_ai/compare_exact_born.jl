# Fixed-design comparison of exact Born trajectory averages and Monte Carlo.
# Run from the repository root: julia --project=. src/Measure/born_ai/compare_exact_born.jl
# This script only writes data/exact_born and never runs Free_M.jl's main scan.
using LinearAlgebra, Random, Statistics, SHA, TOML, Dates
BLAS.set_num_threads(1)
include("Free_M.jl")
include("ExactBorn.jl")

const OUTDIR = joinpath(@__DIR__, "data", "exact_born")
const BETAS = (0., .5, 1., 2., 3.)
const LOCAL_CHAINS = 3
const LOCAL_SAMPLES = 10_000
const LOCAL_BURNIN = 500
const IID_SAMPLES = 30_000
const BLOCK_SIZE = 100
const REFRESH_EVERY = 5
const OBSERVABLES = ("S_A", "S_B", "S_C", "A_minus_B", "D")

sha(path) = bytes2hex(sha256(read(path)))
csv(io, row) = println(io, join(string.(row), ','))
tag(beta) = replace(string(beta), "."=>"p")
configuration_key(E) = join(sort(E), ';')

function setup_case(L)
    K = nnK_Matrix("HoneyComb120", [L,L])
    eigenvalues, V = LAPACK.syevd!('V','L',K)
    n = L^2
    P = V[:,1:n]
    G = P * inv(P'*P) * P'
    # L=2 uses intentionally small test partitions, not a scaling geometry.
    regions = (A=area_index("HoneyComb120",[L,L],([1,1],[1,L])),
               B=area_index("HoneyComb120",[L,L],([1,1],[1,L-1])),
               C=area_index("HoneyComb120",[L,L],([1,1],[L,1])))
    (; P, G, regions, eigenvalues)
end

function observables(three)
    a,b,c = three
    [a,b,c,a-b,(2a+c)/3-b]
end

function entropy_reference(P, E, beta, regions)
    weights = ones(size(P,1)); weights[E] .= exp(-2beta)
    Q = Matrix(qr(weights.*P).Q)[:,1:size(P,2)]
    s = Float64[]
    for idx in values(regions)
        eigenvalues = eigvals(Symmetric(Q[idx,:]*Q[idx,:]'))
        push!(s, -sum(log.(eigenvalues.^2+(1 .-eigenvalues).^2)))
    end
    observables(s)
end

function geyer_g(x)
    maximum(x)-minimum(x)<1e-12 && return 1.
    y=x.-mean(x); norm2=dot(y,y); n=length(x)
    maxlag=min(2000,n÷2)
    rho = vcat(1.,[dot(view(y,1:n-k),view(y,1+k:n))/norm2 for k in 1:maxlag])
    total=0.; previous=Inf
    for k in 1:2:length(rho)-1
        pair=rho[k]+rho[k+1]
        pair<=0 && break
        pair=min(pair,previous); total+=pair; previous=pair
    end
    max(1.,-1+2total)
end

function statistics(chains)
    n=sum(length,chains)
    mu=sum(sum,chains)/n
    lo=minimum(minimum,chains); hi=maximum(maximum,chains)
    if hi-lo<1e-12
        return (;mean=mu,SE=0.,ESS=Float64(n),g=1.,blocks=n÷BLOCK_SIZE)
    end
    blockmeans=Float64[]; denominator=0.
    for x in chains
        length(x)%BLOCK_SIZE==0 || error("block length must divide each fixed sample count")
        append!(blockmeans,[mean(view(x,i:i+BLOCK_SIZE-1)) for i in 1:BLOCK_SIZE:length(x)])
        denominator+=length(x)*geyer_g(x)
    end
    se=std(blockmeans)/sqrt(length(blockmeans))
    ess=n^2/denominator
    (;mean=mu,SE=se,ESS=ess,g=n/ess,blocks=length(blockmeans))
end

function write_exact(L,beta,exact)
    @assert collect(Symbol.(exact.region_names))==[:A,:B,:C]
    @assert size(exact.entropies,2)==3
    @assert abs(sum(exact.probabilities)-1)<1e-11
    path=joinpath(OUTDIR,"exact_born_L$(L)_beta$(tag(beta))_trajectories.csv")
    lookup=Dict{String,Vector{Float64}}()
    mean=zeros(5)
    open(path,"w") do io
        csv(io,("L","beta","ensemble","configuration","E_indices","probability","log_weight",OBSERVABLES...))
        for i in eachindex(exact.probabilities)
            key=configuration_key(exact.configurations[i])
            s=observables(view(exact.entropies,i,:)); lookup[key]=s
            mean .+= exact.probabilities[i].*s
            csv(io,(L,beta,"fixed_half",i,key,exact.probabilities[i],exact.log_weights[i],s...))
        end
    end
    @assert maximum(abs.(mean[1:3]-exact.mean_entropy))<1e-10
    path,lookup,mean
end

function run_chain(L,beta,method,chain,case,lookup,io)
    # Disjoint seed ranges across size, beta, chain, and sampler.
    seed=20_000_000+100_000L+10_000round(Int,2beta)+(method==:iid ? 1_000_000 : 100chain)
    rng=MersenneTwister(seed); P=case.P; G=case.G; N,n=size(P)
    retained=method==:iid ? IID_SAMPLES : LOCAL_SAMPLES
    burnin=method==:iid ? 0 : LOCAL_BURNIN
    M=Measure_EE(beta,N,n;rng)
    sampler=method==:iid ? IIDSampler(P,beta) : nothing
    function fresh_inverse()
        inv(cholesky(Symmetric(I(n)-M.λ*G[M.E_idx,M.E_idx])))
    end
    if method==:local; M.X=fresh_inverse(); end
    temp=zeros(n); temp2=zeros(n,n)
    data=zeros(retained,5)
    accepted=0; proposed=0; max_inverse_error=0.; max_entropy_error=0.
    started=time()
    for step in 1:burnin+retained
        if method==:local
            if (step-1)%REFRESH_EVERY==0
                X=fresh_inverse()
                max_inverse_error=max(max_inverse_error,norm(M.X-X)/norm(X))
                M.X=X
            end
            diagnostic=UpdateMeasure!(M,G,temp,temp2,P;rng)
        else
            M.E_idx.=sample_measurement(sampler;rng)
            diagnostic=(;accepted=0,proposed=0)
        end
        step<=burnin && continue
        sample=step-burnin
        s=entropy_reference(P,M.E_idx,beta,case.regions)
        key=configuration_key(M.E_idx)
        max_entropy_error=max(max_entropy_error,maximum(abs.(s-lookup[key])))
        data[sample,:].=s
        accepted+=diagnostic.accepted; proposed+=diagnostic.proposed
        csv(io,(L,beta,"fixed_half",method,chain,sample,key,s...,diagnostic.accepted,diagnostic.proposed))
    end
    @assert max_entropy_error<1e-9 "independent MC entropy differs from exact trajectory table"
    @assert max_inverse_error<1e-6 "local inverse drift exceeded fixed audit threshold"
    metadata=Dict("L"=>L,"beta"=>beta,"method"=>String(method),"chain"=>chain,"seed"=>seed,
        "retained"=>retained,"burnin"=>burnin,"accepted"=>accepted,"proposed"=>proposed,
        "maximum_inverse_relative_error"=>max_inverse_error,"maximum_entropy_reference_error"=>max_entropy_error,
        "elapsed_seconds"=>time()-started)
    data,metadata
end

function main()
    mkpath(OUTDIR)
    sources=[@__FILE__,joinpath(@__DIR__,"Free_M.jl"),joinpath(@__DIR__,"BornSampler.jl"),joinpath(@__DIR__,"ExactBorn.jl"),
             joinpath(@__DIR__,"..","..","public","Geometry.jl")]
    source_hashes=Dict(abspath(p)=>sha(p) for p in sources)
    started=time(); runs=Dict{String,Any}[]; exact_cases=Dict{String,Any}[]
    summary_path=joinpath(OUTDIR,"exact_born_summary.csv")
    chain_path=joinpath(OUTDIR,"exact_born_chain_statistics.csv")
    open(summary_path,"w") do summary
        csv(summary,("L","beta","ensemble","method","observable","mean","SE","ESS","n","chains","block_size","burnin_per_chain","refresh_every","exact_mean","difference","z_score"))
        open(chain_path,"w") do chainout
            csv(chainout,("L","beta","ensemble","method","chain","observable","mean","SE","ESS","g","n","seed"))
            for L in (2,3)
                case=setup_case(L)
                for beta in BETAS
                    t=time()
                    exact=ExactBorn.born_trajectory_entropy(case.P,beta,case.regions;
                        ensemble=:fixed_half,max_configs=1_000_000,keep_trajectories=true)
                    path,lookup,truth=write_exact(L,beta,exact)
                    count=length(exact.probabilities)
                    push!(exact_cases,Dict("L"=>L,"beta"=>beta,"configurations"=>count,"normalization_error"=>exact.normalization_error,
                        "elapsed_seconds"=>time()-t,"region_A"=>case.regions.A,"region_B"=>case.regions.B,"region_C"=>case.regions.C,
                        "ground_orbital_sha256"=>bytes2hex(sha256(reinterpret(UInt8,vec(case.P))))))
                    for (j,name) in enumerate(OBSERVABLES)
                        csv(summary,(L,beta,"fixed_half","exact",name,truth[j],0.,count,count,0,0,0,0,truth[j],0.,""))
                    end
                    mcpath=joinpath(OUTDIR,"exact_born_L$(L)_beta$(tag(beta))_mc.csv")
                    open(mcpath,"w") do io
                        csv(io,("L","beta","ensemble","method","chain","sample","E_indices",OBSERVABLES...,"accepted","proposed"))
                        for method in (:local,:iid)
                            arrays=Matrix{Float64}[]
                            for chain in 1:(method==:local ? LOCAL_CHAINS : 1)
                                data,meta=run_chain(L,beta,method,chain,case,lookup,io)
                                push!(arrays,data); push!(runs,meta)
                                for (j,name) in enumerate(OBSERVABLES)
                                    stat=statistics([data[:,j]])
                                    csv(chainout,(L,beta,"fixed_half",method,chain,name,stat.mean,stat.SE,stat.ESS,stat.g,size(data,1),meta["seed"]))
                                end
                            end
                            for (j,name) in enumerate(OBSERVABLES)
                                stat=statistics([a[:,j] for a in arrays]); difference=stat.mean-truth[j]
                                z=stat.SE>0 ? difference/stat.SE : ""
                                stat.SE==0 && abs(difference)>1e-10 && error("constant trajectory value differs from exact")
                                csv(summary,(L,beta,"fixed_half",method,name,stat.mean,stat.SE,stat.ESS,sum(a->size(a,1),arrays),
                                    length(arrays),BLOCK_SIZE,method==:local ? LOCAL_BURNIN : 0,method==:local ? REFRESH_EVERY : 0,
                                    truth[j],difference,z))
                            end
                        end
                    end
                    flush(summary); flush(chainout)
                    println("L=$L beta=$beta configurations=$count exact_D=$(truth[5]) elapsed=$(round(time()-t,digits=2))s")
                    flush(stdout)
                end
            end
        end
    end
    @assert all(sha(p)==hash for (p,hash) in source_hashes) "source changed during experiment"
    # Explicitly own only this runner's primary outputs. Derived diagnostics may
    # already exist on reruns and must never become inputs of this manifest.
    primary_paths=[summary_path,chain_path]
    for L in (2,3), beta in BETAS, kind in ("trajectories","mc")
        push!(primary_paths,joinpath(OUTDIR,"exact_born_L$(L)_beta$(tag(beta))_$(kind).csv"))
    end
    @assert length(primary_paths)==22 && length(unique(primary_paths))==22
    files=Dict(replace(relpath(p,@__DIR__),'\\'=>'/')=>sha(p) for p in primary_paths)
    manifest=Dict("status"=>"complete","audit_status"=>"PASS","created_at"=>string(now()),"julia_version"=>string(VERSION),"blas_threads"=>BLAS.get_num_threads(),
        "elapsed_seconds"=>time()-started,"ensemble"=>"fixed_half","L_values"=>[2,3],"beta_values"=>collect(BETAS),
        "local_chains"=>LOCAL_CHAINS,"local_samples_per_chain"=>LOCAL_SAMPLES,"local_burnin_per_chain"=>LOCAL_BURNIN,
        "iid_samples"=>IID_SAMPLES,"block_size"=>BLOCK_SIZE,"refresh_every"=>REFRESH_EVERY,
        "SE_definition"=>"SE of fixed-size block means, pooling blocks from independent chains; derived observables are formed per trajectory",
        "ESS_definition"=>"Geyer positive monotone (rho0+rho1),(rho2+rho3),...; g=max(1,-1+2sum(pairs)); pooled ESS=Ntotal^2/sum(Nchain*gchain)",
        "geometry"=>"L3 A=1x3 B=1x2 C=3x1; L2 A=1x2 B=1x1 C=2x1 are test partitions, not a scaling extrapolation",
        "mean_definition"=>"sum_E p(E) S2(E), not -log(sum_E p(E)*purity(E))",
        "sample_count_policy"=>"Fixed before running and never adjusted based on agreement with exact values",
        "source_sha256"=>source_hashes,"files"=>files,"runs"=>runs,"exact_cases"=>exact_cases)
    open(joinpath(OUTDIR,"exact_born_manifest.toml"),"w") do io; TOML.print(io,manifest); end
    println("PASS elapsed=$(round(time()-started,digits=2))s")
end

if abspath(PROGRAM_FILE)==@__FILE__; main(); end
