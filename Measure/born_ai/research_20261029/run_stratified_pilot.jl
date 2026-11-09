# Frozen small pilot for independent latent-j strata; never modifies old runs.
using LinearAlgebra, Random, SHA, TOML, Dates
BLAS.set_num_threads(1)
include("StratifiedBorn.jl")
using .StratifiedBorn
include("../original_swap_scaling.jl")
using .OriginalSwapScaling
include("../ExactBorn.jl")
using .ExactBorn
const HERE=@__DIR__
const PILOT=joinpath(HERE,"pilot")
const PLAN=joinpath(PILOT,"plan.toml")
const HEADER="L,beta,z,case,method,jlo,jhi,stratum_probability,sample,seed,j,F,E,record_sha256,SA,SB,SC,D,VA,VB,VC,DV,VA_full,VB_full,VC_full,DV_full,DV_subgroup,sampling_seconds,entropy_seconds,fluctuation_seconds"
sha(p)=bytes2hex(sha256(read(p)))
rel(p)=replace(relpath(p,HERE),'\\'=>'/')
save_toml(p,d)=open(io->TOML.print(io,d),p,"w")

function source_hashes()
    files=[@__FILE__,joinpath(HERE,"StratifiedBorn.jl"),joinpath(HERE,"../BornSampler.jl"),
        joinpath(HERE,"../original_swap_scaling.jl"),joinpath(HERE,"../ExactBorn.jl"),
        normpath(joinpath(HERE,"../../../public/Geometry.jl"))]
    Dict(rel(p)=>sha(p) for p in files)
end

function prepare()
    ispath(PILOT) && error("pilot directory exists; preserve its evidence")
    mkpath(PILOT)
    jobs=Dict{String,Any}[];models=Dict{String,Any}[]
    for L in (6,12)
        model=ground_model(L;state=:staggered,mu=1e-8)
        paths=save_model(model,joinpath(PILOT,"models");prefix="stratified")
        metadata=only(filter(p->endswith(p,".toml"),paths))
        push!(models,Dict("L"=>L,"path"=>rel(metadata),"files"=>Dict(rel(p)=>sha(p) for p in paths)))
        cases=[("beta3",3.),("z0p5",.5log(L^2/.5)),("z1p0",.5log(L^2)),("z2p0",.5log(L^2/2.))]
        for (caseid,(label,beta)) in enumerate(cases)
            sampler=StratifiedSampler(model.Q,beta)
            strata=[(j,j) for j in 0:3];push!(strata,(4,model.n))
            for (si,(jlo,jhi)) in enumerate(vcat(strata,[(0,model.n)]))
                method=si<=5 ? "stratified" : "iid_latent"
                count=method=="stratified" ? 64 : 320
                id="L$(L)_$(label)_$(method)_$(si)"
                seed=291_029_000+10000L+100caseid+si
                push!(jobs,Dict("id"=>id,"L"=>L,"beta"=>beta,"z"=>L^2*exp(-2beta),
                    "case"=>label,"method"=>method,"jlo"=>jlo,"jhi"=>jhi,
                    "stratum_probability"=>stratum_probability(sampler,jlo,jhi),
                    "records"=>count,"seed"=>seed,"model"=>rel(metadata)))
            end
        end
    end
    plan=Dict("stage"=>"pilot","created_at"=>string(now()),"mu"=>1e-8,"jobs"=>jobs,"models"=>models,
        "source_sha256"=>source_hashes(),"blas_threads"=>1,
        "independence"=>"Fresh independent auxiliary F for each record, including between strata; no physical sequential measurement",
        "strata"=>"j=0,1,2,3 and exact conditional mixture j>=4; every j covered, no truncation",
        "target"=>"Fixed-half Born ensemble, original Slater state filtered at the same beta",
        "geometry"=>"Origin A=L/3 x L, B=L/3 x 2L/3, C=L x L/3; D=(2SA+SC)/3-SB. Sampling-method pilot, not an asymptotic corner fit.",
        "purpose"=>"Pilot cost/variance and tail diagnostics only; no pilot records enter future production means",
        "fluctuation_observables"=>"V_R=Tr(C_R-C_R^2); DV=(2VA+VC)/3-VB. DV_full is full L^2-translation average, DV_subgroup is nine-shift average.",
        "fluctuation_kernel"=>"DV_full=sum_ij |Gij|^2 h(dx)h(dy), h(d)=min(d,L-d,L/3)/L; both sublattices retained. DV_subgroup=sum_ij |Gij|^2 * (tile_x_i!=tile_x_j)*(tile_y_i!=tile_y_j)/9.",
        "correlation_aggregates"=>"Every job saves mean G and mean |G|^2 matrices. These are diagnostics, not a substitute for each-state nonlinear entropy.",
        "allocation_rule"=>"No production stopping or seed changes based on desired mean or scaling trend; freeze new production plan after pilot diagnostics")
    save_toml(PLAN,plan)
    println("PREPARED $(length(jobs)) jobs, $(sum(j["records"] for j in jobs)) independent records")
end

function kernels(L)
    N=2L^2;w=L÷3
    xs=[((i-1)÷2)%L for i in 1:N];ys=[((i-1)÷2)÷L for i in 1:N]
    Kx=zeros(N,N);Ky=zeros(N,N);Ksub=zeros(N,N)
    for j in 1:N,i in 1:N
        dx=mod(xs[i]-xs[j],L);dy=mod(ys[i]-ys[j],L)
        Kx[i,j]=min(dx,L-dx,w)/L;Ky[i,j]=min(dy,L-dy,w)/L
        Ksub[i,j]=((xs[i]÷w!=xs[j]÷w)&&(ys[i]÷w!=ys[j]÷w))/9
    end
    (;Kx,Ky,Kfull=Kx.*Ky,Ksub)
end

function fluctuation!(G,Gsquare,work,kernel)
    mul!(G,work.Q,work.Q')
    Gsquare.=abs2.(G)
    V=[length(work.regions[r])/4-sum(abs2,work.correlation[r]) for r in 1:3]
    minimum(V)>=-1e-9 || error("negative regional number variance")
    DV=(2V[1]+V[3])/3-V[2]
    VAfull=sum(Gsquare.*kernel.Kx);VCfull=sum(Gsquare.*kernel.Ky)
    DVfull=sum(Gsquare.*kernel.Kfull);DVsub=sum(Gsquare.*kernel.Ksub)
    VBfull=(2VAfull+VCfull)/3-DVfull
    [V...,DV,VAfull,VBfull,VCfull,DVfull,DVsub]
end

function load_model(meta)
    d=TOML.parsefile(meta);N=d["Nsites"];n=d["Nparticles"]
    arrays=Dict{String,Any}()
    for (name,shape) in (("Q",(N,n)),("G",(N,N)),("energies",(N,)))
        file=only(filter(f->endswith(f,"_$(name).bin"),collect(keys(d["files"]))));path=joinpath(dirname(meta),file)
        sha(path)==d["files"][file] || error("model hash changed")
        x=Array{Float64}(undef,shape...);open(io->read!(io,x),path,"r");arrays[name]=x
    end
    GroundModel(d["L"],N,n,arrays["Q"],arrays["G"],arrays["energies"],:staggered,d["mu"],d["null_singular_count"],d["zero_singular_tolerance"],d["orthogonality_error"],d["ground_energy_error"],d["eigenvector_residual"])
end

function reference_entropy(model,E,beta,regions)
    P=copy(model.Q);P[E,:].*=exp(-2beta)
    Q=Matrix(qr(P).Q)[:,1:model.n]
    values=[ExactBorn.regional_entropy(Q,r,model.N) for r in regions]
    [values...,(2values[1]+values[3])/3-values[2]]
end

function run()
    plan=TOML.parsefile(PLAN);planhash=sha(PLAN)
    for (file,hash) in plan["source_sha256"];sha(joinpath(HERE,file))==hash || error("source changed: $file");end
    for m in plan["models"],(file,hash) in m["files"];sha(joinpath(HERE,file))==hash || error("model changed");end
    models=Dict(m["L"]=>load_model(joinpath(HERE,m["path"])) for m in plan["models"])
    modelwork=Dict(L=>EntropyWorkspace(m) for (L,m) in models)
    modelkernels=Dict(L=>kernels(L) for L in keys(models))
    # Fixed discarded warm-up calls exclude compilation from per-record costs.
    for (L,m) in models
        s=StratifiedSampler(m.Q,3.);x=sample_fixed_j(s,1;rng=MersenneTwister(7+L))
        entropy!(modelwork[L],m,x.E,3.)
        reference_entropy(m,x.E,3.,modelwork[L].regions)
    end
    files=Dict{String,String}();receipts=String[]
    for job in plan["jobs"]
        stem=joinpath(PILOT,job["id"]);data=stem*".csv";receipt=stem*"_receipt.toml"
        if isfile(receipt)
            d=TOML.parsefile(receipt)
            d["status"]=="complete" && d["plan_sha256"]==planhash && d["job"]==job || error("incompatible receipt")
            sha(data)==d["files"][rel(data)] || error("completed output changed")
            files[rel(data)]=sha(data);files[rel(receipt)]=sha(receipt);push!(receipts,rel(receipt));continue
        end
        isfile(data) && error("output exists without receipt: $data")
        isfile(data*".partial") && error("partial output exists; do not overwrite")
        model=models[job["L"]];work=modelwork[job["L"]];s=StratifiedSampler(model.Q,job["beta"]);rng=MersenneTwister(job["seed"])
        maxerror=0.;sampling_time=0.;entropy_time=0.;fluctuation_time=0.;wall=time();jcounts=zeros(Int,model.n+1)
        G=zeros(model.N,model.N);Gsquare=similar(G);sumG=zeros(model.N,model.N);sumGsquare=similar(sumG);fill!(sumGsquare,0.)
        open(data*".partial","w") do io
            println(io,HEADER)
            for k in 1:job["records"]
                t=time_ns();draw=sample_stratum(s,job["jlo"],job["jhi"];rng);ts=(time_ns()-t)/1e9
                t=time_ns();entropy=entropy!(work,model,draw.E,job["beta"]);te=(time_ns()-t)/1e9
                y=[entropy.SA,entropy.SB,entropy.SC,entropy.D]
                all(isfinite,y) && minimum(y[1:3])>=-1e-10 || error("invalid entropy")
                t=time_ns();v=fluctuation!(G,Gsquare,work,modelkernels[job["L"]]);tv=(time_ns()-t)/1e9
                sumG.+=G;sumGsquare.+=Gsquare
                if k<=2
                    ref=reference_entropy(model,draw.E,job["beta"],work.regions)
                    maxerror=max(maxerror,maximum(abs.(ref-y)));maxerror<1e-9 || error("independent entropy disagreement")
                end
                sampling_time+=ts;entropy_time+=te;fluctuation_time+=tv;jcounts[draw.j+1]+=1
                row=(job["L"],job["beta"],job["z"],job["case"],job["method"],job["jlo"],job["jhi"],job["stratum_probability"],k,job["seed"],draw.j,join(draw.F,';'),join(draw.E,';'),record_digest(draw.F,draw.E,draw.j),y...,v...,ts,te,tv)
                println(io,join(row,','))
            end
        end
        mv(data*".partial",data)
        meanGpath=stem*"_meanG.bin";meanG2path=stem*"_meanGsquare.bin"
        open(io->write(io,sumG/job["records"]),meanGpath,"w")
        open(io->write(io,sumGsquare/job["records"]),meanG2path,"w")
        result=Dict("status"=>"complete","job"=>job,"plan_sha256"=>planhash,"created_at"=>string(now()),
            "elapsed_seconds"=>time()-wall,"sampling_seconds"=>sampling_time,"entropy_seconds"=>entropy_time,"fluctuation_seconds"=>fluctuation_time,
            "maximum_reference_entropy_error"=>maxerror,"j_counts"=>jcounts,"records"=>job["records"],
            "files"=>Dict(rel(p)=>sha(p) for p in (data,meanGpath,meanG2path)),"julia_version"=>string(VERSION),"blas_threads"=>BLAS.get_num_threads(),
            "matrix_format"=>"Float64 native-endian column-major Nsites x Nsites")
        save_toml(receipt,result);merge!(files,result["files"]);files[rel(receipt)]=sha(receipt);push!(receipts,rel(receipt))
        println("DONE $(job["id"]) n=$(job["records"]) seconds=$(round(time()-wall;digits=3))");flush(stdout)
    end
    files[rel(PLAN)]=sha(PLAN)
    for m in plan["models"];merge!(files,m["files"]);end
    save_toml(joinpath(PILOT,"manifest.toml"),Dict("status"=>"complete","plan_sha256"=>planhash,
        "records"=>sum(j["records"] for j in plan["jobs"]),"source_sha256"=>plan["source_sha256"],"files"=>files,"receipts"=>receipts))
    println("COMPLETE pilot")
end

if "--prepare" in ARGS;prepare();elseif "--run" in ARGS;run();else;error("Use --prepare or --run");end
