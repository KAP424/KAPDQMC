# Additional deterministic beta=0 sizes. Does not modify the original state audit.
using LinearAlgebra, SHA, TOML
BLAS.set_num_threads(1)
include(joinpath(@__DIR__,"..","original_swap_scaling.jl"))
using .OriginalSwapScaling

const BETA0_LARGE_OUT=normpath(joinpath(@__DIR__,"..","data","born_scaling"))
large_sha(p)=bytes2hex(sha256(read(p)))
function large_write_table(path,rows)
    open(path,"w") do io
        println(io,join(string.(keys(first(rows))),','))
        for row in rows;println(io,join(values(row),','));end
    end
end

function beta0_large_sizes()
    mkpath(BETA0_LARGE_OUT)
    input=joinpath(@__DIR__,"born_scaling_state_audit.csv")
    sources=[@__FILE__,joinpath(@__DIR__,"..","original_swap_scaling.jl"),input]
    hashes=Dict(abspath(p)=>large_sha(p) for p in sources)
    combined=NamedTuple[]
    lines=readlines(input); header=split(lines[1],','); columns=Dict(x=>i for (i,x) in enumerate(header))
    for line in lines[2:end]
        fields=split(line,',')
        fields[columns["state"]]=="staggered_mu1e-8" || continue
        push!(combined,(L=parse(Int,fields[columns["L"]]),D=parse(Float64,fields[columns["D"]])))
    end
    rows=NamedTuple[]
    for L in (27,30,36)
        started=time()
        model=ground_model(L;state=:staggered,mu=1e-8)
        regions=region_indices(L)
        entropy=Float64[]
        for indices in values(regions)
            values_=eigvals(Symmetric(model.G[indices,indices]))
            @assert minimum(values_) >= -1e-10 && maximum(values_) <= 1+1e-10
            push!(entropy,-sum(log.(values_.^2 .+(1 .-values_).^2)))
        end
        SA,SB,SC=entropy; D=(2SA+SC)/3-SB
        @assert abs(SA-SC)<1e-8
        push!(rows,(L=L,mu=1e-8,mu_times_L=1e-8L,beta=0.,S_A=SA,S_B=SB,S_C=SC,D=D,
            S_A_minus_S_C=SA-SC,orthogonality_error=model.orthogonality_error,
            ground_energy_error=model.ground_energy_error,eigenvector_residual=model.eigenvector_residual,
            null_singular_count=model.null_singular_count,elapsed_seconds=time()-started))
        push!(combined,(L=L,D=D))
        println("Additional beta0 L=$L D=$D elapsed=$(round(time()-started,digits=2))s")
        flush(stdout)
    end
    sort!(combined;by=r->r.L)
    fits=NamedTuple[]
    for Lmin in (6,9,12,15,18),Lmax in (24,30,36),order in (0,1)
        data=filter(r->Lmin<=r.L<=Lmax,combined)
        length(data)>=order+4 || continue
        sizes=Float64[r.L for r in data]; d=[r.D for r in data]
        design=order==0 ? hcat(ones(length(sizes)),log.(sizes)) : hcat(ones(length(sizes)),log.(sizes),1 ./sizes)
        params=design\d; residual=design*params-d
        push!(fits,(Lmin=Lmin,Lmax=Lmax,inverse_order=order,n=length(sizes),constant=params[1],c_window=params[2],
            inv1=order==1 ? params[3] : 0.,rmse=sqrt(sum(abs2,residual)/length(sizes)),residual_dof=length(sizes)-length(params)))
    end
    secants=NamedTuple[]
    for (left,right) in zip(combined[1:end-1],combined[2:end])
        push!(secants,(L_low=left.L,L_high=right.L,local_log_slope=(right.D-left.D)/log(right.L/left.L)))
    end
    csvpath=joinpath(BETA0_LARGE_OUT,"beta0_mass_large_sizes.csv")
    fitpath=joinpath(BETA0_LARGE_OUT,"beta0_mass_combined_window_fits.csv")
    slopepath=joinpath(BETA0_LARGE_OUT,"beta0_mass_combined_local_slopes.csv")
    large_write_table(csvpath,rows);large_write_table(fitpath,fits);large_write_table(slopepath,secants)
    @assert all(large_sha(p)==h for (p,h) in hashes)
    base=normpath(joinpath(@__DIR__,".."))
    receipt=Dict("status"=>"PASS","new_sizes"=>[27,30,36],"mu"=>1e-8,"beta"=>0.,
        "calculation"=>"Deterministic entropies of the staggered-mass negative-energy Slater. No Monte Carlo and no positive-beta data.",
        "combined_sizes"=>[r.L for r in combined],
        "fit_scope"=>"Unweighted finite-size diagnostics of D=b+c log(L)+d/L (or d=0). No stochastic uncertainty, no fitted model selection, and no assertion that c_window is the asymptotic coefficient.",
        "source_hashes"=>hashes,"files"=>Dict(replace(relpath(p,base),'\\'=>'/')=>large_sha(p) for p in (csvpath,fitpath,slopepath)))
    open(joinpath(BETA0_LARGE_OUT,"beta0_mass_large_sizes.toml"),"w") do io;TOML.print(io,receipt;sorted=true);end
    println("PASS: additional deterministic beta0 sizes")
end

beta0_large_sizes()
