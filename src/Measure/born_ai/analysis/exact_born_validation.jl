# Independent occupation-basis oracle. No correlation-matrix entropy formula.
using LinearAlgebra, Random, SHA, TOML, Dates
BLAS.set_num_threads(1)
const HERE = @__DIR__
const SOURCE = normpath(joinpath(HERE,"..","ExactBorn.jl"))
const SOURCE_HASH = "--oracle-only" in ARGS ? "" : bytes2hex(sha256(read(SOURCE)))
if !("--oracle-only" in ARGS)
    include(SOURCE)
end

function configurations(N; particles=nothing)
    [[i for i in 1:N if (mask>>(i-1))&1==1] for mask in 0:(1<<N)-1
     if isnothing(particles) || count_ones(mask)==particles]
end

function fock_state(P)
    N,n=size(P)
    basis=configurations(N;particles=n)
    amplitudes=ComplexF64[det(P[F,:]) for F in basis]
    norm2=sum(abs2,amplitudes)
    norm2>0 || error("Singular Fock state")
    basis,amplitudes/sqrt(norm2)
end

function fock_entropy(basis,amplitudes,N,region;fermion_sign=true)
    # Global convention |F> = c†_{f1} ... c†_{fn}|0>, f1<...<fn.
    # Tensor-factor convention: region orbitals first, complement orbitals next.
    # Sorting occupied creators into that order produces one minus sign for each
    # occupied complement index b that precedes an occupied region index a.
    R=sort(collect(region)); C=setdiff(1:N,R)
    wave=zeros(ComplexF64,1<<length(R),1<<length(C))
    for (F,amplitude) in zip(basis,amplitudes)
        r=sum(1<<(j-1) for (j,i) in enumerate(R) if i in F;init=0)
        c=sum(1<<(j-1) for (j,i) in enumerate(C) if i in F;init=0)
        inversions=sum(b<a for a in R for b in C if a in F && b in F;init=0)
        sign=fermion_sign && isodd(inversions) ? -1 : 1
        wave[r+1,c+1]+=sign*amplitude
    end
    rho=wave*wave'
    normalization=real(tr(rho))
    @assert abs(normalization-1)<2e-12
    purity=real(tr(rho*rho))
    @assert 0<purity<=1+2e-12
    (;purity,S2=-log(purity))
end

function fock_oracle(P,beta,regions;ensemble)
    N,n=size(P)
    @assert N==2n
    basis,psi=fock_state(P)
    outcomes=configurations(N;particles=ensemble==:fixed_half ? n : nothing)
    probabilities=Float64[]
    raw_filter_weights=Float64[]
    entropies=zeros(length(outcomes),length(regions))
    purities=similar(entropies)
    unsigned_gap=0.
    for (i,E) in enumerate(outcomes)
        wave=copy(psi)
        push!(raw_filter_weights,sum(abs2(amplitude)*exp(-4beta*length(intersect(E,F)))
                                    for (F,amplitude) in zip(basis,psi)))
        for (j,F) in enumerate(basis)
            if ensemble==:fixed_half
                # M_E is diagonal in occupation F, amplitude exp(-2 beta N_E).
                wave[j]*=exp(-2beta*length(intersect(E,F)))
            elseif ensemble==:full_local
                # Direct product of site Kraus diagonal entries, not a determinant.
                factor=1.
                for site in 1:N
                    target_empty=site in E
                    occupied=site in F
                    mismatch=(target_empty && occupied)||(!target_empty && !occupied)
                    factor*= (mismatch ? exp(-beta) : 1.)/sqrt(1+exp(-2beta))
                end
                wave[j]*=factor
            else
                error("Unknown oracle ensemble")
            end
        end
        weight=sum(abs2,wave)
        @assert weight>0
        push!(probabilities,weight)
        wave./=sqrt(weight)
        for (j,R) in enumerate(regions)
            result=fock_entropy(basis,wave,N,R)
            entropies[i,j]=result.S2
            purities[i,j]=result.purity
            complement=fock_entropy(basis,wave,N,setdiff(1:N,R))
            @assert abs(result.S2-complement.S2)<2e-12
            unsigned=fock_entropy(basis,wave,N,R;fermion_sign=false)
            unsigned_gap=max(unsigned_gap,abs(unsigned.S2-result.S2))
        end
    end
    total=sum(probabilities)
    expected_total=ensemble==:fixed_half ? sum(binomial(n,j)^2*exp(-4beta*j) for j in 0:n) : 1.
    @assert abs(total-expected_total)<2e-12*max(1,expected_total)
    probabilities./=total
    mean_entropy=vec(probabilities'*entropies)
    mean_purity=vec(probabilities'*purities)
    log_mean_purity=-log.(mean_purity)
    @assert all(mean_entropy .>= log_mean_purity.-2e-12)
    raw_filter_weight_sum=sum(raw_filter_weights)
    expected_filter_sum=ensemble==:fixed_half ? expected_total : 2.0^n*(1+exp(-4beta))^n
    @assert abs(raw_filter_weight_sum-expected_filter_sum)<2e-12*max(1,expected_filter_sum)
    (;outcomes,probabilities,kraus_weight_sum=total,raw_filter_weights,raw_filter_weight_sum,entropies,purities,
      mean_entropy,mean_purity,log_mean_purity,unsigned_gap)
end

function fixtures()
    rng=MersenneTwister(20261027)
    result=[]
    for N in (4,6)
        n=N÷2
        raw_real=randn(rng,N,n)
        raw_complex=randn(rng,N,n)+im*randn(rng,N,n)
        Q=Matrix(qr(raw_complex).Q)[:,1:n]
        transform=Matrix{ComplexF64}(I,n,n)
        for i in 1:n,j in i:n
            transform[i,j]=i==j ? 1+.3i : .2randn(rng)+.1im*randn(rng)
        end
        product=zeros(ComplexF64,N,n)
        for j in 1:n;product[2j-1,j]=1;end
        for (name,P) in (("real_nonorthogonal",raw_real),("complex_orthogonal",Q),
                         ("complex_same_span_nonorthogonal",Q*transform),("product_interleaved",product))
            push!(result,(;N,name,P))
        end
    end
    result
end

function regions_for(N)
    n=N÷2
    [collect(1:n),collect(n+1:N),collect(1:2:N),collect(2:2:N),unique([1,3,N]),Int[],collect(1:N)]
end

function oracle_self_test()
    cases=0;trajectories=0;max_unsigned_gap=0.;max_jensen_gap=0.
    for f in fixtures(),beta in (0.,.35,1.2,3.),ensemble in (:fixed_half,:full_local)
        regions=regions_for(f.N)
        reference=fock_oracle(f.P,beta,regions;ensemble)
        cases+=1;trajectories+=length(reference.outcomes)
        max_unsigned_gap=max(max_unsigned_gap,reference.unsigned_gap)
        max_jensen_gap=max(max_jensen_gap,maximum(reference.mean_entropy-reference.log_mean_purity))
        f.name=="product_interleaved" && @assert maximum(abs.(reference.entropies))<2e-12
        if beta==0
            @assert maximum(abs.(reference.probabilities.-1/length(reference.outcomes)))<2e-12
            @assert maximum(abs.(reference.entropies.-reference.entropies[1:1,:]))<2e-12
        end
    end
    @assert max_unsigned_gap>1e-3 # Negative control: orbital-order sign matters.
    @assert max_jensen_gap>1e-3 # Negative control: mean entropy != -log mean purity.
    (;cases,trajectories,max_unsigned_gap,max_jensen_gap)
end

function expect_argument_error(f)
    caught=false
    try
        f()
    catch error
        error isa ArgumentError || rethrow(error)
        caught=true
    end
    @assert caught "Expected input/count validation to reject the call"
end

function production_audit()
    source_hash=SOURCE_HASH
    calculate=ExactBorn.born_trajectory_entropy
    rows=Dict{String,Any}[]
    totals=Dict("cases"=>0,"trajectories"=>0,"regional_trajectory_comparisons"=>0,"input_rejections"=>0)
    maxima=Dict("probability_error"=>0.,"log_probability_error"=>0.,"log_weight_error"=>0.,
                "trajectory_entropy_error"=>0.,"mean_entropy_error"=>0.,"mean_purity_error"=>0.,
                "annealed_entropy_error"=>0.,"covariance_error"=>0.,"raw_weight_sum_log_error"=>0.,
                "normalization_error"=>0.,"gauge_invariance_error"=>0.,"noncontiguous_unsigned_entropy_error"=>0.,
                "mean_vs_annealed_entropy_difference"=>0.,"full_conditioned_vs_fixed_probability_error"=>0.)
    cache=Dict{Any,Any}()
    for f in fixtures(),beta in (0.,.35,1.2,3.),ensemble in (:fixed_half,:full_local)
        regions=regions_for(f.N)
        names=(:left,:right,:odd,:even,:mixed,:empty,:all)
        named=NamedTuple{names}(Tuple(regions))
        ref=fock_oracle(f.P,beta,regions;ensemble)
        result=calculate(f.P,beta,named;ensemble)
        compact=calculate(f.P,beta,named;ensemble,keep_trajectories=false)
        @assert result.nconfigurations==length(ref.outcomes)
        @assert result.region_names==collect(names) && result.regions==regions
        @assert result.ensemble==ensemble && result.nsites==f.N && result.nparticles==f.N÷2
        @assert result.probability_sum≈1 && abs(result.probability_mass_before_normalization-1)<2e-12
        @assert result.normalization_error<2e-12 && result.weight_sum_error<2e-12
        @assert all(getproperty(compact,field)===nothing for field in
                    (:configurations,:probabilities,:log_probabilities,:log_weights,:entropies))
        @assert compact.mean_entropy==result.mean_entropy && compact.mean_purity==result.mean_purity
        @assert compact.entropy_covariance==result.entropy_covariance
        lookup=Dict(Tuple(E)=>i for (i,E) in enumerate(ref.outcomes))
        @assert length(unique(Tuple.(result.configurations)))==length(ref.outcomes)
        @assert Set(Tuple.(result.configurations))==Set(keys(lookup))
        permutation=[lookup[Tuple(E)] for E in result.configurations]
        p=ref.probabilities[permutation]
        entropy=ref.entropies[permutation,:]
        raw=ref.raw_filter_weights[permutation]
        centered=ref.entropies.-transpose(ref.mean_entropy)
        covariance=centered'*(ref.probabilities.*centered)
        checks=Dict("probability_error"=>maximum(abs.(result.probabilities-p)),
            "log_probability_error"=>maximum(abs.(result.log_probabilities-log.(p))),
            "log_weight_error"=>maximum(abs.(result.log_weights-log.(raw))),
            "trajectory_entropy_error"=>maximum(abs.(result.entropies-entropy)),
            "mean_entropy_error"=>maximum(abs.(result.mean_entropy-ref.mean_entropy)),
            "mean_purity_error"=>maximum(abs.(result.mean_purity-ref.mean_purity)),
            "annealed_entropy_error"=>maximum(abs.(result.annealed_entropy-ref.log_mean_purity)),
            "covariance_error"=>maximum(abs.(result.entropy_covariance-covariance)),
            "raw_weight_sum_log_error"=>max(abs(result.log_weight_sum-log(ref.raw_filter_weight_sum)),
                                         abs(result.expected_log_weight_sum-log(ref.raw_filter_weight_sum))),
            "normalization_error"=>result.normalization_error)
        for (key,value) in checks
            @assert value<2e-11 (f.name,f.N,beta,ensemble,key,value)
            maxima[key]=max(maxima[key],value)
        end
        f.name=="product_interleaved" && @assert maximum(abs.(result.entropies))<2e-12
        if beta==0
            @assert maximum(abs.(result.probabilities.-1/result.nconfigurations))<2e-12
            @assert maximum(abs.(result.entropies.-result.entropies[1:1,:]))<2e-12
        end
        if f.name=="complex_same_span_nonorthogonal"
            previous=cache[(f.N,"complex_orthogonal",beta,ensemble)]
            gap=max(maximum(abs.(result.probabilities-previous.probabilities)),
                    maximum(abs.(result.entropies-previous.entropies)))
            @assert gap<2e-11
            maxima["gauge_invariance_error"]=max(maxima["gauge_invariance_error"],gap)
        end
        if ensemble==:full_local
            fixed=cache[(f.N,f.name,beta,:fixed_half)]
            idx=findall(E->length(E)==f.N÷2,result.configurations)
            probabilities=result.probabilities[idx]
            probabilities./=sum(probabilities)
            @assert result.configurations[idx]==fixed.configurations
            gap=maximum(abs.(probabilities-fixed.probabilities))
            @assert gap<2e-12
            maxima["full_conditioned_vs_fixed_probability_error"]=max(maxima["full_conditioned_vs_fixed_probability_error"],gap)
        end
        cache[(f.N,f.name,beta,ensemble)]=result
        maxima["noncontiguous_unsigned_entropy_error"]=max(maxima["noncontiguous_unsigned_entropy_error"],ref.unsigned_gap)
        maxima["mean_vs_annealed_entropy_difference"]=max(maxima["mean_vs_annealed_entropy_difference"],maximum(ref.mean_entropy-ref.log_mean_purity))
        totals["cases"]+=1;totals["trajectories"]+=length(ref.outcomes)
        totals["regional_trajectory_comparisons"]+=length(ref.outcomes)*length(regions)
        push!(rows,Dict("state"=>f.name,"N"=>f.N,"beta"=>beta,"ensemble"=>String(ensemble),
                        "outcomes"=>length(ref.outcomes),"max_errors"=>checks,
                        "mean_entropy"=>result.mean_entropy,"annealed_entropy"=>result.annealed_entropy))
    end
    @assert maxima["noncontiguous_unsigned_entropy_error"]>1e-3
    @assert maxima["mean_vs_annealed_entropy_difference"]>1e-3
    f=fixtures()[2]
    for region in (Int[],[4,1],[1,3],collect(1:4))
        reference=fock_oracle(f.P,.35,[region];ensemble=:fixed_half)
        result=calculate(f.P,.35,region)
        @assert result.region_names==[:region]
        @assert maximum(abs.(result.mean_entropy-reference.mean_entropy))<2e-12
    end
    list_result=calculate(f.P,.35,regions_for(4))
    dict_result=calculate(f.P,.35,Dict(:x=>[1,3],:y=>[2,4]))
    @assert length(list_result.region_names)==7 && length(dict_result.region_names)==2
    for (j,name) in enumerate(dict_result.region_names)
        reference=fock_oracle(f.P,.35,[name==:x ? [1,3] : [2,4]];ensemble=:fixed_half)
        @assert abs(dict_result.mean_entropy[j]-reference.mean_entropy[1])<2e-12
    end
    function rejects(callback)
        expect_argument_error(callback)
        totals["input_rejections"]+=1
    end
    for ensemble in (:fixed_half,:full_local)
        count=ensemble==:fixed_half ? 6 : 16
        rejects(()->calculate(f.P,1.,[1,3];ensemble,max_configs=count-1))
        @assert calculate(f.P,1.,[1,3];ensemble,max_configs=count).nconfigurations==count
        huge=Matrix{Float64}(I,64,64)[:,1:32]
        rejects(()->calculate(huge,1.,[1];ensemble,max_configs=1000))
    end
    for beta in (-.1,Inf,NaN,1000.)
        rejects(()->calculate(f.P,beta,[1]))
    end
    rejects(()->calculate(f.P,1.,[1];ensemble=:unknown))
    rejects(()->calculate(f.P,1.,[1,1]))
    rejects(()->calculate(f.P,1.,[0]))
    rejects(()->calculate(f.P,1.,[5]))
    rejects(()->calculate(f.P,1.,[1.5]))
    rejects(()->calculate(zeros(4,2),1.,[1]))
    rejects(()->calculate(fill(NaN,4,2),1.,[1]))
    rejects(()->calculate(ones(5,2),1.,[1]))
    rejects(()->calculate(f.P,1.,[1];max_configs=0))
    rejects(()->calculate(f.P,1.,[1];normalization_tolerance=-1.))
    @assert bytes2hex(sha256(read(SOURCE)))==source_hash "Production source changed during audit"
    receipt=Dict("status"=>"PASS","generated_utc"=>string(now(UTC)),"counts"=>totals,"maxima"=>maxima,
        "method"=>"Full occupation-basis determinant amplitudes, diagonal local Kraus amplitudes, fermionic orbital reorder, explicit reduced density matrix, trace(rho_R^2). No correlation-matrix entropy identity or production helpers used by the oracle.",
        "fermionic_partition_convention"=>"Global creation operators ordered by ascending site. Before partial trace, move all R modes before Rc modes; multiply each occupied F amplitude by (-1)^number{(a,b): a in F intersect R,b in F intersect Rc,b<a}. Reshape into row=R occupation and column=Rc occupation, then rho_R=wave*wave'. This includes noncontiguous regions.",
        "protocols"=>Dict("fixed_half"=>"All C(N,N/2) E; M_E=exp(-2 beta N_E), normalize by independent sum_E norm^2.",
                           "full_local"=>"All 2^N E; product site K+/-=exp[+/-beta(n-1/2)]/sqrt(2cosh beta). Complete POVM probabilities sum to 1 without conditioning."),
        "normalization_checks"=>"Both actual Born mass and raw M_E norm sums checked independently. Full_local conditioned on |E|=N/2 equals fixed_half.",
        "zero_beta_and_product"=>"At beta=0 outcome probabilities are uniform in each selected ensemble and all normalized trajectories share the initial entropy. Product-state trajectories have zero entropy at every tested beta.",
        "negative_controls"=>"Omitting fermionic regrouping signs and replacing mean S2 by -log(mean purity) each produce clearly nonzero differences in these fixtures.",
        "limits"=>"Float64/ComplexF64 small-system cross-checks, beta<=3. No assertion of arbitrary-precision exactness or reliability at arbitrary filtering condition number.",
        "regions"=>["left","right","odd sites","even sites","noncontiguous mixed","empty","all"],
        "seed"=>20261027,"julia_version"=>string(VERSION),"cases"=>rows,
        "files"=>Dict("ExactBorn.jl"=>source_hash,"analysis/exact_born_validation.jl"=>bytes2hex(sha256(read(@__FILE__)))))
    open(joinpath(HERE,"exact_born_validation.toml"),"w") do io;TOML.print(io,receipt;sorted=true);end
    println("PASS: independent Fock-space ExactBorn validation")
    println(totals)
    println(maxima)
end

if "--oracle-only" in ARGS
    println("PASS: independent oracle-only self checks (no production receipt): ",oracle_self_test())
else
    production_audit()
end
