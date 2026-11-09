# Independent deterministic audit of staggered-mass and zero-mode prescriptions.
# No Monte Carlo and no modification of saved states or existing datasets.
push!(LOAD_PATH, normpath(joinpath(@__DIR__, "..", "..", "..")))
using KAPDQMC, LinearAlgebra, SHA, TOML, Dates
BLAS.set_num_threads(1)
include(joinpath(@__DIR__,"..","original_swap_scaling.jl"))

const STATE_AUDIT_SIZES = collect(3:3:24)
const STATE_AUDIT_OUT = @__DIR__
sha_file(path) = bytes2hex(sha256(read(path)))

function region_entropy(C, indices)
    values = eigvals(Symmetric(C[indices, indices]))
    minimum(values) >= -1e-10 && maximum(values) <= 1+1e-10 ||
        error("Correlation eigenvalues outside [0,1]")
    -sum(log.(values.^2 + (1 .- values).^2))
end

function analytic_zero_orbitals(L)
    a=L^2
    # Both sublattices use the SAME analytic valley basis. No SVD-basis pairing.
    theta = [2pi*(x+y)/3 for y in 0:L-1 for x in 0:L-1]
    U = (sqrt(2)/L).*hcat(cos.(theta),sin.(theta))
    Q0 = zeros(2a,2)
    Q0[1:2:end,:] .= U./sqrt(2)
    Q0[2:2:end,:] .= -U./sqrt(2)
    Q0
end

function independent_mass_orbitals(factor,mu)
    sigma=copy(factor.S)
    null=findall(<(1e-10),sigma)
    @assert length(null)==2
    sigma[null].=0.
    epsilon=sqrt.(sigma.^2 .+mu^2)
    n=length(sigma)
    Q=zeros(2n,n)
    Q[1:2:end,:].=factor.U.*(sigma./sqrt.(2 .*epsilon.*(epsilon.+mu)))'
    Q[2:2:end,:].=-factor.V.*sqrt.((epsilon.+mu)./(2 .*epsilon))'
    Q,-epsilon
end

function fourier_mass_projector(L,mu)
    n=L^2
    W=zeros(ComplexF64,n,n); ca=zeros(n); cb=zeros(n); cab=zeros(ComplexF64,n)
    for ky in 0:L-1,kx in 0:L-1
        j=1+kx+L*ky
        f=1+cis(2pi*kx/L)+cis(-2pi*ky/L)
        if (kx==ky==L÷3) || (kx==ky==2L÷3); f=0im; end
        eps=hypot(abs(f),mu)
        ca[j]=(1-mu/eps)/2; cb[j]=(1+mu/eps)/2; cab[j]=-f/(2eps)
        for y in 0:L-1,x in 0:L-1
            W[1+x+L*y,j]=cis(2pi*(kx*x+ky*y)/L)/L
        end
    end
    C=zeros(ComplexF64,2n,2n)
    C[1:2:end,1:2:end]=(W.*ca')*W'
    C[2:2:end,2:2:end]=(W.*cb')*W'
    C[1:2:end,2:2:end]=(W.*transpose(cab))*W'
    C[2:2:end,1:2:end]=C[1:2:end,2:2:end]'
    @assert maximum(abs.(imag.(C)))<1e-10
    real.(C)
end

function state_audit()
    sources=[@__FILE__,normpath(joinpath(@__DIR__,"..","..","..","public","Geometry.jl")),
        normpath(joinpath(@__DIR__,"..","original_swap_scaling.jl"))]
    source_hashes=Dict(abspath(p)=>sha_file(p) for p in sources)
    rows=NamedTuple[]
    comparison=NamedTuple[]
    implementation_checks=NamedTuple[]
    max_verified_error=0.0
    for L in STATE_AUDIT_SIZES
        a=L^2; N=2a
        H=nnK_Matrix("HoneyComb120",[L,L])
        energies,V=LAPACK.syevd!('V','L',copy(H))
        neg=findall(<(-1e-10),energies)
        zero=findall(x->abs(x)<1e-10,energies)
        @assert length(neg)==a-2 && length(zero)==4
        Q0=analytic_zero_orbitals(L)
        zero_residual=norm(H*Q0)
        Ppair=hcat(V[:,neg],Q0)
        Pold=V[:,1:a]
        factor=svd(H[1:2:end,2:2:end])
        Pmass,Emass=independent_mass_orbitals(factor,1e-8)
        Psmall,Esmall=independent_mass_orbitals(factor,1e-9)
        # Explicit mu -> 0+ projector: fill both B-sublattice valley zero modes.
        QBzero=zeros(N,2); QBzero[2:2:end,:].=sqrt(2).*Q0[1:2:end,:]
        Plimit=hcat(V[:,neg],QBzero); Climit=Plimit*Plimit'
        ground_energy=sum(energies[neg])
        A=area_index("HoneyComb120",[L,L],([1,1],[L÷3,L]))
        B=area_index("HoneyComb120",[L,L],([1,1],[L÷3,2L÷3]))
        Cregion=area_index("HoneyComb120",[L,L],([1,1],[L,L÷3]))
        @assert length(A)==length(Cregion)==2a÷3 && length(B)==4a÷9
        @assert length(unique(A))==length(A) && length(unique(B))==length(B)
        px=[2*(mod(x+1,L)+L*y)+s for y in 0:L-1 for x in 0:L-1 for s in 1:2]
        py=[2*(x+L*mod(y+1,L))+s for y in 0:L-1 for x in 0:L-1 for s in 1:2]
        reflection=[2*(y+L*x)+(3-s) for y in 0:L-1 for x in 0:L-1 for s in 1:2]
        gamma=repeat([1.,-1.],a)
        @assert norm(H[reflection,reflection]-H)<1e-12
        bystate=Dict{String,NamedTuple}()
        for (name,P,mu,occupied_energies) in (("staggered_mu1e-8",Pmass,1e-8,Emass),
                ("staggered_mu1e-9",Psmall,1e-9,Esmall),
                ("paired_analytic_minus_background",Ppair,0.,vcat(energies[neg],zeros(2))),
                ("legacy_syevd",Pold,0.,energies[1:a]))
            C=P*P'
            orthogonality=norm(P'*P-I)
            Hstate=H+Diagonal(mu.*gamma)
            energy_error=abs(sum(P.*(Hstate*P))-sum(occupied_energies))
            original_energy_excess=sum(P.*(H*P))-ground_energy
            eig_residual=norm(Hstate*P-P.*occupied_energies')
            tx=norm(C[px,px]-C); ty=norm(C[py,py]-C)
            reflection_error=norm(C[reflection,reflection]-C)
            compensated_reflection_error=norm(C[reflection,reflection]-(I-C).*(gamma*gamma'))
            same_A_error=norm(C[1:2:end,1:2:end]-.5I)
            same_B_error=norm(C[2:2:end,2:2:end]-.5I)
            Sa=region_entropy(C,A); Sb=region_entropy(C,B); Sc=region_entropy(C,Cregion)
            D=(2Sa+Sc)/3-Sb
            row=(L=L,state=name,mu=mu,mu_times_L=mu*L,beta=0.0,S_A=Sa,S_B=Sb,S_C=Sc,D=D,A_minus_B=Sa-Sb,
                 S_A_minus_S_C=Sa-Sc,negative_dimension=length(neg),zero_dimension=length(zero),
                 occupied_zero_dimension=2,orthogonality_error=orthogonality,energy_error=energy_error,
                 original_H_energy_excess=original_energy_excess,eigenvector_residual=eig_residual,
                 zero_H_residual=zero_residual,translation_x_error=tx,translation_y_error=ty,
                 xy_AB_reflection_error=reflection_error,particle_hole_compensated_reflection_error=compensated_reflection_error,
                 same_sublattice_A_halfI_error=same_A_error,same_sublattice_B_halfI_error=same_B_error,
                 site_density_max_deviation=maximum(abs.(diag(C).-0.5)),distance_to_mu0plus_projector=norm(C-Climit))
            push!(rows,row); bystate[name]=row
            @assert orthogonality<1e-9 && energy_error<1e-9 && eig_residual<1e-9
            max_verified_error=max(max_verified_error,orthogonality,energy_error,zero_residual,eig_residual)
            if mu>0
                @assert maximum((tx,ty,compensated_reflection_error,abs(Sa-Sc)))<1e-9
                max_verified_error=max(max_verified_error,tx,ty,compensated_reflection_error,abs(Sa-Sc))
                model=OriginalSwapScaling.ground_model(L;state=:staggered,mu=mu)
                implementation_error=norm(model.G-C)
                fourier_error=norm(fourier_mass_projector(L,mu)-C)
                @assert implementation_error<1e-9 && fourier_error<1e-9
                max_verified_error=max(max_verified_error,implementation_error,fourier_error)
                direct_syevd_error=NaN
                if L<=9
                    _,Vdirect=LAPACK.syevd!('V','L',copy(Hstate))
                    Pdirect=Vdirect[:,1:a]
                    direct_syevd_error=norm(Pdirect*Pdirect'-C)
                    @assert direct_syevd_error<1e-5
                end
                push!(implementation_checks,(L=L,mu=mu,implementation_projector_error=implementation_error,
                    independent_fourier_projector_error=fourier_error,direct_mass_syevd_projector_error=direct_syevd_error))
            elseif name=="paired_analytic_minus_background"
                @assert maximum((tx,ty,reflection_error,same_A_error,same_B_error,abs(Sa-Sc)))<1e-9
                max_verified_error=max(max_verified_error,tx,ty,reflection_error,same_A_error,same_B_error,abs(Sa-Sc))
            end
        end
        main=bystate["staggered_mu1e-8"]; small=bystate["staggered_mu1e-9"]
        paired=bystate["paired_analytic_minus_background"]; legacy=bystate["legacy_syevd"]
        push!(comparison,(L=L,mass_D=main.D,smaller_mass_D=small.D,mass_sensitivity_D=main.D-small.D,
            legacy_D=legacy.D,legacy_minus_mass_D=legacy.D-main.D,paired_background_D=paired.D,
            mass_S_A=main.S_A,smaller_mass_S_A=small.S_A,mass_sensitivity_S_A=main.S_A-small.S_A,
            legacy_S_A=legacy.S_A,legacy_S_A_minus_S_C=legacy.S_A_minus_S_C))
        println("L=$L mass D=$(main.D) mass sensitivity=$(main.D-small.D) legacy D=$(legacy.D)")
        flush(stdout)
    end
    function write_rows(path,table)
        open(path,"w") do io
            println(io,join(string.(keys(first(table))),','))
            for row in table; println(io,join(values(row),',')); end
        end
    end
    csvpath=joinpath(STATE_AUDIT_OUT,"born_scaling_state_audit.csv")
    cmppath=joinpath(STATE_AUDIT_OUT,"born_scaling_state_comparison.csv")
    write_rows(csvpath,rows); write_rows(cmppath,comparison)
    secants=NamedTuple[]
    fits=NamedTuple[]
    for state in ("staggered_mu1e-8","staggered_mu1e-9","paired_analytic_minus_background","legacy_syevd")
        selected=filter(r->r.state==state,rows)
        for (left,right) in zip(selected[1:end-1],selected[2:end])
            push!(secants,(state=state,L_low=left.L,L_high=right.L,
                local_log_slope=(right.D-left.D)/log(right.L/left.L)))
        end
        for Lmin in (3,6,9,12)
            selectedfit=filter(r->r.L>=Lmin,selected)
            sizes=Float64[r.L for r in selectedfit]; d=[r.D for r in selectedfit]
            for order in (0,1)
                design=order==0 ? hcat(ones(length(sizes)),log.(sizes)) :
                    hcat(ones(length(sizes)),log.(sizes),1 ./sizes)
                params=design\d; residual=design*params-d
                push!(fits,(state=state,Lmin=Lmin,Lmax=24,inverse_order=order,n=length(sizes),
                    constant=params[1],c_window=params[2],inv1=order==1 ? params[3] : 0.,
                    rmse=sqrt(sum(abs2,residual)/length(sizes)),residual_dof=length(sizes)-length(params)))
            end
        end
    end
    slopepath=joinpath(STATE_AUDIT_OUT,"born_scaling_state_local_slopes.csv")
    fitpath=joinpath(STATE_AUDIT_OUT,"born_scaling_state_window_fits.csv")
    write_rows(slopepath,secants); write_rows(fitpath,fits)
    implementationpath=joinpath(STATE_AUDIT_OUT,"born_scaling_state_implementation_checks.csv")
    write_rows(implementationpath,implementation_checks)
    @assert all(sha_file(p)==hash for (p,hash) in source_hashes)
    files=Dict(basename(p)=>sha_file(p) for p in (csvpath,cmppath,slopepath,fitpath,implementationpath,@__FILE__))
    receipt=Dict("status"=>"PASS","calculation"=>"Deterministic beta=0 entropies only; no Monte Carlo",
        "sizes"=>STATE_AUDIT_SIZES,"rows"=>length(rows),"maximum_verified_error"=>max_verified_error,
        "primary_state_rule"=>"Negative-energy Slater of H_mu=H+mu diag(+A,-B), mu=1e-8, sensitivity mu=1e-9. Independently built from SVD of H_AB with the two analytically zero singular values set exactly to zero. Stable negative-eigenvector formula explicitly fills the two B zero modes for mu>0.",
        "background_paired_rule"=>"P=[all strictly negative-energy orbitals,Q0]; Q0_A=U/sqrt2, Q0_B=-U/sqrt2; U=(sqrt2/L)[cos(2pi(x+y)/3),sin(2pi(x+y)/3)], x,y=0:L-1. Background only; not the chosen production state.",
        "legacy_rule"=>"First L^2 vectors returned by LAPACK.syevd! for exactly the same Hamiltonian; this is the original algorithm, not a reload of an earlier binary state.",
        "geometry"=>"Both sites per cell. A=(L/3)xL, B=(L/3)x(2L/3), C=Lx(L/3), all start at cell (1,1). D=(2 S_A+S_C)/3-S_B.",
        "mass_symmetries"=>["primitive-cell translations","real orbitals/time reversal","particle-hole compensated x-y and A-B exchange: R C_mu R^T = Gamma (I-C_mu) Gamma; implies equal beta=0 S_A and S_C"],
        "mass_scope"=>"The staggered chemical potential changes every nonzero-energy orbital as well as lifting the four Dirac zero modes. All masses satisfy mu*L << 1 in this audit. Ordinary x-y plus A-B reflection changes the sign of mu and is NOT a symmetry of one chosen massive projector.",
        "not_claimed"=>["full point-group invariance","positive-beta Born averages insensitive to zero-mode choice","finite-window coefficients equal asymptotic corner coefficient"],
        "fit_scope"=>"Unweighted deterministic finite-size diagnostics only, not stochastic errors, accepted scaling models, or asymptotic coefficient estimates.",
        "mass_sensitivity_max_D"=>maximum(abs(r.mass_sensitivity_D) for r in comparison),
        "mass_sensitivity_max_S_A"=>maximum(abs(r.mass_sensitivity_S_A) for r in comparison),
        "source_hashes"=>source_hashes,"files"=>files)
    open(joinpath(STATE_AUDIT_OUT,"born_scaling_state_audit.toml"),"w") do io
        TOML.print(io,receipt;sorted=true)
    end
    println("PASS: deterministic state audit, maximum verified error=",max_verified_error)
end

state_audit()
