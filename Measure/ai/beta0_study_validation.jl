# Independent dense real-space checks for the beta-zero runner. The reference
# lives in a separate module and uses neither the new runner's Bloch model nor
# its cached matrices to construct its finite- or infinite-time orbitals.
include(joinpath(@__DIR__, "beta0_study_experiments.jl"))
module B0RealReference
include(joinpath(@__DIR__, "second_class_extra_validation.jl"))
end

function beta0_study_validation_main()
    BLAS.set_num_threads(1)
    files=unique(vcat(B0_CODE_FILES,["beta0_study_validation.jl",
        "second_class_extra_validation.jl","validate_occupancy_classes.jl"]))
    hashes=Dict(f=>bp_hash(joinpath(@__DIR__,f)) for f in files)
    rows=NamedTuple[]; models=Dict{Tuple{Int,Int},Any}(); references=Dict{Int,Any}()
    for p in beta0_study_patterns()
        L=3p.m; n=2L^2
        if !haskey(references,L)
            h=B0RealReference.validation_hamiltonian(L)
            references[L]=(h,Matrix(exp(Symmetric(-.25h))))
        end
        h,ek=references[L]
        q0,F,E=B0RealReference.validation_initial(p,L)
        qinf,lim=B0RealReference.independent_supported_limit(h,q0)
        ground=abs(lim.energy-lim.ground_energy)<1e-8
        model=get!(()->FE.BlochModel(L,p.m),models,(L,p.m))
        # Shared eigensystem objects are reused among patterns. FE preparation
        # must leave them unchanged, including at degenerate energies.
        model_signature=bytes2hex(sha256(join(string.(vcat(vec.(model.hamiltonians)...)),',')))
        pre=FE.prepare_pattern(model,p.occupied;tol=1e-10)
        comp=FE.prepare_pattern(model,.!p.occupied;tol=1e-10)
        @assert model_signature==bytes2hex(sha256(join(string.(vcat(vec.(model.hamiltonians)...)),',')))
        @assert F==FE.occupied_indices(pre)
        @assert abs(pre.accessible_energy-lim.energy)<1e-8
        isnothing(p.expected_ground) || @assert p.expected_ground==ground
        gamma=Diagonal([isodd(i) ? 1. : -1. for i in 1:n])
        @assert norm(gamma*h+h*gamma)==0
        region(w,hh)=[s+2*((x-1)+L*(y-1)) for y in 1:hh for x in 1:w for s in 1:2]
        regions=(region(L÷3,L),region(L,L÷3),region(L÷3,2L÷3))
        times=p.name in ("uniform_A","near02_irregular3","pure02_irregular4") ?
            (0.,1.,4.,10.,Inf) : (0.,1.,4.,Inf)
        dense=copy(q0); last=0.
        for theta in times
            if isinf(theta)
                reference=qinf
            else
                dense=B0RealReference.validation_dense_evolve(dense,ek,round(Int,4(theta-last)))
                last=theta; reference=dense
            end
            cref=reference*reference'
            state=FE.evolve(pre,theta); c=FE.correlation_matrix(state)
            complement=FE.evolve(comp,theta); ccomp=FE.correlation_matrix(complement)
            covariance_error=norm(c-cref)/L
            complement_covariance_error=norm(ccomp-(I-gamma*c*gamma))/L
            reference_entropies=[B0RealReference.validation_entropy(cref[ix,ix]) for ix in regions]
            e=b0_entropies(state;rectangle_threads=1)
            ec=b0_entropies(complement;rectangle_threads=1)
            entropies=[e.sa,e.sc,e.sb]; complement_entropies=[ec.sa,ec.sc,ec.sb]
            entropy_error=maximum(abs.(entropies-reference_entropies))
            complement_entropy_error=maximum(abs.(entropies-complement_entropies))
            D=(2e.sa+e.sc)/3-e.sb
            reference_D=(2reference_entropies[1]+reference_entropies[2])/3-reference_entropies[3]
            complement_D=(2ec.sa+ec.sc)/3-ec.sb
            D_error=abs(D-reference_D); complement_D_error=abs(D-complement_D)
            @assert covariance_error<1e-7 && entropy_error<1e-7 && D_error<1e-7
            @assert complement_covariance_error<1e-9 && complement_entropy_error<1e-9 && complement_D_error<1e-9
            # Direct entropy evaluation independently checks the theta=0 shortcut.
            if theta==0
                sa0=FE.entropy_strip(state); sc0=FE.entropy_strip(state;orientation=:y)
                sb0=RealEntropyExtension.entropy_rectangle_real(state)
                @assert maximum(abs.([sa0,sc0,sb0]))<1e-10
                @assert maximum(abs.(reference_entropies))<1e-10
            end
            d=FE.diagnostics(state)
            @assert d.projector_error<1e-10 && d.reconstruction_error<1e-9
            isinf(theta) && @assert abs(d.excess_energy-(lim.energy-lim.ground_energy))<1e-8
            push!(rows,(;pattern=p.name,m=p.m,L,theta,beta=0.,occupied_bits=join(Int.(p.occupied)),
                ground_reachable=ground,independent_limit_excess=lim.energy-lim.ground_energy,
                independent_limit_residual=lim.residual,covariance_error,entropy_error,D_error,
                complement_covariance_error,complement_entropy_error,complement_D_error,
                S_A=e.sa,S_C=e.sc,S_B=e.sb,D,reference_D,
                reference_method=isinf(theta) ? "realspace_energy_block_nullspace" : "realspace_exp_qr",
                entropy_method=e.method))
            @printf("%s L=%d theta=%g C=%.3g S2=%.3g complement_C=%.3g\n",
                p.name,L,theta,covariance_error,entropy_error,complement_covariance_error)
            flush(stdout)
        end
        GC.gc(false)
    end
    output=joinpath(@__DIR__,"data","beta0_study_validation.csv")
    open(output,"w") do io
        println(io,join(string.(keys(first(rows))),','))
        for r in rows; println(io,join(values(r),',')); end
    end
    receipt=Dict{String,Any}("status"=>"PASS","generated_utc"=>string(now(Dates.UTC)),
        "states"=>length(rows),"patterns"=>length(beta0_study_patterns()),"code_sha256"=>hashes,
        "csv_sha256"=>bp_hash(output),"blas_threads"=>BLAS.get_num_threads(),"julia"=>string(VERSION),
        "scope"=>"All catalog patterns at L=3m, theta=0,1,4,Inf and three additional theta10 checks. Independent full real-space exp(-0.25H)+QR at finite time, independent energy-block/nullspace support filtration at Inf, full covariance and regional S2/D comparisons. Cached Bloch models are shared between different initial patterns. Theta0 shortcut is checked against direct regional entropy. Chiral complement identity C_complement=I-Gamma*C*Gamma is checked on the complete covariance and all three regional entropies, including blocked states and initial-selected zero modes. No large-size scaling assertion.")
    for field in (:covariance_error,:entropy_error,:D_error,:complement_covariance_error,
                  :complement_entropy_error,:complement_D_error)
        receipt["max_"*string(field)]=maximum(getproperty(r,field) for r in rows)
    end
    all(bp_hash(joinpath(@__DIR__,f))==hashes[f] for f in files) || error("code changed during validation")
    open(io->TOML.print(io,receipt),replace(output,".csv"=>".toml"),"w")
    println("PASS ",length(rows)," states; max C=",receipt["max_covariance_error"],
        " max S2=",receipt["max_entropy_error"]," complement C=",receipt["max_complement_covariance_error"])
end

abspath(PROGRAM_FILE)==abspath(@__FILE__) && beta0_study_validation_main()
