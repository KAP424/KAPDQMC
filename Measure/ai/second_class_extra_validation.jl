# Independent small-system reference for both reachable and blocked product states.
# The accessible infinite-time state is built by full real-space spectral
# support filtration, never by a long QR evolution that can leak to the ground.
include(joinpath(@__DIR__, "validate_occupancy_classes.jl"))
include(joinpath(@__DIR__, "second_class_extra_patterns.jl"))

"""Image of each energy block on the null space of every lower-energy block."""
function independent_supported_limit(h, q0; rank_tol=1e-10, energy_tol=1e-9)
    spectral = eigen(Symmetric(h))
    energies, vectors = spectral.values, spectral.vectors
    overlap = vectors' * q0
    n, particles = size(q0)
    remaining = Matrix{Float64}(I,particles,particles)
    orbitals = Matrix{Float64}(undef,n,particles)
    selected_energies = Float64[]
    rank_increments = Int[]
    max_discarded_sv, min_retained_sv = 0., Inf
    used, first = 0, 1
    while first <= n && used < particles
        last = first
        while last < n && abs(energies[last+1]-energies[first]) < energy_tol
            last += 1
        end
        block = overlap[first:last,:] * remaining
        decomposition = svd(block;full=true)
        added = count(>(rank_tol),decomposition.S)
        append!(rank_increments,added)
        for sigma in decomposition.S
            if sigma > rank_tol
                min_retained_sv = min(min_retained_sv,sigma)
            else
                max_discarded_sv = max(max_discarded_sv,sigma)
            end
        end
        if added > 0
            orbitals[:,used+1:used+added] = vectors[:,first:last]*decomposition.U[:,1:added]
            append!(selected_energies,fill(sum(energies[first:last])/(last-first+1),added))
            used += added
            # Remove exactly the newly-supported directions from the remaining
            # initial-orbital space. Full V also contains a rectangular nullspace.
            remaining = remaining*decomposition.V[:,added+1:end]
        end
        first = last+1
    end
    @assert used == particles && norm(orbitals'*orbitals-I) < 1e-10
    return orbitals,(; energy=sum(selected_energies),
        ground_energy=sum(energies[1:particles]),max_discarded_sv,min_retained_sv,
        residual=norm(h*orbitals-orbitals*(orbitals'*h*orbitals)))
end

function second_class_extra_validation_main()
    BLAS.set_num_threads(1)
    files = ["fermion_experiment.jl","real_entropy_extension.jl",
        "beta_patterns_strip_extension.jl","validate_occupancy_classes.jl",
        "occupancy_classes_patterns.jl","second_class_extra_patterns.jl",
        "second_class_extra_experiments.jl","second_class_extra_validation.jl"]
    hashes = Dict(f=>bytes2hex(sha256(read(joinpath(@__DIR__,f)))) for f in files)
    rows, summaries = NamedTuple[], Dict{String,Any}[]
    for p in second_class_extra_patterns()
        L = p.m == 4 ? 12 : 6
        h = validation_hamiltonian(L)
        q0,F,E = validation_initial(p,L)
        qinf,limit = independent_supported_limit(h,q0)
        ground = abs(limit.energy-limit.ground_energy) < 1e-8
        @assert ground == p.expected_ground
        @assert ground || limit.energy-limit.ground_energy > 1e-6
        @assert limit.residual < 1e-9
        ek = Matrix(exp(Symmetric(-.25h)))
        q4 = validation_dense_evolve(copy(q0),ek,16)
        q7 = validation_dense_evolve(copy(q4),ek,12)
        # A different time-step construction controls finite-time roundoff.
        ek_half = Matrix(exp(Symmetric(-.125h)))
        q7_half = validation_dense_evolve(copy(q0),ek_half,56)
        finite_step_difference = norm(q7*q7'-q7_half*q7_half')/L
        @assert finite_step_difference < 1e-7
        pre = FE.prepare_pattern(FE.BlochModel(L,p.m),p.occupied)
        @assert F == FE.occupied_indices(pre)
        @assert abs(pre.accessible_energy-limit.energy) < 1e-9
        region(w,hh) = [s+2*((x-1)+L*(y-1)) for y in 1:hh for x in 1:w for s in 1:2]
        IA,IC,IB = region(L÷3,L),region(L,L÷3),region(L÷3,2L÷3)
        for theta in (0.,4.,7.,Inf), beta in (0.,1.,2.)
            reference = theta == 0 ? q0 : theta == 4 ? q4 : theta == 7 ? q7 : qinf
            q = validation_dense_measure(reference,E,beta)
            reference_c = q*q'
            state = FE.evolve(pre,theta;beta=beta)
            c = FE.correlation_matrix(state)
            covariance_error = norm(c-reference_c)/L
            covariance_element_error = maximum(abs,c-reference_c)
            reference_entropies = [validation_entropy(reference_c[ix,ix]) for ix in (IA,IC,IB)]
            sa = BetaPatternsStripExtension.entropy_strip_folded(state)
            sc = BetaPatternsStripExtension.entropy_strip_folded(state;orientation=:y)
            sb = RealEntropyExtension.entropy_rectangle_real(state)
            D = (2sa+sc)/3-sb
            reference_D = (2reference_entropies[1]+reference_entropies[2])/3-reference_entropies[3]
            entropy_error = maximum(abs.([sa,sc,sb]-reference_entropies))
            D_error = abs(D-reference_D)
            @assert covariance_error < 1e-8 && entropy_error < 1e-8 && D_error < 1e-8
            theta == 0 && @assert maximum(abs.([sa,sc,sb])) < 1e-10
            push!(rows,(;pattern=p.name,m=p.m,L,theta,beta,ground_reachable=ground,
                limit_state=ground ? "ground_state" : "accessible_excited_state",
                reference_method=isinf(theta) ? "realspace_energy_block_nullspace" : "realspace_exp_qr",
                S_A=sa,S_C=sc,S_B=sb,D,reference_S_A=reference_entropies[1],
                reference_S_C=reference_entropies[2],reference_S_B=reference_entropies[3],reference_D,
                covariance_error,covariance_element_error,entropy_error,D_error,
                independent_limit_energy=limit.energy,independent_ground_energy=limit.ground_energy,
                independent_limit_excess=limit.energy-limit.ground_energy,
                independent_limit_residual=limit.residual,finite_step_difference,
                max_discarded_sv=limit.max_discarded_sv,min_retained_sv=limit.min_retained_sv))
            @printf("%s L=%d theta=%g beta=%g ground=%s C=%.3g S2=%.3g D=%.3g\n",
                p.name,L,theta,beta,ground,covariance_error,entropy_error,D_error)
            flush(stdout)
        end
        push!(summaries,Dict("pattern"=>p.name,"m"=>p.m,"L"=>L,
            "occupied_bits"=>join(Int.(p.occupied)),"ground_reachable"=>ground,
            "limit_state"=>ground ? "ground_state" : "accessible_excited_state",
            "independent_limit_excess"=>limit.energy-limit.ground_energy,
            "independent_limit_residual"=>limit.residual,
            "min_retained_sv"=>limit.min_retained_sv,"max_discarded_sv"=>limit.max_discarded_sv,
            "finite_step_difference"=>finite_step_difference))
        GC.gc()
    end
    out=joinpath(@__DIR__,"data","second_class_extra_validation.csv")
    open(out,"w") do io
        println(io,join(string.(keys(first(rows))),','))
        for r in rows
            println(io,join(values(r),','))
        end
    end
    receipt=Dict{String,Any}("status"=>"PASS","generated_utc"=>string(now(UTC)),
        "states"=>length(rows),"patterns"=>summaries,"code_sha256"=>hashes,
        "csv_sha256"=>bytes2hex(sha256(read(out))),"julia"=>string(VERSION),
        "BLAS_threads"=>BLAS.get_num_threads(),
        "max_covariance_error"=>maximum(r.covariance_error for r in rows),
        "max_covariance_element_error"=>maximum(r.covariance_element_error for r in rows),
        "max_entropy_error"=>maximum(r.entropy_error for r in rows),
        "max_D_error"=>maximum(r.D_error for r in rows),
        "max_finite_step_difference"=>maximum(r.finite_step_difference for r in rows),
        "scope"=>"Four pure-0/2 patterns, including two blocked alternating controls; m4 L12 and m2 L6; theta0,4,7,Inf and beta0,1,2. Finite-time reference is full real-space exact exp(-0.25H) plus QR, cross-checked at step0.125. Infinite-time reference is a separate real-space energy-block/nullspace spectral-support construction, not long QR evolution. Full C, three S2 and weighted D are compared; blocked accessible limits retain positive excitation energy and are never labeled ground states. No scaling claim.")
    for f in files
        @assert bytes2hex(sha256(read(joinpath(@__DIR__,f)))) == hashes[f]
    end
    open(joinpath(@__DIR__,"data","second_class_extra_validation.toml"),"w") do io
        TOML.print(io,receipt)
    end
    println("PASS states=",length(rows)," max_C=",receipt["max_covariance_error"],
        " max_S2=",receipt["max_entropy_error"]," max_D=",receipt["max_D_error"])
end

abspath(PROGRAM_FILE)==abspath(@__FILE__) && second_class_extra_validation_main()
