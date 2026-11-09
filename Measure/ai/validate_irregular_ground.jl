# Dedicated independent real-space ground-projection audit of the two m=4
# patterns. Imported helpers guard their main entry point; old files are not
# regenerated. This script writes only data/irregular_ground_validation.*.
include(joinpath(@__DIR__, "validate_occupancy_classes.jl"))

function validate_irregular_ground()
    BLAS.set_num_threads(1)
    L, step = 12, 0.25
    names = ("pure02_block4", "pure02_irregular4")
    patterns = [only(filter(p -> p.name == name, occupancy_class_patterns())) for name in names]
    files = ["fermion_experiment.jl", "real_entropy_extension.jl",
        "beta_patterns_strip_extension.jl", "occupancy_classes_patterns.jl",
        "validate_occupancy_classes.jl", "validate_irregular_ground.jl"]
    hashes = Dict(f => bytes2hex(sha256(read(joinpath(@__DIR__, f)))) for f in files)
    h = validation_hamiltonian(L)
    eig = eigen(Symmetric(h))
    ground_energy = sum(eig.values[1:L^2])
    zerospace = eig.vectors[:, abs.(eig.values) .< 1e-10]
    @assert size(zerospace, 2) == 4
    ek = Matrix(exp(Symmetric(-step*h)))
    # Construct physical-region indices independently from the Bloch backend.
    region(w, hh) = [s+2*((x-1)+L*(y-1)) for y in 1:hh for x in 1:w for s in 1:2]
    IA, IC, IB = region(L÷3,L), region(L,L÷3), region(L÷3,2L÷3)
    rows = NamedTuple[]
    checks = Dict{String,Any}[]
    failures = String[]
    for p in patterns
        started = time()
        q0, F, E = validation_initial(p,L)
        q64 = validation_dense_evolve(copy(q0),ek,256)
        q128 = validation_dense_evolve(copy(q64),ek,256)
        c64, c128 = q64*q64', q128*q128'
        difference64_128 = norm(c128-c64)/L
        difference128_256 = NaN
        qlimit, reference_theta = q128, 128.
        convergence = difference64_128
        if difference64_128 > 1e-10
            q256 = validation_dense_evolve(copy(q128),ek,512)
            difference128_256 = norm(q256*q256'-c128)/L
            qlimit, reference_theta, convergence = q256, 256., difference128_256
        end
        dense_c0 = qlimit*qlimit'
        dense_ground_excess = tr(h*dense_c0)-ground_energy
        dense_residual = norm(h*qlimit-qlimit*(qlimit'*h*qlimit))
        convergence <= 1e-10 || push!(failures,"$(p.name): independent projection has not converged")
        abs(dense_ground_excess) <= 1e-9 || push!(failures,"$(p.name): dense energy above ground")
        dense_residual <= 1e-9 || push!(failures,"$(p.name): dense ground residual too large")
        pre = FE.prepare_pattern(FE.BlochModel(L,p.m),p.occupied;tol=1e-10)
        @assert F == FE.occupied_indices(pre)
        state0 = FE.evolve(pre,Inf)
        fe_c0 = FE.correlation_matrix(state0)
        zero_sector_error = norm(zerospace'*(fe_c0-dense_c0)*zerospace)
        zero_sector_particles = tr(zerospace'*dense_c0*zerospace)
        zero_sector_error <= 1e-9 || push!(failures,"$(p.name): initial-state-selected zero sector differs")
        for beta in (0.,1.,2.)
            q = validation_dense_measure(qlimit,E,beta)
            dense_c = q*q'
            state = FE.measure(state0,beta)
            c = FE.correlation_matrix(state)
            covariance_error = norm(c-dense_c)/L
            covariance_max_element = maximum(abs,c-dense_c)
            dense_entropies = [validation_entropy(dense_c[ix,ix]) for ix in (IA,IC,IB)]
            sa = BetaPatternsStripExtension.entropy_strip_folded(state)
            sc = BetaPatternsStripExtension.entropy_strip_folded(state;orientation=:y)
            sb = RealEntropyExtension.entropy_rectangle_real(state)
            entropy_error = maximum(abs.([sa,sc,sb]-dense_entropies))
            D = (2sa+sc)/3-sb
            dense_D = (2dense_entropies[1]+dense_entropies[2])/3-dense_entropies[3]
            D_error = abs(D-dense_D)
            covariance_error <= 1e-9 || push!(failures,"$(p.name), beta=$beta: covariance mismatch")
            entropy_error <= 1e-9 || push!(failures,"$(p.name), beta=$beta: regional entropy mismatch")
            D_error <= 1e-9 || push!(failures,"$(p.name), beta=$beta: D mismatch")
            push!(rows,(;pattern=p.name,m=p.m,L,theta=Inf,beta,reference_theta,
                S_A=sa,S_C=sc,S_B=sb,D,
                dense_S_A=dense_entropies[1],dense_S_C=dense_entropies[2],
                dense_S_B=dense_entropies[3],dense_D,
                covariance_error,covariance_max_element,entropy_error,D_error,
                dense_theta64_128_projector_difference=difference64_128,
                dense_theta128_256_projector_difference=difference128_256,
                dense_reference_convergence=convergence,dense_ground_excess,dense_residual,
                premeasurement_zero_sector_error=zero_sector_error,
                premeasurement_zero_sector_particles=zero_sector_particles))
            @printf("%s L=%d beta=%g covariance=%.3g entropy=%.3g D=%.3g zero=%.3g\n",
                p.name,L,beta,covariance_error,entropy_error,D_error,zero_sector_error)
            flush(stdout)
        end
        push!(checks,Dict("pattern"=>p.name,"occupied_bits"=>join(Int.(p.occupied)),
            "reference_theta"=>reference_theta,"theta64_128_convergence"=>difference64_128,
            "theta128_256_convergence"=>difference128_256,"reference_convergence"=>convergence,
            "dense_ground_excess"=>dense_ground_excess,"dense_ground_residual"=>dense_residual,
            "premeasurement_zero_sector_error"=>zero_sector_error,
            "premeasurement_zero_sector_particles"=>zero_sector_particles,"seconds"=>time()-started))
    end
    out = joinpath(@__DIR__,"data","irregular_ground_validation.csv")
    open(out,"w") do io
        println(io,join(string.(keys(first(rows))),','))
        for row in rows
            println(io,join(values(row),','))
        end
    end
    for f in files
        hashes[f] == bytes2hex(sha256(read(joinpath(@__DIR__,f)))) ||
            push!(failures,"Source changed during audit: $f")
    end
    receipt = Dict{String,Any}("status"=>isempty(failures) ? "PASS" : "FAIL",
        "generated_utc"=>string(now(UTC)),"L"=>L,"step"=>step,
        "zero_mode_dimension"=>size(zerospace,2),"states"=>length(rows),
        "BLAS_threads"=>BLAS.get_num_threads(),"blas_config"=>sprint(show,BLAS.get_config()),
        "julia"=>string(VERSION),"code_sha256"=>hashes,
        "csv_sha256"=>bytes2hex(sha256(read(out))),"patterns"=>checks,"failures"=>failures,
        "max_covariance_error"=>maximum(r.covariance_error for r in rows),
        "max_covariance_element_error"=>maximum(r.covariance_max_element for r in rows),
        "max_entropy_error"=>maximum(r.entropy_error for r in rows),
        "max_D_error"=>maximum(r.D_error for r in rows),
        "max_zero_sector_error"=>maximum(r.premeasurement_zero_sector_error for r in rows),
        "scope"=>"Independent real-space Hamiltonian and product orbitals evolved by exp(-0.25H) plus QR to theta64/128 (256 only if needed). Convergence, energy and residual are checked before measurement. The full covariance and the four-dimensional zero-energy sector are compared to FE theta=Inf, validating the initial-state-selected zero modes, followed by beta0,1,2 filtering and three regional entropy and D comparisons. L12 only; this checks implementation, not asymptotic scaling.")
    open(joinpath(@__DIR__,"data","irregular_ground_validation.toml"),"w") do io
        TOML.print(io,receipt)
    end
    println(receipt["status"]," states=",length(rows)," covariance=",receipt["max_covariance_error"],
        " entropy=",receipt["max_entropy_error"]," zero-sector=",receipt["max_zero_sector_error"])
    isempty(failures) || error(join(failures,"; "))
end

abspath(PROGRAM_FILE)==abspath(@__FILE__) && validate_irregular_ground()
