# Reuse only helper definitions: the imported file guards its main entry point.
# This script never overwrites the six-pattern validation or observation files.
include(joinpath(@__DIR__, "validate_occupancy_classes.jl"))
include(joinpath(@__DIR__, "irregular_m3_patterns.jl"))

function validate_irregular_m3()
    L = 9
    p = only(irregular_m3_patterns())
    files = ["fermion_experiment.jl","real_entropy_extension.jl",
        "beta_patterns_strip_extension.jl","validate_occupancy_classes.jl",
        "irregular_m3_patterns.jl","validate_irregular_m3.jl"]
    hashes = Dict(f => bytes2hex(sha256(read(joinpath(@__DIR__,f)))) for f in files)
    pre = FE.prepare_pattern(FE.BlochModel(L,p.m),p.occupied)
    h = validation_hamiltonian(L)
    q0,F,E = validation_initial(p,L)
    @assert F == FE.occupied_indices(pre)
    @assert h == FE.dense_hamiltonian(L)
    ek = Matrix(exp(Symmetric(-0.25h)))
    q4 = validation_dense_evolve(copy(q0),ek,16)
    q64 = validation_dense_evolve(copy(q0),ek,256)
    q128 = validation_dense_evolve(copy(q64),ek,256)
    convergence = norm(q128*q128'-q64*q64')/L
    dense_energy = tr(h*(q128*q128'))
    ground_energy = sum(eigvals(Symmetric(h))[1:L^2])
    dense_ground_excess = dense_energy-ground_energy
    @assert convergence < 1e-10 && abs(dense_ground_excess) < 1e-9
    q4_direct = Matrix(qr(exp(Symmetric(-4h))*q0).Q)[:,1:L^2]
    direct_step_error = norm(q4*q4'-q4_direct*q4_direct')/L
    @assert direct_step_error < 1e-6
    # Construct these indices directly, without the FE regional indexing helper.
    region(w,hh) = [s+2*((x-1)+L*(y-1)) for y in 1:hh for x in 1:w for s in 1:2]
    IA,IC,IB = region(L÷3,L),region(L,L÷3),region(L÷3,2L÷3)
    rows = NamedTuple[]
    for theta in (0.,4.,Inf), beta in (0.,1.,2.)
        q = validation_dense_measure(theta == 0 ? q0 : theta == 4 ? q4 : q128,E,beta)
        dense_c = q*q'
        state = FE.evolve(pre,theta;beta=beta)
        c = FE.correlation_matrix(state)
        covariance_error = norm(c-dense_c)/L
        # This includes the full zero-mode sector: equal energy alone is insufficient.
        dense_entropies = [validation_entropy(dense_c[ix,ix]) for ix in (IA,IC,IB)]
        sa,ca = BetaPatternsStripExtension.entropy_strip_folded(state;with_checks=true)
        sc,cc = BetaPatternsStripExtension.entropy_strip_folded(state;orientation=:y,with_checks=true)
        sb,cr = RealEntropyExtension.entropy_rectangle_real(state;with_checks=true)
        entropy_error = maximum(abs.([sa,sc,sb]-dense_entropies))
        theta == 0 && @assert maximum(abs.([sa,sc,sb])) < 1e-10
        @assert covariance_error < 1e-9 && entropy_error < 1e-9
        diag = FE.diagnostics(state)
        @assert diag.projector_error < 1e-10
        push!(rows,(;pattern=p.name,m=p.m,L,theta,beta,S_A=sa,S_C=sc,S_B=sb,
            D=(2sa+sc)/3-sb,covariance_error,entropy_error,
            dense_theta64_128_projector_difference=convergence,dense_ground_excess,
            direct_step_error,projector_error=diag.projector_error))
        @printf("L=%d theta=%g beta=%g covariance=%.3g entropy=%.3g\n",
            L,theta,beta,covariance_error,entropy_error)
        flush(stdout)
    end
    out = joinpath(@__DIR__,"data","irregular_m3_validation.csv")
    open(out,"w") do io
        println(io,join(string.(keys(first(rows))),','))
        for r in rows
            println(io,join(values(r),','))
        end
    end
    receipt = Dict{String,Any}("status"=>"PASS","generated_utc"=>string(now(UTC)),
        "pattern"=>p.name,"occupied_bits"=>join(Int.(p.occupied)),"L"=>L,
        "states"=>length(rows),"BLAS_threads"=>BLAS.get_num_threads(),
        "julia"=>string(VERSION),"code_sha256"=>hashes,
        "csv_sha256"=>bytes2hex(sha256(read(out))),
        "max_covariance_error"=>maximum(r.covariance_error for r in rows),
        "max_entropy_error"=>maximum(r.entropy_error for r in rows),
        "dense_theta64_128_projector_difference"=>convergence,
        "dense_ground_excess"=>dense_ground_excess,
        "scope"=>"Independent real-space exp(-0.25H) plus QR validates full projectors and all three regional entropies at L=9, theta=0,4,Inf and beta=0,1,2. The infinite-time reference is theta=128, separately compared to theta=64 and to ground energy before measurement. Full covariance comparison checks initial-state-selected zero modes. Small-system correctness only, not a scaling claim.")
    for f in files
        @assert bytes2hex(sha256(read(joinpath(@__DIR__,f)))) == hashes[f]
    end
    open(joinpath(@__DIR__,"data","irregular_m3_validation.toml"),"w") do io
        TOML.print(io,receipt)
    end
    println("PASS states=",length(rows)," ground convergence=",convergence)
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && validate_irregular_m3()
