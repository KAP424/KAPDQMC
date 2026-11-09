# Independent small-system checks for the three occupation classes.
# This file writes only data/occupancy_classes_validation.{csv,toml}.
using LinearAlgebra, SHA, TOML, Dates, Printf
BLAS.set_num_threads(1)
include(joinpath(@__DIR__, "fermion_experiment.jl"))
import .FermionExperiment as FE
include(joinpath(@__DIR__, "real_entropy_extension.jl"))
include(joinpath(@__DIR__, "beta_patterns_strip_extension.jl"))
include(joinpath(@__DIR__, "occupancy_classes_patterns.jl"))

"""Build the real-space reference without Bloch or FE coordinate helpers."""
function validation_hamiltonian(L)
    site(x, y, s) = 2(mod(x, L) + L * mod(y, L)) + s
    h = zeros(2L^2, 2L^2)
    for y in 0:L-1, x in 0:L-1
        a = site(x, y, 1)
        for (bx, by) in ((x, y), (x+1, y), (x, y-1))
            b = site(bx, by, 2)
            h[a, b] += 1
            h[b, a] += 1
        end
    end
    return h
end

function validation_initial(p, L)
    F = Int[]
    for y in 0:L-1, x in 0:L-1, s in 1:2
        local_index = 2(mod(x, p.m) + p.m * mod(y, p.m)) + s
        p.occupied[local_index] && push!(F, 2(x + L*y) + s)
    end
    E = setdiff(1:2L^2, F)
    q = Matrix{Float64}(I, 2L^2, 2L^2)[:, F]
    return q, F, E
end

"""Exact dense exponential steps; QR only changes the orbital basis."""
function validation_dense_evolve(q, ek, steps)
    p = size(q, 2)
    for _ in 1:steps
        q = Matrix(qr(ek*q).Q)[:, 1:p]
    end
    return q
end

function validation_dense_measure(q, E, beta)
    raw = copy(q)
    raw[E, :] .*= exp(-2beta)
    return Matrix(qr(raw).Q)[:, 1:size(q, 2)]
end

function validation_entropy(c)
    vals = eigvals(Hermitian(c))
    minimum(vals) >= -1e-9 && maximum(vals) <= 1+1e-9 ||
        error("Reference covariance spectrum is unphysical")
    vals = clamp.(vals, 0., 1.)
    return -sum(log.(vals.^2 + (1 .- vals).^2))
end

function validation_main()
    files = ["fermion_experiment.jl", "real_entropy_extension.jl",
             "beta_patterns_strip_extension.jl", "occupancy_classes_patterns.jl",
             "occupancy_classes_experiments.jl", "validate_occupancy_classes.jl"]
    hashes = Dict(f => bytes2hex(sha256(read(joinpath(@__DIR__, f)))) for f in files)
    rows = NamedTuple[]
    patterns = occupancy_class_patterns()
    ecache = Dict{Int,Matrix{Float64}}()
    max_dense_covariance_error = 0.
    max_direct_vs_stepped_error = 0.
    class_checks = Any[]
    for p in patterns
        ns = [Int(p.occupied[2j-1]) + Int(p.occupied[2j]) for j in 1:p.m^2]
        n0, n1, n2 = count(==(0), ns), count(==(1), ns), count(==(2), ns)
        @assert count(p.occupied) == p.m^2 && n0 == n2
        @assert p.class_id == (n1 == p.m^2 ? 1 : n1 == 0 ? 2 : 3)
        push!(class_checks, Dict("pattern"=>p.name, "class_id"=>p.class_id,
            "m"=>p.m, "occupied_bits"=>join(Int.(p.occupied)),
            "n0"=>n0, "n1"=>n1, "n2"=>n2))
        sizes = p.m == 2 ? (6, 12) : p.m == 3 ? (9, 18) : (12,)
        for L in sizes
            pre = FE.prepare_pattern(FE.BlochModel(L, p.m), p.occupied; tol=1e-10)
            q0, F, E = validation_initial(p, L)
            @assert F == FE.occupied_indices(pre)
            ek = get!(ecache, L) do
                h = validation_hamiltonian(L)
                @assert h == FE.dense_hamiltonian(L)
                Matrix(exp(Symmetric(-0.25h)))
            end
            q4 = validation_dense_evolve(copy(q0), ek, 16)
            # A separately evaluated one-shot exponential is a useful control.
            # Only the stepped reference is used for the tight covariance test.
            h = validation_hamiltonian(L)
            q4_direct = Matrix(qr(exp(Symmetric(-4h))*q0).Q)[:, 1:L^2]
            direct_vs_step = norm(q4_direct*q4_direct' - q4*q4') / L
            max_direct_vs_stepped_error = max(max_direct_vs_stepped_error, direct_vs_step)
            @assert direct_vs_step < 1e-6
            for theta in (0., 4., Inf)
                state0 = FE.evolve(pre, theta)
                ground = FE.diagnostics(FE.evolve(pre, Inf))
                @assert abs(ground.excess_energy) < 1e-8 && ground.residual < 1e-9
                for beta in (0., 1., 2.)
                    state = FE.measure(state0, beta)
                    C = FE.correlation_matrix(state)
                    IA = FE.rectangle_indices(L; width=L÷3, height=L)
                    IC = FE.rectangle_indices(L; width=L, height=L÷3)
                    IB = FE.rectangle_indices(L)
                    sa_ref = validation_entropy(C[IA, IA])
                    sc_ref = validation_entropy(C[IC, IC])
                    sb_ref = validation_entropy(C[IB, IB])
                    sa_orig = FE.entropy_strip(state)
                    sc_orig = FE.entropy_strip(state; orientation=:y)
                    sa_fold, ca = BetaPatternsStripExtension.entropy_strip_folded(state; with_checks=true)
                    sc_fold, cc = BetaPatternsStripExtension.entropy_strip_folded(state; orientation=:y, with_checks=true)
                    sb_real, cr = RealEntropyExtension.entropy_rectangle_real(state; with_checks=true)
                    strip_error = max(abs(sa_fold-sa_orig), abs(sc_fold-sc_orig))
                    strip_dense_error = max(abs(sa_fold-sa_ref), abs(sc_fold-sc_ref))
                    rectangle_error = abs(sb_real-sb_ref)
                    @assert strip_error < 1e-9 && strip_dense_error < 1e-9 && rectangle_error < 1e-9
                    covariance_error, dense_entropy_error = NaN, NaN
                    if isfinite(theta)
                        q = validation_dense_measure(theta == 0 ? q0 : q4, E, beta)
                        denseC = q*q'
                        covariance_error = norm(C-denseC) / L
                        dense_entropy_error = maximum(abs.([
                            sa_ref-validation_entropy(denseC[IA, IA]),
                            sc_ref-validation_entropy(denseC[IC, IC]),
                            sb_ref-validation_entropy(denseC[IB, IB])]))
                        max_dense_covariance_error = max(max_dense_covariance_error, covariance_error)
                        @assert covariance_error < 1e-8 && dense_entropy_error < 1e-8
                    end
                    zero_error = theta == 0 ? maximum(abs.([sa_fold, sc_fold, sb_real])) : NaN
                    theta == 0 && @assert zero_error < 1e-9
                    @assert ca.total_weight == pre.model.q && cc.total_weight == pre.model.q
                    diag = FE.diagnostics(state)
                    @assert diag.projector_error < 1e-10
                    push!(rows, (; pattern=p.name, class_id=p.class_id, m=p.m, L, theta, beta,
                        S_A=sa_fold, S_C=sc_fold, S_B=sb_real, D=(2sa_fold+sc_fold)/3-sb_real,
                        strip_error, strip_dense_error, rectangle_error, covariance_error,
                        dense_entropy_error, zero_error, direct_vs_step,
                        max_imaginary=cr.max_imaginary,
                        max_bloch_conjugacy=max(ca.max_bloch_conjugacy, cc.max_bloch_conjugacy),
                        projector_error=diag.projector_error,
                        ground_excess=ground.excess_energy, ground_residual=ground.residual))
                end
            end
            @printf("validated %s L=%d; dense covariance max=%.3g\n", p.name, L, max_dense_covariance_error)
            flush(stdout)
        end
    end
    out = joinpath(@__DIR__, "data", "occupancy_classes_validation.csv")
    open(out, "w") do io
        println(io, join(string.(keys(first(rows))), ','))
        for row in rows
            println(io, join(values(row), ','))
        end
    end
    receipt = Dict{String,Any}(
        "status"=>"PASS", "generated_utc"=>string(now(UTC)), "states"=>length(rows),
        "patterns"=>class_checks, "BLAS_threads"=>BLAS.get_num_threads(),
        "blas_config"=>sprint(show, BLAS.get_config()), "julia"=>string(VERSION),
        "max_strip_original_error"=>maximum(r.strip_error for r in rows),
        "max_strip_dense_error"=>maximum(r.strip_dense_error for r in rows),
        "max_rectangle_eigen_error"=>maximum(r.rectangle_error for r in rows),
        "max_dense_covariance_error"=>max_dense_covariance_error,
        "max_dense_entropy_error"=>maximum(r.dense_entropy_error for r in rows if isfinite(r.theta)),
        "max_direct_vs_stepped_covariance_error"=>max_direct_vs_stepped_error,
        "max_theta0_entropy"=>maximum(r.zero_error for r in rows if r.theta == 0),
        "max_ground_excess"=>maximum(abs(r.ground_excess) for r in rows),
        "max_ground_residual"=>maximum(r.ground_residual for r in rows),
        "code_sha256"=>hashes, "csv_sha256"=>bytes2hex(sha256(read(out))),
        "scope"=>"Small-system correctness only; no scaling-fit claim. All six patterns. theta=0,4,Inf and beta=0,1,2. Independent real-space exact dense exponentials and QR validate finite-time covariance. Infinite theta is checked for ground energy/residual and entropy-backend agreement, not a separate dense limiting-state construction.")
    for f in files
        bytes2hex(sha256(read(joinpath(@__DIR__, f)))) == hashes[f] ||
            error("Code changed during validation: $f")
    end
    open(joinpath(@__DIR__, "data", "occupancy_classes_validation.toml"), "w") do io
        TOML.print(io, receipt)
    end
    println("PASS ", length(rows), " states; max dense covariance error ", max_dense_covariance_error)
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && validation_main()
