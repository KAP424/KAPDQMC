using Test, LinearAlgebra, Random

include(joinpath(@__DIR__, "../../src/Measure/Force2/NeqTgrd.jl"))
include(joinpath(@__DIR__, "../../src/public/Geometry.jl"))

function neq_entropy2(Q, region)
    values = eigvals(Hermitian(Q[region, :] * Q[region, :]'))
    return -sum(x -> log(x^2 + (1 - x)^2), values)
end

@testset "Initial-state imaginary-time limit" begin
    rng = MersenneTwister(424)
    @testset "Real and complex ground states" for T in (Float64, ComplexF64)
        V = Matrix(qr(randn(rng, T, 8, 8)).Q)
        E = [-3., -2., -1., 0.5, 1., 2., 3., 4.]
        eK = V * Diagonal(exp.(-0.2 .* E)) * V'
        P0 = randn(rng, T, 8, 3)
        target = V[:, 1:3] * V[:, 1:3]'
        for spectrum in (nothing, (E, V))
            Pt = copy(P0)
            @test NeqTgrd!(Pt, eK; spectrum) === Pt
            @test norm(Pt' * Pt - I) < 1e-12
            @test norm(Pt * Pt' - target) < 1e-11
            @test minimum(svdvals(Matrix(qr(P0).Q)' * Pt)) > 1e-5
        end
    end

    @testset "Forbidden many-body components and degenerate energies" begin
        # Every energy has nonzero initial weight, but the first two rows are
        # dependent: the determinant occupying both lower levels is forbidden.
        E = [-2., -1., 0.]
        V = Matrix{Float64}(I, 3, 3)
        eK = Diagonal(exp.(-E))
        P0 = [1. 0.; 1. 0.; 0. 1.]
        Pt = NeqTgrd!(copy(P0), eK)
        @test norm(Pt * Pt' - Diagonal([1., 0., 1.])) < 1e-12
        @test tr(Pt' * Diagonal(E) * Pt) ≈ -2.0
        @test minimum(svdvals(Matrix(qr(P0).Q)' * Pt)) > 0.7
        # Independent reference: actual finite-time evolution at 256-bit precision.
        reference = setprecision(256) do
            Matrix(qr(Diagonal(exp.(-BigFloat(30) .* E)) * BigFloat.(P0)).Q)
        end
        @test norm(Pt * Pt' - Float64.(reference * reference')) < 1e-11

        E = [-2., -1., -1.]
        eK = Diagonal(exp.(-E))
        P0 = [1. 1.; 1. 0.; 0. 1.]
        target = [1. 0.; 0. inv(sqrt(2)); 0. -inv(sqrt(2))]
        Pt = NeqTgrd!(copy(P0), eK; spectrum=(E, V))
        @test norm(Pt * Pt' - target * target') < 1e-12
        # Both orbital gauge changes and rotations inside degenerate eigenspaces
        # must leave the physical result unchanged.
        Vrot = copy(V)
        Vrot[:, 2:3] = V[:, 2:3] * Matrix(qr(randn(rng, 2, 2)).Q)
        Pt_rot = NeqTgrd!(P0 * [2. 1.; 0. 3.], eK; spectrum=(E, Vrot))
        @test norm(Pt_rot * Pt_rot' - Pt * Pt') < 1e-12
        Pt_identity = NeqTgrd!(copy(P0), Matrix{Float64}(I, 3, 3))
        Q0 = Matrix(qr(P0).Q)
        @test norm(Pt_identity * Pt_identity' - Q0 * Q0') < 1e-12

        # Do not discard a resolvable, genuinely small ground-state component.
        Pt_small = reshape([1e-6, 1.], 2, 1)
        NeqTgrd!(Pt_small, Diagonal([2., 1.]))
        @test abs(Pt_small[1]) ≈ 1.0
    end

    @testset "Honeycomb configuration returns to a product state" for L in (6, 12, 18, 24)
        K = nnK_Matrix("HoneyComb120", [L, L])
        E, V = LAPACK.syevd!('V', 'L', copy(K))
        eK = V * Diagonal(exp.(-3 .* E)) * V'
        Ns = size(K, 1)
        ns = Ns ÷ 2
        # Matches Config_set([L,L], [2,2], [1 1; -1 -1]) and its complement.
        F_idx = [2 * (x + (y - 1) * L) - Int(isodd(x)) for y in 1:L for x in 1:L]
        E_idx = setdiff(1:Ns, F_idx)
        P0 = Matrix{Float64}(I, Ns, Ns)[:, F_idx]
        A = area_index("HoneyComb120", [L, L], ([1, 1], [L ÷ 3, L]))
        B = area_index("HoneyComb120", [L, L], ([1, 1], [L ÷ 3, 2L ÷ 3]))
        @test abs(neq_entropy2(P0, A)) < 1e-12
        Pt = NeqTgrd!(copy(P0), eK; spectrum=(E, V))
        from_propagator = NeqTgrd!(copy(P0), eK)
        @test norm(from_propagator * from_propagator' - Pt * Pt') < 1e-7
        @test minimum(svdvals(from_propagator[F_idx, :])) > 1e-3
        @test norm(Pt' * Pt - I) < 1e-11
        @test norm(K * Pt - Pt * (Pt' * K * Pt)) < 1e-10
        @test minimum(svdvals(Pt[F_idx, :])) > 1e-3
        @test neq_entropy2(Pt, A) > 0.1
        excess = tr(Pt' * K * Pt) - sum(E[1:ns])
        @test excess >= L - 1e-9
        if L <= 12
            @test excess ≈ L atol=1e-9
        end

        # Exact integer eigenvectors: these negative-energy directions have no
        # initial overlap and must remain empty, even in the infinite-time limit.
        W = zeros(Int, Ns, L ÷ 2)
        for (j, x) in enumerate(2:2:L), y in 1:L
            W[2 * (x + (y - 1) * L) - 1, j] = (-1)^y
            W[2 * (mod1(x + 1, L) + (y - 1) * L), j] = -(-1)^y
        end
        @test Int.(K) * W == -W
        @test all(iszero, W[F_idx, :])
        @test norm(W' * Pt) < 1e-10

        n = ones(Ns)
        n[E_idx] .= exp(-2 * 8.0)
        measured = Matrix(qr(n .* Pt).Q)
        @test sum(abs2, measured[E_idx, :]) < 1e-8
        @test abs(neq_entropy2(measured, A)) < 1e-8
        @test abs(neq_entropy2(measured, B)) < 1e-8
    end

    @testset "Invalid inputs do not invent orbitals" begin
        Pt = reshape([1., 1.], 2, 1)
        @test_throws DimensionMismatch NeqTgrd!(copy(Pt), ones(3, 3))
        @test_throws ArgumentError NeqTgrd!(zeros(2, 0), Matrix{Float64}(I, 2, 2))
        @test_throws ArgumentError NeqTgrd!(zeros(2, 1), Matrix{Float64}(I, 2, 2))
        @test_throws ArgumentError NeqTgrd!(ones(2, 2), Matrix{Float64}(I, 2, 2))
        @test_throws ArgumentError NeqTgrd!(copy(Pt), [1. 1.; 0. 1.])
        @test_throws ArgumentError NeqTgrd!(copy(Pt), Diagonal([1., -1.]))
        @test_throws ArgumentError NeqTgrd!(copy(Pt), Diagonal([1., NaN]))
        @test_throws ArgumentError NeqTgrd!(copy(Pt), Diagonal([2., 1.]); tol=0)
        @test_throws ArgumentError NeqTgrd!(copy(Pt), Diagonal([2., 1.]); degeneracy_tol=-1)
        @test_throws DimensionMismatch NeqTgrd!(copy(Pt), Diagonal([2., 1.]); spectrum=([0.], ones(1, 1)))
    end
end
