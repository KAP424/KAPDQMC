# CI entry point for KAPDQMC.
#
# `Pkg.test()` — and therefore `julia-actions/julia-runtest@v1` — requires
# exactly this file. Without it Pkg aborts with
#   "Package KAPDQMC did not provide a `test/runtests.jl` file"
#
# The heavy Monte-Carlo driver scripts (test/tU, test/tV, test/tUV, test/SO3,
# test/VBS, test/M_VBS, test/Free_WaveFunction) are long-running and use
# absolute Windows paths, so they are intentionally NOT wired into CI here.
# Once a script is CI-friendly (relative paths via @__DIR__, short sweeps),
# add it with `include("tU/xxx_runtests.jl")` inside the outer testset.

using Test
using LinearAlgebra
using Random

using KAPDQMC

@testset "KAPDQMC" begin

    @testset "package loads and exports its API" begin
        for sym in (:tU_Hubbard_Para, :tV_Hubbard_Para, :tUV_Hubbard_Para,
                    :SO3_Hubbard_Para, :VBS_Hubbard_Para, :M_VBS_Hubbard_Para)
            @test isdefined(KAPDQMC, sym)
        end
        # unified API stubs extended by the submodules via multiple dispatch
        for f in (:phy_update, :Initial_s, :ctrl_SCEEicr, :ctrl_EEicr,
                  :ctrl_SCDOPicr, :phy_update_M, :ctrl_SCEEicr_M)
            @test isdefined(KAPDQMC, f)
        end
    end

    @testset "lattice names" begin
        @test name_Lattice("SQUARE90") == "□90"
        @test name_Lattice("SQUARE45") == "□45"
        @test name_Lattice("HoneyComb120") == "HC120"
        @test name_Lattice("HoneyComb60") == "HC60"
        @test name_Lattice("triangular90") == "△90"
        @test name_Lattice("triangular45") == "△45"
        @test_throws ErrorException name_Lattice("not-a-lattice")
    end

    @testset "index <-> coordinate roundtrip" begin
        site = [4, 4]
        # two-sublattice ("even index") lattices: xy_i returns an even index,
        # i_xy maps both 2k-1 and 2k back to the same (x, y).
        for lattice in ("SQUARE90", "HoneyComb120")
            for x in 1:site[1], y in 1:site[2]
                i = xy_i(lattice, site, x, y)
                @test iseven(i)
                @test i == 2 * (x + (y - 1) * site[1])
                @test i_xy(lattice, site, i) == (x, y)
                @test i_xy(lattice, site, i - 1) == (x, y)
            end
        end
        # single-sublattice lattices: plain row-major index
        for lattice in ("SQUARE45",)
            for x in 1:site[1], y in 1:site[2]
                i = xy_i(lattice, site, x, y)
                @test i == x + (y - 1) * site[1]
                @test i_xy(lattice, site, i) == (x, y)
            end
        end
        @test_throws ErrorException xy_i("not-a-lattice", site, 1, 1)
        @test_throws ErrorException i_xy("not-a-lattice", site, 1)
    end

    @testset "nearest-neighbour table" begin
        # nnidx_F only lists A-sublattice sites: an 8x4 matrix of
        # (site, neighbour) tuples on a periodic 4x4 square lattice.
        site = [4, 4]
        Ns = prod(site)
        nnidx = nnidx_F("SQUARE45", site)
        @test size(nnidx) == (div(Ns, 2), 4)

        a_sites = Int[]
        for row in 1:size(nnidx, 1)
            i = nnidx[row, 1][1]
            x, y = i_xy("SQUARE45", site, i)
            @test i == xy_i("SQUARE45", site, x, y)   # valid lattice index
            @test (x + y) % 2 == 1                    # A sublattice
            push!(a_sites, i)

            nbrs = [nnidx[row, j][2] for j in 1:4]
            @test all(n -> 1 <= n <= Ns, nbrs)
            @test length(unique(nbrs)) == 4          # no self / no duplicate
            @test all(n -> n != i, nbrs)
            # the neighbour relation must be symmetric
            for n in nbrs
                @test i in KAPDQMC.nn2idx("SQUARE45", site, n)
            end
        end
        # every A-sublattice site appears exactly once
        @test length(unique(a_sites)) == div(Ns, 2)
    end

    @testset "RNG is reproducible" begin
        @test rand(MersenneTwister(2024), 8) == rand(MersenneTwister(2024), 8)
    end

    @testset "linear algebra backend" begin
        @test BLAS.get_num_threads() >= 1
        @test isapprox(det([1.0 2.0; 3.0 4.0]), -2.0; atol = 1e-12)
    end

end
