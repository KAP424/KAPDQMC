#=
Export the half-filled HoneyComb120 state used by test/test.jl.

From the repository root:
    julia --project=. test/Free_WaveFunction/export_green.jl [Lx=12] [Ly=Lx] [output_dir]

G_occupied[i,j] = <c_j^dagger c_i>; the DQMC hole Green matrix is I-G_occupied.
The reference state calls Initial_Pt!("H0") exactly, including its staggered
potential. A second, explicitly Gaussian zero-mode ensemble of the massless
Hamiltonian is exported for comparison; it need not have fixed particle number.
=#
using KAPDQMC
using LinearAlgebra
using DelimitedFiles
using TOML

function export_green(Lx::Int=12, Ly::Int=Lx,
                      output_dir=joinpath(@__DIR__, "data", "L$(Lx)x$(Ly)"))
    Lx >= 3 && Ly >= 3 || error("Use Lx,Ly >= 3 to keep three distinct neighbours.")
    lattice = "HoneyComb120"
    sites = [Lx, Ly]
    K = nnK_Matrix(lattice, sites)
    Ns = size(K, 1)
    Np = Ns ÷ 2
    Pt = zeros(ComplexF64, Ns, Np)
    Initial_Pt!(lattice, "H0", Pt, K)
    # Same occupied-space projector as test/test.jl, using a solve instead of inv.
    G = Pt * ((Pt' * Pt) \ Pt')

    # Independent diagonalization verifies the regulator currently used in GF.jl.
    mass = 1e-5
    H_reference = K + Diagonal(mass .* repeat([-1, 1], Np))
    er, vr = eigen(Hermitian(H_reference))
    @assert norm(G - vr[:, 1:Np] * vr[:, 1:Np]') < 1e-8
    @assert norm(G - G') < 1e-10
    @assert norm(G * G - G) < 1e-10
    @assert abs(real(tr(G)) - Np) < 1e-10

    e0, v0 = eigen(Hermitian(K))
    zero_tol = 1e-10 * max(1.0, opnorm(K, Inf))
    occupations = [e < -zero_tol ? 1.0 : e > zero_tol ? 0.0 : 0.5 for e in e0]
    G_symmetric = v0 * Diagonal(occupations) * v0'
    @assert abs(real(tr(G_symmetric)) - Np) < 1e-10
    @assert maximum(abs.(diag(G_symmetric) .- 0.5)) < 1e-10

    mkpath(output_dir)
    for (name, matrix) in (("G_occupied", G), ("G_symmetric", G_symmetric))
        writedlm(joinpath(output_dir, name * "_real.csv"), real.(matrix), ',')
        writedlm(joinpath(output_dir, name * "_imag.csv"), imag.(matrix), ',')
    end
    writedlm(joinpath(output_dir, "K.csv"), K, ',')
    open(joinpath(output_dir, "spectrum.csv"), "w") do io
        println(io, "level,energy_massless,energy_reference,occupation_massless")
        writedlm(io, hcat(1:Ns, e0, er, occupations), ',')
    end
    # Index convention: A=2*(x+(y-1)*Lx)-1, B=A+1 (Julia, one based).
    a1 = [sqrt(3.0), 0.0]
    a2 = [-sqrt(3.0)/2, 1.5]
    delta = [-sqrt(3.0)/2, 0.5]
    open(joinpath(output_dir, "sites.csv"), "w") do io
        println(io, "site,cell,x,y,sublattice,rx,ry")
        for i in 1:Ns
            x, y = i_xy(lattice, sites, i)
            sub = isodd(i) ? "A" : "B"
            r = (x-1)*a1 + (y-1)*a2 + (isodd(i) ? -0.5 : 0.5)*delta
            println(io, join((i, (i+1)÷2, x, y, sub, r...), ','))
        end
    end
    metadata = Dict(
        "schema_version" => 1, "lattice" => lattice, "Lx" => Lx, "Ly" => Ly,
        "Ns" => Ns, "Np" => Np, "filling" => 0.5, "boundary" => "PBC",
        "hopping" => [1.0, 1.0, 1.0], "hamiltonian" => "H = sum_ij c_i^dagger K_ij c_j",
        "reference_mass" => mass, "reference_potential_AB" => [-mass, mass],
        "green_convention" => "G_occupied[i,j] = <c_j^dagger c_i>; G_hole = I-G_occupied",
        "reference_state" => "Initial_Pt!(HoneyComb120, H0, Pt, K), pure Slater determinant",
        "symmetric_state" => "massless Gaussian ensemble; independent occupation 1/2 of every zero mode",
        "zero_mode_count" => count(abs.(e0) .<= zero_tol), "zero_tolerance" => zero_tol,
        "a1" => a1, "a2" => a2, "delta_AB" => delta, "bond_length" => 1.0,
        "site_order" => "A,B interleaved; x fastest; all exported indices one based",
        "julia_version" => string(VERSION), "source" => "test/test.jl; src/public/Geometry.jl; src/public/GF.jl",
    )
    open(joinpath(output_dir, "metadata.toml"), "w") do io
        TOML.print(io, metadata; sorted=true)
    end
    println("Saved: ", abspath(output_dir))
    println("Ns=$Ns, Np=$Np, massless zero modes=", metadata["zero_mode_count"])
    println("Reference site density range: ", extrema(real.(diag(G))))
    return G
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) <= 3 || error("Usage: export_green.jl [Lx] [Ly] [output_dir]")
    Lx = isempty(ARGS) ? 12 : parse(Int, ARGS[1])
    Ly = length(ARGS) < 2 ? Lx : parse(Int, ARGS[2])
    out = length(ARGS) < 3 ? joinpath(@__DIR__, "data", "L$(Lx)x$(Ly)") : abspath(ARGS[3])
    export_green(Lx, Ly, out)
end
