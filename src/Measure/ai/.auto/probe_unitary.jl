include("D:/JuliaDQMC/code/KAPDQMC/src/Measure/ai/strong_scaling.jl")
const FE = FermionExperiment
using Printf
function xstats(bits::String, m::Int, L::Int)
    occ = BitVector([c == '1' for c in bits])
    model = FE.BlochModel(L, m); prep = FE.prepare_pattern(model, occ); base = FE.evolve(prep, Inf)
    eset, fset = findall(.!occ), findall(occ)
    lam = Inf; smax = 0.0; smin = Inf
    for k in 1:length(base.orbitals)
        qf = base.orbitals[k][fset, :]
        s = svdvals(qf); lam = min(lam, minimum(s)^2)
        Xk = base.orbitals[k][eset, :] / qf
        sx = svdvals(Xk); smax = max(smax, maximum(sx)); smin = min(smin, minimum(sx))
    end
    return sqrt(1/lam - 1), smax, smin
end
# the configurations with theta == 45 exactly, at fixed L, from the dataset
using DelimitedFiles
for (m, L) in ((1,48),(2,48),(3,36),(4,36))
    # uniform_A: all A occupied
    out = fill('0', 2m^2)
    for j in 1:m^2; out[2*(j-1)+1] = '1'; end
    bits = String(out)
    xn, smax, smin = xstats(bits, m, L)
    @printf("m=%d L=%d uniform_A: ||X||=%.10f  sigma_max(X)=%.10f  sigma_min(X)=%.10f  unitary=%s\n",
        m, L, xn, smax, smin, abs(smax-1)<1e-9 && abs(smin-1)<1e-9)
end
