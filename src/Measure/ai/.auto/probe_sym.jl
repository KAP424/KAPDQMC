include("D:/JuliaDQMC/code/KAPDQMC/src/Measure/ai/strong_scaling.jl")
const FE = FermionExperiment
function xnorm_of(bits::String, m::Int, L::Int)
    occ = BitVector([c == '1' for c in bits])
    model = FE.BlochModel(L, m); prep = FE.prepare_pattern(model, occ); base = FE.evolve(prep, Inf)
    eset, fset = findall(.!occ), findall(occ)
    lam = Inf
    for k in 1:length(base.orbitals)
        s = svdvals(base.orbitals[k][fset, :]); lam = min(lam, minimum(s)^2)
    end
    return sqrt(1/lam - 1)
end
pairs = [("110010010111000110","010001101011001101"),
         ("100011100101011100","001011110100011001"),
         ("001110000100111011","001100011011001110")]
for (a,b) in pairs
    xa = xnorm_of(a,3,36); xb = xnorm_of(b,3,36)
    println("$a -> $(round(xa,digits=6))   |   $b -> $(round(xb,digits=6))   ratio=$(round(xa/xb,digits=4))")
end
