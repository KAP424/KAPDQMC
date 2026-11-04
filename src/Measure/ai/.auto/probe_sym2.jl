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
function idx(m,x,y,s); 2*((x-1)+m*(y-1))+s+1; end
function apply_map(bits,m,src)
    out = collect(bits)
    for y in 1:m, x in 1:m
        ax, ay = src(x,y)
        for s in 0:1
            out[idx(m,x,y,s)] = bits[idx(m,ax,ay,s)]
        end
    end
    return String(out)
end
rot180(bits,m) = apply_map(bits,m,(x,y)->(mod(2-x,m)+1, mod(2-y,m)+1))
refd(bits,m)   = apply_map(bits,m,(x,y)->(y,x))
m, L = 3, 36
tests = ["100110011010101010","110010010111000110","111111111000000000"]
for b in tests
    x0 = xnorm_of(b,m,L)
    x1 = xnorm_of(rot180(b,m),m,L)
    x2 = xnorm_of(refd(b,m),m,L)
    println("$b  ||X||=$(round(x0,digits=6))  rot180->$(round(x1,digits=6))  ref_diag->$(round(x2,digits=6))")
    println("      rot180 same pattern? $(rot180(b,m)==b)   ref_diag same? $(refd(b,m)==b)")
end
