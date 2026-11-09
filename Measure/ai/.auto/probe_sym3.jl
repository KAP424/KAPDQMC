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
idx(m,x,y,s) = 2*((x-1)+m*(y-1))+s+1
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
# correct maps, 1-based, x0 = x-1
w1(v,m) = mod(v,m)+1                       # wrap into 1..m
rot180(bits,m) = apply_map(bits,m,(x,y)->(w1(1-x,m), w1(1-y,m)))
refd(bits,m)   = apply_map(bits,m,(x,y)->(y,x))
refx(bits,m)   = apply_map(bits,m,(x,y)->(w1(1-x,m), y))
rot90(bits,m)  = apply_map(bits,m,(x,y)->(w1(1-y,m), x))
m, L = 3, 36
println("check involutions: rot180^2=$(rot180(rot180("100110011010101010",m),m)=="100110011010101010") refd^2=$(refd(refd("100110011010101010",m),m)=="100110011010101010") rot90^4=$(rot90(rot90(rot90(rot90("100110011010101010",m),m),m),m)=="100110011010101010")")
for b in ["111111111000000000","110010010111000110","100011100101011100","001110000100111011"]
    x0 = xnorm_of(b,m,L)
    println("$b ||X||=$(round(x0,digits=6)) | rot180 $(round(xnorm_of(rot180(b,m),m,L),digits=6)) (same=$(rot180(b,m)==b)) | refd $(round(xnorm_of(refd(b,m),m,L),digits=6)) (same=$(refd(b,m)==b)) | rot90 $(round(xnorm_of(rot90(b,m),m,L),digits=6))")
end
