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
w1(v,m) = mod(v,m)+1
rot180(bits,m) = apply_map(bits,m,(x,y)->(w1(1-x,m), w1(1-y,m)))
refd(bits,m)   = apply_map(bits,m,(x,y)->(y,x))
# build class-1 patterns properly: each cell puts its fermion on A or B
function class1(m, cells::Vector{Int})   # cells[j] = 0 -> A, 1 -> B
    out = fill('0', 2m^2)
    for (j,c) in enumerate(cells)
        out[2*(j-1)+1+c] = '1'
    end
    return String(out)
end
m, L = 3, 36
uniA  = class1(m, zeros(Int, 9))
uniB  = class1(m, ones(Int, 9))
stripe= class1(m, [0,1,0,1,0,1,0,1,0])
mixp  = class1(m, [0,0,1,1,0,0,1,1,0])
for (nm,b) in (("uniform_A",uniA),("uniform_B",uniB),("stripe",stripe),("mixed",mixp))
    x0 = xnorm_of(b,m,L)
    println("$nm ||X||=$(round(x0,digits=6)) | rot180 $(round(xnorm_of(rot180(b,m),m,L),digits=6)) | refd $(round(xnorm_of(refd(b,m),m,L),digits=6))")
end
