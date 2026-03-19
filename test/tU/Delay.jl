push!(LOAD_PATH, "D:\\JuliaDQMC\\code\\KAPDQMC\\src\\")
using KAPDQMC
using Test
using Random
using LinearAlgebra


function Gτ(model, s::Array{UInt8,2}, τ::Int64)::Array{ComplexF64,2}
    """
    equal time Green function
    """
    BL::Array{ComplexF64,2} = model.Pt'[:, :]
    BR::Array{ComplexF64,2} = model.Pt[:, :]

    counter = 0
    for i in model.Nt:-1:τ+1
        D = [model.η[x] for x in s[:, i]]
        BL = BL * diagm(exp.(1im * model.α[i] .* D)) * model.eK
        counter += 1
        if counter == model.BatchSize
            counter = 0
            BL = Matrix(qr(BL').Q)'
        end
    end
    counter = 0
    for i in 1:1:τ
        D = [model.η[x] for x in s[:, i]]
        BR = diagm(exp.(1im * model.α[i] .* D)) * model.eK * BR
        counter += 1
        if counter == model.BatchSize
            counter = 0
            BR = Matrix(qr(BR).Q)
        end
    end

    BL = Matrix(qr(BL').Q)'
    BR = Matrix(qr(BR).Q)

    return I(model.Ns) - BR * inv(BL * BR) * BL
end
mutable struct UpdateBuffer_
    r::Matrix{ComplexF64}
    subidx::Vector{Int64}
end
function UpdateBuffer()
    return UpdateBuffer_(
        Matrix{ComplexF64}(undef, 1, 1),
        [0],
    )
end
function get_r!(UPD::UpdateBuffer_, Δs::Float64, Gt)
    @fastmath Δ = cis(Δs) - 1
    @fastmath p = 1 + Δ * (1 - Gt[UPD.subidx[1], UPD.subidx[1]])
    UPD.r[1, 1] = Δ / p
    return abs2(p)
end

rng = MersenneTwister(1234)

model = tU_Hubbard_Para(Ht=1.0, Hu1=6.0, Hu2=3.8, Θrelax=2.1, Θquench=0.3, Lattice="HoneyComb120",
    site=[3, 3], Δt=0.03, BatchSize=10, Initial="V")

s = Initial_s(model, rng)

s1 = copy(s)
s2 = copy(s)

lt = 1
x1 = 1
x2 = 2
s1[x1, lt] = rand(rng, model.samplers_dict[s[x1, lt]])
s2[x2, lt] = rand(rng, model.samplers_dict[s[x2, lt]])

G = Gτ(model, s, lt)
G1 = Gτ(model, s1, lt)
G2 = Gτ(model, s2, lt)

UPD = UpdateBuffer()

# 验证G->G2 和 G1->G2 的概率结果一致
UPD.subidx = [x2]
r1 = get_r!(UPD, model.α[lt] * (model.η[s2[x2, lt]] - model.η[s[x2, lt]]), G)
r2 = get_r!(UPD, model.α[lt] * (model.η[s2[x2, lt]] - model.η[s1[x2, lt]]), G1)

@assert s[x2, lt] == s1[x2, lt]
@assert abs(r1 - r2) < 1e-6