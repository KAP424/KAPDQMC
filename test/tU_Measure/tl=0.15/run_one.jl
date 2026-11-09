# usage: julia run_one.jl <outdir> <beta> <L> <U> <warm> <sweeps> [Theta] [dt]
# 单个 (beta, L, U) 任务：HoneyComb120，Eidx = A 子格（奇数编号）
# 输出 csv 列: Ek Eu CDW0 CDW1 SDW0 SDW1（每个 sweep 一行）
push!(LOAD_PATH, "D:/JuliaDQMC/code/KAPDQMC/src/")
using KAPDQMC
using Random, LinearAlgebra
BLAS.set_num_threads(1)

outdir = ARGS[1]
beta = parse(Float64, ARGS[2])
L = parse(Int, ARGS[3])
U = parse(Float64, ARGS[4])
warm = parse(Int, ARGS[5])
sweeps = parse(Int, ARGS[6])
theta = length(ARGS) >= 7 ? parse(Float64, ARGS[7]) : 2.0 * L
dt = length(ARGS) >= 8 ? parse(Float64, ARGS[8]) : 0.05
mkpath(outdir)

rng = MersenneTwister(hash((beta, L, U)))
model = tU_Hubbard_Para(Ht=1.0, Hu1=U, Hu2=U, Θrelax=theta, Θquench=0.0, Lattice="HoneyComb120", site=[L, L], Δt=dt, BatchSize=5, Initial="V")
s = Initial_s(model, rng)
Eidx = collect(1:2:model.Ns)   # A 子格 = 奇数编号

t0 = time()
s = KAPDQMC.phy_update_M(outdir, model, Eidx, beta, s, warm, false)
t1 = time()
s = KAPDQMC.phy_update_M(outdir, model, Eidx, beta, s, sweeps, true)
println("DONE beta=$(beta) L=$(L) U=$(U) warm=$(round(t1 - t0, digits=1))s record=$(round(time() - t1, digits=1))s")
