push!(LOAD_PATH, "D:\\JuliaDQMC\\code\\KAPDQMC\\src\\")

using KAPDQMC

using DelimitedFiles
using LinearAlgebra

function Initial_Pt(Pt, K, μ)
    Ns = size(K, 1)
    if Initial == "H0"
        KK = copy(K)
        KK .+= μ * diagm(repeat([-1, 1], div(Ns, 2)))
        E, V = LAPACK.syevd!('V', 'L', KK)
        Pt .= V[:, 1:div(Ns, 2)]
    else
        error("Unsupported Initial state: $Initial")
    end
end

Lattice = "HoneyComb120"
filling_num = 0.5
Initial = "H0"

# L_set = [6, 12, 18, 24, 30, 36, 42, 48]
# beta_set = collect(0:0.2:10.0)

L_set = [9, 18, 27, 36]
# L_set = [3, 6, 9, 12, 15, 18, 21]
beta_set = collect(0:0.05:3.0)

csv_dir = @__DIR__
csv_header = [0.0; beta_set]

for filename in ("EA.csv", "EB.csv", "ΔE.csv")
    path = joinpath(csv_dir, filename)
    first_row = isfile(path) ? tryparse.(Float64, strip.(split(readline(path), ','))) : nothing
    if first_row != csv_header
        old_data = isfile(path) ? read(path) : UInt8[]
        open(path, "w") do io
            writedlm(io, permutedims(csv_header), ',')
            write(io, old_data)  # 表头放在最前面，保留原有数据
        end
    end
end

for L in L_set
    EA = zeros(Float64, length(beta_set) + 1)
    EB = zeros(Float64, length(beta_set) + 1)
    ΔE = zeros(Float64, length(beta_set) + 1)
    EA[1] = EB[1] = ΔE[1] = L

    site = [L, L]
    # Half
    idxA = area_index(Lattice, site, ([1, 1], [div(L, 3), L]))
    # HalfHalf
    idxB = area_index(Lattice, site, ([1, 1], [div(L, 3), div(2 * L, 3)]))

    GM_A = zeros(Float64, length(idxA), length(idxA))
    GM_B = zeros(Float64, length(idxB), length(idxB))

    K = nnK_Matrix(Lattice, site)
    Ns = size(K, 1)
    ns = round(Int, Ns * filling_num)
    Pt = zeros(Float64, Ns, ns)  # 当前 H0 初态为实数，避免不必要的复数运算
    Initial_Pt(Pt, K, 1e-5)
    M_Pt = similar(Pt)

    ###########################################################################################

    # 构型一：每个 3×3 原胞块只有一个原胞选择 B 满
    # E_idx = [2 * (x + (y - 1) * L) - Int(x % 3 == 0 && y % 3 == 0)
    #          for y in 1:L for x in 1:L]

    # E_idx = [2 * (x + (y - 1) * L) - Int(x % 3 == 0 && y % 3 == 0) - Int(x % 3 == 1 && y % 3 == 1)
    #          for y in 1:L for x in 1:L]

    # E_idx = [2 * (x + (y - 1) * L) - Int(x % 3 == 1 && y % 3 == 0) - Int(x % 3 == 0 && y % 3 == 1)
    #          for y in 1:L for x in 1:L]

    # E_idx = [2 * (x + (y - 1) * L) - Int(x % 3 == 1 && y % 3 == 2) - Int(x % 3 == 2 && y % 3 == 1)
    #          for y in 1:L for x in 1:L]

    # E_idx = [2 * (x + (y - 1) * L) - Int(x % 3 == 1 && y % 3 == 1) - Int(x % 3 == 2 && y % 3 == 2)
    #          for y in 1:L for x in 1:L]

    # E_idx = [2 * (x + (y - 1) * L) - Int(x % 3 == 1 && y % 3 == 1) - Int(x % 3 == 3 && y % 3 == 3)
    #          for y in 1:L for x in 1:L]

    # 对角失败
    # E_idx = [2 * (x + (y - 1) * L) - Int(x % 3 == y % 3)
    #          for y in 1:L for x in 1:L]

    # 1对角+2非对角
    # E_idx = [2 * (x + (y - 1) * L) - Int(x % 3 == 1 && y % 3 == 1) - Int(x % 3 == 2 && y % 3 == 3) - Int(x % 3 == 3 && y % 3 == 2)
    #          for y in 1:L for x in 1:L]

    # 2对角+1非对角
    E_idx = [2 * (x + (y - 1) * L) - Int(x % 3 == 1 && y % 3 == 1) - Int(x % 3 == 2 && y % 3 == 2) - Int(x % 3 == 3 && y % 3 == 2)
             for y in 1:L for x in 1:L]

    ###########################################################################################

    # 每个 2×2 原胞块：对角2 个 B 满、非对角2 个 A 满
    # E_idx = [2 * (x + (y - 1) * L) - Int(x % 2 == y % 2)
    #          for y in 1:L for x in 1:L]

    # 每个 2×2 原胞块：左列 A 满，右列 B 满。
    # E_idx = [2 * (x + (y - 1) * L) - Int(iseven(x))
    #          for y in 1:L for x in 1:L]

    ###########################################################################################
    n = ones(Ns)
    n[E_idx] .= exp(-2 * 3.0)
    M_Pt .= n .* Pt

    str_M_Pt = zeros(Ns, ns)
    count = 1
    for i in 1:Ns
        if !(i in E_idx)
            str_M_Pt[i, count] = 1
            count += 1
        end
    end

    @assert norm(abs.(Matrix(qr(n .* str_M_Pt).Q)) - str_M_Pt) < 1e-5 "投影算符构建有错"

    overlap_svals = svdvals(Pt' * str_M_Pt)
    overlap_tol = 1e-10  # 当前 μ=1e-5 的可调数值容差，不是严格零的数学证明
    overlap_nzero = sum(overlap_svals .< overlap_tol)
    if overlap_nzero > 0
        log_overlap = sum(log, overlap_svals)  # log|det|，避免重叠随尺寸指数变小而下溢
        println("L = $L, σ_min = $(overlap_svals[end]), log|overlap| = $log_overlap, ",
            "tol = $overlap_tol, 候选零奇异值数 = $overlap_nzero")
        println("重叠在该容差下数值为零，可能正交。")
    end
    ###########################################################################################

    tau = Vector{eltype(Pt)}(undef, ns)
    G = Matrix{eltype(Pt)}(undef, Ns, Ns)


    for i in eachindex(beta_set)
        n = ones(Ns)
        n[E_idx] .= exp(-2 * beta_set[i])
        M_Pt .= n .* Pt  # 等价于 Diagonal(n) * Pt，复用缓冲区
        LAPACK.geqrf!(M_Pt, tau)
        LAPACK.orgqr!(M_Pt, tau)  # 原地生成 Ns × ns 的薄 Q

        G = M_Pt * M_Pt'  # QR 后列已正交

        if beta_set[i] < 4
            GroverMatrix!(GM_A, view(G, idxA, idxA), view(G, idxA, idxA))
            GroverMatrix!(GM_B, view(G, idxB, idxB), view(G, idxB, idxB))
            EA[i+1] = -logdet(GM_A)
            EB[i+1] = -logdet(GM_B)
        else
            # 等价的纯态公式，避免大 beta 时 logdet 丢失极小的熵。
            EA[i+1] = -sum(s -> log1p(-2s^2), svdvals(G[idxA, setdiff(1:Ns, idxA)]))
            EB[i+1] = -sum(s -> log1p(-2s^2), svdvals(G[idxB, setdiff(1:Ns, idxB)]))
        end
        ΔE[i+1] = EA[i+1] - EB[i+1]
    end

    for (filename, data) in (("EA.csv", EA), ("EB.csv", EB), ("ΔE.csv", ΔE))
        open(joinpath(csv_dir, filename), "a") do io
            writedlm(io, data', ',')
        end
    end

end

