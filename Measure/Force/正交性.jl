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


L_set = [3, 6, 9, 12, 15, 18, 21]
beta_set = collect(0:0.05:3.0)

csv_dir = @__DIR__
csv_header = [0.0; beta_set]

for L in L_set
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

    E_idx = collect(2:2:Ns)
    E_idx[2] = 1


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


end

