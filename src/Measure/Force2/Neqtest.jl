push!(LOAD_PATH, "D:\\JuliaDQMC\\code\\KAPDQMC\\src\\")
using KAPDQMC
using DelimitedFiles
using LinearAlgebra


function Config_set(site, subsite, config)
    """
    E = Config_set([6, 6], [3, 3], [1 1 -1; 1 1 1; -1 1 1])
    L = 6
    E_idx = [2 * (x + (y - 1) * L) - Int(x % 3 == 1 && y % 3 == 0) - Int(x % 3 == 0 && y % 3 == 1) for y in 1:L for x in 1:L]
    sort(collect(E)) == sort(collect(E_idx))
    """
    subE = zeros(Int, prod(subsite))
    count = 1
    for i in 1:subsite[1]
        for j in 1:subsite[2]
            if config[i, j] == 1
                subE[count] = 2 * (i + (j - 1) * site[1])
            else
                subE[count] = 2 * (i + (j - 1) * site[1]) - 1
            end
            count += 1
        end
    end
    E = []
    for i in 1:div(site[1], subsite[1])
        for j in 1:div(site[2], subsite[2])
            E = vcat(E, subE .+ 2 * subsite[1] * (i - 1) .+ 2 * subsite[2] * site[2] * (j - 1))
        end
    end
    @assert length(E) == prod(site) "E_idx 长度不正确"
    return E
end


include(joinpath(@__DIR__, "NeqTgrd.jl"))


Lattice = "HoneyComb120"
filling_num = 0.5
subsite = [2, 2]
config = [1 1; -1 -1]

L_set = [6, 12, 18, 24]
# L_set = [9, 18, 27, 36]
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
    site = [L, L]
    E_idx = Config_set(site, subsite, config)

    EA = zeros(Float64, length(beta_set) + 1)
    EB = zeros(Float64, length(beta_set) + 1)
    ΔE = zeros(Float64, length(beta_set) + 1)
    EA[1] = EB[1] = ΔE[1] = L

    # Half
    idxA = area_index(Lattice, site, ([1, 1], [div(L, 3), L]))
    # HalfHalf
    idxB = area_index(Lattice, site, ([1, 1], [div(L, 3), div(2 * L, 3)]))

    GM_A = zeros(Float64, length(idxA), length(idxA))
    GM_B = zeros(Float64, length(idxB), length(idxB))

    K = nnK_Matrix(Lattice, site)
    E, V = LAPACK.syevd!('V', 'L', copy(K))
    eK = V * Diagonal(exp.(-3 .* E)) * V'

    Ns = size(K, 1)
    ns = round(Int, Ns * filling_num)
    Pt = zeros(Float64, Ns, ns)  # 当前 H0 初态为实数，避免不必要的复数运算
    F_idx = setdiff(1:Ns, E_idx)
    for i in eachindex(F_idx)
        Pt[F_idx[i], i] = 1
    end


    n = ones(Ns)
    n[E_idx] .= exp(-2 * 3.0)

    @assert norm(abs.(Matrix(qr(n .* Pt).Q)) - Pt) < 1e-5 "投影算符构建有错"

    # overlap_svals = svdvals(Pt' * str_M_Pt)
    # overlap_tol = 1e-10  # 当前 μ=1e-5 的可调数值容差，不是严格零的数学证明
    # overlap_nzero = sum(overlap_svals .< overlap_tol)
    # if overlap_nzero > 0
    #     log_overlap = sum(log, overlap_svals)  # log|det|，避免重叠随尺寸指数变小而下溢
    #     println("L = $L, σ_min = $(overlap_svals[end]), log|overlap| = $log_overlap, ",
    #         "tol = $overlap_tol, 候选零奇异值数 = $overlap_nzero")
    #     println("重叠在该容差下数值为零，可能正交。")
    # end
    ###########################################################################################

    NeqTgrd!(Pt, eK; spectrum=(E, V))
    tau = Vector{eltype(Pt)}(undef, ns)
    G = Matrix{eltype(Pt)}(undef, Ns, Ns)


    M_Pt = similar(Pt)
    for i in eachindex(beta_set)
        n = ones(Ns)
        n[E_idx] .= exp(-2 * beta_set[i])
        M_Pt .= n .* Pt  # 等价于 Diagonal(n) * Pt，复用缓冲区
        LAPACK.geqrf!(M_Pt, tau)
        LAPACK.orgqr!(M_Pt, tau)  # 原地生成 Ns × ns 的薄 Q

        G = M_Pt * M_Pt'  # QR 后列已正交
        @assert norm(M_Pt' * M_Pt - I) < 1e-10 "M_Pt' * M_Pt is not identity!"

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

