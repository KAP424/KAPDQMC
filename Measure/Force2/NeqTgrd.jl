using LinearAlgebra

"""
    NeqTgrd!(Pt, eK; tol=1e-10, spectrum=nothing, degeneracy_tol=nothing)

原位构造初态 Pt 的归一化虚时演化极限，返回 Pt。
eK 必须是 exp(-Δτ * K)，其中 K 为厄米矩阵，Δτ > 0。
若已有 K 的本征分解，推荐传入 `spectrum=(E, V)`，避免重新对角化 eK。

按能量从低到高逐个处理简并空间，只保留初态实际包含的独立方向。
这样不会通过反复乘法和 QR 放大原本不存在的低能分量；简并空间中的
最终态也由初态决定。初态与全局基态正交时，返回初态可达的最低能态。

tol 是正交化初态在各能量空间中投影的奇异值截断值，必须在 (0, 1) 内。
低于该阈值的真实微小分量也会被视为零，必要时需调整 tol 或提高输入精度。
degeneracy_tol 是谱分组的绝对容差，默认按谱尺度和浮点精度设置。
本实现直接求虚时极限，不再需要 maxiter。
"""
function NeqTgrd!(Pt, eK; tol=1e-10, spectrum=nothing, degeneracy_tol=nothing)
    Ns, ns = size(Pt)
    size(eK) == (Ns, Ns) || throw(DimensionMismatch("eK must have size ($Ns, $Ns)"))
    0 < ns <= Ns || throw(ArgumentError("Pt must have between 1 and Ns columns"))
    isfinite(tol) && 0 < tol < 1 || throw(ArgumentError("tol must be between 0 and 1"))
    all(isfinite, Pt) && all(isfinite, eK) || throw(ArgumentError("Pt and eK must be finite"))
    precision = eps(real(float(one(eltype(Pt)))))
    isapprox(eK, eK'; rtol=100 * Ns * precision, atol=0) ||
        throw(ArgumentError("eK must be Hermitian"))

    # 用 SVD 检查初态秩；秩不足时不能让 QR 自动补出新的物理轨道。
    initial = svd(Pt)
    initial.S[end] > tol * initial.S[1] ||
        throw(ArgumentError("Pt is numerically rank deficient at tol=$tol"))
    P0 = initial.U[:, 1:ns]

    if spectrum === nothing
        decomposition = eigen(Hermitian((eK + eK') / 2))
        minimum(decomposition.values) > 0 ||
            throw(ArgumentError("eK must be positive definite; pass spectrum=(E, V) from K if eK is ill-conditioned"))
        # eK 的本征值越大，K 的能量越低。无需对小本征值取对数。
        levels = -reverse(decomposition.values)
        basis = decomposition.vectors[:, end:-1:1]
    else
        energies, vectors = spectrum
        length(energies) == Ns && size(vectors) == (Ns, Ns) ||
            throw(DimensionMismatch("spectrum must contain Ns energies and an Ns × Ns eigenvector matrix"))
        eltype(energies) <: Real || throw(ArgumentError("energies must be real"))
        all(isfinite, energies) && all(isfinite, vectors) ||
            throw(ArgumentError("spectrum must be finite"))
        order = sortperm(energies)
        levels = energies[order]
        basis = vectors[:, order]
    end

    spectral_precision = max(precision, eps(real(float(one(eltype(levels))))))
    group_tol = isnothing(degeneracy_tol) ?
        100 * Ns * spectral_precision * maximum(abs, levels) : degeneracy_tol
    isfinite(group_tol) && group_tol >= 0 ||
        throw(ArgumentError("degeneracy_tol must be finite and nonnegative"))

    coefficients = basis' * P0
    T = promote_type(eltype(Pt), eltype(basis))
    right_basis = zeros(T, ns, ns)
    result = Matrix{T}(undef, Ns, ns)
    occupied = 0
    first_level = 1

    while first_level <= Ns && occupied < ns
        last_level = first_level
        while last_level < Ns && abs(levels[last_level + 1] - levels[first_level]) <= group_tol
            last_level += 1
        end
        block = first_level:last_level
        projected = copy(coefficients[block, :])

        # 去掉已被更低能轨道占用的初态方向；仅删除各块的零行还不够。
        # 两遍正交化避免跨能级的线性依赖被舍入误差当作新方向。
        previous = view(right_basis, :, 1:occupied)
        for _ in 1:2
            projected -= (projected * previous) * previous'
        end
        decomposition = svd(projected)
        added = count(>(tol), decomposition.S)
        added <= ns - occupied || error("NeqTgrd!: inconsistent numerical rank; increase tol")
        if added > 0
            columns = occupied + 1:occupied + added
            result[:, columns] = basis[:, block] * decomposition.U[:, 1:added]
            right_basis[:, columns] = decomposition.V[:, 1:added]
            occupied += added
        end
        first_level = last_level + 1
    end

    occupied == ns || error("NeqTgrd!: only $occupied of $ns initial directions survived; reduce tol")
    copyto!(Pt, result)
    return Pt
end
