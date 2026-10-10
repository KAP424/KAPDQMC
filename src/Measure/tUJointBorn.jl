module TUJointBorn

using LinearAlgebra
using Random: AbstractRNG
import ..KAPDQMC

const _tU = KAPDQMC.tUDQMC

export ESwap2Buffer,
    measurement_node_index,
    measurement_block_index,
    measurement_scale,
    build_measured_block!,
    build_measured_block_inverse!,
    build_measured_block_pair!,
    prepare_eswap!,
    apply_eswap_green!,
    apply_eswap_block!,
    swap_membership!,
    tu_weight_ratio,
    tu_logweight_ratio,
    conditioned_tu_sweeps!,
    measured_center_green,
    tu_eswap_logratio

"""
    measurement_scale(beta)

Return the one-particle factor in `M_E^2`.  For

    M_E = exp(-2 beta N_E),

the center insertion is `B_E^2 = diag(d2)` with `d2 = exp(-4beta)` on
the sites in `E`.
"""
measurement_scale(beta::Real) = exp(-4 * float(beta))

"""
    measurement_node_index(model)

Return the node at the center of the projector contour.  The tU node layout is
constructed to contain `model.Nt / 2`; checking it here prevents silently
putting the measurement on the wrong side of a time block.
"""
function measurement_node_index(model)
    iseven(model.Nt) || throw(ArgumentError("the measurement center requires even model.Nt"))
    center_time = model.Nt ÷ 2
    node = findfirst(==(center_time), model.nodes)
    isnothing(node) && throw(ArgumentError("model.nodes does not contain the center time $center_time"))
    return node
end

"""
    measurement_block_index(model)

Return the block ending at the center node.  `B_E^2` is *left multiplied* onto
this block.  With that convention the measured center right orbital is
`BR_E = B_E^2 BR`, so the existing DQMC Green update orientation remains valid.
"""
function measurement_block_index(model)
    block = measurement_node_index(model) - 1
    block >= 1 || throw(ArgumentError("the center node has no block on its right-projecting side"))
    return block
end

@inline function _check_measurement_scale(d2::Real)
    isfinite(d2) && d2 > 0 || throw(ArgumentError("d2 must be finite and positive; got $d2"))
    return float(d2)
end

@inline function _check_sites(E_idx, Ns)
    all(i -> 1 <= i <= Ns, E_idx) || throw(BoundsError(1:Ns, E_idx))
    length(unique(E_idx)) == length(E_idx) || throw(ArgumentError("E_idx contains duplicate sites"))
    return nothing
end

@inline function _left_scale_rows!(A::AbstractMatrix, sites, factor)
    @inbounds for site in sites
        @views A[site, :] .*= factor
    end
    return A
end

@inline function _right_scale_columns!(A::AbstractMatrix, sites, factor)
    @inbounds for site in sites
        @views A[:, site] .*= factor
    end
    return A
end

"""
    build_measured_block!(tmpN, tmpNN, BM, model, s, block, E_idx, d2)

Build one ordinary tU propagation block and insert `B_E^2` when `block` is the
block ending at the center.  The insertion order is

    BM_measured = B_E^2 * BM.

This order is deliberate.  It makes a fixed-count measurement swap a left
diagonal update, exactly matching the convention used by the existing Green
update in `public/Gupdate.jl`.
"""
function build_measured_block!(tmpN, tmpNN, BM, model, s, block::Integer, E_idx, d2::Real)
    d2 = _check_measurement_scale(d2)
    _check_sites(E_idx, model.Ns)
    _tU.BM_F!(tmpN, tmpNN, BM, model, s, block)
    if block == measurement_block_index(model)
        _left_scale_rows!(BM, E_idx, d2)
    end
    return BM
end

"""
    build_measured_block_inverse!(tmpN, tmpNN, BMinv, model, s, block, E_idx, d2)

Build the inverse of `build_measured_block!`.  Since

    inv(B_E^2 * BM) = inv(BM) * inv(B_E^2),

the inverse measurement factor must be *right multiplied*, i.e. the selected
columns are divided by `d2`.
"""
function build_measured_block_inverse!(tmpN, tmpNN, BMinv, model, s, block::Integer, E_idx, d2::Real)
    d2 = _check_measurement_scale(d2)
    _check_sites(E_idx, model.Ns)
    _tU.BMinv_F!(tmpN, tmpNN, BMinv, model, s, block)
    if block == measurement_block_index(model)
        _right_scale_columns!(BMinv, E_idx, inv(d2))
    end
    return BMinv
end

function build_measured_block_pair!(tmpN, tmpNN, BM, BMinv, model, s, block::Integer, E_idx, d2::Real)
    build_measured_block!(tmpN, tmpNN, BM, model, s, block, E_idx, d2)
    build_measured_block_inverse!(tmpN, tmpNN, BMinv, model, s, block, E_idx, d2)
    return BM, BMinv
end

"""
Scratch storage for a two-site, fixed-count measurement update.

The tU code stores the hole Green matrix `G = I - C`.  If `e = E[j]` is
removed from `E` and `s = S[i]` is inserted, the center multiplier changes by

    R[e,e] = 1/d2,    R[s,s] = d2.

For `subidx = [e,s]`, define `Delta = R-I` and

    K = I + Delta * C[subidx,subidx]
      = I + Delta * (I-G)[subidx,subidx].

`det(K)` is the one-flavor determinant ratio.  The sign-free tU weight contains
the conjugate flavor as well, hence its physical ratio is `abs2(det(K))`.
"""
mutable struct ESwap2Buffer{T<:Number}
    subidx::Vector{Int}
    delta::Matrix{T}
    kernel::Matrix{T}
    kernel_factor::Matrix{T}
    rinv_delta::Matrix{T}
    n_by_2::Matrix{T}
    two_by_n::Matrix{T}
    n_by_n::Matrix{T}
    single_flavor_ratio::T
    logabs_single_flavor_ratio::Float64
    update_ready::Bool
end

function ESwap2Buffer(::Type{T}, Ns::Integer) where {T<:Number}
    Ns >= 2 || throw(ArgumentError("a fixed-count swap requires at least two sites"))
    return ESwap2Buffer{T}(
        zeros(Int, 2),
        zeros(T, 2, 2),
        zeros(T, 2, 2),
        zeros(T, 2, 2),
        zeros(T, 2, 2),
        zeros(T, Ns, 2),
        zeros(T, 2, Ns),
        zeros(T, Ns, Ns),
        zero(T),
        -Inf,
        false,
    )
end

ESwap2Buffer(G::AbstractMatrix{T}) where {T<:Number} = ESwap2Buffer(T, size(G, 1))

"""
    prepare_eswap!(buffer, G, E_idx, S_idx, i, j, d2)

Prepare the ratio and Woodbury matrix for the swap `S[i] <-> E[j]` without
changing `G`, `E_idx`, or `S_idx`.  The returned named tuple contains

- `single_flavor`: the generally complex ratio `det(K)`;
- `tu_weight`: `abs2(det(K))`, including the conjugate tU flavor;
- `tu_logweight`: `log(abs2(det(K)))`, evaluated without first squaring a tiny
  determinant;
- `update_ready`: false only when the proposal has exactly zero weight and its
  singular Green update must therefore not be applied.

Call `apply_eswap_green!` only after the Metropolis proposal is accepted.
"""
function prepare_eswap!(buffer::ESwap2Buffer, G::AbstractMatrix,
    E_idx, S_idx, i::Integer, j::Integer, d2::Real)

    d2 = _check_measurement_scale(d2)
    size(G, 1) == size(G, 2) || throw(DimensionMismatch("G must be square"))
    size(G, 1) == size(buffer.n_by_n, 1) || throw(DimensionMismatch("buffer and G sizes differ"))
    1 <= i <= length(S_idx) || throw(BoundsError(S_idx, i))
    1 <= j <= length(E_idx) || throw(BoundsError(E_idx, j))

    e_site = E_idx[j]
    s_site = S_idx[i]
    e_site != s_site || throw(ArgumentError("E[j] and S[i] must be different sites"))
    1 <= e_site <= size(G, 1) || throw(BoundsError(axes(G, 1), e_site))
    1 <= s_site <= size(G, 1) || throw(BoundsError(axes(G, 1), s_site))

    buffer.subidx[1] = e_site
    buffer.subidx[2] = s_site

    fill!(buffer.delta, zero(eltype(buffer.delta)))
    buffer.delta[1, 1] = inv(d2) - 1
    buffer.delta[2, 2] = d2 - 1

    # C[subidx,subidx] = I - G[subidx,subidx].
    @inbounds for row in 1:2, col in 1:2
        buffer.kernel[row, col] = -buffer.delta[row, row] * G[buffer.subidx[row], buffer.subidx[col]]
    end
    buffer.kernel[1, 1] += one(eltype(buffer.kernel)) + buffer.delta[1, 1]
    buffer.kernel[2, 2] += one(eltype(buffer.kernel)) + buffer.delta[2, 2]

    buffer.single_flavor_ratio = det(buffer.kernel)
    logabs, _ = logabsdet(buffer.kernel)
    buffer.logabs_single_flavor_ratio = logabs

    # Solve K * X = Delta.  Keep `kernel` intact for diagnostics.
    copyto!(buffer.kernel_factor, buffer.kernel)
    copyto!(buffer.rinv_delta, buffer.delta)
    factor = lu!(buffer.kernel_factor; check=false)
    buffer.update_ready = issuccess(factor)
    if buffer.update_ready
        ldiv!(factor, buffer.rinv_delta)
    else
        # A singular K means that the proposed configuration has zero weight.
        # Metropolis must be able to reject it without trying to form inv(K).
        fill!(buffer.rinv_delta, zero(eltype(buffer.rinv_delta)))
    end

    return (
        single_flavor=buffer.single_flavor_ratio,
        tu_weight=abs2(buffer.single_flavor_ratio),
        tu_logweight=2 * buffer.logabs_single_flavor_ratio,
        update_ready=buffer.update_ready,
    )
end

tu_weight_ratio(buffer::ESwap2Buffer) = abs2(buffer.single_flavor_ratio)
tu_logweight_ratio(buffer::ESwap2Buffer) = 2 * buffer.logabs_single_flavor_ratio

"""
    apply_eswap_green!(G, buffer)

Apply a previously prepared and accepted swap to the hole Green matrix:

    G' = G - G[:,sub] * inv(K) * Delta * (I-G)[sub,:].

The function mutates `G`.  It deliberately does not change `E_idx` or `S_idx`.
The buffer is specific to the `G` passed to `prepare_eswap!`: every replica
must prepare and apply its own buffer.  After all replica Green matrices have
been updated, commit the shared membership lists once with `swap_membership!`.
"""
function apply_eswap_green!(G::AbstractMatrix, buffer::ESwap2Buffer)
    size(G) == size(buffer.n_by_n) || throw(DimensionMismatch("buffer and G sizes differ"))
    buffer.update_ready || throw(ArgumentError("the prepared swap has a singular update matrix and cannot be applied"))

    @views mul!(buffer.n_by_2, G[:, buffer.subidx], buffer.rinv_delta)

    # two_by_n = (I-G)[subidx,:]
    @inbounds for row in 1:2, col in axes(G, 2)
        buffer.two_by_n[row, col] = -G[buffer.subidx[row], col]
    end
    buffer.two_by_n[1, buffer.subidx[1]] += one(eltype(buffer.two_by_n))
    buffer.two_by_n[2, buffer.subidx[2]] += one(eltype(buffer.two_by_n))

    mul!(buffer.n_by_n, buffer.n_by_2, buffer.two_by_n)
    G .-= buffer.n_by_n
    return G
end

"""
    apply_eswap_block!(BM, BMinv, E_idx, S_idx, i, j, d2)

Update an already measured center block after accepting `S[i] <-> E[j]`.
For the forward block, `BM' = R*BM`.  For its inverse,
`BMinv' = BMinv*inv(R)`.  The site lists are not changed here.
"""
function apply_eswap_block!(BM::AbstractMatrix, BMinv::AbstractMatrix,
    E_idx, S_idx, i::Integer, j::Integer, d2::Real)

    d2 = _check_measurement_scale(d2)
    e_site = E_idx[j]
    s_site = S_idx[i]

    @views BM[e_site, :] .*= inv(d2)
    @views BM[s_site, :] .*= d2

    @views BMinv[:, e_site] .*= d2
    @views BMinv[:, s_site] .*= inv(d2)
    return BM, BMinv
end

"""
Commit the fixed-count exchange `S[i] <-> E[j]` to mutable membership lists.

The positions are preserved, so the lists need not remain sorted.  This is
useful for low-level cached updates.  Before using the outcome as a sampler or
persistent-bank label, create its canonical immutable form with
`Tuple(sort(E_idx))`.
"""
function swap_membership!(E_idx, S_idx, i::Integer, j::Integer)
    old_e = E_idx[j]
    E_idx[j] = S_idx[i]
    S_idx[i] = old_e
    return E_idx, S_idx
end

"""
    conditioned_tu_sweeps!(rng, model, field, E_idx, beta, sweeps; observe=nothing)

Run the existing stabilized tU projector-DQMC update at fixed measurement
outcome `E_idx`.  The auxiliary-field weight contains the center insertion

    B_E^2 = exp(-4beta * D_E),

so the invariant distribution is proportional to the unmeasured HS weight
times `F_E`.  The function mutates and returns `field`.  It is the conditional
kernel required by `update_replicas!` and by the inner chain of
`double_metropolis_outcome_step!`.

Every rebuilding of the time block ending at the center reapplies the
measurement insertion; omitting that step after an HS update would silently
return to the unmeasured distribution.

For diagnostics, `inspect_layer(direction, time_index, G, field)` is called
after `G` has been placed on the physical side of the measurement filter for
that layer and immediately before its local HS updates.  It may inspect or
temporarily modify and restore its arguments, but must leave them unchanged on
return.
"""
function conditioned_tu_sweeps!(
    rng::AbstractRNG,
    model,
    field::Array{UInt8,2},
    E_idx,
    beta::Real,
    sweeps::Integer,
    ;
    observe=nothing,
    inspect_layer=nothing,
)
    sweeps >= 0 || throw(ArgumentError("sweeps must be nonnegative"))
    size(field) == (model.Ns, model.Nt) ||
        throw(DimensionMismatch("field must have size (model.Ns, model.Nt)"))
    _check_sites(E_idx, model.Ns)
    sweeps == 0 && return field

    node_count = length(model.nodes)
    physical = _tU.PhyBuffer(model.Ns, node_count)
    update = _tU.UpdateBuffer()
    d2 = measurement_scale(beta)

    G = physical.G
    tau = physical.tau
    pivots = physical.ipiv
    left_orbitals = physical.BLs
    right_orbitals = physical.BRs
    diagonal = physical.N
    scratch_matrix = physical.NN
    block = physical.BM
    tall_scratch = physical.Nn
    overlap = physical.nn
    wide_scratch = physical.nN

    right_orbitals[:, :, 1] .= model.HalfeKinv * model.Pt
    left_orbitals[:, :, node_count] .= model.Pt' * model.HalfeK

    for block_index in node_count-1:-1:1
        build_measured_block!(
            diagonal,
            scratch_matrix,
            block,
            model,
            field,
            block_index,
            E_idx,
            d2,
        )
        mul!(wide_scratch, view(left_orbitals, :, :, block_index + 1), block)
        LAPACK.gerqf!(wide_scratch, tau)
        LAPACK.orgrq!(wide_scratch, tau, model.Ns ÷ 2)
        copyto!(view(left_orbitals, :, :, block_index), wide_scratch)
    end

    node_index = 1
    _tU.get_G!(
        overlap,
        wide_scratch,
        pivots,
        view(left_orbitals, :, :, node_index),
        view(right_orbitals, :, :, node_index),
        G,
    )

    for _ in 1:sweeps
        for time_index in 1:model.Nt
            @inbounds @simd for site in 1:model.Ns
                diagonal[site] = model.exp_αη_pos[time_index, field[site, time_index]]
            end
            _tU.WrapKV!(
                scratch_matrix,
                model.eK,
                model.eKinv,
                diagonal,
                G,
                "Forward",
                "B",
            )
            inspect_layer !== nothing && inspect_layer(:forward, time_index, G, field)
            _tU.UpdatePhyLayer!(
                rng,
                view(field, :, time_index),
                time_index,
                model,
                update,
                physical,
            )

            if time_index in model.nodes
                node_index += 1
                build_measured_block!(
                    diagonal,
                    scratch_matrix,
                    block,
                    model,
                    field,
                    node_index - 1,
                    E_idx,
                    d2,
                )
                mul!(tall_scratch, block, view(right_orbitals, :, :, node_index - 1))
                LAPACK.geqrf!(tall_scratch, tau)
                LAPACK.orgqr!(tall_scratch, tau, model.Ns ÷ 2)
                copyto!(view(right_orbitals, :, :, node_index), tall_scratch)

                _tU.get_G!(
                    overlap,
                    wide_scratch,
                    pivots,
                    view(left_orbitals, :, :, node_index),
                    view(right_orbitals, :, :, node_index),
                    G,
                )
            end
        end

        for time_index in model.Nt:-1:1
            if time_index == model.Nt ÷ 2
                # The stabilized rebuild at the center returns the Green matrix
                # immediately after the measurement filter D = B_E^2.  The
                # center slice, and every earlier slice in this backward pass,
                # lives before that filter.  Cross the discontinuity explicitly:
                #
                #     G_after = D * G_before * inv(D)
                #     G_before = inv(D) * G_after * D.
                #
                # This is essential when a time block contains slices on the
                # earlier side of the center (BatchSize > 1).
                _left_scale_rows!(G, E_idx, inv(d2))
                _right_scale_columns!(G, E_idx, d2)
            end
            inspect_layer !== nothing && inspect_layer(:backward, time_index, G, field)
            _tU.UpdatePhyLayer!(
                rng,
                view(field, :, time_index),
                time_index,
                model,
                update,
                physical,
            )
            @inbounds @simd for site in 1:model.Ns
                diagonal[site] = model.exp_αη_neg[time_index, field[site, time_index]]
            end
            _tU.WrapKV!(
                scratch_matrix,
                model.eK,
                model.eKinv,
                diagonal,
                G,
                "Backward",
                "B",
            )

            if (time_index - 1) in model.nodes
                node_index -= 1
                build_measured_block!(
                    diagonal,
                    scratch_matrix,
                    block,
                    model,
                    field,
                    node_index,
                    E_idx,
                    d2,
                )
                mul!(wide_scratch, view(left_orbitals, :, :, node_index + 1), block)
                LAPACK.gerqf!(wide_scratch, tau)
                LAPACK.orgrq!(wide_scratch, tau, model.Ns ÷ 2)
                copyto!(view(left_orbitals, :, :, node_index), wide_scratch)

                _tU.get_G!(
                    overlap,
                    wide_scratch,
                    pivots,
                    view(left_orbitals, :, :, node_index),
                    view(right_orbitals, :, :, node_index),
                    G,
                )
            end
        end
        observe !== nothing && observe(field)
    end

    return field
end

"""
    measured_center_green(model, field, E_idx, beta)

Rebuild the stabilized hole Green matrix at the measurement center for one HS
configuration and one fixed outcome.  This allocation-heavy reference path is
useful for correctness checks and for wiring the generic joint sampler before
introducing persistent Green caches.
"""
function measured_center_green(model, field::Array{UInt8,2}, E_idx, beta::Real)
    size(field) == (model.Ns, model.Nt) ||
        throw(DimensionMismatch("field must have size (model.Ns, model.Nt)"))
    _check_sites(E_idx, model.Ns)

    node_count = length(model.nodes)
    center_node = measurement_node_index(model)
    physical = _tU.PhyBuffer(model.Ns, node_count)
    d2 = measurement_scale(beta)

    left_orbitals = physical.BLs
    right_orbitals = physical.BRs
    diagonal = physical.N
    scratch_matrix = physical.NN
    block = physical.BM
    tall_scratch = physical.Nn
    wide_scratch = physical.nN
    overlap = physical.nn
    pivots = physical.ipiv
    tau = physical.tau

    right_orbitals[:, :, 1] .= model.HalfeKinv * model.Pt
    left_orbitals[:, :, node_count] .= model.Pt' * model.HalfeK

    for block_index in node_count-1:-1:center_node
        build_measured_block!(
            diagonal,
            scratch_matrix,
            block,
            model,
            field,
            block_index,
            E_idx,
            d2,
        )
        mul!(wide_scratch, view(left_orbitals, :, :, block_index + 1), block)
        LAPACK.gerqf!(wide_scratch, tau)
        LAPACK.orgrq!(wide_scratch, tau, model.Ns ÷ 2)
        copyto!(view(left_orbitals, :, :, block_index), wide_scratch)
    end

    for block_index in 1:center_node-1
        build_measured_block!(
            diagonal,
            scratch_matrix,
            block,
            model,
            field,
            block_index,
            E_idx,
            d2,
        )
        mul!(tall_scratch, block, view(right_orbitals, :, :, block_index))
        LAPACK.geqrf!(tall_scratch, tau)
        LAPACK.orgqr!(tall_scratch, tau, model.Ns ÷ 2)
        copyto!(view(right_orbitals, :, :, block_index + 1), tall_scratch)
    end

    _tU.get_G!(
        overlap,
        wide_scratch,
        pivots,
        view(left_orbitals, :, :, center_node),
        view(right_orbitals, :, :, center_node),
        physical.G,
    )
    return copy(physical.G)
end

"""
    tu_eswap_logratio(model, field, E_from, E_to, beta)

Return the physical two-flavor log weight ratio for a fixed-count proposal
that replaces one element of `E_from` by one element outside it.  The function
rebuilds the measured center Green matrix and then uses the rank-2 determinant
lemma.  It returns `-Inf` for a zero-weight proposal.
"""
function tu_eswap_logratio(model, field::Array{UInt8,2}, E_from, E_to, beta::Real)
    length(E_from) == length(E_to) ||
        throw(ArgumentError("E_from and E_to must have the same size"))
    _check_sites(E_from, model.Ns)
    _check_sites(E_to, model.Ns)

    removed = [site for site in E_from if site ∉ E_to]
    inserted = [site for site in E_to if site ∉ E_from]
    length(removed) == 1 && length(inserted) == 1 ||
        throw(ArgumentError("the proposal must exchange exactly one E site and one S site"))

    S_from = [site for site in 1:model.Ns if site ∉ E_from]
    j = findfirst(==(only(removed)), E_from)
    i = findfirst(==(only(inserted)), S_from)
    isnothing(i) && error("inserted site is not in the complement of E_from")
    isnothing(j) && error("removed site is not in E_from")

    G = measured_center_green(model, field, E_from, beta)
    buffer = ESwap2Buffer(G)
    ratio = prepare_eswap!(
        buffer,
        G,
        E_from,
        S_from,
        i,
        j,
        measurement_scale(beta),
    )
    return ratio.tu_logweight
end

end # module TUJointBorn
