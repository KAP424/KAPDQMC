@inline _is_mutable_value(value) = Base.ismutabletype(typeof(value))

"""
State of the joint Born sampler.

The target distribution is

    Π(E, s1, s2) ∝ μ(s1) F(E, s1) μ(s2) F(E, s2) / Z(E),

where `Z(E) = sum_s μ(s) F(E, s)`.

`replica1` and `replica2` must own independent mutable storage.  The
constructor rejects the simplest case in which both fields are the same
mutable object, but it cannot detect distinct views or wrappers that share a
backing array.
"""
mutable struct JointBornState{E,S}
    outcome::E
    replica1::S
    replica2::S

    function JointBornState(outcome::E, replica1::S, replica2::S) where {E,S}
        if _is_mutable_value(replica1) && replica1 === replica2
            throw(ArgumentError("the two replicas must not alias the same mutable object"))
        end
        return new{E,S}(outcome, replica1, replica2)
    end
end

"""State for the scalable outer Born chain `rho(E,s) proportional to mu(s)F_E(s)`."""
mutable struct BornOutcomeState{E,S}
    outcome::E
    field::S
end

"""
Extended state for the exact persistent-exchange construction.

`auxiliary[H]` stores one persistent conditional configuration for every
allowed outcome `H` except `outcome`.  Keeping those configurations in the
Markov state removes the perfect-draw requirement of the ordinary exchange
algorithm, at the cost of a bank whose logical size is the number of outcomes.
Every mutable replica and bank sample must own independent backing storage;
distinct views of the same array are not safe.
"""
mutable struct PersistentExchangeState{E,S,B}
    outcome::E
    replica1::S
    replica2::S
    auxiliary::B

    function PersistentExchangeState(
        outcome::E,
        replica1::S,
        replica2::S,
        auxiliary::B,
    ) where {E,S,B}
        _is_mutable_value(outcome) &&
            throw(ArgumentError("persistent outcome labels must be immutable; use a sorted Tuple"))
        if _is_mutable_value(replica1) && replica1 === replica2
            throw(ArgumentError("the two replicas must not alias the same mutable object"))
        end

        mutable_samples = Any[]
        _is_mutable_value(replica1) && push!(mutable_samples, replica1)
        _is_mutable_value(replica2) && push!(mutable_samples, replica2)
        for (label, sample) in auxiliary
            _is_mutable_value(label) &&
                throw(ArgumentError("persistent bank labels must be immutable; use sorted Tuples"))
            isequal(label, outcome) &&
                throw(ArgumentError("the current outcome must not have an auxiliary bank entry"))
            if _is_mutable_value(sample)
                any(existing -> existing === sample, mutable_samples) &&
                    throw(ArgumentError("replicas and bank entries must be distinct mutable objects"))
                push!(mutable_samples, sample)
            end
        end
        return new{E,S,B}(outcome, replica1, replica2, auxiliary)
    end
end

Base.@kwdef mutable struct JointBornStats
    outcome_proposals::Int = 0
    outcome_accepts::Int = 0
    conditional_updates::Int = 0
    auxiliary_draws::Int = 0
end

outcome_acceptance(stats::JointBornStats) =
    stats.outcome_proposals == 0 ? 0.0 : stats.outcome_accepts / stats.outcome_proposals

@inline function _accept_logratio(rng::AbstractRNG, logratio::Real)
    isnan(logratio) && throw(ArgumentError("Metropolis log-ratio is NaN"))
    return logratio >= 0 || log(rand(rng)) < logratio
end

@inline _no_outcome_commit!(replica, current, proposed) = replica

"""
    exchange_outcome_ratio_step!(rng, state, propose_outcome,
                                 log_measure_ratio, draw_conditional;
                                 commit_outcome! = _no_outcome_commit!,
                                 stats=nothing)

Numerically stable form of the exact exchange update.  The callback
`log_measure_ratio(Efrom, Eto, s)` returns

    log(F(Eto, s) / F(Efrom, s)).

It can therefore use a determinant lemma or a cached Green-function update
without constructing either determinant separately.  `draw_conditional` must
still be an **independent exact** draw from `mu_E`, conditional only on its
outcome argument.  If a replica stores matrices tied to
the current outcome, `commit_outcome!(replica, Efrom, Eto)` must update those
caches after an accepted move.  It is called once for each main replica and
must not mutate a shared outcome or membership list; `state.outcome` is
committed once by this function.  The default callback is a no-op.
"""
function exchange_outcome_ratio_step!(
    rng::AbstractRNG,
    state::JointBornState,
    propose_outcome,
    log_measure_ratio,
    draw_conditional;
    commit_outcome! = _no_outcome_commit!,
    stats::Union{Nothing,JointBornStats}=nothing,
)
    current = state.outcome
    proposed, log_q_forward, log_q_reverse = propose_outcome(rng, current)
    if isequal(proposed, current)
        if stats !== nothing
            stats.outcome_proposals += 1
            stats.outcome_accepts += 1
        end
        return true, 0.0
    end
    auxiliary = draw_conditional(rng, proposed)

    logratio =
        log_measure_ratio(current, proposed, state.replica1) +
        log_measure_ratio(current, proposed, state.replica2) +
        log_measure_ratio(proposed, current, auxiliary) +
        log_q_reverse - log_q_forward

    if stats !== nothing
        stats.outcome_proposals += 1
        stats.auxiliary_draws += 1
    end

    accepted = _accept_logratio(rng, logratio)
    if accepted
        commit_outcome!(state.replica1, current, proposed)
        commit_outcome!(state.replica2, current, proposed)
        state.outcome = proposed
        stats !== nothing && (stats.outcome_accepts += 1)
    end
    return accepted, logratio
end

"""
    exchange_outcome_step!(rng, state, propose_outcome, log_measure_weight,
                           draw_conditional;
                           commit_outcome! = _no_outcome_commit!,
                           stats=nothing)

Perform one exact exchange-algorithm update of `state.outcome`.

Callbacks:

* `propose_outcome(rng, E)` returns `(Enew, log_q_forward, log_q_reverse)`.
* `log_measure_weight(E, s)` returns `log(F(E,s))`.
* `draw_conditional(rng, E)` must return an **independent exact** draw from
  `μ_E(s) = μ(s)F(E,s)/Z(E)`.

For a symmetric outcome proposal, return zero for both proposal logs.  The
unknown ratio `Z(E)/Z(Enew)` is cancelled by the auxiliary conditional draw.
If `draw_conditional` is replaced by a finite inner MCMC run, this becomes the
approximate double-Metropolis-Hastings algorithm and is no longer exact.
"""
function exchange_outcome_step!(
    rng::AbstractRNG,
    state::JointBornState,
    propose_outcome,
    log_measure_weight,
    draw_conditional;
    commit_outcome! = _no_outcome_commit!,
    stats::Union{Nothing,JointBornStats}=nothing,
)
    log_measure_ratio = (from, to, sample) ->
        log_measure_weight(to, sample) - log_measure_weight(from, sample)
    return exchange_outcome_ratio_step!(
        rng,
        state,
        propose_outcome,
        log_measure_ratio,
        draw_conditional;
        commit_outcome! = commit_outcome!,
        stats=stats,
    )
end

"""
    double_metropolis_outcome_step!(rng, state, propose_outcome,
                                    log_measure_ratio,
                                    initialize_auxiliary,
                                    conditional_kernel;
                                    inner_steps,
                                    commit_outcome! = _no_outcome_commit!,
                                    stats=nothing)

Approximate an exchange update by starting an auxiliary state with
`initialize_auxiliary(rng, E)` and applying `inner_steps` transitions of
`conditional_kernel(rng, s, E)`.  This is the double-Metropolis-Hastings
algorithm.  It is exact only in the limit in which the returned auxiliary
state is distributed exactly as `mu_E`; finite `inner_steps` generally leaves
bias and must be checked by an inner-length convergence study.
"""
function double_metropolis_outcome_step!(
    rng::AbstractRNG,
    state::JointBornState,
    propose_outcome,
    log_measure_ratio,
    initialize_auxiliary,
    conditional_kernel;
    inner_steps::Integer,
    commit_outcome! = _no_outcome_commit!,
    stats::Union{Nothing,JointBornStats}=nothing,
)
    inner_steps >= 0 || throw(ArgumentError("inner_steps must be nonnegative"))
    draw_approximately = function (rng, outcome)
        auxiliary = initialize_auxiliary(rng, outcome)
        for _ in 1:inner_steps
            auxiliary = conditional_kernel(rng, auxiliary, outcome)
            stats !== nothing && (stats.conditional_updates += 1)
        end
        return auxiliary
    end
    return exchange_outcome_ratio_step!(
        rng,
        state,
        propose_outcome,
        log_measure_ratio,
        draw_approximately;
        commit_outcome! = commit_outcome!,
        stats=stats,
    )
end

"""
    born_outcome_step!(rng, state, propose_outcome, log_measure_ratio;
                       commit_outcome! = _no_outcome_commit!, stats=nothing)

Update `E` at fixed auxiliary field for the outer target

    rho(E,s) proportional to mu(s) F_E(s).

Its `E` marginal is exactly the Born distribution, and no `Z_E` ratio appears.
This is the scalable outer chain used by the nested strategy; it is not by
itself a sampler of the two-replica joint target.
"""
function born_outcome_step!(
    rng::AbstractRNG,
    state::BornOutcomeState,
    propose_outcome,
    log_measure_ratio;
    commit_outcome! = _no_outcome_commit!,
    stats::Union{Nothing,JointBornStats}=nothing,
)
    current = state.outcome
    proposed, log_q_forward, log_q_reverse = propose_outcome(rng, current)
    if isequal(proposed, current)
        if stats !== nothing
            stats.outcome_proposals += 1
            stats.outcome_accepts += 1
        end
        return true, 0.0
    end
    logratio =
        log_measure_ratio(current, proposed, state.field) +
        log_q_reverse - log_q_forward

    stats !== nothing && (stats.outcome_proposals += 1)
    accepted = _accept_logratio(rng, logratio)
    if accepted
        commit_outcome!(state.field, current, proposed)
        state.outcome = proposed
        stats !== nothing && (stats.outcome_accepts += 1)
    end
    return accepted, logratio
end

"""Advance the auxiliary field of an outer Born state at fixed outcome."""
function update_born_field!(
    rng::AbstractRNG,
    state::BornOutcomeState,
    conditional_kernel;
    steps::Integer=1,
    stats::Union{Nothing,JointBornStats}=nothing,
)
    steps >= 0 || throw(ArgumentError("steps must be nonnegative"))
    for _ in 1:steps
        state.field = conditional_kernel(rng, state.field, state.outcome)
        stats !== nothing && (stats.conditional_updates += 1)
    end
    return state
end

"""Run fixed-outcome field updates followed by one outer Born outcome move."""
function born_outcome_sweep!(
    rng::AbstractRNG,
    state::BornOutcomeState,
    propose_outcome,
    log_measure_ratio,
    conditional_kernel;
    field_steps::Integer=1,
    commit_outcome! = _no_outcome_commit!,
    stats::Union{Nothing,JointBornStats}=nothing,
)
    update_born_field!(rng, state, conditional_kernel; steps=field_steps, stats=stats)
    return born_outcome_step!(
        rng,
        state,
        propose_outcome,
        log_measure_ratio;
        commit_outcome! = commit_outcome!,
        stats=stats,
    )
end

"""
    persistent_exchange_step!(rng, state, propose_outcome,
                              log_measure_ratio;
                              commit_outcome! = _no_outcome_commit!,
                              stats=nothing)

Perform an exact outcome update on `PersistentExchangeState` without an exact
conditional draw.  If `H` is proposed, its stored persistent configuration
`u_H` enters the acceptance ratio

    F_H(s1) F_H(s2) F_E(u_H)
    ------------------------- * q(E|H) / q(H|E).
    F_E(s1) F_E(s2) F_H(u_H)

On acceptance, `u_H` is relabelled as the auxiliary configuration for the old
outcome `E`.  The update is ordinary Metropolis-Hastings for the extended
target

    Pi(E,s1,s2) * product(mu_H(u_H), H != E),

whose unknown normalizers multiply to `product_H Z_H^(-1)`, independent of
the current main outcome.  Update bank entries with
`update_persistent_auxiliary!` as separate Markov steps.  Those updates may be
finite without the double-MH stationary bias.  The bank must contain every
outcome that the proposal can reach; omitting labels restricts the sampled
outcome space.
"""
function persistent_exchange_step!(
    rng::AbstractRNG,
    state::PersistentExchangeState,
    propose_outcome,
    log_measure_ratio,
    ;
    commit_outcome! = _no_outcome_commit!,
    stats::Union{Nothing,JointBornStats}=nothing,
)
    current = state.outcome
    proposed, log_q_forward, log_q_reverse = propose_outcome(rng, current)
    if isequal(proposed, current)
        if stats !== nothing
            stats.outcome_proposals += 1
            stats.outcome_accepts += 1
        end
        return true, 0.0
    end
    haskey(state.auxiliary, proposed) ||
        throw(KeyError("no persistent auxiliary state for proposed outcome $proposed"))

    auxiliary = state.auxiliary[proposed]

    logratio =
        log_measure_ratio(current, proposed, state.replica1) +
        log_measure_ratio(current, proposed, state.replica2) +
        log_measure_ratio(proposed, current, auxiliary) +
        log_q_reverse - log_q_forward

    stats !== nothing && (stats.outcome_proposals += 1)
    accepted = _accept_logratio(rng, logratio)
    if accepted
        commit_outcome!(state.replica1, current, proposed)
        commit_outcome!(state.replica2, current, proposed)
        commit_outcome!(auxiliary, proposed, current)
        delete!(state.auxiliary, proposed)
        state.auxiliary[current] = auxiliary
        state.outcome = proposed
        stats !== nothing && (stats.outcome_accepts += 1)
    end
    return accepted, logratio
end

"""
    update_persistent_auxiliary!(rng, state, outcome, conditional_kernel;
                                 steps=1, stats=nothing)

Advance one stored bank configuration with a kernel preserving
`mu_outcome`.  This is a separate conditional Markov step and therefore may
use any finite nonnegative `steps`.  `outcome` cannot be the current main
label because that label has no bank entry.
"""
function update_persistent_auxiliary!(
    rng::AbstractRNG,
    state::PersistentExchangeState,
    outcome,
    conditional_kernel;
    steps::Integer=1,
    stats::Union{Nothing,JointBornStats}=nothing,
)
    steps >= 0 || throw(ArgumentError("steps must be nonnegative"))
    haskey(state.auxiliary, outcome) ||
        throw(KeyError("no persistent auxiliary state for outcome $outcome"))
    auxiliary = state.auxiliary[outcome]
    for _ in 1:steps
        auxiliary = conditional_kernel(rng, auxiliary, outcome)
        stats !== nothing && (stats.conditional_updates += 1)
    end
    state.auxiliary[outcome] = auxiliary
    return auxiliary
end

"""
    update_all_persistent_auxiliaries!(rng, state, conditional_kernel;
                                       steps=1, stats=nothing)

Advance every stored conditional configuration.  This is exact but costs one
conditional update per outcome and is intended mainly for small-system
validation.  Large calculations can random-scan bank entries with
`update_persistent_auxiliary!` instead.
"""
function update_all_persistent_auxiliaries!(
    rng::AbstractRNG,
    state::PersistentExchangeState,
    conditional_kernel;
    steps::Integer=1,
    stats::Union{Nothing,JointBornStats}=nothing,
)
    for outcome in collect(keys(state.auxiliary))
        update_persistent_auxiliary!(
            rng,
            state,
            outcome,
            conditional_kernel;
            steps=steps,
            stats=stats,
        )
    end
    return state
end

"""
    update_persistent_replicas!(rng, state, conditional_kernel;
                                steps=1, stats=nothing)

Update the two main replicas of a persistent-exchange state at fixed outcome.
Each transition only needs to preserve `mu_E`; it need not return an
independent draw.
"""
function update_persistent_replicas!(
    rng::AbstractRNG,
    state::PersistentExchangeState,
    conditional_kernel;
    steps::Integer=1,
    stats::Union{Nothing,JointBornStats}=nothing,
)
    steps >= 0 || throw(ArgumentError("steps must be nonnegative"))
    for _ in 1:steps
        state.replica1 = conditional_kernel(rng, state.replica1, state.outcome)
        state.replica2 = conditional_kernel(rng, state.replica2, state.outcome)
        stats !== nothing && (stats.conditional_updates += 2)
    end
    return state
end

"""
    update_replicas!(rng, state, conditional_kernel; steps=1, stats=nothing)

Update both replicas at fixed outcome.  `conditional_kernel(rng, s, E)` must
return a state obtained from a Markov kernel whose invariant distribution is
`μ_E`.  It may be an ordinary conditioned DQMC sweep; it need not be an
independent draw.
"""
function update_replicas!(
    rng::AbstractRNG,
    state::JointBornState,
    conditional_kernel;
    steps::Integer=1,
    stats::Union{Nothing,JointBornStats}=nothing,
)
    steps >= 0 || throw(ArgumentError("steps must be nonnegative"))
    for _ in 1:steps
        state.replica1 = conditional_kernel(rng, state.replica1, state.outcome)
        state.replica2 = conditional_kernel(rng, state.replica2, state.outcome)
        stats !== nothing && (stats.conditional_updates += 2)
    end
    return state
end

"""
    exchange_sweep!(rng, state, propose_outcome, log_measure_weight,
                    draw_conditional, conditional_kernel;
                    replica_steps=1,
                    commit_outcome! = _no_outcome_commit!,
                    stats=nothing)

One sweep of the exact joint sampler: update the two conditional replicas and
then update the outcome with the exchange algorithm.
"""
function exchange_sweep!(
    rng::AbstractRNG,
    state::JointBornState,
    propose_outcome,
    log_measure_weight,
    draw_conditional,
    conditional_kernel;
    replica_steps::Integer=1,
    commit_outcome! = _no_outcome_commit!,
    stats::Union{Nothing,JointBornStats}=nothing,
)
    update_replicas!(rng, state, conditional_kernel; steps=replica_steps, stats=stats)
    return exchange_outcome_step!(
        rng,
        state,
        propose_outcome,
        log_measure_weight,
        draw_conditional;
        commit_outcome! = commit_outcome!,
        stats=stats,
    )
end

"""
    hierarchical_refresh_step!(rng, state, propose_outcome,
                               log_measure_weight, draw_conditional;
                               stats=nothing)

Exact hierarchical alternative to the exchange update.

First update `(E, replica1)` with target `μ(s)F(E,s)`, whose `E` marginal is
the Born distribution.  Then independently redraw both replicas from `μ_E`.
The two redraws must be exact.  Replacing them by a finite number of DQMC
sweeps gives a nested approximation whose bias has to be checked by increasing
the number of inner sweeps.
"""
function hierarchical_refresh_step!(
    rng::AbstractRNG,
    state::JointBornState,
    propose_outcome,
    log_measure_weight,
    draw_conditional;
    stats::Union{Nothing,JointBornStats}=nothing,
)
    current = state.outcome
    proposed, log_q_forward, log_q_reverse = propose_outcome(rng, current)
    self_proposal = isequal(proposed, current)
    logratio = self_proposal ? 0.0 :
        log_measure_weight(proposed, state.replica1) -
        log_measure_weight(current, state.replica1) +
        log_q_reverse - log_q_forward

    stats !== nothing && (stats.outcome_proposals += 1)
    accepted = self_proposal || _accept_logratio(rng, logratio)
    if accepted
        state.outcome = proposed
        stats !== nothing && (stats.outcome_accepts += 1)
    end

    state.replica1 = draw_conditional(rng, state.outcome)
    state.replica2 = draw_conditional(rng, state.outcome)
    if stats !== nothing
        stats.auxiliary_draws += 2
        stats.conditional_updates += 2
    end
    return accepted, logratio
end

"""
    sequential_born_outcome!(rng, sequential_state, sites,
                             local_weights, apply_result!)

Generate one complete binary measurement record by the chain rule used in a
one-site-at-a-time Born measurement.  For the current prefix,
`local_weights(sequential_state, site)` returns the two nonnegative real
weights `(weight_S, weight_E)`.  The function normalizes them, samples whether
`site` belongs to `S` or `E`, and calls
`apply_result!(sequential_state, site, in_E)` before visiting the next site.

The callback must evaluate physical conditional Born weights.  Transition
Green functions of individual auxiliary-field configurations can be complex
or non-positive; such values are rejected here instead of silently being used
as probabilities.

Returns `(E, log_path_probability, sequential_state)`.
"""
function sequential_born_outcome!(
    rng::AbstractRNG,
    sequential_state,
    sites,
    local_weights,
    apply_result!,
)
    outcome = eltype(sites)[]
    log_path_probability = 0.0

    for site in sites
        weight_S, weight_E = local_weights(sequential_state, site)
        weight_S isa Real && weight_E isa Real ||
            throw(ArgumentError("local Born weights must be real"))
        isfinite(weight_S) && isfinite(weight_E) ||
            throw(ArgumentError("local Born weights must be finite"))
        weight_S >= 0 && weight_E >= 0 ||
            throw(ArgumentError("local Born weights must be nonnegative"))

        scale = max(weight_S, weight_E)
        scale > 0 ||
            throw(ArgumentError("at least one local Born weight must be positive"))
        scaled_S = weight_S / scale
        scaled_E = weight_E / scale
        normalization = scaled_S + scaled_E
        normalization > 0 ||
            throw(ArgumentError("at least one local Born weight must be positive"))
        probability_E = scaled_E / normalization
        in_E = rand(rng) < probability_E
        chosen_probability = in_E ? probability_E : 1 - probability_E
        log_path_probability += log(chosen_probability)
        in_E && push!(outcome, site)
        apply_result!(sequential_state, site, in_E)
    end

    return outcome, log_path_probability, sequential_state
end

"""
    fixed_count_sequential_born(rng, initialize_state, sites,
                                local_weights, apply_result!, count;
                                max_attempts=typemax(Int))

Condition a complete sequential Born record on `length(E) == count` by exact
rejection sampling.  `initialize_state(rng)` must return a fresh copy of the
unmeasured state for every attempt.  The result is
`(E, sequential_state, attempts)`.
"""
function fixed_count_sequential_born(
    rng::AbstractRNG,
    initialize_state,
    sites,
    local_weights,
    apply_result!,
    count::Integer;
    max_attempts::Integer=typemax(Int),
)
    0 <= count <= length(sites) || throw(ArgumentError("invalid fixed count"))
    max_attempts > 0 || throw(ArgumentError("max_attempts must be positive"))

    for attempt in 1:max_attempts
        sequential_state = initialize_state(rng)
        outcome, _, sequential_state = sequential_born_outcome!(
            rng,
            sequential_state,
            sites,
            local_weights,
            apply_result!,
        )
        length(outcome) == count && return outcome, sequential_state, attempt
    end
    error("fixed-count sequential Born sampler exceeded max_attempts")
end

"""
    balanced_outcome_acceptance(a, beta)

Exact probability that the binary occupation POVM described in
`相互作用测量方案.md` produces `|E| = a` on `2a` sites in the single-flavor,
fixed-particle-number sector.  The evaluation uses logarithms to avoid
overflow of the binomial coefficients.
"""
function balanced_outcome_acceptance(a::Integer, beta::Real)
    a >= 0 || throw(ArgumentError("a must be nonnegative"))
    beta >= 0 || throw(ArgumentError("beta must be nonnegative"))

    log_terms = Vector{Float64}(undef, a + 1)
    log_binomial = 0.0
    for k in 0:a
        if k > 0
            log_binomial += log(a - k + 1) - log(k)
        end
        log_terms[k + 1] = 2log_binomial - 4beta * k
    end
    largest = maximum(log_terms)
    log_sum = largest + log(sum(exp(value - largest) for value in log_terms))
    return exp(log_sum - 2a * log1p(exp(-2beta)))
end

"""
    collision_joint_draw(rng, draw_outer; same_outcome=(==))

An exact, normalization-free construction of one `JointBornState` when
`draw_outer(rng)` supplies **independent exact** pairs `(E, s)` from

    rho(E, s) = P_B(E) * mu_E(s).

Draw `(E, s1)` once, then draw independent outer pairs until their outcome
equals `E`; the matching sample becomes `s2`.  Conditional on the collision,
`s1` and `s2` are independent draws from `mu_E`, so the output has the desired
joint law.  Its expected collision count equals the number of outcomes with
nonzero Born probability and is generally exponential in system size.  MCMC
states from one correlated outer chain do not satisfy the independence
requirement.

Returns `(state, attempts)`.
"""
function collision_joint_draw(
    rng::AbstractRNG,
    draw_outer;
    same_outcome=(==),
)
    outcome, replica1 = draw_outer(rng)
    attempt = 0
    while true
        attempt += 1
        candidate_outcome, replica2 = draw_outer(rng)
        if same_outcome(candidate_outcome, outcome)
            return JointBornState(outcome, replica1, replica2), attempt
        end
    end
end

"""
    propose_fixed_count_swap(rng, E, number_of_sites)

Symmetrically choose the `i`th element `S[i]` of the complement and the `j`th
element `E[j]`, then exchange them.  The returned tuple is compatible with the
joint-sampler proposal callback:

    (E_new, log_q_forward, log_q_reverse).

Both proposal logs are zero because every state has
`length(E) * length(S)` ordered swaps.  `E` must be sorted, and the proposed
subset is sorted as well, so every physical subset has one representation.
"""
function propose_fixed_count_swap(rng::AbstractRNG, E, number_of_sites::Integer)
    0 < length(E) < number_of_sites ||
        throw(ArgumentError("E and S must both be nonempty"))
    length(unique(E)) == length(E) || throw(ArgumentError("E contains duplicates"))
    issorted(E) || throw(ArgumentError("E must use sorted canonical order"))
    all(site -> 1 <= site <= number_of_sites, E) ||
        throw(BoundsError(1:number_of_sites, E))

    S = [site for site in 1:number_of_sites if site ∉ E]
    i = rand(rng, eachindex(S))
    j = rand(rng, eachindex(E))
    proposed_values = collect(E)
    proposed_values[j] = S[i]
    sort!(proposed_values)
    proposed = E isa Tuple ? Tuple(proposed_values) : proposed_values
    return proposed, 0.0, 0.0
end
