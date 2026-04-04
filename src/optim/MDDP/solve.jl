# ──────────────────────────────────────────────────────────────────────
#  Outer solver for Multiple-Shooting DDP (Pellegrini & Russell)
#
#  Three nested loops:
#    1. Outer AL loop: update Lagrange multipliers and penalty μ
#    2. Node correction loop: Newton step on shooting-node states
#    3. Inner DDP loop (per leg): optimise controls on each leg
#
#  Reference: Pellegrini & Russell, Acta Astronautica 2020, §5.
# ──────────────────────────────────────────────────────────────────────

"""
    solve(prob::MDDPProblem, X0, U0, t, M; opts=MDDPOptions()) -> MDDPSolution

Solve a constrained optimal-control problem with multiple-shooting DDP.

# Arguments
- `prob`  – [`MDDPProblem`](@ref) specification
- `X0`    – initial state trajectory guess (`Vector` of `SVector`, length `N`)
- `U0`    – initial control guess (`Vector` of `SVector`, length `N-1`)
- `t`     – node times (`Vector`, length `N`)
- `M`     – number of multiple-shooting legs
- `opts`  – [`MDDPOptions`](@ref)

The trajectory is split into `M` approximately equal legs.  The initial
state of each leg (except the first, which is fixed) is a decision
variable; continuity between legs is enforced via the node-correction
step using value-function sensitivities and state-transition matrices.

# Returns
A [`MDDPSolution`](@ref) containing the optimised trajectory.

# Algorithm outline
1. Split trajectory into `M` legs
2. **Outer loop** (augmented-Lagrangian):
   a. Run DDP on each leg independently
   b. Correct shooting-node states via Newton step
   c. Update multipliers; increase penalty if needed
"""
function solve(prob::MDDPProblem, X0::Vector{SVector{nx,T}},
               U0::Vector{SVector{nu,T}},
               t::Vector{T}, M::Int;
               opts::MDDPOptions = MDDPOptions{T}()) where {nx, nu, T}

    N = length(X0)
    @assert length(U0) == N - 1
    @assert length(t) == N
    @assert M ≥ 1
    @assert prob.nx == nx
    @assert prob.nu == nu

    # ── Split into legs ─────────────────────────────────────────────
    legs = _split_into_legs(X0, U0, t, M)

    # ── Initialise multipliers (per-leg, using shared helpers) ──────
    λ_eq_legs   = [init_path_multipliers(prob.eq, length(leg.X), T)
                   for leg in legs]
    λ_ineq_legs = [init_path_multipliers(prob.ineq, length(leg.X), T)
                   for leg in legs]
    ν = init_terminal_multipliers(prob.terminal_eq, T)

    μ = opts.μ0
    total_iters = 0
    status = :max_iterations

    # ── Outer augmented-Lagrangian loop ─────────────────────────────
    for outer in 1:opts.max_outer
        total_iters += 1

        # ── Run DDP on each leg ─────────────────────────────────────
        Vx0s  = Vector{SVector{nx,T}}(undef, M)
        Vxx0s = Vector{SMatrix{nx,nx,T}}(undef, M)
        J_total = zero(T)

        for m in 1:M
            is_last = (m == M)
            Vx0, J_leg = leg_ddp!(legs[m], prob, λ_eq_legs[m], λ_ineq_legs[m],
                                  μ, opts, is_last, ν)
            Vx0s[m] = Vx0

            _, Vxx0 = _leg_value_gradient(
                prob, legs[m].X, legs[m].U, legs[m].t,
                λ_eq_legs[m], λ_ineq_legs[m], μ, opts.method, is_last, ν)
            Vxx0s[m] = Vxx0
            J_total += J_leg
        end

        # ── Node correction iterations ──────────────────────────────
        d_max = max_defect(legs)
        opts.verbose && @info "MDDP" outer d_max J_total μ

        for node_iter in 1:opts.max_node_iter
            d_max < opts.dtol && break

            node_correction!(legs, Vx0s, Vxx0s, prob, μ)
            d_max = max_defect(legs)

            # Re-run DDP on affected legs to update controls
            for m in 1:M
                is_last = (m == M)
                Vx0, _ = leg_ddp!(legs[m], prob, λ_eq_legs[m], λ_ineq_legs[m],
                                  μ, opts, is_last, ν)
                Vx0s[m] = Vx0
                _, Vxx0 = _leg_value_gradient(
                    prob, legs[m].X, legs[m].U, legs[m].t,
                    λ_eq_legs[m], λ_ineq_legs[m], μ, opts.method, is_last, ν)
                Vxx0s[m] = Vxx0
            end

            d_max = max_defect(legs)
            opts.verbose && @info "  Node correction" node_iter d_max
        end

        # ── Multiplier update & constraint check ────────────────────
        c_viol = _update_leg_multipliers!(prob, legs, λ_eq_legs, λ_ineq_legs, ν, μ)

        if c_viol < opts.ctol && d_max < opts.dtol
            status = :converged
            break
        end

        μ = min(μ * opts.ϕ_μ, opts.μ_max)
    end

    # ── Assemble full solution ──────────────────────────────────────
    X_full, U_full, t_full = _assemble_solution(legs)
    J_pure = _total_pure_cost(prob, legs)

    λ_eq_flat   = reduce(vcat, λ_eq_legs)
    λ_ineq_flat = reduce(vcat, λ_ineq_legs)

    return MDDPSolution(X_full, U_full, t_full, J_pure,
                        legs, λ_eq_flat, λ_ineq_flat, ν, μ,
                        total_iters, status)
end

# ──────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────

"""
    _split_into_legs(X0, U0, t, M) -> Vector{Leg}

Split a trajectory of `N` nodes into `M` approximately equal legs.
Each leg owns a contiguous slice of the state/control/time arrays.
"""
function _split_into_legs(X0, U0, t, M)
    N = length(X0)
    SX = eltype(X0)
    SU = eltype(U0)

    n_seg = N - 1
    legs = Vector{Leg{eltype(t), SX, SU}}(undef, M)

    for m in 1:M
        k_start = div((m - 1) * n_seg, M) + 1
        k_end   = div(m * n_seg, M) + 1
        legs[m] = Leg(X0[k_start:k_end], U0[k_start:k_end-1], t[k_start:k_end])
    end

    return legs
end

"""
    _assemble_solution(legs) -> (X, U, t)

Reassemble the full trajectory from legs, removing duplicate boundary
nodes between consecutive legs.
"""
function _assemble_solution(legs)
    M = length(legs)
    X = copy(legs[1].X)
    U = copy(legs[1].U)
    t = copy(legs[1].t)

    for m in 2:M
        append!(X, legs[m].X[2:end])
        append!(U, legs[m].U)
        append!(t, legs[m].t[2:end])
    end

    return X, U, t
end

"""
    _update_leg_multipliers!(prob, legs, λ_eq_legs, λ_ineq_legs, ν, μ) -> c_viol

Hestenes-Powell multiplier update across all legs.  Returns the
maximum constraint violation.
"""
function _update_leg_multipliers!(prob, legs, λ_eq_legs, λ_ineq_legs, ν, μ)
    T = eltype(legs[1].X[1])
    c_viol = zero(T)

    for m in eachindex(legs)
        X, U, t = legs[m].X, legs[m].U, legs[m].t
        N = length(X)

        if prob.eq !== nothing
            for k in 1:(N-1)
                gval = prob.eq.g(X[k], U[k], t[k])
                for j in eachindex(λ_eq_legs[m][k])
                    λ_eq_legs[m][k][j] += μ * gval[j]
                end
                c_viol = max(c_viol, maximum(abs, gval))
            end
        end

        if prob.ineq !== nothing
            for k in 1:(N-1)
                hval = prob.ineq.h(X[k], U[k], t[k])
                for j in eachindex(λ_ineq_legs[m][k])
                    λ_ineq_legs[m][k][j] = max(zero(T), λ_ineq_legs[m][k][j] - μ * hval[j])
                end
                for j in eachindex(hval)
                    c_viol = max(c_viol, max(zero(T), -hval[j]))
                end
            end
        end
    end

    # Terminal constraints (last leg only)
    if prob.terminal_eq !== nothing
        X_last = legs[end].X
        ψval = prob.terminal_eq.ψ(X_last[end])
        for j in eachindex(ν)
            ν[j] += μ * ψval[j]
        end
        c_viol = max(c_viol, maximum(abs, ψval))
    end

    return c_viol
end

"""
    _total_pure_cost(prob, legs)

Sum of un-augmented costs across all legs.
"""
function _total_pure_cost(prob, legs)
    T = eltype(legs[1].X[1])
    J = zero(T)
    for m in eachindex(legs)
        J += _leg_pure_cost(prob, legs[m].X, legs[m].U, legs[m].t, m == length(legs))
    end
    return J
end
