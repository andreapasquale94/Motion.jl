# ──────────────────────────────────────────────────────────────────────
#  Outer solver for Multiple-Shooting DDP (Pellegrini & Russell)
#
#  Three nested loops:
#    1. Outer AL loop: update Lagrange multipliers and penalty μ
#    2. Node correction loop: update shooting-node initial conditions
#    3. Inner DDP loop (per leg): optimise controls on each leg
#
#  Reference: Pellegrini & Russell, Acta Astronautica 2020, §5.
# ──────────────────────────────────────────────────────────────────────

using ..DDP: EqualityConstraint, InequalityConstraint, TerminalConstraint

"""
    solve(prob::MDDPProblem, X0, U0, t, M; opts=MDDPOptions()) -> MDDPSolution

Solve a constrained optimal-control problem with multiple-shooting DDP.

# Arguments
- `prob`  – problem definition
- `X0`    – initial state trajectory guess (Vector of SVectors, length N)
- `U0`    – initial control guess (Vector of SVectors, length N-1)
- `t`     – node times (Vector, length N)
- `M`     – number of multiple-shooting legs
- `opts`  – algorithmic options

The trajectory is split into `M` approximately equal legs.  The initial
state of each leg is a decision variable; continuity between legs is
enforced via the node-correction step.

# Returns
A [`MDDPSolution`](@ref) containing the optimised trajectory.
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

    # ── Initialise multipliers (per-leg, per-node) ──────────────────
    λ_eq_legs   = [_init_leg_multipliers(prob.eq,   length(leg.X), T)
                   for leg in legs]
    λ_ineq_legs = [_init_leg_multipliers(prob.ineq, length(leg.X), T)
                   for leg in legs]
    ν = _init_terminal_mult(prob.terminal_eq, T)

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

            # Also get Vxx0 for node correction
            _, Vxx0 = _leg_value_gradient(
                prob, legs[m].X, legs[m].U, legs[m].t,
                λ_eq_legs[m], λ_ineq_legs[m], μ, opts.method, is_last, ν)
            Vxx0s[m] = Vxx0
            J_total += J_leg
        end

        # ── Node correction iterations ──────────────────────────────
        d_max = max_defect(legs)
        if opts.verbose
            @info "MDDP" outer d_max J_total μ
        end

        for node_iter in 1:opts.max_node_iter
            if d_max < opts.dtol
                break
            end

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
            if opts.verbose
                @info "  Node correction" node_iter d_max
            end
        end

        # ── Multiplier update & constraint check ────────────────────
        c_viol = _update_leg_multipliers!(prob, legs, λ_eq_legs, λ_ineq_legs, ν, μ)

        # Combined convergence: constraints + defects
        if c_viol < opts.ctol && d_max < opts.dtol
            status = :converged
            break
        end

        μ = min(μ * opts.ϕ_μ, opts.μ_max)
    end

    # ── Assemble full solution ──────────────────────────────────────
    X_full, U_full, t_full = _assemble_solution(legs)
    J_pure = _total_pure_cost(prob, legs)

    # Flatten per-leg multipliers
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

Split a trajectory into M approximately equal legs.
"""
function _split_into_legs(X0, U0, t, M)
    N = length(X0)
    T = eltype(X0[1])
    SX = eltype(X0)
    SU = eltype(U0)

    # Compute split points (N-1 segments split into M legs)
    n_seg = N - 1
    legs = Vector{Leg{eltype(t), SX, SU}}(undef, M)

    for m in 1:M
        k_start = div((m - 1) * n_seg, M) + 1
        k_end   = div(m * n_seg, M) + 1

        legs[m] = Leg(
            X0[k_start:k_end],
            U0[k_start:k_end-1],
            t[k_start:k_end]
        )
    end

    return legs
end

"""
    _assemble_solution(legs) -> (X, U, t)

Reassemble the full trajectory from legs, removing duplicate boundary nodes.
"""
function _assemble_solution(legs)
    M = length(legs)
    X = copy(legs[1].X)
    U = copy(legs[1].U)
    t = copy(legs[1].t)

    for m in 2:M
        # Skip the first node of subsequent legs (it overlaps with previous leg's last)
        append!(X, legs[m].X[2:end])
        append!(U, legs[m].U)
        append!(t, legs[m].t[2:end])
    end

    return X, U, t
end

function _init_leg_multipliers(::Nothing, N, ::Type{T}) where T
    return [T[] for _ in 1:(N-1)]
end

function _init_leg_multipliers(c::EqualityConstraint, N, ::Type{T}) where T
    return [zeros(T, c.p) for _ in 1:(N-1)]
end

function _init_leg_multipliers(c::InequalityConstraint, N, ::Type{T}) where T
    return [zeros(T, c.q) for _ in 1:(N-1)]
end

function _init_terminal_mult(::Nothing, ::Type{T}) where T
    return T[]
end

function _init_terminal_mult(c::TerminalConstraint, ::Type{T}) where T
    return zeros(T, c.r)
end

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

    # Terminal
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

function _total_pure_cost(prob, legs)
    T = eltype(legs[1].X[1])
    J = zero(T)
    for m in eachindex(legs)
        is_last = (m == length(legs))
        J += _leg_pure_cost(prob, legs[m].X, legs[m].U, legs[m].t, is_last)
    end
    return J
end
