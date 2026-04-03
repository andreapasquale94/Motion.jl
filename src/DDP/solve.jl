# ──────────────────────────────────────────────────────────────────────
#  Outer solver – augmented-Lagrangian DDP with multiplier updates
#
#  Two-loop structure:
#    Outer loop: update Lagrange multipliers λ, ν and penalty μ
#    Inner loop: standard DDP (backward + forward) at fixed (λ, ν, μ)
#
#  Reference: Lantoine & Russell, JOTA 2012, §6.
# ──────────────────────────────────────────────────────────────────────

"""
    solve(prob::DDPProblem, X0, U0, t; opts=DDPOptions()) -> DDPSolution

Solve a constrained optimal-control problem with differential dynamic
programming using an augmented-Lagrangian outer loop.

# Arguments
- `prob`  – problem definition (dynamics, costs, constraints)
- `X0`    – initial state trajectory guess (Vector of SVectors, length N)
- `U0`    – initial control guess (Vector of SVectors, length N-1)
- `t`     – node times (Vector, length N)
- `opts`  – algorithmic options

# Returns
A [`DDPSolution`](@ref) containing the optimised trajectory, controls,
multipliers and convergence information.
"""
function solve(prob::DDPProblem, X0::Vector{SVector{nx,T}},
               U0::Vector{SVector{nu,T}},
               t::Vector{T};
               opts::DDPOptions = DDPOptions{T}()) where {nx, nu, T}

    N = length(X0)
    @assert length(U0) == N - 1
    @assert length(t) == N
    @assert prob.nx == nx
    @assert prob.nu == nu

    # ── Initialise multipliers ───────────────────────────────────────
    λ_eq   = _init_path_multipliers(prob.eq,   N, T)
    λ_ineq = _init_path_multipliers(prob.ineq, N, T)
    ν      = _init_terminal_multipliers(prob.terminal_eq, T)

    μ   = opts.μ0
    reg = opts.reg0

    X = copy(X0)
    U = copy(U0)
    J = _eval_total_cost(prob, X, U, t, λ_eq, λ_ineq, ν, μ)

    total_iters = 0
    status = :max_iterations

    # ── Outer augmented-Lagrangian loop ──────────────────────────────
    for outer in 1:opts.max_outer

        # ── Inner DDP loop ───────────────────────────────────────────
        for iter in 1:opts.max_iter
            total_iters += 1

            # Backward pass
            result, ΔJ1, ΔJ2, ok = backward_pass(
                prob, X, U, t, λ_eq, λ_ineq, ν, μ, reg)

            if !ok
                # Increase regularisation and retry
                reg = (reg <= zero(T)) ? opts.reg_min : reg * opts.reg_factor
                if reg > opts.reg_max
                    status = :regularisation_failure
                    @goto done
                end
                continue
            end

            gains_l, gains_L = result

            # Forward pass (line search)
            X_new, U_new, J_new, α = forward_pass(
                prob, X, U, t, gains_l, gains_L, J, ΔJ1, ΔJ2,
                λ_eq, λ_ineq, ν, μ, opts)

            if α == zero(T)
                # Line search failed – increase regularisation
                reg = (reg <= zero(T)) ? opts.reg_min : reg * opts.reg_factor
                if reg > opts.reg_max
                    status = :line_search_failure
                    @goto done
                end
                continue
            end

            # Accept step – decrease regularisation
            ΔJ = J - J_new
            J = J_new
            X = X_new
            U = U_new
            reg = max(reg / opts.reg_factor, opts.reg_min)

            if opts.verbose
                @info "DDP" outer iter α J ΔJ reg
            end

            # Convergence check (inner)
            if abs(ΔJ) < opts.atol || abs(ΔJ) / (abs(J) + T(1e-16)) < opts.rtol
                break
            end
        end

        # ── Multiplier update & constraint check ────────────────────
        c_viol = _update_multipliers!(prob, X, U, t, λ_eq, λ_ineq, ν, μ)

        if opts.verbose
            @info "AL outer" outer μ c_viol
        end

        if c_viol < opts.ctol
            status = :converged
            @goto done
        end

        # Increase penalty
        μ = min(μ * opts.ϕ_μ, opts.μ_max)

        # Re-evaluate merit with updated multipliers
        J = _eval_total_cost(prob, X, U, t, λ_eq, λ_ineq, ν, μ)
    end

    @label done

    # Recompute pure cost (without AL terms) for reporting
    J_pure = _eval_pure_cost(prob, X, U, t)

    return DDPSolution(X, U, t, J_pure, λ_eq, λ_ineq, ν, μ, total_iters, status)
end

# ──────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────

function _init_path_multipliers(::Nothing, N, ::Type{T}) where T
    return [T[] for _ in 1:(N-1)]
end

function _init_path_multipliers(c::EqualityConstraint, N, ::Type{T}) where T
    return [zeros(T, c.p) for _ in 1:(N-1)]
end

function _init_path_multipliers(c::InequalityConstraint, N, ::Type{T}) where T
    return [zeros(T, c.q) for _ in 1:(N-1)]
end

function _init_terminal_multipliers(::Nothing, ::Type{T}) where T
    return T[]
end

function _init_terminal_multipliers(c::TerminalConstraint, ::Type{T}) where T
    return zeros(T, c.r)
end

"""
    _update_multipliers!(prob, X, U, t, λ_eq, λ_ineq, ν, μ) -> c_viol

First-order multiplier update (Hestenes-Powell rule) and return
the maximum constraint violation.
"""
function _update_multipliers!(prob, X, U, t, λ_eq, λ_ineq, ν, μ)
    N = length(X)
    T = eltype(X[1])
    c_viol = zero(T)

    # Path equality: λ ← λ + μ g
    if prob.eq !== nothing
        for k in 1:(N-1)
            gval = prob.eq.g(X[k], U[k], t[k])
            for j in eachindex(λ_eq[k])
                λ_eq[k][j] += μ * gval[j]
            end
            c_viol = max(c_viol, maximum(abs, gval))
        end
    end

    # Path inequality: λ ← max(0, λ - μ h)
    if prob.ineq !== nothing
        for k in 1:(N-1)
            hval = prob.ineq.h(X[k], U[k], t[k])
            for j in eachindex(λ_ineq[k])
                λ_ineq[k][j] = max(zero(T), λ_ineq[k][j] - μ * hval[j])
            end
            for j in eachindex(hval)
                c_viol = max(c_viol, max(zero(T), -hval[j]))
            end
        end
    end

    # Terminal equality: ν ← ν + μ ψ
    if prob.terminal_eq !== nothing
        ψval = prob.terminal_eq.ψ(X[N])
        for j in eachindex(ν)
            ν[j] += μ * ψval[j]
        end
        c_viol = max(c_viol, maximum(abs, ψval))
    end

    return c_viol
end

"""
    _eval_pure_cost(prob, X, U, t) -> T

Evaluate the original (un-augmented) cost.
"""
function _eval_pure_cost(prob, X, U, t)
    N = length(X)
    T = eltype(X[1])
    J = zero(T)
    for k in 1:(N-1)
        J += prob.stage_cost.ℓ(X[k], U[k], t[k])
    end
    J += prob.terminal_cost.ϕ(X[N])
    return J
end
