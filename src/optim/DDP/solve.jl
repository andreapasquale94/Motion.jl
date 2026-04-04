# ──────────────────────────────────────────────────────────────────────
#  Outer solver – augmented-Lagrangian DDP / iLQR
#
#  Two-loop structure:
#    Outer: update Lagrange multipliers λ, ν and penalty μ
#    Inner: standard DDP (backward + forward) at fixed (λ, ν, μ)
# ──────────────────────────────────────────────────────────────────────

"""
    solve(prob::DDPProblem, X0, U0, t; opts=DDPOptions()) -> DDPSolution

Solve a discrete-time constrained optimal-control problem using
DDP / iLQR with an augmented-Lagrangian outer loop.

# Arguments
- `prob`  – [`DDPProblem`](@ref) specification
- `X0`    – initial state trajectory guess (`Vector` of `SVector`, length `N`)
- `U0`    – initial control guess (`Vector` of `SVector`, length `N-1`)
- `t`     – node times (`Vector`, length `N`)
- `opts`  – [`DDPOptions`](@ref)

# Returns
A [`DDPSolution`](@ref).

# Algorithm outline
1. **Inner loop** (at fixed multipliers / penalty):
   backward pass → gains (l, L) → forward rollout with line-search.
2. **Outer loop**: Hestenes-Powell multiplier update, penalty increase
   when constraint violation exceeds `ctol`.
"""
function solve(prob::DDPProblem, X0::Vector{SVector{nx,T}},
               U0::Vector{SVector{nu,T}},
               t::Vector{T};
               opts::DDPOptions = DDPOptions{T}()) where {nx, nu, T}

    N = length(X0)
    @assert length(U0) == N - 1 "U0 must have length N-1 = $(N-1)"
    @assert length(t) == N      "t must have length N = $N"
    @assert prob.nx == nx && prob.nu == nu

    # ── Initialise ──────────────────────────────────────────────────
    λ_eq   = init_path_multipliers(prob.eq,   N, T)
    λ_ineq = init_path_multipliers(prob.ineq, N, T)
    ν      = init_terminal_multipliers(prob.terminal_eq, T)

    μ   = opts.μ0
    reg = opts.reg0
    X, U = copy(X0), copy(U0)
    J = eval_merit(prob, X, U, t, λ_eq, λ_ineq, ν, μ)

    total_iters = 0
    status = :max_iterations

    # ── Outer augmented-Lagrangian loop ─────────────────────────────
    for outer in 1:opts.max_outer

        # ── Inner DDP loop ──────────────────────────────────────────
        for iter in 1:opts.max_iter
            total_iters += 1

            result, ΔJ1, ΔJ2, ok = backward_pass(
                prob, X, U, t, λ_eq, λ_ineq, ν, μ, reg, opts.method)

            if !ok
                reg = _increase_reg(reg, opts)
                reg > opts.reg_max && (@goto done; status = :regularisation_failure)
                continue
            end

            gains_l, gains_L = result
            X_new, U_new, J_new, α = forward_pass(
                prob, X, U, t, gains_l, gains_L, J, ΔJ1, ΔJ2,
                λ_eq, λ_ineq, ν, μ, opts)

            if α == zero(T)
                reg = _increase_reg(reg, opts)
                reg > opts.reg_max && (@goto done; status = :line_search_failure)
                continue
            end

            ΔJ = J - J_new
            J, X, U = J_new, X_new, U_new
            reg = max(reg / opts.reg_factor, opts.reg_min)

            opts.verbose && @info "DDP" outer iter α J ΔJ reg

            (abs(ΔJ) < opts.atol || abs(ΔJ) / (abs(J) + T(1e-16)) < opts.rtol) && break
        end

        # ── Multiplier update ───────────────────────────────────────
        c_viol = update_multipliers!(
            prob.eq, prob.ineq, prob.terminal_eq, X, U, t, λ_eq, λ_ineq, ν, μ)

        opts.verbose && @info "AL outer" outer μ c_viol

        if c_viol < opts.ctol
            status = :converged
            @goto done
        end

        μ = min(μ * opts.ϕ_μ, opts.μ_max)
        J = eval_merit(prob, X, U, t, λ_eq, λ_ineq, ν, μ)
    end

    @label done
    return DDPSolution(X, U, t, eval_pure_cost(prob, X, U, t),
                       λ_eq, λ_ineq, ν, μ, total_iters, status)
end

# ──────────────────────────────────────────────────────────────────────

@inline function _increase_reg(reg, opts)
    T = typeof(reg)
    return reg <= zero(T) ? opts.reg_min : reg * opts.reg_factor
end
