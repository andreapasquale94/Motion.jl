# ──────────────────────────────────────────────────────────────────────
#  Forward pass – line-search rollout for constrained DDP
#
#  Given the gains (l_k, L_k) from the backward pass, compute a new
#  trajectory via:
#      δx_1 = 0   (or α * l_1 for state perturbation)
#      δu_k = α l_k + L_k δx_k
#      x̄_{k+1} = f(x̄_k, ū_k, t_k, t_{k+1})
#
#  A backtracking line-search on α ∈ (0, 1] enforces sufficient decrease.
#
#  Reference: Lantoine & Russell, JOTA 2012, §5.
# ──────────────────────────────────────────────────────────────────────

"""
    forward_pass(prob, X, U, t, gains, J_old, ΔJ1, ΔJ2,
                 λ_eq, λ_ineq, ν, μ, opts) -> (X_new, U_new, J_new, α)

Backtracking line-search rollout.  Returns the new trajectory and the
accepted step size `α`, or `α = 0` if no sufficient decrease was found.
"""
function forward_pass(prob::DDPProblem, X, U, t,
                      gains_l, gains_L, J_old, ΔJ1, ΔJ2,
                      λ_eq, λ_ineq, ν, μ, opts::DDPOptions)
    nx, nu = prob.nx, prob.nu
    N = length(X)
    T = eltype(X[1])

    β = opts.line_search_β
    γ = opts.line_search_γ

    α = one(T)
    X_new = similar(X)
    U_new = similar(U)

    for _ in 1:20   # max line-search iterations
        X_new[1] = X[1]

        for k in 1:(N-1)
            δx = X_new[k] - X[k]
            δu = α * gains_l[k] + gains_L[k] * δx
            U_new[k] = U[k] + δu
            X_new[k+1] = prob.dynamics(X_new[k], U_new[k], t[k], t[k+1])
        end

        J_new = _eval_total_cost(prob, X_new, U_new, t, λ_eq, λ_ineq, ν, μ)

        expected = α * ΔJ1 + α^2 * ΔJ2
        if expected < zero(T)  # we expect a decrease
            if J_new ≤ J_old + γ * expected
                return X_new, U_new, J_new, α
            end
        else
            # Fallback: accept if cost decreased at all
            if J_new < J_old
                return X_new, U_new, J_new, α
            end
        end

        α *= β
    end

    # Line search failed – return original trajectory
    return X, U, J_old, zero(T)
end

# ──────────────────────────────────────────────────────────────────────
#  Cost evaluation (includes augmented Lagrangian terms)
# ──────────────────────────────────────────────────────────────────────

"""
    _eval_total_cost(prob, X, U, t, λ_eq, λ_ineq, ν, μ) -> T

Total augmented-Lagrangian merit function:
    J = ∑ ℓ_k + ϕ(x_N)
      + ∑ [λ_eq'g + (μ/2)‖g‖²]
      + ∑ [inequality AL terms]
      + ν'ψ(x_N) + (μ/2)‖ψ(x_N)‖²
"""
function _eval_total_cost(prob::DDPProblem, X, U, t,
                          λ_eq, λ_ineq, ν, μ)
    N = length(X)
    T = eltype(X[1])
    J = zero(T)

    # Stage costs
    for k in 1:(N-1)
        J += prob.stage_cost.ℓ(X[k], U[k], t[k])
    end

    # Terminal cost
    J += prob.terminal_cost.ϕ(X[N])

    # Path equality constraint AL terms
    if prob.eq !== nothing
        p = prob.eq.p
        for k in 1:(N-1)
            gval = prob.eq.g(X[k], U[k], t[k])
            λk = λ_eq[k]
            for j in 1:p
                J += λk[j] * gval[j] + (μ / 2) * gval[j]^2
            end
        end
    end

    # Path inequality constraint AL terms  (h ≥ 0)
    if prob.ineq !== nothing
        q = prob.ineq.q
        for k in 1:(N-1)
            hval = prob.ineq.h(X[k], U[k], t[k])
            λk = λ_ineq[k]
            for j in 1:q
                # Active: h_j ≤ 0 or λ_j > 0
                if hval[j] <= zero(T) || λk[j] > zero(T)
                    # Merit: -λ_j h_j + (μ/2) h_j²
                    J += -λk[j] * hval[j] + (μ / 2) * hval[j]^2
                end
            end
        end
    end

    # Terminal equality constraint AL terms
    if prob.terminal_eq !== nothing
        r = prob.terminal_eq.r
        ψval = prob.terminal_eq.ψ(X[N])
        for j in 1:r
            J += ν[j] * ψval[j] + (μ / 2) * ψval[j]^2
        end
    end

    return J
end
