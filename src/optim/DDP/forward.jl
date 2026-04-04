# ──────────────────────────────────────────────────────────────────────
#  Forward pass – line-search rollout
#
#  Given gains (lₖ, Lₖ) from the backward pass, roll out
#      δuₖ = α lₖ + Lₖ δxₖ,   x̄_{k+1} = f(x̄_k, ū_k, tₖ, tₖ₊₁)
#  with backtracking on α ∈ (0, 1] to enforce Armijo decrease.
# ──────────────────────────────────────────────────────────────────────

"""
    forward_pass(prob, X, U, t, l, L, J_old, ΔJ1, ΔJ2,
                 λ_eq, λ_ineq, ν, μ, opts) -> (X_new, U_new, J_new, α)

Backtracking line-search rollout.  Returns `α = 0` when no sufficient
decrease is found (caller should increase regularisation).
"""
function forward_pass(prob::DDPProblem, X, U, t,
                      gains_l, gains_L, J_old, ΔJ1, ΔJ2,
                      λ_eq, λ_ineq, ν, μ, opts::DDPOptions)
    N = length(X)
    T = eltype(X[1])
    β, γ = opts.line_search_β, opts.line_search_γ

    α = one(T)
    X_new = similar(X)
    U_new = similar(U)

    for _ in 1:20
        X_new[1] = X[1]
        for k in 1:(N-1)
            δx = X_new[k] - X[k]
            U_new[k] = U[k] + α * gains_l[k] + gains_L[k] * δx
            X_new[k+1] = prob.dynamics(X_new[k], U_new[k], t[k], t[k+1])
        end

        J_new = eval_merit(prob, X_new, U_new, t, λ_eq, λ_ineq, ν, μ)

        expected = α * ΔJ1 + α^2 * ΔJ2
        if expected < zero(T)
            J_new ≤ J_old + γ * expected && return X_new, U_new, J_new, α
        else
            J_new < J_old && return X_new, U_new, J_new, α
        end
        α *= β
    end

    return X, U, J_old, zero(T)
end

# ──────────────────────────────────────────────────────────────────────
#  Merit / cost evaluation (shared with MDDP via `using ..DDP`)
# ──────────────────────────────────────────────────────────────────────

"""
    eval_merit(prob, X, U, t, λ_eq, λ_ineq, ν, μ) -> T

Total augmented-Lagrangian merit function:
`J = ∑ ℓₖ + ϕ(xN) + AL(equality) + AL(inequality) + AL(terminal)`.
"""
function eval_merit(prob, X, U, t, λ_eq, λ_ineq, ν, μ)
    N = length(X)
    T = eltype(X[1])
    J = zero(T)

    for k in 1:(N-1)
        J += prob.stage_cost.ℓ(X[k], U[k], t[k])
        J += eval_stage_al_cost(prob.eq, prob.ineq, X[k], U[k], t[k],
                                λ_eq[k], λ_ineq[k], μ)
    end

    J += prob.terminal_cost.ϕ(X[N])
    J += eval_terminal_al_cost(prob.terminal_eq, X[N], ν, μ)
    return J
end

"""
    eval_pure_cost(prob, X, U, t) -> T

Un-augmented objective (stage costs + terminal cost only).
"""
function eval_pure_cost(prob, X, U, t)
    N = length(X)
    T = eltype(X[1])
    J = zero(T)
    for k in 1:(N-1)
        J += prob.stage_cost.ℓ(X[k], U[k], t[k])
    end
    return J + prob.terminal_cost.ϕ(X[N])
end
