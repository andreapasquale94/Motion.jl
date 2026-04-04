# ──────────────────────────────────────────────────────────────────────
#  Shared augmented-Lagrangian helpers
#
#  These functions are used by both the DDP and MDDP modules to
#  augment the Q-function expansion with constraint penalty terms
#  and to evaluate the AL merit function.
# ────────────────────────────────────────────���─────────────────────────

"""
    augment_Q_equality!(Qx, Qu, Qxx, Quu, Qux, eq, x, u, t, λk, μ)

Add augmented-Lagrangian terms for path equality constraints
`g(x,u,t) = 0` to the Q-function expansion in-place (returns new values).
"""
function augment_Q_equality(Qx, Qu, Qxx, Quu, Qux,
                            eq::EqualityConstraint, x, u, t, λk, μ)
    p = eq.p
    gval, gx, gu = constraint_derivatives(eq.g, x, u, t, Val(p))
    T = eltype(x)
    λ = SVector{p,T}(λk)
    return (Qx  + gx' * λ + μ * gx' * gval,
            Qu  + gu' * λ + μ * gu' * gval,
            Qxx + μ * gx' * gx,
            Quu + μ * gu' * gu,
            Qux + μ * gu' * gx)
end

"""
    augment_Q_inequality(Qx, Qu, Qxx, Quu, Qux, ineq, x, u, t, λk, μ)

Add augmented-Lagrangian terms for path inequality constraints
`h(x,u,t) ≥ 0` to the Q-function expansion.  Only active constraints
(violated or positive multiplier) contribute.
"""
function augment_Q_inequality(Qx, Qu, Qxx, Quu, Qux,
                              ineq::InequalityConstraint, x, u, t, λk, μ)
    nx = length(x); nu = length(u)
    q = ineq.q
    T = eltype(x)
    hval, hx, hu = constraint_derivatives(ineq.h, x, u, t, Val(q))
    λ_ineq = SVector{q,T}(λk)

    for j in 1:q
        if hval[j] <= zero(T) || λ_ineq[j] > zero(T)
            hx_j = SVector{nx,T}(hx[j, :])
            hu_j = SVector{nu,T}(hu[j, :])
            Qx  = Qx  - λ_ineq[j] * hx_j - μ * hval[j] * hx_j
            Qu  = Qu  - λ_ineq[j] * hu_j - μ * hval[j] * hu_j
            Qxx = Qxx + μ * hx_j * hx_j'
            Quu = Quu + μ * hu_j * hu_j'
            Qux = Qux + μ * hu_j * hx_j'
        end
    end
    return Qx, Qu, Qxx, Quu, Qux
end

"""
    eval_stage_al_cost(eq, ineq, x, u, t, λ_eq, λ_ineq, μ) -> T

Evaluate the augmented-Lagrangian cost contribution from path
constraints at a single stage.
"""
function eval_stage_al_cost(eq, ineq, x, u, t, λ_eq, λ_ineq, μ)
    T = eltype(x)
    J = zero(T)

    if eq !== nothing
        gval = eq.g(x, u, t)
        for j in 1:eq.p
            J += λ_eq[j] * gval[j] + (μ / 2) * gval[j]^2
        end
    end

    if ineq !== nothing
        hval = ineq.h(x, u, t)
        for j in 1:ineq.q
            if hval[j] <= zero(T) || λ_ineq[j] > zero(T)
                J += -λ_ineq[j] * hval[j] + (μ / 2) * hval[j]^2
            end
        end
    end

    return J
end

"""
    eval_terminal_al_cost(terminal_eq, x, ν, μ) -> T

Evaluate the augmented-Lagrangian cost from terminal constraints.
"""
function eval_terminal_al_cost(terminal_eq, x, ν, μ)
    T = eltype(x)
    J = zero(T)
    if terminal_eq !== nothing
        ψval = terminal_eq.ψ(x)
        for j in 1:terminal_eq.r
            J += ν[j] * ψval[j] + (μ / 2) * ψval[j]^2
        end
    end
    return J
end

# ── Multiplier initialisers ─────────────────────────────────────────

init_path_multipliers(::Nothing, N, ::Type{T}) where T = [T[] for _ in 1:(N-1)]
init_path_multipliers(c::EqualityConstraint, N, ::Type{T}) where T =
    [zeros(T, c.p) for _ in 1:(N-1)]
init_path_multipliers(c::InequalityConstraint, N, ::Type{T}) where T =
    [zeros(T, c.q) for _ in 1:(N-1)]

init_terminal_multipliers(::Nothing, ::Type{T}) where T = T[]
init_terminal_multipliers(c::TerminalConstraint, ::Type{T}) where T = zeros(T, c.r)

# ── Multiplier update (Hestenes-Powell) ─────────────────────────────

"""
    update_multipliers!(eq, ineq, terminal_eq, X, U, t, λ_eq, λ_ineq, ν, μ) -> c_viol

First-order multiplier update and return the maximum constraint violation.
"""
function update_multipliers!(eq, ineq, terminal_eq, X, U, t,
                             λ_eq, λ_ineq, ν, μ)
    N = length(X)
    T = eltype(X[1])
    c_viol = zero(T)

    if eq !== nothing
        for k in 1:(N-1)
            gval = eq.g(X[k], U[k], t[k])
            for j in eachindex(λ_eq[k])
                λ_eq[k][j] += μ * gval[j]
            end
            c_viol = max(c_viol, maximum(abs, gval))
        end
    end

    if ineq !== nothing
        for k in 1:(N-1)
            hval = ineq.h(X[k], U[k], t[k])
            for j in eachindex(λ_ineq[k])
                λ_ineq[k][j] = max(zero(T), λ_ineq[k][j] - μ * hval[j])
            end
            for j in eachindex(hval)
                c_viol = max(c_viol, max(zero(T), -hval[j]))
            end
        end
    end

    if terminal_eq !== nothing
        ψval = terminal_eq.ψ(X[N])
        for j in eachindex(ν)
            ν[j] += μ * ψval[j]
        end
        c_viol = max(c_viol, maximum(abs, ψval))
    end

    return c_viol
end
