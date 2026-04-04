# ──────────────────────────────────────────────────────────────────────
#  Per-leg DDP sweep for the MDDP algorithm
#
#  Each leg is an independent sub-problem: given fixed initial state
#  x₀ᵐ and time grid tᵐ, optimise the controls Uᵐ and propagate
#  the states Xᵐ.  Returns the value-function gradient at the start
#  of the leg (Vx₀) which is used by the outer node-correction step.
#
#  This file delegates to shared DDP helpers (_terminal_init,
#  _Q_expansion, augment_Q_*, eval_*_al_cost) rather than duplicating
#  the backward/forward/cost logic.
#
#  Reference: Pellegrini & Russell, Acta Astronautica 2020, §3.
# ──────────────────────────────────────────────────────────────────────

"""
    leg_ddp!(leg, prob, λ_eq_leg, λ_ineq_leg, μ, opts, is_last_leg, ν)

Run DDP iterations on a single shooting leg.  Modifies `leg.X` and
`leg.U` in-place.

Returns `(Vx0, J_leg)` where `Vx0` is the value-function gradient at
the initial state of the leg (used by the node-correction step) and
`J_leg` is the un-augmented cost on this leg.
"""
function leg_ddp!(leg::Leg{T}, prob::MDDPProblem, λ_eq_leg, λ_ineq_leg,
                  μ, opts::MDDPOptions,
                  is_last_leg::Bool, ν) where T
    nx, nu = prob.nx, prob.nu
    X, U, t = leg.X, leg.U, leg.t
    reg = opts.reg0

    for iter in 1:opts.max_ddp_iter
        # ── Backward pass ────────────────────────────────────────────
        result = _leg_backward(prob, X, U, t, λ_eq_leg, λ_ineq_leg,
                               μ, reg, opts.method, is_last_leg, ν)

        if result === nothing
            reg = (reg <= zero(T)) ? opts.reg_min : reg * opts.reg_factor
            reg > opts.reg_max && break
            continue
        end

        gains_l, gains_L, Vx0, ΔJ1, ΔJ2 = result

        # ── Forward pass (line search) ───────────────────────────────
        J_old = _leg_merit(prob, X, U, t, λ_eq_leg, λ_ineq_leg, μ,
                           is_last_leg, ν)

        X_new, U_new, J_new, α = _leg_forward(
            prob, X, U, t, gains_l, gains_L, J_old, ΔJ1, ΔJ2,
            λ_eq_leg, λ_ineq_leg, μ, opts, is_last_leg, ν)

        if α == zero(T)
            reg = (reg <= zero(T)) ? opts.reg_min : reg * opts.reg_factor
            reg > opts.reg_max && break
            continue
        end

        ΔJ = J_old - J_new
        leg.X = X_new
        leg.U = U_new
        X, U = X_new, U_new
        reg = max(reg / opts.reg_factor, opts.reg_min)

        if abs(ΔJ) < opts.atol || abs(ΔJ) / (abs(J_new) + T(1e-16)) < opts.rtol
            break
        end
    end

    # Compute final Vx0 for node correction
    Vx0, _ = _leg_value_gradient(prob, X, U, t, λ_eq_leg, λ_ineq_leg,
                                  μ, opts.method, is_last_leg, ν)
    J_leg = _leg_pure_cost(prob, X, U, t, is_last_leg)
    return Vx0, J_leg
end

# ──────────────────────────────────────────────────────────────────────
#  Backward pass for a single leg
#
#  Uses shared _terminal_init and _Q_expansion from DDP, plus the
#  shared augment_Q_equality / augment_Q_inequality helpers.
# ──────────────────────────────────────────────────────────────────────

function _leg_backward(prob, X, U, t, λ_eq_leg, λ_ineq_leg,
                       μ, reg, method, is_last_leg, ν)
    nx, nu = prob.nx, prob.nu
    N = length(X)
    T = eltype(X[1])

    # Terminal initialisation: last leg gets cost + constraint terms;
    # interior legs start from zero (no terminal cost contribution)
    if is_last_leg
        Sx, Sxx = _terminal_init(prob, X[N], ν, μ)
    else
        Sx  = zero(SVector{nx,T})
        Sxx = zero(SMatrix{nx,nx,T})
    end

    gains_l = Vector{SVector{nu,T}}(undef, N - 1)
    gains_L = Vector{SMatrix{nu,nx,T}}(undef, N - 1)
    ΔJ1 = zero(T)
    ΔJ2 = zero(T)

    for k in (N-1):-1:1
        xk, uk, tk, tkp1 = X[k], U[k], t[k], t[k+1]

        # Bare Q-function expansion (cost + dynamics)
        Qx, Qu, Qxx, Quu, Qux = _Q_expansion(
            prob, xk, uk, tk, tkp1, Sx, Sxx, method)

        # Augment with constraint AL terms
        if prob.eq !== nothing
            Qx, Qu, Qxx, Quu, Qux = augment_Q_equality(
                Qx, Qu, Qxx, Quu, Qux, prob.eq, xk, uk, tk, λ_eq_leg[k], μ)
        end
        if prob.ineq !== nothing
            Qx, Qu, Qxx, Quu, Qux = augment_Q_inequality(
                Qx, Qu, Qxx, Quu, Qux, prob.ineq, xk, uk, tk, λ_ineq_leg[k], μ)
        end

        # Regularise and factorise
        Quu_reg = Quu + reg * SMatrix{nu,nu,T}(I)
        C = cholesky(Symmetric(Matrix(Quu_reg)); check=false)
        if !issuccess(C)
            return nothing
        end

        l_k = -(Quu_reg \ Qu)
        L_k = -(Quu_reg \ Qux)

        gains_l[k] = l_k
        gains_L[k] = L_k

        ΔJ1 += Qu' * l_k
        ΔJ2 += T(0.5) * l_k' * Quu * l_k

        # Propagate value function backwards
        Sx  = Qx  + L_k' * Quu * l_k + L_k' * Qu + Qux' * l_k
        Sxx = Qxx + L_k' * Quu * L_k + L_k' * Qux + Qux' * L_k
        Sxx = T(0.5) * (Sxx + Sxx')
    end

    # Vx0 = Sx at k=1 (value-function gradient at the leg's initial state)
    return gains_l, gains_L, Sx, ΔJ1, ΔJ2
end

# ──────────────────────────────────────────────────────────────────────
#  Forward pass (line-search rollout) for a single leg
# ──────────────────────────────────────────────────────────────────────

function _leg_forward(prob, X, U, t, gains_l, gains_L, J_old, ΔJ1, ΔJ2,
                      λ_eq_leg, λ_ineq_leg, μ, opts, is_last_leg, ν)
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

        J_new = _leg_merit(prob, X_new, U_new, t, λ_eq_leg, λ_ineq_leg,
                           μ, is_last_leg, ν)

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
#  Cost / merit evaluation helpers
#
#  Delegates to shared eval_stage_al_cost / eval_terminal_al_cost.
# ──────────────────────────────────────────────────────────────────────

"""
    _leg_merit(prob, X, U, t, λ_eq_leg, λ_ineq_leg, μ, is_last_leg, ν)

Augmented-Lagrangian merit function on a single leg.
"""
function _leg_merit(prob, X, U, t, λ_eq_leg, λ_ineq_leg, μ,
                    is_last_leg, ν)
    N = length(X)
    T = eltype(X[1])
    J = zero(T)

    for k in 1:(N-1)
        J += prob.stage_cost.ℓ(X[k], U[k], t[k])
        J += eval_stage_al_cost(prob.eq, prob.ineq, X[k], U[k], t[k],
                                λ_eq_leg[k], λ_ineq_leg[k], μ)
    end

    if is_last_leg
        J += prob.terminal_cost.ϕ(X[N])
        J += eval_terminal_al_cost(prob.terminal_eq, X[N], ν, μ)
    end

    return J
end

"""
    _leg_pure_cost(prob, X, U, t, is_last_leg)

Un-augmented cost on a single leg (stage costs, plus terminal cost on
the last leg only).
"""
function _leg_pure_cost(prob, X, U, t, is_last_leg)
    N = length(X)
    T = eltype(X[1])
    J = zero(T)
    for k in 1:(N-1)
        J += prob.stage_cost.ℓ(X[k], U[k], t[k])
    end
    if is_last_leg
        J += prob.terminal_cost.ϕ(X[N])
    end
    return J
end

# ──────────────────────────────────────────────────────────────────────
#  Value-function gradient / Hessian at leg start (for node correction)
#
#  Runs a single backward sweep (using shared helpers) to propagate
#  Sx and Sxx from terminal to initial, returning (Vx₀, Vxx₀).
# ──────────────────────────────────────────────────────────────────────

"""
    _leg_value_gradient(prob, X, U, t, ...) -> (Vx0, Vxx0)

Compute the value-function gradient and Hessian at the initial state
of a leg via a backward Riccati sweep.  Used by the node-correction
step to form the block-tridiagonal Newton system.
"""
function _leg_value_gradient(prob, X, U, t, λ_eq_leg, λ_ineq_leg,
                              μ, method, is_last_leg, ν)
    nx, nu = prob.nx, prob.nu
    N = length(X)
    T = eltype(X[1])

    # Terminal initialisation
    if is_last_leg
        Sx, Sxx = _terminal_init(prob, X[N], ν, μ)
    else
        Sx  = zero(SVector{nx,T})
        Sxx = zero(SMatrix{nx,nx,T})
    end

    for k in (N-1):-1:1
        xk, uk, tk, tkp1 = X[k], U[k], t[k], t[k+1]

        # Q-function expansion via shared helpers
        Qx, Qu, Qxx, Quu, Qux = _Q_expansion(
            prob, xk, uk, tk, tkp1, Sx, Sxx, method)

        if prob.eq !== nothing
            Qx, Qu, Qxx, Quu, Qux = augment_Q_equality(
                Qx, Qu, Qxx, Quu, Qux, prob.eq, xk, uk, tk, λ_eq_leg[k], μ)
        end
        if prob.ineq !== nothing
            Qx, Qu, Qxx, Quu, Qux = augment_Q_inequality(
                Qx, Qu, Qxx, Quu, Qux, prob.ineq, xk, uk, tk, λ_ineq_leg[k], μ)
        end

        # Small regularisation for numerical stability
        Quu_reg = Quu + T(1e-8) * SMatrix{nu,nu,T}(I)
        l_k = -(Quu_reg \ Qu)
        L_k = -(Quu_reg \ Qux)

        Sx  = Qx  + L_k' * Quu * l_k + L_k' * Qu + Qux' * l_k
        Sxx = Qxx + L_k' * Quu * L_k + L_k' * Qux + Qux' * L_k
        Sxx = T(0.5) * (Sxx + Sxx')
    end

    return Sx, Sxx
end
