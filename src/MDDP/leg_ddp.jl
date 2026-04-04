# ──────────────────────────────────────────────────────────────────────
#  Per-leg DDP sweep for the MDDP algorithm
#
#  Each leg is an independent sub-problem: given fixed initial state
#  x₀ᵐ and time grid tᵐ, optimise the controls Uᵐ and propagate
#  the states Xᵐ.  Returns the value-function gradient at the start
#  of the leg (Vx₀) which is used by the outer node-correction step.
#
#  Reference: Pellegrini & Russell, Acta Astronautica 2020, §3.
# ──────────────────────────────────────────────────────────────────────

using ..DDP: cost_derivatives, terminal_cost_derivatives,
             dynamics_derivatives, dynamics_hessians,
             constraint_derivatives, terminal_constraint_derivatives

"""
    leg_ddp!(leg, prob, λ_eq_leg, λ_ineq_leg, μ, opts, is_last_leg, ν)

Run DDP iterations on a single leg.  Modifies `leg.X` and `leg.U` in-place.

Returns `(Vx0, J_leg)` where `Vx0` is the value-function gradient at
the initial state of the leg, used for the outer node-correction step.
"""
function leg_ddp!(leg::Leg{T}, prob::MDDPProblem, λ_eq_leg, λ_ineq_leg,
                  μ, opts::MDDPOptions,
                  is_last_leg::Bool, ν) where T
    nx, nu = prob.nx, prob.nu
    N = length(leg.X)
    X, U, t = leg.X, leg.U, leg.t
    method = opts.method

    reg = opts.reg0

    for iter in 1:opts.max_ddp_iter
        # ── Backward pass ────────────────────────────────────────────
        result = _leg_backward(prob, X, U, t, λ_eq_leg, λ_ineq_leg,
                               μ, reg, method, is_last_leg, ν)

        if result === nothing
            reg = (reg <= zero(T)) ? opts.reg_min : reg * opts.reg_factor
            reg > opts.reg_max && break
            continue
        end

        gains_l, gains_L, Vx0, ΔJ1, ΔJ2 = result

        # ── Forward pass (line search) ───────────────────────────────
        J_old = _leg_cost(prob, X, U, t, λ_eq_leg, λ_ineq_leg, μ,
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
                                  μ, method, is_last_leg, ν)
    J_leg = _leg_pure_cost(prob, X, U, t, is_last_leg)
    return Vx0, J_leg
end

# ──────────────────────────────────────────────────────────────────────
#  Backward pass for a single leg
# ──────────────────────────────────────────────────────────────────────

function _leg_backward(prob, X, U, t, λ_eq_leg, λ_ineq_leg,
                       μ, reg, method, is_last_leg, ν)
    nx, nu = prob.nx, prob.nu
    N = length(X)
    T = eltype(X[1])

    # Terminal initialisation
    if is_last_leg
        Sx, Sxx = terminal_cost_derivatives(prob.terminal_cost.ϕ, X[N])
        if prob.terminal_eq !== nothing
            r = prob.terminal_eq.r
            ψval, ψx = terminal_constraint_derivatives(
                prob.terminal_eq.ψ, X[N], Val(r))
            Sx  = Sx  + ψx' * SVector{r,T}(ν) + μ * ψx' * ψval
            Sxx = Sxx + μ * ψx' * ψx
        end
    else
        Sx  = zeros(SVector{nx,T})
        Sxx = zeros(SMatrix{nx,nx,T})
    end

    gains_l = Vector{SVector{nu,T}}(undef, N - 1)
    gains_L = Vector{SMatrix{nu,nx,T}}(undef, N - 1)
    ΔJ1 = zero(T)
    ΔJ2 = zero(T)

    for k in (N-1):-1:1
        xk, uk, tk, tkp1 = X[k], U[k], t[k], t[k+1]

        ℓx, ℓu, ℓxx, ℓuu, ℓux = cost_derivatives(prob.stage_cost.ℓ, xk, uk, tk)
        fx, fu = dynamics_derivatives(prob.dynamics, xk, uk, tk, tkp1)

        Qx  = ℓx  + fx' * Sx
        Qu  = ℓu  + fu' * Sx
        Qxx = ℓxx + fx' * Sxx * fx
        Quu = ℓuu + fu' * Sxx * fu
        Qux = ℓux + fu' * Sxx * fx

        if method === :DDP
            Qxx_t, Quu_t, Qux_t = dynamics_hessians(
                prob.dynamics, xk, uk, tk, tkp1, Sx)
            Qxx = Qxx + Qxx_t
            Quu = Quu + Quu_t
            Qux = Qux + Qux_t
        end

        # Path equality constraints
        if prob.eq !== nothing
            p = prob.eq.p
            gval, gx, gu = constraint_derivatives(prob.eq.g, xk, uk, tk, Val(p))
            λk = SVector{p,T}(λ_eq_leg[k])
            Qx  = Qx  + gx' * λk + μ * gx' * gval
            Qu  = Qu  + gu' * λk + μ * gu' * gval
            Qxx = Qxx + μ * gx' * gx
            Quu = Quu + μ * gu' * gu
            Qux = Qux + μ * gu' * gx
        end

        # Path inequality constraints
        if prob.ineq !== nothing
            q = prob.ineq.q
            hval, hx, hu = constraint_derivatives(prob.ineq.h, xk, uk, tk, Val(q))
            λk_ineq = SVector{q,T}(λ_ineq_leg[k])
            for j in 1:q
                if hval[j] <= zero(T) || λk_ineq[j] > zero(T)
                    hx_j = SVector{nx,T}(hx[j, :])
                    hu_j = SVector{nu,T}(hu[j, :])
                    Qx  = Qx  - λk_ineq[j] * hx_j - μ * hval[j] * hx_j
                    Qu  = Qu  - λk_ineq[j] * hu_j - μ * hval[j] * hu_j
                    Qxx = Qxx + μ * hx_j * hx_j'
                    Quu = Quu + μ * hu_j * hu_j'
                    Qux = Qux + μ * hu_j * hx_j'
                end
            end
        end

        Quu_reg = Quu + reg * SMatrix{nu,nu,T}(I)
        if !isposdef(Symmetric(Matrix(Quu_reg)))
            return nothing
        end

        Quu_inv = inv(Quu_reg)
        l_k = -Quu_inv * Qu
        L_k = -Quu_inv * Qux

        gains_l[k] = l_k
        gains_L[k] = L_k

        ΔJ1 += Qu' * l_k
        ΔJ2 += T(0.5) * l_k' * Quu * l_k

        Sx  = Qx  + L_k' * Quu * l_k + L_k' * Qu + Qux' * l_k
        Sxx = Qxx + L_k' * Quu * L_k + L_k' * Qux + Qux' * L_k
        Sxx = T(0.5) * (Sxx + Sxx')
    end

    # Vx0 = Sx at k=1 (the value-function gradient at the leg's initial state)
    Vx0 = Sx

    return gains_l, gains_L, Vx0, ΔJ1, ΔJ2
end

# ──────────────────────────────────────────────────────────────────────
#  Forward pass for a single leg
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
            δu = α * gains_l[k] + gains_L[k] * δx
            U_new[k] = U[k] + δu
            X_new[k+1] = prob.dynamics(X_new[k], U_new[k], t[k], t[k+1])
        end

        J_new = _leg_cost(prob, X_new, U_new, t, λ_eq_leg, λ_ineq_leg,
                          μ, is_last_leg, ν)

        expected = α * ΔJ1 + α^2 * ΔJ2
        if expected < zero(T)
            if J_new ≤ J_old + γ * expected
                return X_new, U_new, J_new, α
            end
        else
            if J_new < J_old
                return X_new, U_new, J_new, α
            end
        end
        α *= β
    end

    return X, U, J_old, zero(T)
end

# ──────────────────────────────────────────────────────────────────────
#  Cost evaluation helpers
# ──────────────────────────────────────────────────────────────────────

function _leg_cost(prob, X, U, t, λ_eq_leg, λ_ineq_leg, μ,
                   is_last_leg, ν)
    N = length(X)
    T = eltype(X[1])
    J = zero(T)

    for k in 1:(N-1)
        J += prob.stage_cost.ℓ(X[k], U[k], t[k])
    end

    if is_last_leg
        J += prob.terminal_cost.ϕ(X[N])
        if prob.terminal_eq !== nothing
            r = prob.terminal_eq.r
            ψval = prob.terminal_eq.ψ(X[N])
            for j in 1:r
                J += ν[j] * ψval[j] + (μ / 2) * ψval[j]^2
            end
        end
    end

    # Path equality
    if prob.eq !== nothing
        p = prob.eq.p
        for k in 1:(N-1)
            gval = prob.eq.g(X[k], U[k], t[k])
            λk = λ_eq_leg[k]
            for j in 1:p
                J += λk[j] * gval[j] + (μ / 2) * gval[j]^2
            end
        end
    end

    # Path inequality
    if prob.ineq !== nothing
        q = prob.ineq.q
        for k in 1:(N-1)
            hval = prob.ineq.h(X[k], U[k], t[k])
            λk = λ_ineq_leg[k]
            for j in 1:q
                if hval[j] <= zero(T) || λk[j] > zero(T)
                    J += -λk[j] * hval[j] + (μ / 2) * hval[j]^2
                end
            end
        end
    end

    return J
end

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
#  Value-function gradient at leg start (for node correction)
# ──────────────────────────────────────────────────────────────────────

"""
    _leg_value_gradient(prob, X, U, t, ...) -> (Vx0, Vxx0)

Compute the value-function gradient and Hessian at the initial state
of a leg by running a single backward pass (no gain computation needed,
just propagate Sx and Sxx).
"""
function _leg_value_gradient(prob, X, U, t, λ_eq_leg, λ_ineq_leg,
                              μ, method, is_last_leg, ν)
    nx, nu = prob.nx, prob.nu
    N = length(X)
    T = eltype(X[1])

    if is_last_leg
        Sx, Sxx = terminal_cost_derivatives(prob.terminal_cost.ϕ, X[N])
        if prob.terminal_eq !== nothing
            r = prob.terminal_eq.r
            ψval, ψx = terminal_constraint_derivatives(
                prob.terminal_eq.ψ, X[N], Val(r))
            Sx  = Sx  + ψx' * SVector{r,T}(ν) + μ * ψx' * ψval
            Sxx = Sxx + μ * ψx' * ψx
        end
    else
        Sx  = zeros(SVector{nx,T})
        Sxx = zeros(SMatrix{nx,nx,T})
    end

    for k in (N-1):-1:1
        xk, uk, tk, tkp1 = X[k], U[k], t[k], t[k+1]

        ℓx, ℓu, ℓxx, ℓuu, ℓux = cost_derivatives(prob.stage_cost.ℓ, xk, uk, tk)
        fx, fu = dynamics_derivatives(prob.dynamics, xk, uk, tk, tkp1)

        Qx  = ℓx  + fx' * Sx
        Qu  = ℓu  + fu' * Sx
        Qxx = ℓxx + fx' * Sxx * fx
        Quu = ℓuu + fu' * Sxx * fu
        Qux = ℓux + fu' * Sxx * fx

        if method === :DDP
            Qxx_t, Quu_t, Qux_t = dynamics_hessians(
                prob.dynamics, xk, uk, tk, tkp1, Sx)
            Qxx = Qxx + Qxx_t
            Quu = Quu + Quu_t
            Qux = Qux + Qux_t
        end

        # Constraint terms (same as backward pass)
        if prob.eq !== nothing
            p = prob.eq.p
            gval, gx, gu = constraint_derivatives(prob.eq.g, xk, uk, tk, Val(p))
            λk = SVector{p,T}(λ_eq_leg[k])
            Qx  = Qx  + gx' * λk + μ * gx' * gval
            Qu  = Qu  + gu' * λk + μ * gu' * gval
            Qxx = Qxx + μ * gx' * gx
            Quu = Quu + μ * gu' * gu
            Qux = Qux + μ * gu' * gx
        end

        if prob.ineq !== nothing
            q = prob.ineq.q
            hval, hx, hu = constraint_derivatives(prob.ineq.h, xk, uk, tk, Val(q))
            λk_ineq = SVector{q,T}(λ_ineq_leg[k])
            for j in 1:q
                if hval[j] <= zero(T) || λk_ineq[j] > zero(T)
                    hx_j = SVector{nx,T}(hx[j, :])
                    hu_j = SVector{nu,T}(hu[j, :])
                    Qx  = Qx  - λk_ineq[j] * hx_j - μ * hval[j] * hx_j
                    Qu  = Qu  - λk_ineq[j] * hu_j - μ * hval[j] * hu_j
                    Qxx = Qxx + μ * hx_j * hx_j'
                    Quu = Quu + μ * hu_j * hu_j'
                    Qux = Qux + μ * hu_j * hx_j'
                end
            end
        end

        Quu_reg = Quu + T(1e-8) * SMatrix{nu,nu,T}(I)
        Quu_inv = inv(Quu_reg)
        L_k = -Quu_inv * Qux
        l_k = -Quu_inv * Qu

        Sx  = Qx  + L_k' * Quu * l_k + L_k' * Qu + Qux' * l_k
        Sxx = Qxx + L_k' * Quu * L_k + L_k' * Qux + Qux' * L_k
        Sxx = T(0.5) * (Sxx + Sxx')
    end

    return Sx, Sxx
end
