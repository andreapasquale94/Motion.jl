# ──────────────────────────────────────────────────────────────────────
#  Backward pass – Riccati-like sweep for constrained DDP (HDDP)
#
#  Computes the quadratic value-function approximation
#      V_k(δx) ≈ s_k + Sₓᵀδx + ½ δxᵀ Sₓₓ δx
#  and the affine feedback gains
#      δu* = l_k + L_k δx
#
#  Handles path equality/inequality constraints via augmented
#  Lagrangian terms and terminal equality constraints.
#
#  Reference: Lantoine & Russell, JOTA 2012, §3-4.
# ──────────────────────────────────────────────────────────────────────

"""
    backward_pass!(prob, X, U, t, λ_eq, λ_ineq, ν, μ, reg, opts)

Perform the backward Riccati sweep.

Returns `(gains, ΔJ1, ΔJ2, ok)` where
- `gains = [(l_k, L_k), …]` for k = 1,…,N-1
- `ΔJ1, ΔJ2` are the expected cost reduction from the linear and
  quadratic model, used by the line-search
- `ok` is false when `Quu` is not positive-definite (needs more regularisation)
"""
function backward_pass(prob::DDPProblem, X, U, t,
                       λ_eq, λ_ineq, ν, μ, reg)
    nx, nu = prob.nx, prob.nu
    N = length(X)
    T = eltype(X[1])

    # ── Terminal initialisation ──────────────────────────────────────
    ϕx, ϕxx = terminal_cost_derivatives(prob.terminal_cost.ϕ, X[N])

    Sx  = ϕx
    Sxx = ϕxx

    # Add terminal equality constraint contribution (augmented Lagrangian)
    if prob.terminal_eq !== nothing
        r = prob.terminal_eq.r
        ψval, ψx = terminal_constraint_derivatives(
            prob.terminal_eq.ψ, X[N], Val(r))
        # Augmented Lagrangian: ν'ψ + (μ/2)‖ψ‖²
        Sx  = Sx  + ψx' * SVector{r,T}(ν) + μ * ψx' * ψval
        Sxx = Sxx + μ * ψx' * ψx
    end

    # Pre-allocate gain arrays
    gains_l = Vector{SVector{nu,T}}(undef, N - 1)
    gains_L = Vector{SMatrix{nu,nx,T}}(undef, N - 1)

    ΔJ1 = zero(T)
    ΔJ2 = zero(T)

    # ── Backward sweep k = N-1, …, 1 ────────────────────────────────
    for k in (N-1):-1:1
        xk = X[k]
        uk = U[k]
        tk = t[k]
        tkp1 = t[k+1]

        # Cost derivatives
        ℓx, ℓu, ℓxx, ℓuu, ℓux = cost_derivatives(prob.stage_cost.ℓ, xk, uk, tk)

        # Dynamics linearisation
        fx, fu = dynamics_derivatives(prob.dynamics, xk, uk, tk, tkp1)

        # Q-function expansion (second-order DDP keeps dynamics curvature
        # implicit through the value function Hessian)
        Qx  = ℓx  + fx' * Sx
        Qu  = ℓu  + fu' * Sx
        Qxx = ℓxx + fx' * Sxx * fx
        Quu = ℓuu + fu' * Sxx * fu
        Qux = ℓux + fu' * Sxx * fx

        # ── Path equality constraints ────────────────────────────────
        if prob.eq !== nothing
            p = prob.eq.p
            gval, gx, gu = constraint_derivatives(
                prob.eq.g, xk, uk, tk, Val(p))
            λk = SVector{p,T}(λ_eq[k])
            # Augmented Lagrangian gradient/Hessian contributions
            Qx  = Qx  + gx' * λk + μ * gx' * gval
            Qu  = Qu  + gu' * λk + μ * gu' * gval
            Qxx = Qxx + μ * gx' * gx
            Quu = Quu + μ * gu' * gu
            Qux = Qux + μ * gu' * gx
        end

        # ── Path inequality constraints (active-set via AL) ─────────
        if prob.ineq !== nothing
            q = prob.ineq.q
            hval, hx, hu = constraint_derivatives(
                prob.ineq.h, xk, uk, tk, Val(q))
            λk_ineq = SVector{q,T}(λ_ineq[k])
            for j in 1:q
                # Constraint is h_j ≥ 0.
                # Active if h_j ≤ 0 or multiplier > 0 (complementarity)
                if hval[j] <= zero(T) || λk_ineq[j] > zero(T)
                    # Contribution: -λ_j h_j + (μ/2) h_j²
                    # (sign: we want h ≥ 0, so penalty on violation -h)
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

        # ── Regularisation ───────────────────────────────────────────
        Quu_reg = Quu + reg * SMatrix{nu,nu,T}(I)

        # Check positive-definiteness
        if !isposdef(Symmetric(Matrix(Quu_reg)))
            return nothing, zero(T), zero(T), false
        end

        # ── Gains ────────────────────────────────────────────────────
        Quu_inv = inv(Quu_reg)
        l_k = -Quu_inv * Qu
        L_k = -Quu_inv * Qux

        gains_l[k] = l_k
        gains_L[k] = L_k

        # Expected cost reduction
        ΔJ1 += Qu' * l_k
        ΔJ2 += T(0.5) * l_k' * Quu * l_k

        # ── Propagate value function ─────────────────────────────────
        Sx  = Qx  + L_k' * Quu * l_k + L_k' * Qu + Qux' * l_k
        Sxx = Qxx + L_k' * Quu * L_k + L_k' * Qux + Qux' * L_k
        Sxx = T(0.5) * (Sxx + Sxx')  # enforce symmetry
    end

    return (gains_l, gains_L), ΔJ1, ΔJ2, true
end
