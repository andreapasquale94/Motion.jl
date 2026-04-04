# ──────────────────────────────────────────────────────────────────────
#  Backward pass – Riccati-like sweep
#
#  Computes the quadratic value-function approximation
#      Vₖ(δx) ≈ sₖ + Sₓᵀδx + ½ δxᵀ Sₓₓ δx
#  and the affine feedback gains  δu* = lₖ + Lₖ δx.
#
#  Two flavours selected by `method`:
#    :iLQR – Gauss-Newton (first-order dynamics only)  [2]
#    :DDP  – full second-order (dynamics Hessian tensor) [1]
#
#  References: see types.jl header.
# ──────────────────────────────────────────────────────────────────────

"""
    backward_pass(prob, X, U, t, λ_eq, λ_ineq, ν, μ, reg, method) -> (gains, ΔJ1, ΔJ2, ok)

Backward Riccati sweep over the full trajectory.

Returns `(nothing, 0, 0, false)` when `Quu` is not positive-definite
at some stage (caller should increase regularisation and retry).
Otherwise returns the tuple `((l, L), ΔJ1, ΔJ2, true)` where `l[k]`
and `L[k]` are the feedforward and feedback gains.
"""
function backward_pass(prob::DDPProblem, X, U, t,
                       λ_eq, λ_ineq, ν, μ, reg, method::Symbol)
    nx, nu = prob.nx, prob.nu
    N = length(X)
    T = eltype(X[1])

    # ── Terminal initialisation ──────────────────────────────────────
    Sx, Sxx = _terminal_init(prob, X[N], ν, μ)

    gains_l = Vector{SVector{nu,T}}(undef, N - 1)
    gains_L = Vector{SMatrix{nu,nx,T}}(undef, N - 1)
    ΔJ1 = zero(T)
    ΔJ2 = zero(T)

    # ── Backward sweep ──────────────────────────────────────────────
    for k in (N-1):-1:1
        xk, uk, tk, tkp1 = X[k], U[k], t[k], t[k+1]

        # Q-function quadratic expansion
        Qx, Qu, Qxx, Quu, Qux = _Q_expansion(
            prob, xk, uk, tk, tkp1, Sx, Sxx, method)

        # Augment with constraint penalty terms
        if prob.eq !== nothing
            Qx, Qu, Qxx, Quu, Qux = augment_Q_equality(
                Qx, Qu, Qxx, Quu, Qux, prob.eq, xk, uk, tk, λ_eq[k], μ)
        end
        if prob.ineq !== nothing
            Qx, Qu, Qxx, Quu, Qux = augment_Q_inequality(
                Qx, Qu, Qxx, Quu, Qux, prob.ineq, xk, uk, tk, λ_ineq[k], μ)
        end

        # Regularise and factorise
        Quu_reg = Quu + reg * SMatrix{nu,nu,T}(I)
        if !isposdef(Symmetric(Matrix(Quu_reg)))
            return nothing, zero(T), zero(T), false
        end

        Quu_inv = inv(Quu_reg)
        l_k = -Quu_inv * Qu
        L_k = -Quu_inv * Qux
        gains_l[k] = l_k
        gains_L[k] = L_k

        ΔJ1 += Qu' * l_k
        ΔJ2 += T(0.5) * l_k' * Quu * l_k

        # Propagate value function
        Sx  = Qx  + L_k' * Quu * l_k + L_k' * Qu + Qux' * l_k
        Sxx = Qxx + L_k' * Quu * L_k + L_k' * Qux + Qux' * L_k
        Sxx = T(0.5) * (Sxx + Sxx')
    end

    return (gains_l, gains_L), ΔJ1, ΔJ2, true
end

# ──────────────────────────────────────────────────────────────────────
#  Internal helpers (also reused by MDDP)
# ──────────────────────────────────────────────────────────────────────

"""
    _terminal_init(prob, xN, ν, μ) -> (Sx, Sxx)

Initialise the value-function partials at the terminal node,
including augmented-Lagrangian terminal constraint terms.
"""
function _terminal_init(prob, xN, ν, μ)
    T = eltype(xN)
    Sx, Sxx = terminal_cost_derivatives(prob.terminal_cost.ϕ, xN)
    if prob.terminal_eq !== nothing
        r = prob.terminal_eq.r
        ψval, ψx = terminal_constraint_derivatives(prob.terminal_eq.ψ, xN, Val(r))
        Sx  = Sx  + ψx' * SVector{r,T}(ν) + μ * ψx' * ψval
        Sxx = Sxx + μ * ψx' * ψx
    end
    return Sx, Sxx
end

"""
    _Q_expansion(prob, x, u, tk, tkp1, Sx, Sxx, method) -> (Qx, Qu, Qxx, Quu, Qux)

Build the bare (unconstrained) Q-function quadratic expansion
from cost and dynamics derivatives.
"""
function _Q_expansion(prob, x, u, tk, tkp1, Sx, Sxx, method::Symbol)
    ℓx, ℓu, ℓxx, ℓuu, ℓux = cost_derivatives(prob.stage_cost.ℓ, x, u, tk)
    fx, fu = dynamics_derivatives(prob.dynamics, x, u, tk, tkp1)

    Qx  = ℓx  + fx' * Sx
    Qu  = ℓu  + fu' * Sx
    Qxx = ℓxx + fx' * Sxx * fx
    Quu = ℓuu + fu' * Sxx * fu
    Qux = ℓux + fu' * Sxx * fx

    if method === :DDP
        Qxx_t, Quu_t, Qux_t = dynamics_hessians(prob.dynamics, x, u, tk, tkp1, Sx)
        Qxx = Qxx + Qxx_t
        Quu = Quu + Quu_t
        Qux = Qux + Qux_t
    end

    return Qx, Qu, Qxx, Quu, Qux
end
