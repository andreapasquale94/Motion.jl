# ──────────────────────────────���────────────────────────��──────────────
#  ForwardDiff-based derivatives for cost, dynamics, and constraints.
#
#  Every function here returns StaticArrays so that they compose
#  efficiently in the backward / forward passes.
# ──────────────────────────────────────────────────────────────────────

# ── Index helpers (computed once, reused) ────────────────────────────

@inline _ix(::Val{nx}) where nx = SVector{nx,Int}(ntuple(identity, Val(nx)))
@inline _iu(::Val{nx}, ::Val{nu}) where {nx,nu} =
    SVector{nu,Int}(ntuple(i -> i + nx, Val(nu)))

# ── Stage cost ───────────────────────────────────────────────────────

"""
    cost_derivatives(ℓ, x, u, t) -> (ℓx, ℓu, ℓxx, ℓuu, ℓux)

First- and second-order derivatives of the stage cost `ℓ(x, u, t)`
with respect to `x` and `u` (time `t` is held fixed).
"""
function cost_derivatives(ℓ, x::SVector{nx,T}, u::SVector{nu,T}, t) where {nx,nu,T}
    z = vcat(x, u)
    ix = _ix(Val(nx))
    iu = _iu(Val(nx), Val(nu))

    ℓz(w) = ℓ(SVector{nx}(w[ix]), SVector{nu}(w[iu]), t)

    g = ForwardDiff.gradient(ℓz, z)
    H = ForwardDiff.hessian(ℓz, z)

    return (SVector{nx,T}(g[ix]),                    # ℓx
            SVector{nu,T}(g[iu]),                    # ℓu
            SMatrix{nx,nx,T}(H[ix, ix]),             # ℓxx
            SMatrix{nu,nu,T}(H[iu, iu]),             # ℓuu
            SMatrix{nu,nx,T}(H[iu, ix]))             # ℓux
end

# ── Terminal cost ───────���───────────────────────────────���────────────

"""
    terminal_cost_derivatives(ϕ, x) -> (ϕx, ϕxx)

Gradient and Hessian of the terminal cost `ϕ(x)`.
"""
function terminal_cost_derivatives(ϕ, x::SVector{nx,T}) where {nx,T}
    ϕx  = SVector{nx,T}(ForwardDiff.gradient(ϕ, x))
    ϕxx = SMatrix{nx,nx,T}(ForwardDiff.hessian(ϕ, x))
    return ϕx, ϕxx
end

# ── Dynamics ─────────────────────────────────────────────────────────

"""
    dynamics_derivatives(f, x, u, tk, tkp1) -> (fx, fu)

State- and control-Jacobians of the discrete dynamics `f(x, u, tk, tkp1)`.
"""
function dynamics_derivatives(f, x::SVector{nx,T}, u::SVector{nu,T},
                              tk, tkp1) where {nx,nu,T}
    fx = ForwardDiff.jacobian(w -> f(w, u, tk, tkp1), x)
    fu = ForwardDiff.jacobian(w -> f(x, w, tk, tkp1), u)
    return SMatrix{nx,nx,T}(fx), SMatrix{nx,nu,T}(fu)
end

"""
    dynamics_hessians(f, x, u, tk, tkp1, Sx) -> (Qxx_t, Quu_t, Qux_t)

Second-order dynamics correction (tensor contraction with the
value-function gradient `Sx`) used by full DDP:

    Qzz_tensor = ∑ᵢ Sx[i] ∂²fᵢ/���z²

where `z = [x; u]`.  Only needed when `method == :DDP`.
"""
function dynamics_hessians(f, x::SVector{nx,T}, u::SVector{nu,T},
                           tk, tkp1, Sx::SVector{nx,T}) where {nx,nu,T}
    z  = vcat(x, u)
    ix = _ix(Val(nx))
    iu = _iu(Val(nx), Val(nu))

    Qxx_t = zero(SMatrix{nx,nx,T})
    Quu_t = zero(SMatrix{nu,nu,T})
    Qux_t = zero(SMatrix{nu,nx,T})

    for i in 1:nx
        fi(w) = f(SVector{nx}(w[ix]), SVector{nu}(w[iu]), tk, tkp1)[i]
        Hi = SMatrix{nx+nu,nx+nu,T}(ForwardDiff.hessian(fi, z))
        Qxx_t = Qxx_t + Sx[i] * Hi[ix, ix]
        Quu_t = Quu_t + Sx[i] * Hi[iu, iu]
        Qux_t = Qux_t + Sx[i] * Hi[iu, ix]
    end
    return Qxx_t, Quu_t, Qux_t
end

# ── Path constraints ───────────────────────���─────────────────────────

"""
    constraint_derivatives(g, x, u, t, Val(p)) -> (gval, gx, gu)

Value, state-Jacobian, and control-Jacobian of a vector constraint
`g(x, u, t) -> SVector{p}`.
"""
function constraint_derivatives(g, x::SVector{nx,T}, u::SVector{nu,T},
                                t, ::Val{p}) where {nx,nu,T,p}
    gval = SVector{p,T}(g(x, u, t))
    gx   = SMatrix{p,nx,T}(ForwardDiff.jacobian(w -> g(w, u, t), x))
    gu   = SMatrix{p,nu,T}(ForwardDiff.jacobian(w -> g(x, w, t), u))
    return gval, gx, gu
end

# ── Terminal constraints ─────────────────────────────────────────────

"""
    terminal_constraint_derivatives(ψ, x, Val(r)) -> (ψval, ψx)

Value and Jacobian of the terminal constraint `ψ(x) -> SVector{r}`.
"""
function terminal_constraint_derivatives(ψ, x::SVector{nx,T},
                                         ::Val{r}) where {nx,T,r}
    ψval = SVector{r,T}(ψ(x))
    ψx   = SMatrix{r,nx,T}(ForwardDiff.jacobian(ψ, x))
    return ψval, ψx
end
