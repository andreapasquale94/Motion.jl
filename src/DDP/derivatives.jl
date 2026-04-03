# ──────────────────────────────────────────────────────────────────────
#  Automatic-differentiation helpers for cost / dynamics / constraints
# ──────────────────────────────────────────────────────────────────────

using ForwardDiff

# ── Stage cost derivatives ───────────────────────────────────────────

"""
    cost_derivatives(ℓ, x, u, t) -> (ℓx, ℓu, ℓxx, ℓuu, ℓux)

First and second derivatives of the stage cost `ℓ(x, u, t)`.
"""
function cost_derivatives(ℓ, x::SVector{nx,T}, u::SVector{nu,T}, t) where {nx, nu, T}
    # Concatenate state and control for a single AD pass
    z  = vcat(x, u)

    ℓz(w)  = ℓ(SVector{nx}(w[SVector{nx,Int}(ntuple(identity, Val(nx)))]),
               SVector{nu}(w[SVector{nu,Int}(ntuple(i -> i + nx, Val(nu)))]),
               t)

    g  = ForwardDiff.gradient(ℓz, z)
    H  = ForwardDiff.hessian(ℓz, z)

    ℓx  = SVector{nx,T}(g[SVector{nx,Int}(ntuple(identity, Val(nx)))])
    ℓu  = SVector{nu,T}(g[SVector{nu,Int}(ntuple(i -> i + nx, Val(nu)))])
    ℓxx = SMatrix{nx,nx,T}(H[SVector{nx,Int}(ntuple(identity, Val(nx))),
                              SVector{nx,Int}(ntuple(identity, Val(nx)))])
    ℓuu = SMatrix{nu,nu,T}(H[SVector{nu,Int}(ntuple(i -> i + nx, Val(nu))),
                              SVector{nu,Int}(ntuple(i -> i + nx, Val(nu)))])
    ℓux = SMatrix{nu,nx,T}(H[SVector{nu,Int}(ntuple(i -> i + nx, Val(nu))),
                              SVector{nx,Int}(ntuple(identity, Val(nx)))])
    return ℓx, ℓu, ℓxx, ℓuu, ℓux
end

# ── Terminal cost derivatives ────────────────────────────────────────

"""
    terminal_cost_derivatives(ϕ, x) -> (ϕx, ϕxx)
"""
function terminal_cost_derivatives(ϕ, x::SVector{nx,T}) where {nx, T}
    ϕx  = SVector{nx,T}(ForwardDiff.gradient(ϕ, x))
    ϕxx = SMatrix{nx,nx,T}(ForwardDiff.hessian(ϕ, x))
    return ϕx, ϕxx
end

# ── Dynamics linearisation ───────────────────────────────────────────

"""
    dynamics_derivatives(f, x, u, tk, tkp1) -> (fx, fu)

Jacobians of `f(x, u, tk, tkp1)` w.r.t. `x` and `u`.
"""
function dynamics_derivatives(f, x::SVector{nx,T}, u::SVector{nu,T},
                              tk, tkp1) where {nx, nu, T}
    fx = ForwardDiff.jacobian(w -> f(w, u, tk, tkp1), x)
    fu = ForwardDiff.jacobian(w -> f(x, w, tk, tkp1), u)
    return SMatrix{nx,nx,T}(fx), SMatrix{nx,nu,T}(fu)
end

# ── Constraint derivatives ──────────────────────────────────────────

"""
    constraint_derivatives(g, x, u, t) -> (gval, gx, gu)

Value and Jacobians of a path constraint `g(x,u,t)`.
"""
function constraint_derivatives(g, x::SVector{nx,T}, u::SVector{nu,T},
                                t, ::Val{p}) where {nx, nu, T, p}
    gval = SVector{p,T}(g(x, u, t))
    gx   = SMatrix{p,nx,T}(ForwardDiff.jacobian(w -> g(w, u, t), x))
    gu   = SMatrix{p,nu,T}(ForwardDiff.jacobian(w -> g(x, w, t), u))
    return gval, gx, gu
end

"""
    terminal_constraint_derivatives(ψ, x) -> (ψval, ψx, ψxx)

Value, Jacobian and (vector of) Hessians of terminal constraint `ψ(x)`.
"""
function terminal_constraint_derivatives(ψ, x::SVector{nx,T},
                                         ::Val{r}) where {nx, T, r}
    ψval = SVector{r,T}(ψ(x))
    ψx   = SMatrix{r,nx,T}(ForwardDiff.jacobian(ψ, x))
    return ψval, ψx
end
