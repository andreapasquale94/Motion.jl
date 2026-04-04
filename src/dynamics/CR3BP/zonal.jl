# ──────────────────────────────────────────────────────────────────────────────
#  Zonal gravity perturbations  (A₂, A₄, A₆)
# ──────────────────────────────────────────────────────────────────────────────
#  Aℓ = Jℓ (R/d)^ℓ   pre-normalised, stored directly in the parameter array.
 
@fastmath function gravity_A2(x::AbstractVector{T}, μ, A2) where T
    x, y, z = x[1], x[2], x[3]
    r2  = x*x + y*y + z*z
    ir2 = inv(r2)
    ir  = sqrt(ir2)
    s2  = z*z * ir2
 
    f = T(3)/2 * μ * A2 * ir2 * ir2 * ir          # 3/2 · μ A₂ / r⁵
 
    cxy = f * (5*s2 - 1)
    cz  = f * (5*s2 - 3)
 
    return SVector{3,T}(x*cxy, y*cxy, z*cz)
end

@fastmath function gravity_A24(x::AbstractVector{T}, μ, A2, A4) where T
    x, y, z = x[1], x[2], x[3]
    r2  = x*x + y*y + z*z
    ir2 = inv(r2)
    ir  = sqrt(ir2)
    s2  = z*z * ir2
    s4  = s2 * s2
 
    Mr3 = μ * ir2 * ir
    f2  = T(3)/2 * A2 * Mr3 * ir2
    f4  = T(5)/8 * A4 * Mr3 * ir2 * ir2
 
    cxy = f2 * (5*s2 - 1) +
          f4 * (63*s4 - 42*s2 + 3)
    cz  = f2 * (5*s2 - 3) +
          f4 * (63*s4 - 70*s2 + 15)
 
    return SVector{3,T}(x*cxy, y*cxy, z*cz)
end


@fastmath function gravity_A246(x::AbstractVector{T}, μ, A2, A4, A6) where T
    x, y, z = x[1], x[2], x[3]
    r2  = x*x + y*y + z*z
    ir2 = inv(r2)
    ir  = sqrt(ir2)
    s2  = z*z * ir2
    s4  = s2 * s2
    s6  = s4 * s2
 
    Mr3 = μ * ir2 * ir
    ir4 = ir2 * ir2
    f2  = T(3)/2  * A2 * Mr3 * ir2
    f4  = T(5)/8  * A4 * Mr3 * ir4
    f6  = T(7)/16 * A6 * Mr3 * ir4 * ir2
 
    cxy = f2 * (5*s2 - 1) +
          f4 * (63*s4 - 42*s2 + 3) +
          f6 * (429*s6 - 495*s4 + 135*s2 - 5)
    cz  = f2 * (5*s2 - 3) +
          f4 * (63*s4 - 70*s2 + 15) +
          f6 * (429*s6 - 693*s4 + 315*s2 - 35)
 
    return SVector{3,T}(x*cxy, y*cxy, z*cz)
end

function mean_motion(p::ComponentArray)
    n2 = 1.0
    for key in (:primary, :secondary)
        hasproperty(p, key) || continue
        body = getproperty(p, key)
        hasproperty(body, :A2) && (n2 +=  3/2  * getproperty(body, :A2))
        hasproperty(body, :A4) && (n2 -= 15/8  * getproperty(body, :A4))
        hasproperty(body, :A6) && (n2 += 35/16 * getproperty(body, :A6))
    end
    return sqrt(n2)
end

@inline function zonal_gravity(r⃗::SVector{3,T}, μ::T, body) where T
    if hasproperty(body, :A6)
        return gravity_A246(r⃗, μ, T(body.A2), T(body.A4), T(body.A6))
    elseif hasproperty(body, :A4)
        return gravity_A24(r⃗, μ, T(body.A2), T(body.A4))
    elseif hasproperty(body, :A2)
        return gravity_A2(r⃗, μ, T(body.A2))
    else
        return SVector{3,T}(zero(T), zero(T), zero(T))
    end
end

# ──────────────────────────────────────────────────────────────────────────────
#  Libration points
# ──────────────────────────────────────────────────────────────────────────────

"""
    _equatorial_correction(r, body) → H(r)
 
Zonal correction factor at the equatorial plane (s = 0).
The effective gravitational 1/r² force is multiplied by (1 + H).
"""
@inline function _equatorial_correction(r, body)
    ir2 = inv(r * r)
    h = zero(typeof(r))
    hasproperty(body, :A2) && (h +=  3/2  * body.A2 * ir2)
    hasproperty(body, :A4) && (h -= 15/8  * body.A4 * ir2 * ir2)
    hasproperty(body, :A6) && (h += 35/16 * body.A6 * ir2 * ir2 * ir2)
    return h
end
 
@inline function _h1h2(r1, r2, p)
    h1 = hasproperty(p, :primary)   ? _equatorial_correction(r1, p.primary)   : zero(typeof(r1))
    h2 = hasproperty(p, :secondary) ? _equatorial_correction(r2, p.secondary) : zero(typeof(r2))
    return h1, h2
end

@inline function _compute_libration_point(p::ComponentArray, ::Val{:L1}, tol = 1e-14)
    μ  = p.μ
    n2 = (hasproperty(p, :n) ? p.n : mean_motion(p))^2
 
    f = (ξ, _) -> begin
        r1 = 1 - ξ          # distance from primary
        r2 = ξ              # distance from secondary
        h1, h2 = _h1h2(r1, r2, p)
        n2 * (1 - μ - ξ) - (1 - μ) / (1 - ξ)^2 * (1 + h1) + μ / ξ^2 * (1 + h2)
    end
 
    prob = IntervalNonlinearProblem(f, (1e-6, 1.0 - 1e-6))
    sol  = solve(prob, Ridder(); abstol = tol, reltol = tol)
    return SVector{6,typeof(μ)}(1 - μ - sol.u, 0, 0, 0, 0, 0)
end

@inline function _compute_libration_point(p::ComponentArray, ::Val{:L2}, tol = 1e-14)
    μ  = p.μ
    n2 = (hasproperty(p, :n) ? p.n : mean_motion(p))^2
 
    f = (ξ, _) -> begin
        r1 = 1 + ξ
        r2 = ξ
        h1, h2 = _h1h2(r1, r2, p)
        n2 * (1 - μ + ξ) - (1 - μ) / (1 + ξ)^2 * (1 + h1) - μ / ξ^2 * (1 + h2)
    end
 
    prob = IntervalNonlinearProblem(f, (1e-6, 0.5))
    sol  = solve(prob, Ridder(); abstol = tol, reltol = tol)
    return SVector{6,typeof(μ)}(1 - μ + sol.u, 0, 0, 0, 0, 0)
end

@inline function _compute_libration_point(p::ComponentArray, ::Val{:L3}, tol = 1e-14)
    μ  = p.μ
    n2 = (hasproperty(p, :n) ? p.n : mean_motion(p))^2
 
    f = (ξ, _) -> begin
        r1 = ξ              # distance from primary
        r2 = 1 + ξ          # distance from secondary
        h1, h2 = _h1h2(r1, r2, p)
        n2 * (-μ - ξ) + (1 - μ) / ξ^2 * (1 + h1) - μ / (1 + ξ)^2 * (1 + h2)
    end
 
    prob = IntervalNonlinearProblem(f, (0.5, 1.5))
    sol  = solve(prob, Ridder(); abstol = tol, reltol = tol)
    return SVector{6,typeof(μ)}(-μ - sol.u, 0, 0, 0, 0, 0)
end

function _compute_libration_point(p::ComponentArray, ::Val{:L4}, tol = 1e-14)
    μ  = p.μ
    n2 = (hasproperty(p, :n) ? p.n : mean_motion(p))^2
 
    F = (u, _) -> begin
        x, y = u[1], u[2]
        r1 = sqrt((x + μ)^2 + y^2)
        r2 = sqrt((x - 1 + μ)^2 + y^2)
        h1, h2 = _h1h2(r1, r2, p)
        c1 = (1 - μ) / r1^3 * (1 + h1)
        c2 = μ / r2^3 * (1 + h2)
        SVector(n2 * x - c1 * (x + μ) - c2 * (x - 1 + μ),
                (n2 - c1 - c2) * y)
    end
 
    prob = NonlinearProblem(F, SVector(0.5 - μ, sqrt(3.0) / 2))
    sol  = solve(prob, NewtonRaphson(); abstol = tol, reltol = tol)
    return SVector{6,typeof(μ)}(sol.u[1], sol.u[2], 0, 0, 0, 0)
end

function _compute_libration_point(p::ComponentArray, ::Val{:L5}, tol = 1e-14)
    μ  = p.μ
    n2 = (hasproperty(p, :n) ? p.n : mean_motion(p))^2
 
    F = (u, _) -> begin
        x, y = u[1], u[2]
        r1 = sqrt((x + μ)^2 + y^2)
        r2 = sqrt((x - 1 + μ)^2 + y^2)
        h1, h2 = _h1h2(r1, r2, p)
        c1 = (1 - μ) / r1^3 * (1 + h1)
        c2 = μ / r2^3 * (1 + h2)
        SVector(n2 * x - c1 * (x + μ) - c2 * (x - 1 + μ),
                (n2 - c1 - c2) * y)
    end
 
    prob = NonlinearProblem(F, SVector(0.5 - μ, -sqrt(3.0) / 2))
    sol  = solve(prob, NewtonRaphson(); abstol = tol, reltol = tol)
    return SVector{6,typeof(μ)}(sol.u[1], sol.u[2], 0, 0, 0, 0)
end

# ──────────────────────────────────────────────────────────────────────────────
#  Jacobi constant
# ──────────────────────────────────────────────────────────────────────────────

@inline function _body_zonal_potential(r, z, M, body)
    ir  = inv(r)
    ir2 = ir * ir
    s2   = z * z * ir2
 
    V = zero(typeof(r))
    if hasproperty(body, :A2)
        V -= M * body.A2 * ir2 * ir * (3*s2 - 1) / 2
    end
    if hasproperty(body, :A4)
        V -= M * body.A4 * ir2 * ir2 * ir * (s2 * (35*s2 - 30) + 3) / 8
    end
    if hasproperty(body, :A6)
        V -= M * body.A6 * ir2 * ir2 * ir2 * ir * (s2 * (s2 * (231*s2 - 315) + 105) - 5) / 16
    end
    return V
end

@fastmath function jacobi_constant(x::AbstractVector{T}, p::ComponentArray{<:Number}) where {T}
    μ  = T(getproperty(p, :μ))
    μ1 = one(T) - μ
    n  = hasproperty(p, :n) ? T(getproperty(p, :n)) : one(T)
 
    @inbounds begin
        px, py, pz = x[1], x[2], x[3]
        vx, vy, vz = x[4], x[5], x[6]
 
        Δx1 = px + μ
        Δx2 = px - μ1
        r1  = sqrt(Δx1*Δx1 + py*py + pz*pz)
        r2  = sqrt(Δx2*Δx2 + py*py + pz*pz)
        vsq = vx*vx + vy*vy + vz*vz
 
        CJ = n*n*(px*px + py*py) + 2*(μ1/r1 + μ/r2) - vsq
 
        if hasproperty(p, :primary)
            CJ += 2 * _body_zonal_potential(r1, pz, μ1, getproperty(p, :primary))
        end
        if hasproperty(p, :secondary)
            CJ += 2 * _body_zonal_potential(r2, pz, μ, getproperty(p, :secondary))
        end
        return CJ
    end
end