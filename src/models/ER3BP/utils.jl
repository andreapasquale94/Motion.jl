"""
    __er3bp_scale(f, e) -> T

Return the pulsating scale factor for ER3BP in true-anomaly formulation:

`k(f) = 1 / (1 + e*cos(f))`.
"""
@inline function __er3bp_scale(f::T, e::T) where {T}
    return inv(one(T) + e * cos(f))
end

"""
    jacobian(x, f, μ, e) -> SMatrix{6,6,T}

State Jacobian for ER3BP rotating-pulsating dynamics evaluated at true anomaly `f`.
"""
@fastmath function jacobian(x::AbstractVector{T}, f::Number, μ::Number, e::Number) where {T}
    μT = T(μ)
    eT = T(e)
    k  = __er3bp_scale(T(f), eT)

    @inbounds begin
        px, py, pz = x[1], x[2], x[3]
        px1 = px + μT
        px2 = px - (one(T) - μT)

        tmp = py*py + pz*pz
        r1 = sqrt(px1*px1 + tmp)
        r2 = sqrt(px2*px2 + tmp)

        r1² = r1*r1
        r2² = r2*r2
        r1³ = r1²*r1
        r2³ = r2²*r2

        f13 = (one(T) - μT) / r1³
        f23 = μT / r2³
        f15 = f13 / r1²
        f25 = f23 / r2²

        tmp2 = f15 + f25
        uxx = one(T) - f13 - f23 + 3*px1*px1*f15 + 3*px2*px2*f25
        uyy = one(T) - f13 - f23 + 3*py*py*tmp2
        uzz = - f13 - f23 + 3*pz*pz*tmp2

        uyz = 3*py*pz*tmp2
        tmp3 = px1*f15 + px2*f25
        uxy = 3*py*tmp3
        uxz = 3*pz*tmp3

        return SMatrix{6, 6, T}(
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            k*uxx, k*uxy, k*uxz, 0.0, 2.0, 0.0,
            k*uxy, k*uyy, k*uyz, -2.0, 0.0, 0.0,
            k*uxz, k*uyz, k*uzz, 0.0, 0.0, 0.0,
        )'
    end
end
