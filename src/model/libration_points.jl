@inline function _compute_libration_point(μ::Number, ::Val{:L1}, tol=1e-14)
    f = x -> 1.0 -μ -x -(1-μ)/(1-x)^2 +μ/x^2
    _, x = ridder(f, 0.0, 0.5, 0.0, tol)
    return SVector{6, typeof(μ)}(
        1 - μ - x, 0, 0, 
        0, 0, 0
    )
end

@inline function _compute_libration_point(μ, ::Val{:L2}, tol=1e-14)
    f = x -> 1.0 -μ +x -(1-μ)/(1+x)^2 -μ/x^2
    _, x = ridder(f, 0.0, 0.5, 0.0, tol)
    return SVector{6, typeof(μ)}(
        1 - μ + x, 0, 0, 
        0, 0, 0
    )
end

@inline function _compute_libration_point(μ, ::Val{:L3}, tol=1e-14)
    f = x -> -μ -x +(1-μ)/x^2 -μ/(1+x)^2
    _, x = ridder(f, 0.5, 1.5, 0.0, tol)
    return SVector{6, typeof(μ)}(
        - μ - x, 0, 0, 
        0, 0, 0
    )
end

@inline function _compute_libration_point(μ, ::Val{:L4}, args...)
    return SVector{6, typeof(μ)}(
        0.5 - μ, 0.5*sqrt(3), 0, 0, 0, 0
    )
end

@inline function _compute_libration_point(μ, ::Val{:L5}, args...)
    return SVector{6, typeof(μ)}(
        0.5 - μ, -0.5*sqrt(3), 0, 0, 0, 0
    )
end

@inline function compute_libration_point(μ::Number, val::Val{S}, tol=1e-14) where S
    return _compute_libration_point(μ, val, tol)
end

"""
    libration_points(μ::Number; tol=1e-14)

Return the five CR3BP libration points (L1–L5) for mass parameter `μ`,
as 6-element state vectors in the rotating frame. The points are returned
in order L1, L2, L3, L4, L5.
"""
function libration_points(μ::Number; tol=1e-14)
    return [
        compute_libration_point(μ, Val(:L1), tol),
        compute_libration_point(μ, Val(:L2), tol),
        compute_libration_point(μ, Val(:L3), tol),
        compute_libration_point(μ, Val(:L4), tol),
        compute_libration_point(μ, Val(:L5), tol)
    ]
end
