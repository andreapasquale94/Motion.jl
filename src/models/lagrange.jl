
@inline function _compute_libration_point(μ::Number, ::Val{:L1}, tol=1e-14)
    f = x -> 1.0 -μ -x -(1-μ)/(1-x)^2 +μ/x^2
    _, x = ridder(f, 0.0, 0.5, 0.0, tol)
    return Adim{typeof(μ)}(
        1 - μ - x, 0, 0, 
        0, 0, 0
    )
end

@inline function _compute_libration_point(μ, ::Val{:L2}, tol=1e-14)
    f = x -> 1.0 -μ +x -(1-μ)/(1+x)^2 -μ/x^2
    _, x = ridder(f, 0.0, 0.5, 0.0, tol)
    return Adim{typeof(μ)}(
        1 - μ + x, 0, 0, 
        0, 0, 0
    )
end

@inline function _compute_libration_point(μ, ::Val{:L3}, tol=1e-14)
    f = x -> -μ -x +(1-μ)/x^2 -μ/(1+x)^2
    _, x = ridder(f, 0.5, 1.5, 0.0, tol)
    return Adim{typeof(μ)}(
        - μ - x, 0, 0, 
        0, 0, 0
    )
end

@inline function _compute_libration_point(μ, ::Val{:L4}, args...)
    return Adim{typeof(μ)}(
        0.5 - μ, 0.5*sqrt(3), 0, 0, 0, 0
    )
end

@inline function _compute_libration_point(μ, ::Val{:L5}, args...)
    return Adim{typeof(μ)}(
        0.5 - μ, -0.5*sqrt(3), 0, 0, 0, 0
    )
end

@inline function compute_libration_point(p::CR3BPSystemProperties, val::Val{S}, 
    tol=1e-14
) where S
    return _compute_libration_point(p.μ, val, tol)
end

function libration_points(p::CR3BPSystemProperties; tol=1e-14)
    lps = [
        compute_libration_point(p, Val(:L1), tol),
        compute_libration_point(p, Val(:L2), tol),
        compute_libration_point(p, Val(:L3), tol),
        compute_libration_point(p, Val(:L4), tol),
        compute_libration_point(p, Val(:L5), tol)
    ]
    return lps
end