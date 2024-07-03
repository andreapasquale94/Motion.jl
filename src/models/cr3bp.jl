
function centrifugal_potential(x, μ)
    @inbounds px, py, pz = x[1], x[2], x[3]
    p₁ = px+μ
    p₂ = px-1+μ

    tmp = py*py + pz*pz
    r₁ = sqrt(p₁*p₁ + tmp)
    r₂ = sqrt(p₂*p₂ + tmp)
    return 0.5*(px^2 + py^2) + (1 - μ)/r₁ + μ/r₂  + 0.5*(1 - μ)*μ
end

@fastmath function rhs_cr3bp(x, t, μ)
    @inbounds px, py, pz, vx, vy, vz = x[1], x[2], x[3], x[4], x[5], x[6]
    p₁ = px+μ
    p₂ = px-1+μ

    tmp = py*py + pz*pz
    r₁ = sqrt(p₁*p₁ + tmp)
    r₂ = sqrt(p₂*p₂ + tmp)

    r₁³ = r₁*r₁*r₁
    r₂³ = r₂*r₂*r₂

    f₁ = (1-μ)/r₁³
    f₂ = μ/r₂³
    
    return SVector{6}( 
        vx, vy, vz, 
        px + 2*vy - p₁*f₁ - p₂*f₂, 
        py - 2*vx - py*(f₁ + f₂), 
        - pz*(f₁ + f₂)
    )
end

@fastmath function jacobian_cr3bp(x, t, μ)
    @inbounds px, py, pz = x[1], x[2], x[3]

    p₁ = px+μ
    p₂ = px-1+μ

    tmp = py*py + pz*pz
    r₁ = sqrt(p₁*p₁ + tmp)
    r₂ = sqrt(p₂*p₂ + tmp)

    r₁² = r₁*r₁
    r₂² = r₂*r₂
    r₁³ = r₁²*r₁
    r₂³ = r₂²*r₂

    f₁3 = (1-μ)/r₁³
    f₂3 = μ/r₂³
    f₁5 = f₁3/r₁²
    f₂5 = f₂3/r₂²

    tmp = f₁5 + f₂5
    uxx = 1. - f₁3 - f₂3 + 3*p₁*p₁*f₁5 + 3*p₂*p₂*f₂5
    uyy = 1. - f₁3 - f₂3 + 3*py*py*tmp
    uzz = - f₁3 - f₂3 + 3*pz*pz*tmp

    uyz = 3*py*pz*tmp
    tmp = p₁*f₁5 + p₂*f₂5
    uxy = 3*py*tmp
    uxz = 3*pz*tmp

    return SMatrix{6, 6}(
        0.,     0.,     0.,     1.,     0.,     0.,
        0.,     0.,     0.,     0.,     1.,     0.,
        0.,     0.,     0.,     0.,     0.,     1.,
        uxx,   uxy,    uxz,     0.,     2.,     0., 
        uxy,   uyy,    uyz,    -2.,     0.,     0.,
        uxz,   uyz,    uzz,     0.,     0.,     0.
    )'

end

@fastmath function rhs(x, p::CR3BPSystemProperties, t)
    μ = p.μ
    return rhs_cr3bp(x, t, μ)
end

@fastmath function jacobian(x, p::CR3BPSystemProperties, t)
    μ = p.μ
    return jacobian_cr3bp(x, t, μ)
end

function rhs_stm(x, p::CR3BPSystemProperties, t)
    # State transition matrix
    Φ = reshape(@view(x[7:end]), Size(6, 6))

    # Compute state derivative
    δx = rhs(x, p, t)

    # Compute jacobian
    J = jacobian(x, p, t)
    # Compute stm derivative
    δΦ = J * Φ

    return vcat(δx, reshape(δΦ, Size(36)))
end