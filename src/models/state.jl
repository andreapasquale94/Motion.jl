
abstract type AbstractStateN{N, T} <: FieldVector{N, T} end

abstract type AbstractState6{T} <: AbstractStateN{6, T} end

abstract type AbstractAdimState6{T} <: AbstractState6{T} end

# ----
# Representations 

# This is the CR3BP representation used for the state vector
# It is non-dimensional units, centered at the barycenter of the three body system
struct Adim{N} <: AbstractAdimState6{N}
    pox::N 
    poy::N 
    poz::N 
    vex::N 
    vey::N 
    vez::N
end


# The following state representations are all inertial 
# They are centered either at the primary or the secondary
struct AdimCart{N} <: AbstractAdimState6{N}
    pox::N 
    poy::N 
    poz::N 
    vex::N 
    vey::N 
    vez::N
end

struct Cart{N} <: AbstractState6{N}
    pox::N 
    poy::N 
    poz::N 
    vex::N 
    vey::N 
    vez::N
end

struct Coe{N} <: AbstractState6{N}
    sma::N 
    ecc::N 
    inc::N 
    ran::N
    aop::N 
    tra::N
end

struct CoeRad{N} <: AbstractState6{N}
    rpe::N 
    rap::N 
    inc::N 
    ran::N 
    aop::N 
    tra::N
end

# ----
# Basic conversions

@fastmath function convert6_coerad_to_coe(sv::AbstractVector{T}) where T 
    sma = 0.5 * (sv[1] + sv[2])
    ecc = (sv[2] - sv[1])/(sv[1] + sv[2])
    return Coe{T}(sma, ecc, sv[3], sv[4], sv[5], sv[6])
end

@fastmath function convert6_cart_to_coe(sv::AbstractVector{T}, μ::Number) where T
    @inbounds R = SVector{3}(sv[1], sv[2], sv[3])
    @inbounds V = SVector{3}(sv[4], sv[5], sv[6])

    r = norm(R)
    v² = dot(V, V)
    H = cross(R, V) # momentum vector
    h² = dot(H, H)
    h = sqrt(h²)
    ĥ = H/h

    E = cross(V, H)/μ - R/r # eccentricity vector 
    p = h²/μ # semilatur rectum

    sma = (r*μ)/(2*μ - r*v²) # semimajor axis
    ecc = norm(E)  # eccentricity
    inc = acos(ĥ[3]) # inclination 

    # check inclination is zero
    if H[1]*H[1] + H[2]*H[2] > 1e-12 
        aol = atan(h*R[3], R[2]*H[1] - R[1]*H[2])
        ran = mod2pi( atan(ĥ[1], -ĥ[2]) )
    else 
        aol = atan(R[2], R[1]) * sign(H[3]) # true longitude
        ran = 0.0
    end

    # check circular orbits 
    if ecc > 1e-14
        tra = atan(sqrt(p/μ)*dot(R, V), p-r)
        aop = mod2pi(aol - tra)
    else 
        tra = mod2pi(aol) 
        aop = 0.
    end

    return Cart{T}(sma, ecc, inc, ran, aop, tra)
end

@fastmath function convert6_coe_to_cart(sv::AbstractVector{T}, μ::Number) where T
    @inbounds sma, ecc, inc, ran, aop, tra = @views(sv[1:6]) 
    p = sma*(1-ecc*ecc)

    stan, ctan = sincos(tra)
    r = p/(1 + ecc*ctan)
    sran, cran = sincos(ran)
    sinc, cinc = sincos(inc)
    saop, caop = sincos(aop)
    s1 = saop*cinc 
    s2 = caop*cinc

    c1x = caop*cran - s1*sran
    c1y = saop*cran + s2*sran
    c2x = caop*sran + s1*cran 
    c2y = s2*cran - saop*sran
    c3x = saop*sinc 
    c3y = caop*sinc

    rx = r*ctan
    ry = r*stan
    μop = sqrt(μ/p) 
    vx = -μop*stan
    vy = μop*(ecc + ctan)

    return Cart{T}(
        rx*c1x - ry*c1y, rx*c2x + ry*c2y, rx*c3x + ry*c3y,
        vx*c1x - vy*c1y, vx*c2x + vy*c2y, vx*c3x + vy*c3y
    )
end

# ----
# Translation in non-dim units 

function translate(c::Adim{N}, p::CR3BPSystemProperties, ::Val{:primary}) where N 
    return Adim{N}(c.pox+p.μ, c.poy, c.poz, c.vex, c.vey, c.vez)
end

function translate(c::Adim{N}, p::CR3BPSystemProperties, ::Val{:secondary}) where N 
    return Adim{N}(c.pox-1+p.μ, c.poy, c.poz, c.vex, c.vey, c.vez)
end

# ----
# Simple state rotations (inertial to synodic)

function rotate(::Type{Adim}, s::AdimCart{N}, θ::Number, ∂θ::Number=1.0) where N 
    sθ, cθ = sincos(θ)
    return Adim{N}(
        s.pox*cθ + s.poy*sθ,
        -s.pox*sθ + s.poy*cθ,
        s.poz,
        -s.pox*∂θ*sθ + s.poy*∂θ*cθ + s.vex*cθ + s.vey*sθ,
        -s.pox*∂θ*cθ - s.poy*∂θ*sθ - s.vex*sθ + s.vey*cθ,
        s.vez
    )
end

function rotate(::Type{AdimCart}, s::Adim{N}, θ::Number, ∂θ::Number=1.0) where N 
    sθ, cθ = sincos(θ)
    return AdimCart{N}(
        s.pox*cθ - s.poy*sθ,
        s.pox*sθ + s.poy*cθ,
        s.poz,
        -s.pox*∂θ*sθ - s.poy*∂θ*cθ + s.vex*cθ - s.vey*sθ,
        s.pox*∂θ*cθ - s.poy*∂θ*sθ + s.vex*sθ + s.vey*cθ,
        s.vez
    )
end

# ----
# Transformations between representations 

@inline transform(::Type{A}, s::A, args...) where {A <: AbstractState6} = s
    
function transform(::Type{Cart}, s::AdimCart{N}, p::CR3BPSystemProperties, args...) where N 
    lu = p.L
    vu = p.L/p.T
    return Cart{N}(
        s.pox*lu, s.poy*lu, s.poz*lu, s.vex*vu, s.vey*vu, s.vez*vu
    )
end

function transform(::Type{AdimCart}, s::Cart{N}, p::CR3BPSystemProperties, args...) where N 
    lu = p.L
    vu = p.L/p.T
    return Adim{N}(
        s.pox/lu, s.poy/lu, s.poz/lu, s.vex/vu, s.vey/vu, s.vez/vu
    )
end

# primary/secondary-centric transformations

function transform(
    ::Type{Coe}, s::AdimCart{N}, p::CR3BPSystemProperties, ::Val{:primary}
) where N 
    dim = transform(Cart, s, p, val)
    return convert6_cart_to_coe(dim, p.GM1)
end

function transform(
    ::Type{Coe}, s::AdimCart{N}, p::CR3BPSystemProperties, ::Val{:secondary}
) where N 
    dim = transform(Cart, s, p, val)
    return convert6_cart_to_coe(dim, p.GM2)
end

function transform(
    ::Type{AdimCart}, s::Coe{N}, p::CR3BPSystemProperties, ::Val{:primary}
) where N 
    cart = convert6_coe_to_cart(s, p.GM1)
    adim = transform(AdimCart, cart, p)
    return AdimCart{N}(adim.pox-p.μ, adim.poy, adim.poz, adim.vex, adim.vey, adim.vez)
end

function transform(
    ::Type{AdimCart}, s::Coe{N}, p::CR3BPSystemProperties, ::Val{:secondary}
) where N 
    cart = convert6_coe_to_cart(s, p.GM2)
    adim = transform(AdimCart, cart, p)
    return AdimCart{N}(adim.pox+1-p.μ, adim.poy, adim.poz, adim.vex, adim.vey, adim.vez)
end

function transform(
    ::Type{AdimCart}, s::CoeRad{N}, p::CR3BPSystemProperties, val
) where {N} 
    coe = convert6_coerad_to_coe(s)
    return transform(AdimCart, coe, p, val)
end

function transform(
    ::Type{Adim}, s::R, p::CR3BPSystemProperties, val::Val{:primary}, θ::Number=0.0
) where {R <: AbstractState6}
    x = rotate(Adim, transform(AdimCart, s, p, val), θ, 1.0)
    return x - SA[0, 0, 0, 0, -p.μ, 0] # translate velocity to barycenter 
end

function transform(
    ::Type{Adim}, s::R, p::CR3BPSystemProperties, val::Val{:secondary}, θ::Number=0.0
) where {R <: AbstractState6}
    x = rotate(Adim, transform(AdimCart, s, p, val), θ, 1.0)
    return x + SA[0, 0, 0, 0, 1-p.μ, 0] # translate velocity to barycenter
end