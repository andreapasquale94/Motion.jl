
abstract type AbstractStateN{N, T} <: FieldVector{N, T} end

abstract type AbstractState6{T} <: AbstractStateN{6, T} end

# ----
# Representations 

struct Adim{N} <: AbstractState6{N}
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
# CR3BP

function translate_state(c::Adim{N}, p::CR3BPSystemProperties, ::Val{:primary}) where N 
    return Adim{N}(c.pox+p.μ, c.poy, c.poz, c.vex, c.vey, c.vez)
end

function translate_state(c::Adim{N}, p::CR3BPSystemProperties, ::Val{:secondary}) where N 
    return Adim{N}(c.pox-1+p.μ, c.poy, c.poz, c.vex, c.vey, c.vez)
end

function convert_state(::Type{Cart}, c::Adim{N}, p::CR3BPSystemProperties, args...) where N 
    lu = p.L
    vu = p.L/p.T
    return Cart{N}(
        c.pox*lu, c.poy*lu, c.poz*lu, c.vex*vu, c.vey*vu, c.vez*vu
    )
end

function convert_state(::Type{Adim}, c::Cart{N}, p::CR3BPSystemProperties, args...) where N 
    lu = p.L
    vu = p.L/p.T
    return Adim{N}(
        c.pox/lu, c.poy/lu, c.poz/lu, c.vex/vu, c.vey/vu, c.vez/vu
    )
end

function convert_state(::Type{Cart}, s::Adim{N}, p::CR3BPSystemProperties, val::Val{S}) where {N, S}
    return convert_state(Cart, translate_state(s, p, val), p)
end

function convert_state(::Type{Coe}, s::Adim{N}, p::CR3BPSystemProperties, ::Val{:primary}) where N 
    return convert6_cart_to_coe(convert_state(Cart, s, p, val), p.GM1)
end

function convert_state(::Type{Coe}, s::Adim{N}, p::CR3BPSystemProperties, ::Val{:secondary}) where N 
    return convert6_cart_to_coe(convert_state(Cart, s, p, val), p.GM2)
end

function convert_state(::Type{Adim}, s::Coe{N}, p::CR3BPSystemProperties, ::Val{:primary}) where N 
    adim = convert_state(Adim, convert_state(Cart, s, p), p)
    return Adim{N}(adim.pox-p.μ, adim.poy, adim.poz, adim.vex, adim.vey, adim.vez)
end

function convert_state(::Type{Adim}, s::Coe{N}, p::CR3BPSystemProperties, ::Val{:secondary}) where N 
    adim = convert_state(Adim, convert_state(Cart, s, p), p)
    return Adim{N}(adim.pox+1-p.μ, adim.poy, adim.poz, adim.vex, adim.vey, adim.vez)
end

function convert_state(::Type{Adim}, s::CoeRad{N}, p::CR3BPSystemProperties, val::Val{S}) where {N, S} 
    coe = convert6_coerad_to_coe(s)
    return convert_state(Adim, coe, p, val)
end
