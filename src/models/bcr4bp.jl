
function rhs_bcr4bp_spm(x, t, μ, μ₃, l, ω)
    @inbounds px, py, pz, vx, vy, vz, θ = x[1], x[2], x[3], x[4], x[5], x[6], x[7]

    # sun 
    rsv = SVector{3}(px-μ, py, pz)
    # planet-moon barycenter 
    rb = SVector{3}(px-1+μ, py, pz)

    # vector pointing from planet to moon
    sθ, cθ = sincos(θ) 
    vpm = SVector{3}(cθ, sθ, 0)

    # planet 
    rpv = rb + l*μ₃*vpm 
    # moon 
    rmv = rb - l*(1-μ₃)*vpm

    rs3 = norm(rsv)^3
    rm3 = norm(rmv)^3 
    rp3 = norm(rpv)^3

    return SVector{7}(
        vx, vy, vz,
        2vy + px - (1-μ)*rsv[1]/rs3 - μ*(1-μ₃)*rpv[1]/rp3 - μ*μ₃*rmv[1]/rm3,
        -2vx + py - (1-μ)*rsv[2]/rs3 - μ*(1-μ₃)*rpv[2]/rp3 - μ*μ₃*rmv[2]/rm3,
        -(1-μ)*rsv[3]/rs3 - μ*(1-μ₃)*rpv[3]/rp3 - μ*μ₃*rmv[3]/rm3,
        ω
    )
end

function rhs_bcr4bp_pms(x, t, μ, μ₃, l, ω)
    @inbounds θ = x[7]
    sθ, cθ = sincos(θ)
    rsv = SVector{3}(x[1]-l*cθ, x[2]-l*sθ, x[3])
    rs3 = norm(rsv)^3

    δx = rhs_cr3bp(x, t, μ)
    return SVector{7}(
        δx[1],
        δx[2],
        δx[3],
        δx[4] - μ₃*rsv[1]/rs3 - μ₃*cθ/l^2,
        δx[5] - μ₃*rsv[2]/rs3 - μ₃*sθ/l^2,
        δx[6] - μ₃*rsv[3]/rs3,
        ω
    )
end

function rhs_spm(x, p::BCR4BPSystemProperties, t)
    μ, μ3, a3, ω3 = p.μ, p.μ3, p.a3, p.ω3
    return rhs_bcr4bp_spm(x, t, μ, μ3, a3, ω3)
end

function rhs_pms(x, p::BCR4BPSystemProperties, t)
    μ, μ3, a3, ω3 = p.μ, p.μ3, p.a3, p.ω3
    return rhs_bcr4bp_pms(x, t, μ, μ3, a3, ω3)
end
