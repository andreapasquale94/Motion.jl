"""
    __lvlh_basis(r::SVector{3,T}, v::SVector{3,T}) -> (R̂, T̂, N̂)

Compute the right-handed RTN/LVLH basis from position `r` and velocity `v`.

Definitions (all in the same frame as `r,v`):
- `R̂` (radial)   = `r / ‖r‖`
- `N̂` (normal)   = `(r × v) / ‖r × v‖`
- `T̂` (tangential, along-track) = `N̂ × R̂`

A small `eps(T)` is added to norms for numerical robustness near singularities.
"""
@inline function __lvlh_basis(r::SVector{3,T}, v::SVector{3,T}) where {T}
    ϵ  = eps(T)

    rn = sqrt(dot(r, r) + ϵ)
    R̂  = r / rn

    h  = cross(r, v)
    hn = sqrt(dot(h, h) + ϵ)
    N̂  = h / hn

    T̂  = cross(N̂, R̂)  # completes RTN triad
    return R̂, T̂, N̂
end

"""
    __acceleration_to_cartesian(x, p) -> SVector{3,T}

Convert an acceleration specification `p` into a 3D Cartesian acceleration vector.

`x` is the current state and is only used for RTN/LVLH-based representations.
It is assumed to be ordered as `[rx, ry, rz, vx, vy, vz, ...]`.

The returned scalar type `T` is chosen as:

`T = promote_type(eltype(x), eltype(p))`

Supported representations (detected by fields in `p`):

1. **Cartesian components**: fields `(x, y, z)`
   - Returns `a = (p.x, p.y, p.z)`.

2. **Spherical (RA/DEC) in the integration frame**: fields `(r, ras, dec)`
   - Angles in **radians**.
   - `ras` = right ascension (azimuth in XY plane, from +X toward +Y).
   - `dec` = declination (elevation from XY plane toward +Z).
   - Conversion:
     `ax = r*cos(dec)*cos(ras)`, `ay = r*cos(dec)*sin(ras)`, `az = r*sin(dec)`.

3. **RTN/LVLH components**: fields `(r, t, n)`
   - Here `r,t,n` are components along the instantaneous RTN basis computed from `(r,v)` in `x`.
   - `a = ar*R̂ + at*T̂ + an*N̂`.

4. **RTN spherical**: fields `(r, rtn_ras, rtn_dec)`
   - `r` = magnitude.
   - `rtn_ras` = azimuth in the RT plane from +R toward +T (radians).
   - `rtn_dec` = elevation from the RT plane toward +N (radians).
   - Compute `(aR,aT,aN)` as RA/DEC but in RTN, then map to Cartesian with the RTN basis.

If none match, throws `ArgumentError` listing `propertynames(p)`.
"""
@inline function __acceleration_to_cartesian(x::AbstractVector{<:Number}, p::ComponentArray{<:Number})
    T = promote_type(eltype(x), eltype(p))

    # 1) Cartesian: p.x, p.y, p.z
    if hasproperty(p, :x) && hasproperty(p, :y) && hasproperty(p, :z)
        return SVector{3,T}(
            T(getproperty(p, :x)),
            T(getproperty(p, :y)),
            T(getproperty(p, :z)),
        )
    end

    # 2) Spherical (RA/DEC): p.r, p.ras, p.dec  (radians)
    if hasproperty(p, :r) && hasproperty(p, :ras) && hasproperty(p, :dec)
        r   = T(getproperty(p, :r))
        ras = T(getproperty(p, :ras))
        dec = T(getproperty(p, :dec))

        cdec = cos(dec)
        return SVector{3,T}(
            r * cdec * cos(ras),
            r * cdec * sin(ras),
            r * sin(dec),
        )
    end

    # for RTN-based cases we need r and v from x
    @inbounds begin
        rvec = SVector{3,T}(T(x[1]), T(x[2]), T(x[3]))
        vvec = SVector{3,T}(T(x[4]), T(x[5]), T(x[6]))
        R̂, T̂, N̂ = __lvlh_basis(rvec, vvec)

        # 3) RTN components: p.r, p.t, p.n
        if hasproperty(p, :r) && hasproperty(p, :t) && hasproperty(p, :n)
            ar = T(getproperty(p, :r))
            at = T(getproperty(p, :t))
            an = T(getproperty(p, :n))
            return ar * R̂ + at * T̂ + an * N̂
        end

        # 4) RTN spherical: p.r, p.rtn_ras, p.rtn_dec
        if hasproperty(p, :r) && hasproperty(p, :rtn_ras) && hasproperty(p, :rtn_dec)
            rmag = T(getproperty(p, :r))
            ras  = T(getproperty(p, :rtn_ras))
            dec  = T(getproperty(p, :rtn_dec))

            cdec = cos(dec)
            aR = rmag * cdec * cos(ras)
            aT = rmag * cdec * sin(ras)
            aN = rmag * sin(dec)

            return aR * R̂ + aT * T̂ + aN * N̂
        end
    end

    throw(ArgumentError(
        "Acceleration must have fields (x,y,z) or (r,ras,dec) or (r,t,n) or (r,rtn_ras,rtn_dec). " *
        "Got properties: $(propertynames(p))",
    ))
end
