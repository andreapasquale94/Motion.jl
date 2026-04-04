# ──────────────────────────────────────────────────────────────────────
#  Hermite-Simpson collocation defect constraints
#
#  The dynamics function has signature:
#      dynamics(x::SVector{nx}, u::SVector{nu}, t) -> SVector{nx}
#  returning the time derivative ẋ = f(x, u, t).
#
#  Compressed form  (:HermiteSimpson)
#  ──────────────────────────────────
#  Midpoint state via Hermite interpolation:
#      x̄_m = (x_k + x_{k+1})/2 + (h_k/8)(f_k - f_{k+1})
#  Midpoint control via linear interpolation:
#      ū_m = (u_k + u_{k+1})/2
#  Simpson defect:
#      ζ_k = x_{k+1} - x_k - (h_k/6)(f_k + 4 f̄_m + f_{k+1})
#
#  Separated form  (:HermiteSimpsonSeparated)
#  ──────────────────────────────────────────
#  Midpoint state x_m and control u_m are decision variables.
#  Two defects per segment:
#      ζ_k = x_m - (x_k + x_{k+1})/2 - (h_k/8)(f_k - f_{k+1})
#      η_k = x_{k+1} - x_k - (h_k/6)(f_k + 4 f_m + f_{k+1})
#
#  References: Betts (2010) §4.6; Pritchett (2020) §2.3.
# ──────────────────────────────────────────────────────────────────────

"""
    defects(vars, dynamics, Val(N), Val(nx), Val(nu), Val(:HermiteSimpson))

Compressed Hermite-Simpson collocation defects.

Returns `SVector{nx*(N-1)}` with the Simpson quadrature residuals.
The midpoint states are implicitly computed via cubic Hermite
interpolation; midpoint controls are linearly interpolated.
"""
function defects(
    vars::AbstractVector{T},
    dynamics::F,
    vN::Val{N}, vnx::Val{nx}, vnu::Val{nu},
    ::Val{:HermiteSimpson},
) where {T, F, N, nx, nu}
    _, dt, X, U, t = variables(vars, vN, vnx, vnu)

    blocks = ntuple(Val(N - 1)) do k
        @inbounds begin
            xk   = X[:, k]
            xkp1 = X[:, k + 1]
            uk   = U[:, k]
            ukp1 = U[:, k + 1]
            hk   = dt[k]
            tm   = t[k] + hk / 2

            fk   = dynamics(xk,   uk,   t[k])
            fkp1 = dynamics(xkp1, ukp1, t[k + 1])

            # Hermite midpoint state
            xm = (xk + xkp1) / 2 + (hk / 8) * (fk - fkp1)
            um = (uk + ukp1) / 2
            fm = dynamics(xm, um, tm)

            # Simpson quadrature residual
            xkp1 - xk - (hk / 6) * (fk + 4 * fm + fkp1)
        end
    end
    return SVector{nx * (N - 1), T}(reduce(vcat, blocks))
end

"""
    defects(vars, dynamics, Val(N), Val(nx), Val(nu), Val(:HermiteSimpsonSeparated))

Separated Hermite-Simpson collocation defects.

Returns `SVector{2*nx*(N-1)}` with interleaved Hermite interpolation
and Simpson quadrature residuals for each segment.  Midpoint states
and controls are taken from the decision vector.
"""
function defects(
    vars::AbstractVector{T},
    dynamics::F,
    vN::Val{N}, vnx::Val{nx}, vnu::Val{nu},
    ::Val{:HermiteSimpsonSeparated},
) where {T, F, N, nx, nu}
    _, dt, X, U, Xm, Um, t = variables_separated(vars, vN, vnx, vnu)

    blocks = ntuple(Val(N - 1)) do k
        @inbounds begin
            xk   = X[:, k]
            xkp1 = X[:, k + 1]
            uk   = U[:, k]
            ukp1 = U[:, k + 1]
            xm   = Xm[:, k]
            um   = Um[:, k]
            hk   = dt[k]
            tm   = t[k] + hk / 2

            fk   = dynamics(xk,   uk,   t[k])
            fkp1 = dynamics(xkp1, ukp1, t[k + 1])
            fm   = dynamics(xm,   um,   tm)

            # Hermite interpolation residual (midpoint matching)
            ζ = xm - (xk + xkp1) / 2 - (hk / 8) * (fk - fkp1)
            # Simpson quadrature residual (collocation)
            η = xkp1 - xk - (hk / 6) * (fk + 4 * fm + fkp1)

            vcat(ζ, η)
        end
    end
    return SVector{2 * nx * (N - 1), T}(reduce(vcat, blocks))
end
