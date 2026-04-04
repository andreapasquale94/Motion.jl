# ──────────────────────────────────────────────────────────────────────
#  Mesh refinement for Hermite-Simpson collocation
#
#  After solving the NLP on a given mesh, the truncation error on each
#  segment is estimated.  Segments exceeding a tolerance are bisected
#  (or subdivided further based on the error magnitude) and a new
#  initial guess is constructed via Hermite interpolation.
#
#  These functions operate on trajectory arrays (Vector{SVector}),
#  not on the flat decision vector, because N changes at runtime.
#
#  Reference: Betts (2010) §4.7; Pritchett (2020) §2.4.
# ──────────────────────────────────────────────────────────────────────

"""
    mesh_error(X, U, t, dynamics) -> Vector{T}

Estimate the local truncation error on each segment of a
Hermite-Simpson collocation solution.

For segment `k` with step size `h_k`, the error indicator is

    e_k = max_i |(h_k / 36)(f_{k+1}[i] - 2 f_m[i] + f_k[i])| / (1 + max_i |x_{k+1}[i]|)

where `f_k`, `f_m`, `f_{k+1}` are dynamics evaluations at the left
node, Hermite midpoint, and right node.  The denominator provides
relative scaling.

# Arguments
- `X`  – state trajectory (`Vector{SVector{nx,T}}`, length `N`)
- `U`  – control sequence (`Vector{SVector{nu,T}}`, length `N`)
- `t`  – node times (`Vector{T}`, length `N`)
- `dynamics` – ODE right-hand side `f(x, u, t) -> ẋ`

# Returns
`Vector{T}` of length `N-1` with the per-segment error estimate.
"""
function mesh_error(
    X::Vector{SVector{nx, T}},
    U::Vector{SVector{nu, T}},
    t::Vector{T},
    dynamics,
) where {nx, nu, T}
    N = length(X)
    errors = Vector{T}(undef, N - 1)

    for k in 1:(N - 1)
        hk = t[k + 1] - t[k]
        tm = (t[k] + t[k + 1]) / 2
        um = (U[k] + U[k + 1]) / 2

        fk   = dynamics(X[k],   U[k],   t[k])
        fkp1 = dynamics(X[k+1], U[k+1], t[k+1])

        # Hermite midpoint state
        xm = (X[k] + X[k+1]) / 2 + (hk / 8) * (fk - fkp1)
        fm = dynamics(xm, um, tm)

        # Leading truncation error term (O(h^5) Simpson remainder)
        err_vec = (hk / 36) * (fkp1 - 2 * fm + fk)
        scale   = one(T) + maximum(abs, X[k + 1])
        errors[k] = maximum(abs, err_vec) / scale
    end

    return errors
end

"""
    refine_mesh(X, U, t, dynamics; tol=1e-6, max_subdivisions=4)
        -> (X_new, U_new, t_new, refined)

Refine the collocation mesh by subdividing segments whose error
exceeds `tol`.

The number of sub-intervals inserted into segment `k` is

    n_k = min(ceil((e_k / tol)^{1/4}), max_subdivisions)

where `e_k` is the local error estimate and the exponent `1/4`
reflects the 4th-order convergence of Hermite-Simpson.  New node
states are initialised via cubic Hermite interpolation; controls
are linearly interpolated.

# Arguments
- `X`, `U`, `t`      – current solution trajectory (length `N`)
- `dynamics`          – ODE right-hand side `f(x, u, t) -> ẋ`
- `tol`               – maximum allowable segment error
- `max_subdivisions`  – cap on sub-intervals per segment (default 4)

# Returns
- `X_new`, `U_new`, `t_new` – refined trajectory arrays
- `refined::Bool` – `true` if any segment was subdivided
"""
function refine_mesh(
    X::Vector{SVector{nx, T}},
    U::Vector{SVector{nu, T}},
    t::Vector{T},
    dynamics;
    tol::T = T(1e-6),
    max_subdivisions::Int = 4,
) where {nx, nu, T}
    N = length(X)
    errors = mesh_error(X, U, t, dynamics)

    X_new = SVector{nx, T}[X[1]]; sizehint!(X_new, 2N)
    U_new = SVector{nu, T}[U[1]]; sizehint!(U_new, 2N)
    t_new = T[t[1]];              sizehint!(t_new, 2N)

    for k in 1:(N - 1)
        if errors[k] > tol
            # Number of sub-intervals based on error magnitude
            nsub = min(ceil(Int, (errors[k] / tol)^(one(T) / 4)), max_subdivisions)

            hk   = t[k + 1] - t[k]
            fk   = dynamics(X[k],   U[k],   t[k])
            fkp1 = dynamics(X[k+1], U[k+1], t[k+1])

            # Insert nsub-1 interior nodes via Hermite interpolation
            for j in 1:(nsub - 1)
                τ = T(j) / T(nsub)          # fraction along segment
                tj = t[k] + τ * hk

                # Cubic Hermite basis
                h00 = 2τ^3 - 3τ^2 + 1
                h10 = τ^3  - 2τ^2 + τ
                h01 = -2τ^3 + 3τ^2
                h11 = τ^3  - τ^2

                xj = h00 * X[k] + h10 * hk * fk + h01 * X[k+1] + h11 * hk * fkp1
                uj = (1 - τ) * U[k] + τ * U[k + 1]

                push!(X_new, xj)
                push!(U_new, uj)
                push!(t_new, tj)
            end
        end

        # Always add the right-endpoint node
        push!(X_new, X[k + 1])
        push!(U_new, U[k + 1])
        push!(t_new, t[k + 1])
    end

    refined = length(X_new) > N
    return X_new, U_new, t_new, refined
end
