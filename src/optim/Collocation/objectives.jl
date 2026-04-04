# ──────────────────────────────────────────────────────────────────────
#  Objective functions for Hermite-Simpson collocation
#
#  Running costs are integrated via Simpson quadrature:
#      J ≈ Σ_{k=1}^{N-1} (h_k/6)[L_k + 4 L_m + L_{k+1}]
#
#  For the compressed form, midpoint controls are linearly
#  interpolated: u_m = (u_k + u_{k+1})/2.
#  For the separated form, midpoint controls come from the
#  decision vector.
# ──────────────────────────────────────────────────────────────────────

# ── Compressed-form objectives ──────────────────────────────────────

"""
    objective(vars, Val(N), Val(nx), Val(nu), Val(:FUEL)) -> T

Minimum fuel via Simpson quadrature: `∑ (h_k/6)[‖u_k‖ + 4‖u_m‖ + ‖u_{k+1}‖]`
with smooth approximation `‖u‖ ≈ √(‖u‖² + ε)`.

Compressed form – midpoint controls are linearly interpolated.
"""
function objective(
    vars::AbstractVector{T},
    vN::Val{N}, vnx::Val{nx}, vnu::Val{nu},
    ::Val{:FUEL},
) where {T, N, nx, nu}
    _, dt, _, U, _ = variables(vars, vN, vnx, vnu)
    ε = T(1e-16)
    J = zero(T)
    @inbounds for k in 1:(N - 1)
        uk   = SVector{nu, T}(U[:, k])
        ukp1 = SVector{nu, T}(U[:, k + 1])
        um   = (uk + ukp1) / 2
        Lk   = sqrt(sum(abs2, uk)   + ε)
        Lm   = sqrt(sum(abs2, um)   + ε)
        Lkp1 = sqrt(sum(abs2, ukp1) + ε)
        J += (dt[k] / 6) * (Lk + 4 * Lm + Lkp1)
    end
    return J
end

"""
    objective(vars, Val(N), Val(nx), Val(nu), Val(:ENERGY)) -> T

Minimum energy via Simpson quadrature: `∑ (h_k/6)[‖u_k‖² + 4‖u_m‖² + ‖u_{k+1}‖²]`.

Compressed form.
"""
function objective(
    vars::AbstractVector{T},
    vN::Val{N}, vnx::Val{nx}, vnu::Val{nu},
    ::Val{:ENERGY},
) where {T, N, nx, nu}
    _, dt, _, U, _ = variables(vars, vN, vnx, vnu)
    J = zero(T)
    @inbounds for k in 1:(N - 1)
        uk   = SVector{nu, T}(U[:, k])
        ukp1 = SVector{nu, T}(U[:, k + 1])
        um   = (uk + ukp1) / 2
        J += (dt[k] / 6) * (sum(abs2, uk) + 4 * sum(abs2, um) + sum(abs2, ukp1))
    end
    return J
end

"""
    objective(vars, Val(N), Val(nx), Val(nu), Val(:TIME)) -> T

Minimum time: `∑ dt_k`.
"""
function objective(
    vars::AbstractVector{T},
    vN::Val{N}, vnx::Val{nx}, vnu::Val{nu},
    ::Val{:TIME},
) where {T, N, nx, nu}
    _, idt, _, _ = indexes(vN, vnx, vnu)
    return sum(@view(vars[idt]))
end

# ── Separated-form objectives ──────────────────────────────────────

"""
    objective_separated(vars, Val(N), Val(nx), Val(nu), Val(:FUEL)) -> T

Minimum fuel via Simpson quadrature (separated form).
Midpoint controls are taken from the decision vector.
"""
function objective_separated(
    vars::AbstractVector{T},
    vN::Val{N}, vnx::Val{nx}, vnu::Val{nu},
    ::Val{:FUEL},
) where {T, N, nx, nu}
    _, dt, _, U, _, Um, _ = variables_separated(vars, vN, vnx, vnu)
    ε = T(1e-16)
    J = zero(T)
    @inbounds for k in 1:(N - 1)
        uk   = SVector{nu, T}(U[:, k])
        ukp1 = SVector{nu, T}(U[:, k + 1])
        um   = SVector{nu, T}(Um[:, k])
        Lk   = sqrt(sum(abs2, uk)   + ε)
        Lm   = sqrt(sum(abs2, um)   + ε)
        Lkp1 = sqrt(sum(abs2, ukp1) + ε)
        J += (dt[k] / 6) * (Lk + 4 * Lm + Lkp1)
    end
    return J
end

"""
    objective_separated(vars, Val(N), Val(nx), Val(nu), Val(:ENERGY)) -> T

Minimum energy via Simpson quadrature (separated form).
"""
function objective_separated(
    vars::AbstractVector{T},
    vN::Val{N}, vnx::Val{nx}, vnu::Val{nu},
    ::Val{:ENERGY},
) where {T, N, nx, nu}
    _, dt, _, U, _, Um, _ = variables_separated(vars, vN, vnx, vnu)
    J = zero(T)
    @inbounds for k in 1:(N - 1)
        uk   = SVector{nu, T}(U[:, k])
        ukp1 = SVector{nu, T}(U[:, k + 1])
        um   = SVector{nu, T}(Um[:, k])
        J += (dt[k] / 6) * (sum(abs2, uk) + 4 * sum(abs2, um) + sum(abs2, ukp1))
    end
    return J
end

"""
    objective_separated(vars, Val(N), Val(nx), Val(nu), Val(:TIME)) -> T

Minimum time (separated form, same as compressed).
"""
function objective_separated(
    vars::AbstractVector{T},
    vN::Val{N}, vnx::Val{nx}, vnu::Val{nu},
    ::Val{:TIME},
) where {T, N, nx, nu}
    _, idt, _, _ = indexes(vN, vnx, vnu)
    return sum(@view(vars[idt]))
end
