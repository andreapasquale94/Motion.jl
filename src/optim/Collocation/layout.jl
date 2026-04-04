# ──────────────────────────────────────────────────────────────────────
#  Decision-variable layout for Hermite-Simpson collocation
#
#  Compressed form (same layout as MultipleShooting):
#    [t0, dt₁…dt_{N-1}, X(nx×N), U(nu×N)]
#
#  Separated form (adds explicit midpoint variables):
#    [t0, dt₁…dt_{N-1}, X(nx×N), U(nu×N), Xm(nx×(N-1)), Um(nu×(N-1))]
# ──────────────────────────────────────────────────────────────────────

# ── Compressed layout ───────────────────────────────────────────────

"""
    indexes(Val(N), Val(nx), Val(nu)) -> (it0, idt, iX, iU)

Return StaticArrays indices for the compressed Hermite-Simpson layout.

Decision vector `vars`:
- `vars[it0]`       : start time `t0`
- `vars[idt]`       : `N-1` segment durations `dt₁ … dt_{N-1}`
- `vars[vec(iX)]`   : node states, column-major `nx × N`
- `vars[vec(iU)]`   : node controls, column-major `nu × N`

Total length: `N + nx N + nu N`.
"""
@inline function indexes(::Val{N}, ::Val{nx}, ::Val{nu}) where {N, nx, nu}
    it0 = 1
    idt = SVector{N-1, Int}(2:N)

    x_start = N + 1
    x_end   = x_start + nx * N - 1
    iX      = SMatrix{nx, N, Int}(SVector{nx*N, Int}(x_start:x_end))

    u_start = x_end + 1
    u_end   = u_start + nu * N - 1
    iU      = SMatrix{nu, N, Int}(SVector{nu*N, Int}(u_start:u_end))

    return it0, idt, iX, iU
end

# ── Separated layout ───────────────────────────────────────────────

"""
    indexes_separated(Val(N), Val(nx), Val(nu)) -> (it0, idt, iX, iU, iXm, iUm)

Return StaticArrays indices for the separated Hermite-Simpson layout,
which extends the compressed layout with midpoint decision variables.

Extra indices:
- `vars[vec(iXm)]`  : midpoint states, column-major `nx × (N-1)`
- `vars[vec(iUm)]`  : midpoint controls, column-major `nu × (N-1)`

Total length: `N + nx(2N-1) + nu(2N-1)`.
"""
@inline function indexes_separated(::Val{N}, ::Val{nx}, ::Val{nu}) where {N, nx, nu}
    it0, idt, iX, iU = indexes(Val(N), Val(nx), Val(nu))

    base = N + nx * N + nu * N

    xm_start = base + 1
    xm_end   = xm_start + nx * (N - 1) - 1
    iXm      = SMatrix{nx, N-1, Int}(SVector{nx*(N-1), Int}(xm_start:xm_end))

    um_start = xm_end + 1
    um_end   = um_start + nu * (N - 1) - 1
    iUm      = SMatrix{nu, N-1, Int}(SVector{nu*(N-1), Int}(um_start:um_end))

    return it0, idt, iX, iU, iXm, iUm
end

# ── Time-vector builder (generated for zero-allocation) ────────────

@generated function __build_time_vector(::Val{N}, t0::T, dt) where {N, T}
    if N == 1
        return :(SVector{1, T}(t0))
    end

    tvars = [gensym(:t) for _ in 2:N]
    assigns = Expr[]
    push!(assigns, :($(tvars[1]) = t0 + dt[1]))
    for k in 3:N
        push!(assigns, :($(tvars[k-1]) = $(tvars[k-2]) + dt[$(k-1)]))
    end

    args = Any[:t0]
    append!(args, tvars)

    quote
        @inbounds begin
            $(Expr(:block, assigns...))
            SVector{$N, T}($(args...))
        end
    end
end

# ── Variable unpacking ─────────────────────────────────────────────

"""
    variables(vars, Val(N), Val(nx), Val(nu)) -> (t0, dt, X, U, t)

Unpack a compressed decision vector into structured StaticArrays.
Allocation-free.
"""
@inline function variables(
    vars::AbstractVector{T},
    vN::Val{N}, vnx::Val{nx}, vnu::Val{nu},
) where {T, N, nx, nu}
    it0, idt, iX, iU = indexes(vN, vnx, vnu)
    @inbounds begin
        t0 = vars[it0]
        dt = SVector{N-1, T}(vars[idt])
        X  = SMatrix{nx, N, T}(vars[vec(iX)])
        U  = SMatrix{nu, N, T}(vars[vec(iU)])
        t  = __build_time_vector(vN, t0, dt)
        return t0, dt, X, U, t
    end
end

"""
    variables_separated(vars, Val(N), Val(nx), Val(nu)) -> (t0, dt, X, U, Xm, Um, t)

Unpack a separated decision vector, including midpoint states `Xm`
and controls `Um`.  Allocation-free.
"""
@inline function variables_separated(
    vars::AbstractVector{T},
    vN::Val{N}, vnx::Val{nx}, vnu::Val{nu},
) where {T, N, nx, nu}
    it0, idt, iX, iU, iXm, iUm = indexes_separated(vN, vnx, vnu)
    @inbounds begin
        t0 = vars[it0]
        dt = SVector{N-1, T}(vars[idt])
        X  = SMatrix{nx, N, T}(vars[vec(iX)])
        U  = SMatrix{nu, N, T}(vars[vec(iU)])
        Xm = SMatrix{nx, N-1, T}(vars[vec(iXm)])
        Um = SMatrix{nu, N-1, T}(vars[vec(iUm)])
        t  = __build_time_vector(vN, t0, dt)
        return t0, dt, X, U, Xm, Um, t
    end
end
