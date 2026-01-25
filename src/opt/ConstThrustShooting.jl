module ConstThrustShooting

using StaticArrays
using LinearAlgebra

export indexes, variables, defects, objective

"""
    indexes(Val(N), Val(nx), Val(nu)) -> (it0, idt, iX, iU)

Return StaticArrays indices describing the decision-vector layout for *continuous-thrust*
multiple shooting (piecewise-constant control per segment).

Decision vector `vars` layout:

- `vars[it0]`            : scalar start time `t0`
- `vars[idt]`            : `(N-1)` segment durations `dt₁ … dt_{N-1}`
- `vars[vec(iX)]`        : node states, packed column-major as an `nx×N` matrix
- `vars[vec(iU)]`        : segment controls, packed column-major as a `nu×(N-1)` matrix

Controls are defined **per segment** `k=1..N-1` and applied over `[tₖ, tₖ₊₁]`.
"""
@inline function indexes(::Val{N}, ::Val{nx}, ::Val{nu}) where {N,nx,nu}
    it0 = 1
    idt = SVector{N-1,Int}(2:N)

    x_start = N + 1
    x_end   = x_start + nx*N - 1
    Xvec    = SVector{nx*N,Int}(x_start:x_end)
    iX      = SMatrix{nx,N,Int}(Xvec)          # column-major fill

    u_start = x_end + 1
    u_end   = u_start + nu*(N-1) - 1
    Uvec    = SVector{nu*(N-1),Int}(u_start:u_end)
    iU      = SMatrix{nu,N-1,Int}(Uvec)

    return it0, idt, iX, iU
end

@generated function __build_time_vector(::Val{N}, t0::T, dt) where {N,T}
    if N == 1
        return quote
            SVector{1,T}(t0)
        end
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
            SVector{$N,T}($(args...))
        end
    end
end

"""
    variables(vars, Val(N), Val(nx), Val(nu)) -> (t0, dt, X, U, t)

Unpack decision vector into structured StaticArrays:

- `t0::T`
- `dt::SVector{N-1,T}`
- `X::SMatrix{nx,N,T}`     node states
- `U::SMatrix{nu,N-1,T}`   segment controls
- `t::SVector{N,T}`        node times
"""
@inline function variables(
    vars::AbstractVector{T},
    vN::Val{N},
    vnx::Val{nx},
    vnu::Val{nu},
) where {T,N,nx,nu}
    it0, idt, iX, iU = indexes(vN, vnx, vnu)
    @inbounds begin
        t0 = vars[it0]
        dt = SVector{N-1,T}(vars[idt])
        X  = SMatrix{nx,N,T}(vars[vec(iX)])
        U  = SMatrix{nu,N-1,T}(vars[vec(iU)])
        t  = __build_time_vector(vN, t0, dt)
        return t0, dt, X, U, t
    end
end

"""
    defects(vars, flow, Val(N), Val(nx), Val(nu)) -> SVector{nx*(N-1),T}

Compute multiple-shooting defects for continuous thrust:

`d_k = X_{k+1} - flow(X_k, U_k, t_k, t_{k+1})`, for `k=1..N-1`.

`flow(x,u,t0,t1)` must return a length-`nx` vector (preferably `SVector{nx,T}`).
"""
function defects(
    vars::AbstractVector{T},
    flow::F,
    vN::Val{N},
    vnx::Val{nx},
    vnu::Val{nu},
) where {T,F,N,nx,nu}
    _, _, X, U, t = variables(vars, vN, vnx, vnu)

    blocks = ntuple(Val(N-1)) do k
        @inbounds begin
            xk   = SVector{nx,T}(X[:,k])
            xkp1 = SVector{nx,T}(X[:,k+1])
            uk   = SVector{nu,T}(U[:,k])

            xn = flow(xk, uk, t[k], t[k+1])
            xnS = (xn isa SVector{nx,T}) ? xn : SVector{nx,T}(xn)

            xkp1 - xnS
        end
    end

    return SVector{nx*(N-1),T}(reduce(vcat, blocks))
end

"""
    objective(vars, Val(N), Val(nx), Val(nu), Val(:FUEL)) -> T

Minimum fuel objective for continuous thrust (piecewise-constant per segment): `∑_{k=1}^{N-1} √(‖u_k‖² + ϵ) * dt_k`

Smooth approximation of `∫‖u(t)‖ dt` with `ϵ = T(1e-16)`.
"""
function objective(
    vars::AbstractVector{T},
    vN::Val{N},
    vnx::Val{nx},
    vnu::Val{nu},
    ::Val{:FUEL},
) where {T,N,nx,nu}
    _, dt, _, U, _ = variables(vars, vN, vnx, vnu)
    ϵ = T(1e-16)
    v = zero(T)
    @inbounds for k in 1:(N-1)
        uk = @view U[:,k]
        v += sqrt(sum(abs2, uk) + ϵ) * dt[k]
    end
    return v
end

"""
    objective(vars, Val(N), Val(nx), Val(nu), Val(:ENERGY)) -> T

Minimum energy continuous thrust: `∑_{k=1}^{N-1} ‖u_k‖² dt_k`

Approximates `∫‖u(t)‖² dt` for piecewise-constant control.
"""
function objective(
    vars::AbstractVector{T},
    vN::Val{N},
    vnx::Val{nx},
    vnu::Val{nu},
    ::Val{:ENERGY},
) where {T,N,nx,nu}
    _, dt, _, U, _ = variables(vars, vN, vnx, vnu)
    v = zero(T)
    @inbounds for k in 1:(N-1)
        uk = @view U[:,k]
        v += sum(abs2, uk) * dt[k]
    end
    return v
end

"""
	objective(vars, Val(N), Val(nx), Val(nu), Val(:TIME)) -> T

Minimum time objective: `∑ₖ dtₖ`.
"""
function objective(
	vars::AbstractVector{T},
	vN::Val{N},
	vnx::Val{nx},
	vnu::Val{nu},
	::Val{:TIME},
) where {T, N, nx, nu}
	_, idt, _, _ = indexes(vN, vnx, vnu)
	return sum(@view(vars[idt]))
end

end # module
