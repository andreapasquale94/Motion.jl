"""
    cr3bp_rhs_constant_thrust(x, p, t) -> SVector{6,T}

CR3BP rotating-frame dynamics with *constant thrust acceleration*.

- `x` is the 6D state `[x,y,z, vx,vy,vz]`.
- `p` is a `ComponentArray` with fields:
  - `p.μ`: CR3BP mass parameter
  - `p.acc`: acceleration specification
"""
@fastmath function cr3bp_rhs_constant_thrust(x::AbstractVector{T}, p::ComponentArray{<:Number}, t::Number) where T
	dx = cr3bp_rhs(x, p, t)
	a = __acceleration_to_cartesian(x, p.acc)
	return SVector{6, T}(
		dx[1], dx[2], dx[3],
		dx[4] + a[1], dx[5] + a[2], dx[6] + a[3],
	)
end

# --- Make

@inline function __acc_cr3bp_constant_thrust(::Val{:Cartesian}, u, ::Type{T}) where {T}
	@inbounds return ComponentArray(; x = T(u[1]), y = T(u[2]), z = T(u[3]))
end

@inline function __acc_cr3bp_constant_thrust(::Val{:Spherical}, u, ::Type{T}) where {T}
	@inbounds return ComponentArray(; r = T(u[1]), ras = T(u[2]), dec = T(u[3]))
end

@inline function __acc_cr3bp_constant_thrust(::Val{:RTN}, u, ::Type{T}) where {T}
	@inbounds return ComponentArray(; r = T(u[1]), t = T(u[2]), n = T(u[3]))
end

@inline function __acc_cr3bp_constant_thrust(::Val{:SphericalRTN}, u, ::Type{T}) where {T}
	@inbounds return ComponentArray(; r = T(u[1]), rtn_ras = T(u[2]), rtn_dec = T(u[3]))
end

@inline function __make_cr3bp_constant_thrust(
	::Val{M}, μ::Number, xv0::SVector{6, T}, u, t0::Number, tf::Number,
) where {M, T}
	acc = __acc_cr3bp_constant_thrust(Val(M), u, T)
	p   = ComponentArray(μ = μ, acc = acc)
	return ODEProblem(cr3bp_rhs_constant_thrust, xv0, (t0, tf), p)
end

"""
	make_cr3bp_constant_thrust(μ, x0, u, t0, tf; model=:Cartesian)

Create an `ODEProblem` for CR3BP dynamics with *constant thrust acceleration*.

Arguments
- `μ`: CR3BP parameter.
- `x0`: initial state (length 6): `[x,y,z, vx,vy,vz]`.
- `u`: control parameters (length 3), interpreted depending on `model`.
- `t0, tf`: initial/final times.

Keyword
- `model::Symbol`:
  - `:Cartesian`     → `u = (ax, ay, az)`
  - `:Spherical`     → `u = (r, ras, dec)` (RA/DEC in radians)
  - `:RTN`           → `u = (ar, at, an)` (RTN components)
  - `:SphericalRTN`  → `u = (r, rtn_ras, rtn_dec)` (RTN spherical angles in radians)
"""
function make_cr3bp_constant_thrust(
	μ::Number,
	x0::AbstractVector{<:Number},
	u::AbstractVector{<:Number},
	t0::Number,
	tf::Number;
	model::Symbol = :Cartesian,
)
	length(x0) == 6 || throw(ArgumentError("expected x0 of length 6, got $(length(x0))"))
	length(u) == 3 || throw(ArgumentError("expected u of length 3, got $(length(u))"))
	T = promote_type(typeof(μ), eltype(x0), eltype(u), typeof(t0), typeof(tf))
	xv0 = @inbounds SVector{6, T}(x0[1], x0[2], x0[3], x0[4], x0[5], x0[6])

	return __make_cr3bp_constant_thrust(Val(model), μ, xv0, u, t0, tf)
end

"""
    flow_cr3bp_constant_thrust(μ, x0, u, t0, tf, alg; 
        model=:Cartesian, reltol=..., abstol=..., kwargs...) -> SVector{6,T}

Integrate CR3BP with constant thrust and return the final state `x(tf)`.

- `alg` is the OrdinaryDiffEq algorithm (e.g. `Vern9()`).
"""
function flow_cr3bp_constant_thrust(
    μ::Number,
    x0::AbstractVector{<:Number},
    u::AbstractVector{<:Number},
    t0::Number,
    tf::Number,
    alg;
    model::Symbol = :Cartesian,
    reltol = 1e-12,
    abstol = 1e-12,
    kwargs...,
)
    prob = make_cr3bp_constant_thrust(μ, x0, u, t0, tf; model=model)
    sol  = solve(prob, alg; save_everystep=false, reltol=reltol, abstol=abstol, kwargs...)
    return sol.u[end]
end

"""
    solve_cr3bp_constant_thrust(μ, x0, u, t0, tf, alg; model=:Cartesian, kwargs...) -> Solution

Solve CR3BP with constant thrust and return a `Solution` wrapper.
"""
function solve_cr3bp_constant_thrust(
    μ::Number,
    x0::AbstractVector{<:Number},
    u::AbstractVector{<:Number},
    t0::Number,
    tf::Number,
    alg;
    model::Symbol = :Cartesian,
    kwargs...,
)
    prob = make_cr3bp_constant_thrust(μ, x0, u, t0, tf; model=model)
    sol  = solve(prob, alg; kwargs...)
    return Solution(sol, t0, tf, sol.u[1], sol.u[end])
end
