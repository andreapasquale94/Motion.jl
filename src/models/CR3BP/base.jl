"""
	rhs(x, p::ComponentArray, t) -> SVector{6,T}

CR3BP rotating-frame dynamics (dimensionless).

State ordering:
`x = [px, py, pz, vx, vy, vz]`.

Parameters:
- `p.μ`: mass parameter.
"""
@inline function rhs(x::AbstractVector{T}, p::ComponentArray{<:Number}, t::Number) where {T}
	μ  = T(getproperty(p, :μ))
	μ1 = one(T) - μ

	@inbounds begin
		px, py, pz = x[1], x[2], x[3]
		vx, vy, vz = x[4], x[5], x[6]

		px1 = px + μ
		px2 = px - μ1

		r1sq = px1*px1 + py*py + pz*pz
		r2sq = px2*px2 + py*py + pz*pz

		invr13 = inv(r1sq * sqrt(r1sq))
		invr23 = inv(r2sq * sqrt(r2sq))

		ax = 2*vy + px - μ1*px1*invr13 - μ*px2*invr23
		ay = -2*vx + py - μ1*py*invr13 - μ*py*invr23
		az = - μ1*pz*invr13 - μ*pz*invr23

		return SVector{6, T}(vx, vy, vz, ax, ay, az)
	end
end

# --- Make

"""
	make(μ, x0, t0, tf) -> ODEProblem

Build an `ODEProblem` for CR3BP with parameters stored in a `ComponentArray(μ=...)`.
Promotes `(μ, x0, t0, tf)` to a common scalar type for consistency.
"""
function make(μ::Number, x0::AbstractVector{<:Number}, t0::Number, tf::Number)
	length(x0) == 6 || throw(ArgumentError("expected state of length 6, got $(length(x0))"))
	T = promote_type(typeof(μ), eltype(x0), typeof(t0), typeof(tf))
	x0v = @inbounds SVector{6, T}(x0[1], x0[2], x0[3], x0[4], x0[5], x0[6])
	p = ComponentArray(; μ = T(μ))
	return ODEProblem(rhs, x0v, (T(t0), T(tf)), p)
end


# --- Flow 

"""
	flow(μ, x0, t0, tf, alg; reltol=..., abstol=..., kwargs...) -> SVector{6,T}

Integrate CR3BP and return the final state `x(tf)`.
"""
function flow(
	μ::Number, x0::AbstractVector{<:Number}, t0::Number, tf::Number, alg;
	reltol = 1e-12, abstol = 1e-12, kwargs...,
)
	prob = make(μ, x0, t0, tf)
	sol  = solve(prob, alg; save_everystep = false, reltol = reltol, abstol = abstol, kwargs...)
	return sol.u[end]
end

# --- Solve 

"""
    build_solution(μ, x0, t0, tf, alg; kwargs...) -> Solution

Integrate CR3BP and return a Solution.
"""
function build_solution(
    μ::Number, x0::AbstractVector{<:Number}, t0::Number, tf::Number, alg;
    kwargs...,
)
    prob = make(μ, x0, t0, tf)
    sol  = solve(prob, alg; kwargs...)
    return Solution(sol, t0, tf, sol.u[1], sol.u[end])
end

