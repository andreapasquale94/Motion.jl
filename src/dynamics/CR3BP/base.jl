"""
	rhs(x, p::ComponentArray, t) -> SVector{6,T}

CR3BP rotating-frame dynamics (dimensionless).

State ordering:
`x = [x, y, z, vx, vy, vz]`.

Parameters:
- `p.μ`: mass parameter.
"""
@inline function rhs(x::AbstractVector{T}, p::ComponentArray{<:Number}, t::Number) where {T}
	μ  = T(getproperty(p, :μ))
	μ1 = one(T) - μ
	n   = hasproperty(p, :n) ? T(getproperty(p, :n)) : one(T)

	@inbounds begin
		px, py, pz = x[1], x[2], x[3]
		vx, vy, vz = x[4], x[5], x[6]

		px1 = px + μ
		px2 = px - μ1

		r1sq = px1*px1 + py*py + pz*pz
		r2sq = px2*px2 + py*py + pz*pz

		ir13 = inv(r1sq * sqrt(r1sq))
		ir23 = inv(r2sq * sqrt(r2sq))

		ax = 2n*vy + n*n*px - μ1*px1*ir13 - μ*px2*ir23
		ay = -2n*vx + n*n*py - μ1*py*ir13 - μ*py*ir23
		az = - μ1*pz*ir13 - μ*pz*ir23

		if hasproperty(p, :primary)
			aJ = zonal_gravity(SVector(px1, py, pz), μ1, getproperty(p, :primary))
			ax += aJ[1]
			ay += aJ[2]
			az += aJ[3]
		end

		if hasproperty(p, :secondary)
			aJ = zonal_gravity(SVector(px2, py, pz), μ, getproperty(p, :secondary))
			ax += aJ[1]
			ay += aJ[2]
			az += aJ[3]
		end

		return SVector{6, T}(vx, vy, vz, ax, ay, az)
	end
end

@fastmath function jacobian(x::AbstractVector{T}, μ::Number) where T
	@inbounds x, y, z = x[1], x[2], x[3]

	Δx1 = x+μ
	Δx2 = x-1+μ

	tmp = y*y + z*z
	r₁ = sqrt(Δx1*Δx1 + tmp)
	r₂ = sqrt(Δx2*Δx2 + tmp)

	r₁² = r₁*r₁
	r₂² = r₂*r₂
	r₁³ = r₁²*r₁
	r₂³ = r₂²*r₂

	f₁3 = (1-μ)/r₁³
	f₂3 = μ/r₂³
	f₁5 = f₁3/r₁²
	f₂5 = f₂3/r₂²

	tmp = f₁5 + f₂5
	uxx = 1.0 - f₁3 - f₂3 + 3*Δx1*Δx1*f₁5 + 3*Δx2*Δx2*f₂5
	uyy = 1.0 - f₁3 - f₂3 + 3*y*y*tmp
	uzz = - f₁3 - f₂3 + 3*z*z*tmp

	uyz = 3*y*z*tmp
	tmp = Δx1*f₁5 + Δx2*f₂5
	uxy = 3*y*tmp
	uxz = 3*z*tmp

	return SMatrix{6, 6, T}(
		0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
		0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
		0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
		uxx, uxy, uxz, 0.0, 2.0, 0.0,
		uxy, uyy, uyz, -2.0, 0.0, 0.0,
		uxz, uyz, uzz, 0.0, 0.0, 0.0,
	)'
end

@fastmath function jacobian(x::AbstractVector{T}, p::ComponentArray{<:Number}) where T
	return ForwardDiff.jacobian(xx->rhs(xx, p, 0), x)
end

function rhs_stm(x::AbstractVector{T}, p::ComponentArray{<:Number}, t) where T
	# State transition matrix
	Φ = reshape(@view(x[7:end]), Size(6, 6))
	# Compute state derivative
	δx = rhs(x, p, t)
	# Compute jacobian
	J = jacobian(@view(x[1:6]), p)
	# Compute stm derivative
	δΦ = J * Φ
	return vcat(δx, reshape(δΦ, Size(36)))
end

"""
	jacobi_constant(x, μ) -> T

Return the CR3BP Jacobi constant for rotating-frame state
`x = [x, y, z, vx, vy, vz]` and mass parameter `μ`.
"""
@fastmath function jacobi_constant(x::AbstractVector{T}, μ::Number) where {T}
	μT = T(μ)
	μ1 = one(T) - μT

	@inbounds begin
		x, y, z = x[1], x[2], x[3]
		vx, vy, vz = x[4], x[5], x[6]

		Δx1 = x + μT
		Δx2 = x - μ1

		r1 = sqrt(Δx1*Δx1 + y*y + z*z)
		r2 = sqrt(Δx2*Δx2 + y*y + z*z)

		vsq = vx*vx + vy*vy + vz*vz
		return x*x + y*y + 2*(μ1/r1 + μT/r2) - vsq
	end
end

# --- Make

function make_ode_problem(p::ComponentArray{<:Number}, x0::AbstractVector{<:Number}, t0::Number, tf::Number)
	length(x0) == 6 || throw(ArgumentError("expected state of length 6, got $(length(x0))"))
	T = promote_type(eltype(p), eltype(x0), typeof(t0), typeof(tf))
	x0v = @inbounds SVector{6, T}(x0[1], x0[2], x0[3], x0[4], x0[5], x0[6])
	return ODEProblem(rhs, x0v, (T(t0), T(tf)), p)
end

function make_ode_problem(μ::Number, x0::AbstractVector{<:Number}, t0::Number, tf::Number)
	T = promote_type(typeof(μ), eltype(x0), typeof(t0), typeof(tf))
	p = ComponentArray(; μ = T(μ))
	return make_ode_problem(p, x0, t0, tf)
end

# --- Flow 

"""
	flow(μ, x0, t0, tf, alg; reltol=..., abstol=..., kwargs...) -> SVector{6,T}

Integrate CR3BP and return the final state `x(tf)`.
"""
function flow(
	μ::Number, x0::AbstractVector{<:Number}, t0::Number, tf::Number, alg;
	reltol = 1e-14, abstol = 1e-14, kwargs...,
)
	prob = make_ode_problem(μ, x0, t0, tf)
	sol  = solve(prob, alg; save_everystep = false, reltol = reltol, abstol = abstol, kwargs...)
	return sol.u[end]
end

function flow(
	p::ComponentArray{<:Number}, x0::AbstractVector{<:Number}, t0::Number, tf::Number, alg;
	reltol = 1e-14, abstol = 1e-14, kwargs...,
)
	prob = make_ode_problem(p, x0, t0, tf)
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
	reltol = 1e-14, abstol = 1e-14, kwargs...,
)
	prob = make_ode_problem(μ, x0, t0, tf)
	sol  = solve(prob, alg; reltol = reltol, abstol = abstol, kwargs...)
	return Solution(sol, t0, tf, sol.u[1], sol.u[end])
end

function build_solution(
	p::ComponentArray{<:Number}, x0::AbstractVector{<:Number}, t0::Number, tf::Number, alg;
	reltol = 1e-14, abstol = 1e-14, kwargs...,
)
	prob = make_ode_problem(p, x0, t0, tf)
	sol  = solve(prob, alg; reltol = reltol, abstol = abstol, kwargs...)
	return Solution(sol, t0, tf, sol.u[1], sol.u[end])
end
