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

@fastmath function jacobian(x::AbstractVector{T}, μ::Number) where T
	@inbounds px, py, pz = x[1], x[2], x[3]

	px1 = px+μ
	px2 = px-1+μ

	tmp = py*py + pz*pz
	r₁ = sqrt(px1*px1 + tmp)
	r₂ = sqrt(px2*px2 + tmp)

	r₁² = r₁*r₁
	r₂² = r₂*r₂
	r₁³ = r₁²*r₁
	r₂³ = r₂²*r₂

	f₁3 = (1-μ)/r₁³
	f₂3 = μ/r₂³
	f₁5 = f₁3/r₁²
	f₂5 = f₂3/r₂²

	tmp = f₁5 + f₂5
	uxx = 1.0 - f₁3 - f₂3 + 3*px1*px1*f₁5 + 3*px2*px2*f₂5
	uyy = 1.0 - f₁3 - f₂3 + 3*py*py*tmp
	uzz = - f₁3 - f₂3 + 3*pz*pz*tmp

	uyz = 3*py*pz*tmp
	tmp = px1*f₁5 + px2*f₂5
	uxy = 3*py*tmp
	uxz = 3*pz*tmp

	return SMatrix{6, 6, T}(
		0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
		0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
		0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
		uxx, uxy, uxz, 0.0, 2.0, 0.0,
		uxy, uyy, uyz, -2.0, 0.0, 0.0,
		uxz, uyz, uzz, 0.0, 0.0, 0.0,
	)'
end

function rhs_stm(x::AbstractVector{T}, p::ComponentArray{<:Number}, t) where T
	μ  = T(getproperty(p, :μ))
    # State transition matrix
    Φ = reshape(@view(x[7:end]), Size(6, 6))
    # Compute state derivative
    δx = rhs(x, p, t)
    # Compute jacobian
    J = jacobian(x, μ)
    # Compute stm derivative
    δΦ = J * Φ
    return vcat(δx, reshape(δΦ, Size(36)))
end

"""
	jacobi_constant(x, μ) -> T

Return the CR3BP Jacobi constant for rotating-frame state
`x = [px, py, pz, vx, vy, vz]` and mass parameter `μ`.
"""
@fastmath function jacobi_constant(x::AbstractVector{T}, μ::Number) where {T}
	μT = T(μ)
	μ1 = one(T) - μT

	@inbounds begin
		px, py, pz = x[1], x[2], x[3]
		vx, vy, vz = x[4], x[5], x[6]

		px1 = px + μT
		px2 = px - μ1

		r1 = sqrt(px1*px1 + py*py + pz*pz)
		r2 = sqrt(px2*px2 + py*py + pz*pz)

		vsq = vx*vx + vy*vy + vz*vz
		return px*px + py*py + 2*(μ1/r1 + μT/r2) - vsq
	end
end

# --- Make

function make_ode_problem(μ::Number, x0::AbstractVector{<:Number}, t0::Number, tf::Number)
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
	reltol = 1e-14, abstol = 1e-14, kwargs...,
)
	prob = make_ode_problem(μ, x0, t0, tf)
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

