
@inline function __rhs_bcr4bp_spm(x::AbstractVector{T}, t, μ, μ₃, l, ω) where T
	μ1 = one(T) - μ
	μp = μ * (one(T) - μ₃)
	μm = μ * μ₃

	@inbounds begin
		px, py, pz = x[1], x[2], x[3]
		vx, vy, vz = x[4], x[5], x[6]
		θ = x[7]

		sθ, cθ = sincos(θ)

		rsx = px - μ
		rsy = py
		rsz = pz

		rb_x = px - one(T) + μ
		rb_y = py

		lμ3 = l * μ₃
		lμ3m = l * (one(T) - μ₃)

		rpx = rb_x + lμ3 * cθ
		rpy = rb_y + lμ3 * sθ

		rmx = rb_x - lμ3m * cθ
		rmy = rb_y - lμ3m * sθ

		rs_sq = rsx * rsx + rsy * rsy + rsz * rsz
		rp_sq = rpx * rpx + rpy * rpy + pz * pz
		rm_sq = rmx * rmx + rmy * rmy + pz * pz

		invrs3 = inv(rs_sq * sqrt(rs_sq))
		invrp3 = inv(rp_sq * sqrt(rp_sq))
		invrm3 = inv(rm_sq * sqrt(rm_sq))

		ax = 2 * vy + px - μ1 * rsx * invrs3 - μp * rpx * invrp3 - μm * rmx * invrm3
		ay = -2 * vx + py - μ1 * rsy * invrs3 - μp * rpy * invrp3 - μm * rmy * invrm3
		az = -μ1 * rsz * invrs3 - μp * pz * invrp3 - μm * pz * invrm3

		return SVector{7, T}(vx, vy, vz, ax, ay, az, ω)
	end
end

@inline function __rhs_bcr4bp_pms(x::AbstractVector{T}, t, μ, μ₃, l, ω) where T
	μ1 = one(T) - μ

	@inbounds begin
		px, py, pz = x[1], x[2], x[3]
		vx, vy, vz = x[4], x[5], x[6]
		θ = x[7]

		sθ, cθ = sincos(θ)

		px1 = px + μ
		px2 = px - μ1

		r1sq = px1 * px1 + py * py + pz * pz
		r2sq = px2 * px2 + py * py + pz * pz

		invr13 = inv(r1sq * sqrt(r1sq))
		invr23 = inv(r2sq * sqrt(r2sq))

		ax = 2 * vy + px - μ1 * px1 * invr13 - μ * px2 * invr23
		ay = -2 * vx + py - μ1 * py * invr13 - μ * py * invr23
		az = -μ1 * pz * invr13 - μ * pz * invr23

		rsx = px - l * cθ
		rsy = py - l * sθ
		rsz = pz
		rs_sq = rsx * rsx + rsy * rsy + rsz * rsz
		invrs3 = inv(rs_sq * sqrt(rs_sq))
		invl2 = inv(l * l)

		ax -= μ₃ * rsx * invrs3 + μ₃ * cθ * invl2
		ay -= μ₃ * rsy * invrs3 + μ₃ * sθ * invl2
		az -= μ₃ * rsz * invrs3

		return SVector{7, T}(vx, vy, vz, ax, ay, az, ω)
	end
end

@inline function rhs_spm(x::AbstractVector{T}, p::ComponentArray{<:Number}, t::Number) where T
	μ = T(getproperty(p, :μ))
	μ₃ = T(getproperty(p, :μ₃))
	l = T(getproperty(p, :l))
	ω = T(getproperty(p, :ω))
	return __rhs_bcr4bp_spm(x, t, μ, μ₃, l, ω)
end

@inline function rhs_pms(x::AbstractVector{T}, p::ComponentArray{<:Number}, t::Number) where T
	μ = T(getproperty(p, :μ))
	μ₃ = T(getproperty(p, :μ₃))
	l = T(getproperty(p, :l))
	ω = T(getproperty(p, :ω))
	return __rhs_bcr4bp_pms(x, t, μ, μ₃, l, ω)
end

# --- Make

"""
	make(μ, μ₃, l, ω, x0, t0, tf; model=:spm) -> ODEProblem

Build an `ODEProblem` for BCR4BP with parameters stored in a `ComponentArray(μ=..., μ₃=..., l=..., ω=...)`.
Promotes `(μ, μ₃, l, ω, x0, t0, tf)` to a common scalar type for consistency.
"""
function make(
	μ::Number,
	μ₃::Number,
	l::Number,
	ω::Number,
	x0::AbstractVector{<:Number},
	t0::Number,
	tf::Number;
	model::Symbol = :spm,
)
	length(x0) == 7 || throw(ArgumentError("expected state of length 7, got $(length(x0))"))
	T = promote_type(typeof(μ), typeof(μ₃), typeof(l), typeof(ω), eltype(x0), typeof(t0), typeof(tf))
	x0v = @inbounds SVector{7, T}(x0[1], x0[2], x0[3], x0[4], x0[5], x0[6], x0[7])
	p = ComponentArray(; μ = T(μ), μ₃ = T(μ₃), l = T(l), ω = T(ω))
	rhs = (model === :spm || model === :SPM || model===:SunPlanetMoon) ? rhs_spm :
		  (model === :pms || model === :PMS || model===:PlanetMoonSun) ? rhs_pms :
		  throw(ArgumentError("expected model :SPM or :PMS, got $(model)"))
	return ODEProblem(rhs, x0v, (T(t0), T(tf)), p)
end

# --- Flow

"""
	flow(μ, μ₃, l, ω, x0, t0, tf, alg; model=:spm, reltol=..., abstol=..., kwargs...) -> SVector{7,T}

Integrate BCR4BP and return the final state `x(tf)`.
"""
function flow(
	μ::Number,
	μ₃::Number,
	l::Number,
	ω::Number,
	x0::AbstractVector{<:Number},
	t0::Number,
	tf::Number,
	alg;
	model::Symbol = :spm,
	reltol = 1e-12,
	abstol = 1e-12,
	kwargs...,
)
	prob = make(μ, μ₃, l, ω, x0, t0, tf; model = model)
	sol  = solve(prob, alg; save_everystep = false, reltol = reltol, abstol = abstol, kwargs...)
	return sol.u[end]
end

# --- Solve

"""
	build_solution(μ, μ₃, l, ω, x0, t0, tf, alg; model=:spm, kwargs...) -> Solution

Integrate BCR4BP and return a Solution.
"""
function build_solution(
	μ::Number,
	μ₃::Number,
	l::Number,
	ω::Number,
	x0::AbstractVector{<:Number},
	t0::Number,
	tf::Number,
	alg;
	model::Symbol = :spm,
	kwargs...,
)
	prob = make(μ, μ₃, l, ω, x0, t0, tf; model = model)
	sol  = solve(prob, alg; kwargs...)
	return Solution(sol, t0, tf, sol.u[1], sol.u[end])
end
