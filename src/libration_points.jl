@inline function _compute_libration_point(μ::Number, ::Val{:L1}, tol = 1e-14)
	f = (x, _) -> 1.0 - μ - x - (1-μ)/(1-x)^2 + μ/x^2
	prob = IntervalNonlinearProblem(f, (0.0, 0.5))
	sol = solve(prob, Ridder(); abstol =  reltol=tol )
	return SVector{6, typeof(μ)}(1 - μ - sol.u, 0, 0, 0, 0, 0)
end

@inline function _compute_libration_point(μ::Number, ::Val{:L2}, tol = 1e-14)
	f = (x, _) -> 1.0 - μ + x - (1-μ)/(1+x)^2 - μ/x^2
	prob = IntervalNonlinearProblem(f, (0.0, 0.5))
	sol = solve(prob, Ridder(); abstol =  reltol=tol )
	return SVector{6, typeof(μ)}(1 - μ + sol.u, 0, 0, 0, 0, 0)
end

@inline function _compute_libration_point(μ::Number, ::Val{:L3}, tol = 1e-14)
	f = (x, _) -> -μ - x + (1-μ)/x^2 - μ/(1+x)^2
	prob = IntervalNonlinearProblem(f, (0.5, 1.5))
	sol = solve(prob, Ridder(); abstol =  reltol=tol )
	return SVector{6, typeof(μ)}(- μ - sol.u, 0, 0, 0, 0, 0)
end

@inline function _compute_libration_point(μ::Number, ::Val{:L4}, args...)
	return SVector{6, typeof(μ)}(0.5 - μ, 0.5*sqrt(3), 0, 0, 0, 0)
end

@inline function _compute_libration_point(μ::Number, ::Val{:L5}, args...)
	return SVector{6, typeof(μ)}(0.5 - μ, -0.5*sqrt(3), 0, 0, 0, 0)
end

@inline function libration_point(par, val::Val{S}, tol = 1e-14) where S
	return _compute_libration_point(par, val, tol)
end

"""
	libration_points(par; tol=1e-14)

Return the five CR3BP libration points (L1-L5) given the problem parameters,
as 6-element state vectors in the rotating frame. The points are returned in order L1, L2, L3, L4, L5.
"""
function libration_points(par; tol = 1e-14)
	return [
		libration_point(par, Val(:L1), tol),
		libration_point(par, Val(:L2), tol),
		libration_point(par, Val(:L3), tol),
		libration_point(par, Val(:L4), tol),
		libration_point(par, Val(:L5), tol),
	]
end
