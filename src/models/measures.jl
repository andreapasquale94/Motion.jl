"""
    arclen(x, offset)

Return the instantaneous rate of change of arc length about `offset` for state `x`.
`x` is assumed to be at least 6 elements long with position in `x[1:3]` and
velocity in `x[4:6]`.
"""
function arclen(x::AbstractVector{<:Number}, offset::AbstractVector{<:Number})
	px, py, pz, vx, vy, vz = @view(x[1:6])
	ox, oy, oz = offset
	dx, dy, dz = px - ox, py - oy, pz - oz
	return dx*vx + dy*vy + dz*vz
end

"""
    stretch(x)

Return the speed ‖v‖ for state `x`, assuming velocity is stored in `x[4:6]`.
"""
function stretch(x::AbstractVector{<:Number})
	return norm(@view(x[4:6]))
end

"""
    make_measure(state, measure, t0, tf)

Create an `ODEProblem` that evaluates a scalar `measure` of the state over time
on the interval `(t0, tf)`.
"""
function make_measure(state, measure, t0::Number, tf::Number)
	rhs = function (s, p, t)
		return measure(state(t))
	end
	prob = ODEProblem(rhs, 0.0, (t0, tf))
	return prob
end

"""
    compute_stretch(state, t0, tf, Δs=nothing, args...; kwargs...)

Compute the stretch (speed) measure over `(t0, tf)` by solving an ODE that
evaluates `stretch(state(t))`. If `Δs` is provided, uses a callback to save
only when the measure increases by `Δs`.
"""
function compute_stretch(state, t0::Number, tf::Number, Δs=nothing, args...; kwargs...)
	if isnothing(Δs)
		# compute the stretch function 
		prob = make_measure(state, stretch, t0, tf)
		sol = solve(prob, args...; kwargs...)
		return sol
	end

	s_next = Ref{Float64}(Float64(Δs))
	condition = (u, t, integ)->u[end] - s_next[]
	affect! = integ->s_next[] += Δs
	cb = ContinuousCallback(condition, affect!; save_positions=(false, true))

	prob = make_measure(state, stretch, t0, tf)
	sol = solve(prob, args...; callback = cb, save_everystep = false, kwargs...)
	return sol
end
