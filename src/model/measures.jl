@inbounds function arclen(x::AbstractVector{<:Number}, offset::AbstractVector{<:Number})
	px, py, pz, vx, vy, vz = @view(x[1:6])
	ox, oy, oz = offset
	dx, dy, dz = px - ox, py - oy, pz - oz
	return dx*vx + dy*vy + dz*vz
end

@inbounds function stretch(x::AbstractVector{<:Number})
	return norm(@view(x[4:6]))
end

function make_measure(state, measure, t0::Number, tf::Number)
	rhs = function (s, p, t)
		return measure(state(t))
	end
	prob = ODEProblem(rhs, 0.0, (t0, tf))
	return prob
end

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
