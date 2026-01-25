"""
	Solution{S, N, T, V}

Container for a single trajectory solution.

Fields
- `sol`: callable solution object
- `t0`, `tf`: initial and final times
- `x0`, `xf`: initial and final states
"""
struct Solution{S, N, T, V}
    sol::S 
    t0::T 
    tf::T
    x0::SVector{N, V}
    xf::SVector{N, V}
end

function (cache::Solution)(args...) 
	return cache.sol(args...)
end

"""
	SensitivitySolution{S, N, T, V}

Container for a single trajectory solution with sensitivities.

Fields
- `sol`: callable solution object
- `t0`, `tf`: initial and final times
- `x0`, `xf`: initial and final states
- `dx_dx0`: state sensitivity w.r.t. initial state
- `dx_dtf`: state sensitivity w.r.t. final time
- `dx_dt0`: state sensitivity w.r.t. initial time
"""
struct SensitivitySolution{S, N, T, V}
	sol::S
    t0::T 
    tf::T
	x0::SVector{N, V} 
	xf::SVector{N, V}
    # gradients
	dx_dx0::SMatrix{N, N, V}
	dx_dtf::SVector{N, V}
	dx_dt0::SVector{N, V}
end

function (cache::SensitivitySolution)(args...)
	return cache.sol(args...)
end

"""
	BatchSolution{S, T, V}
	
Container for a batch of trajectory solutions.

Fields
- `sol`: callable solution object
- `t0`, `tf`: vectors of initial and final times
- `x0`, `xf`: matrices of initial and final states (columns correspond to trajectories)
"""
struct BatchSolution{S, T, V}
	sol::S
	t0::Vector{T}
	tf::Vector{T}
	x0::Matrix{V}
	xf::Matrix{V}
end

function (cache::BatchSolution)(args...)
	return cache.sol(args...)
end
