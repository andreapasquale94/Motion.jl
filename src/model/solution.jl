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
