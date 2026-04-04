abstract type AbstractShootingLayout end

abstract type AbstractSingleShootingLayout <: AbstractShootingLayout end

is_free_T(L::AbstractSingleShootingLayout) = L.free_T

struct SingleShootingLayout <: AbstractSingleShootingLayout
	nx::Int
	free_T::Bool
	T_fixed::Float64
	function SingleShootingLayout(nx::Int, free_T::Bool = true; T_fixed::Float64 = NaN)
		!free_T && isnan(T_fixed) && throw(ArgumentError("T_fixed must be provided when free_T = false"))
		return new(nx, free_T, T_fixed)
	end
end

nx(L::SingleShootingLayout)   = L.nx

nvar(L::SingleShootingLayout) = L.nx + L.free_T

@inline function unpack(L::SingleShootingLayout, z::AbstractVector)
	@boundscheck length(z) == nvar(L) || throw(DimensionMismatch(
		"z has length $(length(z)); expected $(nvar(L))",
	))
	x = @view z[1:L.nx]
	T = L.free_T ? z[L.nx+1] : L.T_fixed
	return (x = x, T = T)
end

@inline function pack(L::SingleShootingLayout, x::AbstractVector, T::Number)
	L.free_T ? vcat(x, T) : Vector(x)
end

struct SingleShootingReducedLayout <: AbstractSingleShootingLayout
	nx::Int
	free_x::Vector{Int}
	free_T::Bool
	T_fixed::Float64
	function SingleShootingReducedLayout(nx::Int, free_x::AbstractVector{Int}, free_T::Bool = true; T_fixed::Float64 = NaN)
		all(1 .<= free_x .<= nx) || throw(ArgumentError("free_x contains out-of-range indices"))
		!free_T && isnan(T_fixed) && throw(ArgumentError("T_fixed must be provided when free_T = false"))
		return new(nx, free_x, free_T, T_fixed)
	end
end

nx(L::SingleShootingReducedLayout) = L.nx

nvar(L::SingleShootingReducedLayout) = length(L.free_x) + L.free_T

@inline function unpack(L::SingleShootingReducedLayout, z::AbstractVector)
	@boundscheck length(z) == nvar(L) || throw(DimensionMismatch(
		"z has length $(length(z)); expected $(nvar(L))",
	))
	x = zeros(eltype(z), L.nx)
	@inbounds for k in 1:length(L.free_x)
		x[L.free_x[k]] = z[k]
	end
	T = L.free_T ? z[end] : L.T_fixed
	return (x = x, T = T)
end

@inline function pack(L::SingleShootingReducedLayout, x::AbstractVector, T::Number)
	L.free_T ? vcat(x[L.free_x], T) : x[L.free_x]
end