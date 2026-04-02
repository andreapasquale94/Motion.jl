# TODO: handling of fixed T (now always assuming T is free)

abstract type AbstractShootingLayout end

abstract type AbstractSingleShootingLayout <: AbstractShootingLayout end

struct SingleShootingLayout <: AbstractSingleShootingLayout
	nx::Int
	free_T::Bool
	SingleShootingLayout(nx::Int, free_T::Bool = true) = new(nx, free_T)
end

nx(L::SingleShootingLayout)   = L.nx

nvar(L::SingleShootingLayout) = L.nx + L.free_T

@inline function unpack(L::SingleShootingLayout, z::AbstractVector)
	@boundscheck length(z) == nvar(L) || throw(DimensionMismatch(
		"z has length $(length(z)); expected $(nvar(L))",
	))
	x = @view z[1:L.nx]
	T  = z[L.nx+1]
	return (x = x, T = T)
end
@inline pack(::SingleShootingLayout, x::AbstractVector, T::Number) = vcat(x, T)

struct SingleShootingReducedLayout <: AbstractSingleShootingLayout
	nx::Int
	free_x::Vector{Int}
	free_T::Bool
	function SingleShootingReducedLayout(nx::Int, free_x::AbstractVector{Int}, free_T::Bool = true)
		all(1 .<= free_x .<= nx) || throw(ArgumentError("free_x contains out-of-range indices"))
		return new(nx, free_x, free_T)
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
	T = z[end]
	return (x = x, T = T)
end

@inline pack(L::SingleShootingReducedLayout, x::AbstractVector, T::Number) = vcat(x[L.free_x], T)