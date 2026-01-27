
abstract type AbstractShootingLayout end

struct SingleShootingLayout <: AbstractShootingLayout
	nx::Int
end

nx(L::SingleShootingLayout)   = L.nx
nvar(L::SingleShootingLayout) = L.nx + 1

@inline function unpack(L::SingleShootingLayout, z::AbstractVector)
	@boundscheck length(z) == nvar(L) || throw(DimensionMismatch(
		"z has length $(length(z)); expected $(nvar(L))",
	))
	x0 = @view z[1:L.nx]
	T  = z[L.nx+1]
	return (x0 = x0, T = T)
end

"""
	VarMap(full_len, free_idx)

Maps reduced variables z_free into positions of a full vector z_full.
`free_idx` are indices in full decision vector that are free.
"""
struct VarMap{VI <: AbstractVector{Int}}
	full_len::Int
	free_idx::VI
	function VarMap(full_len::Int, free_idx::AbstractVector{Int})
		all(1 .<= free_idx .<= full_len) || throw(ArgumentError("free_idx contains out-of-range indices"))
		return new{typeof(free_idx)}(full_len, free_idx)
	end
end

nfree(vm::VarMap) = length(vm.free_idx)

@inline function __resolve_layout!(template::AbstractVector, vm::VarMap, z_free::AbstractVector)
	@boundscheck length(template) == vm.full_len || throw(DimensionMismatch())
	@boundscheck length(z_free) == nfree(vm) || throw(DimensionMismatch())
	@inbounds for k in 1:nfree(vm)
		template[vm.free_idx[k]] = z_free[k]
	end
	return template
end

"""
	ReducedLayout(full_layout, vmap, z0_full_template)

- Stores a Float64 template/cache `zfull::Vector{Float64}`
- If `z_free` is Float64: writes into cache (no alloc)
- Otherwise (Dual, etc.): allocates a new vector with eltype(z_free), seeded from cache
"""
struct ReducedLayout{L, VM, VZ} <: AbstractShootingLayout
	full::L
	vmap::VM
	template::VZ  # Float64 template/cache
end

nx(L::ReducedLayout)   = nx(L.full)
nvar(L::ReducedLayout) = nfree(L.vmap)

function ReducedLayout(full::L, vmap::VM, z0_full::AbstractVector{<:Real}) where {L, VM <: VarMap}
	nvar(full) == vmap.full_len || throw(DimensionMismatch(
		"full expects nvar=$(nvar(full)) but vmap.full_len=$(vmap.full_len)",
	))
	zcache = Vector{Float64}(z0_full)  # copy to Float64 cache
	length(zcache) == vmap.full_len || throw(DimensionMismatch("z0_full length mismatch"))
	return ReducedLayout{L, VM, typeof(zcache)}(full, vmap, zcache)
end

@inline function unpack(L::ReducedLayout, z_free::AbstractVector)
	z_full = similar(z_free, length(L.template))
	copyto!(z_full, L.template)
	__resolve_layout!(z_full, L.vmap, z_free)
	return unpack(L.full, z_full)
end