"""
	AbstractShooting

A shooting transcription that can be used in continuation.

A concrete shooting must provide:

- `nx(sh)::Int`        (state dimension)
- `nvar(sh)::Int`      (decision vector length seen by the corrector/continuation)
- `unpack(layout, z)`  (views into the *full* decision variables needed by the residual)
- `shoot(sh, x0, T, λ) -> xf`
- `residual!(r, sh, z, λ)`
"""
abstract type AbstractShooting end

nx(::AbstractShooting)   = error("nx(::AbstractShooting) not implemented")
nvar(::AbstractShooting) = error("nvar(::AbstractShooting) not implemented")

abstract type AbstractShootingLayout end

# ----------------------------------------------------------------------------------------------------------

"""
	SingleShootingLayout(nx)

Full decision vector layout for single shooting:

	z_full = [ x0 (nx) ; T (1) ]

`unpack(layout, z_full)` returns `(x0 = view, T = scalar)`.
"""
struct SingleShootingLayout <: AbstractShootingLayout
	nx::Int
end

nx(L::SingleShootingLayout) = L.nx
nvar(L::SingleShootingLayout) = L.nx + 1

@inline function unpack(L::SingleShootingLayout, z::AbstractVector)
	@boundscheck length(z) == nvar(L) || throw(DimensionMismatch("z has length $(length(z)); expected $(nvar(L))"))
	x0 = @view z[1:L.nx]
	T  = z[L.nx+1]
	return (x0 = x0, T = T)
end

# ----------------------------------------------------------------------------------------------------------

"""
	VarMap(full_len, free_idx)

Map between a reduced decision vector `z_free` and a full vector `z_full`.

- `full_len` : length of z_full
- `free_idx` : positions in z_full that are free (1-based indices)
"""
struct VarMap{T, VI <: AbstractVector{Int}}
	full_len::Int
	free_idx::VI
end

function VarMap(full_len::Int, free_idx::AbstractVector{Int}) where {T}
	all(1 .<= free_idx .<= full_len) || throw(ArgumentError("free_idx contains out-of-range indices"))
	return VarMap{Float64, typeof(free_idx)}(full_len, free_idx)
end

nfree(vm::VarMap) = length(vm.free_idx)

@inline function __resolve_layout(z_full::AbstractVector, vm::VarMap, z_free::AbstractVector)
	@boundscheck length(z_full) == vm.full_len || throw(DimensionMismatch("z_full has length $(length(z_full)); expected $(vm.full_len)"))
	@boundscheck length(z_free) == nfree(vm) || throw(DimensionMismatch("z_free has length $(length(z_free)); expected $(nfree(vm))"))
	@inbounds for k in 1:nfree(vm)
		z_full[vm.free_idx[k]] = z_free[k]
	end
	return z_full
end

"""
	ReducedLayout(full_layout, vmap)

A layout wrapper that exposes a reduced decision vector `z_free` to the solver,
but provides `unpack(...)` for the *full* layout via an internal `z_full` cache.

This is the recommended way to "continue on a subset of the state and/or period".
"""
struct ReducedLayout{L, VM, VZ} <: AbstractShootingLayout
	full::L
	vmap::VM
	zfull::VZ
end

nx(L::ReducedLayout) = nx(L.full)
nvar(L::ReducedLayout) = nfree(L.vmap)

function ReducedLayout(full::L, vmap::VM) where {L, VM <: VarMap}
	nvar(full) == vmap.full_len || throw(DimensionMismatch("full layout expects nvar=$(nvar(full)) but vmap.full_len=$(vmap.full_len)"))
	zfull = zeros(Float64, vmap.full_len)
	return ReducedLayout{L, VM, typeof(zfull)}(full, vmap, zfull)
end

@inline function unpack(L::ReducedLayout, z_free::AbstractVector)
	z_full = zeros(eltype(z_free), length(L.zfull))
	copyto!(z_full, L.zfull)
	__resolve_layout(z_full, L.vmap, z_free)
	return unpack(L.full, z_full)
end

# ----------------------------------------------------------------------------------------------------------

"""
	SingleShooting(flow; phase = nothing, layout = SingleShootingLayout(nx))

Single shooting transcription.

`flow(x0, T, λ) -> xf` must return a vector-like `xf` of length `nx`.

Residual (periodic orbit):
- `r[1:nx] = xf - x0`
- optionally appends `phase(x0, T, λ)` (scalar or vector)
"""
struct SingleShooting{F, PH, L} <: AbstractShooting
	flow::F
	phase::PH
	layout::L
end

function SingleShooting(flow; phase = nothing, nx = nothing, layout = nothing)
	if layout === nothing
		nx === nothing && error("Provide `nx=...` (or an explicit `layout=`).")
		layout = SingleShootingLayout(Int(nx))
	end
	return SingleShooting(flow, phase, layout)
end

nx(sh::SingleShooting)   = nx(sh.layout)
nvar(sh::SingleShooting) = nvar(sh.layout)

@inline _phase_is_none(phase) = phase === nothing

@inline function _phase_eval!(out::AbstractVector, phase, x0, T, λ)
	v = phase(x0, T, λ)
	if v isa Number
		@boundscheck length(out) == 1 || throw(DimensionMismatch("phase returned scalar; out must have length 1"))
		out[1] = v
	else
		@boundscheck length(out) == length(v) || throw(DimensionMismatch("phase length mismatch"))
		copyto!(out, v)
	end
	return out
end

@inline function shoot(sh::SingleShooting, x0, T, λ)
	xf = sh.flow(x0, T, λ)
	return xf
end

function residual!(r::AbstractVector, sh::SingleShooting, z::AbstractVector, λ)
	L  = sh.layout
	u  = unpack(L, z)
	x0 = u.x0
	T  = u.T

	xf = shoot(sh, x0, T, λ)

	nx_ = nx(L)
	@boundscheck length(r) >= nx_ || throw(DimensionMismatch("r too small: $(size(r))"))
	@inbounds @views r[1:nx_] .= xf .- x0

	if !_phase_is_none(sh.phase)
		@inbounds @views _phase_eval!(r[(nx_+1):end], sh.phase, x0, T, λ)
	else
		@boundscheck length(r) == nx_ || throw(DimensionMismatch("phase is nothing, so r must have length nx"))
	end

	return r
end
