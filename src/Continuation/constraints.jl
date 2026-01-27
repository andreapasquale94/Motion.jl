
abstract type AbstractConstraint end

Base.size(::AbstractConstraint) = error("size(constraint, shooting) not implemented")

function constraint!(out, c::AbstractConstraint, x0, T, xf, λ)
	error("constraint!(...) not implemented")
end

"Lightweight closure-based constraint."
struct GenericConstraint{F} <: AbstractConstraint
	func::F
	m::Int
end
Base.size(c::GenericConstraint) = c.m
@inline function constraint!(out, c::GenericConstraint, x0, T, xf, λ)
	c.func(out, x0, T, xf, λ);
	out
end

"Half-period symmetry"
struct HalfPeriodSymmetry{IDX} <: AbstractConstraint
	idx::IDX  
end

Base.size(c::HalfPeriodSymmetry) = length(c.idx)

@inline function constraint!(out, c::HalfPeriodSymmetry, x0, T, xf, λ)
	@inbounds for (k, i) in pairs(c.idx)
		out[k] = xf[i]
	end
	return out
end

"Full periodicity"
struct Periodicity{IDX} <: AbstractConstraint
	idx::IDX  
end

Base.size(c::Periodicity) = length(c.idx)

@inline function constraint!(out, c::Periodicity, x0, T, xf, λ)
	@inbounds for (k, i) in pairs(c.idx)
		out[k] = xf[i] - x0[i]
	end
	return out
end