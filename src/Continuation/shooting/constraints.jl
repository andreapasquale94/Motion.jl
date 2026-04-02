
abstract type AbstractConstraint end

struct Periodicity{L <: AbstractShootingLayout} <: AbstractConstraint
	layout::L
end

@inline function residual(c::Periodicity, x0, x, T, λ)
    r = pack(c.layout, x, T) - pack(c.layout, x0, T)
    return r[1:end-1] # remove period
end

struct HalfPeriodSymmetry <: AbstractConstraint
	idx::Vector{Int}
end

@inline function residual(c::HalfPeriodSymmetry, x0, x, T, λ)
    return x[c.idx]
end