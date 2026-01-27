
"""
	AbstractShooting

Propagation/transcription only.

Must provide:
- `nx(sh)::Int`
- `nvar(sh)::Int`         (decision vector length exposed to solvers)
- `unpack(layout, z)`     -> NamedTuple with at least (x0, T)
- `shoot(sh, x0, T, λ)`   -> xf
"""
abstract type AbstractShooting end

nx(::AbstractShooting)   = error("nx(::AbstractShooting) not implemented")

nvar(::AbstractShooting) = error("nvar(::AbstractShooting) not implemented")


"""
	ShootingArc(flow; layout)

`flow` may be either:
- `flow(x0, T)` or
- `flow(x0, T, λ)`

Returns final state `xf`.
"""
struct ShootingArc{F, L} <: AbstractShooting
	flow::F
	layout::L
end

nx(sh::ShootingArc)   = nx(sh.layout)
nvar(sh::ShootingArc) = nvar(sh.layout)

@inline function shoot(sh::ShootingArc, x0, T, λ)
	f = sh.flow
	if applicable(f, x0, T, λ)
		return f(x0, T, λ)
	elseif applicable(f, x0, T)
		return f(x0, T)
	else
		throw(MethodError(f, (x0, T, λ)))
	end
end

"""
A residual system usable by continuation / corrector.
Must provide:
- `nvar(sys)::Int`
- `Base.size(sys)::Int`
- `residual!(out, sys, z, λ)`
"""
abstract type AbstractResidual end

Base.size(::AbstractResidual) = error("Base.size(::AbstractResidual) not implemented")

struct SingleShootingResidual{SH, C} <: AbstractResidual
	shooter::SH
	constraint::C
end

nvar(sys::SingleShootingResidual) = nvar(sys.shooter)

Base.size(sys::SingleShootingResidual) = size(sys.constraint)

function residual!(out::AbstractVector, sys::SingleShootingResidual, z::AbstractVector, λ)
	u  = unpack(sys.shooter.layout, z)
	x0 = u.x0
	T  = u.T
	xf = shoot(sys.shooter, x0, T, λ)
	constraint!(out, sys.constraint, x0, T, xf, λ)
	return out
end