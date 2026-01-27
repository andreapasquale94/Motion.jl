# --- Simple natural parameter -----------------------------------------------------------------------------

struct SimpleNaturalParameter <: AbstractPredictor
	index::Int
	sign::Float64
end

SimpleNaturalParameter(idx::Int) = SimpleNaturalParameter(idx, +1.0)

function predict(p::SimpleNaturalParameter, history::Vector{ContinuationPoint{T}}, ds::Real) where {T}
	pk = history[end]
	λpred = pk.λ + T(p.sign * ds)
	zpred = copy(pk.z)
	zpred[p.index] = λpred
	return zpred, λpred
end

struct SimpleNaturalParameterShootingResidual{SYS} <: AbstractResidual
	sys::SYS
	index::Int
end

function residual!(out::AbstractVector, nat::SimpleNaturalParameterShootingResidual, z::AbstractVector, λ::Real)
	F = @view(out[1:(end-1)])
	residual!(F, nat.sys, z, λ)
	out[end] = z[nat.index] - λ
	return out
end

nvar(nat::SimpleNaturalParameterShootingResidual) = nvar(nat.sys)

Base.size(nat::SimpleNaturalParameterShootingResidual) = size(nat.sys) + 1

function step!(
	cp::ContinuationProblem{<:Any, SimpleNaturalParameter, <:Any}, history::Vector{ContinuationPoint{T}};
	ds::Real,
) where {T}
	zpred, λpred = predict(cp.predictor, history, ds)
    nat = SimpleNaturalParameterShootingResidual(cp.sys, cp.predictor.index)
	znew, stat = SciMLBase.solve(nat, cp.corrector, zpred, λpred)
	return ContinuationPoint{T}(znew, λpred), stat
end