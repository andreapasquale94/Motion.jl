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
