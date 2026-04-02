struct NaturalParameterShootingResidual{R <: SingleShootingResidual} <: AbstractResidual
	r::R
	index::Int
end

function residual(nat::NaturalParameterShootingResidual, z::AbstractVector, λ::Real)
	r0 = residual(nat.r, z, λ)
	dλ = z[nat.index] - λ
	return vcat(r0, dλ)
end

function step!(
	cp::ContinuationProblem{<:Any, <:SimpleNaturalParameter, <:Any},
	history::Vector{ContinuationPoint{T}};
	ds::Real,
) where {T}
	zpred, λpred = predict(cp.pre, history, ds)
	nat = NaturalParameterShootingResidual(cp.sys, cp.pre.index)
	znew, stat = SciMLBase.solve(nat, cp.corr, zpred, λpred)
	return ContinuationPoint{T}(znew, λpred), stat
end
