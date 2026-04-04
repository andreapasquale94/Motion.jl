struct PseudoArcLengthShootingResidual{R <: SingleShootingResidual} <: AbstractResidual
	r::R
    z0::Vector{Float64}
	t::Vector{Float64}
	ds::Float64
end

function residual(ps::PseudoArcLengthShootingResidual, z::AbstractVector, λ)
    r0 = residual(ps.r, z, λ)
    palc = dot(ps.t, z - ps.z0) - ps.ds
    return vcat(r0, palc)
end

function step!(
	cp::ContinuationProblem{<:Any, <:SimplePseudoArcLength, <:Any},
	history::Vector{ContinuationPoint{T}};
	ds::Real,
) where {T}
	zpred, λpred, t = predict(cp.pre, history, ds)
	nat = PseudoArcLengthShootingResidual(cp.sys, history[end].z, t, ds)
	znew, stat = SciMLBase.solve(nat, cp.corr, zpred, λpred)
	return ContinuationPoint{T}(znew, λpred), stat
end