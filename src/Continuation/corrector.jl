struct SciMLCorrector{ALG, KW} <: AbstractCorrector
	alg::ALG
	kwargs::KW
end
SciMLCorrector(alg = SimpleNewtonRaphson(); kwargs...) = SciMLCorrector(alg, (; kwargs...))

struct CorrectorStats
	success::Bool
	residuals::Vector{Float64}
end

function SciMLBase.solve(sys::AbstractResidual, corr::SciMLCorrector, z0::Vector{T}, λ) where {T}
	func = (z, _) -> residual(sys, z, λ)
	prob = SciMLBase.NonlinearProblem(func, z0)
	sol = SciMLBase.solve(prob, corr.alg; corr.kwargs...)
	znew = Vector{T}(sol.u)
	stats = CorrectorStats(SciMLBase.successful_retcode(sol), sol.resid)
	return znew, stats
end