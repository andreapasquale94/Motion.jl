abstract type AbstractPredictor end

abstract type AbstractCorrector end

struct Corrector{ALG, KW} <: AbstractCorrector 
	alg::ALG
	kwargs::KW
end
Corrector(alg; kwargs...) = Corrector(alg, (; kwargs...))

struct CorrectorStats
	success::Bool
	residuals::Vector{Float64}
end

function SciMLBase.solve(sys::AbstractResidual, corr::Corrector, z0::Vector{T}, λ) where {T}
	func! = (out, z, _) -> residual!(out, sys, z, λ)
	prob = SciMLBase.NonlinearProblem(func!, z0)
	sol = SciMLBase.solve(prob, corr.alg; corr.kwargs...)
	znew = Vector{T}(sol.u)
	stats = CorrectorStats(SciMLBase.successful_retcode(sol), sol.resid)
	return znew, stats
end

struct ContinuationProblem{SYS, PR, CR}
	sys::SYS
	predictor::PR
	corrector::CR
end

function ContinuationProblem(sys::AbstractResidual;
	predictor::AbstractPredictor = PseudoArcLength(),
	corrector::AbstractCorrector)
	return ContinuationProblem{typeof(sys), typeof(predictor), typeof(corrector)}(sys, predictor, corrector)
end

struct ContinuationPoint{T}
	z::Vector{T}
	λ::T
end