
struct ContinuationProblem{M, PR, CR}
	internal::M
	predictor::PR
	corrector::CR
end

function ContinuationProblem(internal; predictor = PseudoArcLength(), corrector::AbstractCorrector) 
	ContinuationProblem(internal, predictor, corrector)
end

"""
    step!(cp, history; ds)

Predict + correct once. Returns a new point + stats.
"""
function step!(cp::ContinuationProblem, history::Vector{ContinuationPoint{T}}; ds::Real) where {T}
    zpred, λpred, wk, t = predict(cp.predictor, history, ds)
	znew, λnew, st = correct(cp.internal, cp.predictor, cp.corrector, zpred, λpred, wk, t, ds)
	return ContinuationPoint{T}(znew, λnew), st
end