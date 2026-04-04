abstract type AbstractPredictor end

abstract type AbstractCorrector end

abstract type AbstractResidual end

struct ContinuationPoint{T}
	z::Vector{T}
	λ::T
end

struct ContinuationProblem{R <: AbstractResidual, PR <: AbstractPredictor, CR <: AbstractCorrector}
	sys::R
	pre::PR
	corr::CR
end

function ContinuationProblem(r::AbstractResidual;
	predictor::AbstractPredictor,
	corrector::AbstractCorrector = SciMLCorrector(SimpleNewtonRaphson(); reltol=1e-12, abstol=1e-12))
	return ContinuationProblem{typeof(r), typeof(predictor), typeof(corrector)}(r, predictor, corrector)
end


function predict end

function step! end
