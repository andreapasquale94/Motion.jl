module Continuation

using LinearAlgebra
using SciMLBase: SciMLBase
using NonlinearSolve
using ForwardDiff

export ContinuationProblem, ContinuationPoint
include("problem.jl")

export SimpleNaturalParameter
include("napar.jl")

export SimplePseudoArcLength
include("palc.jl")

export SciMLCorrector
include("corrector.jl")

export SingleShootingLayout, SingleShootingReducedLayout, SingleShooting, SingleShootingResidual
include("shooting/layout.jl")
include("shooting/constraints.jl")
include("shooting/model.jl")

export NaturalParameterShootingResidual
include("shooting/natpar.jl")

export PseudoArcLengthShootingResidual
include("shooting/palc.jl")

end
