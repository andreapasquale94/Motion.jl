module Continuation

using LinearAlgebra
using SciMLBase: SciMLBase

export SingleShootingLayout, VarMap, ReducedLayout, unpack
include("layout.jl")

export AbstractConstraint, constraint!, GenericConstraint, HalfPeriodSymmetry, Periodicity
include("constraints.jl")

export ShootingArc, SingleShootingResidual
include("shooting.jl")

export AbstractResidual, AbstractPredictor, AbstractPredictor
export ContinuationPoint, nvar, ContinuationProblem, Corrector, solve
include("problem.jl")

export SimpleNaturalParameter
include("natural_param.jl")

export PseudoArcLength
include("pseudo_arclen.jl")

end