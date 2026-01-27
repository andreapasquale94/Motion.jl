module Continuation

using LinearAlgebra
using SciMLBase: SciMLBase

export SingleShootingLayout, VarMap, ReducedLayout, unpack
include("layout.jl")

export AbstractConstraint, constraint!, GenericConstraint, HalfPeriodSymmetry, Periodicity
include("constraints.jl")

export ShootingArc, SingleShootingResidual
include("shooting.jl")

export Corrector, solve
include("corrector.jl")

end