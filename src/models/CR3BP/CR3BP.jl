module CR3BP

using StaticArrays
using LinearAlgebra
using ComponentArrays

using SciMLBase: ODEProblem, ContinuousCallback, remake, solve
using SciMLBase: EnsembleProblem, EnsembleThreads

using ..Motion: Solution, __acceleration_to_cartesian

include("utils.jl")
include("base.jl")
include("batch.jl")
include("constant_thrust.jl")

export rhs, rhs_const_thrust, jacobian
export make, flow, build_solution
export make_const_thrust, flow_const_thrust, build_solution_const_thrust

end
