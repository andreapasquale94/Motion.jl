module ER3BP

using StaticArrays
using LinearAlgebra
using ComponentArrays

using SciMLBase: ODEProblem, solve

using ..Motion: Solution

include("utils.jl")
include("base.jl")

export rhs, jacobian
export make, flow, build_solution

end
