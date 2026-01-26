module Continuation

using SciMLBase: SciMLBase

export SingleShootingLayout, ReducedLayout, nx, nvar 
export SingleShooting, shoot, residual!, restrict
include("shooting.jl")
include("model.jl")
include("problem.jl")

end