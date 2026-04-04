# ──────────────────────────────────────────────────────────────────────
#  Hermite-Simpson Direct Collocation
#
#  Transcribes a continuous-time optimal-control problem into an NLP
#  by enforcing the dynamics via collocation constraints rather than
#  numerical integration.
#
#  Two formulations are provided:
#    - **Compressed** (`:HermiteSimpson`): midpoint states are computed
#      implicitly via Hermite interpolation.  Decision variables are
#      states and controls at nodes only.
#    - **Separated** (`:HermiteSimpsonSeparated`): midpoint states and
#      controls are explicit decision variables, yielding a sparser
#      Jacobian at the cost of more variables.
#
#  Mesh refinement is supported via error estimation and bisection.
#
#  References
#  ----------
#  [1] Pritchett, "Strategies for Low-Thrust Trajectory Design Based
#      on Direct Collocation Techniques", Purdue PhD, 2020.
#  [2] Betts, "Practical Methods for Optimal Control and Estimation
#      Using Nonlinear Programming", SIAM, 2010, Ch. 4.
#  [3] Herman & Conway, "Direct optimization using collocation based
#      on high-order Gauss-Lobatto quadrature rules", JGCD, 1996.
# ──────────────────────────────────────────────────────────────────────
module Collocation

using StaticArrays
using LinearAlgebra

export indexes, indexes_separated
export variables, variables_separated
export defects, objective, objective_separated
export mesh_error, refine_mesh

include("layout.jl")
include("defects.jl")
include("objectives.jl")
include("mesh_refinement.jl")

end # module
