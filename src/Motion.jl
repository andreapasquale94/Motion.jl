"""
    Motion

A multibody trajectory design toolkit for semi-analytical models used in
mission analysis (CR3BP, BCR4BP, etc.).

Provides dynamical models (`CR3BP`), continuation methods (`Continuation`),
and trajectory optimisation algorithms (`MultipleShooting`, `DDP`, `MDDP`).

# Submodules
- `Motion.CR3BP`          – Circular Restricted Three-Body Problem dynamics
- `Motion.Continuation`   – Numerical continuation (natural parameter, pseudo-arc-length)
- `Motion.MultipleShooting` – Multiple-shooting transcription helpers
- `Motion.DDP`            – Constrained Differential Dynamic Programming (iLQR/DDP)
- `Motion.MDDP`           – Multiple-Shooting DDP (Pellegrini & Russell)
"""
module Motion

using StaticArrays
using LinearAlgebra
using ComponentArrays

using SciMLBase: ODEProblem, ContinuousCallback, remake, solve
using SciMLBase: EnsembleProblem, EnsembleThreads
using NonlinearSolve

# ── Common types and utilities ───────────────────────────────────────

export Solution, SensitivitySolution, BatchSolution
include("solution.jl")

export libration_points, libration_point
include("libration_points.jl")

export compute_stretch
include("measures.jl")

# ── Dynamical models ────────────────────────────────────────────────

include("dynamics/CR3BP/CR3BP.jl")

# ── Optimisation algorithms ─────────────────────────────────────────

include("Continuation/Continuation.jl")
include("optim/MultipleShooting.jl")
include("optim/DDP/DDP.jl")
include("optim/MDDP/MDDP.jl")

end
