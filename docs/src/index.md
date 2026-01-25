# Motion.jl

**Motion.jl** is a fast, composable, and research-friendly Julia toolbox for exploring 
*simplified multi-body astrodynamics models*—with a workflow that stays close to the math while 
still scaling to real numerical experiments.

It aims to provide a clear path from *model definition → propagation → analysis → optimization*, 
balancing performance, ergonomics, and extensibility for research code.

## What you can do with Motion.jl

- Define and simulate dynamics in common restricted models (e.g. CR3BP, ER3BP, BCR4BP)
- Integrate trajectories with events, stitching, and reproducible numerical settings
- Compute variational dynamics (STMs) for sensitivity, targeting, and continuation workflows
- Explore geometry (libration points, manifolds, invariant structures) for mission design
- Prototype optimization setups (multiple shooting, constraints, costs)
- Visualize trajectories and phase-space diagnostics with lightweight plotting helpers
