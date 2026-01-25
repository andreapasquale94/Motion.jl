"""
	make(μ, x0, t0, tf) -> EnsembleProblem

Build an `EnsembleProblem` for CR3BP dynamics with different initial conditions and time spans.

Inputs
- `μ::Number`: CR3BP mass parameter.
- `x0::AbstractMatrix{<:Number}`: state matrix of size `(6, M)`; each column is `[x,y,z,vx,vy,vz]`.
- `t0::AbstractVector{<:Number}`: start times, length `M`.
- `tf::AbstractVector{<:Number}`: final times, length `M`.
"""
function make(
	μ::Number,
	x0::AbstractMatrix{<:Number},
	t0::AbstractVector{<:Number},
	tf::AbstractVector{<:Number},
)
	size(x0, 1) == 6 || throw(ArgumentError("expected x0 with 6 rows, got size(x0)=$(size(x0))"))
	M = size(x0, 2)
	length(t0) == M || throw(ArgumentError("t0 length must match number of columns in x0 (M=$M), got $(length(t0))"))
	length(tf) == M || throw(ArgumentError("tf length must match number of columns in x0 (M=$M), got $(length(tf))"))

	Tp = promote_type(typeof(μ), eltype(x0), eltype(t0), eltype(tf))
	p  = ComponentArray(; μ = Tp(μ))

	@inbounds xb0 = SVector{6, Tp}(x0[1, 1], x0[2, 1], x0[3, 1], x0[4, 1], x0[5, 1], x0[6, 1])
	base_prob = ODEProblem(cr3bp_rhs, xb0, (Tp(t0[1]), Tp(tf[1])), p)

	function prob_func(prob, i, _)
		@inbounds begin
			u0 = SVector{6, Tp}(x0[1, i], x0[2, i], x0[3, i], x0[4, i], x0[5, i], x0[6, i])
			return remake(prob; u0 = u0, tspan = (Tp(t0[i]), Tp(tf[i])))
		end
	end
	return EnsembleProblem(base_prob; prob_func = prob_func, safetycopy = false)
end

# --- Flow 

"""
	flow(μ, x0, t0, tf, [alg...]; kwargs...) -> Matrix

Solve a batch (ensemble) of CR3BP trajectories.

Inputs
- `x0`: matrix (6, M) of initial states.
- `t0`, `tf`: vectors length M of start/end times.
- `alg...` (optional): ODE solver algorithm and options, passed to `solve`.
"""
function flow(
	μ::Number,
	x0::AbstractMatrix{<:Number},
	t0::AbstractVector{<:Number},
	tf::AbstractVector{<:Number},
	args...;
	abstol = 1e-12, reltol = 1e-12, save_everystep = false, kwargs...,
)
	M = size(x0, 2)
	prob = make(μ, x0, t0, tf)

	ens_sol = solve(
        prob, args..., EnsembleThreads(); 
        trajectories = M, save_everystep = save_everystep, reltol = reltol, abstol = abstol, 
        kwargs...
    )

	Tp = promote_type(typeof(μ), eltype(x0), eltype(t0), eltype(tf))
	xf = Matrix{Tp}(undef, 6, M)
	@inbounds for i in 1:M
		xf[:, i] .= ens_sol.u[i].u[end]
	end
	xf
end
