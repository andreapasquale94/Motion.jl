struct ContinuationPoint{T}
	z::Vector{T}     # decision vector seen by solver (could be reduced)
	λ::T             # scalar continuation parameter (keep MVP simple)
end

# ---- Predictors ----

abstract type AbstractPredictor end

"""
	NaturalParameter()

Predict by changing λ only:
	λ_pred = λ_k + ds
	z_pred = z_k
"""
struct NaturalParameter <: AbstractPredictor end

"""
	PseudoArcLength()

Predict along secant in extended space (z, λ).
Requires at least 2 previous points.
"""
struct PseudoArcLength <: AbstractPredictor end

function predict(n::NaturalParameter, history::Vector{ContinuationPoint{T}}, ds::Real) where {T}
	pk = history[end]
	return copy(pk.z), pk.λ + n.Δλ_sign * T(ds), nothing, nothing
end

function predict(::PseudoArcLength, history::Vector{ContinuationPoint{T}}, ds::Real) where {T}
	length(history) ≥ 2 || throw(ArgumentError("PseudoArcLength needs at least 2 points"))
	pkm1 = history[end-1]
	pk   = history[end]

	wk   = vcat(pk.z, pk.λ)
	wkm1 = vcat(pkm1.z, pkm1.λ)

	t = wk .- wkm1
	nt = norm(t)
	nt == 0 && throw(ArgumentError("Identical consecutive points; cannot build tangent"))
	t ./= nt

	wpred = wk .+ T(ds) .* t
	zpred = @view(wpred[1:(end-1)])
	λpred = wpred[end]
	return zpred, λpred, wk, t
end

# ---- Correctors ----

abstract type AbstractCorrector end

"""
	Corrector(alg; kwargs...)

Rootfinding corrector based on SciMLBase `NonlinearProblem` + `solve`.

- `alg` is a solver algorithm object (e.g. `NewtonRaphson()`, `TrustRegion()`)
  provided by a loaded solver package (typically NonlinearSolve).
- `kwargs...` are forwarded to `solve`.
"""
struct Corrector{ALG, KW} <: AbstractCorrector
	alg::ALG
	kwargs::KW  # NamedTuple
end
Corrector(alg; kwargs...) = Corrector(alg, (; kwargs...))

struct CorrectorStats
	success::Bool
	iters::Int
	resnorm::Float64
end

@inline function _iters(sol)
	st = getproperty(sol, :stats, nothing)
	if st === nothing
		return -1
	end
	return getproperty(st, :iters, getproperty(st, :nsteps, getproperty(st, :nf, -1)))
end

function correct(sh::SingleShooting, ::NaturalParameter, corr::Corrector,
	z0::Vector{T}, λfixed::T, args...) where {T}
	f! = (out, z, p) -> begin
		residual!(out, sh, z, λfixed)
		return out
	end
	prob = SciMLBase.NonlinearProblem(f!, z0)
	sol = SciMLBase.solve(prob, corr.alg; corr.kwargs...)
	znew = Vector{T}(sol.u)
	stats = CorrectorStats(
		SciMLBase.successful_retcode(sol),
		_iters(sol),
		0.0,
	)
	return znew, λfixed, stats
end

function correct(sh::SingleShooting, ::PseudoArcLength, corr::Corrector,
	zpred::Vector{T}, λpred::T, wk::Vector{T}, t::Vector{T}, ds::Real) where {T}

	f!   = (out, w, p) -> begin
		z = @view w[1:(end-1)]
		λ = w[end]
		F = @view out[1:(end-1)]
		residual!(F, sh, z, λ)
		out[end] = dot(t, w .- wk) - ds
		return out
	end
	prob = SciMLBase.NonlinearProblem(f!, w0)
	sol  = SciMLBase.solve(prob, corr.alg; corr.kwargs...)
	w    = Vector{T}(sol.u)

	znew = @view(w[1:(end-1)])
	λnew = w[end]

	stats = CorrectorStats(
		SciMLBase.successful_retcode(sol),
		_iters(sol),
		0.0,
	)
	return znew, λnew, stats
end
