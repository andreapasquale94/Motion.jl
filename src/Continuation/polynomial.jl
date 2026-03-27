"""
	PolynomialPredictor(; step0, tangent0 = nothing, hmin = step0 / 10, hmax = 10 * step0,
		hfail = step0 / 2, dhmax = 2.0, max_degree = 4, err_abs = sqrt(eps(Float64)),
		err_rel = 1e-6)

Adaptive Newton divided-difference predictor in the extended continuation space `(z, λ)`.

- With a single seed point, `tangent0` must provide the initial tangent in extended space.
- With two or more points, the predictor builds a Newton interpolation polynomial using the
  last corrected points and evaluates it at the current continuation step.
- The correction step remains pseudo-arclength based, so the continuation problem should
  satisfy the same square-system assumptions as [`PseudoArcLength`](@ref).

The predictor stores the current step size and polynomial degree internally. Call
[`step!`](@ref) without `ds` to reuse the adapted step from the previous iteration, or pass
`ds = ...` to override the step used for the current correction.
"""
mutable struct PolynomialPredictor{T, VT <: Union{Nothing, Vector{T}}} <: AbstractPredictor
	h0::T
	hk::T
	t0::VT
	hmin::T
	hmax::T
	hfail::T
	dhmax::T
	max_degree::Int
	degree::Int
	err_abs::T
	err_rel::T
end

function PolynomialPredictor(;
	step0::Real,
	tangent0 = nothing,
	hmin::Real = step0 / 10,
	hmax::Real = 10 * step0,
	hfail::Real = step0 / 2,
	dhmax::Real = 2.0,
	max_degree::Integer = 4,
	err_abs::Real = sqrt(eps(Float64)),
	err_rel::Real = 1e-6,
)
	T = promote_type(
		typeof(float(step0)),
		typeof(float(hmin)),
		typeof(float(hmax)),
		typeof(float(hfail)),
		typeof(float(dhmax)),
		typeof(float(err_abs)),
		typeof(float(err_rel)),
	)

	step0T = T(step0)
	hminT = T(hmin)
	hmaxT = T(hmax)
	hfailT = T(hfail)
	dhmaxT = T(dhmax)
	err_absT = T(err_abs)
	err_relT = T(err_rel)

	step0T > zero(T) || throw(ArgumentError("step0 must be positive"))
	hminT > zero(T) || throw(ArgumentError("hmin must be positive"))
	hmaxT >= hminT || throw(ArgumentError("hmax must be at least hmin"))
	hfailT > zero(T) || throw(ArgumentError("hfail must be positive"))
	dhmaxT >= one(T) || throw(ArgumentError("dhmax must be at least 1"))
	max_degree >= 1 || throw(ArgumentError("max_degree must be at least 1"))
	err_absT >= zero(T) || throw(ArgumentError("err_abs must be non-negative"))
	err_relT >= zero(T) || throw(ArgumentError("err_rel must be non-negative"))

	tangent = tangent0 === nothing ? nothing : Vector{T}(tangent0)
	return PolynomialPredictor{T, typeof(tangent)}(
		step0T,
		step0T,
		tangent,
		hminT,
		hmaxT,
		hfailT,
		dhmaxT,
		max_degree,
		1,
		err_absT,
		err_relT,
	)
end

stepsize(p::PolynomialPredictor) = p.hk

polynomial_degree(p::PolynomialPredictor) = p.degree

function _extended_point(point::ContinuationPoint{T}) where {T}
	w = Vector{T}(undef, length(point.z) + 1)
	w[1:(end-1)] .= point.z
	w[end] = point.λ
	return w
end

function _extended_history_matrix(history::AbstractVector{ContinuationPoint{T}}) where {T}
	npt = length(history)
	nvar = length(history[1].z) + 1
	X = Matrix{T}(undef, npt, nvar)
	for i in eachindex(history)
		X[i, 1:(end-1)] .= history[i].z
		X[i, end] = history[i].λ
	end
	return X
end

function _euclidean_abscissa(X::AbstractMatrix{T}) where {T}
	npt = size(X, 1)
	s = Vector{T}(undef, npt)
	xk = @view X[end, :]
	for i in 1:npt
		s[i] = -norm(xk .- @view(X[i, :]))
	end
	s[end] = zero(T)
	return s
end

function _reference_tangent(p::PolynomialPredictor{T}, history::Vector{ContinuationPoint{T}}) where {T}
	wk = _extended_point(history[end])
	t = if length(history) == 1
		p.t0 === nothing && throw(ArgumentError(
			"PolynomialPredictor needs tangent0 when history contains only one point",
		))
		length(p.t0) == length(wk) || throw(DimensionMismatch(
			"tangent0 has length $(length(p.t0)); expected $(length(wk))",
		))
		copy(p.t0)
	else
		wkm1 = _extended_point(history[end-1])
		wk .- wkm1
	end

	nt = norm(t)
	nt == 0 && throw(ArgumentError("Identical consecutive points; cannot build tangent"))
	t ./= nt
	return wk, t
end

function _polynomial_prediction(
	p::PolynomialPredictor{T},
	history::Vector{ContinuationPoint{T}},
	ds::T,
) where {T}
	length(history) >= 1 || throw(ArgumentError(
		"PolynomialPredictor needs at least one point in the continuation history",
	))

	wk, t = _reference_tangent(p, history)

	if length(history) == 1
		wpred = wk .+ ds .* t
		return wpred, wk, t, nothing, 1
	end

	degree = min(p.degree, length(history) - 1)
	window = history[(end - degree):end]
	X = _extended_history_matrix(window)
	s = _euclidean_abscissa(X)
	poly = NewtonPolynomial(s, X)
	wpred = poly(ds)
	return wpred, wk, t, poly, degree
end

function predict(p::PolynomialPredictor, history::Vector{ContinuationPoint{T}}, ds::Real) where {T}
	wpred, wk, t, _, _ = _polynomial_prediction(p, history, T(ds))
	zpred = Vector{T}(wpred[1:(end-1)])
	λpred = wpred[end]
	return zpred, λpred, wk, t
end

function _update_polynomial_predictor!(
	p::PolynomialPredictor{T},
	history::Vector{ContinuationPoint{T}},
	point::ContinuationPoint{T},
	success::Bool,
	wpred::AbstractVector{T},
	poly::Union{Nothing, NewtonPolynomial{T}},
	degree_used::Int,
	hused::T,
) where {T}
	if success
		wnew = _extended_point(point)
		wnorm = norm(wnew)
		wnorm = wnorm == 0 ? one(T) : wnorm

		new_degree = degree_used
		lowered_order = false
		E_rel_m = norm(wnew .- wpred) / wnorm
		E_rel = E_rel_m

		if degree_used > 1 && poly !== nothing
			wguess_mm1 = reduced_polynomial(poly, 1)(hused)
			E_rel_mm1 = norm(wnew .- wguess_mm1) / wnorm

			if E_rel_mm1 < E_rel_m
				new_degree = degree_used - 1
				lowered_order = true
				E_rel = E_rel_mm1
			end

			if degree_used > 2
				wguess_mm2 = reduced_polynomial(poly, 2)(hused)
				E_rel_mm2 = norm(wnew .- wguess_mm2) / wnorm

				if E_rel_mm2 < E_rel_mm1 || E_rel_mm2 < E_rel_m
					new_degree = degree_used - 2
					lowered_order = true
					E_rel = E_rel_mm2
				end
			end
		end

		gmk = one(T)
		if length(history) + 1 > 2
			histn = [history; point]
			window = histn[(end - degree_used - 1):end]
			Xn = _extended_history_matrix(window)
			sn = _euclidean_abscissa(Xn)
			fit_mp1 = NewtonPolynomial(sn, Xn)
			next_h = solve_consecutive_step(hused, fit_mp1, wnew, p.err_rel)
			gmk = isfinite(next_h) ? next_h / hused : p.dhmax
		else
			E_abs = E_rel * wnorm
			if E_abs < p.err_abs
				gmk = T(1.5)
			end
			if E_rel < p.err_rel
				gmk = T(2)
			end
		end

		p.hk = min(max(T(0.1), gmk), p.dhmax) * hused

		if length(history) + 1 > 2 && gmk > one(T) && !lowered_order
			new_degree = degree_used + 1
		end

		p.degree = min(new_degree, p.max_degree)
	else
		p.hk = min(p.hfail, hused)
		if p.degree > 3
			p.degree -= 1
		end
	end

	p.hk = min(p.hmax, max(p.hk, p.hmin))
	return p
end

function step!(
	cp::ContinuationProblem{SYS, P, C},
	history::Vector{ContinuationPoint{T}};
	ds::Real = cp.predictor.hk,
) where {SYS, P <: PolynomialPredictor, C, T}
	hused = T(ds)
	hused > zero(T) || throw(ArgumentError("PolynomialPredictor expects a positive ds"))
	cp.predictor.hk = hused

	wpred, wk, t, poly, degree_used = _polynomial_prediction(cp.predictor, history, hused)
	zpred = Vector{T}(wpred[1:(end-1)])
	λpred = wpred[end]

	palc = PseudoArcLengthShootingResidual(cp.sys, wk, t, hused)
	wnew, stat = SciMLBase.solve(palc, cp.corrector, vcat(zpred, λpred), zero(T))
	znew = Vector{T}(wnew[1:(end-1)])
	λnew = wnew[end]
	point = ContinuationPoint{T}(znew, λnew)

	_update_polynomial_predictor!(
		cp.predictor,
		history,
		point,
		stat.success,
		wpred,
		poly,
		degree_used,
		hused,
	)

	return point, stat
end
