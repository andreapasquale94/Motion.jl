
struct PseudoArcLength <: AbstractPredictor end

function predict(::PseudoArcLength, hist::Vector{ContinuationPoint{T}}, ds::Real) where {T}
	length(hist) ≥ 2 || throw(ArgumentError("PseudoArcLength needs at least 2 points"))
	pkm1 = hist[end-1]
	pk   = hist[end]
	wk   = vcat(pk.z, pk.λ)
	wkm1 = vcat(pkm1.z, pkm1.λ)

	t = wk .- wkm1
	nt = norm(t)
	nt == 0 && throw(ArgumentError("Identical consecutive points; cannot build tangent"))
	t ./= nt

	wpred = wk .+ T(ds) .* t
	zpred, λpred = @view(wpred[1:(end-1)]), wpred[end]
	return zpred, T(λpred), wk, t
end

struct PseudoArcLengthShootingResidual{SYS, TW, TT, S} <: AbstractResidual
	sys::SYS
	wk::TW
	t::TT
	ds::S
end

nvar(ps::PseudoArcLengthShootingResidual) = nvar(ps.sys) + 1

Base.size(ps::PseudoArcLengthShootingResidual) = size(ps.sys) + 1

function residual!(out::AbstractVector, ps::PseudoArcLengthShootingResidual, w::AbstractVector, _)
	z = @view w[1:(end-1)]
	λ = w[end]
	F = @view out[1:(end-1)]
	residual!(F, ps.sys, z, λ)
	out[end] = dot(ps.t, w .- ps.wk) - ps.ds
	return out
end

function step!(
	cp::ContinuationProblem{SYS, PseudoArcLength, C}, history::Vector{ContinuationPoint{T}};
	ds::Real,
) where {SYS, C, T}
	zpred, λpred, wk, t = predict(cp.predictor, history, ds)
	palc = PseudoArcLengthShootingResidual(cp.sys, wk, t, ds)
	wnew, stat = solve(palc, cp.corrector, vcat(zpred, λpred), zero(T))
	znew = @view(wnew[1:(end-1)])
	λnew = wnew[end]
	return ContinuationPoint{T}(znew, λnew), stat
end
