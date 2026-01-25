
@fastmath function cr3bp_jacobian(x::AbstractVector{T}, μ::Number) where T
	@inbounds px, py, pz = x[1], x[2], x[3]

	px1 = px+μ
	px2 = px-1+μ

	tmp = py*py + pz*pz
	r₁ = sqrt(px1*px1 + tmp)
	r₂ = sqrt(px2*px2 + tmp)

	r₁² = r₁*r₁
	r₂² = r₂*r₂
	r₁³ = r₁²*r₁
	r₂³ = r₂²*r₂

	f₁3 = (1-μ)/r₁³
	f₂3 = μ/r₂³
	f₁5 = f₁3/r₁²
	f₂5 = f₂3/r₂²

	tmp = f₁5 + f₂5
	uxx = 1.0 - f₁3 - f₂3 + 3*px1*px1*f₁5 + 3*px2*px2*f₂5
	uyy = 1.0 - f₁3 - f₂3 + 3*py*py*tmp
	uzz = - f₁3 - f₂3 + 3*pz*pz*tmp

	uyz = 3*py*pz*tmp
	tmp = px1*f₁5 + px2*f₂5
	uxy = 3*py*tmp
	uxz = 3*pz*tmp

	return SMatrix{6, 6, T}(
		0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
		0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
		0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
		uxx, uxy, uxz, 0.0, 2.0, 0.0,
		uxy, uyy, uyz, -2.0, 0.0, 0.0,
		uxz, uyz, uzz, 0.0, 0.0, 0.0,
	)'
end
