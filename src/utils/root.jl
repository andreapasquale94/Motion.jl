
function newton(f, ∂f, x0::T, xtol, ftol, maxiter=20) where T
    x = x0
    for _ in 1:maxiter
        fval = f( x )
        if abs(fval) ≤ ftol
            return true, x 
        end
        dx = -fval / ∂f(x)
        x += dx 

        if abs(dx) ≤ xtol 
            return true, x 
        end
    end
    return false, zero(T)
end

function ridder(f, a::T, b::T, xtol, ftol, maxiter=20) where T
    fa = f(a)
    fb = f(b)

    if fa*fb > 0
        error("f(a) and f(b) must have different signs")
    end

    if abs(fb) ≤ ftol 
        return true, b
    end 

    # initial guess for the root
    xLast = a - fa / fb * (b - a)

    for _ in 1:maxiter 
        xm = a + 0.5 * (b - a)
        fm = f(xm)
        s = sqrt(fm * fm - fa * fb)

        # update value 
        x = xm + (xm - a) * ( ( fa ≥ fb ? 1.0 : -1.0 ) * fm / s )
        fx = f(x)

        if abs(x - xLast) ≤ xtol 
            return true, x 
        end

        if abs(fx) ≤ ftol 
            return true, x 
        end

        if fx*fm < 0 
            a, fa = xm, fm  
            b, fb = x, fx 
        elseif fx*fa < 0 
            b, fb = x, fx 
        elseif fx*fb < 0 
            a, fa = x, fx 
        else 
            throw(error("Root not bracketed"))
        end

        xLast = x
    end
    return false, zero(T)
end
