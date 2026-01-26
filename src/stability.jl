@inline function stability_index(M::AbstractMatrix)
    trM = tr(M)
    a = 2 - trM
    a2 = a * a
    b = (a2 - tr(M * M)) / 2

    disc = a2 - 4 * b + 8
    s = sqrt(disc)
    p = (a + s) / 2
    q = (a - s) / 2
    return p, q
end
