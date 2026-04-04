using Motion.Continuation
using LinearAlgebra
using Test

@testset "Bifurcation" verbose = true begin

    # ── critical_stability_index ──────────────────────────────────────
    @testset "critical_stability_index" begin
        @test critical_stability_index(TANGENT)         ≈  2.0
        @test critical_stability_index(PERIOD_DOUBLING) ≈ -2.0
        @test critical_stability_index(PERIOD_TRIPLING)    ≈ -1.0
        @test critical_stability_index(PERIOD_QUADRUPLING) ≈  0.0 atol = 1e-15

        # General period-N
        @test critical_stability_index(1)  ≈  2.0          # 2cos(2π) = 2
        @test critical_stability_index(2)  ≈ -2.0          # 2cos(π)  = -2
        @test critical_stability_index(3)  ≈ -1.0           # 2cos(2π/3) = -1
        @test critical_stability_index(4)  ≈  0.0  atol=1e-15  # 2cos(π/2) = 0
        @test critical_stability_index(6)  ≈  1.0           # 2cos(π/3) = 1

        @test_throws ArgumentError critical_stability_index(0)
    end

    # ── period_multiple ───────────────────────────────────────────────
    @testset "period_multiple" begin
        @test period_multiple(TANGENT)         == 1
        @test period_multiple(PERIOD_DOUBLING) == 2
        @test period_multiple(PERIOD_TRIPLING)    == 3
        @test period_multiple(PERIOD_QUADRUPLING) == 4
        @test period_multiple(5)                  == 5
    end

    # ── BifurcationDetector constructors ──────────────────────────────
    @testset "BifurcationDetector" begin
        d1 = BifurcationDetector(TANGENT)
        @test d1.s_critical ≈ 2.0
        @test d1.period_mul == 1

        d2 = BifurcationDetector(PERIOD_DOUBLING)
        @test d2.s_critical ≈ -2.0
        @test d2.period_mul == 2

        d3 = BifurcationDetector(4)
        @test d3.s_critical ≈ 0.0 atol = 1e-15
        @test d3.period_mul == 4
    end

    # ── stability_index ───────────────────────────────────────────────
    @testset "stability_index" begin
        # Identity matrix: trivial monodromy → all eigenvalues at 1
        # stability indices should both be 2
        M = Matrix{Float64}(I, 6, 6)
        p, q = Continuation.stability_index(M)
        @test p ≈ 2.0 atol = 1e-12
        @test q ≈ 2.0 atol = 1e-12

        # Construct a block-diagonal symplectic matrix with known stability indices
        # Block 1 (trivial): I₂  → eigenvalues (1,1)
        # Block 2: rotation by θ → eigenvalues e^{±iθ}, s = 2cos(θ)
        # Block 3: hyperbolic    → eigenvalues (σ, 1/σ), s = σ + 1/σ
        θ  = π / 4   # s = 2cos(π/4) = √2
        σ  = 3.0     # s = 3 + 1/3 = 10/3

        R = [cos(θ) -sin(θ); sin(θ) cos(θ)]
        H = [σ 0.0; 0.0 1.0/σ]
        M2 = zeros(6, 6)
        M2[1:2, 1:2] = I(2)
        M2[3:4, 3:4] = R
        M2[5:6, 5:6] = H

        p2, q2 = Continuation.stability_index(M2)
        expected = sort([2cos(θ), σ + 1/σ]; rev = true)
        actual   = sort([p2, q2]; rev = true)
        @test actual[1] ≈ expected[1] atol = 1e-12
        @test actual[2] ≈ expected[2] atol = 1e-12
    end

    # ── detect_bifurcations ───────────────────────────────────────────
    @testset "detect_bifurcations" begin
        # Synthetic stability index sequence: p crosses +2 between indices 3→4
        si = [
            (1.5, -0.5),
            (1.8, -0.8),
            (1.95, -1.2),
            (2.1, -1.5),    # p crosses 2.0 between 3 and 4
            (2.3, -1.8),
            (2.5, -2.1),    # q crosses -2.0 between 5 and 6
        ]

        # Tangent detector (s = +2)
        det_tang = BifurcationDetector(TANGENT)
        evts = detect_bifurcations(si, det_tang)
        @test length(evts) == 1
        @test evts[1].index == 3
        @test evts[1].which == :p
        @test evts[1].s_critical ≈ 2.0

        # Period-doubling detector (s = -2)
        det_pd = BifurcationDetector(PERIOD_DOUBLING)
        evts_pd = detect_bifurcations(si, det_pd)
        @test length(evts_pd) == 1
        @test evts_pd[1].index == 5
        @test evts_pd[1].which == :q

        # No crossings for period-tripling in this sequence
        det_pt = BifurcationDetector(PERIOD_TRIPLING)
        evts_pt = detect_bifurcations(si, det_pt)
        # q crosses -1 between indices 2→3 and the p values don't cross -1
        @test any(e -> e.which == :q, evts_pt)
    end

    # ── _find_critical_eigenpair ──────────────────────────────────────
    @testset "_find_critical_eigenpair" begin
        # Matrix with a known eigenvalue at +1
        M = Matrix{Float64}(I, 6, 6)
        eval, evec = Continuation._find_critical_eigenpair(M, 2.0)
        @test abs(eval - 1.0) < 1e-10

        # Matrix with eigenvalue at -1
        M2 = copy(M)
        M2[3, 3] = -1.0
        M2[4, 4] = -1.0
        eval2, evec2 = Continuation._find_critical_eigenpair(M2, -2.0)
        @test abs(eval2 - (-1.0)) < 1e-10
    end

    # ── locate_bifurcation (mock) ─────────────────────────────────────
    @testset "locate_bifurcation" begin
        # Create a mock family where the stability index linearly
        # crosses +2 as a function of λ.
        # Monodromy = block-diag(I₂, R(θ(λ)), H) with θ(λ) tuned so
        # that s = 2cos(θ) crosses 2 at λ = 0.5.

        function mock_monodromy(cp)
            # s(λ) = 2cos(θ(λ)) with θ(λ) = π*(1-λ)/4
            # s = 2 when θ = 0, i.e. λ = 1
            # We want crossing near λ_bif ≈ 0.5
            # Use s(λ) = 2 - 4*(1 - λ), so s(0.5) = 0, s(1) = 2
            # Actually let's just build a diagonal monodromy
            # with s_p = 2*λ (crosses 2 at λ=1) to be simple.
            # Eigenvalue pair: λ_eig satisfies λ_eig^2 - s*λ_eig + 1 = 0
            # We just use a diagonal with eigenvalues tuned accordingly.
            λ = cp.λ
            # Simple: s = 4*λ - 2, crosses 2 at λ = 1.0
            # At λ=0.4: s = -0.4, at λ=0.6: s = 0.4
            # Let's just target s crosses 2 at λ = 0.5 for easy verification
            # s = 4*(λ - 0.25), so s(0.25) = 0, s(0.5) = 1, s(0.75) = 2
            # Use s = 8/3*(λ), so s(0.75) = 2.0
            # Actually let's just linearly interpolate:
            # p(λ) = 4λ, crosses 2 at λ = 0.5
            s_val = 4.0 * λ

            # Build block diagonal: [I₂, rotation with angle, I₂]
            # For the stability index to be s_val, we need eigenvalues
            # satisfying λ^2 - s_val*λ + 1 = 0
            # If |s_val| < 2: complex eigenvalues → rotation
            # If |s_val| >= 2: real eigenvalues
            M = zeros(6, 6)
            M[1, 1] = 1.0; M[2, 2] = 1.0  # trivial pair
            M[5, 5] = 1.0; M[6, 6] = 1.0  # second trivial pair → q ≈ 2

            if abs(s_val) < 2.0
                θ = acos(s_val / 2.0)
                M[3, 3] = cos(θ);  M[3, 4] = -sin(θ)
                M[4, 3] = sin(θ);  M[4, 4] = cos(θ)
            else
                disc = s_val^2 - 4.0
                e1 = (s_val + sqrt(max(disc, 0.0))) / 2.0
                e2 = 1.0 / e1
                M[3, 3] = e1; M[4, 4] = e2
            end

            return M
        end

        # Identity correction: just return z as-is (orbits are "exact")
        correct_fn(z, λ) = (z, true)

        # Build a simple family: λ goes from 0.3 to 0.7
        family = [ContinuationPoint{Float64}(rand(4), λ) for λ in 0.3:0.05:0.7]

        # Compute stability indices
        si = [Continuation.stability_index(mock_monodromy(cp)) for cp in family]

        # Detect tangent bifurcation
        det = BifurcationDetector(TANGENT)
        evts = detect_bifurcations(si, det)
        @test length(evts) >= 1

        # Locate
        bp = locate_bifurcation(family, evts[1], correct_fn, mock_monodromy; tol = 1e-8)
        @test abs(bp.cp.λ - 0.5) < 1e-3
        @test abs(bp.stability_indices[1] - 2.0) < 1e-3 || abs(bp.stability_indices[2] - 2.0) < 1e-3
    end

    # ── exploit_bifurcation (mock) ────────────────────────────────────
    @testset "exploit_bifurcation" begin
        # Create a BifurcationPoint manually
        z0 = [1.0, 0.0, 0.0, 0.5]
        cp = ContinuationPoint{Float64}(z0, 0.5)
        M  = Matrix{Float64}(I, 6, 6)
        ev = zeros(Complex{Float64}, 6); ev[3] = 1.0  # z-direction
        bp = BifurcationPoint{Float64}(cp, M, (2.0, 2.0), 1.0 + 0.0im, ev, 1)

        # Mock corrector just returns the perturbed state
        correct_fn(z, λ) = (z, true)

        cp_new = exploit_bifurcation(bp, correct_fn; ε = 1e-3)
        @test cp_new.λ ≈ 0.5
        @test cp_new.z[3] ≈ 1e-3 atol = 1e-10  # perturbed in z-direction
        @test cp_new.z[1] ≈ 1.0                 # unchanged

        # Custom perturbation
        pert = [0.0, 0.0, 0.0, 1.0]
        cp_new2 = exploit_bifurcation(bp, correct_fn; ε = 0.01, perturbation = pert)
        @test cp_new2.z[4] ≈ 0.5 + 0.01 atol = 1e-10
    end

    # ── BrouckeData / broucke_diagram ────────────────────────────────
    @testset "broucke_diagram" begin
        # Reuse mock monodromy from locate_bifurcation test
        function mock_mono_bd(cp)
            λ = cp.λ
            s_val = 4.0 * λ  # p(λ) = 4λ, crosses 2 at λ = 0.5
            M = zeros(6, 6)
            M[1, 1] = 1.0; M[2, 2] = 1.0
            M[5, 5] = 1.0; M[6, 6] = 1.0
            if abs(s_val) < 2.0
                θ = acos(s_val / 2.0)
                M[3, 3] = cos(θ);  M[3, 4] = -sin(θ)
                M[4, 3] = sin(θ);  M[4, 4] = cos(θ)
            else
                disc = s_val^2 - 4.0
                e1 = (s_val + sqrt(max(disc, 0.0))) / 2.0
                e2 = 1.0 / e1
                M[3, 3] = e1; M[4, 4] = e2
            end
            return M
        end

        family = [ContinuationPoint{Float64}(rand(4), λ) for λ in 0.1:0.1:0.9]

        # Without detectors
        bd = broucke_diagram(family, mock_mono_bd)
        @test length(bd.parameter) == length(family)
        @test length(bd.p) == length(family)
        @test length(bd.q) == length(family)
        @test isempty(bd.bifurcations)

        # With detectors
        bd2 = broucke_diagram(family, mock_mono_bd; detectors = ALL_STANDARD_DETECTORS)
        @test !isempty(bd2.bifurcations)
        # Should detect tangent crossing (p crosses 2 at λ ≈ 0.5)
        tang_events = filter(e -> e.s_critical ≈ 2.0, bd2.bifurcations)
        @test !isempty(tang_events)

        # From pre-computed stability indices
        si = [(bd.p[k], bd.q[k]) for k in eachindex(bd.p)]
        bd3 = broucke_diagram(si, bd.parameter;
            detectors = [BifurcationDetector(TANGENT)])
        @test bd3.p ≈ bd.p
        @test bd3.q ≈ bd.q
        @test !isempty(bd3.bifurcations)

        # Custom parameter function
        bd4 = broucke_diagram(family, mock_mono_bd;
            parameter_fn = cp -> 2 * cp.λ)
        @test bd4.parameter ≈ 2 .* bd.parameter
    end

end
