# The maps: their derivatives against ForwardDiff's, past every box; the
# frame against the inverse Jacobian; the Piola identity; the number types;
# and the patch properties.

const NPOINTS = 40

@testset "Every layout is isbits in Float64 and Float32" begin
    # A layout that is not isbits cannot be a kernel argument on a device.
    for (name, layout) in LAYOUTS
        @test isbits(layout)
        L32 = typeof(layout).name.wrapper{Float32}
        @test isbitstype(L32)
    end
    @test isbits(SevenPatchBall{Float32}(1, 3, (2, 2)))
end

@testset "The Jacobian and the Hessian are ForwardDiff's, past the box too: $name" for (name, layout) in LAYOUTS
    # A hand-derived derivative that is wrong somewhere, or only inside the
    # box, where an interpolated ghost's position needs it past the box.
    rng = Xoshiro(1)
    D = dimension(layout)
    for p in 1:npatches(layout), _ in 1:NPOINTS
        ξ = random_point(rng, layout, p)
        f(y) = SVector(chart_map(layout, Tuple(y), p))
        J = chart_jacobian(layout, ξ, p)
        Jf = ForwardDiff.jacobian(f, SVector(ξ))
        @test maximum(abs(J[i][a] - Jf[i, a]) for i in 1:D, a in 1:D) <=
              1e-14 * max(1, maxabs(Jf))
        H = chart_hessian(layout, ξ, p)
        g(y) = SVector(flat(chart_jacobian(layout, Tuple(y), p))...)
        Hf = ForwardDiff.jacobian(g, SVector(ξ))    # row (i-1)D + a, column b
        @test maximum(abs(H[i][a][b] - Hf[(i - 1) * D + a, b])
                      for i in 1:D, a in 1:D, b in 1:D) <= 1e-13 * max(1, maxabs(Hf))
        # The Hessian is symmetric in its two logical indices.
        symmetric = all(H[i][a][b] == H[i][b][a] for i in 1:D, a in 1:D, b in 1:D)
        @test symmetric
    end
end

@testset "The frame is √g and √g ∇ξ, past the box too: $name" for (name, layout) in LAYOUTS
    # Area vectors formed from the cofactors with a wrong sign or index,
    # or a √g that is the signed determinant on a left-handed patch.
    rng = Xoshiro(2)
    D = dimension(layout)
    for p in 1:npatches(layout), _ in 1:NPOINTS
        ξ = random_point(rng, layout, p)
        Jf = ForwardDiff.jacobian(y -> SVector(chart_map(layout, Tuple(y), p)), SVector(ξ))
        J, S = patch_frame(layout, ξ, p)
        @test J > 0
        @test J ≈ abs(det(Jf)) rtol = 1e-14
        K = abs(det(Jf)) * inv(Jf)
        @test maximum(abs(S[d][i] - K[d, i]) for d in 1:D, i in 1:D) <=
              1e-13 * maxabs(K)
        @test volume_element(layout, ξ, p) == J
        @test area_vectors(layout, ξ, p) == S
        # `S[d] ⋅ ∂_d x = √g`, the sign that makes the flux outward.
        @test all(sum(S[d][i] * Jf[i, d] for i in 1:D) ≈ J for d in 1:D)
    end
end

@testset "Σ_d ∂_d S[d] = 0 to rounding on every patch: $name" for (name, layout) in LAYOUTS
    # The Piola identity, which a discrete divergence of the flux relies on
    # to keep a constant constant; a division in forming `S` would break it
    # at the rounding of the division, not of the products.
    rng = Xoshiro(3)
    D = dimension(layout)
    for p in 1:npatches(layout), _ in 1:NPOINTS
        ξ = random_point(rng, layout, p)
        s(y) = SVector(flat(area_vectors(layout, Tuple(y), p))...)
        dS = ForwardDiff.jacobian(s, SVector(ξ))    # row (d-1)D + i, column b
        div = [sum(dS[(d - 1) * D + i, d] for d in 1:D) for i in 1:D]
        @test maximum(abs, div) <= 1e-14 * max(1, maxabs(dS))
    end
end

@testset "The maps evaluate in Float64x2 and BigFloat, and ForwardDiff differentiates them there: $name" for (name, layout) in LAYOUTS
    # A constant rounded to the layout's Float64, a Float64 literal, or a
    # function that MultiFloats does not have, which would limit the
    # extended-precision tables to Float64's accuracy or fail outright.
    rng = Xoshiro(4)
    D = dimension(layout)
    setprecision(BigFloat, 256) do
        for p in 1:npatches(layout), _ in 1:10
            ξ = random_point(rng, layout, p)
            ξ2, ξb = Float64x2.(ξ), BigFloat.(ξ)
            for f in (chart_map, chart_jacobian, chart_hessian, volume_element,
                      area_vectors)
                a = flat(f(layout, ξ2, p))
                b = flat(f(layout, ξb, p))
                @test eltype(a) == Float64x2
                @test eltype(b) == BigFloat
                @test maximum(abs.(BigFloat.(a) .- b)) <= 1e-29 * max(1, maxabs(b))
            end
            Jf = ForwardDiff.jacobian(y -> SVector(chart_map(layout, Tuple(y), p)),
                                      SVector(ξ2))
            J = chart_jacobian(layout, ξ2, p)
            @test maximum(abs(BigFloat(J[i][a] - Jf[i, a])) for i in 1:D, a in 1:D) <=
                  1e-29 * max(1, maxabs(BigFloat.(Jf)))
            # The inverse too, at a point inside the box, which every
            # layout's domain holds.
            x = chart_map(layout, Float64x2.(random_inside(rng, layout, p)), p)
            loc = chart_locate(layout, x)
            @test loc !== nothing
            @test maximum(abs.(BigFloat.(chart_map(layout, loc[1], loc[2])) .-
                               BigFloat.(x))) <= 1e-29 * max(1, maxabs(BigFloat.(x)))
        end
    end
    # A Float32 layout evaluates in Float32, and a Float32 point in a
    # Float64 layout in Float64.
    L32 = typeof(layout).name.wrapper{Float32}
    l32 = L32(layout)
    @test l32 isa L32
    ξ = ntuple(_ -> 0.25f0, D)
    @test eltype(flat(chart_jacobian(l32, ξ, 2))) == Float32
    @test eltype(flat(chart_jacobian(layout, ξ, 2))) == Float64
end

@testset "The volume factors multiply to √g: $name" for (name, layout) in LAYOUTS
    # A patch that declares a factorization its √g does not have.
    rng = Xoshiro(5)
    D = dimension(layout)
    for p in 1:npatches(layout)
        k = factored_volume(layout, p)
        k == 0 && continue
        for _ in 1:NPOINTS
            ξ = random_point(rng, layout, p)
            ξ⊥ = Tuple(ξ[e] for e in 1:D if e != k)
            @test radial_volume(layout, ξ[k], p) * angular_volume(layout, ξ⊥, p) ≈
                  volume_element(layout, ξ, p) rtol = 1e-14
            # One class, one angular function.
            q = findfirst(q -> angular_class(layout, q) == angular_class(layout, p),
                          1:npatches(layout))
            @test angular_volume(layout, ξ⊥, q) == angular_volume(layout, ξ⊥, p)
        end
    end
    # The factorizations each layout declares: the thirteen-patch ball's
    # outer shells and the annulus's patches, along their radius.
    if layout isa ThirteenPatchBall
        @test [factored_volume(layout, p) for p in 1:13] ==
              [0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 3, 3]
    elseif layout isa PolarAnnulus
        @test all(p -> factored_volume(layout, p) == 2, 1:4)
    else
        @test all(p -> factored_volume(layout, p) == 0, 1:7)
    end
    @test factored_volume(layout, 1) == (layout isa PolarAnnulus ? 2 : 0)
end

@testset "The logical speed is √λ_max of the inverse metric: $name" for (name, layout) in LAYOUTS
    # A closed-form eigenvalue that is wrong for a sheared frame, or that
    # loses the cube's equal eigenvalues to a division by zero.
    rng = Xoshiro(6)
    D = dimension(layout)
    for p in 1:npatches(layout), _ in 1:NPOINTS
        ξ = random_point(rng, layout, p)
        Jf = ForwardDiff.jacobian(y -> SVector(chart_map(layout, Tuple(y), p)), SVector(ξ))
        K = inv(Jf)
        λ = maximum(eigvals(Symmetric(Matrix(K * K'))))
        @test logical_speed(layout, ξ, p) ≈ sqrt(λ) rtol = 1e-12
    end
    if !(layout isa PolarAnnulus)
        n = layout.roots[1]
        @test logical_speed(layout, (0.3, 0.7, 1.1), 1) == n / (2layout.Lc)
    end
end

@testset "affine_map is the cube's map, and refused elsewhere: $name" for (name, layout) in LAYOUTS
    # An affine map that places the cube's nodes off its chart_map, which a
    # scheme working in physical coordinates in the cube would misplace.
    D = dimension(layout)
    if layout isa PolarAnnulus
        @test !any(p -> affine_patch(layout, p), 1:4)
        @test_throws "has no affine map" affine_map(layout, 1)
        return
    end
    @test affine_patch(layout, 1)
    @test !any(p -> affine_patch(layout, p), 2:npatches(layout))
    origin, scale = affine_map(layout, 1)
    for ξ in ((0.0, 0.0, 0.0), (0.5, 1.25, 2.0), (-0.5, 1.0, 2.5))
        @test chart_map(layout, ξ, 1) == origin .+ scale .* ξ
    end
    @test_throws "is a shell, whose map is not affine" affine_map(layout, 2)
end

@testset "Every shell's radial coordinate runs outward" begin
    # The orientation this package changes from TreeWaveOLDG's: a shell
    # over a low cube face whose `ξ_k` still ran inward would put the outer
    # sphere on its low face.
    for (name, layout) in LAYOUTS
        layout isa PolarAnnulus && continue
        for p in 2:npatches(layout)
            k = radial_dimension(layout, p)
            n = patch_roots(layout, p)
            ξ = ntuple(d -> d == k ? 0.5 : n[d] / 2, 3)
            J = chart_jacobian(layout, ξ, p)
            x = chart_map(layout, ξ, p)
            @test sum(J[i][k] * x[i] for i in 1:3) > 0
        end
        @test radial_dimension(layout, 1) == 0
    end
    a = PolarAnnulus(1, 2, (2, 1))
    @test all(p -> radial_dimension(a, p) == 2, 1:4)
    @test outer_radius(a) == 2
end

@testset "The constructors refuse what does not map one to one" begin
    # A layout whose maps fold over, or a typo in the roots, found when it
    # is built rather than as a wrong answer.
    @test_throws "0 < r0 < r1" PolarAnnulus(2, 1, (2, 1))
    @test_throws "positive number of roots" PolarAnnulus(1, 2, (0, 1))
    @test_throws "must exceed √3 Lc" SevenPatchBall(1, 1.7, (2, 2))
    @test_throws "must be positive" SevenPatchBall(0, 3, (2, 2))
    @test_throws "positive number of roots" SevenPatchBall(1, 3, (2, 0))
    @test_throws "must be finite" SevenPatchBall(1, 3, (2, 2); grading=Inf)
    @test_throws "the middle radius Rm" ThirteenPatchBall(1, 1.5, 3, (2, 1, 1))
    @test_throws "must exceed the middle radius" ThirteenPatchBall(1, 2, 2, (2, 1, 1))
    @test_throws "roots = (n, m, m′)" ThirteenPatchBall(1, 2, 3, (2, 1))
end
