# The atlas: chart_locate against chart_map inside the patches, on their
# glued faces and corners, past their boxes and outside the domain; and
# the alignment each layout declares.

normalize_tuple(u) = u ./ sqrt(sum(abs2, u))

# Whether `ξ` lies in patch `q`'s closed box, as TreeAMR's contract asks.
inbox(layout, ξ, q) = all(d -> 0 <= ξ[d] <= patch_roots(layout, q)[d], eachindex(ξ))

# The distance of `ξ` from patch `q`'s box boundary, in roots.
boundary_distance(layout, ξ, q) =
    minimum(d -> min(ξ[d], patch_roots(layout, q)[d] - ξ[d]), eachindex(ξ))

@testset "chart_locate inverts chart_map at random points: $name" for (name, layout) in LAYOUTS
    # An inverse that picks the wrong patch, or a wrong closed form for a
    # coordinate, away from any face.
    rng = Xoshiro(11)
    for p in 1:npatches(layout), _ in 1:100
        ξ = random_inside(rng, layout, p)
        loc = chart_locate(layout, chart_map(layout, ξ, p))
        @test loc !== nothing
        ξ′, q = loc
        @test q == p
        @test maximum(abs.(ξ′ .- ξ)) <= 1e-13 * maximum(patch_roots(layout, p))
    end
end

@testset "chart_locate inverts chart_map on glued faces and at patch corners: $name" for (name, layout) in LAYOUTS
    # A point on a face, edge or corner, where two to four patches hold the
    # point and the inverse may name any of them, but must name one whose
    # closed box holds the `ξ` it returns, and the same physical point.
    rng = Xoshiro(12)
    D = dimension(layout)
    for p in 1:npatches(layout), _ in 1:200
        n = patch_roots(layout, p)
        ξ = random_inside(rng, layout, p)
        for d in randperm(rng, D)[1:rand(rng, 1:D)]
            ξ = Base.setindex(ξ, rand(rng, (0.0, Float64(n[d]))), d)
        end
        x = chart_map(layout, ξ, p)
        loc = chart_locate(layout, x)
        @test loc !== nothing
        ξ′, q = loc
        @test inbox(layout, ξ′, q)
        @test maximum(abs.(chart_map(layout, ξ′, q) .- x)) <= 1e-13 * max(1, maxabs(x))
        # Another patch holds it only on its own boundary.
        q == p || @test boundary_distance(layout, ξ′, q) <= 1e-12
    end
end

@testset "chart_locate inverts chart_map past the box, where the ghosts are: $name" for (name, layout) in LAYOUTS
    # A ghost across a glue is located from the target's map past its
    # box: the inverse must name a patch whose closed box holds the point,
    # or none beyond the domain, and never a `ξ` outside the box.
    rng = Xoshiro(13)
    found, outside = 0, 0
    for p in 1:npatches(layout), _ in 1:200
        ξ = random_point(rng, layout, p)
        x = chart_map(layout, ξ, p)
        loc = chart_locate(layout, x)
        if loc === nothing
            outside += 1
            continue
        end
        found += 1
        ξ′, q = loc
        @test inbox(layout, ξ′, q)
        @test maximum(abs.(chart_map(layout, ξ′, q) .- x)) <= 1e-13 * max(1, maxabs(x))
    end
    @test found > 0
    @test outside > 0
end

@testset "A point just past a glued face is in the patch the glue names: $name" for (name, layout) in LAYOUTS
    # The atlas and the glue disagreeing about which patch lies across a
    # face, which would send a ghost to the wrong neighbor.
    rng = Xoshiro(14)
    D = dimension(layout)
    for g in layout_glue(layout), _ in 1:10
        p, d, side = g.from
        n = patch_roots(layout, p)
        ξ = ntuple(e -> e == d ? (side == 1 ? -0.05 : n[d] + 0.05) :
                        0.1 + (n[e] - 0.2) * rand(rng), D)
        loc = chart_locate(layout, chart_map(layout, ξ, p))
        @test loc !== nothing
        @test loc[2] == g.to[1]
    end
end

@testset "A point outside the domain is in no patch: $name" for (name, layout) in LAYOUTS
    # An inverse that clamps a point far outside onto the boundary, which
    # would give a ghost beyond the outer face a value instead of the hook.
    rng = Xoshiro(15)
    D = dimension(layout)
    for _ in 1:50
        u = normalize_tuple(ntuple(_ -> randn(rng), D))
        if layout isa PolarAnnulus
            @test chart_locate(layout, layout.r1 * (1 + 1e-9) .* u) === nothing
            @test chart_locate(layout, layout.r0 * (1 - 1e-9) .* u) === nothing
            @test chart_locate(layout, layout.r1 * (1 - 1e-9) .* u) !== nothing
            @test chart_locate(layout, layout.r0 * (1 + 1e-9) .* u) !== nothing
        else
            @test chart_locate(layout, layout.R1 * (1 + 1e-9) .* u) === nothing
            @test chart_locate(layout, layout.R1 * (1 - 1e-9) .* u) !== nothing
        end
    end
    layout isa PolarAnnulus && @test chart_locate(layout, (0.0, 0.0)) === nothing
    layout isa PolarAnnulus || @test chart_locate(layout, (0.0, 0.0, 0.0)) == ((n = layout.roots[1] / 2; (n, n, n)), 1)
end

# Every interpolated ghost of patch `p` at spacing `1/N` and stagger `c`
# (0 for a vertex-centered grid, ½ for a cell-centered one) within `w`
# spacings past `p`'s box that `chart_locate` places in another patch:
# `(ξ, q, ξ′)` per ghost.
function ghost_points(layout, p, N, c, w)
    n = patch_roots(layout, p)
    D = length(n)
    ghosts = Tuple{NTuple{D,Float64},Int,NTuple{D,Float64}}[]
    for I in CartesianIndices(ntuple(d -> -w:(n[d] * N + w - 1), D))
        i = Tuple(I)
        ξ = ntuple(d -> (i[d] + c) / N, D)
        all(d -> 0 <= ξ[d] <= n[d], 1:D) && continue
        loc = chart_locate(layout, chart_map(layout, ξ, p))
        loc === nothing && continue
        ξ′, q = loc
        q == p && continue
        push!(ghosts, (ξ, q, ξ′))
    end
    return ghosts
end

# How far `t` is from a node of the grid of spacing `1/N` and stagger `c`,
# in spacings.
offnode(t, N, c) = (u = t * N - c; abs(u - round(u)))

@testset "Every declared alignment holds at the ghosts past each box: $name" for (name, layout) in LAYOUTS
    # A declaration the maps do not have, which TreeAMR refuses when it
    # builds the schedule, or which, rounded to a node, would put a ghost
    # where the map does not; and a declaration missing where TreeAMR's
    # interpolation could have width one.
    D = dimension(layout)
    pairs = Dict{Tuple{Int,Int},Int}()
    worst = 0.0
    offnodes = 0
    shapes = true
    # Three spacings past the box at `N = 6` and 8 points per root, as a
    # fourth-order stencil's ghosts are; the annulus's map past its inner
    # wall holds while `r > 0`, which a coarser grid's ghosts would cross.
    for p in 1:npatches(layout), N in (6, 8), c in (0.0, 0.5)
        for (ξ, q, ξ′) in ghost_points(layout, p, N, c, 3)
            al = chart_aligned(layout, p, q)
            shapes &= al isa NTuple{D,Bool}
            for e in 1:D
                δ = offnode(ξ′[e], N, c)
                if al[e]
                    worst = max(worst, δ)
                    pairs[(p, q)] = get(pairs, (p, q), 0) + 1
                else
                    offnodes += δ > 1e-3
                end
            end
        end
    end
    @test shapes
    # To rounding, far below the thousandth of a spacing TreeAMR checks.
    @test worst <= 1e-12
    # The test discriminates: an undeclared coordinate is off the nodes
    # somewhere, by more than TreeAMR's tolerance.
    @test offnodes > 0 || layout isa PolarAnnulus
    if layout isa PolarAnnulus
        @test all(p -> haskey(pairs, (p, mod1(p + 1, 4))) && haskey(pairs, (mod1(p + 1, 4), p)),
                  1:4)
        @test offnodes == 0
    elseif layout isa ThirteenPatchBall
        # Both kinds of declared pair are reached by ghosts, on every face.
        for (k, s) in TreeMultiPatch.SHELLS
            i = TreeMultiPatch.shell_index(k, s)
            @test haskey(pairs, (i, i + 6)) && haskey(pairs, (i + 6, i))
            @test count(q -> haskey(pairs, (i + 6, q)), 8:13) == 4
        end
    else
        @test isempty(pairs)
    end
end

@testset "The declarations are the ones the plan names" begin
    # A declaration that drifts from what TreeAMR's M20 plan and its tests
    # expect of the thirteen-patch ball.
    b = ThirteenPatchBall(1, 2, 3, (2, 1, 1))
    @test chart_aligned(b, 8, 10) == (false, true, false)       # radius of patch 10
    @test chart_aligned(b, 10, 8) == (true, false, false)
    @test chart_aligned(b, 2, 8) == (false, true, true)         # both angles, face (1, -1)
    @test chart_aligned(b, 8, 2) == (false, true, true)
    @test chart_aligned(b, 4, 10) == (true, false, true)
    @test chart_aligned(b, 2, 10) == (false, false, false)      # different faces
    @test chart_aligned(b, 1, 2) == (false, false, false)       # cube to shell
    @test chart_aligned(b, 2, 4) == (false, false, false)       # two inner shells
    @test chart_aligned(b, 5, 5) == (true, true, true)
    s = SevenPatchBall(1, 3, (2, 2))
    @test chart_aligned(s, 2, 4) == (false, false, false)
    @test chart_aligned(PolarAnnulus(1, 2, (2, 1)), 1, 2) == (true, true)
end

@testset "chart_aligned is TreeAMR's own function" begin
    # A declaration that TreeAMR's glue schedule would not see, because it
    # is a method of another function than the one TreeAMR calls.
    @test chart_aligned === TreeAMR.chart_aligned
    @test hasmethod(chart_aligned, Tuple{ThirteenPatchBall{Float64},Int,Int})
end
