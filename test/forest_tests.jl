# The forests: each layout's builds, and its check accepts it and refuses
# others; the glue joins the maps; the outer faces are where the layout
# says.

@testset "Every layout's forest builds and passes its check: $name" for (name, layout) in LAYOUTS
    # A glue TreeAMR refuses, or a check that refuses the forest its own
    # builder makes, uniform or refined.
    for N in (4, 8)
        forest = layout_forest(layout; N=N)
        @test forest isa Forest{dimension(layout),Float64}
        @test nleaves(forest) == sum(p -> prod(patch_roots(layout, p)), 1:npatches(layout))
        @test check_layout(layout, forest) === nothing
    end
    @test layout_forest(Float32, layout; N=4) isa Forest{dimension(layout),Float32}
    # Refined: leaves at two levels, so the check reads the glue below the
    # root, where a turn of the axes within a root shows.
    forest = layout_forest(layout; N=4)
    refine!(forest, forest.leaves[1:3:end])
    balance!(forest)
    @test maxlevel(forest) == 1
    @test check_layout(layout, forest) === nothing
end

@testset "layout_forest passes the forest's keywords, and refuses the glue" begin
    # A builder that drops a keyword, or lets the caller replace the glue
    # its maps are written for.
    a = PolarAnnulus(1, 2, (2, 1))
    walls = ((false, false), (true, true))
    forest = layout_forest(a; N=4, reflecting=walls)
    @test forest.reflecting == walls
    @test check_layout(a, forest) === nothing
    @test_throws "takes the glue from the layout" layout_forest(a; N=4, glue=[])
    @test_throws "pass no extents" layout_forest(a; N=4, extents=((0, 1), (0, 1)))
    @test_throws UndefKeywordError layout_forest(a)
    # A ball's outer sphere cannot reflect: its cube is glued on every face.
    @test_throws ArgumentError layout_forest(SevenPatchBall(1, 3, (2, 2)); N=4,
                                             reflecting=((true, true), (false, false),
                                                         (false, false)))
end

# The seven-patch ball's glue as TreeWaveOLDG has it, with the shells over
# the cube's low faces running inward: a forest of the same patches that is
# not this package's layout.
function inward_sevenpatch_glue()
    side(s) = s == 1 ? 2 : 1
    index = TreeMultiPatch.shell_index
    glue = TreeMultiPatch.glue_list(Val(3))
    for (k, s) in TreeMultiPatch.SHELLS
        p = index(k, s)
        push!(glue, (from=(1, k, side(s)), to=(p, k, side(-s)), map=(1, 2, 3)))
        push!(glue, (from=(p, k, side(-s)), to=(1, k, side(s)), map=(1, 2, 3)))
        for (l, t) in TreeMultiPatch.SHELLS
            l == k && continue
            σ = ntuple(e -> e == k ? (s == t ? l : -l) : e == l ? (s == t ? -k : k) : e, 3)
            push!(glue, (from=(p, l, side(t)), to=(index(l, t), k, side(s)), map=σ))
        end
    end
    return glue
end

@testset "check_layout refuses a forest that is not the layout's" begin
    # A forest built for another layout, other roots or another glue,
    # which the maps would misplace without a word.
    a = PolarAnnulus(1, 2, (2, 1))
    b = SevenPatchBall(1, 3, (2, 2))
    c = ThirteenPatchBall(1, 2, 3, (2, 1, 1))
    @test_throws "is 3-dimensional, and the forest has D = 2" check_layout(
        b, layout_forest(a; N=4))
    @test_throws "has 13 patches" check_layout(c, layout_forest(b; N=4))
    @test_throws "has 4 patches" check_layout(a, Forest((8, 1); N=4))
    @test_throws "has the roots" check_layout(SevenPatchBall(1, 3, (2, 1)),
                                              layout_forest(b; N=4))
    @test_throws "maps the box of roots (1, 2)" check_layout(PolarAnnulus(1, 2, (1, 2)),
                                              layout_forest(PolarAnnulus(1, 2, (2, 1)); N=4))
    # The old orientation: the same patches, glued as TreeWaveOLDG glues
    # them, at the root level and refined.
    old = Forest([(roots=patch_roots(b, p),) for p in 1:7]; N=4,
                 glue=inward_sevenpatch_glue())
    @test_throws "the forest's glue is not the layout's" check_layout(b, old)
    refine!(old, old.leaves[1:2:end])
    balance!(old)
    @test_throws "the forest's glue is not the layout's" check_layout(b, old)
    # A ring cut open: a glued face the forest leaves unglued.
    glue = layout_glue(a)
    cut = Forest([(roots=(2, 1),) for _ in 1:4]; N=4,
                 glue=filter(g -> !(g.from[1] in (1, 2) && g.to[1] in (1, 2)), glue))
    @test_throws "and the forest leaves it unglued" check_layout(a, cut)
end

@testset "A glued face is one surface under both patches' maps: $name" for (name, layout) in LAYOUTS
    # A glue whose axis map turns the wrong way, which TreeAMR accepts as
    # long as it is a signed permutation, and which would copy or
    # interpolate a ghost from the wrong place.
    rng = Xoshiro(21)
    D = dimension(layout)
    for g in layout_glue(layout), _ in 1:20
        p, d, side = g.from
        n = patch_roots(layout, p)
        ξ = ntuple(e -> e == d ? (side == 1 ? 0.0 : Float64(n[d])) : n[e] * rand(rng), D)
        ξ′ = glue_point(layout, g, ξ)
        x = chart_map(layout, ξ, p)
        @test maximum(abs.(chart_map(layout, ξ′, g.to[1]) .- x)) <= 1e-14 * max(1, maxabs(x))
        # And a step out of `p` is a step into `q`.
        ξout = Base.setindex(ξ, ξ[d] + (side == 1 ? -0.25 : 0.25), d)
        ξin = glue_point(layout, g, ξout)
        @test 0 < ξin[g.to[2]] < patch_roots(layout, g.to[1])[g.to[2]]
    end
    # Both directions of every gluing, each the other's inverse.
    glue = Dict(g.from => g for g in layout_glue(layout))
    @test all(g -> haskey(glue, g.to) && glue[g.to].to == g.from, values(glue))
end

@testset "The outer boundary is the high face of every outermost shell: $name" for (name, layout) in LAYOUTS
    # The orientation the boundary hook relies on: on the balls the outer
    # sphere is the high face along `k` of every outermost shell, and no
    # other face is unglued.
    D = dimension(layout)
    glued = Set(g.from for g in layout_glue(layout))
    open = [(p, d, side) for p in 1:npatches(layout), d in 1:D, side in 1:2
            if !((p, d, side) in glued)]
    if layout isa PolarAnnulus
        @test sort(open) == sort([(p, 2, side) for p in 1:4 for side in 1:2])
    else
        outermost = layout isa ThirteenPatchBall ? (8:13) : (2:7)
        @test sort(open) == sort([(p, radial_dimension(layout, p), 2) for p in outermost])
        rng = Xoshiro(22)
        for (p, k, _) in open, _ in 1:10
            n = patch_roots(layout, p)
            ξ = ntuple(e -> e == k ? Float64(n[k]) : n[e] * rand(rng), 3)
            @test sqrt(sum(abs2, chart_map(layout, ξ, p))) ≈ outer_radius(layout) rtol = 1e-15
        end
    end
end
