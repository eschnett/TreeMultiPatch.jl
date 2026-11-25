# The balls reduced by a symmetry (TreeAMR's M21, step 5): their forests,
# their glue, the atlas's fold of every point of the full ball, their
# volumes, and TreeAMR's ghosts across them against the full ball's.

const SYMMETRY_NAMES = (:bitant, :quadrant, :octant, :quarter, :half, :quarter_bitant,
                        :half_bitant)
const SYMMETRIC_BALLS = [
    "seven-patch (2, 2), graded" => (; symmetry) -> SevenPatchBall(1, 2.5, (2, 2);
                                                                    grading=0.7,
                                                                    symmetry=symmetry),
    "thirteen-patch (4, 2, 3)" => (; symmetry) -> ThirteenPatchBall(0.75, 2, 4, (4, 2, 3);
                                                                     symmetry=symmetry)]

# The physical symmetry of a fold, `x = R^r F x′`: `F` the mirrors by axis,
# `R` the quarter turn about `z` taking `e_1` to `e_2`.
function unfold(x, (mirrors, r))
    y = ntuple(a -> mirrors[a] ? -x[a] : x[a], 3)
    for _ in 1:r
        y = (-y[2], y[1], y[3])
    end
    return y
end

# The share of the full ball a reduction keeps.
function kept_share(name)
    mirrors, turn = TreeMultiPatch.symmetry_parts(name)
    return 1 / (2^count(mirrors) * (turn == 1 ? 4 : turn == 2 ? 2 : 1))
end

@testset "A reduced ball's forest builds and is its layout: $name, $sym" for
        (name, make) in SYMMETRIC_BALLS, sym in SYMMETRY_NAMES
    # The patches a reduction keeps — the cube and every shell but those
    # over a cut-away face — their boxes, mirror faces and glue: a forest
    # TreeAMR refuses, or one `check_layout` finds otherwise glued, would
    # put a ghost somewhere else.
    b = make(; symmetry=sym)
    full = make(; symmetry=nothing)
    mirrors, turn = TreeMultiPatch.symmetry_parts(sym)
    cut = TreeMultiPatch.cut_axes(mirrors, turn)
    layers = full isa ThirteenPatchBall ? 2 : 1
    @test npatches(b) == 1 + layers * (6 - count(cut))
    forest = layout_forest(b; N=8)
    @test check_layout(b, forest) === nothing
    @test TreeAMR.patch_turn(forest) === (turn == 0 ? nothing : ((1, 2), turn == 2))
    for p in 1:npatches(b), a in 1:3
        o = b.offset[p][a]
        @test TreeAMR.mirror_label(forest, p, a, 1) == (mirrors[a] && o > 0 ? a : 0)
    end
end

@testset "A reduced ball's glued faces are one surface, or a turn's images: $name, $sym" for
        (name, make) in SYMMETRIC_BALLS, sym in SYMMETRY_NAMES
    # An ordinary glue joins two patches' maps on their common face; a
    # turned one joins a face to the image of the face across under its
    # turns. A turned entry derived with the wrong axes, sign or turn puts
    # the seam's ghosts somewhere else.
    b = make(; symmetry=sym)
    rng = Xoshiro(5)
    worst = 0.0
    for g in layout_glue(b), _ in 1:4
        p, d, side = g.from
        ξ = Base.setindex(random_inside(rng, b, p), side == 1 ? 0.0 : patch_roots(b, p)[d], d)
        xp = chart_map(b, ξ, p)
        xq = chart_map(b, glue_point(b, g, ξ), g.to[1])
        worst = max(worst, maxabs(xp .- unfold(xq, ((false, false, false), g.turn))))
    end
    @test worst < 1e-12
end

@testset "The atlas folds every point of the full ball: $name, $sym" for
        (name, make) in SYMMETRIC_BALLS, sym in SYMMETRY_NAMES
    # Points of every patch of the full ball, the removed shells' wedges
    # included: each is located as its image in a kept patch's box, and the
    # symmetry reported carries the image back to the point (TreeAMR's
    # contract, as Erik widened it). A wrong turn count, a mirror reported
    # on the wrong axis, or a patch's offset left out, misses.
    b = make(; symmetry=sym)
    full = make(; symmetry=nothing)
    rng = Xoshiro(9)
    worst, outside = 0.0, 0.0
    for P in 1:npatches(full), _ in 1:20
        x = chart_map(full, random_inside(rng, full, P), P)
        ξ, p, s = chart_locate(b, x)
        n = patch_roots(b, p)
        outside = max(outside, maximum(d -> max(-ξ[d], ξ[d] - n[d]), 1:3))
        worst = max(worst, maxabs(x .- unfold(chart_map(b, ξ, p), s)))
    end
    @test outside < 1e-12
    @test worst < 1e-12
end

@testset "√g integrates to a reduced ball's share of the volume: $name, $sym" for
        (name, make) in SYMMETRIC_BALLS, sym in SYMMETRY_NAMES
    # The reduced patches' boxes are the kept part of the full ball, no more
    # and no less: an offset off by a root, or a shell kept that the cut
    # removes, changes the volume.
    b = make(; symmetry=sym)
    vol = sum(p -> patch_volume(b, p, 12), 1:npatches(b))
    @test vol ≈ kept_share(sym) * 4π * b.full.R1^3 / 3 rtol = 1e-11
end

@testset "Interpolated ghosts on a reduced ball are the full ball's: $sym" for
        sym in SYMMETRY_NAMES
    # TreeAMR's whole M21 on a real layout: a vertex set interpolated across
    # the seven-patch ball's kinked glues, over the reduced ball with its
    # mirror faces and turned seams, filled from data with every symmetry,
    # against the full ball's fill at the same points. The edge ghosts
    # folded through the atlas, the turned copies, the corners at the turn's
    # axis and the half turn's slab must all give the full ball's values, to
    # rounding.
    full = SevenPatchBall(1.0, 3.0, (2, 1))
    b = SevenPatchBall(1.0, 3.0, (2, 1); symmetry=sym)
    mirrors, turn = TreeMultiPatch.symmetry_parts(sym)
    s(x) = cos(x[1]) + cos(x[2]) + cos(1.3x[3]) + x[1]^2 * x[2]^2
    ops = Operators(prolongation=4, restriction=4, glue=4)
    function filled(layout, kw)
        forest = layout_forest(layout; N=8)
        fs = FieldSet(forest, 1; G=2, centering=vertexcentered(3), glue=:interpolate, kw...)
        f = WithPatch((ξ, p, v) -> s(chart_map(layout, ξ, p)))
        fill_by_coordinates!(f, fs)
        fill_ghosts!(fs, GhostSchedule(fs, ops; atlas=layout);
                     boundary=CellBoundary(WithPatch((ξ, p, v, δ) ->
                                                         s(chart_map(layout, ξ, p)))))
        return fs
    end
    kw = (any(mirrors) ? (parity=[(EvenParity, EvenParity, EvenParity)],) : (;))
    turn == 1 && (kw = merge(kw, (rotation=(1,),)))
    turn == 2 && (kw = merge(kw, (halfturn=(1,),)))
    fr, ff = filled(b, kw), filled(full, (;))
    key(fs, layout, b′, shift) = (k = blockkey(fs, b′);
                                  (shift(patch(fs.forest, k), block_origin(fs.forest, k)),
                                   level(k)))
    index = Dict(key(ff, full, j, (P, o) -> (P, o)) => j for j in 1:nblocks(ff))
    worst = 0.0
    for j in 1:nblocks(fr)
        (P, o), l = key(fr, b, j, (p, o) -> (Int(b.patch[p]), o .+ b.offset[p]))
        jf = index[((P, o), l)]
        worst = max(worst, maximum(abs.(fr.work[:, :, :, :, j] .- ff.work[:, :, :, :, jf])))
    end
    @test !any(isnan, fr.work)
    @test worst < 1e-12
end

@testset "A reduced ball runs in a kernel: $name, $sym" for (name, make) in SYMMETRIC_BALLS,
                                                           sym in SYMMETRY_NAMES
    # Plain bits, and every method at a point inferred to one concrete type,
    # as every layout's are (CODE.md, "Code structure"): a reduced ball is a
    # device kernel's argument too.
    b = make(; symmetry=sym)
    @test isbits(b)
    ξ = (0.3, 0.4, 0.2)
    x = (-0.7, 0.4, -1.1)
    @test (@inferred chart_map(b, ξ, 2)) isa NTuple{3,Float64}
    @test (@inferred chart_jacobian(b, ξ, 2)) isa NTuple{3,NTuple{3,Float64}}
    @test (@inferred Nothing chart_locate(b, x)) !== nothing
end
