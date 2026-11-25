# The balls reduced by a symmetry (TreeAMR's M21): a half, a quarter or an
# eighth of a seven- or thirteen-patch ball, under mirror planes through
# the coordinate axes and turns about `z`. A reduced ball is sub-boxes of
# the full ball's patches, cut at their mid-planes, with the full maps:
# each reduced patch names a patch of the full ball and the offset of its
# box there, so its own box starts at 0 as every layout's does, and every
# cut is a low face. A cut is a mirror face, labelled by the axis its plane
# is normal to, or a turn's seam, glued by a turned entry. See "Symmetric
# balls" in CODE.md.

# The symmetries by name: the mirrors, by axis, and the turn about `z`,
# 0 for none, 1 for a quarter turn, 2 for a half turn.
function symmetry_parts(name::Symbol)
    name === :bitant && return (false, false, true), 0
    name === :quadrant && return (true, true, false), 0
    name === :octant && return (true, true, true), 0
    name === :quarter && return (false, false, false), 1
    name === :half && return (false, false, false), 2
    name === :quarter_bitant && return (false, false, true), 1
    name === :half_bitant && return (false, false, true), 2
    throw(ArgumentError(
        "symmetry = :$name is none of :bitant, :quadrant, :octant (mirror planes " *
        "through the axes, z ≥ 0, x, y ≥ 0 and all three), :quarter and :half (turns " *
        "about z by 90° and 180°), :quarter_bitant and :half_bitant (the turns with " *
        "the mirror z ≥ 0)"))
end

# The axes whose positive side a reduction keeps: the mirrors', and the
# quarter turn's x and y, or the half turn's x.
cut_axes(mirrors, turn) =
    ntuple(a -> mirrors[a] || (turn == 1 && a <= 2) || (turn == 2 && a == 1), Val(3))

"""
    SymmetricBall

A seven- or thirteen-patch ball reduced by a symmetry, as
`SevenPatchBall(…; symmetry)` and `ThirteenPatchBall(…; symmetry)` build
it (TreeAMR's M21): the part of the ball on the positive side of each
mirror plane and of each turn's cut, as sub-boxes of the full ball's
patches with the full maps. A patch of the full ball that the reduction
leaves nothing of — a shell over a cut-away face — is dropped, and the
others are numbered in the full ball's order. `layout_forest` gives the
reduced patches their mirror faces, label `a` on the cut normal to `x_a`,
and glues each turn's seam by turned entries, declaring the turn's plane
`(1, 2)`; `chart_locate` folds any point of the full ball into the
reduced one and reports the symmetry that carries it back. The cube's
roots `n` must be even, so that its mid-planes are root boundaries.
"""
struct SymmetricBall{T,L<:PatchLayout{3,T},NP} <: PatchLayout{3,T}
    full::L
    mirrors::NTuple{3,Bool}
    turn::Int8
    patch::NTuple{NP,Int8}
    offset::NTuple{NP,NTuple{3,Int}}
end

function symmetric(full::PatchLayout{3,T}, name::Symbol) where {T}
    mirrors, turn = symmetry_parts(name)
    cut = cut_axes(mirrors, turn)
    n = full.roots[1]
    iseven(n) || throw(ArgumentError(
        "symmetry = :$name cuts the cube at its mid-planes, which are root boundaries " *
        "only for an even number of roots across it, got n = $n"))
    patches, offsets = Int8[], NTuple{3,Int}[]
    for P in 1:npatches(full)
        k, s, _ = ball_shell(full, P)
        k != 0 && cut[k] && s == -1 && continue
        push!(patches, Int8(P))
        push!(offsets, ntuple(a -> cut[a] && a != k ? n ÷ 2 : 0, Val(3)))
    end
    return SymmetricBall{T,typeof(full),length(patches)}(full, mirrors, Int8(turn),
                                                          Tuple(patches), Tuple(offsets))
end

npatches(::SymmetricBall{T,L,NP}) where {T,L,NP} = NP
function layout_name(b::SymmetricBall)
    names = Dict(((false, false, true), 0) => "bitant", ((true, true, false), 0) => "quadrant",
                 ((true, true, true), 0) => "octant", ((false, false, false), 1) => "quarter",
                 ((false, false, false), 2) => "half", ((false, false, true), 1) => "quarter_bitant",
                 ((false, false, true), 2) => "half_bitant")
    return "the $(names[(b.mirrors, Int(b.turn))]) of $(layout_name(b.full))"
end
outer_radius(b::SymmetricBall) = outer_radius(b.full)

# The full patch of reduced patch `p`, and `ξ` in the full patch's
# coordinates.
@inline full_patch(b::SymmetricBall, p) = Int(tuple_pick(b.patch, p))
@inline full_point(b::SymmetricBall, ξ, p) =
    (o = tuple_pick(b.offset, p); ntuple(a -> ξ[a] + o[a], Val(3)))

# Entry `i` of a tuple by static recursion, which a device compiles to
# selects rather than a load from a run-time index.
@inline tuple_pick(t::Tuple{Any}, i) = t[1]
@inline tuple_pick(t::Tuple, i) = i == 1 ? t[1] : tuple_pick(Base.tail(t), i - 1)

function patch_roots(b::SymmetricBall, p)
    full = patch_roots(b.full, full_patch(b, p))
    o = b.offset[p]
    return ntuple(a -> o[a] > 0 ? full[a] - o[a] : full[a], Val(3))
end
radial_dimension(b::SymmetricBall, p) = radial_dimension(b.full, full_patch(b, p))
affine_patch(b::SymmetricBall, p) = affine_patch(b.full, full_patch(b, p))
function affine_map(b::SymmetricBall, p)
    origin, scale = affine_map(b.full, full_patch(b, p))
    o = b.offset[p]
    return ntuple(a -> origin[a] + scale * o[a], Val(3)), scale
end
factored_volume(b::SymmetricBall, p) = factored_volume(b.full, full_patch(b, p))
radial_volume(b::SymmetricBall, ξk, p) =
    (P = full_patch(b, p); radial_volume(b.full, ξk + b.offset[p][radial_dimension(b, p)], P))
function angular_volume(b::SymmetricBall, ξ⊥, p)
    P = full_patch(b, p)
    k = radial_dimension(b, p)
    o = b.offset[p]
    across = ntuple(i -> i < k ? i : i + 1, Val(2))
    return angular_volume(b.full, ntuple(i -> ξ⊥[i] + o[across[i]], Val(2)), P)
end
# Reduced patches share an angular factor where their full patches do and
# their boxes start alike.
function angular_class(b::SymmetricBall, p)
    c = angular_class(b.full, full_patch(b, p))
    return findfirst(q -> angular_class(b.full, full_patch(b, q)) == c &&
                          b.offset[q] == b.offset[p], 1:npatches(b))
end

@inline TreeAMR.chart_map(b::SymmetricBall, ξ, p) =
    TreeAMR.chart_map(b.full, full_point(b, ξ, p), full_patch(b, p))
@inline TreeAMR.chart_jacobian(b::SymmetricBall, ξ, p) =
    TreeAMR.chart_jacobian(b.full, full_point(b, ξ, p), full_patch(b, p))
@inline TreeAMR.chart_hessian(b::SymmetricBall, ξ, p) =
    TreeAMR.chart_hessian(b.full, full_point(b, ξ, p), full_patch(b, p))
chart_aligned(b::SymmetricBall, p, q) =
    chart_aligned(b.full, full_patch(b, p), full_patch(b, q))

# The image of `x` in the reduced ball and the symmetry that carries it
# back: the mirrors, by axis, of the negative coordinates they keep
# positive, then the turns about `z` that bring the point into the turn's
# cut, `x = R^r F x′` with `R` taking `e_1` to `e_2` and `e_2` to `-e_1`.
@inline function fold_physical(b::SymmetricBall, x)
    m = ntuple(a -> b.mirrors[a] && x[a] < 0, Val(3))
    y = ntuple(a -> m[a] ? -x[a] : x[a], Val(3))
    r = 0
    if b.turn == 1
        r = y[1] >= 0 ? (y[2] >= 0 ? 0 : 3) : (y[2] >= 0 ? 1 : 2)
    elseif b.turn == 2
        r = y[1] >= 0 ? 0 : 2
    end
    # `R^{-r}`: one quarter turn back is `(x, y) ↦ (y, -x)`.
    z = r == 0 ? y : r == 1 ? (y[2], -y[1], y[3]) : r == 2 ? (-y[1], -y[2], y[3]) :
        (-y[2], y[1], y[3])
    return z, m, r
end

# Any point of the full ball, folded into the reduced one: its image's
# position and reduced patch, and the symmetry (TreeAMR's contract, M21).
# The image lies in a patch the reduction keeps, on the cuts' positive
# side, so its position is in that patch's box but for rounding.
@inline function TreeAMR.chart_locate(b::SymmetricBall, x)
    z, m, r = fold_physical(b, x)
    loc = TreeAMR.chart_locate(b.full, z)
    loc === nothing && return nothing
    ξ, P = loc
    p = reduced_patch(b, P)
    o = tuple_pick(b.offset, p)
    return ntuple(a -> ξ[a] - o[a], Val(3)), p, (m, r)
end

@inline function reduced_patch(b::SymmetricBall{T,L,NP}, P) where {T,L,NP}
    for p in 1:NP
        Int(tuple_pick(b.patch, p)) == P && return p
    end
    return 1
end

# The patches with their mirror faces: label `a` on the low face of every
# axis `a` the patch is cut along by a mirror.
function layout_patches(b::SymmetricBall)
    return [(roots=patch_roots(b, p),
             mirror=ntuple(a -> (b.mirrors[a] && b.offset[p][a] > 0 ? a : 0, 0), Val(3)))
            for p in 1:npatches(b)]
end

# The turn's plane, for `layout_forest`.
forest_keywords(b::SymmetricBall) =
    b.turn == 1 ? (rotating=(1, 2),) : b.turn == 2 ? (halfturn=(1, 2),) : (;)

# The glue: the full ball's between the kept faces, and each turn's seam
# by a turned entry, derived from the maps: a point just past the cut is
# folded into the reduced ball, which names the face across and the turn,
# and steps along each axis name the axes across.
function layout_glue(b::SymmetricBall)
    full = Dict(g.from => g for g in layout_glue(b.full))
    index = Dict(Int(b.patch[p]) => p for p in 1:npatches(b))
    glue = []
    for p in 1:npatches(b), a in 1:3, side in 1:2
        o = b.offset[p]
        if side == 1 && o[a] > 0
            b.mirrors[a] && continue
            push!(glue, turned_entry(b, p, a))
            continue
        end
        g = get(full, (full_patch(b, p), a, side), nothing)
        g === nothing && continue
        q = index[g.to[1]]
        push!(glue, (from=(p, a, side), to=(q, g.to[2], g.to[3]), map=g.map, turn=0))
    end
    return [NamedTuple{(:from, :to, :map, :turn),
                       Tuple{NTuple{3,Int},NTuple{3,Int},NTuple{3,Int},Int}}(Tuple(g))
            for g in glue]
end

function turned_entry(b::SymmetricBall, p, a)
    n = patch_roots(b, p)
    δ, ε = 1 / 8, 1 / 16
    ξ0 = ntuple(e -> e == a ? -δ : n[e] / 2, Val(3))
    image(ξ) = TreeAMR.chart_locate(b, TreeAMR.chart_map(b, ξ, p))
    ξq, q, (_, r) = image(ξ0)
    nq = patch_roots(b, q)
    bq = argmin(e -> b.offset[q][e] > 0 ? abs(ξq[e] - δ) : Inf, 1:3)
    σ = ntuple(Val(3)) do e
        ξe = ntuple(i -> ξ0[i] + (i == e ? ε : 0), Val(3))
        ξ′, q′, _ = image(ξe)
        q′ == q || error("a step from the seam of patch $p leaves patch $q; this is a bug")
        Δ = ξ′ .- ξq
        c = argmax(i -> abs(Δ[i]), 1:3)
        return Δ[c] > 0 ? c : -c
    end
    return (from=(p, a, 1), to=(q, bq, 1), map=σ, turn=r)
end
