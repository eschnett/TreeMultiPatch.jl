# What every layout shares: the interface, the forest it builds and the
# check that a forest is it, the frame formed from the Jacobian, the
# logical speed of light, and the patch properties' defaults.
#
# A layout is a set of patches, logical boxes `[0, n_d]` of `n` roots
# each, the glue between their faces in TreeAMR's form, and the map of
# each box to physical space. The layout is its own atlas: TreeAMR's
# `chart_*` functions are methods on it. Each layout writes out
# `chart_map`, `chart_jacobian`, `chart_hessian` and `chart_locate`; the
# frame (`√g` and the area vectors) and the logical speed follow from the
# Jacobian here. See "Layouts" and "The maps" in CODE.md.

if isdefined(TreeAMR, :chart_aligned)
    # A TreeAMR that has the declaration (M20): the methods below extend
    # its function, which its glue schedule calls.
    import TreeAMR: chart_aligned
else
    function chart_aligned end
end

@doc """
    chart_aligned(atlas, p, q) -> NTuple{D,Bool}

Along which dimensions of patch `q` a point at a node of patch `p`'s grid
lands exactly on a node of `q`'s grid at the same level: there the logical
coordinate `ξ′_e` that `chart_locate` returns in `q` equals `p`'s own
`ξ_e`, or its image under a whole number of roots, for every point of `p`'s
map, past `p`'s box too. A field set interpolated across a glue then reads
no neighbor along such a dimension, since the ghost's coordinate there is a
node's. `p == q` is aligned along every dimension.

Declared by each layout:

- [`ThirteenPatchBall`](@ref): the radius between two outer shells, and
  both angles between an inner and an outer shell over the same cube face;
- [`PolarAnnulus`](@ref): both dimensions for every pair;
- [`SevenPatchBall`](@ref): none between different patches.

This is TreeAMR's contract for `TreeAMR.chart_aligned` (M20, "Finite
differences across a glue" in TreeAMR's `CODE.md`). Where the TreeAMR in
use has that function, this is it, and these are its methods; with a
TreeAMR that does not, it is this package's own function of the same
contract.
""" chart_aligned

"""
    PatchLayout{D,T}

A multi-patch layout in `D` dimensions with parameters of type `T`: its
patches, the glue between their faces, and the map of each patch's
logical box `[0, n₁] × … × [0, n_D]` to physical space, `n` the patch's
roots. A layout is `isbits` for an `isbits` `T`, so that it is an argument
of a kernel on any backend, and it is TreeAMR's atlas for a forest it
built ([`layout_forest`](@ref)). Each layout provides

- `TreeAMR.chart_map(layout, ξ, p)`, the physical position of logical `ξ`
  in patch `p`, and `TreeAMR.chart_jacobian`, `TreeAMR.chart_hessian`
  and `TreeAMR.chart_locate` with TreeAMR's contracts;
- [`chart_aligned`](@ref);
- [`layout_patches`](@ref) and [`layout_glue`](@ref);
- the patch properties [`affine_patch`](@ref), [`affine_map`](@ref),
  [`factored_volume`](@ref), [`radial_dimension`](@ref) and
  [`outer_radius`](@ref).

The maps are generic in the number type: they evaluate in the promotion
of `T` and the type of `ξ`, so a `Float64` layout with exactly
representable parameters gives the same map to `Float64x2` or `BigFloat`
accuracy at a `Float64x2` or `BigFloat` `ξ`, and ForwardDiff
differentiates them. They are defined past each patch's box, by a few
spacings at least, since a ghost across a glue sits at the point the
target's own map names there. From the Jacobian follow
[`patch_frame`](@ref), [`volume_element`](@ref), [`area_vectors`](@ref)
and [`logical_speed`](@ref).

[`PolarAnnulus`](@ref), [`SevenPatchBall`](@ref) and
[`ThirteenPatchBall`](@ref) are the ones implemented.
"""
abstract type PatchLayout{D,T} end

# The number type a map evaluates in at `ξ`, and `ξ` as a tuple of it.
@inline evaltype(::PatchLayout{D,T}, ξ) where {D,T} =
    promote_type(T, ntuple(d -> typeof(ξ[d]), Val(D))...)
@inline function evalpoint(layout::PatchLayout{D}, ξ) where {D}
    R = evaltype(layout, ξ)
    return ntuple(d -> convert(R, ξ[d]), Val(D)), R
end

# `sin` and `cos` of an angle in about `[-π/4, π/2 + π/4]`, and `atan(y,
# x)` for `x, y ≥ 0`: Base's, which MultiFloats does not have; the
# extensions give its numbers, and ForwardDiff's duals over them, their
# own. `sin` and `cos` apart, not `sincos`, whose Float32 method reduces
# its argument in Float64, which a Metal kernel does not have.
@inline sincos_local(x) = (sin(x), cos(x))
@inline atan_local(y, x) = atan(y, x)

"""
    layout_patches(layout) -> Vector

The patches of `layout` as TreeAMR's `Forest` takes them, one
`(roots = (n₁, …, n_D),)` each; every patch's logical box is the default,
`[0, n_d]` along each dimension, so that a root is the unit cube in every
patch.
"""
layout_patches(layout::PatchLayout) =
    [(roots = patch_roots(layout, p),) for p in 1:npatches(layout)]

"""
    layout_glue(layout) -> Vector

The glue of `layout` in the form of TreeAMR's `Forest`: one
`(from = (p, d, side), to = (q, d′, side′), map = σ)` per glued face, both
directions of every gluing, `σ` a signed permutation in scatter form,
`σ[e] = ±a` sending axis `e` of `p` to axis `a` of `q`, reversed for a
minus.
"""
function layout_glue end

# The glue entries' type, a vector of which `layout_glue` returns.
glue_list(::Val{D}) where {D} =
    NamedTuple{(:from, :to, :map),Tuple{NTuple{3,Int},NTuple{3,Int},NTuple{D,Int}}}[]

"""
    layout_forest([T,] layout; N, kwargs...) -> Forest

TreeAMR's forest of `layout`'s patches and glue, `Forest{T}(patches; N,
glue, kwargs...)`: the forest's own keywords, `N` and whichever of
`reflecting`, `outflow`, `leaves` and `comm` the caller passes, with the
layout's glue. `T`, the forest's geometry type, defaults to the layout's
parameter type. The glue is the layout's, so a `glue` keyword is refused.

The face kinds are the caller's: TreeAMR's default makes every unglued
face an outer face, filled by the boundary hook. On the annulus,
`reflecting = ((false, false), (true, true))` makes the walls `r = r0` and
`r = r1` solid; a ball's only unglued faces are the outer sphere's, and
TreeAMR refuses a reflecting face kind there, since the cube is glued on
every face.
"""
layout_forest(layout::PatchLayout{D,T}; kwargs...) where {D,T} =
    layout_forest(T, layout; kwargs...)

function layout_forest(::Type{T}, layout::PatchLayout; N, kwargs...) where {T}
    haskey(kwargs, :glue) && throw(ArgumentError(
        "layout_forest takes the glue from the layout, $(layout_name(layout)); pass the " *
        "forest's other keywords only"))
    haskey(kwargs, :extents) && throw(ArgumentError(
        "every patch of $(layout_name(layout)) has the logical box [0, n] of its roots, " *
        "which its map is written for; pass no extents"))
    return Forest{T}(layout_patches(layout); N=N, glue=layout_glue(layout), kwargs...)
end

# The point of patch `q` that logical `ξ` of patch `p` lands on under the
# glue entry `g` from `p`'s face `(d, side)`: along the normal, a point
# past `p`'s face by `u` is inside `q`'s face by `u`; along the face, axis
# `e` of `p` is axis `|σ[e]|` of `q`, reversed for a minus. Exact on root
# boundaries, which is how `check_layout` reads it, and on the face itself
# the same physical point under both patches' maps (a test).
function glue_point(layout::PatchLayout{D}, g, ξ) where {D}
    p, d, side = g.from
    q, d′, side′ = g.to
    σ = g.map
    np = patch_roots(layout, p)
    nq = patch_roots(layout, q)
    u = side == 2 ? ξ[d] - np[d] : -ξ[d]
    return ntuple(Val(D)) do a
        e = findfirst(e -> abs(σ[e]) == a, 1:D)
        e == d ? (side′ == 1 ? u : nq[a] - u) : σ[e] > 0 ? ξ[e] : nq[a] - ξ[e]
    end
end

"""
    check_layout(layout, forest)

Refuse a `forest` that is not `layout`'s, saying why: its dimension, its
number of patches, a patch's box, or a face glued otherwise, or not glued
where the layout glues it. It reads the forest through TreeAMR's exports
only, the leaves and their neighbors across each patch face, so the glue
is checked where a leaf on one side meets a leaf on the other: which
patch, which face, and where the leaf across lies. A turn of the axes
within a single root is not seen at the root level; on a refined forest
the leaves' positions show it. The forest's face kinds are not checked.
"""
function check_layout(layout::PatchLayout{D}, forest::Forest{E}) where {D,E}
    name = layout_name(layout)
    E == D || throw(ArgumentError(
        "$name is $D-dimensional, and the forest has D = $E"))
    np = npatches(layout)
    leaves = forest.leaves
    counts = zeros(Int, max(np, maximum(k -> patch(forest, k), leaves)))
    for k in leaves
        counts[patch(forest, k)] += 1
    end
    length(counts) == np && all(>(0), counts) || throw(ArgumentError(
        "$name has $np patches, and the forest's leaves lie in " *
        "$(count(>(0), counts)) patches numbered up to $(length(counts)): build the " *
        "forest with layout_forest"))
    roots = [Set{Int32}() for _ in 1:np]
    for k in leaves
        push!(roots[patch(forest, k)], k.root)
    end
    for p in 1:np
        length(roots[p]) == prod(patch_roots(layout, p)) || throw(ArgumentError(
            "patch $p of $name has the roots $(patch_roots(layout, p)), and the forest's " *
            "patch $p has $(length(roots[p])) roots: build the forest with layout_forest"))
    end
    lo = [fill(Inf, D) for _ in 1:np]
    hi = [fill(-Inf, D) for _ in 1:np]
    for k in leaves
        p = patch(forest, k)
        ext = block_extent(Float64, forest, k)
        for d in 1:D
            lo[p][d] = min(lo[p][d], ext[d][1])
            hi[p][d] = max(hi[p][d], ext[d][2])
        end
    end
    for p in 1:np
        n = patch_roots(layout, p)
        all(d -> lo[p][d] == 0 && hi[p][d] == n[d], 1:D) || throw(ArgumentError(
            "patch $p of the forest covers the box $(Tuple(zip(lo[p], hi[p]))), and " *
            "$name maps the box of roots $n there: build the forest with layout_forest"))
    end
    glue = Dict(g.from => g for g in layout_glue(layout))
    for k in leaves
        check_leaf_glue(layout, forest, glue, k, name)
    end
    return nothing
end

# The faces of its patch that leaf `k` lies on, each against the layout's
# glue: an unglued face has no leaf across, and a glued one the leaf at
# the image of `k` mirrored across the face, in the patch the glue names.
function check_leaf_glue(layout::PatchLayout{D}, forest, glue, k, name) where {D}
    p = patch(forest, k)
    n = patch_roots(layout, p)
    ext = block_extent(Float64, forest, k)
    for d in 1:D, side in 1:2
        ext[d][side] == (side == 1 ? 0 : n[d]) || continue
        δ = ntuple(e -> e == d ? 2side - 3 : 0, Val(D))
        nbrs = neighbor_keys(forest, k, δ)
        g = get(glue, (p, d, side), nothing)
        if g === nothing
            isempty(nbrs) || throw(ArgumentError(
                "the forest glues patch $p's face (d = $d, side = $side) to patch " *
                "$(patch(forest, first(nbrs))), and $name leaves that face unglued: build " *
                "the forest with layout_forest"))
            continue
        end
        isempty(nbrs) && throw(ArgumentError(
            "$name glues patch $p's face (d = $d, side = $side) to patch $(g.to[1])'s " *
            "face (d = $(g.to[2]), side = $(g.to[3])), and the forest leaves it unglued: " *
            "build the forest with layout_forest"))
        h = ext[d][2] - ext[d][1]
        a = ntuple(e -> e == d ? (side == 1 ? -h : Float64(n[d])) : ext[e][1], Val(D))
        b = ntuple(e -> e == d ? (side == 1 ? 0.0 : n[d] + h) : ext[e][2], Val(D))
        a′, b′ = glue_point(layout, g, a), glue_point(layout, g, b)
        want = ntuple(e -> minmax(a′[e], b′[e]), Val(D))
        for nbr in nbrs
            got = block_extent(Float64, forest, nbr)
            patch(forest, nbr) == g.to[1] && got == want || throw(ArgumentError(
                "across patch $p's face (d = $d, side = $side) the forest has a leaf of " *
                "patch $(patch(forest, nbr)) at $got, and $name's glue puts patch " *
                "$(g.to[1])'s leaf at $want there: the forest's glue is not the " *
                "layout's; build the forest with layout_forest"))
        end
    end
    return nothing
end

# The frame from the Jacobian `J[i][a] = ∂xⁱ/∂ξᵃ`: `√g = |det J|` and the
# area vectors `S[d] = √g ∇ξ^d`, the rows of `|det J| J⁻¹`, which are the
# cofactors of `J` up to the sign of `det J`: in 3D `S[d] = ±(∂_e x × ∂_f
# x)` with `(d, e, f)` cyclic. No division, so `Σ_d ∂_d S[d] = 0`, the
# Piola identity, holds to rounding.
@inline function frame_from_jacobian(J::NTuple{2,NTuple{2,R}}) where {R}
    det = J[1][1] * J[2][2] - J[1][2] * J[2][1]
    σ = det < 0 ? -one(R) : one(R)
    S = ((σ * J[2][2], -σ * J[1][2]), (-σ * J[2][1], σ * J[1][1]))
    return σ * det, S
end

@inline function frame_from_jacobian(J::NTuple{3,NTuple{3,R}}) where {R}
    col(a) = (J[1][a], J[2][a], J[3][a])
    cross(u, v) = (u[2] * v[3] - u[3] * v[2], u[3] * v[1] - u[1] * v[3],
                   u[1] * v[2] - u[2] * v[1])
    c1, c2, c3 = col(1), col(2), col(3)
    a1, a2, a3 = cross(c2, c3), cross(c3, c1), cross(c1, c2)
    det = c1[1] * a1[1] + c1[2] * a1[2] + c1[3] * a1[3]
    σ = det < 0 ? -one(R) : one(R)
    return σ * det, (σ .* a1, σ .* a2, σ .* a3)
end

"""
    patch_frame(layout, ξ, p) -> (√g, S)

At logical `ξ` of patch `p`, the volume element `√g = |det ∂x/∂ξ|` and the
area vectors `S[d] = √g ∇ξ^d`, `S[d][i] = √g ∂ξ^d/∂xⁱ`, one `D`-tuple of
Cartesian components per logical dimension `d`. `√g` is the absolute
value: the annulus's patches and the balls' shells over the cube's low
faces are left-handed, and `S` carries the sign that makes `S[d] ⋅ ∂_d x =
√g`. The area vectors are the cofactors of the Jacobian, formed without a
division, so that `Σ_d ∂_d S[d] = 0` holds to rounding.
"""
@inline patch_frame(layout::PatchLayout, ξ, p) =
    frame_from_jacobian(TreeAMR.chart_jacobian(layout, ξ, p))

"""
    volume_element(layout, ξ, p)

`√g` at logical `ξ` of patch `p`, the first of [`patch_frame`](@ref).
"""
@inline volume_element(layout::PatchLayout, ξ, p) = first(patch_frame(layout, ξ, p))

"""
    area_vectors(layout, ξ, p)

The area vectors `S[d] = √g ∇ξ^d` at logical `ξ` of patch `p`, the second
of [`patch_frame`](@ref).
"""
@inline area_vectors(layout::PatchLayout, ξ, p) = last(patch_frame(layout, ξ, p))

"""
    logical_speed(layout, ξ, p)

The largest speed of light in logical coordinates at `ξ` of patch `p`,
`√λ_max(g^{ab})` with `g^{ab} = ∇ξ^a ⋅ ∇ξ^b = S[a] ⋅ S[b]/g` the inverse
metric: a scheme's time step is its logical one, `cfl ⋅ h`, over the
largest of these at its points. Closed form in 2D, and in 3D the
symmetric eigenvalue by the trigonometric formula, which needs `acos`
and `cos` of the number type (Base's floats and ForwardDiff's duals have
them; MultiFloats does not, and does not need a time step).
"""
@inline function logical_speed(layout::PatchLayout{D}, ξ, p) where {D}
    J, S = patch_frame(layout, ξ, p)
    g(a, b) = sum(i -> S[a][i] * S[b][i], 1:D) / J^2
    return sqrt(symmetric_eigmax(g, Val(D)))
end

@inline function symmetric_eigmax(g, ::Val{2})
    a, b, c = g(1, 1), g(1, 2), g(2, 2)
    m = (a - c) / 2
    return (a + c) / 2 + sqrt(m^2 + b^2)
end

# The largest eigenvalue of the symmetric 3×3 matrix `g(a, b)` (Smith,
# 1961): the eigenvalues of `B = (A - qI)/p` are `2cos(φ + 2πj/3)` with
# `cos 3φ = det(B)/2`. A diagonal matrix, the cube's, is taken exactly.
@inline function symmetric_eigmax(g, ::Val{3})
    a11, a22, a33 = g(1, 1), g(2, 2), g(3, 3)
    a12, a13, a23 = g(1, 2), g(1, 3), g(2, 3)
    p1 = a12^2 + a13^2 + a23^2
    iszero(p1) && return max(a11, a22, a33)
    q = (a11 + a22 + a33) / 3
    b11, b22, b33 = a11 - q, a22 - q, a33 - q
    p = sqrt((b11^2 + b22^2 + b33^2 + 2p1) / 6)
    det = b11 * (b22 * b33 - a23^2) - a12 * (a12 * b33 - a23 * a13) +
          a13 * (a12 * a23 - b22 * a13)
    r = clamp(det / (2p^3), -one(q), one(q))
    return q + 2p * cos(acos(r) / 3)
end

"""
    affine_patch(layout, p) -> Bool

Whether patch `p`'s map is affine, so that `√g`, the area vectors and the
metric are constants there: the balls' cube, and no other patch. `false`
unless a layout says otherwise. A scheme whose coupling across a
coarse-fine face is exact only where `√g` is constant refines there
alone; that rule is the scheme's.
"""
affine_patch(::PatchLayout, p) = false

"""
    affine_map(layout, p) -> (origin, scale)

Patch `p`'s map as `x = origin + scale ⋅ ξ`, for a patch whose map is
affine ([`affine_patch`](@ref)) with one scale along every dimension: a
`D`-tuple and a number. Refused for any other patch.
"""
affine_map(layout::PatchLayout, p) = throw(ArgumentError(
    "patch $p of $(layout_name(layout)) has no affine map: only a patch for which " *
    "affine_patch is true has one"))

"""
    factored_volume(layout, p) -> k

The logical dimension `k` along which patch `p`'s volume element splits
off, `√g(ξ) = radial_volume(layout, ξ_k, p) ⋅ angular_volume(layout, ξ_⊥,
p)` with `ξ_⊥` the `D - 1` other coordinates in increasing order of
dimension, or `0` when it does not factorize so, and on an affine patch,
whose `√g` is constant; `0` unless a layout says otherwise. The
thirteen-patch ball's outer shells and the annulus's patches factorize
along their radius. Both factors are generic in the number type, and their
product is [`volume_element`](@ref)'s `√g` to rounding (a test).
[`angular_class`](@ref) names the patches whose `angular_volume` is one
function of `ξ_⊥`.
"""
factored_volume(::PatchLayout, p) = 0

"""
    radial_volume(layout, ξ_k, p)

The factor of patch `p`'s `√g` along its dimension
[`factored_volume`](@ref)`(layout, p)`.
"""
function radial_volume end

"""
    angular_volume(layout, ξ_⊥, p)

The factor of patch `p`'s `√g` over the dimensions other than
[`factored_volume`](@ref)`(layout, p)`, a `(D-1)`-tuple in increasing
order of dimension.
"""
function angular_volume end

"""
    angular_class(layout, p)

Patches of one class have one [`angular_volume`](@ref) as a function of
`ξ_⊥`, so that a scheme's tables over it are shared; `p` unless a layout
says otherwise.
"""
angular_class(::PatchLayout, p) = p

"""
    radial_dimension(layout, p) -> k

The logical dimension of patch `p` that runs outward, `0` for a patch
without one, the balls' cube. Every shell's radial coordinate increases
outward, so a ball's outer sphere is the high face along `k` of its
outermost shells, and the annulus's outer wall the high face along 2 of
every patch.
"""
function radial_dimension end

"""
    outer_radius(layout)

The radius of the layout's outer boundary: the sphere of a ball, the
outer wall `r1` of the annulus.
"""
function outer_radius end

# The size of the layout's own patch list.
function npatches end

# The roots of patch `p`, an `NTuple{D,Int}`.
function patch_roots end

# The layout's name in a refusal.
function layout_name end
