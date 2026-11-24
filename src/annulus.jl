# The four-patch annulus: a ring of patches in polar coordinates, the
# angle along dimension 1 and the radius along dimension 2. Its maps join
# smoothly, `θ` linear through every glue, so its atlas declares both
# coordinates aligned. See "The annulus" in CODE.md.

"""
    PolarAnnulus{T}(r0, r1, roots)
    PolarAnnulus(r0, r1, roots)
    PolarAnnulus{T}(annulus)

The annulus `r0 ≤ r ≤ r1` as four patches of `roots = (n₁, n₂)` roots
each, the angle along dimension 1 and the radius along dimension 2. Patch
`p`'s logical box `[0, n₁] × [0, n₂]` maps to

    θ = (p - 1)π/2 + θ′ξ₁,    θ′ = (π/2)/n₁
    r = r0 + r′ξ₂,            r′ = (r1 - r0)/n₂
    x = (r cos θ, r sin θ)

so that `√g = r θ′ r′`. The quarter turn `(p - 1)π/2` is applied exactly,
as a rotation of `(cos θ′ξ₁, sin θ′ξ₁)`, and the map continues past the
box: `θ` past a glue is the next patch's, and `r` below `r0` or above `r1`
is the same line continued, which holds below the inner wall while `r > 0`,
`r0/r′` roots. Each patch's high face along dimension 1 is
glued to the next patch's low face by the identity; the faces `r = r0`
and `r = r1` are unglued, and their face kind is the forest's
([`layout_forest`](@ref)). The patches are left-handed, `det ∂x/∂ξ < 0`.
`PolarAnnulus{T}(annulus)` is the same annulus with its parameters in
`T`, `Float32` for a device.
"""
struct PolarAnnulus{T} <: PatchLayout{2,T}
    r0::T
    r1::T
    roots::NTuple{2,Int}

    function PolarAnnulus{T}(r0, r1, roots::NTuple{2,Integer}) where {T}
        0 < r0 < r1 || throw(ArgumentError(
            "an annulus needs 0 < r0 < r1, got r0 = $r0 and r1 = $r1"))
        all(>(0), roots) || throw(ArgumentError(
            "every patch of the annulus needs a positive number of roots along each " *
            "dimension, got $roots"))
        return new{T}(T(r0), T(r1), Int.(roots))
    end
end

PolarAnnulus{T}(r0, r1, roots) where {T} = PolarAnnulus{T}(r0, r1, Tuple(roots))
PolarAnnulus{T}(a::PolarAnnulus) where {T} = PolarAnnulus{T}(a.r0, a.r1, a.roots)
PolarAnnulus(r0, r1, roots) =
    PolarAnnulus{float(promote_type(typeof(r0), typeof(r1)))}(r0, r1, roots)

npatches(::PolarAnnulus) = 4
patch_roots(a::PolarAnnulus, p) = a.roots
layout_name(::PolarAnnulus) = "the four-patch annulus"
outer_radius(a::PolarAnnulus) = a.r1
radial_dimension(::PolarAnnulus, p) = 2

function layout_glue(::PolarAnnulus)
    glue = glue_list(Val(2))
    for p in 1:4
        q = mod1(p + 1, 4)
        push!(glue, (from=(p, 1, 2), to=(q, 1, 1), map=(1, 2)))
        push!(glue, (from=(q, 1, 1), to=(p, 1, 2), map=(1, 2)))
    end
    return glue
end

# `(c, s)` turned by `q` quarter turns.
@inline quarter_turn(c, s, q) =
    q == 0 ? (c, s) : q == 1 ? (-s, c) : q == 2 ? (-c, -s) : (s, -c)

# At logical `ξ` of patch `p`: `r`, `r′`, `θ′` and `(cos θ, sin θ)`, in
# the map's number type.
@inline function annulus_parts(a::PolarAnnulus, ξ, p)
    (ξ1, ξ2), R = evalpoint(a, ξ)
    n1, n2 = a.roots
    θ′ = convert(R, π) / (2n1)
    r′ = (convert(R, a.r1) - convert(R, a.r0)) / n2
    r = convert(R, a.r0) + r′ * ξ2
    s, c = sincos_local(θ′ * ξ1)
    c, s = quarter_turn(c, s, p - 1)
    return r, r′, θ′, c, s
end

@inline function TreeAMR.chart_map(a::PolarAnnulus, ξ, p)
    r, _, _, c, s = annulus_parts(a, ξ, p)
    return (r * c, r * s)
end

@inline function TreeAMR.chart_jacobian(a::PolarAnnulus, ξ, p)
    r, r′, θ′, c, s = annulus_parts(a, ξ, p)
    rθ′ = r * θ′
    return ((-rθ′ * s, r′ * c), (rθ′ * c, r′ * s))
end

@inline function TreeAMR.chart_hessian(a::PolarAnnulus, ξ, p)
    r, r′, θ′, c, s = annulus_parts(a, ξ, p)
    z = zero(r)
    rθ′² = r * θ′^2
    θ′r′ = θ′ * r′
    return (((-rθ′² * c, -θ′r′ * s), (-θ′r′ * s, z)),
            ((-rθ′² * s, θ′r′ * c), (θ′r′ * c, z)))
end

# The patch is the quadrant, found from the signs of `x` alone, and the
# angle within it from `x` turned back by whole quarter turns, so that no
# multiple of `π` is rounded. A point on a glue goes to the patch whose
# low face it is, a point within a few roundings of a wall to the wall.
@inline function TreeAMR.chart_locate(a::PolarAnnulus{T}, x) where {T}
    R = promote_type(T, typeof(x[1]), typeof(x[2]))
    X, Y = convert(R, x[1]), convert(R, x[2])
    r0, r1 = convert(R, a.r0), convert(R, a.r1)
    n1, n2 = a.roots
    r = sqrt(X^2 + Y^2)
    tol = 8 * eps(R)
    r0 * (1 - tol) <= r <= r1 * (1 + tol) || return nothing
    p = Y >= 0 ? (X > 0 ? 1 : 2) : (X < 0 ? 3 : 4)
    U, V = p == 1 ? (X, Y) : p == 2 ? (Y, -X) : p == 3 ? (-X, -Y) : (-Y, X)
    θ′ = convert(R, π) / (2n1)
    r′ = (r1 - r0) / n2
    ξ1 = clamp(atan_local(V, U) / θ′, zero(R), convert(R, n1))
    ξ2 = clamp((r - r0) / r′, zero(R), convert(R, n2))
    return (ξ1, ξ2), p
end

# `θ` is linear through every glue and `r` the same function of `ξ₂` on
# every patch, so a node lands on a node along both.
chart_aligned(::PolarAnnulus, p, q) = (true, true)

# `√g = (r r′) ⋅ θ′`: the radius's factor times a constant, the same on all
# four patches.
factored_volume(::PolarAnnulus, p) = 2
angular_class(::PolarAnnulus, p) = 1

@inline function radial_volume(a::PolarAnnulus, ξ2, p)
    R = promote_type(typeof(a.r0), typeof(ξ2))
    r′ = (convert(R, a.r1) - convert(R, a.r0)) / a.roots[2]
    return (convert(R, a.r0) + r′ * ξ2) * r′
end

@inline function angular_volume(a::PolarAnnulus, ξ⊥, p)
    R = promote_type(typeof(a.r0), typeof(ξ⊥[1]))
    return convert(R, π) / (2a.roots[1])
end
