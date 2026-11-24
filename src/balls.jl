# The seven-patch and the thirteen-patch ball: a cube, six shells over
# its faces that blend its flat faces into a sphere, and for the
# thirteen-patch ball six more shells outside that sphere whose radial
# coordinate is the radius. Every shell's radial coordinate increases
# outward, so the outer sphere is every outermost shell's high face. See
# "The balls" in CODE.md.
#
# Each shell's map has the form `x = (A + B/|f|) f`, with `f = Lc c` the
# point of the cube's face over which `ξ` lies, `c_k = s`, `c_j = 2ξ_j/n
# - 1` across it, and `A`, `B` functions of the radial coordinate `ξ_k`
# alone (`blended_profile`, `outer_profile`). So `x` is parallel to `f`,
# which is what inverts the map in closed form, and one routine
# differentiates every shell (`shell_jacobian`, `shell_hessian`).

# The shells' cube faces `(k, s)`, in patch order 2–7.
const SHELLS = ((1, -1), (1, 1), (2, -1), (2, 1), (3, -1), (3, 1))

# The `i`-th of them by arithmetic, `SHELLS[i]` without the run-time index
# into a global tuple, which a device kernel would read from memory it may
# not reach.
@inline shell_of(i) = ((i - 1) ÷ 2 + 1, isodd(i) ? -1 : 1)

# The patch number of the (inner) shell over face `(k, s)`.
shell_index(k, s) = 1 + 2(k - 1) + (s == 1 ? 2 : 1)

# The face side, 1 or 2, a step along the sign `s` leaves through.
shell_side(s) = s == 1 ? 2 : 1

# The cube's `x = Lc(2ξ/n - 1)`, written `(2Lc/n)ξ - Lc`, as `affine_map`
# gives it: exact where the step `2Lc/n` and `ξ` are dyadic, so that the
# cube's nodes lie where a brick of the same step places them.
@inline cube_position(Lc, n, ξ) = (2Lc / n) .* ξ .- Lc

@inline function cube_jacobian(Lc, n, ::Type{R}) where {R}
    a, z = 2Lc / n, zero(R)
    return ((a, z, z), (z, a, z), (z, z, a))
end

@inline function cube_hessian(::Type{R}) where {R}
    z = zero(R)
    Z = ((z, z, z), (z, z, z), (z, z, z))
    return (Z, Z, Z)
end

# The point of the cube's face `(k, s)`, of half-width `Lc` and `n` roots
# along each tangential dimension, over which logical `ξ` lies; past the
# box, `|c_j| > 1`, it is the face's plane continued.
@inline shell_face(Lc, n, ξ, k, s) =
    ntuple(j -> j == k ? s * Lc : Lc * (2 * ξ[j] / n - 1), Val(3))

# The blended shell's `A = 1 - ρ` and `B = Rρ` with their first and second
# derivatives along `ξ_k`, `ρ` the fraction of the way from the cube's
# face to the sphere `|x| = R` at `η = ξ_k/m`, graded by `α`:
# `ρ = (e^{αη} - 1)/(e^α - 1)`, `ρ = η` for `α = 0`. Past the box `η < 0`
# reaches into the cube and `η > 1` beyond the sphere.
@inline function blended_profile(ξk, m, R, α)
    η = ξk / m
    if iszero(α)
        ρ, ρ′, ρ″ = η, one(η), zero(η)
    else
        E = exp(α) - 1
        e = exp(α * η)
        ρ, ρ′, ρ″ = (e - 1) / E, α * e / E, α^2 * e / E
    end
    return (1 - ρ, -ρ′ / m, -ρ″ / m^2, ρ * R, R * ρ′ / m, R * ρ″ / m^2)
end

# The outer shell's `A = 0` and `B = r = Rm + (R1 - Rm)ξ_k/m′`, the radius.
@inline function outer_profile(ξk, m′, Rm, R1)
    z = zero(ξk)
    return (z, z, z, Rm + (R1 - Rm) * (ξk / m′), (R1 - Rm) / m′, z)
end

@inline shell_position(f, A, B) =
    (nf = sqrt(f[1]^2 + f[2]^2 + f[3]^2); ntuple(i -> A * f[i] + (B / nf) * f[i], Val(3)))

# The shell's `G = A + Bq`, `q = 1/|f|`, and its first derivatives, with
# `q`'s: the parts both the Jacobian and the Hessian need. Across `k`,
# `∂_j f = a e_j` with `a = 2Lc/n`, so `∂_j q = -a f_j q³`; `∂_k q = 0`.
@inline function shell_parts(f, k, a, prof)
    A, A′, _, B, B′, _ = prof
    q = inv(sqrt(f[1]^2 + f[2]^2 + f[3]^2))
    q3 = q^3
    dq = ntuple(e -> e == k ? zero(q) : -a * f[e] * q3, Val(3))
    G = A + B * q
    dG = ntuple(e -> (e == k ? A′ + B′ * q : zero(q)) + B * dq[e], Val(3))
    return q, q3, dq, G, dG
end

# `∂_e f_i`: `a` on the diagonal across `k`, zero otherwise.
@inline face_derivative(e, i, k, a) = e != k && i == e ? a : zero(a)

# `∂_e x_i = ∂_e G f_i + G ∂_e f_i`.
@inline function shell_jacobian(f, k, a, prof)
    _, _, _, G, dG = shell_parts(f, k, a, prof)
    return ntuple(i -> ntuple(e -> dG[e] * f[i] + G * face_derivative(e, i, k, a),
                              Val(3)), Val(3))
end

# `∂_e∂_b x_i = ∂_eb G f_i + ∂_e G ∂_b f_i + ∂_b G ∂_e f_i`, `f` being
# affine in `ξ`, with `∂_eb q = a²(3 f_e f_b q⁵ - δ_eb q³)` across `k`.
@inline function shell_hessian(f, k, a, prof)
    _, _, A″, B, B′, B″ = prof
    q, q3, dq, _, dG = shell_parts(f, k, a, prof)
    q5 = q3 * q^2
    ddq(e, b) = e == k || b == k ? zero(q) :
        a^2 * (3 * (f[e] * f[b]) * q5 - (e == b ? q3 : zero(q)))
    ddG(e, b) = (e == k && b == k ? A″ + B″ * q : zero(q)) +
                (e == k ? B′ * dq[b] : zero(q)) + (b == k ? B′ * dq[e] : zero(q)) +
                B * ddq(e, b)
    return ntuple(i -> ntuple(e -> ntuple(b -> ddG(e, b) * f[i] +
                                               dG[e] * face_derivative(b, i, k, a) +
                                               dG[b] * face_derivative(e, i, k, a),
                                          Val(3)), Val(3)), Val(3))
end

# Where `x` lies relative to the cube `[-Lc, Lc]³`: inside it, with the
# cube's `ξ`; or over its face `(k, s)`, `k` the first dimension of largest
# `|x_k|`, with the face's tangential `ξ` and `c = x/|x_k|`, since a
# shell's `x` is parallel to its `f = Lc c`.
@inline function ball_face(x, Lc, n)
    R = typeof(Lc)
    ax = abs.(x)
    k = ax[1] >= ax[2] ? (ax[1] >= ax[3] ? 1 : 3) : (ax[2] >= ax[3] ? 2 : 3)
    if ax[k] <= Lc
        ξ = ntuple(j -> clamp((x[j] + Lc) / (2Lc / n), zero(R), convert(R, n)), Val(3))
        return true, ξ, k, 1, ax[k], ax[k]
    end
    s = x[k] < 0 ? -1 : 1
    c = ntuple(j -> x[j] / ax[k], Val(3))
    ξ = ntuple(j -> j == k ? zero(R) : clamp(n * (c[j] + 1) / 2, zero(R), convert(R, n)),
               Val(3))
    return false, ξ, k, s, ax[k], sqrt(c[1]^2 + c[2]^2 + c[3]^2)
end

# The radial `ξ_k` of a blended shell at `x` over face `k`: `|x_k| = G Lc`
# and `G = 1 + ρ(R/|f| - 1)`, so `ρ = (|x_k|/Lc - 1)/(R/(Lc|c|) - 1)`,
# and `η` inverts the grading.
@inline function blended_radial(axk, nc, Lc, m, R, α)
    ρ = clamp((axk / Lc - 1) / (R / (Lc * nc) - 1), zero(R), one(R))
    η = iszero(α) ? ρ : log(1 + ρ * (exp(α) - 1)) / α
    return clamp(m * η, zero(R), oftype(R, m))
end

# `ξ` with `ξk` along `k`: `Base.setindex` with a run-time index infers
# as a tuple of `Any`, which a device kernel cannot compile.
@inline with_radial(ξ, ξk, k) = ntuple(j -> j == k ? ξk : ξ[j], Val(3))

# A point within a few roundings of the sphere `|x| = R1` is on it.
@inline inside_sphere(r2, R1) = r2 <= (R1 * (1 + 8 * eps(typeof(R1))))^2

"""
    SevenPatchBall{T}(Lc, R1, roots; grading=0)
    SevenPatchBall(Lc, R1, roots; grading=0)
    SevenPatchBall{T}(ball)

The ball `|x| ≤ R1` as seven patches, with `roots = (n, m)`:

- patch 1, the cube `[-Lc, Lc]³`, of `(n, n, n)` roots: its logical box
  `[0, n]³` maps to `x = Lc (2ξ/n - 1)`;
- patches 2–7, the shells over the cube's faces `(k, s)` in the order
  `(1, -1), (1, 1), (2, -1), (2, 1), (3, -1), (3, 1)`, each of `m` roots
  along its radial dimension `k` and `n` along the other two. Its logical
  box maps to

      c_k = s,   c_j = 2ξ_j/n - 1 (j ≠ k),   f = Lc c
      η = ξ_k/m,   ρ = (e^{αη} - 1)/(e^α - 1)   (ρ = η for α = 0)
      x = (1 - ρ) f + ρ R1 f/|f|

  so that its inner face, `ξ_k = 0`, is the cube's face, flat, and its
  outer face, `ξ_k = m`, the sphere `|x| = R1`. Each tangential dimension
  `j` of a shell runs along `x_j`, and `ξ_k` increases outward on every
  shell, so that the sphere is every shell's high face along `k`; the
  shells over the cube's low faces, `s = -1`, are left-handed.

The cube is glued to each shell's inner face, by the identity over the
cube's high faces and by a reversal of `k` over its low faces, which are
glued low to low; the shells are glued to each other by quarter turns
that carry each one's radial dimension onto the other's. The map is
defined past every box: `c_j` past `±1` continues the face's plane, `η < 0`
reaches into the cube, `η > 1` past the sphere.

The radial coordinate is uniform for `grading = α = 0`. For `α > 0` the
shells' radial elements widen outward by one ratio from each to the next,
the outermost `e^α` times as wide as the innermost. `R1 > √3 Lc`, so
that the cube's corners lie inside the sphere and the map is one to one.
`SevenPatchBall{T}(ball)` is the same ball with its parameters in `T`.
"""
struct SevenPatchBall{T} <: PatchLayout{3,T}
    Lc::T
    R1::T
    roots::NTuple{2,Int}
    grading::T                   # α

    function SevenPatchBall{T}(Lc, R1, roots::NTuple{2,Integer}; grading=0) where {T}
        0 < Lc || throw(ArgumentError(
            "the cube's half-width Lc must be positive, got $Lc"))
        R1 > sqrt(3) * Lc || throw(ArgumentError(
            "the outer radius R1 = $R1 must exceed √3 Lc = $(sqrt(3) * Lc), so that " *
            "the cube's corners lie inside the sphere and every shell maps one to one"))
        all(>(0), roots) || throw(ArgumentError(
            "every patch of the ball needs a positive number of roots along each " *
            "dimension, got $roots"))
        isfinite(grading) || throw(ArgumentError(
            "the shells' radial grading must be finite, got $grading"))
        return new{T}(T(Lc), T(R1), Int.(roots), T(grading))
    end
end

SevenPatchBall{T}(Lc, R1, roots; grading=0) where {T} =
    SevenPatchBall{T}(Lc, R1, Tuple(roots); grading=grading)
SevenPatchBall{T}(b::SevenPatchBall) where {T} =
    SevenPatchBall{T}(b.Lc, b.R1, b.roots; grading=b.grading)
SevenPatchBall(Lc, R1, roots; grading=0) =
    SevenPatchBall{float(promote_type(typeof(Lc), typeof(R1)))}(Lc, R1, roots;
                                                                grading=grading)

"""
    ThirteenPatchBall{T}(Lc, Rm, R1, roots)
    ThirteenPatchBall(Lc, Rm, R1, roots)
    ThirteenPatchBall{T}(ball)

The ball `|x| ≤ R1` as thirteen patches, with `roots = (n, m, m′)`: the
seven-patch ball of [`SevenPatchBall`](@ref) out to the middle sphere
`|x| = Rm`, ungraded, and six shells around it whose radial coordinate is
the radius.

- patch 1, the cube `[-Lc, Lc]³`, of `(n, n, n)` roots, `x = Lc (2ξ/n - 1)`;
- patches 2–7, the inner shells over the cube's faces `(k, s)` in the
  order `(1, -1), (1, 1), (2, -1), (2, 1), (3, -1), (3, 1)`, of `m` roots
  along `k` and `n` along the other two: [`SevenPatchBall`](@ref)'s shells
  with `Rm` for `R1`;
- patches 8–13, the outer shells over the same faces in the same order,
  of `m′` roots along `k` and `n` along the other two:

      c_k = s,   c_j = 2ξ_j/n - 1 (j ≠ k)
      x = (Rm + (R1 - Rm) ξ_k/m′) c/|c|

  so that `ξ_k` is a function of `r = |x|` alone, increasing outward, and
  the other two coordinates of `x/r` alone, the same functions as on the
  inner shells' outer face.

The cube–shell glue is the seven-patch ball's; each inner shell's outer
face is glued to the outer shell's inner face over the same cube face by
the identity, high to low; and the shells of each layer are glued to each
other by the seven-patch ball's quarter turns. So the outer sphere is
every outer shell's high face along `k`. Because the coordinates are
shared, [`chart_aligned`](@ref) declares the radius between two outer
shells and both angles between an inner and an outer shell over the same
face. On an outer shell `√g = r² r′ ⋅ (2/n)² (1 + c_a² + c_b²)^{-3/2}`,
`r′ = (R1 - Rm)/m′`, a radial factor times an angular one that is the
same on all six ([`factored_volume`](@ref)). `Rm > √3 Lc` and `R1 > Rm`.
`ThirteenPatchBall{T}(ball)` is the same ball with its parameters in `T`.
"""
struct ThirteenPatchBall{T} <: PatchLayout{3,T}
    Lc::T
    Rm::T
    R1::T
    roots::NTuple{3,Int}

    function ThirteenPatchBall{T}(Lc, Rm, R1, roots::NTuple{3,Integer}) where {T}
        0 < Lc || throw(ArgumentError(
            "the cube's half-width Lc must be positive, got $Lc"))
        Rm > sqrt(3) * Lc || throw(ArgumentError(
            "the middle radius Rm = $Rm must exceed √3 Lc = $(sqrt(3) * Lc), so that " *
            "the cube's corners lie inside the middle sphere and every inner shell " *
            "maps one to one"))
        R1 > Rm || throw(ArgumentError(
            "the outer radius R1 = $R1 must exceed the middle radius Rm = $Rm"))
        all(>(0), roots) || throw(ArgumentError(
            "every patch of the ball needs a positive number of roots along each " *
            "dimension, got $roots"))
        return new{T}(T(Lc), T(Rm), T(R1), Int.(roots))
    end
end

function ThirteenPatchBall{T}(Lc, Rm, R1, roots) where {T}
    length(roots) == 3 || throw(ArgumentError(
        "the thirteen-patch ball takes roots = (n, m, m′), the cube's per side and the " *
        "inner and outer shells' radial ones, got $(repr(roots))"))
    return ThirteenPatchBall{T}(Lc, Rm, R1, Tuple(roots))
end
ThirteenPatchBall{T}(b::ThirteenPatchBall) where {T} =
    ThirteenPatchBall{T}(b.Lc, b.Rm, b.R1, b.roots)
ThirteenPatchBall(Lc, Rm, R1, roots) =
    ThirteenPatchBall{float(promote_type(typeof(Lc), typeof(Rm), typeof(R1)))}(
        Lc, Rm, R1, roots)

const Ball{T} = Union{SevenPatchBall{T},ThirteenPatchBall{T}}

npatches(::SevenPatchBall) = 7
npatches(::ThirteenPatchBall) = 13
layout_name(::SevenPatchBall) = "the seven-patch ball"
layout_name(::ThirteenPatchBall) = "the thirteen-patch ball"
outer_radius(b::Ball) = b.R1

# The face `(k, s)` patch `p` lies over and whether it is an outer shell;
# `k = 0` for the cube.
@inline function ball_shell(b::SevenPatchBall, p)
    p == 1 && return 0, 1, false
    k, s = shell_of(p - 1)
    return k, s, false
end
@inline function ball_shell(b::ThirteenPatchBall, p)
    p == 1 && return 0, 1, false
    p <= 7 && return (shell_of(p - 1)..., false)
    return (shell_of(p - 7)..., true)
end

function patch_roots(b::Ball, p)
    n, m = b.roots[1], b.roots[2]
    k, _, outer = ball_shell(b, p)
    k == 0 && return (n, n, n)
    return ntuple(j -> j == k ? (outer ? b.roots[3] : m) : n, 3)
end

radial_dimension(b::Ball, p) = first(ball_shell(b, p))

# The radial profile of shell `p`, `(k, s, A, A′, A″, B, B′, B″)` at `ξ_k`,
# in `R`.
@inline function shell_profile(b::SevenPatchBall, p, ξ, ::Type{R}) where {R}
    k, s, _ = ball_shell(b, p)
    return k, s, blended_profile(ξ[k], b.roots[2], convert(R, b.R1),
                                 convert(R, b.grading))
end
@inline function shell_profile(b::ThirteenPatchBall, p, ξ, ::Type{R}) where {R}
    k, s, outer = ball_shell(b, p)
    prof = outer ?
        outer_profile(ξ[k], b.roots[3], convert(R, b.Rm), convert(R, b.R1)) :
        blended_profile(ξ[k], b.roots[2], convert(R, b.Rm), zero(R))
    return k, s, prof
end

@inline function TreeAMR.chart_map(b::Ball, ξ, p)
    ξ, R = evalpoint(b, ξ)
    n = b.roots[1]
    Lc = convert(R, b.Lc)
    p == 1 && return cube_position(Lc, n, ξ)
    k, s, prof = shell_profile(b, p, ξ, R)
    return shell_position(shell_face(Lc, n, ξ, k, s), prof[1], prof[4])
end

@inline function TreeAMR.chart_jacobian(b::Ball, ξ, p)
    ξ, R = evalpoint(b, ξ)
    n = b.roots[1]
    Lc = convert(R, b.Lc)
    p == 1 && return cube_jacobian(Lc, n, R)
    k, s, prof = shell_profile(b, p, ξ, R)
    return shell_jacobian(shell_face(Lc, n, ξ, k, s), k, 2Lc / n, prof)
end

@inline function TreeAMR.chart_hessian(b::Ball, ξ, p)
    ξ, R = evalpoint(b, ξ)
    n = b.roots[1]
    Lc = convert(R, b.Lc)
    p == 1 && return cube_hessian(R)
    k, s, prof = shell_profile(b, p, ξ, R)
    return shell_hessian(shell_face(Lc, n, ξ, k, s), k, 2Lc / n, prof)
end

# The cube if `x` is in it; otherwise the shell over the face `x` is
# parallel to, whose tangential coordinates `c = x/|x_k|` gives, and the
# radial one from `|x|`. A point on a face between two patches goes to
# either, a point within a few roundings of the sphere to the sphere.
@inline function TreeAMR.chart_locate(b::SevenPatchBall{T}, y) where {T}
    R = promote_type(T, typeof(y[1]), typeof(y[2]), typeof(y[3]))
    x = ntuple(i -> convert(R, y[i]), Val(3))
    n, m = b.roots
    Lc, R1 = convert(R, b.Lc), convert(R, b.R1)
    incube, ξ, k, s, axk, nc = ball_face(x, Lc, n)
    incube && return ξ, 1
    inside_sphere(x[1]^2 + x[2]^2 + x[3]^2, R1) || return nothing
    ξk = blended_radial(axk, nc, Lc, m, R1, convert(R, b.grading))
    return with_radial(ξ, ξk, k), shell_index(k, s)
end

@inline function TreeAMR.chart_locate(b::ThirteenPatchBall{T}, y) where {T}
    R = promote_type(T, typeof(y[1]), typeof(y[2]), typeof(y[3]))
    x = ntuple(i -> convert(R, y[i]), Val(3))
    n, m, m′ = b.roots
    Lc, Rm, R1 = convert(R, b.Lc), convert(R, b.Rm), convert(R, b.R1)
    incube, ξ, k, s, axk, nc = ball_face(x, Lc, n)
    incube && return ξ, 1
    r2 = x[1]^2 + x[2]^2 + x[3]^2
    if r2 <= Rm^2
        ξk = blended_radial(axk, nc, Lc, m, Rm, zero(R))
        return with_radial(ξ, ξk, k), shell_index(k, s)
    end
    inside_sphere(r2, R1) || return nothing
    ξk = clamp(m′ * ((sqrt(r2) - Rm) / (R1 - Rm)), zero(R), convert(R, m′))
    return with_radial(ξ, ξk, k), 6 + shell_index(k, s)
end

# The seven-patch ball's maps kink at every glue, and no coordinate is
# shared.
chart_aligned(::SevenPatchBall, p, q) = ntuple(_ -> p == q, Val(3))

# On the thirteen-patch ball an outer shell's radial coordinate is a
# function of `|x|` alone, the same on all six, and an inner and an outer
# shell over one face share their tangential coordinates, functions of
# `x/|x|` alone.
function chart_aligned(b::ThirteenPatchBall, p, q)
    p == q && return (true, true, true)
    kp, sp, op = ball_shell(b, p)
    kq, sq, oq = ball_shell(b, q)
    op && oq && return ntuple(e -> e == kq, Val(3))
    kp != 0 && (kp, sp) == (kq, sq) && op != oq && return ntuple(e -> e != kq, Val(3))
    return (false, false, false)
end

# The cube's map is affine; the shells' are not.
affine_patch(::Ball, p) = p == 1

function affine_map(b::Ball, p)
    p == 1 || throw(ArgumentError(
        "patch $p of $(layout_name(b)) is a shell, whose map is not affine; only the " *
        "cube, patch 1, has an affine map"))
    return ntuple(_ -> -b.Lc, Val(3)), 2b.Lc / b.roots[1]
end

# On an outer shell over the face `(k, s)`, `x = r(ξ_k) c/|c|`: the
# Jacobian's columns are `r′ c/|c|` along `k` and `r ∂(c/|c|)/∂ξ_j`
# across it, and `|c/|c| ⋅ (∂_a(c/|c|) × ∂_b(c/|c|))| = (2/n)² |c_k|/|c|³`,
# the gnomonic projection's area element, with `|c_k| = 1`. So `√g = r² r′
# ⋅ (2/n)² (1 + c_a² + c_b²)^{-3/2}`, the angular factor the same on every
# outer shell.
factored_volume(b::ThirteenPatchBall, p) = p >= 8 ? radial_dimension(b, p) : 0
angular_class(::ThirteenPatchBall, p) = p >= 8 ? 8 : p

@inline function radial_volume(b::ThirteenPatchBall, ξk, p)
    R = promote_type(typeof(b.Rm), typeof(ξk))
    Rm, R1 = convert(R, b.Rm), convert(R, b.R1)
    m′ = b.roots[3]
    r = Rm + (R1 - Rm) * (ξk / m′)
    return r^2 * ((R1 - Rm) / m′)
end

@inline function angular_volume(b::ThirteenPatchBall, ξ⊥, p)
    n = b.roots[1]
    ca = 2 * ξ⊥[1] / n - 1
    cb = 2 * ξ⊥[2] / n - 1
    q = 1 + ca^2 + cb^2
    return (2 / oftype(ca, n))^2 / (q * sqrt(q))
end

# The glue, both directions of every gluing. The cube's face `(k, s)` onto
# the inner face of shell `p` over it: high to low by the identity for
# `s = 1`, low to low by the reversal of `k` for `s = -1`, since the shell's
# `ξ_k` runs outward either way.
function radial_glue!(glue, p, k, s)
    σ = ntuple(e -> e == k ? s * k : e, 3)
    push!(glue, (from=(1, k, shell_side(s)), to=(p, k, 1), map=σ))
    push!(glue, (from=(p, k, 1), to=(1, k, shell_side(s)), map=σ))
    return glue
end

# Inner shell `p`'s outer face onto outer shell `q`'s inner face, high to
# low by the identity.
function layer_glue!(glue, p, q, k)
    push!(glue, (from=(p, k, 2), to=(q, k, 1), map=(1, 2, 3)))
    push!(glue, (from=(q, k, 1), to=(p, k, 2), map=(1, 2, 3)))
    return glue
end

# Shell `(k, s)`'s faces towards the other shells of its layer, whose
# patch numbers are `index(l, t)`, onto each one's face towards `(k, s)`:
# the radial `k` onto the radial `l`, both outward; the normal `l` onto
# the normal `k`, reversed where both faces are high or both low; the third
# axis onto itself.
function quarter_turns!(glue, index, k, s)
    p = index(k, s)
    for (l, t) in SHELLS
        l == k && continue
        σ = ntuple(e -> e == k ? l : e == l ? (s == t ? -k : k) : e, 3)
        push!(glue, (from=(p, l, shell_side(t)), to=(index(l, t), k, shell_side(s)),
                     map=σ))
    end
    return glue
end

function layout_glue(::SevenPatchBall)
    glue = glue_list(Val(3))
    for (k, s) in SHELLS
        radial_glue!(glue, shell_index(k, s), k, s)
        quarter_turns!(glue, shell_index, k, s)
    end
    return glue
end

function layout_glue(::ThirteenPatchBall)
    glue = glue_list(Val(3))
    outer(k, s) = 6 + shell_index(k, s)
    for (k, s) in SHELLS
        radial_glue!(glue, shell_index(k, s), k, s)
        layer_glue!(glue, shell_index(k, s), outer(k, s), k)
        quarter_turns!(glue, shell_index, k, s)
        quarter_turns!(glue, outer, k, s)
    end
    return glue
end
