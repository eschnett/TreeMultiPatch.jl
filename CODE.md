# TreeMultiPatch.jl — Design

**Status.** Version 0.1, unregistered. The three layouts, their forests,
maps and atlases are implemented and tested, on the CPU and on Metal: step
1 of TreeAMR's M20. No application uses the package yet (`PLAN.md`).

**Contents**

- [What this is](#what-this-is)
- [Code structure](#code-structure)
- [Layouts](#layouts)
  - [The forest and its check](#the-forest-and-its-check)
  - [The annulus](#the-annulus)
  - [The balls](#the-balls)
  - [Shell orientation](#shell-orientation)
- [The maps](#the-maps)
  - [Number types](#number-types)
  - [Past the box](#past-the-box)
  - [The frame and the logical speed](#the-frame-and-the-logical-speed)
- [The atlas](#the-atlas)
  - [The inverse map](#the-inverse-map)
  - [Aligned coordinates](#aligned-coordinates)
- [Patch properties](#patch-properties)
- [What stays in the applications](#what-stays-in-the-applications)
- [Departures from TreeWaveOLDG](#departures-from-treewaveoldg)
- [Tests](#tests)
- [Known limitations and open questions](#known-limitations-and-open-questions)

## What this is

TreeAMR builds forests of patches, logical boxes glued at their faces
(its M17), and keeps its coordinates logical. The map of each box to
physical space, its Jacobian, metric and inverse, belong to whoever owns
the patch system. TreeWaveOLDG owned the annulus and the two balls; every
application on them needs the same maps (TreeWave next, then
TreeGeneralizedHarmonic and TreeGRRMHD), so they live here, in a package
with no physics. TreeAMR stays free of the maps' dependencies, and a new
layout costs no TreeAMR release (TreeAMR's `PLAN.md`, M20, "Why a package
and not TreeAMR").

Dependencies: TreeAMR, through its exports and its public `chart_*`
functions only, with `[compat] TreeAMR = "0.2"`, an ordinary range: a
need for an internal would mean the thing belongs in TreeAMR, exported.
MultiFloats and ForwardDiff are weak dependencies, for the extensions
under [Number types](#number-types).

## Code structure

| File | What |
|---|---|
| `src/TreeMultiPatch.jl` | the module, its exports |
| `src/layout.jl` | `PatchLayout`, `chart_aligned`, `layout_forest`, `check_layout`, the glue's point map, the frame from the Jacobian, `logical_speed`, the patch properties' defaults |
| `src/annulus.jl` | `PolarAnnulus` |
| `src/balls.jl` | `SevenPatchBall`, `ThirteenPatchBall`: the cube, the blended and the outer shells, their glue |
| `ext/TreeMultiPatchMultiFloatsExt.jl` | `sin`, `cos`, `atan` for MultiFloats' numbers |
| `ext/TreeMultiPatchForwardDiffExt.jl` | the same for ForwardDiff's duals over any number |

Rules that span files:

- **A layout is `isbits`** for an `isbits` parameter type, and everything
  evaluated at a point (`chart_*`, `patch_frame`, `logical_speed`, the
  volume factors) runs in a kernel on any backend: no allocation, no
  global read at run time, and every return type concrete. A run-time
  index into a tuple through `Base.setindex`, or a variable captured by a
  closure and reassigned, infers as `Any` and fails on a device; the
  suite checks the inferred types (`device_tests.jl`).
- **Every map promotes.** It evaluates in `promote_type(T, eltype(ξ))`,
  with derived constants (`π/(2n₁)`, `2Lc/n`, the grading's `e^α − 1`)
  formed in that type at the call, not stored rounded to `T`.
- **Glue in TreeAMR's form**: `(from = (p, d, side), to = (q, d′, side′),
  map = σ)`, `σ` a signed permutation in scatter form, both directions
  listed ("A multi-patch forest" in TreeAMR's `CODE.md`).

## Layouts

`PatchLayout{D,T}` is the supertype. A layout's patches are boxes
`[0, n₁] × … × [0, n_D]` of `n` roots each, the default extents, so a root
is the unit cube in every patch and every patch has the one cell size
TreeAMR requires. Each layout provides `layout_patches`, `layout_glue`,
the four `chart_*` methods, `chart_aligned` and the patch properties. The
layout is TreeAMR's atlas itself: no wrapper, nothing to keep in step.

### The forest and its check

`layout_forest([T,] layout; N, kwargs...)` is `Forest{T}(layout_patches(
layout); N, glue = layout_glue(layout), kwargs...)`: the forest's own
keywords (`N`, `reflecting`, `outflow`, `leaves`, `comm`), not a scheme's
`P` and `E`, so every application builds the same forest. `N` has no
default. The glue is the layout's, so `glue` is refused, and `extents`,
since the maps are written for the default boxes. The face kinds are the
caller's: on the annulus `reflecting = ((false, false), (true, true))`
makes the walls solid, which is what TreeWaveOLDG's `annulus_forest`
passed; a ball's unglued faces are its outer sphere's, and TreeAMR
refuses a reflecting kind there, since its face kinds are per
`(dim, side)` and the cube is glued on every face.

`check_layout(layout, forest)` refuses a forest that is not the layout's,
saying why: the dimension, the number of patches, a patch's root count or
box, or a face glued otherwise. TreeAMR 0.2 does not export `npatches`,
`patch_roots` or `patch_keywords`, so the check reads the forest through
what it does export: `forest.leaves` (documented as the tree), `patch`,
`block_extent` and `neighbor_keys`. For every leaf on a face of its patch
it asks for the neighbor across: none where the layout leaves the face
unglued, and otherwise the leaf at the image, under the layout's glue
(`glue_point`), of this leaf mirrored across the face, in the patch the
glue names. TreeAMR keeps a glued face conforming, so that neighbor is one
leaf of the same size, and its box is compared exactly. This sees which
patch and face lie across and where, at every level; a turn of the axes
within a single root is invisible at the root level, and shows on a
refined forest (a test refines TreeWaveOLDG's inward glue and is refused).
The forest's face kinds are not checked. The check reads only replicated
state, so it gives the same answer on every rank.

### The annulus

`PolarAnnulus(r0, r1, (n₁, n₂))`: four patches, the angle along dimension
1 and the radius along 2,

    θ = (p − 1)π/2 + θ′ξ₁,   θ′ = (π/2)/n₁,   r = r0 + r′ξ₂,   r′ = (r1 − r0)/n₂

and `x = r(cos θ, sin θ)`, so `√g = r θ′ r′`. The quarter turn is exact:
`(cos θ′ξ₁, sin θ′ξ₁)` turned by `p − 1` quarter turns by swapping and
negating, so no multiple of `π` is rounded, and the four patches are
images of one another bit for bit. Each patch's high face along 1 is glued
to the next one's low face by the identity. The patches are left-handed
(`det ∂x/∂ξ = −r θ′ r′`). The ring has no patch corner.

### The balls

`SevenPatchBall(Lc, R1, (n, m); grading = α)`: patch 1 the cube
`[−Lc, Lc]³` of `n³` roots, `x = (2Lc/n)ξ − Lc`; patches 2–7 the shells
over its faces `(k, s)` in the order `(1,−1), (1,1), (2,−1), (2,1), (3,−1),
(3,1)`, of `m` roots along `k` and `n` across. A shell blends the flat
face into the sphere: with `c_k = s`, `c_j = 2ξ_j/n − 1` and `f = Lc c`,

    η = ξ_k/m,   ρ = (e^{αη} − 1)/(e^α − 1) (ρ = η for α = 0),   x = (1 − ρ) f + ρ R1 f/|f|

`ThirteenPatchBall(Lc, Rm, R1, (n, m, m′))`: the seven-patch ball out to
`|x| = Rm`, ungraded (patches 1–7), and six outer shells over the same
faces (8–13), `x = (Rm + (R1 − Rm)ξ_k/m′) c/|c|`, whose `ξ_k` is a function
of `|x|` alone and whose angles of `x/|x|` alone. `R1 > √3 Lc`
(`Rm > √3 Lc` and `R1 > Rm`), so that the maps are one to one.

Every shell is `x = (A(ξ_k) + B(ξ_k)/|f|) f`: `A = 1 − ρ`, `B = Rρ` for a
blended shell, `A = 0`, `B = r` for an outer one. That one form gives the
closed-form inverse (`x` is parallel to `f`) and one routine for the
derivatives of every shell: with `G = A + Bq`, `q = 1/|f|`, `∂_j f = a e_j`
across `k` (`a = 2Lc/n`),

    ∂_e x = ∂_e G f + G ∂_e f
    ∂_e∂_b x = ∂_eb G f + ∂_e G ∂_b f + ∂_b G ∂_e f

with `∂_j q = −a f_j q³` and `∂_ij q = a²(3 f_i f_j q⁵ − δ_ij q³)` across
`k` (`shell_jacobian`, `shell_hessian`). The position keeps TreeWaveOLDG's
expression, `A f + (B/|f|) f`.

The glue (`layout_glue`): the cube's face `(k, s)` to the shell's inner
face, high to low by the identity for `s = 1` and low to low by the
reversal `σ[k] = −k` for `s = −1`; each inner shell's outer face to the
outer shell's inner face over the same cube face, high to low by the
identity; and within each layer the quarter turns, shell `(k, s)`'s face
`(l, side(t))` to shell `(l, t)`'s face `(k, side(s))` with `σ[k] = +l`
(radial onto radial, both outward), `σ[l] = −k` where both faces are on
the same side and `+k` otherwise, and the third axis onto itself. The
balls have patch corners at the cube's edges and corners, and the
thirteen-patch ball patch edges along the middle sphere.

### Shell orientation

Every shell's radial coordinate increases outward, including the three
over the cube's low faces, which in TreeWaveOLDG ran inward. So the outer
sphere is the high face along `k` of every outermost shell, and every
inner–outer glue is high to low; a boundary hook or an appendix of closed
ownership (TreeAMR's M20) treats all six shells alike. Within a shell the
tangential axes still run along `x_j` as the cube's do, so only `ξ_k`
flipped on those three shells: their frames are left-handed, which `√g =
|det J|` and the signed cofactors absorb ([The
frame](#the-frame-and-the-logical-speed)). The price is the cube–shell
glue on the low faces, now low to low with `k` reversed, a form TreeAMR's
glue table takes; keeping the frames right-handed instead would have
reversed a tangential axis on those shells and in their glue to the cube.

## The maps

`TreeAMR.chart_map(layout, ξ, p) -> x`, `TreeAMR.chart_jacobian -> J`,
`J[i][a] = ∂xⁱ/∂ξᵃ`, and `TreeAMR.chart_hessian -> H`, `H[i][a][b] =
∂²xⁱ/∂ξᵃ∂ξᵇ`, all as tuples. Both derivatives are written out by hand,
not taken by ForwardDiff inside the package: a device kernel then runs
plain arithmetic, the Hessian is exactly symmetric, and ForwardDiff is
what the tests check them against.

### Number types

A map evaluates in the promotion of the layout's `T` and `ξ`'s type, so
a `Float64` layout whose parameters are exactly representable (`Lc = 1`,
`R1 = 3`, …) gives the same map in `Float64x2` or `BigFloat` to that
type's accuracy, which is what extended-precision tables need; a layout
whose parameters are not exact in `T` is still one map, the one of its
rounded parameters. `PolarAnnulus{U}(annulus)` and its kind give the same
layout in `U`, `Float32` for a device. ForwardDiff differentiates every
map, at any of these types.

MultiFloats defines no `sin`, `cos` or `atan`, which the annulus needs.
The package calls its own `sincos_local` and `atan_local`, which are
Base's by default, and the MultiFloats extension gives MultiFloats'
numbers a Taylor series after reduction to the nearest multiple of `π/2`
and Newton's iteration from `Float64`'s angle; the ForwardDiff extension
applies the chain rule through `sincos_local` of the dual's value, since
ForwardDiff's own rule would call the `sin` MultiFloats lacks. Nothing of
Base is extended for a type this package does not own. `sin` and `cos`
are called apart rather than as `sincos`, whose `Float32` method reduces
its argument in `Float64`, which Metal does not have.

### Past the box

TreeAMR's interpolated ghosts sit at the point the target patch's map
names past its box ("Finite differences across a glue" in TreeAMR's
`CODE.md`), so every map continues past it: the annulus's `θ` and `r`
are the same lines continued, the shells' `c_j` past `±1` continue the
face's plane, `η < 0` reaches into the cube and `η > 1` past the sphere,
and the cube is affine. The tests take the derivatives, the frame and the
Piola identity half a root past every box. Two limits: the annulus's
continuation below its inner wall holds while `r > 0`, `r0/r′` roots,
and a blended shell's into the cube while `G > 0`; both are far beyond
any ghost region at the sizes applications run (the alignment test runs
three spacings past at 6 and 8 points per root).

### The frame and the logical speed

`patch_frame(layout, ξ, p) -> (√g, S)`, with `√g = |det J|` and the area
vectors `S[d] = √g ∇ξ^d`, `S[d][i] = √g ∂ξ^d/∂xⁱ`: the rows of
`sign(det J) · adj J`, in 3D `±(∂_e x × ∂_f x)` for `(d, e, f)` cyclic.
No division, so `Σ_d ∂_d S[d] = 0`, the Piola identity a flux-form scheme
relies on to keep a constant constant, holds to the rounding of the
products (measured below 1e-14 relative). The sign makes `S[d] ⋅ ∂_d x =
√g > 0` on the left-handed patches too. `volume_element` and
`area_vectors` are its two halves.

`logical_speed(layout, ξ, p)` is `√λ_max(g^{ab})`, `g^{ab} = S[a] ⋅ S[b]/g`,
the largest speed of light in logical coordinates; a scheme's step is
its logical one over the largest of these at its points, which is
TreeWaveOLDG's `oldg_timestep` rule. The eigenvalue is closed-form: in 2D
directly, in 3D by Smith's trigonometric formula, the diagonal case (the
cube) exactly. It needs `acos` and `cos`, so not MultiFloats, which a
time step does not need.

## The atlas

### The inverse map

`TreeAMR.chart_locate(layout, x) -> (ξ, p)` or `nothing`, in closed
form, with `ξ` clamped into `p`'s closed box as TreeAMR's contract asks:

- **The annulus.** The quadrant from the signs of `x` alone, `x` turned
  back by whole quarter turns, then `ξ₁ = atan(y′, x′)/θ′` and
  `ξ₂ = (r − r0)/r′`. A point on a glue goes to the patch whose low face it
  is.
- **The balls.** The cube if `max|x_k| ≤ Lc`. Otherwise `k` is the first
  dimension of largest `|x_k|` and `s` its sign, and since a shell's `x`
  is parallel to its `f`, `c = x/|x_k|` gives the tangential coordinates.
  A blended shell then has `|x_k| = G Lc`, so
  `ρ = (|x_k|/Lc − 1)/(R/(Lc|c|) − 1)` and `η` inverts the grading; an
  outer shell has `ρ = (|x| − Rm)/(R1 − Rm)`. On the thirteen-patch ball
  `|x| ≤ Rm` chooses the inner layer.

A point within eight roundings of an outer boundary (the annulus's walls,
the sphere) is on it; anything farther out is `nothing`, which TreeAMR
reads as beyond an outer face, the boundary hook's. A ghost across an
outer face lies a spacing or more out, far beyond that tolerance. On a face
between two patches either may be returned; the tests check that the
patch returned holds the `ξ` in its closed box and maps it back to `x`.

### Aligned coordinates

`chart_aligned(layout, p, q) -> NTuple{D,Bool}`: along which dimensions of
`q` a point at a node of `p`'s grid lands on a node of `q`'s grid at the
same level, so that TreeAMR's interpolation there has width one. The
contract is TreeAMR's (its `src/glue.jl` on the M20 branch). Declared:

- **Thirteen-patch ball**: between two outer shells, the radius of `q`,
  since `ξ_k = m′(|x| − Rm)/(R1 − Rm)` on all six; between an inner and an
  outer shell over the same face, both angles of `q`, since both layers'
  tangential coordinates are `n(c_j + 1)/2` of `x/|x|`.
- **Annulus**: both dimensions for every pair, since `θ` is linear
  through every glue and `r` the same function of `ξ₂` on every patch.
- **Seven-patch ball**: nothing between different patches; its maps kink
  at every glue.
- `p == q` is aligned along every dimension.

Where the TreeAMR in use defines `chart_aligned` (the M20 branch, a later
release), `chart_aligned` here *is* `TreeAMR.chart_aligned`, imported when
the package is compiled, and these are its methods, which TreeAMR's glue
schedule finds with `hasmethod`; with TreeAMR 0.2.10, which does not have
it, it is this package's own function of the same contract. A test checks
which of the two holds.

## Patch properties

- `affine_patch(layout, p)`: the cube on the balls; no patch of the
  annulus. `affine_map(layout, p) -> (origin, scale)` for it, `x = origin
  + scale ⋅ ξ`, the same expression as `chart_map`, bit for bit.
- `factored_volume(layout, p) -> k`, with `radial_volume(layout, ξ_k, p)`,
  `angular_volume(layout, ξ_⊥, p)` and `angular_class(layout, p)`: the
  thirteen-patch ball's outer shells, `√g = r² r′ ⋅ (2/n)²(1 + c_a² +
  c_b²)^{−3/2}`, one angular class; the annulus, `√g = (r r′) ⋅ θ′`, one
  class. `0` on the cube, whose `√g` is constant, and on the blended
  shells, whose `√g` does not factorize.
- `radial_dimension(layout, p)`: the shell's `k`, `0` for the cube, `2`
  on the annulus. With every radial coordinate outward, the outer
  boundary is the high face along it of every outermost shell.
- `outer_radius(layout)`: `R1`, or the annulus's `r1`.

## What stays in the applications

The scheme: TreeWaveOLDG's flux coefficients `A_d = S[d] ⋅ S[d]/√g` and
its per-dimension `jacobian_factor` (the annulus's are its
`angular_volume` and `radial_volume`), its truncation tables, its
Cartesian gradient set, its rule that only affine patches refine and the
refined-cube builder that follows from it; and every application's exact
solutions, boundary hooks and time-step driver.

## Departures from TreeWaveOLDG

TreeWaveOLDG's `src/geometry.jl` is the origin. What changed:

- **The shells over the cube's low faces run outward**, `η = ξ_k/m` on
  every shell where those had `1 − ξ_k/m` ([Shell
  orientation](#shell-orientation)). Their glue changes with it: the
  cube–shell glue on the low faces is low to low with `σ[k] = −k`, where
  it was low to high by the identity, and in the quarter turns
  `σ[k] = +l` always, where it was `±l` by `s == t`. The physical points
  are the same; on those three shells `ξ_k` is now `m − ξ_k`.
- **The builders take the forest's keywords and return the forest**:
  `layout_forest(layout; N, …)` in place of `annulus_forest(T, P, E; …)
  -> (forest, geometry)` and its kind; the layout is built first and
  passed. No `refined` keyword: refining the cube is TreeWaveOLDG's rule,
  and `leaves` passes through. The annulus's walls are not made
  reflecting by default.
- **The annulus's quarter turns are exact**, rotations of `(cos θ′ξ₁, sin
  θ′ξ₁)`, where TreeWaveOLDG took `cos((p − 1)π/2 + θ′ξ₁)` with `π/2`
  rounded; positions differ by rounding, the four patches are now
  bitwise images of one another. `θ′` is `T(π)/(2n₁)` formed at the call,
  equal to TreeWaveOLDG's rounded `BigFloat` value for `n₁` a power of two
  and within an ulp otherwise.
- **The frame is analytic**: the Jacobian written out and the area
  vectors its cofactors, where TreeWaveOLDG took ForwardDiff's Jacobian
  and `inv`. They agree to rounding; the Piola identity now holds to the
  rounding of the products.
- **One supertype**, `PatchLayout`, for TreeWaveOLDG's `Geometry` and
  `FullGeometry`, which are its scheme's distinction (a diagonal metric
  or not). `geometry_frame` is `patch_frame`, `physical_coordinates` is
  `TreeAMR.chart_map`, `check_geometry(geo, forest)` is
  `check_layout(layout, forest)`. `geometry_terms` and `jacobian_factor`
  are the scheme's and stay there.
- **The annulus declares its factorization** (`factored_volume = 2`),
  which TreeWaveOLDG's default left at 0; its diagonal path never read
  it.
- **Parameters are stored as given**, and constants derived from them
  formed at the call in the evaluation type ([Number
  types](#number-types)); TreeWaveOLDG's annulus stored `θ′` and `r′`
  rounded to `T`.

## Tests

`julia --project=. -e 'using Pkg; Pkg.test()'`, about 35 s, or one file
with `julia --project=test test/runtests.jl atlas_tests`. Each file runs
on its own after `test/helpers/layouts.jl`, which holds the layouts under
test (each at its default shape and at one whose roots differ along every
dimension, a graded seven-patch ball among them), random points in and
half a root past a box, and Gauss–Legendre quadrature.

| File | What it asserts |
|---|---|
| `map_tests.jl` | `isbits`; the Jacobian to 1e-14 and the Hessian to 1e-13 against ForwardDiff, half a root past every box; the frame against `|det J| J⁻¹`; `Σ_d ∂_d S[d] = 0` to 1e-14 of `∂S`; `Float64x2` against `BigFloat` to 1e-29, with ForwardDiff over `Float64x2`; the volume factors; `logical_speed` against `eigvals`; `affine_map`; outward radial axes; the constructors' refusals |
| `atlas_tests.jl` | `chart_locate` inverts `chart_map` inside, on faces, edges and corners, and past the boxes; a point just past a glued face is in the patch the glue names; outside points are `nothing`; every declared alignment to 1e-12 of a spacing at the ghosts three spacings past each box, at 6 and 8 points per root, vertex- and cell-centered, every declared pair reached; an undeclared coordinate off the nodes somewhere; the declarations named in TreeAMR's plan; `chart_aligned` is TreeAMR's where TreeAMR has it |
| `forest_tests.jl` | each forest builds, uniform and refined, and passes its check; keywords pass through, `glue` and `extents` refused; the check refuses other layouts, roots, a cut ring and TreeWaveOLDG's inward glue, at the root level and refined; a glued face is one surface under both maps; the outer boundary is the outermost shells' high faces |
| `volume_tests.jl` | `√g` integrates to `π(r1² − r0²)` and `4πR³/3` to 1e-13 and 1e-12, the cube to `(2Lc)³`, each layer's shells alike |
| `device_tests.jl` | every map, `chart_locate` included, in a KernelAbstractions kernel against the host; every map infers a concrete type in `Float32` |

The kernel test runs on Metal in `Float32` (to 1e-4 of the `Float64`
values) from an environment that has Metal, since the test environment
does not (a scratch environment that develops this checkout and adds
Metal and the test dependencies; stacking the global environment onto
`test` mixes KernelAbstractions versions):

```bash
TREEMULTIPATCH_TEST_BACKEND=metal julia --project=<env> test/runtests.jl device_tests
```

It passed on 2026-11-23 with Metal 1.12.1, all six layouts.

## Known limitations and open questions

- **The check cannot see a turn within a single root** at the root
  level, since it reads the glue through neighbors; TreeAMR exporting
  `patch_keywords` (with `npatches`, `patch_roots`, `patch_extents`)
  would let it compare the glue tables directly. **(open)**
- **`logical_speed` needs `acos`**, so it does not run in MultiFloats.
- **No equiangular shells**, no radial map on the outer shells other
  than linear in `r`, and no grading on the thirteen-patch ball's inner
  shells, as in TreeWaveOLDG.
- **No per-patch face kinds** in TreeAMR: with the outer sphere now
  every outermost shell's high face, the unglued faces of a ball are all
  `(p, k, 2)`, but the cube is glued on all six faces, so a per
  `(dim, side)` kind still cannot reach them.
