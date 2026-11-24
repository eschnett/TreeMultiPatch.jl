# TreeMultiPatch.jl

`TreeMultiPatch` holds the multi-patch layouts that applications of
[TreeAMR](https://github.com/eschnett/TreeAMR.jl) share: a domain cut into
logical boxes, *patches*, glued at their faces, with the maps from each
box to physical space and the atlas that inverts them. TreeAMR's
coordinates stay logical; the maps live here. There is no physics in this
package.

**Contents**

- [What it provides](#what-it-provides)
- [Installation](#installation)
- [Example](#example)
- [Status](#status)
- [Design](#design)

## What it provides

Three layouts:

- `PolarAnnulus(r0, r1, (n₁, n₂))`: the annulus `r0 ≤ r ≤ r1` as a ring of
  four patches in polar coordinates;
- `SevenPatchBall(Lc, R1, (n, m); grading)`: the ball `|x| ≤ R1` as a cube
  and six shells that blend its faces into the sphere;
- `ThirteenPatchBall(Lc, Rm, R1, (n, m, m′))`: the seven-patch ball out to
  `|x| = Rm` and six shells around it whose radial coordinate is the
  radius.

For each, the forest (`layout_forest`) and the check that a forest is that
layout (`check_layout`); the map, its Jacobian and second derivatives
(TreeAMR's `chart_map`, `chart_jacobian`, `chart_hessian`), `√g` and the
area vectors `S[d] = √g ∇ξ^d` (`patch_frame`), and the largest logical
speed of light (`logical_speed`); the inverse map (`chart_locate`) and the
coordinates that coincide across a glue (`chart_aligned`); and the patch
properties a scheme asks for (`affine_patch`, `affine_map`,
`factored_volume`, `radial_dimension`, `outer_radius`). A layout is
`isbits` and is TreeAMR's atlas itself. The maps are generic in the number
type (Float32 on a device, `Float64x2`, `BigFloat`, ForwardDiff's duals)
and defined past each patch's box, where a ghost across a glue sits.

## Installation

The package is not registered. From a checkout:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

## Example

```julia
using TreeAMR, TreeMultiPatch

ball = ThirteenPatchBall(1, 2, 3, (4, 2, 2))
forest = layout_forest(ball; N = 16)
check_layout(ball, forest)

ξ, p = (1.5, 0.5, 2.0), 9                 # a point of an outer shell
x = TreeAMR.chart_map(ball, ξ, p)         # its physical position
J = TreeAMR.chart_jacobian(ball, ξ, p)    # J[i][a] = ∂xⁱ/∂ξᵃ
√g, S = patch_frame(ball, ξ, p)
TreeAMR.chart_locate(ball, x)             # (ξ, 9), to rounding
chart_aligned(ball, 9, 11)                # (false, true, false): the radius
```

## Status

Version 0.1, unregistered: step 1 of TreeAMR's milestone M20. The layouts
moved here from TreeWaveOLDG's `src/geometry.jl`, with the balls' shells
reoriented so that every radial coordinate runs outward. TreeWaveOLDG and
TreeWave do not use it yet. See [PLAN.md](PLAN.md).

## Design

[CODE.md](CODE.md) is the design: the layouts, the maps, the atlas, the
tests and why each is the way it is. [HISTORY.md](HISTORY.md) has how the
package came to be.
