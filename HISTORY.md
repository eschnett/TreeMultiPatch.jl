# TreeMultiPatch.jl — History

**Contents**

- [Origin](#origin)
- [Layouts](#layouts)
- [The maps](#the-maps)
- [The atlas](#the-atlas)
- [Milestones](#milestones)
  - [M1 — The layouts, their maps and atlases](#m1--the-layouts-their-maps-and-atlases)

## Origin

Planned in TreeAMR's M20 (proposed 2026-11-22, its decisions taken
2026-11-23 by Erik): the patch systems move out of TreeWaveOLDG's
`src/geometry.jl` into a package of their own, so that every application
on them shares one copy, TreeAMR keeps logical coordinates only, and the
maps' dependencies stay out of TreeAMR. Created 2026-11-23 as M20's
step 1.

## Layouts

**Shell orientation** (Erik, M20's plan). TreeWaveOLDG's shells over the
cube's low faces ran inward, which made every cube–shell glue the
identity. M20 wanted the outer sphere to be every shell's high face, for
the boundary hook and for the appendix of closed ownership, so all six
shells now run outward. Of the two ways to do it, flipping `ξ_k` alone
(left-handed frames on three shells, the cube–shell glue on the low faces
low to low with `k` reversed) and flipping a tangential axis too (all
frames right-handed, a tangential reversal in the glue to the cube), the
first was taken: the tangential coordinates stay those of the cube's
face on all six shells, so the thirteen-patch ball's inner and outer
layers share them with no reversal, and `√g = |det J|` absorbs the sign.

**The forest check through exports only.** TreeWaveOLDG compared
`TreeAMR.patch_keywords(forest).glue` with its own glue list; that and
`npatches`, `patch_roots`, `patch_extents` are TreeAMR internals in 0.2,
and the package takes TreeAMR's exports only. The check walks the leaves
and their neighbors across each patch face instead. What that cannot see
is recorded under "Known limitations" in `CODE.md`.

**The builders.** TreeWaveOLDG's `annulus_forest(T, P, E; …)` built
`N = E(P + 1)` and returned the forest with its geometry, and refined the
balls' cubes under its affine-only rule. Here the layout is built first,
and `layout_forest` takes the forest's keywords, since `P` and `E` are
the scheme's and so is the refinement rule.

## The maps

**Analytic derivatives.** TreeWaveOLDG formed the balls' frame from
ForwardDiff's Jacobian and `inv`. The derivatives here are written out,
for both shell kinds at once through `x = (A + B/|f|) f`: no dual numbers
in a device kernel, an exactly symmetric Hessian (one rounding in
`3 f_e f_b` was found by the test and ordered away), and area vectors as
cofactors, with no division, so that the Piola identity holds to the
products' rounding.

**Trigonometry for MultiFloats.** The annulus's map needs `sin` and
`cos`, which MultiFloats 3.3 does not define (`sin`, `cos`, `atan`,
`expm1`, `log1p` are missing; `sqrt`, `exp`, `log` are there). Extending
Base for MultiFloats' type would be piracy, so the package calls its own
`sincos_local`, which the extensions specialize. The graded shells keep
TreeWaveOLDG's `exp(αη) − 1` rather than `expm1` for the same reason.

**A device kernel's types.** The first Metal run compiled the annulus and
refused both balls with "unsupported dynamic function invocation": the
balls' `chart_locate` rebound `x` inside a closure that read it, which
boxes it, and set the radial coordinate with `Base.setindex` at a
run-time index, which infers a tuple of `Any`. Both run on the CPU. The
suite now asserts concrete inferred types for every map.

## The atlas

**`chart_aligned` before TreeAMR has it.** TreeAMR's `chart_aligned` is
on its M20 branch, not in 0.2.10. The package defines its own function of
that contract, and imports TreeAMR's instead when `isdefined(TreeAMR,
:chart_aligned)` at compile time, so that against the branch, and
against the release that has it, the methods are TreeAMR's with no
change here.

## Milestones

### M1 — The layouts, their maps and atlases

2026-11-23. `PolarAnnulus`, `SevenPatchBall` and `ThirteenPatchBall` with
their glue, `layout_forest`, `check_layout`, the four `chart_*` methods,
`chart_aligned`, the frame, `logical_speed` and the patch properties; the
shells reoriented outward. Every item of M20 step 1's *Done when* holds:
the derivatives agree with ForwardDiff's half a root past every box (the
Jacobian to 1e-14, the Hessian to 1e-13, relative); `Σ_d ∂_d S[d]` is
below 1e-14 of `∂S`; `chart_locate` inverts `chart_map` inside, on faces,
edges and corners, and past the boxes; every declared alignment holds to
1e-12 of a spacing at the ghosts three spacings past each box; `√g`
integrates to each volume to 1e-12; each forest builds, uniform and
refined, and passes its check; the maps run in a kernel on the CPU and,
in `Float32`, on Metal (Metal 1.12.1). The suite takes about 35 s.
