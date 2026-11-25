# TreeMultiPatch.jl — Plan

**Contents**

- [Milestones](#milestones)
  - [M1 — The layouts, their maps and atlases](#m1--the-layouts-their-maps-and-atlases)
  - [M2 — TreeWaveOLDG and TreeWave on TreeMultiPatch](#m2--treewaveoldg-and-treewave-on-treemultipatch)
  - [M3 — Release](#m3--release)

The package is TreeAMR's M20 step 1; its milestones follow that plan
(TreeAMR's `PLAN.md`, "M20 — TreeMultiPatch, and finite differences on
patches").

## Milestones

### M1 — The layouts, their maps and atlases

Done 2026-11-23; the design is in [CODE.md](CODE.md), the record in
[HISTORY.md](HISTORY.md#m1--the-layouts-their-maps-and-atlases).

### M2 — TreeWaveOLDG and TreeWave on TreeMultiPatch

TreeAMR's M20 steps 2, 3 and 5, in those packages. What this package
owes them, if they ask:

- whatever TreeWaveOLDG's port turns up missing: the port replaces
  `annulus_forest(T, P, E; …)` and its kind by `layout_forest(layout;
  N = E(P+1), …)`, `geometry_frame` by `patch_frame`,
  `physical_coordinates` by `TreeAMR.chart_map`, and recomputes its
  flux coefficients and `jacobian_factor` from the frame and the volume
  factors; the three shells over the cube's low faces now run outward
  (CODE.md, "Departures from TreeWaveOLDG").

*Done when* TreeWaveOLDG's convergence tables and energies on the
annulus and both balls are what they were, and TreeWave's annulus and
seven-patch ball run on these layouts.

### M3 — Release

Register TreeMultiPatch (Erik's step). Its `[compat] TreeAMR` is already
`"0.2.12"`, the release that declares `chart_aligned`
([HISTORY.md](HISTORY.md#the-atlas)).

*Done when* TreeMultiPatch is registered and TreeWaveOLDG and TreeWave
resolve it.
