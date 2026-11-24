"""
    TreeMultiPatch

Multi-patch layouts for [TreeAMR](https://github.com/eschnett/TreeAMR.jl)'s
forests of patches: the four-patch annulus, the seven-patch ball and the
thirteen-patch ball, each with its roots and glue, its maps from the
patches' logical coordinates to physical space, and its atlas. TreeAMR
keeps its coordinates logical; the maps, their Jacobians, metrics and
inverses live here, so that every application on a patch system shares
them. There is no physics in this package.

A layout is an `isbits` value that is also TreeAMR's atlas: it has methods
of `TreeAMR.chart_locate`, `chart_map`, `chart_jacobian` and
`chart_hessian`, and of [`chart_aligned`](@ref). The maps are written out
by hand, generic in the number type, and defined past each patch's box.

The interface common to all layouts is in `layout.jl`, the annulus in
`annulus.jl`, the two balls in `balls.jl`. See `CODE.md` in the package
root for the design.
"""
module TreeMultiPatch

using TreeAMR
using TreeAMR: Forest, block_extent, neighbor_keys, patch

# The layouts
export PatchLayout, PolarAnnulus, SevenPatchBall, ThirteenPatchBall

# Forests
export layout_forest, check_layout, layout_patches, layout_glue

# The maps
export patch_frame, volume_element, area_vectors, logical_speed

# Patch properties
export affine_patch, affine_map, factored_volume, radial_volume, angular_volume,
       angular_class, radial_dimension, outer_radius

# The atlas
export chart_aligned

include("layout.jl")
include("annulus.jl")
include("balls.jl")

end # module TreeMultiPatch
