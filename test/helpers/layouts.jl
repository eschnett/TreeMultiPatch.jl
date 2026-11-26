# What every test file shares: the layouts under test, random points in
# and past a patch's box, the glue's point map, and Gauss–Legendre
# quadrature. Every test file runs on its own after this one.

using TreeAMR
using TreeMultiPatch
using TreeMultiPatch: glue_point, npatches, patch_roots
using ForwardDiff
using LinearAlgebra: det, eigvals, inv, Symmetric
using MultiFloats: Float64x2
using Random
using StaticArrays: SVector, SMatrix
using Test

const chart_map = TreeAMR.chart_map
const chart_jacobian = TreeAMR.chart_jacobian
const chart_hessian = TreeAMR.chart_hessian
const chart_locate = TreeAMR.chart_locate

# Each layout at its default shape and at one whose roots differ along
# every dimension, a graded seven-patch ball among them, so that a
# confusion of `n` and `m` or of the shells' dimensions shows.
const LAYOUTS = [
    "annulus (2, 1)" => PolarAnnulus(1, 2, (2, 1)),
    "annulus (3, 2)" => PolarAnnulus(0.5, 2.25, (3, 2)),
    "seven-patch (2, 2)" => SevenPatchBall(1, 3, (2, 2)),
    "seven-patch (3, 2), graded" => SevenPatchBall(1, 2.5, (3, 2); grading=0.7),
    "thirteen-patch (2, 1, 1)" => ThirteenPatchBall(1, 2, 3, (2, 1, 1)),
    "thirteen-patch (4, 2, 3)" => ThirteenPatchBall(0.75, 2, 4, (4, 2, 3)),
]

dimension(::PatchLayout{D}) where {D} = D

# How far past its box a map must hold: half a root, four spacings at
# `N = 8` and more than any ghost region at the sizes the applications run.
const MARGIN = 0.5

# A random point of patch `p`'s box widened by `margin` roots on every side.
random_point(rng, layout, p; margin=MARGIN) =
    ntuple(d -> -margin + (patch_roots(layout, p)[d] + 2margin) * rand(rng),
           dimension(layout))

# A random point strictly inside patch `p`'s box, a tenth of a root from
# its faces.
random_inside(rng, layout, p) = random_point(rng, layout, p; margin=-0.1)

# The largest entry of a nested tuple or array, for scale-aware tolerances.
maxabs(x::Number) = abs(x)
maxabs(x) = maximum(maxabs, x)
flat(x::Number) = [x]
flat(x) = reduce(vcat, map(flat, collect(x)))

# Nodes and weights of `n`-point Gauss–Legendre quadrature on `[-1, 1]`,
# by Newton's iteration on the Legendre polynomial.
function gauss_legendre(n)
    x, w = zeros(n), zeros(n)
    # The polynomial and its derivative at `z`, in BigFloat so that the
    # weights are right to the last bit.
    function legendre(z)
        p0, p1 = one(z), z
        for k in 2:n
            p0, p1 = p1, ((2k - 1) * z * p1 - (k - 1) * p0) / k
        end
        return p1, n * (z * p1 - p0) / (z^2 - 1)
    end
    for i in 1:n
        z = big(cos(π * (i - 0.25) / (n + 0.5)))
        for _ in 1:100
            p1, dp = legendre(z)
            δ = p1 / dp
            z -= δ
            abs(δ) < 1e-40 && break
        end
        dp = legendre(z)[2]
        x[i], w[i] = Float64(z), Float64(2 / ((1 - z^2) * dp^2))
    end
    return x, w
end

# `∫ √g dξ` over patch `p`, `nq` Gauss points per root and dimension.
function patch_volume(layout, p, nq)
    D = dimension(layout)
    x, w = gauss_legendre(nq)
    n = patch_roots(layout, p)
    # Pairwise, a root at a time, so that the sum's rounding stays below
    # what the test asserts.
    return sum(vec(map(CartesianIndices(ntuple(d -> 0:(n[d] - 1), D))) do root
        sum(vec(map(CartesianIndices(ntuple(_ -> nq, D))) do q
            ξ = ntuple(d -> root[d] + (x[q[d]] + 1) / 2, D)
            prod(d -> w[q[d]] / 2, 1:D) * volume_element(layout, ξ, p)
        end))
    end))
end
