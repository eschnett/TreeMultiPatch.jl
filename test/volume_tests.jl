# The volume element integrated over each layout: Gauss–Legendre
# quadrature per root, which the maps' smoothness inside a patch makes
# converge to rounding.

@testset "√g integrates to each layout's volume: $name" for (name, layout) in LAYOUTS
    # A volume element that is wrong by a factor (a spacing, a sign, a
    # Jacobian of the old orientation's coordinates) on some patch, which
    # a weighted norm would carry into every measured error.
    vols = [patch_volume(layout, p, 12) for p in 1:npatches(layout)]
    if layout isa PolarAnnulus
        @test sum(vols) ≈ π * (layout.r1^2 - layout.r0^2) rtol = 1e-13
        @test all(v -> v ≈ sum(vols) / 4, vols)
    else
        @test sum(vols) ≈ 4π * layout.R1^3 / 3 rtol = 1e-12
        @test vols[1] ≈ (2layout.Lc)^3 rtol = 1e-13
        # The six shells of a layer are one another's images.
        @test all(v -> v ≈ vols[2], vols[2:7])
        if layout isa ThirteenPatchBall
            @test sum(vols[1:7]) ≈ 4π * layout.Rm^3 / 3 rtol = 1e-12
            @test all(v -> v ≈ vols[8], vols[8:13])
        end
    end
end
