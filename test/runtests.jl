# The test suite. Each file runs on its own after the helpers, so that a
# single one can be run as `julia --project=test test/runtests.jl
# atlas_tests`.

const TEST_FILES = ["map_tests", "atlas_tests", "forest_tests", "volume_tests",
                    "device_tests"]

include(joinpath(@__DIR__, "helpers", "layouts.jl"))

const SELECTED = isempty(ARGS) ? TEST_FILES : ARGS

@testset "TreeMultiPatch" begin
    for name in SELECTED
        @testset "$name" begin
            include(joinpath(@__DIR__, name * ".jl"))
        end
    end
end
