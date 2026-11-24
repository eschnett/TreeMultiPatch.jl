# The maps inside a KernelAbstractions kernel: every function a ghost
# fill, a coefficient set or a time step evaluates on a device, against
# the same functions on the host. The suite runs it on the CPU; with
# `TREEMULTIPATCH_TEST_BACKEND=metal` and Metal on the load path, in
# `Float32` on Metal (CODE.md, "Tests"):
#
#     TREEMULTIPATCH_TEST_BACKEND=metal JULIA_LOAD_PATH="@:@v#.#:@stdlib" \
#         julia --project=test test/runtests.jl device_tests

using KernelAbstractions: @kernel, @index, CPU, allocate, synchronize, get_backend
import KernelAbstractions

const DEVICE = get(ENV, "TREEMULTIPATCH_TEST_BACKEND", "cpu")

if DEVICE == "metal"
    using Metal: MetalBackend
    const BACKEND = MetalBackend()
    const DEVICE_T = Float32
else
    DEVICE == "cpu" || error("TREEMULTIPATCH_TEST_BACKEND is $DEVICE, and the tests " *
                             "know cpu and metal")
    const BACKEND = CPU()
    const DEVICE_T = Float64
end

# A nested tuple as one flat tuple, by recursion that compiles away.
@inline tflat(x::Number) = (x,)
@inline tflat(::Tuple{}) = ()
@inline tflat(t::Tuple) = (tflat(first(t))..., tflat(Base.tail(t))...)

# Everything at one point: the position, the Jacobian, the Hessian, the
# frame, the logical speed, and the patch and position `chart_locate`
# returns for the position, NaN where it returns nothing.
@inline function everything(layout, ξ, p)
    T = eltype(ξ)
    x = TreeAMR.chart_map(layout, ξ, p)
    loc = TreeAMR.chart_locate(layout, x)
    located = loc === nothing ? (T(NaN), ntuple(_ -> T(NaN), length(ξ))...) :
              (T(loc[2]), loc[1]...)
    return (tflat(x)..., tflat(TreeAMR.chart_jacobian(layout, ξ, p))...,
            tflat(TreeAMR.chart_hessian(layout, ξ, p))...,
            tflat(patch_frame(layout, ξ, p))..., logical_speed(layout, ξ, p),
            located...)
end

@kernel function everything_kernel!(out, layout, ξs, ps)
    j = @index(Global, Linear)
    v = everything(layout, ξs[j], ps[j])
    for i in 1:length(v)
        out[i, j] = v[i]
    end
end

@testset "The maps run in a kernel on $DEVICE: $name" for (name, layout) in LAYOUTS
    # A map that allocates, reads a global, or calls a function the device
    # does not compile (`sincos`'s Float64 reduction on Metal), or a layout
    # that is not isbits.
    T = DEVICE_T
    L = typeof(layout).name.wrapper{T}(layout)
    @test isbits(L)
    rng = Xoshiro(31)
    pts = [(random_inside(rng, layout, p), p) for p in 1:npatches(layout) for _ in 1:16]
    ξs = [ntuple(d -> T(ξ[d]), dimension(layout)) for (ξ, _) in pts]
    ps = Int32[p for (_, p) in pts]
    K = length(everything(L, ξs[1], 1))
    dξs = allocate(BACKEND, eltype(ξs), length(ξs))
    copyto!(dξs, ξs)
    dps = allocate(BACKEND, Int32, length(ps))
    copyto!(dps, ps)
    out = allocate(BACKEND, T, K, length(ξs))
    everything_kernel!(BACKEND, 64)(out, L, dξs, dps; ndrange=length(ξs))
    synchronize(BACKEND)
    got = Array(out)
    # The host's values, in the device's type and in Float64.
    host = reduce(hcat, [collect(everything(L, ξs[j], Int(ps[j]))) for j in eachindex(ξs)])
    ref = reduce(hcat, [collect(everything(layout, Float64.(ξs[j]), Int(ps[j])))
                        for j in eachindex(ξs)])
    @test all(isfinite, got)
    # Every point is located in its own patch, at its own position.
    D = dimension(layout)
    @test got[K - D, :] == T.(ps)
    @test maximum(abs.(got[(K - D + 1):K, :] .- reduce(hcat, collect.(ξs)))) <=
          100 * eps(T) * maximum(patch_roots(layout, 1))
    scale = max.(1, abs.(ref))
    if T == Float64
        @test maximum(abs.(got .- host) ./ scale) <= 4eps(T)
    else
        # A device's own `sin`, `sqrt` and `exp` round differently from the
        # host's; what matters is that they are the map to Float32's
        # accuracy.
        @test maximum(abs.(got .- ref) ./ scale) <= 1e-4
    end
end

@testset "Every map infers a concrete type in Float32: $name" for (name, layout) in LAYOUTS
    # A boxed variable or a run-time tuple index, which the CPU runs and a
    # device kernel refuses to compile ("unsupported dynamic function
    # invocation" on Metal); checked here so that the suite sees it
    # without a device.
    L = typeof(layout).name.wrapper{Float32}(layout)
    D = dimension(layout)
    ξ = NTuple{D,Float32}
    for f in (TreeAMR.chart_map, TreeAMR.chart_jacobian, TreeAMR.chart_hessian,
              patch_frame, volume_element, area_vectors, logical_speed)
        @test isconcretetype(only(Base.return_types(f, (typeof(L), ξ, Int32))))
    end
    loc = only(Base.return_types(TreeAMR.chart_locate, (typeof(L), ξ)))
    @test loc == Union{Nothing,Tuple{ξ,Int}}
end
