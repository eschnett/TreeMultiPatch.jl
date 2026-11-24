# `sin`, `cos` and `atan` for MultiFloats' numbers, which MultiFloats does
# not define: the annulus's map and its inverse need them, and the maps
# are generic in the number type ("The maps" in CODE.md). Each is used
# where TreeMultiPatch's own `sincos_local` and `atan_local` are, so that
# nothing of Base is extended for a type this package does not own.
module TreeMultiPatchMultiFloatsExt

using MultiFloats: MultiFloat
import TreeMultiPatch: sincos_local, atan_local, quarter_turn

# The Taylor series of `sin` and `cos` about 0, to the number type's
# precision, for `|t| ≤ π/4`.
function series_sincos(t::F) where {F}
    t2 = t * t
    s, c = t, one(F)
    ts, tc = t, one(F)
    for i in 1:60
        ts *= -t2 / ((2i) * (2i + 1))
        tc *= -t2 / ((2i - 1) * (2i))
        s += ts
        c += tc
        abs(ts) <= eps(F) * abs(s) && abs(tc) <= eps(F) * abs(c) && break
    end
    return s, c
end

# `t` reduced by the nearest multiple `k` of `π/2`, the series there, and
# the result turned back by `k` quarter turns exactly.
function sincos_local(x::F) where {F<:MultiFloat}
    k = round(Int, Float64(x) / (π / 2))
    s, c = series_sincos(x - k * (F(π) / 2))
    c, s = quarter_turn(c, s, mod(k, 4))
    return s, c
end

# Newton's iteration on `y cos θ - x sin θ = 0` from `Float64`'s angle,
# each step doubling the digits.
function atan_local(y::F, x::F) where {F<:MultiFloat{Float64}}
    θ = F(atan(Float64(y), Float64(x)))
    for _ in 1:3
        s, c = sincos_local(θ)
        θ += (y * c - x * s) / (x * c + y * s)
    end
    return θ
end

end # module TreeMultiPatchMultiFloatsExt
