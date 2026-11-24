# `sin` and `cos` of a dual number through TreeMultiPatch's own
# `sincos_local` of its value, so that ForwardDiff differentiates the
# annulus's map over any number type, MultiFloats' included, which
# ForwardDiff's own rule would hand to a `sin` MultiFloats does not have.
module TreeMultiPatchForwardDiffExt

using ForwardDiff: Dual, partials, value
import TreeMultiPatch: sincos_local

function sincos_local(d::Dual{T}) where {T}
    s, c = sincos_local(value(d))
    return Dual{T}(s, c * partials(d)), Dual{T}(c, -s * partials(d))
end

end # module TreeMultiPatchForwardDiffExt
