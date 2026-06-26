"""
Priors for the field-level inference.

The white-noise field carries a unit-variance i.i.d. Gaussian prior (the IC map and
nLPT encode P(k), so the latent ω is white) — the BORG-style field prior, which also
preconditions the optimizer.  Bias parameters get broad Gaussian priors.
"""

"""    gaussian_prior(ω) = ½‖ω‖²  (−log of a unit-variance white-noise prior)

The VALUE is accumulated in Float64 even for an F32 field (a custom rrule maps each term
to Float64 before the reduction) — at res=128 the ~2M-element sum would otherwise carry
~1e-4 relative F32 round-off, i.e. ~100 absolute on a ~1e6 prior, enough to corrupt the
NUTS ΔH.  The GRADIENT is the exact `ω` broadcast (no accumulation), kept in ω's own
precision so the GPU/Zygote adjoint stays simple — differentiating *through* a widening
map is what broke before, so we replace it with this rrule instead. """
gaussian_prior(ω::AbstractArray) = mapreduce(x -> Float64(abs2(x)), +, ω) / 2

function rrule(::typeof(gaussian_prior), ω::AbstractArray)
    v = gaussian_prior(ω)
    gaussian_prior_pullback(v̄) = (NoTangent(), v̄ .* ω)
    return v, gaussian_prior_pullback
end

"""    bias_prior(b, b0, σb) = ½‖(b−b0)/σb‖²"""
bias_prior(b, b0, σb) = sum(((b .- b0) ./ σb) .^ 2) / 2
