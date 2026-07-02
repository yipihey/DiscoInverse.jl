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

"""    redshift_prior(χ, χ_obs, σχ) = ½ Σ ((χ_obs−χ)/σχ)²

Photometric-redshift Gaussian likelihood as a prior on the per-quasar **comoving radial
distance** χ (Quaia field-level reconstruction).  Parametrizing by χ rather than z keeps
the embedding `x = χ·n̂ + shift` LINEAR in the free parameter — differentiating z→χ through
`comoving_distance` (an interpolation) inside the forward segfaults Zygote — so χ_obs and the
χ-space width σχ = χ(z_obs+σ_z) − χ(z_obs) are precomputed constants and the prior is a plain
quadratic.  The custom rrule returns the exact analytic gradient (χ−χ_obs)/σχ² (no tape through
the broadcast), with the value accumulated in Float64 for sampler-ΔH robustness — mirroring
[`gaussian_prior`](@ref).  Gradient w.r.t. χ only; χ_obs, σχ are data."""
redshift_prior(χ, χ_obs, σχ) = sum(x -> Float64(abs2(x)), (χ_obs .- χ) ./ σχ) / 2

function rrule(::typeof(redshift_prior), χ, χ_obs, σχ)
    v = redshift_prior(χ, χ_obs, σχ)
    redshift_prior_pullback(v̄) = (NoTangent(), v̄ .* (χ .- χ_obs) ./ σχ .^ 2, NoTangent(), NoTangent())
    return v, redshift_prior_pullback
end
