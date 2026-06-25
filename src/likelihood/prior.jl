"""
Priors for the field-level inference.

The white-noise field carries a unit-variance i.i.d. Gaussian prior (the IC map and
nLPT encode P(k), so the latent ω is white) — the BORG-style field prior, which also
preconditions the optimizer.  Bias parameters get broad Gaussian priors.
"""

"""    gaussian_prior(ω) = ½‖ω‖²  (−log of a unit-variance white-noise prior)"""
gaussian_prior(ω::AbstractArray) = sum(abs2, ω) / 2

"""    bias_prior(b, b0, σb) = ½‖(b−b0)/σb‖²"""
bias_prior(b, b0, σb) = sum(((b .- b0) ./ σb) .^ 2) / 2
