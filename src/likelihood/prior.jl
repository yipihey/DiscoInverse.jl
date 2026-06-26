"""
Priors for the field-level inference.

The white-noise field carries a unit-variance i.i.d. Gaussian prior (the IC map and
nLPT encode P(k), so the latent ω is white) — the BORG-style field prior, which also
preconditions the optimizer.  Bias parameters get broad Gaussian priors.
"""

"""    gaussian_prior(ω) = ½‖ω‖²  (−log of a unit-variance white-noise prior)

Accumulated in Float64 even for an F32 field: this enters the HMC/NUTS Hamiltonian, and an
F32 sum over millions of cells has ~O(1) round-off that would swamp the O(1) ΔH used for
accept/reject (the reason naive-F32 sampling fails to mix)."""
gaussian_prior(ω::AbstractArray) = sum(x -> Float64(abs2(x)), ω) / 2

"""    bias_prior(b, b0, σb) = ½‖(b−b0)/σb‖²"""
bias_prior(b, b0, σb) = sum(((b .- b0) ./ σb) .^ 2) / 2
