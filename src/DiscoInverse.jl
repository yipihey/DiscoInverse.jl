"""
DiscoInverse.jl — field-level inference of cosmological initial conditions and
galaxy-bias parameters from the ECHOES completed CMASS-South catalogs.

The differentiable forward model (built on DiscoDJNative) maps a white-noise field ω
and bias parameters b to a model galaxy density on the past lightcone:

    ω → φ(k) → nLPT ψ shapes → {δ_L, s²}  →  w(q) = 1 + b₁δ_L + (b₂/2)(δ_L²−σ²) + b_s2(s²−⟨s²⟩)
      → lightcone crossing x_obs(q) (+RSD)
      → tetrahedral CDM-sheet deposit of mass·w  →  δ_g(x)
      → Poisson likelihood vs the data (PROV=0 hard, PROV≥1 soft) + ½‖ω‖² prior.

Inference optimizes ω (and b) by gradient descent through this chain (Zygote).
"""
module DiscoInverse

using DiscoDJNative
using FFTW
using LinearAlgebra
using Statistics: mean
using ChainRulesCore: @ignore_derivatives
using Random

# Re-export the DiscoDJNative differentiable forward primitives this package builds on.
export white_noise_to_fphi, ic_operator, nlpt_kernels, compute_core_exact, compute_core,
       lpt_displacement, lagrangian_grid_3d, sheet_deposit, cic_deposit,
       lightcone_cross_ad, exact_shape_stack, Cosmology, linear_power_spectrum,
       comoving_distance, growth_D1

# ── Forward: 2nd-order Lagrangian bias ────────────────────────────────────────
export BiasOperators, bias_operators, bias_fields, bias_weight, bias_moments
include("forward/bias.jl")

# ── Forward: galaxy field on the lightcone ────────────────────────────────────
export GalaxyModel, galaxy_model, galaxy_density
include("forward/galaxy_field.jl")

end # module
