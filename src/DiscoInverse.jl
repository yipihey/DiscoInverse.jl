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
using ChainRulesCore: @ignore_derivatives, NoTangent
import ChainRulesCore: rrule
using Random

# Re-export the DiscoDJNative differentiable forward primitives this package builds on.
export white_noise_to_fphi, ic_operator, nlpt_kernels, compute_core_exact, compute_core,
       lpt_displacement, lagrangian_grid_3d, sheet_deposit, cic_deposit,
       lightcone_cross_ad, exact_shape_stack, Cosmology, linear_power_spectrum,
       comoving_distance, growth_D1

# Move a forward model to the GPU (implemented in the CUDA extension).
"""    gpu(gm::GalaxyModel) -> GalaxyModel on the device (requires `using CUDA`)"""
gpu(::Any) = error("DiscoInverse.gpu requires the CUDA extension — run `using CUDA` first.")
export gpu

# ── Forward: 2nd-order Lagrangian bias ────────────────────────────────────────
export BiasOperators, bias_operators, bias_fields, bias_weight, bias_moments
include("forward/bias.jl")

# ── Forward: galaxy field on the lightcone ────────────────────────────────────
export GalaxyModel, galaxy_model, galaxy_density
include("forward/galaxy_field.jl")

# ── ECHOES data IO ────────────────────────────────────────────────────────────
export EchoesCatalog, load_echoes_realization, load_echoes_randoms, n_realizations, prov_mask, prov_weights
include("data/echoes_io.jl")
export QuaiaCatalog, load_quaia, load_quaia_randoms
include("data/quaia_io.jl")

# ── Geometry: wedge → comoving box ────────────────────────────────────────────
export fiducial_cosmology, radec_z_to_cartesian, BoxGeometry, box_geometry, embed, embed_radec_z
include("geometry/embedding.jl")

# ── Survey window ─────────────────────────────────────────────────────────────
export survey_window, bin_galaxies
include("window/window.jl")

# ── Likelihood + prior + loss ─────────────────────────────────────────────────
export poisson_nll, overdispersed_nll, gaussian_prior, bias_prior, redshift_prior
include("likelihood/poisson.jl")
include("likelihood/prior.jl")
export PhotozMixture, calibrate_photoz, radial_posterior_ensemble, coverage_pit
include("likelihood/photoz.jl")
export InferenceProblem, inference_problem, inference_problem_overdispersed,
       galaxy_model_for, prov_weights, model_density, loss
include("likelihood/loss.jl")

# ── Inference driver ──────────────────────────────────────────────────────────
export map_optimize, map_optimize_alternating, adam_optimize, lbfgs_optimize, progressive_optimize
export upsample_white_noise
include("infer/driver.jl")
export hmc_sample
include("infer/hmc.jl")
export nuts_sample, nuts_chains, rhat, ess
include("infer/nuts.jl")

# ── Validation: mock injection + recovery diagnostics ─────────────────────────
export inject_mock, model_lambda, overall_correlation, cross_spectrum_r
include("validation/mock.jl")
include("validation/diagnostics.jl")

# ── Grid-free sheet-on-lightcone forward + point-process likelihood (P4) ───────
export galaxy_density_sheet, galaxy_density_sheet_c0, galaxy_density_sheet_c0_masked, SheetProblem, sheet_problem, inject_mock_sheet
include("forward/sheet_field.jl")

# ── Fixed-amplitude (Angulo–Pontzen) phase parametrization — Stage-1 MAP ───────
export phase_field, phase_loss, phase_map_optimize, cosmic_variance_b1
include("infer/fixed_amplitude.jl")

# ── Quaia field-level redshift reconstruction (χ-parametrized forward + ensemble) ──
export QuaiaProblem, quaia_problem, quaia_positions, sky_directions, reconstruct_quaia, quaia_ensemble
include("forward/quaia.jl")

# ── Quaia-constrained periodic IC box (coarse-constrain → fine-realize refinement) ──
export constrained_ic_box, constrained_ic_box_ensemble, refine_phases, export_white_noise, ic_box_snapshot
include("forward/constrained_box.jl")

# ── Multi-tracer joint field reconstruction (Quaia + DESI + BOSS + eBOSS + …) ──
export Tracer, tracer, MultiTracerProblem, multitracer_problem, multitracer_phase_loss, reconstruct_joint_field
export LensingConstraint, lensing_constraint, kappa_map
export VelocityConstraint, velocity_constraint, radial_velocity
include("forward/multitracer.jl")

# ── Cosmicflows-4 IO + peculiar-velocity error model ──────────────────────────
export CF4Catalog, load_cf4_groups, cf4_hubble, cf4_peculiar_velocity, cf4_box_geometry, cf4_velocity_constraint
include("data/cf4_io.jl")

# ── Perturb-and-MAP constrained realizations (Gaussian-ω posterior; velocity/lensing) ──
export wiener_mean, constrained_realizations
include("infer/constrained_realizations.jl")

end # module
