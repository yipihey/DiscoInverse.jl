# DiscoInverse.jl

**Field-level Bayesian inference of cosmological initial conditions from galaxy, quasar, and CMB-lensing
data** — a differentiable, GPU-capable inverse problem in pure Julia, built on
[DiscoDJ.jl](https://github.com/yipihey/DiscoDJ.jl).

Given a survey, DiscoInverse reconstructs the primordial white-noise field ω whose ΛCDM evolution
reproduces the observed large-scale structure — a BORG/constrained-realization-style inverse of the
differentiable forward model. The reconstructed field is a **constrained realization**: consistent with
the data within its errors, ΛCDM-consistent elsewhere, and tightening as more data are folded in. Its main
data product, Quaia-constrained initial conditions for simulations, is released at
[**QuaiaICs**](https://github.com/yipihey/QuaiaICs).

## Installation

```julia
import Pkg
Pkg.add(url="https://github.com/yipihey/DiscoInverse.jl")
```

DiscoInverse builds on **DiscoDJNative**, the differentiable ΛCDM engine (IC operator, nLPT, lightcone,
tetrahedral CDM-sheet), which lives inside the DiscoDJ.jl repository at
[`native/DiscoDJNative`](https://github.com/yipihey/DiscoDJ.jl/tree/main/native/DiscoDJNative). The
package's `[sources]` resolves it automatically from there — nothing to set up by hand. GPU acceleration
is a package extension: just `using CUDA`.

For local co-development of the engine, override the git source with your checkout:
`Pkg.develop(path="../DiscoDJ.jl/native/DiscoDJNative")`.

## What it does

The differentiable forward maps the white-noise field to observables:

```
ω → √P(k)·ω → nLPT displacements → past-lightcone → tetrahedral CDM-sheet density ρ(x)
                                                       ├─ bias-weighted → galaxy/quasar number density
                                                       └─ unbiased, LOS-integrated → CMB-lensing κ(n̂)
```

Everything is Zygote-differentiable end-to-end (FD-validated), so ω (and bias, and per-object redshifts)
can be optimized or sampled to match the data. Capabilities:

- **Constrained IC boxes** — fixed-P(k) white noise constrained by a survey in the box centre, random
  elsewhere; **coarse-constrain → fine-realize** so a box realizes at *any* resolution (1024³, 2048³, …)
  far beyond the forward's memory ceiling. (`constrained_ic_box`, `refine_phases`, `export_white_noise`)
- **Multi-tracer joint field** — one field constrained jointly by many surveys, each with its own bias and
  window (Quaia + DESI + BOSS + eBOSS …). (`MultiTracerProblem`, `reconstruct_joint_field`)
- **CMB-lensing convergence constraint** — an all-sky, *unbiased* total-matter, line-of-sight constraint
  integrated natively through the sheet tessellation; pins the field where there are no tracers.
  (`LensingConstraint`, `kappa_map`)
- **Field-level redshift reconstruction + calibrated redshift posteriors** — resolve photo-z along the
  line of sight; calibrate against spectroscopy and fold spec-z in as hard constraints.
  (`reconstruct_quaia`, `calibrate_photoz`, `radial_posterior_ensemble`, `coverage_pit`)

## Key entry points

| task | functions |
|---|---|
| forward density / bias | `galaxy_model`, `galaxy_density`, `SheetProblem` |
| constrained IC box | `box_geometry`, `constrained_ic_box`, `refine_phases`, `export_white_noise` |
| multi-tracer joint field | `tracer`, `multitracer_problem`, `reconstruct_joint_field` |
| CMB lensing | `lensing_constraint`, `kappa_map` |
| Quaia redshifts | `quaia_problem`, `reconstruct_quaia`, `calibrate_photoz`, `radial_posterior_ensemble` |
| inference | `map_optimize`, `phase_map_optimize`, `hmc_sample`, `nuts_sample` |

## Built on

- **[DiscoDJ.jl](https://github.com/yipihey/DiscoDJ.jl)** — the differentiable ΛCDM forward (native-Julia
  DISCO-DJ port; `native/DiscoDJNative`). DiscoInverse's tetrahedral CDM-sheet estimator is the
  phase-space-sheet method of Abel, Hahn & Kaehler (2012).
- **[MUSIC](https://github.com/cosmo-sims/MUSIC)** (Hahn & Abel 2011) — the multi-scale white-noise
  paradigm behind coarse-constrain → fine-realize.

## Tests

`julia --project=. test/runtests.jl` — 70 tests covering the forward, geometry, windows, the constrained
IC box, multi-tracer joint reconstruction, CMB-lensing, and the redshift calibration/coverage.
