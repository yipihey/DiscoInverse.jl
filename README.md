# DiscoInverse.jl

Field-level inference of cosmological initial conditions and galaxy-bias parameters
from the [ECHOES](../ECHOES) completed CMASS-South catalogs, using the differentiable
LPT forward model in [DiscoDJNative](../DiscoDJ.jl/native/DiscoDJNative).

A BORG-style inverse problem, pure Julia: optimise a white-noise field ω (the initial
conditions) and bias parameters b so the forward model reproduces the observed
galaxies — truly observed galaxies (PROV=0) weighted most, completed points (PROV≥1)
soft.

## Differentiable forward model

```
ω  → φ(k)                                   white_noise_to_fphi   (DiscoDJNative)
   → ψ exact-growth shapes                  compute_core_exact
   → {δ_L, s²}                              bias_fields           (forward/bias.jl)
   → w(q)=1+b₁δ_L+(b₂/2)(δ_L²−σ²)+b_s2(s²−⟨s²⟩)   bias_weight
   → per-particle lightcone crossing x_obs  lightcone_cross_ad    (DiscoDJNative, IFT)
   → tetrahedral CDM-sheet deposit of mass·w → n_g(x)   sheet_deposit (Abel+ 2012)
```

`galaxy_density(gm, ω, b)` runs the whole chain; it is Zygote-differentiable w.r.t. ω
and b (validated against finite differences: ∂/∂ω ~1e-5, ∂/∂b ~1e-12).

## Status

- **P0** (in DiscoDJNative): differentiable tetrahedral CDM-sheet + CIC deposit;
  differentiable lightcone crossing (implicit-function theorem). ✓
- **P1**: 2nd-order Lagrangian bias (`forward/bias.jl`); galaxy-field forward
  (`forward/galaxy_field.jl`). ✓  ECHOES geometry/window/IO — in progress.
- **P2**: Poisson likelihood + ½‖ω‖² prior; MAP/HMC inference; injection-recovery. — todo.

See the plan in `~/.claude/plans/`.
