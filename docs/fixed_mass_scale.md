# The fixed Lagrangian mass scale: M = 1×10¹⁴ M⊙

To make bias and LPT-order comparisons interpretable *across surveys, resolutions, and boxes*, the
sheet's smallest resolved scale is anchored to a single **absolute Lagrangian mass** — in solar masses,
**no little-h**, so it is cosmology-independent. All analysis (2LPT vs 3LPT, bias parameters b₁/b₂/b_s²)
is defined relative to this one physical scale.

## The anchor

**M = 1×10¹⁴ M⊙** (exact), Planck 2018 base-ΛCDM (Ω_m=0.3153, Ω_b=0.0493, h=0.6736, σ₈=0.8111,
n_s=0.9649; ρ̄_m,0 = 3.9705×10¹⁰ M⊙/Mpc³).

The radius depends on the filter (same mass, different shape):

| filter | M = | radius (phys Mpc) | radius (Mpc/h) |
|---|---|---|---|
| **top-hat** | (4π/3)·ρ̄·R³ | **8.440** | 5.685 |
| **Gaussian** (used by the code's `W_R`) | (2π)^{3/2}·ρ̄·R³ | **5.428** | 3.656 |
| cubic cell (side) | ρ̄·L³ | 13.606 | 9.166 |

R_TH/R_G = 1.555. The forward's bias smoothing is a **Gaussian** filter, so the value fed to
`galaxy_model` is the Gaussian **R = 3.656 Mpc/h**; the top-hat 8.44 Mpc is the same mass in the
Press-Schechter/halo convention.

## Why this scale (sheet-measured, Planck 2018, grid-converged)

Measured directly in our tetrahedral phase-space sheet (fraction of Lagrangian mass in sign-flipped /
folded tetrahedra; 0% undisplaced, grid-converged res 128→320):

| | shell-crossed 2LPT | shell-crossed 3LPT | 2LPT↔3LPT displacement |
|---|---|---|---|
| **z=1** | 0.07% | 0.13% | **1.33%** |
| z=0 | 5.55% | 6.34% | 3.64% |

At z=1 the anchor is **nearly linear** (~0.1% of the mass has shell-crossed) and 2LPT vs 3LPT agree
to ~1.3% on the displacement — a scale where the forward model is well-converged. For reference, the
"3% of the sheet mass shell-crossed at z=1" criterion falls at ~3 Mpc (M≈1.7×10¹³ M⊙), and the "2LPT/3LPT
agree to 1% on the displacement" criterion at ~7 Mpc; **1×10¹⁴ M⊙ (5.43 Mpc Gaussian) sits between them**
and is the chosen round anchor.

## API

```julia
using DiscoInverse
c = fiducial_cosmology(Omega_m=0.3153, Omega_b=0.0493, h=0.6736, sigma8=0.8111, n_s=0.9649)  # Planck 2018
R  = smoothing_from_mass(ANCHOR_MASS, c)     # 3.656 Mpc/h  (Gaussian, for galaxy_model)
M  = mass_from_smoothing(R, c)               # 1e14  (inverse)
Rt = tophat_radius_from_mass(ANCHOR_MASS, c) # 8.440 phys Mpc  (top-hat / halo radius)

gm = galaxy_model(res, boxsize, c, pk; R=R, ...)   # bias defined at the fixed 1e14 scale
```

`ANCHOR_MASS = 1e14`. Use a grid fine enough to resolve the smoothing: **Δq = boxsize/res ≲ R/1.8 ≈
2 Mpc/h**. The physical mass stays fixed at every resolution/box because R is derived from the mass and
cosmology, not from the grid.
