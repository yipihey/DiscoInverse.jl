# The fixed cubic mass cell: 16×10¹³ M⊙ per cube

The entire forward is anchored to **one** scale: a cubic tessellation cell holding a fixed **absolute**
Lagrangian mass — solar masses, **no little-h**, cosmology-independent. Bias (b₁/b₂/b_s²) and the
2LPT-vs-3LPT systematic are all defined relative to this single cell, so they are directly comparable
across surveys, resolutions, boxes, and cosmologies.

## The anchor

> **M = 16×10¹³ M⊙ exactly** (`ANCHOR_MASS = 1.6e14`), per cubic cell.

The mass is fixed; the cube's **side follows from the cosmology**, L = (M/ρ̄_m)^{1/3} with
ρ̄_m = Ω_m·ρ_crit,0 = Ω_m·2.775×10¹¹·h² M⊙/Mpc³. **Any cosmology change keeps the same 16×10¹³ M⊙**
and lets the side move. At Planck 2018 (Ω_m=0.3153, h=0.6736): **L = 15.91 physical Mpc = 10.72 Mpc/h** —
the mnemonic **"16 Mpc ↔ 16×10¹³ M⊙"** (exact to ~1.6%).

## The only filter in the chain is this cube

There is **no Gaussian anywhere**. The single filter in the whole forward is the **cubic top-hat that
matches the tessellation cell**:

  W(**k**) = ∏ᵢ sinc(kᵢ·L/2),   sinc(x) = sin(x)/x,

applied to the linear density and tidal fields in `bias_operators` (δ_L(k) = −k²·φ(k)·W(k)). The cube is
the sheet's native element — each Lagrangian cube tessellates into 6 tetrahedra — and the natural cell
for counting galaxies. So sheet cell = bias filter = counting cell = **one cube**.

## Sheet characterization (Planck 2018, grid = the cube)

Fraction of Lagrangian mass in sign-flipped (folded) tetrahedra (0% undisplaced):

| | shell-crossed 2LPT / 3LPT | 2LPT↔3LPT displacement |
|---|---|---|
| **z=1** | 0.10% / 0.15% | 1.83% |
| z=0 | 5.62% / 6.82% | 5.03% |

Conservative — essentially linear at z=1 (~0.15% shell-crossed), 2LPT vs 3LPT agree to ~1.8%. The sharp
cube keeps a little more near-Nyquist power than a Gaussian would, but well inside the tens-of-percent
regime the crude forward lives in. 16 phys Mpc is also a clean power-of-2 for grid layout.

## API

```julia
using DiscoInverse
c = fiducial_cosmology(Omega_m=0.3153, Omega_b=0.0493, h=0.6736, sigma8=0.8111, n_s=0.9649)  # Planck 2018
ANCHOR_MASS                       # 1.6e14 Msun (= 16e13, exact, cosmology-independent)
cube_side_from_mass(ANCHOR_MASS, c)  # 15.91 phys Mpc
anchor_cube_side(c)               # 10.72 Mpc/h  — the cubic-filter side to pass galaxy_model
cube_mass(L_phys, c)              # rho_m * L^3

L = anchor_cube_side(c)
gm = galaxy_model(res, res*L, c, pk; R=L, ...)   # R is the cubic-cell side; grid Δq = box/res = L
```

`galaxy_model`'s `R` argument is now the **cubic top-hat cell side** [Mpc/h] (not a Gaussian scale). For
the pure anchor, choose the box/res so **Δq = boxsize/res = anchor_cube_side(cosmo)** — then the grid cell
*is* the mass cube and the filter matches the tessellation exactly. The physical mass stays 16×10¹³ M⊙ at
every resolution/box and under any cosmology because it is the fixed primitive.
