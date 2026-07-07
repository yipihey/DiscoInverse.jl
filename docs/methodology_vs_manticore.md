# Our approach vs Manticore/BORG: compare, contrast, and what to adopt

A methodological comparison of our field-level reconstruction (DiscoInverse.jl + DiscoDJNative) against
the Manticore project (McAlpine et al. 2025, [Manticore I](https://arxiv.org/abs/2505.10682); [Manticore II](https://arxiv.org/abs/2606.10020)),
which builds on [BORG](https://arxiv.org/abs/1806.11117). We describe our own code as "a
BORG/constrained-realization-style inverse of a differentiable forward model," so this is a comparison
*within one family* — the differences are design choices along shared axes, not different paradigms.

## The one-paragraph contrast

**Manticore** spends **~30 million CPU-hours** to draw a genuine nonlinear posterior over the initial
white-noise field with a **COLA** forward at 4 Mpc/h and a **calibrated nonlinear (STDP) bias +
generalized-Poisson** likelihood, then **resimulates** ~15–50 posterior IC fields with **full N-body
(SWIFT, up to 4096³)** to deliver a complete halo-level digital twin. It buys **fidelity, halo products,
and a real posterior** — at enormous cost, and with **tiling and per-tile-lightcone approximations forced
by that cost**. **We** spend **GPU-hours** on a differentiable **nLPT + tetrahedral-sheet MAP** with an
**exact continuous per-particle lightcone** and **native lensing/velocity constraints**, buying **speed,
exact geometry, a superior field-level density/velocity estimator, and multi-probe constraints** — at the
cost of **nonlinear fidelity, halo catalogs, and (currently) a fully sampled posterior**. They are the
heavy/high-fidelity/complete-twin end of the BORG family; we are the light/differentiable/multi-probe/
exact-geometry end.

## Axis-by-axis

| axis | Manticore / BORG | Ours (DiscoInverse + DiscoDJNative) |
|---|---|---|
| **Gravity (inference)** | **COLA** (2LPT + PM residual), differentiable, ~4 Mpc/h, redshift space | **nLPT** orders 1–3, differentiable (Zygote), ~384³ ceiling, real or redshift space |
| **Gravity (products)** | **full N-body (SWIFT)**, 2LPT ICs at z=69, 4096³ | none — nLPT only (no PM/N-body) |
| **Density estimator** | PM / CIC on the grid; halos from N-body | **tetrahedral phase-space sheet (AHK)** — caustic-resolving, shot-noise-free, exact multistream + deformation eigenvalues |
| **Bias model** | **STDP**: 5-param sigmoid-truncated *double* power-law + ergodic-mean renorm; per luminosity×z×flux subcatalogue (32 in I; per-tile in II) | 2nd-order Lagrangian (McDonald–Roy: b₁,b₂,b_s²) available, but **linear b₁ in production**; one bias per tracer |
| **Noise model** | **generalized Poisson**, density-dependent overdispersion | Poisson point-process / counts-in-cells; `overdispersed_nll` exists (completion variance) |
| **Selection** | full Aquila: angular completeness + spectroscopic + **Schechter radial selection** | survey **window from randoms** (geometry); no luminosity-dependent or explicit radial selection |
| **RSD / lightcone** | RSD in forward; **lightcone approximate** — each tile at a *single fixed* scale factor | RSD available; **exact continuous per-particle lightcone** (`lightcone_cross_ad`, a_cross per particle) |
| **Inference** | **HMC-within-Gibbs** (HMC field \| slice-sample bias); genuine nonlinear posterior | **fixed-amplitude MAP + perturb-and-MAP** realizations (primary); NUTS/HMC exists, used for low-S/N CF4 |
| **Uncertainty** | 50 (I) / 15 (II) posterior samples | Wiener mean + per-voxel std from perturb-and-MAP (exact only for near-linear model); ensemble over free phases |
| **Cosmology** | **fixed** (DES Y3) | **fixed** (fiducial) — *equal; neither marginalizes* |
| **P(k)** | fixed (CLASS), field amplitudes sampled | **fixed, and exactly enforced** (Angulo–Pontzen fixed-amplitude) |
| **Scale strategy** | **64-tile** decomposition of (4 Gpc/h)³ @ 4 Mpc/h (single box for the 1 Gpc local) | **single global** coarse constraint (≤256³) → **fine spectral refinement** to 1024³+ (tape-free) |
| **Constrained modes** | 4 Mpc/h *everywhere*, but tiling **drops k ≲ 0.006 h/Mpc** (largest coherent modes) | **largest modes coherent** (global box), but small scales random **beyond the coarse Nyquist** |
| **Multi-probe** | galaxy counts constrain the field; **PV/lensing used only to validate** | **lensing (κ ray-march) and CF4 velocities are constraints in the likelihood** (native, differentiable) |
| **Products** | N-body snapshots, **halo/subhalo catalogs** (HBT+/SOAP), velocity fields, IC fields, HEALPix shells | constrained ω + manifest, LPT density/velocity, **sheet density + web-classifier eigenvalues**, environment catalogs, reconstructed redshifts |
| **Validation** | posterior-predictive P(k)/bispectrum/HMF (~1%), **CMB-lensing×δ 7.4σ, kSZ 3.5σ**, PV Bayesian evidence (beats CSiBORG2), cluster counterparts | FD gradient checks, injection-recovery, `cross_spectrum_r`, held-out spec-z PIT/coverage, **DES/κ prediction**, **cross-recon vs Manticore (r=0.76)** |
| **Compute** | **~30M CPU-hours** (II inference) + 6.5×10⁵/resim → only 15 samples | GPU-hours (3–4 orders cheaper) → many realizations feasible |

## Where Manticore is genuinely ahead — and what we should adopt

**Tier 1 — high value, and we already have most of the parts:**

1. **Nonlinear bias + overdispersed likelihood.** Their STDP double-power-law bias + density-dependent
   generalized-Poisson variance captures the real galaxy–density relation (high-density super-Poisson
   scatter, low-density suppression) that our production **linear b₁** misses. **We have the machinery**:
   `bias.jl` already implements the McDonald–Roy b₂/b_s² basis (unused in production), and
   `overdispersed_nll` already exists. *Actions:* (a) turn on b₂/b_s² in production fits; (b) add a
   sigmoid/exponential truncation option to `bias_weight` (small change); (c) generalize `overdispersed_nll`
   to a density-dependent variance. Low effort, direct fidelity gain.

2. **N-body resimulation → halo catalogs → a complete digital twin.** This is our biggest *product* gap.
   Manticore resimulates posterior ICs with SWIFT and extracts HBT+/SOAP halos, HMFs, lightcone shells.
   **We already have the tooling on this machine** (`~/codes/music`, `~/codes/gadget4`, and
   `export_white_noise` writes MUSIC/N-GenIC-ready ω). *Action:* pipe a few constrained ω fields through
   MUSIC/monofonic → GADGET-4/SWIFT → a halo finder. This converts our IC fields into a halo-level twin
   and unlocks HMF validation + HOD mocks. Medium effort, high value.

3. **Close the posterior-honesty gap.** Their HMC-within-Gibbs gives a genuine nonlinear posterior;
   our released default is fixed-amplitude MAP + perturb-and-MAP (exact only for the near-linear velocity
   model — our own notes flag it as "over-confident / factorized"). **We have NUTS** (`infer/nuts.jl`) but
   shelved it (stiff identity-mass geometry). *Actions:* adopt their **Gibbs structure** — HMC on the field
   with the unit-mass white-basis preconditioner we already have, **slice-sample the bias in a separate
   block** (decouples the stiff bias–field coupling that collapses joint HMC) — and/or **validate
   perturb-and-MAP against a real HMC posterior** on a controlled case, so we know when the Gaussian
   approximation is safe. Medium–high effort; the biggest rigor upgrade.

**Tier 2 — matches their validation/data rigor:**

4. **A posterior-predictive + external-cross-correlation validation suite.** Add z=0 P(k)/bispectrum and
   HMF checks (HMF needs #2), and a **kSZ cross-correlation** — we produce velocity fields from the sheet,
   so kSZ is a natural, currently-unused validation (their II reports 3.5σ). Our **DES/κ work is already the
   same class of test as their CMB-lensing 7.4σ** — we should frame and grow it that way.

5. **Luminosity/flux sub-catalogue bias + explicit radial selection.** They split bias by luminosity×z×flux
   with a Schechter radial selection; we fit one bias per tracer from a random-based window. Our value-added
   catalogs already carry magnitudes, so luminosity-binned bias is available to us.

**Tier 3 — deeper, larger effort:**

6. **A differentiable COLA/PM forward option.** Their inference gravity is COLA, not LPT — better nonlinear
   structure inside the loop. DISCO-DJ upstream *has* a PM solver; our native port is LPT-only. Porting the
   differentiable PM would let us match their forward fidelity while keeping AD + GPU. Biggest effort, but
   the principled path beyond LPT.

## Where our approach is genuinely ahead (keep these)

1. **Exact continuous per-particle lightcone.** `lightcone_cross_ad` places each particle at its own crossing
   scale factor. Manticore II's **own stated limitation #1** is that it evolves each tile to a *single fixed*
   scale factor and "assumes minimal evolution across a 1024 h⁻¹ Mpc tile." For a z≈0.7 BOSS-volume
   reconstruction, we solve exactly what they approximate.

2. **Global large-scale coherence vs their tiling.** Manticore II's 64-tile split **drops the largest
   coherent modes (k ≲ 0.006 h/Mpc)** — their stated limitation #2, with sCOLA boundary fixes as future
   work. Our coarse-global-constrain → fine-realize keeps the largest modes coherent by construction. (The
   symmetric cost: *our* small scales beyond the coarse Nyquist are random, where *their* 4 Mpc/h is
   constrained everywhere. The two methods sacrifice opposite ends of the spectrum.)

3. **The tetrahedral phase-space sheet** is a strictly better *field-level* density/velocity estimator than
   PM/CIC — caustic-resolving, shot-noise-free, with exact multistream densities and deformation-tensor
   eigenvalues (cosmic-web classification) for free. (They recover true nonlinear densities, but only via
   full N-body at resimulation.)

4. **Native multi-probe constraints.** We fold **CMB/galaxy lensing (κ) and CF4 peculiar velocities directly
   into the likelihood** via AD; Manticore constrains on galaxy counts and uses lensing/PV only to *validate*.
   Constraining on lensing + velocities is a genuine capability difference (with the usual care about
   non-circularity).

5. **Cost and cadence.** We are 3–4 orders of magnitude cheaper. Manticore's 15 samples and single-chain-
   per-tile convergence are **explicitly forced by the 30M-CPU-hour budget**; our GPU-hour MAP/perturb-and-MAP
   makes large ensembles and rapid iteration feasible.

6. **Exactly-fixed P(k)** (Angulo–Pontzen) — useful for seeding sims without sample-variance scatter; they
   sample field amplitudes.

## Honest caveats on this comparison

Not recovered from the Manticore papers' rendered text (their Appendix A / references hold them): the **COLA
time-step count**, **HMC mass-matrix / leapfrog settings**, the **exact STDP + generalized-Poisson
equations and prior ranges**, and total chain lengths. The gravity-fidelity and posterior-rigor gaps are
real regardless of those details. And two "gaps" are actually **ties**: neither project marginalizes
cosmology, and both fix P(k).

## Suggested near-term program (cheap → expensive)

1. Turn on **b₂/b_s² + density-dependent overdispersion** in production (parts already exist). *(days)*
2. **Resimulate** ~3 constrained ω through MUSIC→GADGET-4 → halo catalog + HMF check (tooling in `~/codes`). *(week)*
3. **kSZ + posterior-predictive** validation on the resulting twin. *(week, after 2)*
4. **HMC-vs-perturb-and-MAP** validation on a controlled case; if the gap is real, adopt Gibbs-blocked HMC. *(weeks)*
5. Research: **differentiable PM** forward option. *(months)*
