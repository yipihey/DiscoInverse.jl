# A cheap, statistically-sound uncertainty budget for a crude forward

Our forward model (nLPT + Lagrangian bias + fixed cosmology) carries **systematic error of order
tens of percent** on the scales it constrains. Estimating *statistical* error bars to finer precision
than that is meaningless — the model error dominates. So the goal is not an exact posterior; it is an
**honest, cheap, conservative** uncertainty at the tens-of-percent level. "Statistically sound" here
means three things, all achievable without a sampler:

1. **Conservative** — the bar is ≥ the true uncertainty (over-confidence is the only real sin).
2. **Honest about what is constrained** — report *which scales* the data informs, not fake per-voxel precision.
3. **Includes the systematic floor** — never claim to be tighter than the model's own bias.

Full posterior sampling (NUTS/BORG-style) is the wrong tool here: it costs ~1–2 h per converged chain
even at res 8, and it pins down a *statistical* width whose precision the model doesn't warrant. (This is
the same wall that limits Manticore to 15 samples at ~30M CPU-hours.)

## The recipe (a few forward evals + ~8 cheap re-MAPs — minutes, not hours)

Report three things, per |k|-band:

1. **r(k) — the primary UQ.** `cross_spectrum_r(ω̂, ω_ref)`: the correlation of the reconstruction with
   truth as a function of scale. This *is* the honest statement of trustworthiness — "faithful to r(k)=0.9
   down to k≈X, prior-dominated below." Nearly free (one FFT).
2. **Statistical width** — a small **parametric bootstrap** of the counts (resample n∼Poisson(n_obs),
   re-MAP), K≈8 draws. Its per-band std is the statistical error. Accuracy ~1/√K ≈ tens of percent, which
   is all the model supports — more draws are wasted effort. (Perturb-and-MAP / the linear Wiener width are
   equally valid, equally cheap alternatives.)
3. **Systematic floor** — re-reconstruct under alternative modelling choices and take the spread:
   - **LPT order** (2LPT vs 3LPT) — the gravity-model systematic.
   - **Bias amplitude** (b₁ ± ~20%) and **bias shape** (drop b₂,b_s² → linear).
   The reported bar is `max(statistical, systematic)` per band (or quadrature).

## Demonstrated numbers (res=8 pilot, true bias [1.8, 0.5, 0.3], data-dominated)

σ relative to each band's own fiducial amplitude:

| band | r(k) | statistical | LPT (2↔3) | drop b₂,b_s² | b₁ ±20% |
|---|---|---|---|---|---|
| 1 (largest) | **+0.91** | 0.50 | 0.15 | 0.02 | 0.08 |
| 2 | +0.56 | 0.89 | 0.30 | 0.01 | 0.08 |
| 3 | +0.25 | 1.01 | 0.37 | 0.01 | 0.15 |
| 4 (smallest) | −0.03 | 0.89 | 0.39 | 0.01 | 0.18 |

**Hierarchy on the constrained scales (r > 0.5): statistical (≥0.50) ≫ LPT (~0.15) ≈ b₁-amplitude (~0.08)
≫ b₂,b_s² shape (~0.02).** Reading:

- **Statistical uncertainty dominates every band** — even where the field is best reconstructed (r=0.91),
  the statistical width is 50% of the band amplitude. The reconstruction is prior-dominated except at the
  largest scales. ⇒ *lead with r(k).*
- **The nonlinear bias *shape* (b₂, b_s²) is negligible (1–2%)** at the quasi-linear scales we constrain —
  dropping it (our production default) costs almost nothing. What matters is the bias **amplitude** b₁
  (8–18%), because b₁ is degenerate with the field amplitude.
- **LPT order barely matters on constrained scales** (0.15, and it never changes r(k₁)) — "2LPT vs 3LPT
  makes little difference," now with a number. Its apparent growth to small scales is in the already-
  unconstrained (r<0) modes and is partly grid noise.

All terms are comfortably tens-of-percent — so a tens-of-percent error bar is not just adequate, it is the
*correct* precision. Finer would be false.

Caveat: these are res-8 pilot magnitudes; the **hierarchy** (statistical ≫ systematics; shape ≪ amplitude)
is the robust, transferable result. Absolute numbers shift with resolution, smoothing R, and tracer density.

## Usage

`examples/uncertainty_budget.jl` runs the full budget on a controlled `InferenceProblem` (the code that
produced the table above). For a real reconstruction: reuse the reconstruction as the fiducial, run the
bootstrap for the statistical width, and re-reconstruct at (n_order±1, b₁±20%, linear-vs-full-bias) for the
systematic floor. Report the per-band table alongside r(k). Do **not** reach for NUTS/SBC unless a specific
result genuinely hinges on the statistical width being known to better than ~30%.
