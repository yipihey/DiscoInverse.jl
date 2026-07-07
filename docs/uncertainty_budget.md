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

## Production numbers (res=192, L=768, R=8 Mpc ≈ 2 voxels, GPU F32, mean of 2 realizations)

σ relative to each band's own fiducial amplitude (`scratch/ubudget_res192.npz`, figure
`scratch/uncertainty_budget_res192.png`). Reconstruction is faithful (r>0.9) out to **k ≈ 0.73 h/Mpc**;
beyond that the field is prior-dominated and the ratios diverge (÷ vanishing amplitude) — not meaningful.

| k [h/Mpc] | r(k) | statistical | LPT (2↔3) | b₂,b_s² shape | **b₁ ±20%** |
|---|---|---|---|---|---|
| 0.05 | 0.97 | 0.10 | 0.15 | 0.10 | **0.20** |
| 0.15 | 0.94 | 0.05 | 0.07 | 0.04 | **0.14** |
| 0.24 | 0.97 | 0.06 | 0.10 | 0.04 | **0.29** |
| 0.34 | 0.98 | 0.06 | 0.11 | 0.04 | **0.36** |
| 0.44 | 0.98 | 0.06 | 0.13 | 0.04 | **0.38** |
| 0.53 | 0.96 | 0.07 | 0.14 | 0.05 | **0.39** |
| 0.63 | 0.95 | 0.09 | 0.17 | 0.07 | **0.41** |
| 0.73 | 0.95 | 0.12 | 0.22 | 0.11 | **0.44** |

**Hierarchy on trustworthy scales (r > 0.9, k ≲ 0.73): b₁-amplitude (0.14–0.44) ≫ LPT (0.07–0.22) >
statistical (0.05–0.12) ≈ b₂,b_s²-shape (0.04–0.11).**

- **The bias *amplitude* (b₁) is the dominant uncertainty by far — 14→44%**, growing with k, swamping
  everything else. This is the b₁–field-amplitude degeneracy: the data constrains b₁·δ, so a b₁ error maps
  almost directly onto δ. ⇒ **pinning/marginalizing b₁ (jointly with the field amplitude) is the single most
  important thing for a reliable reconstruction** — more than any of the other terms combined.
- **LPT order (2↔3) is the *second* systematic — 7–22%, growing toward smaller scales**, and is **not**
  negligible at production resolution. (A res-96 pass gave 5–12%; the res-8 pilot gave ~2% — the LPT
  systematic *grows with resolution* as the 3LPT correction resolves. Note the 3LPT variant was warm-started
  from the 2LPT solution and lightly iterated, so these are a lower bound.) So "2LPT vs 3LPT makes little
  difference" holds only at coarse resolution; at production res it is the runner-up term.
- **The reconstruction is statistically *tight* where constrained** — stat is 5–12%. Report r(k) as the map
  of *where* it's trustworthy; the statistical bar is small there.
- **Nonlinear bias *shape* (b₂,b_s²) is small on constrained scales (4–11%)** — comparable to statistical,
  subdominant to both b₁-amplitude and LPT order.

All constrained-scale terms are tens-of-percent, so a tens-of-percent error bar is the *correct* precision.
Two realizations agree, so this hierarchy is robust — the term that *sets* the budget is **b₁**, with **LPT
order second**; the statistical width and nonlinear-bias shape are the least of the worries.

### ⚠ Why res-8 is not enough (a cautionary result)

An earlier res-8 / R=40 pilot gave the **opposite** conclusion — "statistical dominates (≥0.50), bias shape
negligible (0.02)." Both were artifacts: res-8 is prior-starved (few constrained modes → inflated
statistical width), and R=40 (box/10) over-smoothed the density so the nonlinear-bias terms vanished. The
lesson: **this budget must be run at production resolution with R ≈ 2 voxels** — a small-box pilot inverts
the hierarchy and is actively misleading. The res-96 numbers above are the ones to trust; higher res would
refine but not change the b₁-dominated conclusion.

## Performance

At res-8 CPU a single MAP was ~80 s; a res-96 budget (≈13 reconstructions) on CPU would be *hours*. The
enabling move is **GPU + Float32**: the whole budget runs in **~4–5 min** (fiducial 24 s steady after
compile; warm variants ~10–20 s each; K=8 bootstrap ~80 s). Levers, in order of impact:
- **GPU + F32** — the decisive one (makes production res feasible at all).
- **Warm-start every variant/bootstrap from the fiducial ω** — they are perturbations, so ~8–10 L-BFGS iters
  suffice instead of 45–60 (used here; could be cut further).
- Build the device `GalaxyModel` **once** and reuse it (was rebuilt per reconstruction) — minor here
  (`gpu(gm)` is ~0.3 s) but tidy; the real per-iter cost is the sheet forward+gradient (~0.4–0.5 s/iter at
  res 96, dominated by the L-BFGS line search doing a few evals/iter). 3LPT variants cost ~2× the 2LPT ones.
- K need only be ~6–8 for a tens-of-% statistical width; more is wasted.

## Usage

`examples/uncertainty_budget.jl` runs the full budget on a controlled `InferenceProblem` (the code that
produced the table above). For a real reconstruction: reuse the reconstruction as the fiducial, run the
bootstrap for the statistical width, and re-reconstruct at (n_order±1, b₁±20%, linear-vs-full-bias) for the
systematic floor. Report the per-band table alongside r(k). Do **not** reach for NUTS/SBC unless a specific
result genuinely hinges on the statistical width being known to better than ~30%.
