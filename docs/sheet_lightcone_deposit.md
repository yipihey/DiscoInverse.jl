# Grid-free sheet-on-lightcone density estimator — design spec

**Status:** design / not yet implemented. This is the strongly-enabling core that makes
the DiscoInverse pipeline unique: field-level inference evaluated directly on the
Abel–Hahn–Kaehler (AHK) phase-space sheet, on the past lightcone, with **no Eulerian
grid**. It removes the memory wall (the grid + CIC deposit was the ~14 GB AD-tape hog),
is physically more faithful (exact multi-stream density, adaptive resolution, no CIC
smoothing), and replaces counts-in-cells with a resolution-free point-process likelihood
matched to the survey's (footprint × redshift) wedge.

## 0. Why this works (the two enabling facts)

1. **The trajectory is analytic.** nLPT gives `x(q,a) = q + Σ_k D_k(a)·ψ_k(q)` in closed
   form, and the lightcone crossing `a_cross(q)` is a single detached Newton step
   (`lightcone_cross_ad`). So the AD tape of the *trajectory* is ≈0 (measured), and the
   deposit is a **single geometric pass** — no time loop, no crossing detection, no
   double-counting.
2. **The lightcone is radially foliated.** Adjacent Lagrangian vertices cross within ~one
   grid cell of each other, so each tetrahedron spans a **thin, known redshift interval**.
   The 3-D point-location collapses to a radial sort (built once on the fixed galaxies) +
   a small angular test.

Everything upstream — `white_noise_to_fphi` → `compute_core_exact` → `bias_fields`/
`bias_weight` → `lightcone_cross_ad` (giving per-vertex `x_obs`, `a_cross`, `v_r`) — is
**unchanged and already FD-validated**. This spec replaces only `sheet_deposit` + the
counts-in-cells likelihood.

## 1. The mathematical object

AHK tessellates Lagrangian space into 6 tetrahedra per cube (Kuhn decomposition, already
in `_TETS`/`_CUBE_CORNERS`). Each tet's 4 vertices are advected to their lightcone
crossing positions `y_1..y_4 = x_obs(q_i, a_cross_i)` (in redshift space — `x_obs`
already carries the RSD shift). The **lightcone tet** is the simplex spanned by those 4
crossing positions.

Bias-weighted matter (galaxy) density at an Eulerian/redshift point `x`:

    ρ_g(x) = Σ_{T : x ∈ T}  m_T · w_T / |V_T|

- `V_T = (1/6)·det[y_2−y_1, y_3−y_1, y_4−y_1]` — signed Eulerian tet volume.
- `m_T = 1/6` — conserved Lagrangian mass (cube mass 1 → 6 tets).
- `w_T` — galaxy bias weight (mean of the 4 vertices' `w = 1 + b₁δ_L + (b₂/2)(δ_L²−σ²) +
  b_s2(s²−⟨s²⟩)`), already computed per vertex.
- The sum runs over **all** tets whose deformed image contains `x` → **multi-streaming**
  is captured exactly (a halo point is inside many tets; their densities add).

This is the AHK estimator used in its native form; the grid was only ever a way to plug
it into a grid likelihood.

## 2. The likelihood — inhomogeneous Poisson point process (grid-free)

    log L = Σ_g log λ(x_g) − ∫ λ(x) dx,    λ(x) = N_tot · W(x) · ρ_g(x) / Z

With `W(x)` the survey selection and `Z = ∫ W ρ_g dx` pinning `∫λ = N_tot` (fixes the
amplitude → removes the n̄–bias degeneracy, as in the current grid likelihood). Two clean
pieces:

- **Data term** (over galaxies): `Σ_g u_g log ρ_g(x_g)` — evaluate the scattered density
  at each observed galaxy. `log W(x_g)` and `log N_tot` are data constants → drop.
- **Normalization** (over tets, *analytic*): a tet's integrated mass is `m_T·w_T`, so
  `Z = Σ_T m_T·w_T·W(x_T)` (centroid-windowed; the `|V_T|` cancels `∫_T W dx ≈ W·|V_T|`).
  No point queries needed for `Z`.

Negative log-posterior (the single Zygote entry point):

    −log P = −Σ_g u_g log ρ_g(x_g) + U_tot·log Z + ½‖ω‖² + ½‖(b−b₀)/σ_b‖²,  U_tot = Σ_g u_g

`u_g` = per-galaxy PROV weight (1 for observed PROV=0, soft for completed) — the point
process handles per-galaxy weighting natively, cleaner than grid over-dispersion.

## 3. Differentiability — exactly what is detached vs. flows

Same detach-the-assignment / differentiate-the-value pattern as `cic_deposit` and the
crossing:

- **DETACHED** (combinatorial, piecewise-constant in θ): the set `{T ∋ x_g}` (which tets
  contain each galaxy), the galaxy grid hash, the tet→shell assignment, `W(x_T)`.
- **DIFFERENTIABLE** (smooth in θ = ω, bias): `V_T` (via the 4 vertex positions `x_obs`),
  `w_T` (via δ_L, s² → ω, and the bias params), hence `ρ_T` and `Z`.

Gradient flow, given the detached containment:

    ∂(−logP)/∂ρ_g = −u_g/ρ_g ;   ∂(−logP)/∂Z = U_tot/Z
    ρ̄_T  = Σ_{g∈T} (−u_g/ρ_g)                         # gather galaxy cotangents to tets
    w̄_T  = ρ̄_T·(m_T/|V_T|)  +  (U_tot/Z)·m_T·W_T       # data + normalization
    V̄_T  = ρ̄_T·(−m_T w_T·sign(V_T)/V_T²)              # Z detaches W ⇒ no V_T term
    ȳ_i  = V̄_T·∂V_T/∂y_i ,   ∂V_T/∂y_2 = (1/6)(y_3−y_1)×(y_4−y_1), …   # det cofactors
    → scatter ȳ_i to x̄_obs (4 vertices), w̄_T/4 to w̄ (4 vertices)

`x̄_obs`, `w̄` (both O(N)) then flow through the **existing** `lightcone_cross_ad` and
`bias_fields` rrules to `ω` and the bias params. The **one** new hand-written rrule is the
tet→galaxy deposit — structurally identical to `cic_deposit` (forward atomic scatter,
backward gather), just retargeted from grid cells to galaxy points and emitting
`(x̄_obs, w̄)` instead of `(x̄, w̄)`.

The deposit primitive is `(ρ_g, Z) = sheet_density_at_points(x_obs, w, galaxy_index, W,
…)`; the loss layer `−Σ u log ρ_g + U log Z` is a trivial differentiable scalar of
`(ρ_g, Z)` (Zygote handles it). Clean separation: **one** hand rrule.

## 4. Spatial lookup — fixed-galaxy index + parallel-over-tets scatter

**Invert the chaining mesh: index the data, scatter from the model.** Galaxy positions
are fixed; the mesh deforms each forward.

- **Once:** build a 3-D uniform grid hash (Morton sort + cell-start array) on the fixed
  galaxy positions. Cell ≈ grid spacing. The lightcone's radial thinness ⇒ each tet's
  deformed AABB overlaps ~few cells; the footprint is 7.7% of the box ⇒ **~92% of tets
  hit only empty cells and early-out**.
- **Per forward (parallel over the 6N tets — dense, coalesced, no ragged cell lists):**
  each thread loads its 4 vertices (implicit connectivity via `_TETS`), computes `V_T`,
  `w_T`, AABB; queries overlapping galaxy cells; tests barycentric containment
  (`[y_2−y_1,y_3−y_1,y_4−y_1] λ = x_g−y_1`, inside iff all λ ≥ −ε); **atomic-adds `ρ_T`**
  to each contained galaxy's accumulator. Also reduces `Z += m_T w_T W(x_T)`.

All detached (in `@ignore_derivatives`); the rrule recomputes (or caches) the containment
in the backward. **F64 accumulators** for `ρ_g` and the gradient scatters (the F32-sum
lesson). Atomic contention only where many tets pile onto a halo galaxy (warp-aggregate
if it bites).

## 5. The one real design decision: continuity for HMC

The plain AHK density is **piecewise-constant per tet** ⇒ `ρ_g(x_g)` jumps as a galaxy
crosses a tet face under ω-moves. The assignment is detached so the *gradient* is smooth
a.e., but the *loss value* has O(1) steps → HMC energy-conservation suffers near caustics
(cf. the `max(ng,0)` kink, but many of them). What HMC actually needs is continuity of
`ρ_g(x_g)` **in ω** — i.e. the value must match across a face so a galaxy crossing it
produces no jump. Two routes to a continuous (linear) density:

**(A) AHK nodal averaging (recommended for this pipeline).** Give each vertex the
volume-weighted density of its incident tets `ρ_v = Σ_{T∋v} m_T w_T / Σ_{T∋v} |V_T|`, then
`ρ_g(x_g) = Σ_{T∋g} Σ_i λ_i^T(x_g)·ρ_{v_i(T)}` (barycentric interp; sum over containing
tets for multi-streaming). C⁰ because shared vertices carry one value and the face
barycentrics match. **Crucially it keeps the LINEAR point-in-tet locator** — the whole
reason the lookup is cheap and the rrule is a `cic_deposit` twin. Cost: one extra
tet→vertex reduction + the interp; a two-stage adjoint (galaxy → {λ (vertex positions),
ρ_v} → incident tets → V_T, w_T). Piecewise-constant is the `λ_i→` special case, so the
infrastructure is shared (build PWC first, add the nodal layer).

**(B) Multi-linear (hexahedral/"quadrilateral") elements** — the adaptive
phase-space-element route (Hahn & Angulo). A tri-linear map `x(q)` per cube → a Jacobian
density that is smooth *within* an element and adaptively refinable near caustics. But for
a **point-process** we must *locate* each galaxy in the sheet, and a tri-linear map is
**nonlinear to invert** — finding `q : x(q)=x_g` (and all multi-stream roots) is a 3-D
Newton solve per candidate element, per galaxy. That reintroduces an iterative root-find
inside the deposit, fights the grid-free single-pass GPU design, and is harder to
differentiate/parallelize than the linear barycentric test. Two more caveats: even
multi-linear elements have Jacobian discontinuities *across* element faces (the normal
derivative jumps), so C⁰ still wants nodal averaging either way — element order buys map
fidelity, not continuity per se; and its extra fidelity is **sub-galaxy-spacing**
(~15 Mpc/h), which ECHOES cannot constrain. So (B) is the higher-fidelity option for a
future variant where sub-resolution caustic structure matters, not for this data-limited
inference.

**Plan:** PWC for validation (matches the grid in the single-stream limit, simplest FD) →
**route (A)** for production HMC (shared infra, linear locator preserved, C⁰ value) →
the caustic `|V_T|` floor (§6) handles the residual C¹ roughness at folds. Reach for (B)
only if a science case needs the refined sheet below the galaxy spacing.

## 6. Edge cases (decided)

- **Caustics** `V_T→0`: floor `|V_T| ≥ ε·V_T^Lagrangian` (finite sheet thickness) — caps
  infinite density, regularizes the gradient.
- **Inverted/folded tets** `det<0`: use `|V_T|` (stream present, orientation flipped);
  `∂|V_T|/∂V_T = sign(V_T)`.
- **Galaxy in no tet** `ρ_g=0`: floor `ρ_g ≥ ρ_floor` (sheet should cover the box;
  rare/edge only).
- **Shell-straddling tets**: handled by the full AABB (no special logic).
- **Box-boundary periodic connectivity**: boundary-wrapping tets are outside the interior
  footprint (W=0, no galaxies) → culled.
- **`n_sub` is gone**: the density is exact per tet; no sub-sampling.
- **Window on points**: `W(x)` = analytic angular mask × radial n̄(χ) from the randoms,
  evaluated at tet centroids (detached) for `Z`.

## 7. Memory / performance expectations

- **Memory:** O(N) vertices + O(N_tet) in-register + O(N_gal) accumulator + galaxy hash ≈
  **~100–200 MB at res=128** (vs ~14 GB grid+deposit). The memory wall is gone ⇒ res≥128,
  multi-chain, 2/3LPT all become feasible; resolution is set by the sheet, not the GPU.
- **Compute:** dense parallel-over-tets (~1 M footprint tets) + few-galaxy lookup +
  scatter. The FFTs likely still dominate; profile the lookup (built-once index; per-tet
  AABB is the new cost).

## 8. Implementation phases

- **P1 — per-tet differentiable density** (`sheet_density_at_points` core, no lookup):
  `V_T`, `ρ_T`, `Z` (W=1) via a KA kernel + analytic det. FD-validate ∂Z/∂(x_obs,w) and
  mass conservation `Σ_T m_T = N`. Reuses `_TETS`. CPU first. *(DiscoDJNative)*
- **P2 — galaxy index + lookup** (detached): 3-D grid hash on fixed galaxies; per-tet AABB
  query + barycentric. Validate single-stream: each point in exactly one tet. *(DiscoDJNative)*
- **P3 — scatter + hand rrule**: tet→galaxy atomic deposit → `ρ_g`; gather adjoint →
  `x̄_obs, w̄`. FD-validate ∂ρ_g/∂(x_obs,w) and ∂(−logP)/∂ω. F64 accumulators. The core new
  primitive (analog of `cic_deposit`). *(DiscoDJNative)*
- **P4 — point-process likelihood**: `−Σ u_g log ρ_g + U log Z`, `W(x)` from randoms, PROV
  weights → grid-free `galaxy_density_sheet`/`loss`. End-to-end FD. *(DiscoInverse)*
- **P5 — validation suite**: single-stream vs grid (large-scale agreement), mass
  conservation, multi-stream (planar-collapse 3-stream analytic), GPU≡CPU,
  injection-recovery via NUTS (bias posteriors + r(k)).
- **P6 — C⁰ linear estimator** if HMC divergence rate demands it (§5).
- **P7 — real ECHOES run** at res≥128, the memory-unbounded unique pipeline.

## 9. Open questions / risks (ranked)

1. **Continuity vs HMC** (§5) — assess piecewise-constant divergence rate early; the
   linear estimator is the fallback. *Highest risk.*
2. **Lookup cost** per forward × thousands of NUTS grads — profile; footprint cull + thin
   AABBs should keep it cheap.
3. **Caustic floor `ε`** — sets the effective resolution in halos; tune.
4. **Z's W-detachment** — dropping ∇W(centroid); check it doesn't bias the amplitude.
5. **Atomic contention** in deep multi-stream halos — warp-aggregate if needed.
6. **Containment cache vs recompute** in the backward — memory of the (tet,galaxy) pairs
   in multi-stream regions; default to recompute.

## 10. Reuse summary

Unchanged & reused: `white_noise_to_fphi`, `compute_core_exact`, `exact_shape_stack`,
`bias_fields`/`bias_weight`, `lightcone_cross_ad` (x_obs, a_cross, v_r), `_TETS`,
geometry/embedding, the `cic_deposit` rrule pattern. New: `sheet_density_at_points` +
its rrule + galaxy grid hash *(DiscoDJNative)*; point-process loss + `W(x)` queryable
*(DiscoInverse)*. ~600–900 lines, mostly KA kernels.
