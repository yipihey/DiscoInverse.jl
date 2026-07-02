"""
Grid-free galaxy-field forward + point-process likelihood on the AHK lightcone sheet (P4).

The whole upstream is reused unchanged — `ω → φ → ψ → {δ_L,s²} → w(q)` and the
differentiable lightcone crossing `→ x_obs` (the per-vertex positions in redshift space).
The grid + CIC deposit is replaced by `sheet_density_at_points` (DiscoDJNative P3): the AHK
density `ρ_g(x_g) = Σ_{T∋g} m_T w_T/|V_T|` evaluated directly at the (fixed) galaxy
positions, plus the analytic normalisation `Z = Σ_T m_T w_T`.

The likelihood is the inhomogeneous Poisson point process (W=1 here; the survey window
enters Z later):

    −log P = −Σ_g u_g log ρ_g(x_g) + U_tot·log Z + ½‖ω‖² + bias prior

`galaxy_density_sheet(gm, ω, b, pts, cl)` and `loss(::SheetProblem, ω, b)` are
Zygote-differentiable end-to-end (the deposit's hand rrule + the existing crossing/bias
rrules).  `inject_mock_sheet` samples a galaxy catalogue from the model for
injection-recovery.
"""

# Shared, BIAS-INDEPENDENT geometry: ω → (x_obs vertices on the lightcone, δ_L, s²). Computed once and
# reused across tracers of different bias in the multi-tracer forward (forward/multitracer.jl); the
# expensive LPT + lightcone happen here, only the per-vertex weight below depends on the bias.
function _sheet_geometry(gm::GalaxyModel{T}, ω::AbstractArray{T,3}) where {T}
    fphi   = white_noise_to_fphi(gm.op, ω)
    Psi    = exact_shape_stack(compute_core_exact(fphi, gm.K; n_order=gm.n_order))
    δL, s2 = bias_fields(fphi, gm.ops)
    lc     = lightcone_cross_ad(Psi, gm.qflat, gm.cosmo, gm.observer, gm.a_far, gm.a_near; rsd=gm.rsd)
    x      = lc.x_obs
    if gm.rsd
        o1, o2, o3 = T(gm.observer[1]), T(gm.observer[2]), T(gm.observer[3])
        d1 = x[:,1] .- o1; d2 = x[:,2] .- o2; d3 = x[:,3] .- o3
        s  = lc.v_r ./ max.(sqrt.(d1.^2 .+ d2.^2 .+ d3.^2), T(1e-30))
        x  = hcat(x[:,1] .+ s.*d1, x[:,2] .+ s.*d2, x[:,3] .+ s.*d3)
    end
    return reshape(x, gm.res, gm.res, gm.res, 3), δL, s2
end

# per-vertex bias weight w(q) for bias b, from the shared (δ_L, s²)
_sheet_weight(gm::GalaxyModel, δL, s2, b) = bias_weight(δL, s2, gm.sigma2, gm.s2mean; b1=b[1], b2=b[2], bs2=b[3])

# Shared upstream: ω,b → (x_obs vertices on the lightcone, per-vertex bias weight).
function _sheet_inputs(gm::GalaxyModel, ω, b)
    xg, δL, s2 = _sheet_geometry(gm, ω)
    return xg, _sheet_weight(gm, δL, s2, b)
end

# Per-vertex weight w(q) ← max(bias(q), 0)·W(q):
#   - clip the linear bias 1+b₁δ_L+… at 0 — it goes negative in deep voids (δ_L < −1/b₁), which
#     would make ρ_g and the normalization Z negative (unphysical; log Z throws). Galaxies sit in
#     over-densities so this only touches empty voids; the loss's ρ_g-floor handles any galaxy there.
#   - W is the (fixed) footprint window, so Z = Σ_T m_T w_T integrates only over the observed
#     volume and the gradient vanishes outside the footprint (no data there).
@inline _apply_window(wg, window) =
    (w = max.(wg, zero(eltype(wg))); window === nothing ? w : w .* window)

"""    galaxy_density_sheet(gm, ω, b, pts, cl; floor_frac=1e-3, window=nothing) -> (ρ_g::(N,), Z)"""
function galaxy_density_sheet(gm::GalaxyModel, ω, b, pts, cl; floor_frac::Real=1e-3, window=nothing)
    xg, wg = _sheet_inputs(gm, ω, b)
    return sheet_density_at_points(xg, _apply_window(wg, window), pts, cl, gm.res, gm.boxsize; floor_frac=floor_frac)
end

"""    galaxy_density_sheet_c0(gm, ω, b, pts, cl; floor_frac=1e-3) -> (ρ_g, Z)  [C⁰ nodal-averaged]

Continuous density: vertex densities `ρ_v` (nodal_density) then barycentric interpolation at
the galaxies (interp_sheet_at_points).  Continuous across tet faces ⇒ the loss is smoothly
optimizable, where the piecewise-constant version is not (P5 finding)."""
function galaxy_density_sheet_c0(gm::GalaxyModel, ω, b, pts, cl; floor_frac::Real=1e-3, window=nothing)
    xg, wg = _sheet_inputs(gm, ω, b)
    ρv, Z = nodal_density(xg, _apply_window(wg, window), gm.res, gm.boxsize; floor_frac=floor_frac)
    ρg = interp_sheet_at_points(xg, ρv, pts, cl, gm.res)
    return (ρg, Z)
end

"""    galaxy_density_sheet_c0_masked(gm, ω, b, pts, cl, active; floor_frac=1e-3, window=nothing) -> (ρ_g, Z)

Sheet-on-mask C⁰ galaxy density restricted to the footprint trace-back `active` mask — only the ~10%
active tets are processed (nodal_density_masked + interp_sheet_at_points_masked).  Exact at the
footprint galaxies (their containing tets are active); ~9× fewer tets.  Differentiable w.r.t. ω, b."""
function galaxy_density_sheet_c0_masked(gm::GalaxyModel, ω, b, pts, cl, active; floor_frac::Real=1e-3, window=nothing)
    xg, wg = _sheet_inputs(gm, ω, b)
    ρv, Z = nodal_density_masked(xg, _apply_window(wg, window), gm.res, gm.boxsize, active; floor_frac=floor_frac)
    ρg = interp_sheet_at_points_masked(xg, ρv, pts, cl, gm.res, active)
    return (ρg, Z)
end

struct SheetProblem{T<:AbstractFloat, GM, P, C, U, W, A}
    gm::GM
    pts::P                  # (N_gal, 3) galaxy positions (fixed; redshift space)
    cl::C                   # chaining-mesh cell list on pts (built once)
    u::U                    # PROV per-galaxy weights (host Vector or CuArray after gpu())
    Utot::T
    b0::Vector{T}; σb::Vector{T}
    ρfloor::T               # floor for log ρ_g (galaxies in no tet)
    floor_frac::T           # caustic floor: |V_T| ≥ floor_frac·V_Lagrangian (caps ρ_T)
    c0::Bool                # C⁰ nodal-averaged density (true, optimizable) vs piecewise-constant
    window::W               # survey footprint mask (res³, Lagrangian) folded into w_T, or nothing
    active::A               # footprint trace-back mask (Bool res³) → sheet-on-mask (~9× fewer tets), or nothing
end

"""    sheet_problem(gm, pts; u, b0, σb, ρfloor=1e-8, floor_frac=1e-3, c0=true, window=nothing, h=cell) -> SheetProblem

`window` (res³ footprint field, e.g. from `survey_window(geom, randoms)`) folds into the
per-vertex weight so the normalization integrates only over the observed volume — required
for real survey data (else the optimizer games Z with spurious out-of-footprint structure)."""
function sheet_problem(gm::GalaxyModel{T}, pts::AbstractMatrix; u=nothing,
                       b0=[1.0,0,0], σb=[5.0,5,5], ρfloor::Real=1e-8, floor_frac::Real=1e-3,
                       c0::Bool=true, window=nothing, active=nothing, h=nothing) where {T}
    P = T.(pts); hh = h === nothing ? gm.boxsize/gm.res : T(h)
    cl = build_cell_list(P, hh)
    uu = u === nothing ? ones(T, size(P,1)) : Vector{T}(u)
    win = window === nothing ? nothing : Array{T,3}(window)
    act = active === nothing ? nothing : Array{Bool,3}(active)
    return SheetProblem{T, typeof(gm), typeof(P), typeof(cl), typeof(uu), typeof(win), typeof(act)}(gm, P, cl, uu, sum(uu),
                          Vector{T}(b0), Vector{T}(σb), T(ρfloor), T(floor_frac), c0, win, act)
end

# mixed-precision NUTS hook: an F32 SheetProblem evaluates its analytic forward in F32
# while the sampler keeps its leapfrog state in F64 (see infer/nuts.jl `_loss_grad`).
_model_T(prob::SheetProblem{T}) where {T} = T

_sheet_dens(prob::SheetProblem, ω, b) =
    prob.active !== nothing ?
        galaxy_density_sheet_c0_masked(prob.gm, ω, b, prob.pts, prob.cl, prob.active; floor_frac=prob.floor_frac, window=prob.window) :
    prob.c0 ?
        galaxy_density_sheet_c0(prob.gm, ω, b, prob.pts, prob.cl; floor_frac=prob.floor_frac, window=prob.window) :
        galaxy_density_sheet(prob.gm, ω, b, prob.pts, prob.cl; floor_frac=prob.floor_frac, window=prob.window)

"""    loss(prob::SheetProblem, ω, b) -> −Σ u_g log ρ_g + U log Z + priors  (Zygote entry point)"""
function loss(prob::SheetProblem, ω, b)
    ρg, Z = _sheet_dens(prob, ω, b)
    return -sum(prob.u .* log.(max.(ρg, prob.ρfloor))) + prob.Utot * log(Z) +
           gaussian_prior(ω) + bias_prior(b, prob.b0, prob.σb)
end

model_density(prob::SheetProblem, ω, b) = _sheet_dens(prob, ω, b)[1]

"""
    inject_mock_sheet(gm, ω*, b*, ntot; seed=0) -> pts::(N,3)

Sample a galaxy catalogue from the model point process: tet T gets Poisson(ntot·m_T w_T/Z)
galaxies placed uniformly inside it (random barycentric), so the catalogue's intensity is
λ ∝ ρ_g.  Used for injection-recovery.
"""
function inject_mock_sheet(gm::GalaxyModel{T}, ωstar::AbstractArray{T,3}, bstar, ntot::Real; seed::Int=0) where {T}
    xg, wg = _sheet_inputs(gm, ωstar, bstar); xg = Array(xg); wg = Array(wg)
    res = gm.res; nc = res-1; off = DiscoDJNative._TET_OFFSETS; mT = 1/6
    Z = 0.0
    @inbounds for i in 1:nc, j in 1:nc, k in 1:nc, t in 1:6
        Z += mT * (wg[i+off[t,1,1],j+off[t,1,2],k+off[t,1,3]] + wg[i+off[t,2,1],j+off[t,2,2],k+off[t,2,3]] +
                   wg[i+off[t,3,1],j+off[t,3,2],k+off[t,3,3]] + wg[i+off[t,4,1],j+off[t,4,2],k+off[t,4,3]]) / 4
    end
    rng = MersenneTwister(seed); pts = NTuple{3,T}[]
    @inbounds for i in 1:nc, j in 1:nc, k in 1:nc, t in 1:6
        v = ntuple(vv -> (xg[i+off[t,vv,1],j+off[t,vv,2],k+off[t,vv,3],1],
                          xg[i+off[t,vv,1],j+off[t,vv,2],k+off[t,vv,3],2],
                          xg[i+off[t,vv,1],j+off[t,vv,2],k+off[t,vv,3],3]), 4)
        wT = (wg[i+off[t,1,1],j+off[t,1,2],k+off[t,1,3]] + wg[i+off[t,2,1],j+off[t,2,2],k+off[t,2,3]] +
              wg[i+off[t,3,1],j+off[t,3,2],k+off[t,3,3]] + wg[i+off[t,4,1],j+off[t,4,2],k+off[t,4,3]]) / 4
        n = _rand_poisson(rng, ntot * mT * wT / Z)
        for _ in 1:n
            e1=-log(rand(rng)); e2=-log(rand(rng)); e3=-log(rand(rng)); e4=-log(rand(rng)); es=e1+e2+e3+e4
            l1=e1/es; l2=e2/es; l3=e3/es; l4=e4/es
            push!(pts, (T(l1*v[1][1]+l2*v[2][1]+l3*v[3][1]+l4*v[4][1]),
                        T(l1*v[1][2]+l2*v[2][2]+l3*v[3][2]+l4*v[4][2]),
                        T(l1*v[1][3]+l2*v[2][3]+l3*v[3][3]+l4*v[4][3])))
        end
    end
    P = Matrix{T}(undef, length(pts), 3)
    @inbounds for g in eachindex(pts); P[g,1]=pts[g][1]; P[g,2]=pts[g][2]; P[g,3]=pts[g][3]; end
    return P
end
