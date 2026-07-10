"""
Multi-tracer joint field reconstruction — one shared ΛCDM field constrained by several tracer
populations at once.

Folding more spectroscopy into the *field* (not per-object redshifts) is where "more data → more
stringent" pays off: a field constrained by Quaia (all-sky, photometric) plus DESI/BOSS/eBOSS
(spectroscopic) recovers the cosmic web that any single survey's blur or footprint misses. Each tracer
contributes an inhomogeneous-Poisson point-process term `−Σ log ρ_g(x) + U log Z` with its own linear
bias, survey window, and (fixed) positions; the fixed-amplitude phases are optimized against the sum.
The LPT + lightcone geometry (`_sheet_geometry`) is computed once and shared across tracers — only the
bias-weighted density and the tracer-position evaluation differ — so N tracers cost ≈1.4×, not N×, the
single-tracer forward.  Positions are held fixed: clustering does not sharpen individual redshifts (see
`likelihood/photoz.jl`), so the multi-tracer value is entirely at the field / large-scale level.
"""

using Zygote
using Random: MersenneTwister
using Statistics: mean

"""One tracer population for the joint reconstruction: fixed positions, cell list, window, bias, weights."""
struct Tracer{P, C, W, U, T<:AbstractFloat}
    pts::P; cl::C; window::W; b0::Vector{T}; u::U; Utot::T
end

"""
CMB-lensing convergence constraint — an all-sky, **unbiased total-matter**, line-of-sight-integral
constraint on the field, complementary to the (biased, local) tracer point processes.

The convergence is computed *natively through the tessellation*: rays from the observer (box centre)
sample the continuous CDM-sheet density along each sky direction and radial shell, and the Born kernel
`W_k = (3/2)Ω_m(H₀/c)² χ_k(1−χ_k/χ_s)/a_k` (as in DiscoDJNative's `density_shells_to_kappa`) integrates
the per-shell overdensity: `κ(n̂) = Σ_k W_k δ_k(n̂) dχ`.  It reuses `interp_sheet_at_points` on a fixed
sky×radial grid, so it is differentiable w.r.t. ω through the existing sheet rrules — no new primitive.
Because lensing sees the total matter it uses `w=1` (no galaxy bias), sharing the same sheet geometry the
tracers use.  Only the in-box redshift range is constrained; the high-z tail (z>z_far→z_src) is left to
the noise `invN`."""
struct LensingConstraint{T<:AbstractFloat, P, C, V, W3}
    pts::P               # (ndir·nshell, 3) ray points, fixed
    cl::C                # cell list of the ray points
    nd::Int; ns::Int     # n directions, n shells
    Wdχ::V               # (nshell,) Born kernel × dχ per shell
    κ_obs::V             # (ndir,) observed convergence
    invN::V              # (ndir,) inverse noise variance (or uniform)
    ones3::W3            # (res³) unit weight — total matter (no bias)
    ρfloor::T
end

"""
    lensing_constraint(gm, geom, cosmo, nhat, κ_obs; invN=nothing, nshell=32, z_source=1100, zpad=0.05) -> LensingConstraint

Build a CMB-lensing constraint. `nhat` is an `(ndir, 3)` array of sky directions (e.g. HEALPix pixel unit
vectors), `κ_obs` the observed convergence per direction. Radial shells span the box's lightcone range
`[z_near, z_far]` (padded); the kernel uses `z_source` (CMB ≈ 1100)."""
function lensing_constraint(gm::GalaxyModel{T}, geom, cosmo, nhat::AbstractMatrix, κ_obs::AbstractVector;
                            invN=nothing, nshell::Int=32, z_source::Real=1100.0, zpad::Real=0.05,
                            ρfloor::Real=1e-8) where {T}
    res = gm.res; L = gm.boxsize; dx = L / res; shift = vec(geom.shift); ndir = size(nhat, 1)
    znear = 1/geom.a_near - 1; zfar = 1/geom.a_far - 1
    zed = collect(range(znear + zpad, zfar - zpad; length=nshell + 1))
    χe = [comoving_distance(cosmo, 1/(1+z)) for z in zed]
    χc = (χe[1:end-1] .+ χe[2:end]) ./ 2; dχ = abs.(diff(χe))
    ac = [1/(1 + (zed[k]+zed[k+1])/2) for k in 1:nshell]
    χs = comoving_distance(cosmo, 1/(1+z_source)); Om = cosmo.Omega_c + cosmo.Omega_b
    H0 = 100*cosmo.h; ck = 299792.458
    W = [χc[k] <= χs ? 1.5*Om*(H0/ck)^2 * χc[k]*(1 - χc[k]/χs)/ac[k] : 0.0 for k in 1:nshell]
    pts = Matrix{T}(undef, ndir*nshell, 3); i = 1
    @inbounds for p in 1:ndir, k in 1:nshell
        pts[i,1] = χc[k]*nhat[p,1] + shift[1]; pts[i,2] = χc[k]*nhat[p,2] + shift[2]; pts[i,3] = χc[k]*nhat[p,3] + shift[3]; i += 1
    end
    iN = invN === nothing ? ones(T, ndir) : Vector{T}(invN)
    return LensingConstraint{T, Matrix{T}, typeof(build_cell_list(pts,dx)), Vector{T}, Array{T,3}}(
        pts, build_cell_list(pts, dx), ndir, nshell, T.(W .* dχ), Vector{T}(κ_obs), iN, ones(T,res,res,res), T(ρfloor))
end

# differentiable convergence map from the shared sheet geometry (unbiased total matter)
function _kappa_model(lc::LensingConstraint, gm, xg)
    ρv, _ = nodal_density(xg, lc.ones3, gm.res, gm.boxsize)
    ρ  = interp_sheet_at_points(xg, ρv, lc.pts, lc.cl, gm.res)     # (ndir·nshell,), shell-fastest
    ρm = permutedims(reshape(ρ, lc.ns, lc.nd))                     # (ndir, nshell)
    δ  = ρm ./ max.(mean(ρm; dims=1), lc.ρfloor) .- 1              # per-shell overdensity
    return δ * lc.Wdχ                                              # (ndir,) convergence
end
_lensing_loss(lc::LensingConstraint, gm, xg) =
    (κ = _kappa_model(lc, gm, xg); 0.5 * sum(lc.invN .* (κ .- lc.κ_obs) .^ 2))

"""    kappa_map(lc::LensingConstraint, gm, ω) -> κ::(ndir,)  — the differentiable CMB-lensing forward"""
kappa_map(lc::LensingConstraint, gm::GalaxyModel, ω) = _kappa_model(lc, gm, _sheet_geometry(gm, ω)[1])

"""
Peculiar-velocity (cosmic-flows) constraint — a **dynamical, galaxy-bias-free**, local constraint on the
field, complementary to the (biased) tracer point processes and the (integrated) lensing term.

The model radial peculiar velocity at each tracer is read straight off the tetrahedral sheet: the sheet
carries a velocity vector `v(q)=Σ_k f₁ D_k Ψ_k` at every vertex (`_sheet_geometry_v`), interpolated to the
tracer positions with the same `interp_sheet_at_points` used for density — accurate up to shell crossing,
i.e. exactly the quasi-linear regime Cosmicflows constrains. It is projected onto each tracer's radial
direction and scaled by `vnorm = 100·a·E(a)` (comoving Mpc/h → km/s). Differentiable w.r.t. ω through the
existing sheet rrules — no new primitive. The likelihood is Gaussian with inverse-variance `invN` (the
distance-error velocity noise, which grows ∝ distance); `submean` marginalizes the monopole/zero-point."""
struct VelocityConstraint{T<:AbstractFloat, P, C, V, V3}
    pts::P               # (N,3) tracer real-space positions
    cl::C                # cell list of pts
    rhat::V3             # (N,3) radial unit vectors from the observer
    v_obs::V             # (N,) observed radial peculiar velocity [km/s] (monopole removed if submean)
    invN::V              # (N,) inverse variance [(km/s)^-2]
    vnorm::T             # comoving Mpc/h → km/s
    submean::Bool        # subtract the (inverse-variance-weighted) monopole before comparing
end

_wmean(x, w) = sum(w .* x) / sum(w)

"""
    velocity_constraint(gm, geom, cosmo, pts, v_obs; sigma_v=nothing, submean=true) -> VelocityConstraint

Build a peculiar-velocity constraint. `pts` are the tracers' REAL-space box positions (from their
distances), `v_obs` the observed radial peculiar velocities [km/s], `sigma_v` the per-tracer velocity
errors [km/s] (uniform if omitted). The observer is the box centre (`geom.shift`)."""
function velocity_constraint(gm::GalaxyModel{T}, geom, cosmo, pts::AbstractMatrix, v_obs::AbstractVector;
                             sigma_v=nothing, submean::Bool=true) where {T}
    P = T.(pts); N = size(P, 1); obs = vec(geom.shift)
    d = P .- obs'; r = sqrt.(sum(d .^ 2; dims=2)); rh = T.(d ./ max.(r, T(1e-30)))
    Om = cosmo.Omega_c + cosmo.Omega_b; a = geom.a_near              # low-z edge ≈ observation epoch
    vnorm = 100 * a * sqrt(Om / a^3 + (1 - Om))                      # 100·a·E(a)  [km/s per Mpc/h]
    iN = sigma_v === nothing ? ones(T, N) : T.(1 ./ Vector{Float64}(sigma_v) .^ 2)
    vo = Vector{T}(v_obs); vo = submean ? vo .- T(_wmean(vo, iN)) : vo
    return VelocityConstraint{T, Matrix{T}, typeof(build_cell_list(P, gm.boxsize/gm.res)), Vector{T}, Matrix{T}}(
        P, build_cell_list(P, gm.boxsize/gm.res), rh, vo, iN, T(vnorm), submean)
end

# model radial peculiar velocity [km/s] at the tracers, from the shared geometry + velocity vector
function _velocity_model(vc::VelocityConstraint, gm, xg, vg)
    vx = interp_sheet_at_points(xg, vg[:, :, :, 1], vc.pts, vc.cl, gm.res)
    vy = interp_sheet_at_points(xg, vg[:, :, :, 2], vc.pts, vc.cl, gm.res)
    vz = interp_sheet_at_points(xg, vg[:, :, :, 3], vc.pts, vc.cl, gm.res)
    return vc.vnorm .* (vx .* vc.rhat[:, 1] .+ vy .* vc.rhat[:, 2] .+ vz .* vc.rhat[:, 3])
end
function _velocity_loss(vc::VelocityConstraint, gm, xg, vg)
    vm = _velocity_model(vc, gm, xg, vg)
    vm = vc.submean ? vm .- _wmean(vm, vc.invN) : vm
    return 0.5 * sum(vc.invN .* (vm .- vc.v_obs) .^ 2)
end

"""    radial_velocity(vc::VelocityConstraint, gm, ω) -> v_r::(N,) [km/s]  — the differentiable PV forward"""
radial_velocity(vc::VelocityConstraint, gm::GalaxyModel, ω) =
    (g = _sheet_geometry_v(gm, ω); _velocity_model(vc, gm, g[1], g[4]))

"""
    velocity_amplitude(vc, gm, ω) -> (; amplitude, sigma)

Best-fit amplitude `A` (∝ fσ₈ / fσ₈_fiducial) of the observed peculiar velocities relative to the fiducial-
growth model at field `ω`, with its Gaussian error — the linear **density–velocity fσ₈ estimator**.  When `ω`
is the density field constrained by *galaxies* (∝ bσ₈), fitting `A` to independent *velocities* measures
fσ₈ separately from the bias — this is how peculiar velocities break the growth–bias degeneracy.  `A=1`
means the flows match the fiducial-growth prediction."""
function velocity_amplitude(vc::VelocityConstraint, gm::GalaxyModel, ω)
    g = _sheet_geometry_v(gm, ω); vm = _velocity_model(vc, gm, g[1], g[4])
    vm = vc.submean ? vm .- _wmean(vm, vc.invN) : vm
    S = sum(vc.invN .* vm .^ 2)
    return (amplitude = sum(vc.invN .* vm .* vc.v_obs) / S, sigma = 1 / sqrt(S))
end

"""    tracer(gm, pts; b1, window, u=nothing) -> Tracer

Build a tracer from box positions `pts` (N×3), linear bias `b1`, and a survey `window` (res³, e.g.
`survey_window(geom, randoms)`).  The chaining-mesh cell list is built once."""
function tracer(gm::GalaxyModel{T}, pts::AbstractMatrix; b1::Real, window, u=nothing) where {T}
    P = T.(pts); cl = build_cell_list(P, gm.boxsize / gm.res)
    uu = u === nothing ? ones(T, size(P, 1)) : Vector{T}(u)
    return Tracer(P, cl, Array{T,3}(window), T[b1, 0, 0], uu, T(sum(uu)))
end

# ── Sheet-native tracer: NO grid window (no CIC). The survey footprint enters as the RANDOM POINTS, and
# the Poisson normalization Z is the Monte-Carlo integral of the tessellation density over them,
# Z = ⟨ρ_sheet(x_random)⟩ — so both the data term (ρ at galaxies) and Z (ρ at randoms) are evaluated
# purely on the phase-space sheet via `interp_sheet_at_points`.  The only grid/FFT anywhere is the 2LPT
# calculation of ψ; `survey_window`/`bin_galaxies` (CIC deposits) are never touched.
struct SheetTracer{P, C, RP, RC, U, T<:AbstractFloat}
    pts::P; cl::C; ran_pts::RP; ran_cl::RC; b0::Vector{T}; u::U; Utot::T
end

"""    sheet_tracer(gm, gal_pts, ran_pts; b1, u=nothing) -> SheetTracer

Sheet-native tracer from galaxy box positions `gal_pts` and survey random positions `ran_pts` (both N×3).
The footprint is the randoms themselves (Monte-Carlo Z = ⟨ρ_sheet(randoms)⟩) — no `survey_window`/CIC grid."""
function sheet_tracer(gm::GalaxyModel{T}, gal_pts::AbstractMatrix, ran_pts::AbstractMatrix;
                      b1::Real, u=nothing) where {T}
    G = T.(gal_pts); R = T.(ran_pts); h = gm.boxsize / gm.res
    gcl = build_cell_list(G, h); rcl = build_cell_list(R, h)
    uu = u === nothing ? ones(T, size(G, 1)) : Vector{T}(u)
    return SheetTracer(G, gcl, R, rcl, T[b1, 0, 0], uu, T(sum(uu)))
end

struct MultiTracerProblem{T<:AbstractFloat, GM, TR, LC, VC}
    gm::GM
    tracers::TR                 # Vector{Tracer}
    lensing::LC                 # LensingConstraint or Nothing
    velocity::VC                # VelocityConstraint or Nothing
    ρfloor::T
    floor_frac::T
end

"""    multitracer_problem(gm, tracers; lensing=nothing, velocity=nothing, ρfloor=1e-8, floor_frac=1e-3)

Joint problem over an arbitrary tracer list, optionally with a CMB-lensing convergence constraint
(`lensing::LensingConstraint`, all-sky unbiased integrated) and/or a peculiar-velocity constraint
(`velocity::VelocityConstraint`, dynamical bias-free local) — each adds a term to the field loss.
The tracer list may be empty for a lensing- or velocity-only reconstruction."""
function multitracer_problem(gm::GalaxyModel{T}, tracers; lensing=nothing, velocity=nothing,
                             ρfloor::Real=1e-8, floor_frac::Real=1e-3) where {T}
    return MultiTracerProblem{T, typeof(gm), typeof(tracers), typeof(lensing), typeof(velocity)}(
        gm, tracers, lensing, velocity, T(ρfloor), T(floor_frac))
end

_model_T(::MultiTracerProblem{T}) where {T} = T

# per-tracer point-process term, given the shared (δ_L, s², x_obs) geometry
function _tracer_pp_loss(gm, δL, s2, xg, tr::Tracer, ρfloor, floor_frac)
    wg = _sheet_weight(gm, δL, s2, tr.b0)
    ρv, Z = nodal_density(xg, _apply_window(wg, tr.window), gm.res, gm.boxsize; floor_frac=floor_frac)
    ρg = interp_sheet_at_points(xg, ρv, tr.pts, tr.cl, gm.res)
    return -sum(tr.u .* log.(max.(ρg, ρfloor))) + tr.Utot * log(Z)
end

# sheet-native (no grid window): Z = ⟨ρ_sheet(randoms)⟩ (Monte-Carlo over the footprint), all on the sheet
function _tracer_pp_loss(gm, δL, s2, xg, tr::SheetTracer, ρfloor, floor_frac)
    wg = _apply_window(_sheet_weight(gm, δL, s2, tr.b0), nothing)     # max(bias,0); NO window/CIC
    ρv, _ = nodal_density(xg, wg, gm.res, gm.boxsize; floor_frac=floor_frac)
    ρg = interp_sheet_at_points(xg, ρv, tr.pts, tr.cl, gm.res)        # sheet density at galaxies
    ρr = interp_sheet_at_points(xg, ρv, tr.ran_pts, tr.ran_cl, gm.res) # …at randoms → normalization
    return -sum(tr.u .* log.(max.(ρg, ρfloor))) + tr.Utot * log(max(mean(ρr), ρfloor))
end

# sum of the per-tracer point-process terms (handles an empty tracer list: lensing-/velocity-only)
_tracer_losses(mtp, δL, s2, xg) = isempty(mtp.tracers) ? zero(_model_T(mtp)) :
    sum(map(tr -> _tracer_pp_loss(mtp.gm, δL, s2, xg, tr, mtp.ρfloor, mtp.floor_frac), mtp.tracers))

# core data loss given the field ω (no prior) — shared by the fixed-amplitude MAP and the Gaussian-ω sampler
function _mtp_data_loss(mtp::MultiTracerProblem, ω)
    if mtp.velocity === nothing                          # no velocity term → skip the velocity vector
        xg, δL, s2 = _sheet_geometry(mtp.gm, ω)
        l = _tracer_losses(mtp, δL, s2, xg)
        return mtp.lensing === nothing ? l : l + _lensing_loss(mtp.lensing, mtp.gm, xg)
    else
        xg, δL, s2, vg = _sheet_geometry_v(mtp.gm, ω)
        l = _tracer_losses(mtp, δL, s2, xg) + _velocity_loss(mtp.velocity, mtp.gm, xg, vg)
        return mtp.lensing === nothing ? l : l + _lensing_loss(mtp.lensing, mtp.gm, xg)
    end
end

"""    multitracer_phase_loss(mtp, φ) -> scalar

Fixed-amplitude (Angulo–Pontzen) MAP loss: the field `ω = phase_field(φ)` constrained by all tracers +
any lensing/velocity terms.  Differentiable w.r.t. `φ` (Zygote)."""
multitracer_phase_loss(mtp::MultiTracerProblem, φ) = _mtp_data_loss(mtp, phase_field(φ))

"""    loss(mtp::MultiTracerProblem, ω, b) -> scalar

Gaussian-ω posterior loss (data terms + ½‖ω‖² ΛCDM prior), the target for HMC/`nuts_sample`.  Samples the
POSTERIOR over the field — honest UQ, where the fixed-amplitude MAP overfits low-SNR data (e.g. CF4
peculiar velocities, σ_v ≫ signal).  Tracers carry their own fixed bias, so the sampler's `b`-block is a
decoupled N(0,I) phantom (pass `b0=zeros(3)`); its samples are ignored."""
loss(mtp::MultiTracerProblem, ω::AbstractArray, b) = _mtp_data_loss(mtp, ω) + gaussian_prior(ω) + 0.5 * sum(abs2, b)

"""
    reconstruct_joint_field(mtp, seed; device=identity, iters=35) -> (; ω, φ)

Reconstruct the fixed-amplitude field jointly constrained by all tracers (device-resident L-BFGS on the
phases).  Returns the host white-noise field `ω` and the phases `φ` — the latter is a constraint carrier
that `refine_phases` / `constrained_ic_box` turn into a periodic IC box.  `device=CuArray` runs on GPU."""
function reconstruct_joint_field(mtp::MultiTracerProblem{T}, seed::Integer; device=identity,
                                 iters::Int=35) where {T}
    res = mtp.gm.res
    φ0 = device(2π .* rand(MersenneTwister(seed), res ÷ 2 + 1, res, res))
    f = φ -> multitracer_phase_loss(mtp, φ)
    φ, _, _ = _lbfgs_generic(f, φ -> Zygote.gradient(f, φ)[1], φ0; iters=iters)
    return (ω = Array(phase_field(φ)), φ = Array(φ))
end
