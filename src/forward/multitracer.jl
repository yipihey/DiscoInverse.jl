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

"""    tracer(gm, pts; b1, window, u=nothing) -> Tracer

Build a tracer from box positions `pts` (N×3), linear bias `b1`, and a survey `window` (res³, e.g.
`survey_window(geom, randoms)`).  The chaining-mesh cell list is built once."""
function tracer(gm::GalaxyModel{T}, pts::AbstractMatrix; b1::Real, window, u=nothing) where {T}
    P = T.(pts); cl = build_cell_list(P, gm.boxsize / gm.res)
    uu = u === nothing ? ones(T, size(P, 1)) : Vector{T}(u)
    return Tracer(P, cl, Array{T,3}(window), T[b1, 0, 0], uu, T(sum(uu)))
end

struct MultiTracerProblem{T<:AbstractFloat, GM, TR, LC}
    gm::GM
    tracers::TR                 # Vector{Tracer}
    lensing::LC                 # LensingConstraint or Nothing
    ρfloor::T
    floor_frac::T
end

"""    multitracer_problem(gm, tracers; lensing=nothing, ρfloor=1e-8, floor_frac=1e-3) -> MultiTracerProblem

Joint problem over an arbitrary tracer list, optionally with a CMB-lensing convergence constraint
(`lensing::LensingConstraint`) that adds an all-sky, unbiased, line-of-sight term to the field loss."""
function multitracer_problem(gm::GalaxyModel{T}, tracers; lensing=nothing,
                             ρfloor::Real=1e-8, floor_frac::Real=1e-3) where {T}
    return MultiTracerProblem{T, typeof(gm), typeof(tracers), typeof(lensing)}(
        gm, tracers, lensing, T(ρfloor), T(floor_frac))
end

_model_T(::MultiTracerProblem{T}) where {T} = T

# per-tracer point-process term, given the shared (δ_L, s², x_obs) geometry
function _tracer_pp_loss(gm, δL, s2, xg, tr::Tracer, ρfloor, floor_frac)
    wg = _sheet_weight(gm, δL, s2, tr.b0)
    ρv, Z = nodal_density(xg, _apply_window(wg, tr.window), gm.res, gm.boxsize; floor_frac=floor_frac)
    ρg = interp_sheet_at_points(xg, ρv, tr.pts, tr.cl, gm.res)
    return -sum(tr.u .* log.(max.(ρg, ρfloor))) + tr.Utot * log(Z)
end

"""    multitracer_phase_loss(mtp, φ) -> scalar

Fixed-amplitude phase loss: the field `ω = phase_field(φ)` constrained by ALL tracers (sum of their
point-process terms).  Differentiable w.r.t. `φ` (Zygote)."""
function multitracer_phase_loss(mtp::MultiTracerProblem, φ)
    ω = phase_field(φ)
    xg, δL, s2 = _sheet_geometry(mtp.gm, ω)
    l = sum(map(tr -> _tracer_pp_loss(mtp.gm, δL, s2, xg, tr, mtp.ρfloor, mtp.floor_frac), mtp.tracers))
    return mtp.lensing === nothing ? l : l + _lensing_loss(mtp.lensing, mtp.gm, xg)
end

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
