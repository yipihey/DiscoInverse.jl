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

"""One tracer population for the joint reconstruction: fixed positions, cell list, window, bias, weights."""
struct Tracer{P, C, W, U, T<:AbstractFloat}
    pts::P; cl::C; window::W; b0::Vector{T}; u::U; Utot::T
end

"""    tracer(gm, pts; b1, window, u=nothing) -> Tracer

Build a tracer from box positions `pts` (N×3), linear bias `b1`, and a survey `window` (res³, e.g.
`survey_window(geom, randoms)`).  The chaining-mesh cell list is built once."""
function tracer(gm::GalaxyModel{T}, pts::AbstractMatrix; b1::Real, window, u=nothing) where {T}
    P = T.(pts); cl = build_cell_list(P, gm.boxsize / gm.res)
    uu = u === nothing ? ones(T, size(P, 1)) : Vector{T}(u)
    return Tracer(P, cl, Array{T,3}(window), T[b1, 0, 0], uu, T(sum(uu)))
end

struct MultiTracerProblem{T<:AbstractFloat, GM, TR}
    gm::GM
    tracers::TR                 # Vector{Tracer}
    ρfloor::T
    floor_frac::T
end

"""    multitracer_problem(gm, tracers; ρfloor=1e-8, floor_frac=1e-3) -> MultiTracerProblem"""
function multitracer_problem(gm::GalaxyModel{T}, tracers; ρfloor::Real=1e-8, floor_frac::Real=1e-3) where {T}
    return MultiTracerProblem{T, typeof(gm), typeof(tracers)}(gm, tracers, T(ρfloor), T(floor_frac))
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
    return sum(map(tr -> _tracer_pp_loss(mtp.gm, δL, s2, xg, tr, mtp.ρfloor, mtp.floor_frac), mtp.tracers))
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
