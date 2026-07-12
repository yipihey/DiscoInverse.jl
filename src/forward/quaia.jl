"""
Quaia field-level redshift reconstruction — χ-parametrized forward + alternating-MAP driver.

The photo-z radial blur (~120 Mpc/h) is resolved by inferring each quasar's true comoving
distance χ_i from the 3D ΛCDM clustering jointly with the white-noise field ω.  The one new
capability over the CMASS sheet forward is that the **quasar positions are free, differentiable
parameters**: with the query-point gradient in `interp_sheet_at_points` (DiscoDJNative), the
sheet density ρ_g is differentiable w.r.t. the points, and the χ-linear embedding
`x_i = χ_i·n̂_i + shift` (n̂ = exact sky direction) makes ∂x/∂χ trivial — no interpolation
gradient (differentiating z→χ through `comoving_distance` segfaults Zygote; see `redshift_prior`).

`reconstruct_quaia` runs the per-seed **alternating MAP**: a fixed-amplitude (Angulo–Pontzen)
phase step `phase_map_optimize` at the current χ (so the realization keeps full ΛCDM power with
seed-dependent unconstrained phases), then a χ step (L-BFGS on the point-process + photo-z prior
at the current field).  Different seeds → different unconstrained phases → a constrained-realization
ensemble whose per-quasar spread is the field-level radial uncertainty.  `quaia_ensemble` drives a
set of seeds and (optionally) writes each realization out.

CPU/GPU: pass `device=CuArray` (after `using CUDA`) to run the whole reconstruction on the device;
the default `device=identity` runs on the host (used by the FD/recovery validation).
"""

using Zygote
using Random: MersenneTwister

"""    sky_directions(ra, dec) -> (N,3) unit vectors  (ra, dec in degrees)"""
sky_directions(ra, dec) =
    hcat(cosd.(dec) .* cosd.(ra), cosd.(dec) .* sind.(ra), sind.(dec))

# χ → z by inverting the comoving-distance table (monotone): linear interp on a z-grid.
# Replaces Interpolations.jl (not a dependency) with a searchsorted lookup — matches the
# production pipeline exactly.  Realizations convert the reconstructed χ̂ back to ẑ with this.
function build_z_of_χ(cosmo; zmax::Real=5.0, dz::Real=0.01)
    zg = collect(0.0:dz:zmax)
    χg = [comoving_distance(cosmo, 1 / (1 + z)) for z in zg]
    return function (v)
        v <= χg[1]   && return zg[1]
        v >= χg[end] && return zg[end]
        i = searchsortedfirst(χg, v)
        return zg[i-1] + (v - χg[i-1]) / (χg[i] - χg[i-1]) * (zg[i] - zg[i-1])
    end
end

struct QuaiaProblem{T<:AbstractFloat, GM, BG, F}
    gm::GM                  # GalaxyModel (host; moved to device per-round when device≠identity)
    geom::BG                # BoxGeometry (all-sky)
    nhat::Matrix{T}         # (N,3) exact sky directions
    shift::Matrix{T}        # (1,3) comoving→box shift
    χ_obs::Vector{T}        # photo-z comoving distance (the prior centre)
    σχ::Vector{T}           # photo-z radial width = χ(z_obs+σ_z) − χ(z_obs)
    u::Vector{T}            # per-quasar weights
    window::Array{T,3}      # LEGACY grid window (res³); used only if ran_pts is empty
    ran_pts::Matrix{T}      # sheet-native: survey RANDOM box positions (N_ran,3) → Z=⟨ρ_sheet(randoms)⟩ (empty ⇒ legacy)
    res::Int
    boxsize::T
    dx::T
    b1::T                   # quasar linear bias
    z_of_χ::F               # χ → z closure
end

Base.length(prob::QuaiaProblem) = length(prob.χ_obs)

"""    quaia_problem(cat, geom, gm, window; b1=2.5, u=nothing) -> QuaiaProblem

Precompute the χ-space quantities (sky directions, χ_obs, σχ, the χ→z table) for the catalog.
`window` is e.g. `survey_window(geom, randoms)`; `gm` a `galaxy_model(res, L, …; rsd=false)`
(RSD ≪ σ_z, so real-space).  `b1` is held fixed during the phase step."""
function quaia_problem(cat::QuaiaCatalog{T}, geom::BoxGeometry, gm::GalaxyModel, window;
                       b1::Real=2.5, u=nothing, randoms=nothing) where {T}
    cosmo = geom.cosmo
    nhat  = sky_directions(cat.ra, cat.dec)
    χ_obs = T[comoving_distance(cosmo, 1 / (1 + zi)) for zi in cat.z_obs]
    σχ    = T[comoving_distance(cosmo, 1 / (1 + cat.z_obs[i] + cat.σz[i])) - χ_obs[i]
              for i in eachindex(cat.z_obs)]
    shift = reshape(T.(geom.shift), 1, 3)
    uu    = u === nothing ? ones(T, length(cat)) : Vector{T}(u)
    res   = geom.res; L = T(geom.boxsize); dx = L / res
    # sheet-native footprint: the survey randoms as box positions (Z = ⟨ρ_sheet(randoms)⟩; no grid window)
    rp    = randoms === nothing ? Matrix{T}(undef, 0, 3) :
            T.(radec_z_to_cartesian(randoms.ra, randoms.dec, randoms.z, cosmo)) .+ shift
    win   = window === nothing ? ones(T, res, res, res) : Array{T,3}(window)
    return QuaiaProblem{T, typeof(gm), typeof(geom), typeof(build_z_of_χ(cosmo))}(
        gm, geom, Matrix{T}(nhat), shift, χ_obs, σχ, uu, win, rp,
        res, L, T(dx), T(b1), build_z_of_χ(cosmo))
end

"""    quaia_positions(prob, χ) -> (N,3) box coordinates  x = χ·n̂ + shift  (backend follows χ/n̂)"""
quaia_positions(nhat, shift, χ) = χ .* nhat .+ shift

# χ-step loss at a FIXED field (xg = sheet vertices, ρv = nodal densities): the point-process
# data term + the photo-z prior.  The gradient w.r.t. χ flows through the query-point rrule of
# interp_sheet_at_points; the discrete cell list is detached (rebuilt forward-only each call).
function _quaia_chi_loss(χ, xg, ρv, nhat, shift, χ_obs, σχ, u, dx, res)
    x  = quaia_positions(nhat, shift, χ)
    cl = @ignore_derivatives build_cell_list(Array(x), dx)
    ρq = interp_sheet_at_points(xg, ρv, x, cl, res)
    return -sum(u .* log.(max.(ρq, 1e-8))) + redshift_prior(χ, χ_obs, σχ)
end

"""
    reconstruct_quaia(prob, seed; device=identity, rounds=3, phase_iters=25, chi_iters=25,
                      b1=prob.b1) -> (; ω, χ, z)

One constrained realization: alternate a fixed-amplitude phase MAP (`phase_map_optimize` at the
current χ, warm-started across rounds) and a χ MAP (`_lbfgs_generic` on the point-process +
photo-z prior at the current field).  Returns the host white-noise field ω, the reconstructed
comoving distances χ, and the redshifts z = z_of_χ(χ).  `device=CuArray` runs on the GPU."""
function reconstruct_quaia(prob::QuaiaProblem{T}, seed::Integer; device=identity,
                           rounds::Int=3, phase_iters::Int=25, chi_iters::Int=25,
                           b1::Real=prob.b1) where {T}
    usegpu = device !== identity
    res = prob.res; L = prob.boxsize; dx = prob.dx; bb = [T(b1), zero(T), zero(T)]
    sheet = size(prob.ran_pts, 1) > 0                                     # sheet-native (randoms) vs legacy grid window
    # device copies of the fixed χ-step data (no-ops on host)
    nhat_d = device(prob.nhat); shift_d = device(prob.shift)
    χobs_d = device(prob.χ_obs); σχ_d = device(prob.σχ); u_d = device(prob.u)
    win_d  = sheet ? nothing : device(prob.window)
    gm_d   = usegpu ? gpu(prob.gm) : prob.gm
    χ = copy(prob.χ_obs); φ = nothing; local ω
    for _ in 1:rounds
        pos = quaia_positions(prob.nhat, prob.shift, χ)                       # host positions
        sp  = sheet ? sheet_problem(prob.gm, pos; ran_pts=prob.ran_pts, u=prob.u, b0=[T(b1), 0, 0], σb=[5.0, 5.0, 5.0]) :
                      sheet_problem(prob.gm, pos; window=prob.window, u=prob.u, b0=[T(b1), 0, 0], σb=[5.0, 5.0, 5.0])
        spd = usegpu ? gpu(sp) : sp
        φ0  = device(φ === nothing ? T(2π) .* rand(MersenneTwister(seed), T, res ÷ 2 + 1, res, res) : φ)
        r   = phase_map_optimize(spd, φ0; b1_grid=[T(b1)], phase_iters=phase_iters, b2=0.0, bs2=0.0)
        φ   = Array(r.φ); ω = r.ω
        xg, wg = _sheet_inputs(gm_d, ω, bb)
        ρv, _  = nodal_density(xg, _apply_window(wg, win_d), res, L)          # win_d=nothing ⇒ windowless (sheet)
        f   = cc -> _quaia_chi_loss(cc, xg, ρv, nhat_d, shift_d, χobs_d, σχ_d, u_d, dx, res)
        cg, _, _ = _lbfgs_generic(f, cc -> Zygote.gradient(f, cc)[1], device(χ); iters=chi_iters)
        χ   = Array(cg)
    end
    # φ is the converged fixed-amplitude phases (host, rfft shape (res÷2+1,res,res)) — the
    # constrained-IC-box refinement embeds these into a finer grid (forward/constrained_box.jl).
    return (ω = Array(ω), χ = χ, z = prob.z_of_χ.(χ), φ = φ)
end

"""
    quaia_ensemble(prob, seeds; device=identity, save=nothing, kw...) -> (; χ, z, seeds)

Drive a set of phase seeds into a constrained-realization ensemble.  Returns the (N×K) matrices
of reconstructed comoving distances `χ` and redshifts `z`.  `save(k, seed, realization)` is called
after each realization (e.g. to `npzwrite` a per-seed zero-redshift-error catalog).  Extra kwargs
(`rounds`, `phase_iters`, `chi_iters`, `b1`) pass through to `reconstruct_quaia`."""
function quaia_ensemble(prob::QuaiaProblem{T}, seeds; device=identity, save=nothing, kw...) where {T}
    N = length(prob); K = length(seeds)
    χmat = Matrix{T}(undef, N, K); zmat = Matrix{Float64}(undef, N, K); ss = collect(seeds)
    for (k, seed) in enumerate(ss)
        r = reconstruct_quaia(prob, seed; device=device, kw...)
        χmat[:, k] .= r.χ; zmat[:, k] .= r.z
        save === nothing || save(k, seed, r)
    end
    return (; χ = χmat, z = zmat, seeds = ss)
end
