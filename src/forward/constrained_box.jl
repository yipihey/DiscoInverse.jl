"""
Quaia-constrained periodic IC box — coarse-constrain → fine-realize.

Emit a periodic cosmological initial-conditions box of any length ≥ the Quaia volume and any
resolution (1024³, 2048³, …), whose center carries ICs consistent with the Quaia quasars + clustering
(within the photo-z errors) and the fiducial cosmology, the rest unconstrained random phases, with
**perfectly fixed P(k)**.

The enabling insight (resolution decoupling): the differentiable forward (LPT + lightcone + sheet +
Zygote tape) is the memory bottleneck — it caps at res ≈ 384 on the A6000 — but the Quaia constraint
is band-limited to large scales (the ~120 Mpc/h photo-z blur informs only k ≲ 0.05 h/Mpc, no small-
scale information).  So we run the constraint (`reconstruct_quaia`) at a COARSE `res_constrain` (≤256,
lossless), then realize the box at a FINE `res_box` by **spectral white-noise refinement**: a pure FFT
upsample with no AD tape, so it scales far beyond the forward's ceiling.  This is the MUSIC multi-scale
white-noise paradigm (Hahn & Abel) applied to a field-level constraint.

Refinement works in PHASE space (the path to exact fixed P(k)): the constrained coarse fixed-amplitude
phases are embedded into the fine rfft grid at matching physical k, the new high-k modes get fresh
uniform-random phases, and `phase_field` gives a unit-modulus (⇒ exact fiducial P(k)) real field whose
large scales carry the constraint and whose small scales + outer shell are a uniform-random draw.
"""

using FFTW
using Random: MersenneTwister
using Statistics: std

# Coarse full-axis rfft index `ic` (1-based) → fine full-axis index at the SAME physical k (same L_box
# ⇒ same integer frequency).  numpy/FFTW order [0,1,…,N/2-1, -N/2,…,-1]: positive freqs to the front,
# negative to the back; the coarse Nyquist (−Nc/2) lands at a regular interior fine mode.
function _embed_indices(Nc::Int, Nb::Int)
    idx = Vector{Int}(undef, Nc)
    @inbounds for ic in 1:Nc
        f = (ic - 1) < Nc ÷ 2 ? (ic - 1) : (ic - 1 - Nc)   # freq ∈ [-Nc/2, Nc/2-1]
        idx[ic] = f ≥ 0 ? f + 1 : Nb + f + 1
    end
    return idx
end

"""
    refine_phases(φ_coarse, res_box; seed=0, fixed_amplitude=true) -> ω_box::(res_box,res_box,res_box)

Spectral white-noise refinement.  `φ_coarse` are the constrained fixed-amplitude phases (rfft shape
`(Nc÷2+1, Nc, Nc)` from `reconstruct_quaia(...).φ`).  Embeds them into a finer rfft grid of size
`res_box` at matching physical k (same `L_box`), fills the new high-k modes with fresh uniform-random
phases (`seed`), and returns the real white-noise field `ω_box = phase_field(φ_box)`.

`fixed_amplitude=true` (default, honoring "perfectly fixed P(k)") gives every mode unit modulus ⇒ the
field through `ic_operator(res_box, L_box, pk)` has exactly the fiducial P(k) at all scales.
`false` gives the unconstrained high-k modes Rayleigh amplitudes (a standard small-scale GRF) while the
constrained low-k band stays fixed.

The interior modes are embedded bit-exactly; the kz=0 / Nyquist planes are projected by `irfft` (as in
all fixed-amplitude work here), so the field-level fidelity to the coarse constraint is ≳99% and →1 as
`res_box/res_constrain` grows.
"""
function refine_phases(φ_coarse::AbstractArray, res_box::Int; seed::Integer=0,
                       fixed_amplitude::Bool=true)
    Nc = size(φ_coarse, 2)
    res_box ≥ Nc      || error("res_box ($res_box) must be ≥ res_constrain ($Nc)")
    iseven(res_box)   || error("res_box ($res_box) must be even")
    φc = Array(φ_coarse)
    φ_box = 2π .* rand(MersenneTwister(seed), res_box ÷ 2 + 1, res_box, res_box)  # fresh random phases
    i = _embed_indices(Nc, res_box)
    @inbounds for c3 in 1:Nc, c2 in 1:Nc
        @views φ_box[1:Nc ÷ 2 + 1, i[c2], i[c3]] .= φc[:, c2, c3]                 # embed the constraint
    end
    fixed_amplitude && return phase_field(φ_box)
    # Gaussian small-scale variant: constrained low-k stay unit modulus; new high-k get Rayleigh |a|
    # with ⟨|a|²⟩=1 (matches the unit-modulus power, so P(k) is continuous, only the scatter differs).
    A = exp.(im .* φ_box)
    keep = trues(res_box ÷ 2 + 1, res_box, res_box)
    @inbounds for c3 in 1:Nc, c2 in 1:Nc; @views keep[1:Nc ÷ 2 + 1, i[c2], i[c3]] .= false; end
    amp = sqrt.(.-log.(rand(MersenneTwister(seed + 1), res_box ÷ 2 + 1, res_box, res_box)))
    A[keep] .*= amp[keep]
    ω = irfft(A, res_box)
    return ω ./ std(ω)
end

# Provenance manifest written alongside ω (exportable to npz; scalars + small vectors).
function _ic_box_manifest(geom::BoxGeometry, cosmo, res_constrain, res_box, b1, seed, cat, fixed_amplitude)
    zmax = maximum(cat.z_obs)
    return Dict{String,Any}(
        "res_constrain"      => res_constrain,
        "res_box"            => res_box,
        "boxsize"            => geom.boxsize,
        "Omega_m"            => cosmo.Omega_c + cosmo.Omega_b,
        "Omega_b"            => cosmo.Omega_b,
        "h"                  => cosmo.h,
        "sigma8"             => cosmo.sigma8,
        "n_s"                => cosmo.n_s,
        "observer"           => collect(geom.observer),
        "shift"              => collect(geom.shift),
        "constrained_radius" => comoving_distance(cosmo, 1 / (1 + zmax)),  # χ(z_max): the central sphere
        "z_min"              => minimum(cat.z_obs),
        "z_max"              => zmax,
        "b1"                 => b1,
        "seed"               => seed,
        "fixed_amplitude"    => fixed_amplitude,
        "n_quasars"          => length(cat),
    )
end

"""
    constrained_ic_box(cat, randoms, cosmo; L_box, res_constrain=256, res_box=1024, b1=2.5,
                       n_order=1, seed=101, device=identity, z_fixed=nothing,
                       fixed_amplitude=true, rounds=3, phase_iters=25, chi_iters=25, R=nothing)
        -> (; ω_box, res_box, L_box, manifest, φ_coarse, z)

One Quaia-constrained periodic IC box.  Builds the geometry/forward at `res_constrain` in the
periodic box of side `L_box` (≥ the survey extent, survey centered), constrains the fixed-amplitude
phases with `reconstruct_quaia` (`device=CuArray` for the GPU), then `refine_phases` realizes the box
at `res_box`.  `z_fixed` (e.g. ẑ from a prior `quaia_ensemble`) fixes the radial positions and skips
the χ-solve (redshifts are physical / box-independent), so only the φ-MAP runs in the big box.
"""
function constrained_ic_box(cat::QuaiaCatalog, randoms, cosmo; L_box::Real,
                            res_constrain::Int=256, res_box::Int=1024, b1::Real=2.5,
                            n_order::Int=1, seed::Integer=101, device=identity,
                            z_fixed=nothing, fixed_amplitude::Bool=true,
                            rounds::Int=3, phase_iters::Int=25, chi_iters::Int=25, R=nothing)
    pk   = linear_power_spectrum(cosmo)
    geom = box_geometry(randoms, cosmo; res=res_constrain, boxsize=L_box)
    L    = geom.boxsize; dx = L / res_constrain
    W    = survey_window(geom, randoms)
    Rsm  = R === nothing ? max(2dx, 80.0) : Float64(R)
    gm   = galaxy_model(res_constrain, L, cosmo, pk; R=Rsm, observer=geom.observer,
                        a_far=geom.a_far, a_near=geom.a_near, n_order=n_order, rsd=false)
    # z_fixed → fix the radial positions at ẑ and run only the φ-MAP (chi_iters=0).
    ci = chi_iters
    if z_fixed !== nothing
        cat = QuaiaCatalog(cat.ra, cat.dec, eltype(cat.z_obs).(z_fixed), cat.σz)
        ci  = 0
    end
    prob = quaia_problem(cat, geom, gm, W; b1=b1)
    r    = reconstruct_quaia(prob, seed; device=device, rounds=rounds,
                             phase_iters=phase_iters, chi_iters=ci, b1=b1)
    ω_box = refine_phases(r.φ, res_box; seed=seed, fixed_amplitude=fixed_amplitude)
    manifest = _ic_box_manifest(geom, cosmo, res_constrain, res_box, b1, seed, cat, fixed_amplitude)
    return (; ω_box=ω_box, res_box=res_box, L_box=L, manifest=manifest, φ_coarse=r.φ, z=r.z)
end

"""
    constrained_ic_box_ensemble(cat, randoms, cosmo; seeds, save=nothing, kw...) -> Vector

A constrained-realization ensemble: each seed shares the central / large-scale Quaia constraint but
gets **independent** unconstrained (high-k + outer-shell) phases — a valid uniform-random draw outside
the data.  `save(k, seed, box)` is called per realization (e.g. `export_white_noise`).  `kw...` pass
through to `constrained_ic_box`."""
function constrained_ic_box_ensemble(cat::QuaiaCatalog, randoms, cosmo; seeds, save=nothing, kw...)
    out = Vector{Any}(undef, length(seeds))
    for (k, seed) in enumerate(seeds)
        box = constrained_ic_box(cat, randoms, cosmo; seed=seed, kw...)
        out[k] = box
        save === nothing || save(k, seed, box)
    end
    return out
end

"""    export_white_noise(path, box; T=Float32) -> (; field, manifest)

Write the constrained white-noise box as a **raw little-endian `T` array** (column-major, res_box³
contiguous) to `path`, and the provenance manifest (res, boxsize, cosmology, observer,
constrained_radius=χ(z_max), seed, …) to `path * ".manifest.npz"`.  Raw binary — not npz — because the
ZIP32 container behind `.npz` cannot hold a ≥4 GB array (a 1024³ Float32 grid is exactly 4 GB) and raw
white-noise is what an external IC code (MUSIC "white noise from file", N-GenIC) ingests anyway; match
its grid ordering / normalization convention on first external use (flagged in the manifest).

The compact `φ_coarse` (a few tens of MB) plus the manifest fully determine the box — `refine_phases`
re-realizes ω at any resolution — so saving the carrier alone is the storage-light option."""
function export_white_noise(path::AbstractString, box; T::Type=Float32)
    open(io -> write(io, Array{T}(box.ω_box)), path, "w")
    mpath = path * ".manifest.npz"
    npzwrite(mpath, box.manifest)
    return (field = path, manifest = mpath)
end

"""    ic_box_snapshot(ω_box, L_box, cosmo, a_init; n_order=2, exact_growth=true) -> pos::(res,res,res,3)

Optional convenience (NOT the default output): evolve the constrained white-noise box to a single
scale factor `a_init` with nLPT and return periodic-wrapped particle positions for a sim snapshot —
`ic_operator` → `white_noise_to_fphi` → `lpt_displacement` → `q + ψ (mod L)`."""
function ic_box_snapshot(ω_box::AbstractArray{<:Real,3}, L_box::Real, cosmo, a_init::Real;
                         n_order::Int=2, exact_growth::Bool=true)
    res  = size(ω_box, 1); pk = linear_power_spectrum(cosmo)
    op   = ic_operator(res, L_box, pk)
    fphi = white_noise_to_fphi(op, ω_box)
    K    = nlpt_kernels(res, L_box)
    ψ    = lpt_displacement(fphi, K, cosmo, a_init; n_order=n_order, exact_growth=exact_growth)
    q    = lagrangian_grid_3d(res, L_box)
    return mod.(q .+ ψ, L_box)
end
