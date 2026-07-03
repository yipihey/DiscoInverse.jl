"""
Cosmicflows-4 group catalog IO + peculiar-velocity error model (for the `VelocityConstraint`).

CF4 (Tully et al. 2023) gives distances (distance moduli DM ± e_DM) and CMB-frame velocities Vcmb for
55 877 galaxies in 38 065 groups.  The distance error is **lognormal** — Gaussian in DM, i.e. a fixed
*fractional* distance error σ_lnd = ln(10)/5 · e_DM (~0.23 for the 0.5-mag median) — so the peculiar-
velocity noise σ_v = H₀·d·σ_lnd grows ∝ distance (median ~3500 km/s: CF4 is deeply noise-dominated
beyond the Local Volume, which is why the field must be sampled, not MAP'd — see the HMC path).

The observed radial peculiar velocity is v_pec = Vcmb − H₀·d.  H₀ is taken as the effective Vcmb–d slope
of the sample (≈72), which zero-points the peculiar velocities (its residual monopole is marginalized by
the constraint's `submean`); this decouples the reconstruction from the distance-scale / H₀-tension.
The **homogeneous Malmquist** correction d ← d·exp(3.5 σ_lnd²) de-biases the lognormal-scattered distances
(more volume at larger d ⇒ observed distances are biased low).  CF4's own `Vpec` is a smoothed/model-based
estimate (2300 km/s residual vs the raw Vcmb−H₀d) and is deliberately NOT used — we forward-model the raw.

Staged once to `.npz` (the package interchange format) with columns `ra, dec, dist, e_dm, vcmb, ngal`.
"""

using NPZ

struct CF4Catalog{T<:AbstractFloat}
    ra::Vector{T}      # degrees
    dec::Vector{T}     # degrees
    dist::Vector{T}    # Mpc (distance from DMav)
    e_dm::Vector{T}    # distance-modulus error [mag]
    vcmb::Vector{T}    # CMB-frame velocity [km/s]
    ngal::Vector{T}    # galaxies in the group
end

Base.length(cat::CF4Catalog) = length(cat.dist)

"""    load_cf4_groups(path; dist_min=1, dist_max=Inf, ngal_min=1, T=Float64) -> CF4Catalog

Load the staged CF4 groups `.npz`.  `dist_max` cuts the noisy far tail (σ_v ∝ d); `ngal_min` keeps only
multi-galaxy groups (lower distance error) if desired."""
function load_cf4_groups(path::AbstractString; dist_min::Real=1.0, dist_max::Real=Inf,
                         ngal_min::Real=1, T::Type{<:AbstractFloat}=Float64)
    d = npzread(path)
    ra=T.(d["ra"]); dec=T.(d["dec"]); dist=T.(d["dist"]); edm=T.(d["e_dm"]); vcmb=T.(d["vcmb"]); ng=T.(d["ngal"])
    keep = isfinite.(dist) .& (dist .> dist_min) .& (dist .< dist_max) .& isfinite.(edm) .& (edm .> 0) .&
           isfinite.(vcmb) .& isfinite.(ra) .& (ng .>= ngal_min)
    return CF4Catalog{T}(ra[keep], dec[keep], dist[keep], edm[keep], vcmb[keep], ng[keep])
end

"""    cf4_hubble(cat) -> H₀  — effective Vcmb–distance slope Σ(V·d)/Σ(d²) [km/s/Mpc]"""
cf4_hubble(cat::CF4Catalog) = sum(cat.vcmb .* cat.dist) / sum(cat.dist .^ 2)

"""
    cf4_peculiar_velocity(cat; H0=nothing, malmquist=true) -> (; dist, vpec, σv, H0)

Radial peculiar velocity and its lognormal-distance-error velocity noise per group:
`σ_lnd = ln(10)/5·e_DM`, homogeneous-Malmquist `d ← d·exp(3.5 σ_lnd²)`, `v_pec = Vcmb − H₀·d`,
`σ_v = H₀·d·σ_lnd` (all km/s / Mpc).  `H0` defaults to `cf4_hubble(cat)`."""
function cf4_peculiar_velocity(cat::CF4Catalog{T}; H0=nothing, malmquist::Bool=true) where {T}
    σlnd = T(log(10) / 5) .* cat.e_dm
    dist = malmquist ? cat.dist .* exp.(T(3.5) .* σlnd .^ 2) : cat.dist
    H0v  = H0 === nothing ? T(sum(cat.vcmb .* dist) / sum(dist .^ 2)) : T(H0)   # fit on the distances used
    vpec = cat.vcmb .- H0v .* dist
    σv   = H0v .* dist .* σlnd
    return (dist=dist, vpec=vpec, σv=σv, H0=H0v)
end

"""    cf4_box_geometry(cat, cosmo; res, pad_frac=0.2, boxsize=nothing) -> BoxGeometry

Box built from the CF4 groups (redshift z = Vcmb/c), observer at the box centre — for a CF4-only or
CF4-anchored local reconstruction."""
function cf4_box_geometry(cat::CF4Catalog{T}, cosmo; res::Int, pad_frac::Real=0.2, boxsize=nothing) where {T}
    z = max.(cat.vcmb ./ T(299792.458), T(1e-4))
    return box_geometry((ra=cat.ra, dec=cat.dec, z=z), cosmo; res=res, pad_frac=pad_frac, boxsize=boxsize)
end

"""
    cf4_velocity_constraint(gm, geom, cosmo, cat; H0=nothing, malmquist=true, submean=true) -> VelocityConstraint

Place each CF4 group at its real-space comoving position (distance → `d·h` [Mpc/h] along its sky
direction) in `geom`'s box, and attach the radial peculiar velocity + distance-error velocity noise.
Drop-in for `multitracer_problem(gm, tracers; velocity=…)`."""
function cf4_velocity_constraint(gm::GalaxyModel{T}, geom, cosmo, cat::CF4Catalog;
                                 H0=nothing, malmquist::Bool=true, submean::Bool=true) where {T}
    pv  = cf4_peculiar_velocity(cat; H0=H0, malmquist=malmquist)
    rar = deg2rad.(T.(cat.ra)); decr = deg2rad.(T.(cat.dec)); cd = cos.(decr)
    û   = hcat(cd .* cos.(rar), cd .* sin.(rar), sin.(decr))     # (N,3) unit sky directions
    χ   = T.(pv.dist) .* T(cosmo.h)                              # Mpc → Mpc/h comoving (local)
    pts = embed(geom, û .* reshape(χ, :, 1))                     # (N,3) box coordinates
    return velocity_constraint(gm, geom, cosmo, pts, pv.vpec; sigma_v=pv.σv, submean=submean)
end
