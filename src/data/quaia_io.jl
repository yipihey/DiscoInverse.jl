"""
Quaia quasar catalog IO (field-level redshift reconstruction).

Quaia (Gaia×unWISE; Storey-Fisher et al. 2024) gives **spectro-photometric** redshifts with
large per-object errors — σ_z ~ 0.03–0.12, i.e. ~120 Mpc/h of radial blur — while the sky
positions (ra, dec) are exact.  We reconstruct zero-redshift-error realizations by inferring
each quasar's true radial distance from the 3D ΛCDM clustering (see `forward/quaia.jl`).

The catalog is staged once to an `.npz` (the package's interchange format, as for the ECHOES
randoms) with columns `ra, dec` (degrees), `z` (photo-z), `zerr` (σ_z), the Galactic-plane cut
(|b|≥10°) and finite/z>0 selection already applied when the npz is built from `quaia_G20.0.fits`.
`load_quaia` reapplies a defensive finite/σ_z>0 filter; pass `b`/`l` columns to cut here instead.
"""

using NPZ

struct QuaiaCatalog{T<:AbstractFloat}
    ra::Vector{T}      # degrees
    dec::Vector{T}     # degrees
    z_obs::Vector{T}   # photometric redshift
    σz::Vector{T}      # photo-z error
end

Base.length(cat::QuaiaCatalog) = length(cat.z_obs)

"""    load_quaia(path; b_min=10, T=Float64) -> QuaiaCatalog

Load the staged Quaia `.npz` (`ra, dec, z, zerr`).  Keeps only finite rows with `z>0` and
`zerr>0`; if a galactic-latitude column `b` is present, also applies `|b| ≥ b_min` (degrees).
"""
function load_quaia(path::AbstractString; b_min::Real=10, T::Type{<:AbstractFloat}=Float64)
    d   = npzread(path)
    ra  = T.(d["ra"]); dec = T.(d["dec"]); z = T.(d["z"]); zerr = T.(d["zerr"])
    keep = isfinite.(ra) .& isfinite.(dec) .& isfinite.(z) .& isfinite.(zerr) .& (z .> 0) .& (zerr .> 0)
    haskey(d, "b") && (keep .&= abs.(T.(d["b"])) .>= T(b_min))
    return QuaiaCatalog{T}(ra[keep], dec[keep], z[keep], zerr[keep])
end

"""    load_quaia_randoms(path; T=Float64) -> (; ra, dec, z)

The Quaia angular+radial randoms (footprint × n̄(z)); same `(ra, dec, z)` npz layout as the
ECHOES randoms, used to size the all-sky box (`box_geometry`) and the window (`survey_window`).
"""
load_quaia_randoms(path::AbstractString; T::Type{<:AbstractFloat}=Float64) =
    load_echoes_randoms(path; T=T)
