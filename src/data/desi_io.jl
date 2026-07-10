"""
DESI DR1 QSO clustering IO (a high-z tracer; only its z<z_box shell is in the res-384 box).

The DESI DR1 QSO clustering sample (`QSO_{NGC,SGC}_clustering.dat.fits`, z∈[0.80, 3.50]) staged
as `desi_qso_data.npz` (`ra, dec, z, weight`), with `desi_qso_randoms.npz` (`ra, dec, z`) the
matching angular+radial randoms.  The bulk sits at z~1–2 (outside the fixed-cube res-384 box,
χ(0.87)≈2058 Mpc/h = box_half), so at res 384 only the z∈[0.80, z_box] shell enters — pass
`zmax = z_box` to cut both galaxies and randoms.  At res 512+ (box → z<1.3) the full sample is
in-box.  `WEIGHT` (the DESI completeness×systematics weight) rides through as the per-galaxy `u`.
"""

struct DESIQSOCatalog{T<:AbstractFloat}
    ra::Vector{T}; dec::Vector{T}; z::Vector{T}; weight::Vector{T}
    randoms::@NamedTuple{ra::Vector{T}, dec::Vector{T}, z::Vector{T}}
end
Base.length(cat::DESIQSOCatalog) = length(cat.z)

"""    load_desi_qso(data_path, randoms_path; zmin=0.0, zmax=Inf, T=Float64) -> DESIQSOCatalog

Load the DESI QSO `.npz` (galaxies `ra, dec, z, weight`) + randoms `.npz` (`ra, dec, z`), with a
`[zmin, zmax]` cut applied to both (use `zmax = z_box` for the in-box shell).
"""
function load_desi_qso(data_path::AbstractString, randoms_path::AbstractString;
                       zmin::Real=0.0, zmax::Real=Inf, T::Type{<:AbstractFloat}=Float64)
    d = npzread(data_path)
    ra, dec, z = T.(d["ra"]), T.(d["dec"]), T.(d["z"])
    w = haskey(d, "weight") ? T.(d["weight"]) : ones(T, length(z))
    keep = (z .≥ T(zmin)) .& (z .≤ T(zmax)) .& isfinite.(z)
    r = npzread(randoms_path)
    rra, rdec, rz = T.(r["ra"]), T.(r["dec"]), T.(r["z"])
    rkeep = (rz .≥ T(zmin)) .& (rz .≤ T(zmax)) .& isfinite.(rz)
    randoms = (ra=rra[rkeep], dec=rdec[rkeep], z=rz[rkeep])
    return DESIQSOCatalog{T}(ra[keep], dec[keep], z[keep], w[keep], randoms)
end
