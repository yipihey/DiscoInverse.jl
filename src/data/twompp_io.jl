"""
2M++ (2MASS++ redshift compilation; Lavaux & Hudson 2011) IO — the local (z<0.1) density tracer.

The staged `local_2mpp_observed.npz` (from the ECHOES local-neighbourhood line) carries `ra, dec, cz,
dist_mpc, ksmag, prov` for the Ks<11.5 magnitude-limited sample (ZoA |b|<5° cut, d<300 Mpc).  2M++ is
magnitude-limited (no survey randoms), so `load_twompp` synthesizes a selection window: randoms with the
observed radial n(cz) (bootstrap of the galaxy redshifts, which averages out clustering over many draws)
and a uniform full-sky angular distribution — i.e. window ≈ angular_mask × n̄(r).  Redshift z = cz/c.
"""

const _C_KMS = 299792.458

struct TwoMPPCatalog{T<:AbstractFloat}
    ra::Vector{T}; dec::Vector{T}; z::Vector{T}
    randoms::@NamedTuple{ra::Vector{T}, dec::Vector{T}, z::Vector{T}}
end
Base.length(cat::TwoMPPCatalog) = length(cat.z)

"""    load_twompp(path; zmin=1e-4, zmax=0.08, n_randoms=400_000, seed=0, T=Float64) -> TwoMPPCatalog

Load the staged 2M++ `.npz` (`ra, dec, cz`).  z = cz/c cut to `[zmin, zmax]`.  Randoms are drawn with the
observed radial n(cz) (bootstrap) × uniform sky → the magnitude-limited selection window for `survey_window`.
"""
function load_twompp(path::AbstractString; zmin::Real=1e-4, zmax::Real=0.08,
                     n_randoms::Integer=400_000, seed::Integer=0, T::Type{<:AbstractFloat}=Float64)
    d = npzread(path)
    ra, dec = T.(d["ra"]), T.(d["dec"])
    z = T.(d["cz"]) ./ T(_C_KMS)
    keep = (z .≥ T(zmin)) .& (z .≤ T(zmax)) .& isfinite.(z) .& isfinite.(ra)
    ra, dec, z = ra[keep], dec[keep], z[keep]
    rng = MersenneTwister(seed)
    rz  = z[rand(rng, 1:length(z), n_randoms)]                      # radial n(z) from the data (bootstrap)
    rra = T(360) .* rand(rng, T, n_randoms)                         # uniform sky
    rdec = T.(rad2deg.(asin.(2 .* rand(rng, T, n_randoms) .- 1)))
    return TwoMPPCatalog{T}(ra, dec, z, (ra=rra, dec=rdec, z=rz))
end
