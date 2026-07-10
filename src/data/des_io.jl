"""
DES Y3 redMaGiC galaxy-clustering IO (a res-384-in-box tracer for the joint reconstruction).

The DES Y3 `y3a2_gold2.2.1_redmagic_highdens` sample (Rozo et al. 2016 red-sequence
photometric LRGs), staged as `des_redmagic_fullz.npz` with the galaxy `(ra, dec, z)` and the
survey `(rra, rdec, rz)` randoms in one file.  z∈[0.15, 0.65] sits entirely inside the fixed
16×10¹³ M⊙ / res-384 box (χ(0.65) ≈ 1560 Mpc/h < box_half 2058), so the whole sample is a
clean in-box clustering tracer — no redshift cut needed.  `load_des_redmagic` returns the
galaxy catalogue and the randoms in the `(; ra, dec, z)` layout `box_geometry` / `survey_window`
/ `tracer` consume.
"""

struct DESRedmagicCatalog{T<:AbstractFloat}
    ra::Vector{T}; dec::Vector{T}; z::Vector{T}
    randoms::@NamedTuple{ra::Vector{T}, dec::Vector{T}, z::Vector{T}}
end
Base.length(cat::DESRedmagicCatalog) = length(cat.z)

"""    load_des_redmagic(path; zmin=0.0, zmax=Inf, n_randoms=nothing, T=Float64) -> DESRedmagicCatalog

Load the DES redMaGiC `.npz` (`ra, dec, z` galaxies + `rra, rdec, rz` randoms).  Optional
`[zmin, zmax]` cut (applied to both galaxies and randoms); `n_randoms` subsamples the (14 M)
randoms for a lighter window build (the window only needs the footprint × n̄(z) sampled well).
"""
function load_des_redmagic(path::AbstractString; zmin::Real=0.0, zmax::Real=Inf,
                           n_randoms::Union{Nothing,Integer}=nothing, T::Type{<:AbstractFloat}=Float64)
    d = npzread(path)
    ra, dec, z = T.(d["ra"]), T.(d["dec"]), T.(d["z"])
    keep = (z .≥ T(zmin)) .& (z .≤ T(zmax)) .& isfinite.(z)
    rra, rdec, rz = T.(d["rra"]), T.(d["rdec"]), T.(d["rz"])
    rkeep = (rz .≥ T(zmin)) .& (rz .≤ T(zmax)) .& isfinite.(rz)
    ridx = findall(rkeep)
    if n_randoms !== nothing && length(ridx) > n_randoms
        ridx = ridx[round.(Int, range(1, length(ridx); length=n_randoms))]   # deterministic thin
    end
    randoms = (ra=rra[ridx], dec=rdec[ridx], z=rz[ridx])
    return DESRedmagicCatalog{T}(ra[keep], dec[keep], z[keep], randoms)
end
