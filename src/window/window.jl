"""
Survey window W(x) and number-binning on the inference mesh.

The window is the (normalised) random density CIC-deposited onto the same mesh as the
forward model; it encodes both the angular footprint and the radial selection n̄(χ)
(CMASS-South is ~99% complete, so the randoms are ≈uniform).  Cells with W≈0 are
outside the survey and excluded from the likelihood.
"""

"""
    survey_window(geom, randoms) -> W::(res,res,res)

CIC-deposit the survey randoms onto the mesh and normalise so the mean over the
footprint (W>0 cells) is 1.  W is data (held fixed; not differentiated).
"""
function survey_window(geom::BoxGeometry{T}, randoms) where {T}
    xb = embed_radec_z(geom, randoms.ra, randoms.dec, randoms.z)
    W  = cic_deposit(xb, ones(T, size(xb, 1)), geom.res, geom.boxsize)
    occ = W .> 0
    return W ./ (sum(W) / sum(occ))           # ⟨W⟩ over footprint = 1
end

"""
    bin_galaxies(geom, ra, dec, z; weights=nothing) -> counts::(res,res,res)

CIC-deposit a galaxy catalog (optionally per-galaxy weighted) onto the mesh → the
binned count field used as the Poisson data.
"""
function bin_galaxies(geom::BoxGeometry{T}, ra, dec, z;
                      weights::Union{Nothing,AbstractVector}=nothing) where {T}
    xb = embed_radec_z(geom, ra, dec, z)
    w  = weights === nothing ? ones(T, size(xb, 1)) : T.(weights)
    return cic_deposit(xb, w, geom.res, geom.boxsize)
end
