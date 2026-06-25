"""
ECHOES catalog / randoms IO.

The completed-catalog draws (`draw_samples.draw(pkg, seed)`) are produced once in
Python and stored in a realizations `.npz` (per-seed `ra_k,dec_k,z_k,prov_k`); the
survey randoms are the released `cmass_south_randoms.npz` (`ra,dec,z`).  Loaded here as
plain `Float64` arrays.  PROV: 0 = observed (fixed/hard), 1/2 = completed missing-z at a
real position (soft), 3 = systot synthetic, 5 = mask-hole inpaint.
"""

using NPZ

struct EchoesCatalog{T<:AbstractFloat}
    ra::Vector{T}      # degrees
    dec::Vector{T}     # degrees
    z::Vector{T}       # redshift
    prov::Vector{Int8}
end

"""    load_echoes_realization(path, seed; T=Float64) -> EchoesCatalog"""
function load_echoes_realization(path::AbstractString, seed::Integer; T::Type{<:AbstractFloat}=Float64)
    d = npzread(path)
    EchoesCatalog{T}(T.(d["ra_$seed"]), T.(d["dec_$seed"]), T.(d["z_$seed"]), Int8.(d["prov_$seed"]))
end

"""    n_realizations(path) -> Int"""
n_realizations(path::AbstractString) = Int(npzread(path)["K"])

"""    load_echoes_randoms(path; T=Float64) -> (; ra, dec, z)"""
function load_echoes_randoms(path::AbstractString; T::Type{<:AbstractFloat}=Float64)
    d = npzread(path)
    (ra=T.(d["ra"]), dec=T.(d["dec"]), z=T.(d["z"]))
end

"""    prov_mask(cat, group) -> BitVector  (group ∈ :observed, :completed, :all)"""
function prov_mask(cat::EchoesCatalog, group::Symbol)
    group === :observed  ? (cat.prov .== 0) :
    group === :completed ? (cat.prov .>= 1) :
    group === :all       ? trues(length(cat.prov)) :
    error("group must be :observed, :completed, or :all")
end
