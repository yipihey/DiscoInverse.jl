"""
Embed the curved CMASS-South wedge into a comoving Cartesian box for the IC field.

`(ra, dec, z)` → comoving Cartesian (observer at the origin) via the fiducial
cosmology's χ(z); then a box is sized to contain the survey (from the randoms) with
padding, and a shift maps comoving → box coordinates.  The forward model's IC field
lives on `[0,L)³`; the observer sits at the box-relative position `geom.observer`
(outside the box), and the lightcone shell `[a_far, a_near]` is set by the z-range.
"""

"""
    fiducial_cosmology(; Omega_m=0.315, Omega_b=0.049, h=0.674, sigma8=0.81, n_s=0.965)

ECHOES fiducial cosmology (matches the catalog's comoving convention), with timetables
computed so `comoving_distance` works.
"""
function fiducial_cosmology(; Omega_m::Real=0.315, Omega_b::Real=0.049, h::Real=0.674,
                            sigma8::Real=0.81, n_s::Real=0.965, T::Type{<:AbstractFloat}=Float64)
    c = Cosmology{T}(; Omega_c=T(Omega_m - Omega_b), Omega_b=T(Omega_b), h=T(h),
                     sigma8=T(sigma8), n_s=T(n_s))
    return compute_timetables(c)
end

"""
    radec_z_to_cartesian(ra, dec, z, cosmo) -> (N,3) comoving [Mpc/h], observer at origin
"""
function radec_z_to_cartesian(ra::AbstractVector{T}, dec::AbstractVector{T},
                              z::AbstractVector{T}, cosmo) where {T}
    χ   = [T(comoving_distance(cosmo, 1 / (1 + zi))) for zi in z]
    rar = deg2rad.(ra); decr = deg2rad.(dec)
    cd  = cos.(decr)
    return hcat(χ .* cd .* cos.(rar), χ .* cd .* sin.(rar), χ .* sin.(decr))
end

struct BoxGeometry{T<:AbstractFloat, CO}
    res::Int
    boxsize::T
    shift::Vector{T}        # x_box = x_comoving + shift
    observer::Vector{T}     # comoving origin in box coords (= shift)
    a_far::T                # = 1/(1+zmax)
    a_near::T               # = 1/(1+zmin)
    cosmo::CO
end

"""
    box_geometry(randoms, cosmo; res, pad_frac=0.15, boxsize=nothing) -> BoxGeometry

Size the comoving box from the survey randoms (the full footprint) with fractional
padding, and place the observer.  `randoms` = `(; ra, dec, z)`.

Pass an explicit `boxsize` (≥ the survey extent) to make a **periodic box larger than the
survey** with the footprint kept CENTERED (`shift = L/2 − center`, observer at box center) —
the constrained-IC-box mode (`forward/constrained_box.jl`), where the survey occupies the
central sub-region and the rest is unconstrained padding.  `nothing` keeps the footprint
auto-sizing.  `a_far`/`a_near` are redshift-derived and unaffected by the box size.
"""
function box_geometry(randoms, cosmo; res::Int, pad_frac::Real=0.15, boxsize=nothing)
    T = eltype(randoms.ra)
    cart = radec_z_to_cartesian(randoms.ra, randoms.dec, randoms.z, cosmo)
    lo = vec(minimum(cart; dims=1)); hi = vec(maximum(cart; dims=1))
    extent = maximum(hi .- lo)
    L = boxsize === nothing ? extent * (1 + 2 * T(pad_frac)) : T(boxsize)
    boxsize === nothing || L ≥ extent ||
        error("boxsize ($(L) Mpc/h) must be ≥ the survey extent ($(extent) Mpc/h)")
    center = (lo .+ hi) ./ 2
    shift  = L / 2 .- center                 # center the survey in the box
    a_far  = T(1 / (1 + maximum(randoms.z)))  # high z → small a (far)
    a_near = T(1 / (1 + minimum(randoms.z)))
    return BoxGeometry{T, typeof(cosmo)}(res, T(L), shift, copy(shift), a_far, a_near, cosmo)
end

"""    embed(geom, cart) -> (N,3) box coordinates"""
embed(geom::BoxGeometry, cart::AbstractMatrix) = cart .+ reshape(geom.shift, 1, 3)

"""    embed_radec_z(geom, ra, dec, z) -> (N,3) box coordinates"""
embed_radec_z(geom::BoxGeometry, ra, dec, z) =
    embed(geom, radec_z_to_cartesian(ra, dec, z, geom.cosmo))
