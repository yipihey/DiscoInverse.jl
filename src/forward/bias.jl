"""
2nd-order Lagrangian bias operators on the linear field.

From the initial Fourier potential φ(k) (the faithful −1/k² gauge from
`white_noise_to_fphi`) we build, all as differentiable FFT broadcasts:

  * the linear density  δ_L(k) = −k²·φ(k)·W_R(k)   (Gaussian-smoothed at Lagrangian
    scale R),
  * the tidal shear     s_ij = (kᵢkⱼ/k² − δᵢⱼ/3)·δ_L,  s² = Σ_ij s_ij²,
  * the per-Lagrangian-particle galaxy weight (McDonald & Roy / Lagrangian bias)
      w(q) = 1 + b₁ δ_L + (b₂/2)(δ_L² − σ²) + b_{s2}(s² − ⟨s²⟩),

with σ² = ⟨δ_L²⟩ and ⟨s²⟩ field means held off the AD tape (they renormalise the
operators and depend on P(k),R, not on the realisation at leading order).
"""

# ── Precomputed spectral operators (k², kᵢkⱼ/k², Gaussian smoothing) ───────────
struct BiasOperators{T<:AbstractFloat, A<:AbstractArray{T,3}}
    res::Int
    boxsize::T
    R::T
    k2::A
    kxkx::A; kyky::A; kzkz::A; kxky::A; kxkz::A; kykz::A   # kᵢkⱼ/k² (0 at DC)
    W_R::A
end

"""
    bias_operators(res, boxsize, R; T=Float64) -> BiasOperators

Build the spectral bias operators on the rfft grid; `R` is the Gaussian Lagrangian
smoothing scale [Mpc/h].
"""
function bias_operators(res::Int, boxsize::Real, R::Real; T::Type{<:AbstractFloat}=Float64)
    grid = get_fourier_grid(res, T(boxsize); T=T)
    kx, ky, kz = grid.k_vecs                      # reshaped 1-D (res,1,1),(1,res,1),(1,1,res÷2+1)
    k2 = grid.k2
    invk2 = @. ifelse(k2 == 0, zero(T), inv(k2))
    KX = kx .* ones(T, 1, res, 1) .* ones(T, 1, 1, res ÷ 2 + 1)   # broadcast to full rfft shape
    KY = ky .* ones(T, res, 1, 1) .* ones(T, 1, 1, res ÷ 2 + 1)
    KZ = kz .* ones(T, res, 1, 1) .* ones(T, 1, res, 1)
    bias = BiasOperators{T, typeof(k2)}(res, T(boxsize), T(R), k2,
        (KX .* KX) .* invk2, (KY .* KY) .* invk2, (KZ .* KZ) .* invk2,
        (KX .* KY) .* invk2, (KX .* KZ) .* invk2, (KY .* KZ) .* invk2,
        @.(exp(-k2 * T(R)^2 / 2)))
    return bias
end

# ── Fixed physical Lagrangian mass scale ──────────────────────────────────────
# The sheet's smallest resolved scale is anchored to an ABSOLUTE Lagrangian mass (solar
# masses, no little-h) so that bias and LPT-order comparisons are made at one fixed physical
# scale across resolutions, boxes, and surveys. The chosen anchor (Tom, 2026-07): the Gaussian
# filter mass M = ρ̄_m (2π)^{3/2} R³ = 1e14 M⊙, giving R_G ≈ 5.43 physical Mpc (3.66 Mpc/h at
# Planck-2018 h). At z=1 this scale is ~linear (~0.1% of the sheet mass shell-crossed) and 2LPT
# vs 3LPT agree to ~1.3%; at z=0 ~6% has shell-crossed. Set the `galaxy_model` smoothing to
# `smoothing_from_mass(ANCHOR_MASS, cosmo)` and use a grid fine enough to resolve it (Δq ≲ R/1.8).
const ANCHOR_MASS = 1e14        # M⊙ (physical, no h) — the fixed Lagrangian mass scale

"""    smoothing_from_mass(M_sun, cosmo) -> R [Mpc/h]

Gaussian Lagrangian smoothing whose enclosed mass M = ρ̄_m·(2π)^{3/2}·R³ equals `M_sun`
(physical solar masses, no h).  ρ̄_m = Ω_m·ρ_crit,0 with ρ_crit,0 = 2.775e11·h² M⊙/Mpc³.
Inverse of [`mass_from_smoothing`]."""
function smoothing_from_mass(M_sun::Real, cosmo)
    Ωm = cosmo.Omega_c + cosmo.Omega_b
    ρm = Ωm * 2.775e11 * cosmo.h^2                 # physical M⊙/Mpc³
    Rphys = cbrt(M_sun / (ρm * (2π)^1.5))          # physical Mpc
    return Rphys * cosmo.h                          # Mpc/h (box units)
end

"""    mass_from_smoothing(R_h, cosmo) -> M [M⊙]  — Gaussian-filter Lagrangian mass of scale `R_h` [Mpc/h]."""
function mass_from_smoothing(R_h::Real, cosmo)
    Ωm = cosmo.Omega_c + cosmo.Omega_b
    ρm = Ωm * 2.775e11 * cosmo.h^2
    return ρm * (2π)^1.5 * (R_h / cosmo.h)^3
end

"""    tophat_radius_from_mass(M_sun, cosmo) -> R_TH [phys Mpc]

Real-space top-hat radius, M = (4π/3)·ρ̄_m·R³ (the Press-Schechter/halo mass radius).  For the
ANCHOR_MASS=1e14 M⊙ at Planck-2018 this is 8.44 physical Mpc (vs the Gaussian smoothing 5.43
phys Mpc = R_TH/1.555 that the code's `W_R` actually applies)."""
function tophat_radius_from_mass(M_sun::Real, cosmo)
    Ωm = cosmo.Omega_c + cosmo.Omega_b
    ρm = Ωm * 2.775e11 * cosmo.h^2                 # physical M⊙/Mpc³
    return cbrt(3 * M_sun / (4π * ρm))             # physical Mpc
end

# Device-aware irfft (CPU [3,1,2]; CuArray permute-wrapped via the DiscoDJNative ext).
_irfftn(f, res) = DiscoDJNative._irfftn(f, res)

"""
    bias_fields(fphi, ops) -> (δ_L, s²)

Real-space linear density and tidal-shear-squared (each `(res,res,res)`), smoothed
at the Lagrangian scale R.  Differentiable w.r.t. `fphi` (→ ω).
"""
function bias_fields(fphi::AbstractArray{Complex{T},3}, ops::BiasOperators{T}) where {T}
    res = ops.res
    fdL = (.-ops.k2 .* ops.W_R) .* fphi           # δ_L(k) = −k²·W_R·φ
    δL  = _irfftn(fdL, res)
    sxx = _irfftn(ops.kxkx .* fdL, res) .- δL ./ 3
    syy = _irfftn(ops.kyky .* fdL, res) .- δL ./ 3
    szz = .-(sxx .+ syy)                           # tidal tensor is trace-free → no 7th irfft
    sxy = _irfftn(ops.kxky .* fdL, res)
    sxz = _irfftn(ops.kxkz .* fdL, res)
    syz = _irfftn(ops.kykz .* fdL, res)
    s2  = sxx.^2 .+ syy.^2 .+ szz.^2 .+ 2 .* (sxy.^2 .+ sxz.^2 .+ syz.^2)
    return δL, s2
end

"""
    bias_moments(δL, s2) -> (σ², ⟨s²⟩)

Reference field moments for the 2nd-order operators.  Compute ONCE (from a reference
field or theory) and hold fixed across the inference — the Lagrangian bias subtracts
the *ensemble* variance σ²(R), not the per-realization mean.
"""
bias_moments(δL::AbstractArray{T,3}, s2::AbstractArray{T,3}) where {T} = (mean(δL .^ 2), mean(s2))

"""
    bias_weight(δL, s2, σ², s2mean; b1, b2, bs2) -> w(q)::(res,res,res)

2nd-order Lagrangian galaxy weight per particle.  `σ²`, `s2mean` are the FIXED
reference moments (from `bias_moments`) — passed in, not recomputed, so the operators
δ_L²−σ² and s²−⟨s²⟩ renormalise against a constant.  Differentiable w.r.t. δ_L, s²
(→ ω) and the bias params.
"""
function bias_weight(δL::AbstractArray{T,3}, s2::AbstractArray{T,3},
                     σ²::Real, s2mean::Real; b1::Real, b2::Real, bs2::Real) where {T}
    return @. 1 + T(b1) * δL + T(b2) / 2 * (δL^2 - T(σ²)) + T(bs2) * (s2 - T(s2mean))
end
