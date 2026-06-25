"""
The differentiable galaxy-field forward model.

`galaxy_density(gm, ω, b)` chains the whole forward:

    ω → φ(k)                                   [white_noise_to_fphi]
      → ψ exact-growth shapes                  [compute_core_exact, n_order=3]
      → {δ_L, s²}                              [bias_fields]
      → w(q)=1+b₁δ_L+(b₂/2)(δ_L²−σ²)+b_s2(s²−⟨s²⟩)   [bias_weight]
      → per-particle lightcone crossing x_obs(q) (+ optional RSD)   [lightcone_cross_ad]
      → tetrahedral CDM-sheet deposit of mass·w  →  n_g(x)          [sheet_deposit]

The result `n_g` is the model galaxy number-density field on the inference mesh (the
comoving box), up to the n̄·W normalisation applied by the likelihood.  Everything is
Zygote-differentiable w.r.t. ω and the bias parameters b=(b₁,b₂,b_s2); the cosmology
and growth factors are held fixed.

`GalaxyModel` precomputes the grid/operators and the FIXED reference moments σ²,⟨s²⟩
(from a reference white-noise draw) so the 2nd-order bias operators renormalise against
a constant.
"""

struct GalaxyModel{T<:AbstractFloat, OP, KER, BOP, CO}
    res::Int
    boxsize::T
    n_order::Int
    n_sub::Int
    rsd::Bool
    a_far::T
    a_near::T
    op::OP
    K::KER
    ops::BOP
    qflat::AbstractMatrix{T}     # Array or CuArray (device-aware)
    observer::Vector{T}
    cosmo::CO
    sigma2::T
    s2mean::T
end

"""
    galaxy_model(res, boxsize, cosmo, pk_table; R, observer, a_far, a_near,
                 n_order=3, n_sub=1, rsd=false, ref_seed=0, T=Float64) -> GalaxyModel

Precompute the forward-model operators for an `res³` comoving box.  `R` = Gaussian
Lagrangian bias-smoothing scale [Mpc/h]; `observer` = comoving position [Mpc/h];
`[a_far, a_near]` = lightcone shell.  The reference moments σ²,⟨s²⟩ are computed once
from a `ref_seed` white-noise draw.
"""
function galaxy_model(res::Int, boxsize::Real, cosmo, pk_table::Dict;
                      R::Real, observer::AbstractVector, a_far::Real, a_near::Real,
                      n_order::Int=3, n_sub::Int=1, rsd::Bool=false, ref_seed::Int=0,
                      T::Type{<:AbstractFloat}=Float64)
    n_order in (2, 3) || error("n_order must be 2 (2LPT, ~7× faster) or 3 (3LPT)")
    op  = ic_operator(res, boxsize, pk_table; T=T)
    K   = nlpt_kernels(res, boxsize; T=T)
    ops = bias_operators(res, boxsize, R; T=T)
    qflat = reshape(lagrangian_grid_3d(res, boxsize; T=T), res^3, 3)
    ωref = randn(MersenneTwister(ref_seed), T, res, res, res)
    δL, s2 = bias_fields(white_noise_to_fphi(op, ωref), ops)
    σ2, s2m = bias_moments(δL, s2)
    return GalaxyModel{T, typeof(op), typeof(K), typeof(ops), typeof(cosmo)}(
        res, T(boxsize), n_order, n_sub, rsd, T(a_far), T(a_near),
        op, K, ops, qflat, Vector{T}(observer), cosmo, T(σ2), T(s2m))
end

"""
    galaxy_density(gm, ω, b) -> n_g::(res,res,res)

Differentiable model galaxy number-density field (bias-weighted CDM-sheet density on
the lightcone).  `b = (b₁, b₂, b_s2)`.
"""
function galaxy_density(gm::GalaxyModel{T}, ω::AbstractArray{T,3}, b) where {T}
    fphi   = white_noise_to_fphi(gm.op, ω)
    shapes = compute_core_exact(fphi, gm.K; n_order=gm.n_order)
    Psi    = exact_shape_stack(shapes)
    δL, s2 = bias_fields(fphi, gm.ops)
    wg     = bias_weight(δL, s2, gm.sigma2, gm.s2mean; b1=b[1], b2=b[2], bs2=b[3])
    lc     = lightcone_cross_ad(Psi, gm.qflat, gm.cosmo, gm.observer, gm.a_far, gm.a_near; rsd=gm.rsd)
    x      = lc.x_obs
    if gm.rsd                                            # redshift-space radial shift
        o1, o2, o3 = T(gm.observer[1]), T(gm.observer[2]), T(gm.observer[3])   # scalars (GPU-safe)
        d1 = x[:, 1] .- o1; d2 = x[:, 2] .- o2; d3 = x[:, 3] .- o3
        s  = lc.v_r ./ max.(sqrt.(d1.^2 .+ d2.^2 .+ d3.^2), T(1e-30))
        x  = hcat(x[:, 1] .+ s .* d1, x[:, 2] .+ s .* d2, x[:, 3] .+ s .* d3)
    end
    xg = reshape(x, gm.res, gm.res, gm.res, 3)
    return sheet_deposit(xg, wg, gm.res, gm.boxsize; n_sub=gm.n_sub)
end
