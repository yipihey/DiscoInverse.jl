"""
The inference problem: forward model + data + priors, with a single differentiable
`loss(prob, ω, b)` entry point for the optimizer / sampler.

    loss = −logL_Poisson(n_g(ω,b); n_obs, W, N_tot) + ½‖ω‖² + ½‖(b−b₀)/σ_b‖²

Gradients flow only to ω and the bias b; the cosmology and survey geometry are fixed.
"""

struct InferenceProblem{T<:AbstractFloat, GM, A<:AbstractArray{T,3}}
    gm::GM                  # GalaxyModel (forward)
    W::A                    # survey window           (Array or CuArray — device-aware)
    mask::A                 # footprint 0/1 (W>0)
    n_obs::A                # binned data counts
    ntot::T
    b0::Vector{T}; σb::Vector{T}   # bias prior (3 params, stays on host)
    λfloor::T
end
# Inner-type-eliding constructor: infer A from the arrays (so CuArrays aren't widened).
InferenceProblem{T,GM}(gm, W, mask, n_obs, ntot, b0, σb, λfloor) where {T,GM} =
    InferenceProblem{T,GM,typeof(W)}(gm, W, mask, n_obs, ntot, b0, σb, λfloor)

"""    galaxy_model_for(geom, pk_table; R, n_order=3, n_sub=1, rsd=false, ref_seed=0)"""
function galaxy_model_for(geom::BoxGeometry, pk_table::Dict;
                          R::Real, n_order::Int=3, n_sub::Int=1, rsd::Bool=false, ref_seed::Int=0)
    T = eltype(geom.shift)
    galaxy_model(geom.res, geom.boxsize, geom.cosmo, pk_table; R=R, observer=geom.observer,
                 a_far=geom.a_far, a_near=geom.a_near, n_order=n_order, n_sub=n_sub,
                 rsd=rsd, ref_seed=ref_seed, T=T)
end

"""
    prov_weights(cat; soft=0.5) -> per-galaxy weights (PROV=0 → 1, PROV≥1 → soft)
"""
prov_weights(cat::EchoesCatalog{T}; soft::Real=0.5) where {T} =
    [p == 0 ? one(T) : T(soft) for p in cat.prov]

"""
    inference_problem(geom, randoms, catalog, pk_table; R, n_order=3, n_sub=1, rsd=false,
                      ref_seed=0, b0=[1.5,0,0], σb=[2,2,2], λfloor=1e-6,
                      galaxy_weights=nothing) -> InferenceProblem

Build the inference problem: forward `GalaxyModel`, survey window `W` (from randoms),
binned data counts `n_obs` (from `catalog`, optional per-galaxy `galaxy_weights`).
"""
function inference_problem(geom::BoxGeometry{T}, randoms, catalog, pk_table::Dict;
                           R::Real, n_order::Int=3, n_sub::Int=1, rsd::Bool=false, ref_seed::Int=0,
                           b0=[1.5, 0.0, 0.0], σb=[2.0, 2.0, 2.0], λfloor::Real=1e-6,
                           galaxy_weights=nothing) where {T}
    gm    = galaxy_model_for(geom, pk_table; R=R, n_order=n_order, n_sub=n_sub, rsd=rsd, ref_seed=ref_seed)
    W     = survey_window(geom, randoms)
    mask  = T.(W .> 0)
    n_obs = bin_galaxies(geom, catalog.ra, catalog.dec, catalog.z; weights=galaxy_weights) .* mask
    return InferenceProblem{T, typeof(gm)}(gm, W, mask, n_obs, T(sum(n_obs)),
                                           Vector{T}(b0), Vector{T}(σb), T(λfloor))
end

"""    model_density(prob, ω, b) -> n_g  (the forward model field)"""
model_density(prob::InferenceProblem, ω, b) = galaxy_density(prob.gm, ω, b)

"""    loss(prob, ω, b) -> scalar  (−logL − logPrior; the single Zygote entry point)"""
function loss(prob::InferenceProblem, ω, b)
    ng = galaxy_density(prob.gm, ω, b)
    return poisson_nll(ng, prob.n_obs, prob.W, prob.mask, prob.ntot; λfloor=prob.λfloor) +
           gaussian_prior(ω) + bias_prior(b, prob.b0, prob.σb)
end
