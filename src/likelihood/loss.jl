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
    n_obs::A                # binned data counts  (over-dispersed: the data MEAN per cell)
    ntot::T
    b0::Vector{T}; σb::Vector{T}   # bias prior (3 params, stays on host)
    λfloor::T
    data_var::Union{Nothing,A}     # nothing → Poisson; per-cell Var_miss → over-dispersed
end
# Inner-type-eliding constructor: infer A from the arrays (so CuArrays aren't widened);
# `data_var` defaults to nothing (plain Poisson), so all existing 8-arg call sites hold.
InferenceProblem{T,GM}(gm, W, mask, n_obs, ntot, b0, σb, λfloor, data_var=nothing) where {T,GM} =
    InferenceProblem{T,GM,typeof(W)}(gm, W, mask, n_obs, ntot, b0, σb, λfloor, data_var)

"""    galaxy_model_for(geom, pk_table; R, n_order=3, n_sub=1, rsd=false, ref_seed=0)"""
function galaxy_model_for(geom::BoxGeometry, pk_table::Dict;
                          R::Real, n_order::Int=3, n_sub::Int=1, rsd::Bool=false, ref_seed::Int=0,
                          ext::Union{Nothing,Int}=nothing)
    T = eltype(geom.shift)
    galaxy_model(geom.res, geom.boxsize, geom.cosmo, pk_table; R=R, observer=geom.observer,
                 a_far=geom.a_far, a_near=geom.a_near, n_order=n_order, n_sub=n_sub,
                 rsd=rsd, ref_seed=ref_seed, ext=ext, T=T)
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

"""
    inference_problem_overdispersed(geom, randoms, path, pk_table; seeds=nothing, R,
        n_order=2, n_sub=1, rsd=false, ref_seed=0, b0, σb, λfloor=1e-6) -> InferenceProblem

PROV-weighted over-dispersed problem from the K completion realizations in `path`.  The
observed (PROV=0) galaxies are identical across draws → binned once as the fixed counts;
the completed (PROV≥1) galaxies vary → their per-cell mean and variance across the draws
give `data_mean = n_obs0 + ⟨n_miss⟩` and `data_var = Var(n_miss)`.  The likelihood is then
`overdispersed_nll`, so cells whose counts come mostly from uncertain completions
contribute less gradient — "observed galaxies weighted most, realizations may differ from
the completed points".
"""
function inference_problem_overdispersed(geom::BoxGeometry{T}, randoms, path::AbstractString,
        pk_table::Dict; seeds=nothing, R::Real, n_order::Int=2, n_sub::Int=1, rsd::Bool=false,
        ref_seed::Int=0, b0=[1.0, 0.0, 0.0], σb=[0.3, 1e-3, 1e-3], λfloor::Real=1e-6) where {T}
    gm   = galaxy_model_for(geom, pk_table; R=R, n_order=n_order, n_sub=n_sub, rsd=rsd, ref_seed=ref_seed)
    W    = survey_window(geom, randoms)
    mask = T.(W .> 0)
    sds  = seeds === nothing ? (0:(n_realizations(path) - 1)) : collect(seeds)
    bin(c, grp) = (m = prov_mask(c, grp); bin_galaxies(geom, c.ra[m], c.dec[m], c.z[m]) .* mask)
    c0      = load_echoes_realization(path, first(sds); T=T)
    n_fixed = bin(c0, :observed)                                  # PROV=0, same every draw
    comps   = [bin(load_echoes_realization(path, s; T=T), :completed) for s in sds]   # PROV≥1
    cmean   = reduce(.+, comps) ./ length(comps)
    cvar    = reduce(.+, [(c .- cmean).^2 for c in comps]) ./ length(comps)
    dmean   = n_fixed .+ cmean
    return InferenceProblem{T, typeof(gm)}(gm, W, mask, dmean, T(sum(dmean)),
                Vector{T}(b0), Vector{T}(σb), T(λfloor), cvar)
end

"""    model_density(prob, ω, b) -> n_g  (the forward model field)"""
model_density(prob::InferenceProblem, ω, b) = galaxy_density(prob.gm, ω, b)

"""    loss(prob, ω, b) -> scalar  (−logL − logPrior; the single Zygote entry point)"""
function loss(prob::InferenceProblem, ω, b)
    ng  = galaxy_density(prob.gm, ω, b)
    nll = prob.data_var === nothing ?
        poisson_nll(ng, prob.n_obs, prob.W, prob.mask, prob.ntot; λfloor=prob.λfloor) :
        overdispersed_nll(ng, prob.n_obs, prob.data_var, prob.W, prob.mask, prob.ntot; λfloor=prob.λfloor)
    return nll + gaussian_prior(ω) + bias_prior(b, prob.b0, prob.σb)
end
