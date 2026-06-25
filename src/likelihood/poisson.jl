"""
Poisson counts-in-cells likelihood.

The expected galaxy count in cell x is λ(x) = N_tot · W(x)·ρ_g(x) / Z, with
ρ_g = max(n_g, 0) the (positivity-floored) model density, W the survey window, and
Z = Σ_footprint W·ρ_g the normalisation that pins Σλ = N_tot (so the overall amplitude
is fixed by the total count, not a free parameter — it removes the n̄–bias degeneracy).
The negative log-likelihood (dropping the data-only log n! term), summed over footprint
cells (mask = W>0):

    −logL = Σ_x mask(x) · [ λ(x) − n_obs(x)·log λ(x) ].

Differentiable in n_g (→ ω, bias); W, n_obs, mask, N_tot are fixed data.
"""

"""
    poisson_nll(ng, n_obs, W, mask, ntot; λfloor=1e-6) -> Float

Counts-in-cells Poisson NLL.  `mask` is the 0/1 footprint (W>0).
"""
function poisson_nll(ng::AbstractArray{T,3}, n_obs::AbstractArray, W::AbstractArray,
                     mask::AbstractArray, ntot::Real; λfloor::Real=1e-6) where {T}
    ρ      = max.(ng, zero(T))                 # positivity (subgradient at the kink)
    lam_u  = W .* ρ
    Z      = sum(mask .* lam_u)
    λ      = (T(ntot) / Z) .* lam_u .+ T(λfloor)
    return sum(mask .* (λ .- n_obs .* log.(λ)))
end

"""
    overdispersed_nll(ng, data_mean, data_var, W, mask, ntot; λfloor=1e-6) -> Float

Gaussian over-dispersed counts-in-cells NLL for the PROV-weighted real-data fit: the
per-cell variance is λ + Var_miss (Poisson + completion uncertainty), so cells whose
counts are dominated by the uncertain completion (large Var_miss) contribute less
gradient — encoding "observed galaxies weighted most, realizations may differ from the
completed points".  `data_mean = n_obs0 + ⟨n_miss⟩`, `data_var = Var_miss`.
"""
function overdispersed_nll(ng::AbstractArray{T,3}, data_mean::AbstractArray,
                           data_var::AbstractArray, W::AbstractArray, mask::AbstractArray,
                           ntot::Real; λfloor::Real=1e-6) where {T}
    ρ     = max.(ng, zero(T))
    lam_u = W .* ρ
    Z     = sum(mask .* lam_u)
    λ     = (T(ntot) / Z) .* lam_u .+ T(λfloor)
    v     = λ .+ data_var .+ T(λfloor)         # Poisson + completion variance
    return sum(mask .* (T(0.5) .* (data_mean .- λ).^2 ./ v .+ T(0.5) .* log.(v)))
end
