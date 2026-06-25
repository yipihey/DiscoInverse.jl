"""
Mock generation for injection-recovery validation.

`inject_mock(prob, ω*, b*; seed)` Poisson-samples the model expected counts λ*(ω*,b*)
inside the survey window and returns an `InferenceProblem` whose `n_obs` is that mock —
so we can run the optimizer and check it recovers ω* and b*.
"""

# Knuth Poisson (small λ) / normal approximation (large λ) — avoids a Distributions dep.
function _rand_poisson(rng, λ::Real)
    λ <= 0 && return 0
    if λ < 30
        L = exp(-λ); k = 0; p = 1.0
        while true
            k += 1; p *= rand(rng)
            p <= L && return k - 1
        end
    else
        return max(0, round(Int, λ + sqrt(λ) * randn(rng)))
    end
end

"""    model_lambda(prob, ω, b; ntot=prob.ntot) -> λ::(res,res,res)  (expected counts, Σ=ntot)"""
function model_lambda(prob::InferenceProblem{T}, ω, b; ntot::Real=prob.ntot) where {T}
    ng = galaxy_density(prob.gm, ω, b)
    ρ  = max.(ng, zero(T)); lam_u = prob.W .* ρ
    Z  = sum(prob.mask .* lam_u)
    return (T(ntot) / Z) .* lam_u
end

"""    inject_mock(prob, ω_true, b_true; ntot=prob.ntot, seed=0) -> InferenceProblem (n_obs=Poisson(λ*))"""
function inject_mock(prob::InferenceProblem{T}, ω_true, b_true; ntot::Real=prob.ntot, seed::Int=0) where {T}
    λ = model_lambda(prob, ω_true, b_true; ntot=ntot)
    rng = MersenneTwister(seed)
    counts = zeros(T, size(λ))
    @inbounds for i in eachindex(λ)
        prob.mask[i] > 0 && (counts[i] = T(_rand_poisson(rng, λ[i])))
    end
    return InferenceProblem{T, typeof(prob.gm)}(prob.gm, prob.W, prob.mask, counts,
                                                T(sum(counts)), prob.b0, prob.σb, prob.λfloor)
end
