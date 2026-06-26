"""
Joint (ω, b) Hamiltonian Monte Carlo for the field-level posterior.

Samples the white-noise field ω AND the bias parameters b from the posterior
π(ω,b) ∝ exp(−loss(prob, ω, b)), reusing the differentiable `loss` and its Zygote
gradient.  The white-noise prior makes the ω-block's prior Hessian the identity, so a
unit mass matrix is well-conditioned for the field (the BORG construction); the bias
block gets its own mass `b_mass` (≈ posterior precision, so both blocks mix on a common
step).  The field ω lives on its array's backend (CuArray → device HMC); the 3 bias
params stay on the host — the same split as `adam_optimize`, so the GPU forward is reused
with no scalar indexing.

Leapfrog HMC with dual-averaging step-size adaptation (Hoffman & Gelman 2014, target
`target_accept`) over `nwarmup` iterations; `nleap` steps per proposal (jittered to
avoid resonances).  This is the estimator the bias study calls for: marginalizing the
field gives a PROPER, bounded posterior on the bias amplitude where the MAP ran away.

Returns `(; b_samples, b_mean, b_std, ω_mean, ω_std, ω_draws, accept, ε, divergences,
loss_trace)` — only the 3-vector bias chain, a few field draws, and the field mean/std
are kept (the full ω chain would be res³·nsamples).
"""

using Zygote
using Random: MersenneTwister, randn!
using Statistics: mean, std

# Backend-aware standard-normal fill (CUDA RNG for CuArray, host rng otherwise).
_fill_randn!(rng, p::Array) = randn!(rng, p)
_fill_randn!(_, p) = randn!(p)            # CuArray → device RNG

function hmc_sample(prob::InferenceProblem, ω0::AbstractArray{T,3}, b0::AbstractVector;
                    nsamples::Int=500, nwarmup::Int=300, nleap::Int=25,
                    target_accept::Real=0.7, b_mass::Real=25.0, ε0::Real=0.02,
                    keep_fields::Int=5, seed::Int=0, jitter::Bool=true,
                    show_every::Int=0) where {T}
    rng = MersenneTwister(seed)
    ω  = copy(ω0); b = collect(float.(b0))
    pω = similar(ω); mb = T(1 / b_mass)                  # field M=I; bias M=b_mass
    Ugrad(w, bb) = Zygote.withgradient((x, y) -> loss(prob, x, y), w, bb)
    res = ndims(ω) == 3 ? size(ω, 1) : error("ω must be (res,res,res)")

    r = Ugrad(ω, b); val = r.val; gω = r.grad[1]; gb = r.grad[2]

    logε = log(T(ε0)); μ = log(T(10) * T(ε0)); logε̄ = zero(T); H̄ = zero(T)
    γ = T(0.05); t0 = T(10); κ = T(0.75)

    b_samples = Vector{Vector{Float64}}(); loss_trace = Float64[]
    ωsum = fill!(similar(ω), zero(T)); ωsq = fill!(similar(ω), zero(T))
    ω_draws = Vector{Array{T,3}}(); nacc = 0; ndiv = 0; nkept = 0
    total = nwarmup + nsamples

    for m in 1:total
        ε = exp(logε)
        L = jitter ? max(1, round(Int, nleap * (T(0.8) + T(0.4) * rand(rng)))) : nleap
        _fill_randn!(rng, pω)                            # pω ~ N(0, I)
        pb = randn(rng, 3) .* sqrt(b_mass)               # pb ~ N(0, b_mass·I)
        ω0c = copy(ω); b0c = copy(b)
        K0 = 0.5 * sum(x -> Float64(abs2(x)), pω) + 0.5 * mb * sum(abs2, pb)   # F64 accum
        H0 = val + K0

        gωc = gω; gbc = gb; valc = val
        @. pω -= (ε / 2) * gωc; @. pb -= (ε / 2) * gbc   # initial half-kick
        for s in 1:L
            @. ω += ε * pω; @. b += ε * mb * pb          # drift (M⁻¹: 1 for ω, mb for b)
            rr = Ugrad(ω, b); valc = rr.val; gωc = rr.grad[1]; gbc = rr.grad[2]
            c = (s < L) ? ε : ε / 2                       # interior full kick / final half
            @. pω -= c * gωc; @. pb -= c * gbc
        end
        K1 = 0.5 * sum(x -> Float64(abs2(x)), pω) + 0.5 * mb * sum(abs2, pb)   # F64 accum
        Δ = H0 - (valc + K1)
        diverged = !isfinite(Δ) || abs(Δ) > 1000
        α = diverged ? 0.0 : min(1.0, exp(Float64(Δ)))
        if !diverged && log(rand(rng)) < Δ               # accept
            val = valc; gω = gωc; gb = gbc; nacc += 1
        else                                             # reject → restore state
            copyto!(ω, ω0c); copyto!(b, b0c); diverged && (ndiv += 1)
        end

        if m <= nwarmup                                  # dual-averaging adaptation
            H̄ = (1 - 1/(m + t0)) * H̄ + (1/(m + t0)) * (T(target_accept) - T(α))
            logε = μ - sqrt(T(m)) / γ * H̄
            η = T(m)^(-κ); logε̄ = η * logε + (1 - η) * logε̄
        else
            logε = logε̄                                  # freeze ε at the adapted value
            push!(b_samples, copy(b)); push!(loss_trace, val)
            @. ωsum += ω; @. ωsq += ω * ω; nkept += 1
            length(ω_draws) < keep_fields && push!(ω_draws, Array(ω))
        end
        show_every > 0 && m % show_every == 0 &&
            @info "hmc" iter=m phase=(m <= nwarmup ? "warmup" : "sample") ε=exp(logε) accept=round(nacc/m, digits=2)
    end

    B  = reduce(hcat, b_samples)'                         # (nsamples, 3)
    ωm = ωsum ./ nkept
    ωv = ωsq ./ nkept .- ωm .^ 2
    return (b_samples = B,
            b_mean = vec(mean(B, dims=1)), b_std = vec(std(B, dims=1)),
            ω_mean = Array(ωm), ω_std = Array(sqrt.(max.(ωv, zero(T)))), ω_draws = ω_draws,
            accept = nacc / total, ε = exp(logε̄), divergences = ndiv, loss_trace = loss_trace)
end
