"""
No-U-Turn Sampler (NUTS) for the joint (ω, b) field-level posterior, with multi-chain
convergence diagnostics.

NUTS (Hoffman & Gelman 2014, efficient recursive Alg. 3 + dual-averaging Alg. 6)
auto-tunes the trajectory length per sample by doubling a binary tree of leapfrog steps
until the trajectory makes a U-turn — removing the fixed `nleap` guess of `hmc_sample`
and giving far higher effective sample size for the slowly-mixing field block.  Same
two-block geometry as `hmc_sample`: field ω on its backend (CuArray → device NUTS, unit
mass / BORG construction), the 3 bias params on the host with mass `b_mass`.

`nuts_sample` runs one chain; `nuts_chains` runs several from dispersed starts and reports
split-R̂ and multi-chain ESS (Vehtari et al. 2021) on the bias parameters.  Reuses the
differentiable `loss` + Zygote gradient; only dependency is Zygote.
"""

using Zygote
using Random: MersenneTwister
using Statistics: mean, std, var
# `_fill_randn!` is defined in infer/hmc.jl (same module, included first).

# ── phase point: position (ω,b), momentum (pω,pb), cached grad (gω,gb) and potential U
struct _PP{A,V}
    ω::A; b::V; pω::A; pb::V; gω::A; gb::V; U::Float64
end
# `mb` is the bias block's inverse mass M_b⁻¹ (3-vector) — per-parameter so the tight b1
# and the wide b2/bs2 mix on a common step.  Field block has unit mass (M⁻¹=1).
_kin(pt::_PP, mb) = 0.5 * (sum(x -> Float64(abs2(x)), pt.pω) + sum(mb .* pt.pb .^ 2))  # F64 accum (F32 momentum sum → O(1) ΔH error)
_ham(pt::_PP, mb) = pt.U + _kin(pt, mb)

# one leapfrog step of (signed) size ε; recomputes & caches the gradient at the new point
function _leap(prob, pt::_PP, ε, mb)
    pω = pt.pω .- (ε / 2) .* pt.gω
    pb = pt.pb .- (ε / 2) .* pt.gb
    ω  = pt.ω .+ ε .* pω
    b  = pt.b .+ ε .* (mb .* pb)
    r  = Zygote.withgradient((x, y) -> loss(prob, x, y), ω, b)
    gω = r.grad[1]; gb = r.grad[2]
    pω = pω .- (ε / 2) .* gω
    pb = pb .- (ε / 2) .* gb
    return _PP(ω, b, pω, pb, gω, gb, r.val)
end

# generalized no-U-turn: (Δq)·(M⁻¹ r) ≥ 0 at both ends  (M⁻¹ = 1 for ω, mb for b)
function _no_uturn(minus::_PP, plus::_PP, mb)
    Δω = plus.ω .- minus.ω; Δb = plus.b .- minus.b
    cm = sum(Δω .* minus.pω) + sum(mb .* Δb .* minus.pb)
    cp = sum(Δω .* plus.pω)  + sum(mb .* Δb .* plus.pb)
    return (cm >= 0) && (cp >= 0)
end

const _ΔMAX = 1000.0

function _build_tree(prob, pt::_PP, logu, v, j, ε, H0, mb, rng)
    if j == 0
        pt2 = _leap(prob, pt, v * ε, mb)
        H2  = _ham(pt2, mb)
        n   = (logu <= -H2) ? 1 : 0
        div = !(logu < _ΔMAX - H2)
        s   = div ? 0 : 1
        return (minus=pt2, plus=pt2, prop=pt2, n=n, s=s,
                α=min(1.0, exp(H0 - H2)), nα=1, ndiv=(div ? 1 : 0))
    end
    sub = _build_tree(prob, pt, logu, v, j - 1, ε, H0, mb, rng)
    sub.s == 0 && return sub
    if v == -1
        sub2 = _build_tree(prob, sub.minus, logu, v, j - 1, ε, H0, mb, rng)
        minus = sub2.minus; plus = sub.plus
    else
        sub2 = _build_tree(prob, sub.plus, logu, v, j - 1, ε, H0, mb, rng)
        minus = sub.minus; plus = sub2.plus
    end
    ntot = sub.n + sub2.n
    prop = (ntot > 0 && rand(rng) < sub2.n / ntot) ? sub2.prop : sub.prop
    s    = sub2.s * (_no_uturn(minus, plus, mb) ? 1 : 0)
    return (minus=minus, plus=plus, prop=prop, n=ntot, s=s,
            α=sub.α + sub2.α, nα=sub.nα + sub2.nα, ndiv=sub.ndiv + sub2.ndiv)
end

# one NUTS transition from `pt0` (whose grad/U are cached); returns (new_pt, accept_stat, depth, ndiv)
# `Mb` is the bias mass (3-vector M_b); mb = M_b⁻¹, momentum pb ~ N(0, M_b).
function _nuts_step(prob, pt0::_PP, ε, Mb, max_depth, rng)
    mb = 1 ./ Mb
    pω = similar(pt0.ω); _fill_randn!(rng, pω)
    pb = randn(rng, 3) .* sqrt.(Mb)
    pt = _PP(pt0.ω, pt0.b, pω, pb, pt0.gω, pt0.gb, pt0.U)
    H0 = _ham(pt, mb)
    logu = log(rand(rng)) - H0
    left = pt; right = pt; prop = pt
    n = 1; s = 1; j = 0; α = 0.0; nα = 1; ndiv = 0
    while s == 1 && j < max_depth
        v = rand(rng) < 0.5 ? -1 : 1
        if v == -1
            sub = _build_tree(prob, left, logu, -1, j, ε, H0, mb, rng); left = sub.minus
        else
            sub = _build_tree(prob, right, logu, 1, j, ε, H0, mb, rng); right = sub.plus
        end
        if sub.s == 1 && rand(rng) < min(1.0, sub.n / n)
            prop = sub.prop
        end
        n += sub.n
        s = sub.s * (_no_uturn(left, right, mb) ? 1 : 0)
        α = sub.α; nα = sub.nα; ndiv += sub.ndiv
        j += 1
    end
    return prop, α / nα, j, ndiv
end

"""
    nuts_sample(prob, ω0, b0; nsamples=400, nwarmup=300, max_depth=8, b_mass=25,
                target_accept=0.8, ε0=0.05, keep_fields=5, seed=0, show_every=0)

Single NUTS chain over (ω, b).  Returns `(; b_samples, b_mean, b_std, ω_mean, ω_std,
ω_draws, accept, ε, depths, divergences, loss_trace)`.
"""
function nuts_sample(prob::InferenceProblem, ω0::AbstractArray{T,3}, b0::AbstractVector;
                     nsamples::Int=400, nwarmup::Int=300, max_depth::Int=8,
                     b_mass=25.0, target_accept::Real=0.8, ε0::Real=0.05,
                     keep_fields::Int=5, seed::Int=0, show_every::Int=0, gc_every::Int=10,
                     reclaim=() -> nothing) where {T}
    rng = MersenneTwister(seed)
    ω = copy(ω0); b = collect(float.(b0))
    Mb = b_mass isa Number ? fill(float(b_mass), 3) : collect(float.(b_mass))   # per-param bias mass
    r0 = Zygote.withgradient((x, y) -> loss(prob, x, y), ω, b)
    pt = _PP(ω, b, similar(ω), b .* 0, r0.grad[1], r0.grad[2], r0.val)

    logε = log(T(ε0)); μ = log(T(10) * T(ε0)); logε̄ = zero(Float64); H̄ = 0.0
    γ = 0.05; t0 = 10.0; κ = 0.75

    b_samples = Vector{Vector{Float64}}(); loss_trace = Float64[]; depths = Int[]
    ωsum = fill!(similar(ω), zero(T)); ωsq = fill!(similar(ω), zero(T))
    ω_draws = Vector{Array{T,3}}(); αsum = 0.0; ndiv = 0; nkept = 0
    total = nwarmup + nsamples

    for m in 1:total
        ε = exp(logε)
        ptnew, astat, depth, nd = _nuts_step(prob, pt, ε, Mb, max_depth, rng)
        pt = ptnew; ndiv += nd; m > nwarmup && (αsum += astat)
        # Cap the CUDA pool: each NUTS draw churns ~2^depth leapfrog gradients, and Julia's
        # lazy GC lets the pool balloon far past the ~live working set (the 31→45 GB we saw).
        # A full collect + caller-injected `reclaim` (e.g. CUDA.reclaim) returns freed blocks
        # all the way to the driver, holding the run at its live footprint (~5 GB res96).
        gc_every > 0 && m % gc_every == 0 && (GC.gc(); reclaim())
        if m <= nwarmup
            H̄ = (1 - 1/(m + t0)) * H̄ + (1/(m + t0)) * (target_accept - astat)
            logε = μ - sqrt(m) / γ * H̄
            η = m^(-κ); logε̄ = η * logε + (1 - η) * logε̄
        else
            logε = logε̄
            push!(b_samples, copy(pt.b)); push!(loss_trace, pt.U); push!(depths, depth)
            @. ωsum += pt.ω; @. ωsq += pt.ω * pt.ω; nkept += 1
            length(ω_draws) < keep_fields && push!(ω_draws, Array(pt.ω))
        end
        show_every > 0 && m % show_every == 0 &&
            @info "nuts" iter=m phase=(m <= nwarmup ? "warmup" : "sample") ε=exp(logε) depth=depth div=ndiv
    end

    B  = reduce(hcat, b_samples)'
    ωm = ωsum ./ nkept; ωv = ωsq ./ nkept .- ωm .^ 2
    return (b_samples = B, b_mean = vec(mean(B, dims=1)), b_std = vec(std(B, dims=1)),
            ω_mean = Array(ωm), ω_std = Array(sqrt.(max.(ωv, zero(T)))), ω_draws = ω_draws,
            accept = αsum / nsamples, ε = exp(logε̄), depths = depths,
            divergences = ndiv, loss_trace = loss_trace)
end

# ── multi-chain convergence diagnostics (one parameter; chains = (C, n) matrix) ──
"""    rhat(chains::AbstractMatrix) -> split-R̂  (chains rows = chains, cols = draws)"""
function rhat(chains::AbstractMatrix)
    C, n = size(chains); m = n ÷ 2
    h = vcat(chains[:, 1:m], chains[:, m+1:2m])          # 2C split half-chains
    means = vec(mean(h, dims=2)); vars = vec(var(h, dims=2, corrected=true))
    W = mean(vars); B = m * var(means, corrected=true)
    return sqrt(((m - 1) / m * W + B / m) / W)
end

"""    ess(chains::AbstractMatrix) -> multi-chain effective sample size (Geyer truncation)"""
function ess(chains::AbstractMatrix)
    C, n = size(chains)
    means = vec(mean(chains, dims=2))
    W = mean(vec(var(chains, dims=2, corrected=true)))
    B = n * var(means, corrected=true)
    var_plus = (n - 1) / n * W + B / n
    var_plus <= 0 && return Float64(C * n)
    ρ = zeros(n)                                          # combined autocorrelation
    for t in 0:(n - 1)
        acov = 0.0
        for c in 1:C
            x = @view chains[c, :]; mc = means[c]
            s = 0.0
            for i in 1:(n - t); s += (x[i] - mc) * (x[i + t] - mc); end
            acov += s / n
        end
        ρ[t + 1] = 1 - (W - acov / C) / var_plus
    end
    τ = 1.0; t = 1                                        # Geyer initial-positive pairs
    while t + 1 < n
        p = ρ[t + 1] + ρ[t + 2]
        p <= 0 && break
        τ += 2 * p; t += 2
    end
    return C * n / τ
end

"""
    nuts_chains(prob, ω0, b0; nchains=4, dispersion=1.0, seed=0, kw...)
        -> (; chains, b_mean, b_std, rhat, ess, accept, divergences)

Run `nchains` NUTS chains from over-dispersed field starts (ω0 + dispersion·N(0,1) per
chain) and pool the bias parameters with split-R̂ and multi-chain ESS.  `kw...` forwards
to `nuts_sample`.
"""
function nuts_chains(prob::InferenceProblem, ω0::AbstractArray{T,3}, b0::AbstractVector;
                     nchains::Int=4, dispersion::Real=1.0, seed::Int=0, kw...) where {T}
    chains = map(1:nchains) do c
        ω0c = ω0 .+ T(dispersion) .* (c == 1 ? zero(ω0) : begin
            z = similar(ω0); _fill_randn!(MersenneTwister(1000 + seed + c), z); z end)
        nuts_sample(prob, ω0c, b0; seed=seed + c, kw...)
    end
    nB = minimum(size(ch.b_samples, 1) for ch in chains)
    B = [ch.b_samples[1:nB, k] for ch in chains, k in 1:3]   # (nchains, 3) of vectors
    stack(k) = permutedims(reduce(hcat, [B[c, k] for c in 1:nchains]))   # (nchains, nB)
    allb = reduce(vcat, [ch.b_samples[1:nB, :] for ch in chains])
    return (chains = chains,
            b_mean = vec(mean(allb, dims=1)), b_std = vec(std(allb, dims=1)),
            rhat = [rhat(stack(k)) for k in 1:3],
            ess  = [ess(stack(k))  for k in 1:3],
            accept = mean(ch.accept for ch in chains),
            divergences = sum(ch.divergences for ch in chains))
end
