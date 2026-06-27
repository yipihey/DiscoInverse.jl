"""
Fixed-amplitude (Angulo–Pontzen) field parametrization — Stage 1 of the bias/cosmology +
cosmic-variance decomposition.

The white-noise field is parametrized by PHASES only: ω(φ) = A·irfft(exp(iφ)), with every
Fourier mode at the same fixed modulus |ω̂_k| = A (A calibrated so ω has unit variance, which
is φ-independent since the power is fixed).  Through the IC operator this gives a field with
*exactly* the cosmological P(k) — no per-mode amplitude scatter.  Removing the amplitude degrees
of freedom breaks the b₁↔field-amplitude degeneracy, so **b₁ is identifiable at the MAP** (a plain
optimization, no sampler), and the reconstructed field keeps full power at all scales (random
phases where the data is silent) instead of the Gaussian-prior MAP shrinking small scales to ~0.

There is no white-noise prior here (½‖ω‖² = N/2 is constant under fixed amplitude) and a flat
prior on the phases; only the bias keeps its prior.  The *conditional* (cosmic-variance-free)
best-fit and error come from this stage; the amplitude scatter (sample variance) is added back
carefully in Stage 2 as a perturbation around this well-conditioned point.
"""

using Zygote
using Random: MersenneTwister
using Statistics: std, mean
using LinearAlgebra: dot

# generic device-resident L-BFGS over an arbitrary loss f (gradient g); state x is any array/vector
# (CuArray phases or a host bias vector) — only dot/broadcast are used, so it stays on x's backend.
function _lbfgs_generic(f, g, x0; iters::Int=30, m::Int=10, c1::Real=1e-4, ρ::Real=0.5, max_ls::Int=25)
    T = real(eltype(x0))                                     # keep all array ops in x's precision (F32 res=256)
    x = copy(x0); fx = f(x); gx = g(x)
    S = typeof(x)[]; Y = typeof(x)[]; ρs = T[]; hist = Float64[fx]
    for _ in 1:iters
        q = copy(gx); k = length(S); al = zeros(T, k)
        for i in k:-1:1; al[i] = T(ρs[i]*dot(S[i], q)); q .-= al[i].*Y[i]; end
        γ = k == 0 ? one(T) : T(dot(S[end], Y[end]) / dot(Y[end], Y[end])); r = γ .* q
        for i in 1:k; β = T(ρs[i]*dot(Y[i], r)); r .+= (al[i]-β).*S[i]; end
        d = .-r; gd = dot(gx, d); gd ≥ 0 && (d = .-gx; gd = dot(gx, d))
        a = one(T); xn = x .+ a.*d; fn = f(xn); ls = 0
        while (fn > fx + c1*a*gd || !isfinite(fn)) && ls < max_ls; a *= T(ρ); xn = x .+ a.*d; fn = f(xn); ls += 1; end
        gn = g(xn); s = xn .- x; y = gn .- gx; sy = dot(s, y)
        sy > 1e-12 && (push!(S,s); push!(Y,y); push!(ρs,T(1/sy)); length(S) > m && (popfirst!(S);popfirst!(Y);popfirst!(ρs)))
        x = xn; fx = fn; gx = gn; push!(hist, fx)
    end
    return x, fx, hist
end

"""    phase_field(φ) -> ω = irfft(exp(iφ)) / std   (unit-variance fixed-amplitude white noise)

`φ` is a real array of rfft shape `(res÷2+1, res, res)` (CPU or CuArray).  Every Fourier mode has
the same modulus (only phases vary) and the field is normalized to unit variance, so ‖ω‖² = N is
exactly constant (the white-noise prior is constant ⇒ dropped) — the field power is fixed to the
cosmological P(k) through the IC operator, with no per-mode amplitude scatter."""
function phase_field(φ::AbstractArray)
    ω = irfft(exp.(im .* φ), size(φ, 2))
    return ω ./ std(ω)
end

"""    phase_loss(prob, φ, b) -> −Σ u log ρ_g + U log Z + bias_prior   (no ω-prior)"""
function phase_loss(prob::SheetProblem, φ, b)
    ρg, Z = _sheet_dens(prob, phase_field(φ), b)
    return -sum(prob.u .* log.(max.(ρg, prob.ρfloor))) + prob.Utot * log(Z) + bias_prior(b, prob.b0, prob.σb)
end

"""
    phase_map_optimize(prob, φ0; b1_grid=1.0:0.5:3.0, phase_iters=80, b2=0, bs2=0)
        -> (; b1, σ_b1_cond, φ, ω, b1_grid, losses)

Stage-1 fixed-amplitude (Angulo–Pontzen) b₁ estimate via the **conditional profile likelihood**:
at each b₁ on the grid, fully optimize the phases φ (fresh start, `phase_iters` L-BFGS steps) with
the per-mode amplitudes fixed and b₂/bs₂ held (linear bias — the cleanly identifiable one), and
record the converged data+Z.  b₁ lands at the minimum (it is NOT degenerate once the amplitudes
are fixed AND the phases are well-converged — under-converged φ makes the bias walk uphill in b₁,
the failure mode of the alternating MAP).  `σ_b1_cond` is the **conditional** (cosmic-variance-
free, over-confident) error from the local parabola curvature — Stage 2 adds the amplitude/sample
variance.  Returns the best-fit phases/field at b₁* too.  Robust where the alternating (φ,b) MAP
is fragile, because each b₁ is converged independently.
"""
function phase_map_optimize(prob, φ0; b1_grid=collect(1.0:0.5:3.0), phase_iters::Int=80,
                            b2::Real=0.0, bs2::Real=0.0)
    bs = collect(float.(b1_grid)); losses = fill(Inf, length(bs))
    φstar = copy(φ0); istar = 1; lstar = Inf
    for (i, b1) in enumerate(bs)
        f = φv -> phase_loss(prob, φv, [b1, b2, bs2])
        φ, l, _ = _lbfgs_generic(f, φv -> Zygote.gradient(f, φv)[1], copy(φ0); iters=phase_iters)
        losses[i] = l
        l < lstar && (lstar = l; φstar = copy(φ); istar = i)
    end
    # local 3-point parabola at the (interior) minimum → vertex b₁* + conditional σ
    b1star = bs[istar]; σ = NaN
    if 1 < istar < length(bs)
        h = bs[istar+1] - bs[istar]
        κ = (losses[istar-1] - 2losses[istar] + losses[istar+1]) / h^2   # curvature = conditional Fisher
        if κ > 0
            b1star = bs[istar] - h*(losses[istar+1] - losses[istar-1]) / (2*(losses[istar-1] - 2losses[istar] + losses[istar+1]))
            σ = 1/sqrt(κ)
        end
    end
    return (b1 = b1star, σ_b1_cond = σ, φ = φstar, ω = phase_field(φstar), b1_grid = bs, losses = losses)
end

"""
    cosmic_variance_b1(prob, φstar; K=24, b1_grid=1.0:0.25:3.5, seed=100)
        -> (; b1_mean, σ_cosmic, b1_samples)

Stage-2 cosmic-variance budget around a Stage-1 best-fit.  Holding the best-fit phases `φstar`,
draw `K` per-mode amplitude realizations (Rayleigh, ⟨a²⟩=1 — the amplitude scatter Stage-1 froze
out) and re-fit b₁ (1-D, field fixed) for each; the spread `σ_cosmic` is the sample-variance error
that inflates the over-confident Stage-1 conditional σ into the realistic one.  Frozen-phase
approximation (the phases are not re-optimized per realization) ⇒ a slight under-estimate.
"""
function cosmic_variance_b1(prob, φstar; K::Int=24, b1_grid=collect(1.0:0.25:3.5), seed::Int=100)
    res = size(φstar, 2); sφ = exp.(im .* φstar); bs = collect(float.(b1_grid))
    fit1d(ω) = begin
        ls = [let r = _sheet_dens(prob, ω, [b1, 0.0, 0.0])
                  -sum(prob.u .* log.(max.(r[1], prob.ρfloor))) + prob.Utot * log(r[2])
              end for b1 in bs]
        i = argmin(ls)
        (1 < i < length(ls)) ? bs[i] - (bs[2]-bs[1])*(ls[i+1]-ls[i-1]) / (2*(ls[i-1]-2ls[i]+ls[i+1])) : bs[i]
    end
    b1s = Float64[]
    for j in 1:K
        g1 = randn(MersenneTwister(seed+j), size(φstar)); g2 = randn(MersenneTwister(seed+K+j), size(φstar))
        a = similar(φstar); copyto!(a, sqrt.(g1.^2 .+ g2.^2) ./ sqrt(2))   # Rayleigh amplitudes on φ's backend
        w = irfft(a .* sφ, res)
        push!(b1s, fit1d(w ./ std(w)))
    end
    return (b1_mean = mean(b1s), σ_cosmic = std(b1s), b1_samples = b1s)
end
