"""
Perturb-and-MAP constrained realizations of the Gaussian-ω posterior.

For a (near-)linear-Gaussian constraint — the peculiar-velocity term is linear in ω to leading order, and
the ½‖ω‖² prior is Gaussian — the posterior is Gaussian, and an EXACT posterior sample is obtained by a
single MAP solve with a perturbed prior mean and perturbed data (Papandreou–Yuille / Bardsley
randomize-then-optimize):

    ω_s = argmin_ω  ½‖ω − ε_p‖² + ½ Σ_i invN_i (v_model_i(ω) − v_obs_i − ε_d,i)²,
    ε_p ~ N(0, I)  (a white-noise field),   ε_d,i ~ N(0, σ_v,i²)  (a per-group velocity-noise draw).

E[ω_s] is the Wiener-filter posterior mean and Cov[ω_s] is the posterior covariance.  This sidesteps the
stiff geometry that collapses identity-mass HMC (velocities ∝ 1/k constrain the large scales tightly): each
MAP is well-conditioned for LBFGS.  Exact for a linear model; an excellent approximation at CF4's
quasi-linear scales.  The MAP with `ε_p = ε_d = 0` is the plain Wiener mean.
"""

using Optim: optimize, LBFGS, Options, minimizer
import Optim
using FFTW: rfft, irfft
using Statistics: mean, std
using Random: MersenneTwister

# minimize ½‖ω − ω_c‖² + data_loss(mtp, ω), warm-started at ω0.  LBFGS handles the stiff Wiener geometry
# that collapses identity-mass HMC; backtracking line search keeps forward evals per step low.
function _wiener_map(mtp::MultiTracerProblem, ω0, ω_c; iters::Int=200)
    f(ω) = _mtp_data_loss(mtp, ω) + 0.5 * sum(abs2, ω .- ω_c)
    function g!(G, ω)
        G .= Zygote.gradient(w -> _mtp_data_loss(mtp, w), ω)[1] .+ (ω .- ω_c)
        return G
    end
    r = optimize(f, g!, ω0, LBFGS(m=20, linesearch=Optim.LineSearches.BackTracking(order=2)),
                 Options(iterations=iters, g_tol=1e-7))
    return minimizer(r)
end

# a copy of the velocity constraint with perturbed data v_obs + ε_d (ε_d on the same device as v_obs)
_perturb_velocity(vc::VelocityConstraint, εd) =
    VelocityConstraint(vc.pts, vc.cl, vc.rhat, vc.v_obs .+ εd, vc.invN, vc.vnorm, vc.submean)

"""    wiener_mean(mtp; iters=200, device=identity) -> ω  — the posterior-mean field (Wiener filter)."""
function wiener_mean(mtp::MultiTracerProblem{T}; iters::Int=200, device=identity) where {T}
    ω0 = device(zeros(T, mtp.gm.res, mtp.gm.res, mtp.gm.res))
    return Array(_wiener_map(mtp, ω0, ω0; iters=iters))
end

"""
    constrained_realizations(mtp, K; iters=200, device=identity, seed=0) -> (; omega_mean, omega_std, draws)

`K` perturb-and-MAP posterior samples of the field constrained by the velocity term — exact posterior draws
for the near-linear-Gaussian model, without the stiff-geometry HMC.  `omega_mean` ≈ the Wiener mean,
`omega_std` the per-voxel posterior uncertainty, `draws` the constrained realizations (host arrays).
`device=CuArray` runs each MAP on the GPU."""
function constrained_realizations(mtp::MultiTracerProblem{T}, K::Int;
                                  iters::Int=200, device=identity, seed::Int=0) where {T}
    res = mtp.gm.res; vc = mtp.velocity
    vc === nothing && error("constrained_realizations requires a velocity constraint")
    σ = Array(1 ./ sqrt.(vc.invN))                       # per-group velocity noise σ_v (host)
    draws = Vector{Array{T,3}}(undef, K)
    for k in 1:K
        rng = MersenneTwister(seed + k)
        εp  = device(T.(randn(rng, res, res, res)))       # prior draw ε_p ~ N(0,I)
        εd  = device(T.(randn(rng, length(σ)) .* σ))      # noise draw ε_d ~ N(0, σ_v²)
        mtpp = multitracer_problem(mtp.gm, mtp.tracers; lensing=mtp.lensing,
                                   velocity=_perturb_velocity(vc, εd), ρfloor=mtp.ρfloor, floor_frac=mtp.floor_frac)
        draws[k] = Array(_wiener_map(mtpp, εp, εp; iters=iters))
    end
    A = cat(draws...; dims=4)
    return (omega_mean = dropdims(mean(A; dims=4); dims=4),
            omega_std  = dropdims(std(A; dims=4); dims=4),
            draws = draws)
end

# high-pass mask on the rfft grid: true for |k| ≥ frac·Nyquist (spherical, matching-physical-k)
function _highpass_mask(N::Int, frac::Real)
    kf = [(i-1) < N÷2 ? i-1 : i-1-N for i in 1:N]; kh = [i-1 for i in 1:N÷2+1]; ks = frac * (N/2)
    return [sqrt(kh[c1]^2 + kf[c2]^2 + kf[c3]^2) >= ks for c1 in 1:N÷2+1, c2 in 1:N, c3 in 1:N]
end

"""
    constrained_zoom(ω_parent, mtp, k_split_frac; iters=150) -> ω_zoom

Nested / multi-scale constrained realization — the primitive for putting a **fine local constraint inside
a coarse global field**. The field's large-scale modes (|k| < `k_split_frac`·Nyquist) are held **fixed to
`ω_parent`** (the globally-constrained field, e.g. the Quaia/lensing carrier) while the smaller scales are
reconstructed against the fine data in `mtp` (e.g. a Cosmicflows-4 `VelocityConstraint`) — so the result
carries the global large scales *and* the local fine structure, **consistent by construction** (the parent
modes are preserved bit-exactly). Above the split the field is pinned by the fine data where it informs and
is a fresh ΛCDM draw where it doesn't. Optimized with LBFGS (backtracking) through the sheet forward + an
FFT high-pass projection; the ½‖ω_high‖² prior acts only on the free modes."""
function constrained_zoom(ω_parent::AbstractArray{T,3}, mtp::MultiTracerProblem, k_split_frac::Real;
                          iters::Int=150) where {T}
    N = size(ω_parent, 1)
    mask  = _highpass_mask(N, k_split_frac)
    ω_low = irfft(rfft(ω_parent) .* .!mask, N)         # parent's large scales (fixed)
    hp(θ) = irfft(rfft(θ) .* mask, N)                  # high-pass projection of the free field
    f(θ)  = _mtp_data_loss(mtp, ω_low .+ hp(θ)) + 0.5 * sum(abs2, hp(θ))
    g!(G, θ) = (G .= Zygote.gradient(f, θ)[1]; G)
    r = optimize(f, g!, zeros(T, N, N, N),
                 LBFGS(m=15, linesearch=Optim.LineSearches.BackTracking(order=2)),
                 Options(iterations=iters))
    return ω_low .+ hp(minimizer(r))
end
