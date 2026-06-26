"""
MAP inference driver: minimise `loss(prob, ω, b)` over the white-noise field ω and the
bias parameters b by gradient descent (Optim.jl L-BFGS with Zygote gradients).

The Gaussian ½‖ω‖² prior preconditions the high-dimensional ω (its Hessian is the
identity in the white latent), so plain L-BFGS works without a mass matrix.  ω (res³)
and b (3) are optimised jointly on a flat vector; for stiff problems use
`map_optimize(...; alternate=true)` to alternate ω-steps and b-steps.
"""

using Zygote
using Optim

# Flatten / unflatten (ω, b) ↔ a single vector for Optim.
_flat(ω, b) = vcat(vec(ω), b)
function _unflat(v, res::Int)
    N = res^3
    return reshape(@view(v[1:N]), res, res, res), v[N+1:end]
end

"""
    map_optimize(prob, ω0, b0; iters=100, method=LBFGS(), show_trace=false,
                 fix_bias=false) -> (; ω, b, loss, result)

Joint MAP over (ω, b).  `fix_bias=true` optimises ω only (b held at b0).
"""
function map_optimize(prob, ω0::AbstractArray{T,3}, b0::AbstractVector;
                      iters::Int=100, method=Optim.LBFGS(), show_trace::Bool=false,
                      fix_bias::Bool=false) where {T}
    res = prob.gm.res; N = res^3
    bfix = Vector{T}(b0)
    f(v) = fix_bias ? loss(prob, reshape(@view(v[1:N]), res, res, res), bfix) :
                      loss(prob, _unflat(v, res)...)
    function g!(G, v)
        if fix_bias
            ω = reshape(v[1:N], res, res, res)
            gω = Zygote.gradient(w -> loss(prob, w, bfix), ω)[1]
            G[1:N] .= vec(gω)
        else
            ω, b = _unflat(v, res)
            gω, gb = Zygote.gradient((w, bb) -> loss(prob, w, bb), ω, b)
            G[1:N] .= vec(gω); G[N+1:end] .= gb
        end
        return G
    end
    v0 = fix_bias ? vec(ω0) : _flat(ω0, bfix)
    r  = Optim.optimize(f, g!, collect(v0), method,
                        Optim.Options(iterations=iters, show_trace=show_trace))
    vmin = Optim.minimizer(r)
    ωh = reshape(vmin[1:N], res, res, res)
    bh = fix_bias ? bfix : vmin[N+1:end]
    return (ω=ωh, b=bh, loss=Optim.minimum(r), result=r)
end

"""
    adam_optimize(prob, ω0, b0; iters=200, lr=0.05, fix_bias=false, b_lr=lr,
                  β1=0.9, β2=0.999, ϵ=1e-8, show_every=0) -> (; ω, b, loss, history)

Device-resident Adam MAP. ω stays on whatever backend `ω0` lives on (`Array` or
`CuArray`) — the whole update is broadcasts + reductions, so a GPU `prob` (`gpu(prob)`)
and a `CuArray` ω0 keep the entire optimization on the device, driving the validated GPU
forward.  The 3 bias params stay on the host.  `fix_bias=true` optimises ω only.  Adam
(rather than L-BFGS) avoids the host scalar line-search that would sync the GPU every
step; the ½‖ω‖² prior keeps the problem well-scaled so a fixed lr converges.
"""
function adam_optimize(prob, ω0::AbstractArray{T,3}, b0::AbstractVector;
                       iters::Int=200, lr::Real=0.05, fix_bias::Bool=false, b_lr::Real=lr,
                       β1::Real=0.9, β2::Real=0.999, ϵ::Real=1e-8, show_every::Int=0) where {T}
    ω  = copy(ω0); b = collect(float.(b0))
    mω = zero(ω); vω = zero(ω); mb = zero(b); vb = zero(b)
    lrω = T(lr); β1ω = T(β1); β2ω = T(β2); ϵω = T(ϵ)
    history = Float64[]
    for t in 1:iters
        if fix_bias
            val, gs = Zygote.withgradient(w -> loss(prob, w, b), ω); gω = gs[1]
        else
            val, gs = Zygote.withgradient((w, bb) -> loss(prob, w, bb), ω, b)
            gω, gb = gs
        end
        push!(history, val)
        bc1 = 1 - β1^t; bc2 = 1 - β2^t
        @. mω = β1ω * mω + (1 - β1ω) * gω
        @. vω = β2ω * vω + (1 - β2ω) * gω * gω
        @. ω -= lrω * (mω / bc1) / (sqrt(vω / bc2) + ϵω)
        if !fix_bias
            @. mb = β1 * mb + (1 - β1) * gb
            @. vb = β2 * vb + (1 - β2) * gb * gb
            @. b -= b_lr * (mb / bc1) / (sqrt(vb / bc2) + ϵ)
        end
        show_every > 0 && (t == 1 || t % show_every == 0) &&
            @info "adam" iter=t loss=val b=(fix_bias ? b0 : b)
    end
    return (ω=ω, b=b, loss=loss(prob, ω, b), history=history)
end

"""
    map_optimize_alternating(prob, ω0, b0; rounds=5, ω_iters=30, b_iters=20) -> (; ω, b, loss)

Alternate ω-optimisation (L-BFGS, bias fixed) and bias-optimisation (L-BFGS, ω fixed) —
robust against the ω-amplitude/bias degeneracy early on.
"""
function map_optimize_alternating(prob::InferenceProblem{T}, ω0::AbstractArray{T,3},
                                  b0::AbstractVector; rounds::Int=5, ω_iters::Int=30,
                                  b_iters::Int=20, show_trace::Bool=false) where {T}
    ω = copy(ω0); b = Vector{T}(b0)
    local ℓ = loss(prob, ω, b)
    for _ in 1:rounds
        rω = map_optimize(prob, ω, b; iters=ω_iters, fix_bias=true, show_trace=show_trace)
        ω = rω.ω
        # bias step: tiny 3-D problem, optimise directly
        fb(bb) = loss(prob, ω, bb)
        rb = Optim.optimize(bb -> fb(bb), bb -> Zygote.gradient(fb, bb)[1], b,
                            Optim.LBFGS(), Optim.Options(iterations=b_iters); inplace=false)
        b = Optim.minimizer(rb); ℓ = Optim.minimum(rb)
    end
    return (ω=ω, b=b, loss=ℓ)
end
