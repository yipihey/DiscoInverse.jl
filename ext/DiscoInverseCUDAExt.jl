"""
DiscoInverseCUDAExt — move a forward model to the GPU.

`gpu(gm)` rebuilds the `GalaxyModel` with its spectral operators, kernels and the
Lagrangian grid on the device (cosmology stays on the host — the lightcone crossing
copies its growth/χ tables to the device on demand).  The whole differentiable forward
`galaxy_density(gm, ω, b)` then runs on `CuArray`s (cuFFT nLPT + vectorised crossing +
KA atomic-scatter deposit) — ~30× the CPU forward, ~33× the gradient (res=64).
"""
module DiscoInverseCUDAExt

using DiscoInverse
using DiscoInverse: GalaxyModel, BiasOperators, InferenceProblem, SheetProblem, Tracer, SheetTracer, MultiTracerProblem, LensingConstraint, VelocityConstraint
using DiscoDJNative
using CUDA

DiscoInverse.gpu(ops::BiasOperators{T}) where {T} =
    BiasOperators{T, CuArray{T,3}}(ops.res, ops.boxsize, ops.R,
        CuArray(ops.k2), CuArray(ops.kxkx), CuArray(ops.kyky), CuArray(ops.kzkz),
        CuArray(ops.kxky), CuArray(ops.kxkz), CuArray(ops.kykz), CuArray(ops.W_R))

function DiscoInverse.gpu(gm::GalaxyModel{T}) where {T}
    op = DiscoDJNative.to_gpu(gm.op)
    K  = DiscoDJNative.to_gpu(gm.K)
    ops = DiscoInverse.gpu(gm.ops)
    qf  = CuArray(gm.qflat)
    return GalaxyModel{T, typeof(op), typeof(K), typeof(ops), typeof(gm.cosmo)}(
        gm.res, gm.boxsize, gm.n_order, gm.n_sub, gm.rsd, gm.a_far, gm.a_near,
        op, K, ops, qf, gm.observer, gm.cosmo, gm.sigma2, gm.s2mean)
end

# Move the whole inference problem to the device: GPU forward + the data fields
# (window/mask/counts) as CuArrays, so `adam_optimize(gpu(prob), CuArray(ω0), b0)` runs
# the entire MAP loop on the A6000.  The bias prior (3 params) stays on the host.
function DiscoInverse.gpu(prob::InferenceProblem{T}) where {T}
    gmg = DiscoInverse.gpu(prob.gm)
    dv  = prob.data_var === nothing ? nothing : CuArray(prob.data_var)
    return InferenceProblem{T, typeof(gmg)}(gmg, CuArray(prob.W), CuArray(prob.mask),
        CuArray(prob.n_obs), prob.ntot, prob.b0, prob.σb, prob.λfloor, dv)
end

# Move a grid-free sheet problem to the device: GPU forward + the per-galaxy weights `u`
# as a CuArray (loss broadcasts `u .* log ρ_g`, so `u` must match the device ρ_g).  The
# galaxy positions / cell list stay on the host — the deposit kernels copy them to the
# backend per call anyway (`mv` in sheet_density.jl); the bias 3-vectors stay on the host.
function DiscoInverse.gpu(prob::SheetProblem{T}) where {T}
    gmg = DiscoInverse.gpu(prob.gm); ug = CuArray(prob.u)
    win = prob.window  === nothing ? nothing : CuArray(prob.window)
    act = prob.active  === nothing ? nothing : CuArray(prob.active)
    rp  = prob.ran_pts === nothing ? nothing : CuArray(prob.ran_pts)   # randoms → device; cell list stays host
    return SheetProblem{T, typeof(gmg), typeof(prob.pts), typeof(prob.cl), typeof(ug), typeof(win), typeof(act), typeof(rp), typeof(prob.ran_cl)}(
        gmg, prob.pts, prob.cl, ug, prob.Utot, prob.b0, prob.σb, prob.ρfloor, prob.floor_frac, prob.c0, win, act, rp, prob.ran_cl)
end

# Move a joint multi-tracer problem to the device: GPU forward + each tracer's positions/window/weights
# as CuArrays (cell lists stay host — the deposit kernels copy points to the backend per call).
function DiscoInverse.gpu(tr::Tracer)
    return Tracer(CuArray(tr.pts), tr.cl, CuArray(tr.window), tr.b0, CuArray(tr.u), tr.Utot)
end
function DiscoInverse.gpu(tr::SheetTracer)   # gal + random points to device; cell lists stay host
    return SheetTracer(CuArray(tr.pts), tr.cl, CuArray(tr.ran_pts), tr.ran_cl, tr.b0, CuArray(tr.u), tr.Utot)
end
DiscoInverse.gpu(lc::LensingConstraint) = LensingConstraint(
    CuArray(lc.pts), lc.cl, lc.nd, lc.ns, CuArray(lc.Wdχ), CuArray(lc.κ_obs), CuArray(lc.invN),
    CuArray(lc.ones3), lc.ρfloor)

# cell list stays host (the deposit/interp kernels copy pts to the backend per call); rhat/v_obs/invN → device
DiscoInverse.gpu(vc::VelocityConstraint) = VelocityConstraint(
    CuArray(vc.pts), vc.cl, CuArray(vc.rhat), CuArray(vc.v_obs), CuArray(vc.invN), vc.vnorm, vc.submean)

function DiscoInverse.gpu(mtp::MultiTracerProblem{T}) where {T}
    gmg = DiscoInverse.gpu(mtp.gm); trs = [DiscoInverse.gpu(tr) for tr in mtp.tracers]
    lc  = mtp.lensing === nothing ? nothing : DiscoInverse.gpu(mtp.lensing)
    vc  = mtp.velocity === nothing ? nothing : DiscoInverse.gpu(mtp.velocity)
    return MultiTracerProblem{T, typeof(gmg), typeof(trs), typeof(lc), typeof(vc)}(
        gmg, trs, lc, vc, mtp.ρfloor, mtp.floor_frac)
end

end # module
