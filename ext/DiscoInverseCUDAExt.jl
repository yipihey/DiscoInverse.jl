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
using DiscoInverse: GalaxyModel, BiasOperators
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

end # module
