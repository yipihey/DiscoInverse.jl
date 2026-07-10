using DiscoInverse, CUDA, Random, Zygote, Printf, Statistics, NPZ
const DI=DiscoInverse; T=Float32; CUDA.allowscalar(false)
c = fiducial_cosmology(Omega_m=0.3153, Omega_b=0.0493, h=0.6736, sigma8=0.8111, n_s=0.9649)  # Planck 2018
pk = DI.linear_power_spectrum(c)
DESI = "/zpool/nvme/data/tabel_scratch/graphGP-cosmology/data/desi/desi_qso_data.npz"
DESIR= "/home/tabel/Projects/ECHOES/data_release/desi_qso/desi_qso_randoms.npz"
res=384; ext=let e=Int(round(res*1.25)); e+e%2 end       # reduced de-aliasing (0.9% err, ~30% faster)
zmax=2.1; b1=2.3                                          # DESI QSO clustering range + bias

desi = load_desi_qso(DESI, DESIR; zmax=zmax, T=T)
geom = box_geometry(desi.randoms, c; res=res, boxsize=nothing)   # auto COARSE box (Δq set by box/res)
Δq   = geom.boxsize/res
gm   = DI.galaxy_model_for(geom, pk; R=T(Δq), n_order=2, ext=ext)
sh   = reshape(T.(geom.shift), 1, 3)
pts  = T.(radec_z_to_cartesian(desi.ra, desi.dec, desi.z, c)) .+ sh
tr   = tracer(gm, pts; b1=b1, window=survey_window(geom, desi.randoms), u=desi.weight)
mtp  = DI.gpu(multitracer_problem(gm, [tr]))
@printf("HIGH-Z COARSE CONSTRAINT: res=%d  box=%.0f Mpc/h (%.1f Gpc)  Δq=%.1f Mpc/h  z∈[%.2f,%.2f]\n",
        res, geom.boxsize, geom.boxsize/1000, Δq, minimum(desi.z), maximum(desi.z))
@printf("  DESI QSO (spectroscopic) N=%d  observer=%s  a∈[%.3f,%.3f]\n",
        length(desi), string(round.(geom.observer)), geom.a_far, geom.a_near)
memGB()=(CUDA.reclaim();(CUDA.total_memory()-CUDA.free_memory())/2^30)

φ = CuArray(2f0π .* rand(MersenneTwister(202), T, res÷2+1, res, res))
f = φ->DI.multitracer_phase_loss(mtp, φ)
v=f(φ); CUDA.@sync Zygote.gradient(f,φ)
t0=time(); tg=(CUDA.@sync Zygote.gradient(f,φ); time()-t0)
@printf("single fwd+grad=%.2fs  loss=%.5e  peakGPU=%.1f/%.1f GB\n", tg, v, memGB(), CUDA.total_memory()/2^30)
ITERS=parse(Int,get(ENV,"ITERS","40")); lr=0.1f0
m=zero(φ); vv=zero(φ); hist=Float64[]; times=Float64[]
for t in 1:ITERS
    ti=time(); val,gs=Zygote.withgradient(f,φ); gφ=gs[1]
    b1c=1-0.9f0^t; b2c=1-0.999f0^t
    @. m=0.9f0*m+0.1f0*gφ; @. vv=0.999f0*vv+0.001f0*gφ*gφ
    @. φ -= lr*(m/b1c)/(sqrt(vv/b2c)+1f-8)
    gφ=nothing; CUDA.synchronize(); push!(hist,val); push!(times,time()-ti)
    (t%4==0)&&(GC.gc(false);CUDA.reclaim())
    (t==1||t%5==0) && @printf("  iter %3d  loss=%.6e  %.2fs\n", t, val, times[end])
end
ω=Array(DI.phase_field(φ))
@printf("\nDONE %d iters  median %.2fs/iter  total %.1fs  loss %.5e→%.5e (%.1f%%)\n",
        ITERS, median(times), sum(times), hist[1], hist[end], 100*(hist[1]-hist[end])/abs(hist[1]))
out="/zpool/nvme/data/tabel_scratch/graphGP-cosmology/data/desi/desi_highz_recon_r384.npz"
npzwrite(out, Dict("omega"=>ω, "loss_history"=>hist, "iter_times"=>times,
                   "boxsize"=>Float64(geom.boxsize), "res"=>res, "dq"=>Float64(Δq),
                   "shift"=>Float64.(geom.shift), "n_desi"=>length(desi)))
println("saved ", out); println("ALLDONE")
