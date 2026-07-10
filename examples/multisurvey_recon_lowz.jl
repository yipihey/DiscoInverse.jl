using DiscoInverse, CUDA, Random, Zygote, Printf, Statistics, NPZ
const DI=DiscoInverse; T=Float32
CUDA.allowscalar(false)
c = fiducial_cosmology(Omega_m=0.3153, Omega_b=0.0493, h=0.6736, sigma8=0.8111, n_s=0.9649)  # Planck 2018
pk = DI.linear_power_spectrum(c)
DES  = "/zpool/nvme/data/tabel_scratch/graphGP-cosmology/data/des/des_redmagic_fullz.npz"
DESI = "/zpool/nvme/data/tabel_scratch/graphGP-cosmology/data/desi/desi_qso_data.npz"
DESIR= "/home/tabel/Projects/ECHOES/data_release/desi_qso/desi_qso_randoms.npz"
b1_des, b1_desi = 1.8, 2.5   # DES redMaGiC LRG-like, DESI QSO

# z at the box half-radius (for the DESI cut + reporting)
zbox_of(L) = (χh=L/2; zg=range(0.05,1.5,length=400);
              χg=[DI.comoving_distance(c, 1/(1+z))*c.h for z in zg]; zg[searchsortedfirst(χg, χh)])

function build(res; anchor::Bool, n_rand=2_000_000, use_desi::Bool=true)
    Rc = T(anchor_cube_side(c)); L = anchor ? T(res*Rc) : nothing
    ef = parse(Float64, get(ENV,"EXT_FRAC","1.5"))            # de-aliasing ext = ef·res (1.5 = exact)
    EXT = ef >= 1.5 ? nothing : (e=Int(round(res*ef)); e+e%2)
    des  = load_des_redmagic(DES; n_randoms=n_rand, T=T)
    sh_of(g) = reshape(T.(g.shift), 1, 3)
    if use_desi
        zb   = anchor ? zbox_of(res*Rc) : 0.87
        desi = load_desi_qso(DESI, DESIR; zmax=zb, T=T)
        allr = (ra=vcat(des.randoms.ra, desi.randoms.ra), dec=vcat(des.randoms.dec, desi.randoms.dec),
                z=vcat(des.randoms.z, desi.randoms.z))
        geom = box_geometry(allr, c; res=res, boxsize=L)
        gm = DI.galaxy_model_for(geom, pk; R=(anchor ? Rc : T(geom.boxsize/res)), n_order=2, ext=EXT)
        sh = sh_of(geom)
        tr_des = tracer(gm, T.(radec_z_to_cartesian(des.ra,des.dec,des.z,c)).+sh; b1=b1_des, window=survey_window(geom, des.randoms))
        tr_desi= tracer(gm, T.(radec_z_to_cartesian(desi.ra,desi.dec,desi.z,c)).+sh; b1=b1_desi, window=survey_window(geom, desi.randoms), u=desi.weight)
        mtp = multitracer_problem(gm, [tr_des, tr_desi]); ndesi = length(desi)
    else
        geom = box_geometry(des.randoms, c; res=res, boxsize=L)
        gm = DI.galaxy_model_for(geom, pk; R=(anchor ? Rc : T(geom.boxsize/res)), n_order=2, ext=EXT)
        sh = sh_of(geom)
        tr_des = tracer(gm, T.(radec_z_to_cartesian(des.ra,des.dec,des.z,c)).+sh; b1=b1_des, window=survey_window(geom, des.randoms))
        mtp = multitracer_problem(gm, [tr_des]); ndesi = 0
    end
    @printf("res=%d box=%.0f Δq=%.2f  DES N=%d  DESI N=%d  observer=%s\n",
            res, geom.boxsize, geom.boxsize/res, length(des), ndesi, string(round.(geom.observer)))
    return mtp, geom, length(des), ndesi
end

memGB()=(CUDA.reclaim();(CUDA.total_memory()-CUDA.free_memory())/2^30)

# ---------- 1) cheap wiring validation (res 128, auto-box) ----------
if get(ENV,"SKIP_VAL","0")!="1"
    println("=== VALIDATION (res 128, auto-box) ===")
    mtp,geom = build(128; anchor=false)
    mtpg = DI.gpu(mtp); res=mtp.gm.res
    φ0 = CuArray(2f0π .* rand(MersenneTwister(0), T, res÷2+1, res, res))
    l0 = DI.multitracer_phase_loss(mtpg, φ0)
    g0 = CUDA.@sync Zygote.gradient(φ->DI.multitracer_phase_loss(mtpg, φ), φ0)[1]
    @printf("loss=%.4e  finite=%s  |grad|=%.3e  grad-finite=%s\n",
            l0, isfinite(l0), sqrt(sum(abs2,g0)), all(isfinite, Array(g0)[1:100]))
    mtpg=nothing; g0=nothing; CUDA.reclaim(); GC.gc(); CUDA.reclaim()
end

if get(ENV,"RUN384","0")=="1"
    # ---------- 2) production res-384 anchor run, instrumented ----------
    println("\n=== PRODUCTION (res 384, fixed 16e13 cube) ===")
    mtp,geom,nd,nq = build(384; anchor=true, use_desi=false)
    mtpg = DI.gpu(mtp); res=384
    φ = CuArray(2f0π .* rand(MersenneTwister(101), T, res÷2+1, res, res))
    f = φ->DI.multitracer_phase_loss(mtpg, φ)
    # warm + single-eval cost + memory
    v=f(φ); CUDA.@sync Zygote.gradient(f,φ)
    t0=time(); v1=CUDA.@sync f(φ); tf=time()-t0
    t0=time(); g=CUDA.@sync Zygote.gradient(f,φ)[1]; tg=time()-t0
    @printf("single: fwd=%.2fs  fwd+grad=%.2fs  loss=%.5e  peakGPU=%.1f/%.1f GB\n",
            tf, tg, v1, memGB(), CUDA.total_memory()/2^30)
    # instrumented Adam on phases (fixed amplitude), timing each iter
    ITERS = parse(Int, get(ENV,"ITERS","50")); lr=parse(Float32, get(ENV,"LR","0.1"))
    m=zero(φ); vv=zero(φ); hist=Float64[]; times=Float64[]
    for t in 1:ITERS
        ti=time(); val,gs=Zygote.withgradient(f, φ); gφ=gs[1]
        b1c=1-0.9f0^t; b2c=1-0.999f0^t
        @. m=0.9f0*m+0.1f0*gφ; @. vv=0.999f0*vv+0.001f0*gφ*gφ
        @. φ -= lr*(m/b1c)/(sqrt(vv/b2c)+1f-8)
        gφ=nothing; CUDA.synchronize(); push!(hist,val); push!(times,time()-ti)
        (t%4==0) && (GC.gc(false); CUDA.reclaim())
        (t==1||t%5==0) && @printf("  iter %3d  loss=%.6e  %.2fs  pool=%.1fGB\n", t, val, times[end], memGB())
    end
    ω = Array(DI.phase_field(φ))
    @printf("\nDONE %d iters  median %.2fs/iter  total %.1fs  loss %.5e→%.5e (%.1f%%)\n",
            ITERS, median(times), sum(times), hist[1], hist[end], 100*(hist[1]-hist[end])/abs(hist[1]))
    out="/zpool/nvme/data/tabel_scratch/graphGP-cosmology/data/des/joint_recon_r384.npz"
    npzwrite(out, Dict("omega"=>ω, "loss_history"=>hist, "iter_times"=>times,
                       "boxsize"=>Float64(geom.boxsize), "res"=>384, "n_des"=>nd, "n_desi"=>nq))
    println("saved ", out)
end
println("ALLDONE")
