using DiscoInverse, CUDA, Random, Zygote, Printf, Statistics, NPZ
const DI=DiscoInverse; T=Float32; CUDA.allowscalar(false)
c=fiducial_cosmology(Omega_m=0.3153,Omega_b=0.0493,h=0.6736,sigma8=0.8111,n_s=0.9649); pk=DI.linear_power_spectrum(c)
CF4="/home/tabel/Projects/ECHOES/data/local/cf4/cf4_groups.npz"
cat=load_cf4_groups(CF4; dist_max=250.0, ngal_min=1, T=T)          # well-measured local groups
Rc=T(anchor_cube_side(c)); res=64; L=T(res*Rc)                     # 686 Mpc/h fixed-cube local box
geom=cf4_box_geometry(cat, c; res=res, boxsize=L)                  # observer at box centre
gm=DI.galaxy_model_for(geom, pk; R=Rc, n_order=2)
vc=cf4_velocity_constraint(gm, geom, c, cat)
mtp=DI.gpu(multitracer_problem(gm, DI.Tracer[]; velocity=vc))
@printf("CF4 NESTED LOCAL: res=%d box=%.0f Mpc/h Δq=%.2f  N_groups=%d (dist<250 Mpc)  H0_eff=%.1f  a_near=%.3f\n",
        res, L, L/res, length(cat), DI.cf4_hubble(cat), geom.a_near)
# parent = ΛCDM large-scale carrier (no big-box clustering data reaches z<0.12; large scales are prior)
ω_parent=CuArray(randn(MersenneTwister(7), T, res,res,res))
ksplit=parse(Float64,get(ENV,"KSPLIT","0.03"))                    # ~0 ⇒ CF4 constrains ALL scales incl. the large-scale flow
t0=time(); r=DI.constrained_zoom_realizations(ω_parent, mtp, ksplit, 8; iters=150, seed=1); tr=time()-t0
@printf("k_split_frac=%.2f (modes |k|>%.2f·Nyq free to CF4; smaller ⇒ CF4 owns the large-scale flow)\n", ksplit, ksplit)
ωm=CuArray(T.(r.omega_mean))
# validate: radial-velocity recovery + the density–velocity fσ8 amplitude
amp=DI.velocity_amplitude(mtp.velocity, mtp.gm, ωm)
vr=Array(DI.radial_velocity(mtp.velocity, mtp.gm, ωm)); vobs=Array(mtp.velocity.v_obs)
@printf("\nDONE [%.0fs, 8 draws]: velocity recovery r(model,CF4)=%.3f  |  fσ8 amplitude A=%.3f ± %.3f (A=1 ⇒ fiducial growth)\n",
        tr, cor(vr,vobs), amp.amplitude, amp.sigma)
@printf("  fine-scale posterior spread <ω_std>=%.3f (perturb-and-MAP; CF4 is noise-dominated so this is real UQ)\n", mean(r.omega_std))
out="/zpool/nvme/data/tabel_scratch/graphGP-cosmology/data/local_cf4_nested_r64.npz"
npzwrite(out, Dict("omega_mean"=>r.omega_mean,"omega_std"=>r.omega_std,"boxsize"=>Float64(L),"res"=>res,
                   "vr_model"=>vr,"v_obs"=>vobs,"amplitude"=>amp.amplitude,"sigma"=>amp.sigma,"n_groups"=>length(cat)))
println("saved ",out); println("ALLDONE")
