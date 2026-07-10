using DiscoInverse, CUDA, Random, Zygote, Printf, Statistics, NPZ
const DI=DiscoInverse; T=Float32; CUDA.allowscalar(false)
c=fiducial_cosmology(Omega_m=0.3153,Omega_b=0.0493,h=0.6736,sigma8=0.8111,n_s=0.9649); pk=DI.linear_power_spectrum(c)
CF4="/home/tabel/Projects/ECHOES/data/local/cf4/cf4_groups.npz"; TWOMPP="/home/tabel/Projects/ECHOES/data_release/local/local_2mpp_observed.npz"
Rc=T(anchor_cube_side(c)); res=64; L=T(res*Rc)
cat=load_cf4_groups(CF4; dist_max=250.0, T=T); tmp=load_twompp(TWOMPP; zmax=0.08, n_randoms=400_000, T=T)
zc=max.(cat.vcmb./T(299792.458),T(1e-4))
allr=(ra=vcat(cat.ra,tmp.randoms.ra),dec=vcat(cat.dec,tmp.randoms.dec),z=vcat(zc,tmp.randoms.z))  # box sizing only (no deposit)
geom=box_geometry(allr, c; res=res, boxsize=L)
gm=DI.galaxy_model_for(geom, pk; R=Rc, n_order=2); sh=reshape(T.(geom.shift),1,3)
vc=cf4_velocity_constraint(gm, geom, c, cat)                       # velocity: already sheet-native (interp at CF4 pts)
gal=T.(radec_z_to_cartesian(tmp.ra,tmp.dec,tmp.z,c)).+sh
ran=T.(radec_z_to_cartesian(tmp.randoms.ra,tmp.randoms.dec,tmp.randoms.z,c)).+sh
tr2=sheet_tracer(gm, gal, ran; b1=1.4)                            # sheet-native density — NO survey_window/CIC
mtp=DI.gpu(multitracer_problem(gm, [tr2]; velocity=vc))
@printf("SHEET-NATIVE LOCAL (only 2LPT uses a grid; NO survey_window, NO bin_galaxies):\n  res=%d box=%.0f Δq=%.2f  CF4=%d groups  2M++=%d gal + %d randoms\n",
        res, L, L/res, length(cat), length(tmp), size(ran,1))
ω_parent=CuArray(randn(MersenneTwister(7), T, res,res,res))
t0=time(); r=DI.constrained_zoom_realizations(ω_parent, mtp, 0.03, 8; iters=150, seed=1); tr=time()-t0
ωm=CuArray(T.(r.omega_mean))
amp=DI.velocity_amplitude(mtp.velocity, mtp.gm, ωm); vr=Array(DI.radial_velocity(mtp.velocity, mtp.gm, ωm)); vobs=Array(mtp.velocity.v_obs)
# SHEET-NATIVE density validation: model tessellation density at 2M++ galaxies vs at randoms (no binning)
xg,wg=DI._sheet_inputs(mtp.gm, ωm, T[1.4,0,0]); wg=max.(wg, 0f0)
ρv,_=DI.nodal_density(xg, wg, res, L)
ρgal=Array(DI.interp_sheet_at_points(xg, ρv, mtp.tracers[1].pts, mtp.tracers[1].cl, res))
ρran=Array(DI.interp_sheet_at_points(xg, ρv, mtp.tracers[1].ran_pts, mtp.tracers[1].ran_cl, res))
ratio=mean(ρgal)/mean(ρran); dlog=mean(log.(max.(ρgal,1f-8)))-mean(log.(max.(ρran,1f-8)))
@printf("\nDONE [%.0fs]: CF4 velocity r=%.3f, fσ8 A=%.3f±%.3f | SHEET density: <ρ_gal>/<ρ_ran>=%.3f (>1 ⇒ 2M++ in model overdensities), Δ⟨logρ⟩=%.3f\n",
        tr, cor(vr,vobs), amp.amplitude, amp.sigma, ratio, dlog)
out="/zpool/nvme/data/tabel_scratch/graphGP-cosmology/data/local_sheet_cf4_2mpp_r64.npz"
npzwrite(out, Dict("omega_mean"=>r.omega_mean,"omega_std"=>r.omega_std,"boxsize"=>Float64(L),"res"=>res,
                   "vr_model"=>vr,"v_obs"=>vobs,"rho_gal"=>ρgal,"rho_ran"=>ρran,
                   "ratio"=>ratio,"amplitude"=>amp.amplitude,"n_cf4"=>length(cat),"n_2mpp"=>length(tmp)))
println("saved ",out); println("ALLDONE")
