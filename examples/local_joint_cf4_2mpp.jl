using DiscoInverse, CUDA, Random, Zygote, Printf, Statistics, NPZ
const DI=DiscoInverse; T=Float32; CUDA.allowscalar(false)
c=fiducial_cosmology(Omega_m=0.3153,Omega_b=0.0493,h=0.6736,sigma8=0.8111,n_s=0.9649); pk=DI.linear_power_spectrum(c)
CF4="/home/tabel/Projects/ECHOES/data/local/cf4/cf4_groups.npz"
TWOMPP="/home/tabel/Projects/ECHOES/data_release/local/local_2mpp_observed.npz"
Rc=T(anchor_cube_side(c)); res=64; L=T(res*Rc)                    # 686 Mpc/h fixed-cube local box
cat=load_cf4_groups(CF4; dist_max=250.0, ngal_min=1, T=T)
tmp=load_twompp(TWOMPP; zmax=0.08, n_randoms=400_000, T=T)        # 2M++ density (z<0.08 ≈ 240 Mpc/h)
# geometry from BOTH local samples (observer at box centre)
zc=max.(cat.vcmb./T(299792.458), T(1e-4))
allr=(ra=vcat(cat.ra, tmp.randoms.ra), dec=vcat(cat.dec, tmp.randoms.dec), z=vcat(zc, tmp.randoms.z))
geom=box_geometry(allr, c; res=res, boxsize=L)
gm=DI.galaxy_model_for(geom, pk; R=Rc, n_order=2)
# CF4 velocity (dynamical) + 2M++ density (tracer)
vc=cf4_velocity_constraint(gm, geom, c, cat)
sh=reshape(T.(geom.shift),1,3)
tr2=tracer(gm, T.(radec_z_to_cartesian(tmp.ra,tmp.dec,tmp.z,c)).+sh; b1=1.4, window=survey_window(geom, tmp.randoms))
mtp=DI.gpu(multitracer_problem(gm, [tr2]; velocity=vc))
@printf("LOCAL JOINT (nested): res=%d box=%.0f Δq=%.2f  CF4=%d groups (velocity)  2M++=%d gal (density)\n",
        res, L, L/res, length(cat), length(tmp))
ω_parent=CuArray(randn(MersenneTwister(7), T, res,res,res))       # ΛCDM carrier; hold only the bulk-flow monopole
t0=time(); r=DI.constrained_zoom_realizations(ω_parent, mtp, 0.03, 8; iters=150, seed=1); tr=time()-t0
ωm=CuArray(T.(r.omega_mean))
amp=DI.velocity_amplitude(mtp.velocity, mtp.gm, ωm)
vr=Array(DI.radial_velocity(mtp.velocity, mtp.gm, ωm)); vobs=Array(mtp.velocity.v_obs)
# density recovery: model δ_L vs 2M++ counts (block-averaged in the footprint)
δL=Array(DI.bias_fields(white_noise_to_fphi(mtp.gm.op, ωm), mtp.gm.ops)[1])
W=survey_window(geom, tmp.randoms); nb=DI.bin_galaxies(geom, tmp.ra, tmp.dec, tmp.z)
f=4; mblk=res÷f
blk(A)=(B=zeros(Float64,mblk,mblk,mblk); A3=reshape(Float64.(A),res,res,res); for k in 1:mblk,j in 1:mblk,i in 1:mblk; s=0.0; for cc in 1:f,b in 1:f,a in 1:f; s+=A3[(i-1)*f+a,(j-1)*f+b,(k-1)*f+cc]; end; B[i,j,k]=s; end; B)
Wb=blk(W); nbb=blk(nb); keep=findall(vec(Wb).>0.3*maximum(Wb)); Wn=Wb.*(sum(nbb)/sum(Wb))
δg=[w>0 ? n/w-1 : 0.0 for (n,w) in zip(nbb,Wn)]; dlb=blk(δL)
aa=vec(dlb)[keep].-mean(vec(dlb)[keep]); bb=vec(δg)[keep].-mean(vec(δg)[keep]); rho=sum(aa.*bb)/sqrt(sum(abs2,aa)*sum(abs2,bb))
@printf("\nDONE [%.0fs]: CF4 velocity recovery r=%.3f | 2M++ density recovery r=%.3f @%.0f Mpc/h | fσ8 A=%.3f±%.3f\n",
        tr, cor(vr,vobs), rho, f*L/res, amp.amplitude, amp.sigma)
out="/zpool/nvme/data/tabel_scratch/graphGP-cosmology/data/local_joint_cf4_2mpp_r64.npz"
npzwrite(out, Dict("omega_mean"=>r.omega_mean,"omega_std"=>r.omega_std,"boxsize"=>Float64(L),"res"=>res,
                   "vr_model"=>vr,"v_obs"=>vobs,"amplitude"=>amp.amplitude,"sigma"=>amp.sigma,
                   "n_cf4"=>length(cat),"n_2mpp"=>length(tmp),"r_vel"=>cor(vr,vobs),"r_dens"=>rho))
println("saved ",out); println("ALLDONE")
