using DiscoInverse, CUDA, Random, Zygote, Printf, Statistics, NPZ
const DI=DiscoInverse; T=Float32; CUDA.allowscalar(false)
c=fiducial_cosmology(Omega_m=0.3153,Omega_b=0.0493,h=0.6736,sigma8=0.8111,n_s=0.9649); pk=DI.linear_power_spectrum(c)
DESI="/zpool/nvme/data/tabel_scratch/graphGP-cosmology/data/desi/desi_qso_data.npz"
DESIR="/home/tabel/Projects/ECHOES/data_release/desi_qso/desi_qso_randoms.npz"
res=384; ext=let e=Int(round(res*1.25)); e+e%2 end; b1=2.3
desi=load_desi_qso(DESI,DESIR; zmax=2.1, T=T)
geom=box_geometry(desi.randoms, c; res=res, boxsize=nothing); L=T(geom.boxsize); Δq=L/res    # box sizing only (no deposit)
gm=DI.galaxy_model_for(geom, pk; R=T(Δq), n_order=2, ext=ext); sh=reshape(T.(geom.shift),1,3)
nr=min(1_500_000, length(desi.randoms.z)); ridx=round.(Int, range(1,length(desi.randoms.z);length=nr))
gal=T.(radec_z_to_cartesian(desi.ra,desi.dec,desi.z,c)).+sh
ran=T.(radec_z_to_cartesian(desi.randoms.ra[ridx],desi.randoms.dec[ridx],desi.randoms.z[ridx],c)).+sh
tr=sheet_tracer(gm, gal, ran; b1=b1, u=desi.weight)              # sheet-native — NO survey_window
mtp=DI.gpu(multitracer_problem(gm, [tr]))
@printf("SHEET-NATIVE DESI HIGH-Z (no CIC): res=%d box=%.0f Δq=%.1f  DESI=%d gal + %d randoms\n",
        res, L, Δq, length(desi), nr)
φ=CuArray(2f0π .* rand(MersenneTwister(202), T, res÷2+1,res,res)); f=φ->DI.multitracer_phase_loss(mtp, φ)
v=f(φ); CUDA.@sync Zygote.gradient(f,φ)
ITERS=parse(Int,get(ENV,"ITERS","40")); lr=0.1f0; m=zero(φ); vv=zero(φ); hist=Float64[]; tt=Float64[]
for t in 1:ITERS
    ti=time(); val,gs=Zygote.withgradient(f,φ); gφ=gs[1]; b1c=1-0.9f0^t; b2c=1-0.999f0^t
    @. m=0.9f0*m+0.1f0*gφ; @. vv=0.999f0*vv+0.001f0*gφ*gφ; @. φ -= lr*(m/b1c)/(sqrt(vv/b2c)+1f-8)
    gφ=nothing; CUDA.synchronize(); push!(hist,val); push!(tt,time()-ti); (t%4==0)&&(GC.gc(false);CUDA.reclaim())
    (t==1||t%10==0)&&@printf("  iter %3d loss=%.5e %.2fs\n",t,val,tt[end])
end
ωm=DI.phase_field(φ)
# SHEET-NATIVE validation: model tessellation density at DESI QSO vs at randoms (no binning)
xg,wg=DI._sheet_inputs(mtp.gm, ωm, T[b1,0,0]); wg=max.(wg,0f0); ρv,_=DI.nodal_density(xg,wg,res,L)
ρgal=Array(DI.interp_sheet_at_points(xg,ρv,mtp.tracers[1].pts,mtp.tracers[1].cl,res))
ρran=Array(DI.interp_sheet_at_points(xg,ρv,mtp.tracers[1].ran_pts,mtp.tracers[1].ran_cl,res))
@printf("\nDONE %d iters median %.2fs/iter loss %.5e→%.5e | SHEET density: <ρ_gal>/<ρ_ran>=%.3f (>1 ⇒ QSO in model overdensities) Δ⟨logρ⟩=%.3f\n",
        ITERS, median(tt), hist[1], hist[end], mean(ρgal)/mean(ρran), mean(log.(max.(ρgal,1f-8)))-mean(log.(max.(ρran,1f-8))))
println("ALLDONE")
