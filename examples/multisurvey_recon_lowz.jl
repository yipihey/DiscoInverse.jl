# Sheet-native low-z reconstruction on the fixed 16e13 cube — DES redMaGiC (z<0.87).
# Purely phase-space sheet: sheet_tracer (Z = ⟨ρ_sheet(randoms)⟩), NO survey_window / bin_galaxies / CIC.
# The only grid/FFT anywhere is the 2LPT ψ calculation.
using DiscoInverse, CUDA, Random, Zygote, Printf, Statistics, NPZ
const DI=DiscoInverse; T=Float32; CUDA.allowscalar(false)
c=fiducial_cosmology(Omega_m=0.3153,Omega_b=0.0493,h=0.6736,sigma8=0.8111,n_s=0.9649); pk=DI.linear_power_spectrum(c)
DES="/zpool/nvme/data/tabel_scratch/graphGP-cosmology/data/des/des_redmagic_fullz.npz"
res=384; Rc=T(anchor_cube_side(c)); L=T(res*Rc); ext=let e=Int(round(res*1.25)); e+e%2 end; b1=1.8
des=load_des_redmagic(DES; n_randoms=400_000, T=T)              # ~400k randoms is plenty for the Monte-Carlo Z
geom=box_geometry(des.randoms, c; res=res, boxsize=L)          # box sizing only (radec→cart extents; no deposit)
gm=DI.galaxy_model_for(geom, pk; R=Rc, n_order=2, ext=ext); sh=reshape(T.(geom.shift),1,3)
gal=T.(radec_z_to_cartesian(des.ra,des.dec,des.z,c)).+sh
ran=T.(radec_z_to_cartesian(des.randoms.ra,des.randoms.dec,des.randoms.z,c)).+sh
tr=sheet_tracer(gm, gal, ran; b1=b1)                          # sheet-native — NO survey_window
mtp=DI.gpu(multitracer_problem(gm, [tr]))
@printf("SHEET-NATIVE DES low-z (no CIC): res=%d box=%.0f Δq=%.2f  DES=%d gal + %d randoms\n",
        res, L, L/res, length(des), size(ran,1))
φ=CuArray(2f0π .* rand(MersenneTwister(101), T, res÷2+1,res,res)); f=φ->DI.multitracer_phase_loss(mtp, φ)
f(φ); CUDA.@sync Zygote.gradient(f,φ)
ITERS=parse(Int,get(ENV,"ITERS","40")); lr=0.1f0; m=zero(φ); vv=zero(φ); hist=Float64[]; tt=Float64[]
for t in 1:ITERS
    ti=time(); val,gs=Zygote.withgradient(f,φ); gφ=gs[1]; b1c=1-0.9f0^t; b2c=1-0.999f0^t
    @. m=0.9f0*m+0.1f0*gφ; @. vv=0.999f0*vv+0.001f0*gφ*gφ; @. φ -= lr*(m/b1c)/(sqrt(vv/b2c)+1f-8)
    gφ=nothing; CUDA.synchronize(); push!(hist,val); push!(tt,time()-ti); (t%4==0)&&(GC.gc(false);CUDA.reclaim())
    (t==1||t%10==0)&&@printf("  iter %3d loss=%.5e %.2fs\n",t,val,tt[end])
end
ωm=DI.phase_field(φ)
xg,wg=DI._sheet_inputs(mtp.gm, ωm, T[b1,0,0]); wg=max.(wg,0f0); ρv,_=DI.nodal_density(xg,wg,res,L)
ρgal=Array(DI.interp_sheet_at_points(xg,ρv,mtp.tracers[1].pts,mtp.tracers[1].cl,res))
ρran=Array(DI.interp_sheet_at_points(xg,ρv,mtp.tracers[1].ran_pts,mtp.tracers[1].ran_cl,res))
@printf("\nDONE %d iters median %.2fs/iter loss %.5e→%.5e | SHEET density <ρ_gal>/<ρ_ran>=%.3f (>1 ⇒ DES in model overdensities)\n",
        ITERS, median(tt), hist[1], hist[end], mean(ρgal)/mean(ρran))
npzwrite("/zpool/nvme/data/tabel_scratch/graphGP-cosmology/data/des/des_lowz_sheet_r384.npz",
    Dict("omega"=>Array(ωm),"boxsize"=>Float64(L),"res"=>res,"ratio"=>mean(ρgal)/mean(ρran),"n_des"=>length(des)))
println("ALLDONE")
