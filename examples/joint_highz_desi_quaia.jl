# Sheet-native joint high-z reconstruction — DESI QSO (spec) + Quaia (photo-z, χ-inferred).
# Purely phase-space sheet: sheet_tracer for both (Z = ⟨ρ_sheet(randoms)⟩), χ-step density windowless.
# NO survey_window / bin_galaxies / CIC anywhere; the only grid/FFT is the 2LPT ψ.
using DiscoInverse, CUDA, Random, Zygote, Printf, Statistics, NPZ
const DI=DiscoInverse; T=Float32; CUDA.allowscalar(false)
c=fiducial_cosmology(Omega_m=0.3153,Omega_b=0.0493,h=0.6736,sigma8=0.8111,n_s=0.9649); pk=DI.linear_power_spectrum(c)
DESI="/zpool/nvme/data/tabel_scratch/graphGP-cosmology/data/desi/desi_qso_data.npz"
DESIR="/home/tabel/Projects/ECHOES/data_release/desi_qso/desi_qso_randoms.npz"
QUAIA="/home/tabel/Projects/DiscoInverse.jl/scratch/quaia_cat.npz"; QUAIAR="/home/tabel/Projects/ECHOES/data_release/quaia/quaia_randoms.npz"
res=parse(Int,get(ENV,"RES","320")); ext=let e=Int(round(res*1.25)); e+e%2 end; b1_desi=2.3; b1_quaia=2.0
sub(v,n)= length(v)<=n ? collect(eachindex(v)) : round.(Int, range(1,length(v);length=n))
desi=load_desi_qso(DESI,DESIR; zmax=2.1, T=T)
q0=load_quaia(QUAIA;T=T); qr0=load_quaia_randoms(QUAIAR;T=T)
qk=(q0.z_obs.<2.5).&(q0.z_obs.>0.5f0); rk=(qr0.z.<2.5).&(qr0.z.>0.5f0)
quaia=DI.QuaiaCatalog{T}(q0.ra[qk],q0.dec[qk],q0.z_obs[qk],q0.σz[qk]); qran=(ra=qr0.ra[rk],dec=qr0.dec[rk],z=qr0.z[rk])
allr=(ra=vcat(desi.randoms.ra,qran.ra),dec=vcat(desi.randoms.dec,qran.dec),z=vcat(desi.randoms.z,qran.z))
geom=box_geometry(allr,c;res=res,boxsize=nothing); L=T(geom.boxsize); dx=L/res
gm=DI.galaxy_model_for(geom,pk;R=T(dx),n_order=2,ext=ext); sh=reshape(T.(geom.shift),1,3)
# random points (subsampled) for the sheet Monte-Carlo Z — NO survey_window
di=sub(desi.randoms.z,400_000); desi_ran=T.(radec_z_to_cartesian(desi.randoms.ra[di],desi.randoms.dec[di],desi.randoms.z[di],c)).+sh
qi=sub(qran.z,400_000); quaia_ran=T.(radec_z_to_cartesian(qran.ra[qi],qran.dec[qi],qran.z[qi],c)).+sh
tr_desi=sheet_tracer(gm, T.(radec_z_to_cartesian(desi.ra,desi.dec,desi.z,c)).+sh, desi_ran; b1=b1_desi, u=desi.weight)
pq=quaia_problem(quaia, geom, gm, ones(T,res,res,res); b1=b1_quaia)   # dummy window (unused; χ-space quantities only)
nhat_d=CuArray(pq.nhat); shift_d=CuArray(pq.shift); χo_d=CuArray(pq.χ_obs); σχ_d=CuArray(pq.σχ); u_d=CuArray(pq.u)
@printf("SHEET-NATIVE JOINT (no CIC): res=%d box=%.0f Δq=%.1f  DESI=%d(spec)  Quaia=%d(photo-z)  randoms %d+%d\n",
        res, L, dx, length(desi), length(quaia), length(di), length(qi))
rounds=parse(Int,get(ENV,"ROUNDS","3")); pit=parse(Int,get(ENV,"PIT","25")); cit=parse(Int,get(ENV,"CIT","15"))
χ=copy(pq.χ_obs); φ=nothing; local ω; hist=Float64[]
for rd in 1:rounds
    global χ,φ,ω
    tr_q=sheet_tracer(gm, DI.quaia_positions(pq.nhat,pq.shift,χ), quaia_ran; b1=b1_quaia, u=pq.u)
    CUDA.reclaim(); mtp=DI.gpu(multitracer_problem(gm, [tr_desi, tr_q]))
    φg=CuArray(φ===nothing ? 2f0π.*rand(MersenneTwister(303),T,res÷2+1,res,res) : φ); f=φ->DI.multitracer_phase_loss(mtp,φ)
    mm=zero(φg); vv=zero(φg)
    for t in 1:pit
        val,gs=Zygote.withgradient(f,φg); gφ=gs[1]; b1c=1-0.9f0^t; b2c=1-0.999f0^t
        @. mm=0.9f0*mm+0.1f0*gφ; @. vv=0.999f0*vv+0.001f0*gφ*gφ; @. φg -= 0.1f0*(mm/b1c)/(sqrt(vv/b2c)+1f-8)
        gφ=nothing; (t%4==0)&&(GC.gc(false);CUDA.reclaim())
    end
    φ=Array(φg); ω=Array(DI.phase_field(φg)); push!(hist,f(φg)); mm=nothing;vv=nothing;CUDA.reclaim()
    xg,wg=DI._sheet_inputs(mtp.gm, CuArray(ω), T[b1_quaia,0,0]); wg=max.(wg,0f0)   # χ-step density: windowless (sheet)
    ρv,_=DI.nodal_density(xg, wg, res, L)
    fc=cc->DI._quaia_chi_loss(cc, xg, ρv, nhat_d, shift_d, χo_d, σχ_d, u_d, dx, res)
    cg,_,_=DI._lbfgs_generic(fc, cc->Zygote.gradient(fc,cc)[1], CuArray(χ); iters=cit); χ=Array(cg)
    @printf("  round %d phase-loss=%.5e <Δχ>=%.1f Mpc/h\n", rd, hist[end], mean(abs.(χ.-pq.χ_obs))); flush(stdout)
end
# sheet validation: model density at each tracer's galaxies vs its randoms
ωm=CuArray(T.(ω)); mtp=DI.gpu(multitracer_problem(gm,[tr_desi, sheet_tracer(gm, DI.quaia_positions(pq.nhat,pq.shift,χ), quaia_ran; b1=b1_quaia, u=pq.u)]))
xg,wgD=DI._sheet_inputs(mtp.gm, ωm, T[b1_desi,0,0]); ρvD,_=DI.nodal_density(xg,max.(wgD,0f0),res,L)
rgD=mean(Array(DI.interp_sheet_at_points(xg,ρvD,mtp.tracers[1].pts,mtp.tracers[1].cl,res)))/mean(Array(DI.interp_sheet_at_points(xg,ρvD,mtp.tracers[1].ran_pts,mtp.tracers[1].ran_cl,res)))
_,wgQ=DI._sheet_inputs(mtp.gm, ωm, T[b1_quaia,0,0]); ρvQ,_=DI.nodal_density(xg,max.(wgQ,0f0),res,L)
rgQ=mean(Array(DI.interp_sheet_at_points(xg,ρvQ,mtp.tracers[2].pts,mtp.tracers[2].cl,res)))/mean(Array(DI.interp_sheet_at_points(xg,ρvQ,mtp.tracers[2].ran_pts,mtp.tracers[2].ran_cl,res)))
@printf("\nDONE joint: phase-loss %.5e→%.5e median z_rec=%.2f | SHEET density <ρ_gal>/<ρ_ran>: DESI=%.2f Quaia=%.2f (>1 ⇒ in model overdensities)\n",
        hist[1], hist[end], median(pq.z_of_χ.(χ)), rgD, rgQ)
npzwrite("/zpool/nvme/data/tabel_scratch/graphGP-cosmology/data/desi/joint_highz_sheet_r$(res).npz",
    Dict("omega"=>ω,"chi_quaia"=>Float64.(χ),"boxsize"=>Float64(L),"res"=>res,"rD"=>rgD,"rQ"=>rgQ))
println("ALLDONE")
