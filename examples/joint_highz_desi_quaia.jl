using DiscoInverse, CUDA, Random, Zygote, Printf, Statistics, NPZ
const DI=DiscoInverse; T=Float32; CUDA.allowscalar(false)
c = fiducial_cosmology(Omega_m=0.3153, Omega_b=0.0493, h=0.6736, sigma8=0.8111, n_s=0.9649)
pk = DI.linear_power_spectrum(c)
DESI ="/zpool/nvme/data/tabel_scratch/graphGP-cosmology/data/desi/desi_qso_data.npz"
DESIR="/home/tabel/Projects/ECHOES/data_release/desi_qso/desi_qso_randoms.npz"
QUAIA="/home/tabel/Projects/DiscoInverse.jl/scratch/quaia_cat.npz"
QUAIAR="/home/tabel/Projects/ECHOES/data_release/quaia/quaia_randoms.npz"
res=parse(Int,get(ENV,"RES","320")); ext=let ef=parse(Float64,get(ENV,"EXT_FRAC","1.25")); e=Int(round(res*ef)); e+e%2 end
zmax=2.5; b1_desi=2.3; b1_quaia=2.0

# ---- data ----
desi = load_desi_qso(DESI, DESIR; zmax=2.1, T=T)
q0   = load_quaia(QUAIA; T=T); qr0 = load_quaia_randoms(QUAIAR; T=T)
qk   = (q0.z_obs .< zmax) .& (q0.z_obs .> 0.5f0)                      # focus the high-z band
quaia= DI.QuaiaCatalog{T}(q0.ra[qk], q0.dec[qk], q0.z_obs[qk], q0.σz[qk])
rk   = (qr0.z .< zmax) .& (qr0.z .> 0.5f0)
qran = (ra=qr0.ra[rk], dec=qr0.dec[rk], z=qr0.z[rk])
# ---- geometry (box covers DESI + Quaia to z<2.5) ----
allr = (ra=vcat(desi.randoms.ra,qran.ra), dec=vcat(desi.randoms.dec,qran.dec), z=vcat(desi.randoms.z,qran.z))
geom = box_geometry(allr, c; res=res, boxsize=nothing); L=T(geom.boxsize); dx=L/res
gm   = DI.galaxy_model_for(geom, pk; R=T(dx), n_order=2, ext=ext)
sh   = reshape(T.(geom.shift),1,3)
@printf("JOINT HIGH-Z: res=%d box=%.0f Mpc/h Δq=%.1f  DESI N=%d (spec)  Quaia N=%d (photo-z)\n",
        res, L, dx, length(desi), length(quaia))
# ---- DESI tracer (fixed spectroscopic positions) ----
tr_desi = tracer(gm, T.(radec_z_to_cartesian(desi.ra,desi.dec,desi.z,c)).+sh; b1=b1_desi,
                 window=survey_window(geom, desi.randoms), u=desi.weight)
# ---- Quaia χ-space (photo-z; χ inferred) ----
qwin = survey_window(geom, qran)
pq   = quaia_problem(quaia, geom, gm, qwin; b1=b1_quaia)
win_d = CuArray(pq.window)
nhat_d=CuArray(pq.nhat); shift_d=CuArray(pq.shift); χo_d=CuArray(pq.χ_obs); σχ_d=CuArray(pq.σχ); u_d=CuArray(pq.u)
memGB()=(CUDA.reclaim();(CUDA.total_memory()-CUDA.free_memory())/2^30)

# ---- alternating joint reconstruction ----
rounds=parse(Int,get(ENV,"ROUNDS","3")); pit=parse(Int,get(ENV,"PIT","20")); cit=parse(Int,get(ENV,"CIT","15"))
χ = copy(pq.χ_obs); φ=nothing; local ω; hist=Float64[]
for rd in 1:rounds
    global χ, φ, ω
    qpts = DI.quaia_positions(pq.nhat, pq.shift, χ)                              # host Quaia positions at χ
    tr_q = tracer(gm, qpts; b1=b1_quaia, window=pq.window, u=pq.u)
    CUDA.reclaim()
    mtp  = DI.gpu(multitracer_problem(gm, [tr_desi, tr_q]))
    φg   = CuArray(φ===nothing ? 2f0π.*rand(MersenneTwister(303), T, res÷2+1,res,res) : φ)
    f    = φ->DI.multitracer_phase_loss(mtp, φ)
    mm=zero(φg); vv=zero(φg)                                                     # adam (no L-BFGS history → lighter)
    for t in 1:pit
        val,gs=Zygote.withgradient(f, φg); gφ=gs[1]
        b1c=1-0.9f0^t; b2c=1-0.999f0^t
        @. mm=0.9f0*mm+0.1f0*gφ; @. vv=0.999f0*vv+0.001f0*gφ*gφ
        @. φg -= 0.1f0*(mm/b1c)/(sqrt(vv/b2c)+1f-8)
        gφ=nothing; (t%4==0)&&(GC.gc(false);CUDA.reclaim())
    end
    φ = Array(φg); ω = Array(DI.phase_field(φg))
    push!(hist, f(φg)); mm=nothing; vv=nothing; CUDA.reclaim()
    # χ step: infer Quaia distances at the joint field (Quaia bias/window)
    xg,wg = DI._sheet_inputs(mtp.gm, CuArray(ω), T[b1_quaia,0,0])
    ρv,_  = DI.nodal_density(xg, DI._apply_window(wg, win_d), res, L)
    fc    = cc->DI._quaia_chi_loss(cc, xg, ρv, nhat_d, shift_d, χo_d, σχ_d, u_d, dx, res)
    cg,_,_= DI._lbfgs_generic(fc, cc->Zygote.gradient(fc,cc)[1], CuArray(χ); iters=cit)
    χ = Array(cg)
    @printf("  round %d: phase-loss=%.5e  <Δχ>=%.1f Mpc/h  peakGPU=%.1f GB\n",
            rd, hist[end], mean(abs.(χ.-pq.χ_obs)), memGB()); flush(stdout)
end
z_rec = pq.z_of_χ.(χ)
@printf("\nDONE joint DESI+Quaia: phase-loss %.5e→%.5e  median z_rec=%.2f\n", hist[1], hist[end], median(z_rec))
out="/zpool/nvme/data/tabel_scratch/graphGP-cosmology/data/desi/joint_highz_desi_quaia_r384.npz"
npzwrite(out, Dict("omega"=>ω, "chi_quaia"=>Float64.(χ), "z_quaia"=>Float64.(z_rec),
                   "boxsize"=>Float64(L), "res"=>res, "n_desi"=>length(desi), "n_quaia"=>length(quaia),
                   "shift"=>Float64.(geom.shift)))
println("saved ", out); println("ALLDONE")
