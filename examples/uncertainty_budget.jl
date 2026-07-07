# Cheap, statistically-sound uncertainty budget for the crude nLPT+bias forward.
# Reports, per |k|-band: r(k) [primary UQ] + statistical width [counts bootstrap]
# + systematic floors [LPT order, bias amplitude, bias shape]. See docs/uncertainty_budget.md.
# Run: julia --project=. examples/uncertainty_budget.jl   (~13 min CPU at res 8)
using DiscoInverse, Random, Statistics, Printf, LinearAlgebra, FFTW
const DI = DiscoInverse
T = Float64
cosmo = fiducial_cosmology(); pk = linear_power_spectrum(cosmo)
res = 8; L = 400.0; obs = [-1300.0, L/2, L/2]
gm2 = galaxy_model(res, L, cosmo, pk; R=40.0, observer=obs, a_far=0.4, a_near=1.0, n_order=2, n_sub=1, rsd=false)
gm3 = galaxy_model(res, L, cosmo, pk; R=40.0, observer=obs, a_far=0.4, a_near=1.0, n_order=3, n_sub=1, rsd=false)
W=ones(res,res,res); mask=ones(res,res,res); btrue=[1.8,0.5,0.3]; ntot=10.0*res^3
probt = DI.InferenceProblem{T,typeof(gm2)}(gm2,W,mask,zeros(res,res,res),ntot,btrue,[1e-4,1e-4,1e-4],1e-6)
ωtrue = randn(MersenneTwister(1), res,res,res)
prob0 = DI.inject_mock(probt, ωtrue, btrue; ntot=ntot, seed=101); nobs=prob0.n_obs
nb=4
# per-k-band membership on the rfft grid
kf=[(i-1)<res÷2 ? i-1 : i-1-res for i in 1:res]; kh=[i-1 for i in 1:res÷2+1]
kg=[sqrt(kh[a]^2+kf[b]^2+kf[c]^2) for a in 1:res÷2+1, b in 1:res, c in 1:res]
edges=range(0, maximum(kg)+1e-6, length=nb+1)
band(k)= clamp(searchsortedlast(edges, k), 1, nb)
bandRMS(δ) = (F=rfft(δ); [sqrt(mean(abs2.(F[band.(kg).==bb]))) for bb in 1:nb])   # per-band RMS amplitude
function recon(gm,bassume;counts=nobs,iters=45,seed=0)
    p=DI.InferenceProblem{T,typeof(gm)}(gm,W,mask,counts,ntot,bassume,[1e-4,1e-4,1e-4],1e-6)
    DI.map_optimize(p, 1e-3 .* randn(MersenneTwister(seed),res,res,res), copy(bassume); iters=iters).ω
end
rk(ω)=DI.cross_spectrum_r(ω,ωtrue;nbins=nb,boxsize=L).r
@printf("UBUDGET2 res=%d ntot=%.0f true bias=%s — BAND-RESOLVED\n",res,ntot,btrue); flush(stdout)
ωfid=recon(gm2,btrue;seed=1); Ffid=bandRMS(ωfid)
rkf=rk(ωfid)
# statistical per band (bootstrap K=6)
boot=Vector{Array{T,3}}()
for k in 1:6
    rng=MersenneTwister(700+k); nbc=similar(nobs)
    for i in eachindex(nobs); nbc[i]=T(DI._rand_poisson(rng,max(nobs[i],0.0))); end
    push!(boot, recon(gm2,btrue;counts=nbc,seed=800+k))
end
# statistical per band: std across bootstrap of the band-projected field
function bandproj(δ,bb); F=rfft(δ); F[band.(kg).!=bb].=0; irfft(F,res); end
statb=[ (S=std(cat([bandproj(boot[k],bb) for k in 1:length(boot)]...;dims=4);dims=4); sqrt(mean(S.^2))) for bb in 1:nb]
# systematics per band (RMS difference from fiducial, band-projected)
ω3=recon(gm3,btrue;seed=1); ωlin=recon(gm2,[btrue[1],0,0];seed=1)
ωhi=recon(gm2,[btrue[1]*1.2,btrue[2],btrue[3]];seed=1); ωlo=recon(gm2,[btrue[1]*0.8,btrue[2],btrue[3]];seed=1)
sysband(ωa)=[ (d=bandproj(ωa.-ωfid,bb); sqrt(mean(d.^2))) for bb in 1:nb]
sLPT=sysband(ω3); slin=sysband(ωlin); sb1=[ (d=bandproj((ωhi.-ωlo)./2,bb); sqrt(mean(d.^2))) for bb in 1:nb]
fidb=[ (d=bandproj(ωfid,bb); sqrt(mean(d.^2))) for bb in 1:nb]
@printf("\n band |  r(k)  | stat/σb | LPT/σb | dropB2/σb | b1±20%%/σb   (σb = fiducial band amplitude)\n")
for bb in 1:nb
  @printf("  %d   |  %+.2f  |  %.2f   |  %.2f  |   %.2f    |  %.2f\n",
    bb, rkf[bb], statb[bb]/fidb[bb], sLPT[bb]/fidb[bb], slin[bb]/fidb[bb], sb1[bb]/fidb[bb])
end
@printf("\nINTERPRETATION: on CONSTRAINED bands (r>0.5), compare stat vs systematics.\n")
println("DONE")
