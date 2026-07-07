using DiscoInverse, CUDA, Random, Statistics, Printf, LinearAlgebra, FFTW, NPZ
const DI = DiscoInverse
T = Float32
cosmo = fiducial_cosmology(); pk = linear_power_spectrum(cosmo)
res = 96; L = 400f0; obs = Float32[-1300.0, L/2, L/2]
Rb = T(max(2*L/res, 6.0))
mkgm(n)=galaxy_model(res,L,cosmo,pk; R=Rb,observer=obs,a_far=0.4f0,a_near=1.0f0,n_order=n,n_sub=1,rsd=false,T=T)
gm2=mkgm(2); gm3=mkgm(3)
Wh=ones(T,res,res,res); maskh=ones(T,res,res,res); btrue=T[1.8,0.5,0.3]; ntot=T(10.0*res^3); σb=T[1e-4,1e-4,1e-4]; λf=T(1e-6)
probt=DI.InferenceProblem{T,typeof(gm2)}(gm2,Wh,maskh,zeros(T,res,res,res),ntot,btrue,σb,λf)
ωtrue=randn(MersenneTwister(1),T,res,res,res)
t0=time(); prob0=DI.inject_mock(probt,ωtrue,btrue;ntot=ntot,seed=101); nobs=prob0.n_obs
@printf("UBUDGET_GPU2 res=%d R=%.1f mock %.1fs\n",res,Rb,time()-t0); flush(stdout)
# PERF: build device gm + static arrays ONCE (was rebuilt every recon)
tg=time(); gmg2=DI.gpu(gm2); gmg3=DI.gpu(gm3); Wg=CuArray(Wh); maskg=CuArray(maskh); nobsg=CuArray(nobs)
@printf("gpu(gm) x2 + arrays built once: %.1fs\n",time()-tg); flush(stdout)
recon(gmg,bassume;counts=nobsg,iters=45,ω0=nothing)=begin
    p=DI.InferenceProblem{T,typeof(gmg)}(gmg,Wg,maskg,counts,ntot,bassume,σb,λf)
    w0=ω0===nothing ? CUDA.zeros(T,res,res,res) : ω0
    DI.lbfgs_optimize(p,w0,CuArray(bassume);iters=iters,m=8,fix_bias=true).ω
end
CUDA.@sync recon(gmg2,btrue;iters=2)   # warmup/compile
t0=time(); ωfidg=CUDA.@sync recon(gmg2,btrue;iters=60); @printf("fiducial STEADY: %.1fs (60it)\n",time()-t0); flush(stdout)
ωfid=Array(ωfidg)
t0=time()
ω3=Array(CUDA.@sync recon(gmg3,btrue;iters=20,ω0=copy(ωfidg)))
ωlin=Array(recon(gmg2,T[btrue[1],0,0];iters=20,ω0=copy(ωfidg)))
ωhi=Array(recon(gmg2,T[btrue[1]*1.2,btrue[2],btrue[3]];iters=20,ω0=copy(ωfidg)))
ωlo=Array(recon(gmg2,T[btrue[1]*0.8,btrue[2],btrue[3]];iters=20,ω0=copy(ωfidg)))
@printf("4 warm variants: %.1fs (%.1f each)\n",time()-t0,(time()-t0)/4); flush(stdout)
K=8; boot=Vector{Array{T,3}}(); t0=time()
for k in 1:K
    rng=MersenneTwister(700+k); nbc=similar(nobs)
    for i in eachindex(nobs); nbc[i]=T(DI._rand_poisson(rng,max(Float64(nobs[i]),0.0))); end
    push!(boot,Array(recon(gmg2,btrue;counts=CuArray(nbc),iters=20,ω0=copy(ωfidg))))
end
@printf("bootstrap K=%d: %.1fs (%.1f each)\n",K,time()-t0,(time()-t0)/K); flush(stdout)
nb=10
kf=[(i-1)<res÷2 ? i-1 : i-1-res for i in 1:res]; kh=[i-1 for i in 1:res÷2+1]
kg=Float32[sqrt(kh[a]^2+kf[b]^2+kf[c]^2) for a in 1:res÷2+1,b in 1:res,c in 1:res]
edges=range(0,maximum(kg)+1f-3,length=nb+1); band(k)=clamp(searchsortedlast(edges,k),1,nb); bmask=[band.(kg).==bb for bb in 1:nb]
bandproj(δ,bb)=(F=rfft(Float64.(δ)); F[.!bmask[bb]].=0.0+0.0im; irfft(F,res))
r0=DI.cross_spectrum_r(ωfid,ωtrue;nbins=nb,boxsize=Float64(L)).r
statb=[(S=std(cat([bandproj(boot[k],bb) for k in 1:K]...;dims=4);dims=4); sqrt(mean(S.^2))) for bb in 1:nb]
sysb(a)=[(d=bandproj(a.-ωfid,bb); sqrt(mean(d.^2))) for bb in 1:nb]
sL=sysb(ω3); sl=sysb(ωlin); sb1=[(d=bandproj((ωhi.-ωlo)./2,bb); sqrt(mean(d.^2))) for bb in 1:nb]
fidb=[(d=bandproj(ωfid,bb); sqrt(mean(d.^2))) for bb in 1:nb]
println("\nBANDTABLE band r(k) stat LPT dropB2 b1pm20 (frac of band amplitude)")
for bb in 1:nb
  @printf("ROW %2d  rk=%+.2f  stat=%.2f  LPT=%.2f  dropB2=%.2f  b1pm=%.2f\n", bb,r0[bb],statb[bb]/fidb[bb],sL[bb]/fidb[bb],sl[bb]/fidb[bb],sb1[bb]/fidb[bb])
end
npzwrite("/home/tabel/Projects/DiscoInverse.jl/scratch/ubudget_res96.npz",
  Dict("rk"=>r0,"stat"=>statb./fidb,"LPT"=>sL./fidb,"dropB2"=>sl./fidb,"b1pm"=>sb1./fidb))
println("DONE")
