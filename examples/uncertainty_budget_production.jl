using DiscoInverse, CUDA, Random, Statistics, Printf, LinearAlgebra, FFTW, NPZ
const DI = DiscoInverse
T = Float32
cosmo = fiducial_cosmology(); pk = linear_power_spectrum(cosmo)
res = 192; L = 768f0; obs = Float32[-1300.0, L/2, L/2]
Rb = T(2*L/res)                                    # 8 Mpc = 2 voxels (same physical regime as res-96)
mkgm(n)=galaxy_model(res,L,cosmo,pk; R=Rb,observer=obs,a_far=0.4f0,a_near=1.0f0,n_order=n,n_sub=1,rsd=false,T=T)
gm2=mkgm(2); gm3=mkgm(3)
Wh=ones(T,res,res,res); maskh=ones(T,res,res,res); btrue=T[1.8,0.5,0.3]; ntot=T(8.0*res^3); σb=T[1e-4,1e-4,1e-4]; λf=T(1e-6)
@printf("UBUDGET_FINAL res=%d L=%.0f R=%.1f ntot=%.2e\n",res,L,Rb,Float64(ntot)); flush(stdout)
tg=time(); gmg2=DI.gpu(gm2); gmg3=DI.gpu(gm3); Wg=CuArray(Wh); maskg=CuArray(maskh)
@printf("device gm x2 built once: %.1fs\n",time()-tg); flush(stdout)
probg(counts,b,gmg)=DI.InferenceProblem{T,typeof(gmg)}(gmg,Wg,maskg,counts,ntot,b,σb,λf)
recon(gmg,b;counts,iters,ω0=nothing)=begin
    w0=ω0===nothing ? CUDA.zeros(T,res,res,res) : ω0
    DI.lbfgs_optimize(probg(counts,b,gmg),w0,CuArray(b);iters=iters,m=8,fix_bias=true).ω
end
# k-band setup
nb=14
kf=[(i-1)<res÷2 ? i-1 : i-1-res for i in 1:res]; kh=[i-1 for i in 1:res÷2+1]
kg=Float32[sqrt(kh[a]^2+kf[b]^2+kf[c]^2) for a in 1:res÷2+1,b in 1:res,c in 1:res]
edges=range(0,maximum(kg)+1f-3,length=nb+1); bnd(k)=clamp(searchsortedlast(edges,k),1,nb); bmask=[bnd.(kg).==bb for bb in 1:nb]
bandproj(δ,bb)=(F=rfft(Float64.(δ)); F[.!bmask[bb]].=0.0+0.0im; irfft(F,res))
kmid=[0.5*(edges[bb]+edges[bb+1])*2π/Float64(L) for bb in 1:nb]
CUDA.@sync recon(gmg2,btrue;counts=CUDA.zeros(T,res,res,res).+ntot/res^3,iters=2)  # warmup
allrows=[]
for real in 1:2
    ωtrue=randn(MersenneTwister(real),T,res,res,res)
    λ=Array(DI.model_lambda(probg(CUDA.zeros(T,res,res,res),btrue,gmg2), CuArray(ωtrue), btrue; ntot=ntot))  # GPU inject
    rng=MersenneTwister(100+real); nobs=similar(λ); for i in eachindex(λ); nobs[i]=T(DI._rand_poisson(rng,max(Float64(λ[i]),0.0))); end
    nobsg=CuArray(nobs)
    t0=time(); ωfidg=CUDA.@sync recon(gmg2,btrue;counts=nobsg,iters=55); ωfid=Array(ωfidg)
    @printf("[real %d] fiducial %.1fs\n",real,time()-t0); flush(stdout)
    t0=time()
    ω3=Array(recon(gmg3,btrue;counts=nobsg,iters=15,ω0=copy(ωfidg)))
    ωlin=Array(recon(gmg2,T[btrue[1],0,0];counts=nobsg,iters=15,ω0=copy(ωfidg)))
    ωhi=Array(recon(gmg2,T[btrue[1]*1.2,btrue[2],btrue[3]];counts=nobsg,iters=15,ω0=copy(ωfidg)))
    ωlo=Array(recon(gmg2,T[btrue[1]*0.8,btrue[2],btrue[3]];counts=nobsg,iters=15,ω0=copy(ωfidg)))
    @printf("[real %d] 4 variants %.1fs\n",real,time()-t0); flush(stdout)
    K=6; boot=Vector{Array{T,3}}(); t0=time()
    for k in 1:K
        r2=MersenneTwister(700+10*real+k); nbc=similar(nobs); for i in eachindex(nobs); nbc[i]=T(DI._rand_poisson(r2,max(Float64(nobs[i]),0.0))); end
        push!(boot,Array(recon(gmg2,btrue;counts=CuArray(nbc),iters=15,ω0=copy(ωfidg))))
    end
    @printf("[real %d] bootstrap K=%d %.1fs\n",real,K,time()-t0); flush(stdout)
    r0=DI.cross_spectrum_r(ωfid,ωtrue;nbins=nb,boxsize=Float64(L)).r
    fidb=[sqrt(mean(bandproj(ωfid,bb).^2)) for bb in 1:nb]
    statb=[(S=std(cat([bandproj(boot[k],bb) for k in 1:K]...;dims=4);dims=4); sqrt(mean(S.^2)))/fidb[bb] for bb in 1:nb]
    sL=[sqrt(mean(bandproj(ω3.-ωfid,bb).^2))/fidb[bb] for bb in 1:nb]
    sl=[sqrt(mean(bandproj(ωlin.-ωfid,bb).^2))/fidb[bb] for bb in 1:nb]
    sb1=[sqrt(mean(bandproj((ωhi.-ωlo)./2,bb).^2))/fidb[bb] for bb in 1:nb]
    push!(allrows,(r0,statb,sL,sl,sb1))
    npzwrite("/home/tabel/Projects/DiscoInverse.jl/scratch/ubudget_res192.npz",
        Dict("kmid"=>kmid,"nreal"=>real,
             "rk"=>hcat([a[1] for a in allrows]...),"stat"=>hcat([a[2] for a in allrows]...),
             "LPT"=>hcat([a[3] for a in allrows]...),"dropB2"=>hcat([a[4] for a in allrows]...),
             "b1pm"=>hcat([a[5] for a in allrows]...)))
    @printf("[real %d] SAVED\n",real); flush(stdout)
end
# report mean over realizations
rkm=mean([a[1] for a in allrows]); stm=mean([a[2] for a in allrows]); Lm=mean([a[3] for a in allrows])
lm=mean([a[4] for a in allrows]); b1m=mean([a[5] for a in allrows])
println("\nFINAL res=192 budget (mean over $(length(allrows)) realizations):")
for bb in 1:nb
  @printf("ROW %2d k=%.3f rk=%+.2f stat=%.2f LPT=%.2f dropB2=%.2f b1pm=%.2f\n",bb,kmid[bb],rkm[bb],stm[bb],Lm[bb],lm[bb],b1m[bb])
end
println("DONE")
