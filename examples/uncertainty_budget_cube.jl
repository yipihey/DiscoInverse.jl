using DiscoInverse, Random, Statistics, Printf, LinearAlgebra, FFTW
const DI = DiscoInverse; T = Float64
c = fiducial_cosmology(Omega_m=0.3153, Omega_b=0.0493, h=0.6736, sigma8=0.8111, n_s=0.9649)  # Planck 2018
pk = linear_power_spectrum(c)
Rcube = anchor_cube_side(c)                     # cubic-cell side [Mpc/h] for 16e13 Msun
res = 96; L = res*Rcube                          # grid cell IS the cube:  Δq = L/res = Rcube
obs = [-1300.0, L/2, L/2]
mkgm(n) = galaxy_model(res, L, c, pk; R=Rcube, observer=obs, a_far=0.4, a_near=1.0, n_order=n, n_sub=1, rsd=false)
gm2 = mkgm(2); gm3 = mkgm(3)
W = ones(res,res,res); mask = ones(res,res,res); btrue=[1.8,0.5,0.3]; ntot=8.0*res^3; σb=[1e-4,1e-4,1e-4]; λf=1e-6
@printf("UBUDGET_CUBE: 16e13 Msun cubic cell — R=%.3f Mpc/h, res=%d, L=%.0f Mpc/h (Δq=Rcube), cubic filter\n", Rcube, res, L); flush(stdout)
recon(gm,b;counts,iters,ω0=nothing)=begin
    p=DI.InferenceProblem{T,typeof(gm)}(gm,W,mask,counts,ntot,b,σb,λf)
    w0 = ω0===nothing ? zeros(T,res,res,res) : ω0
    DI.lbfgs_optimize(p, w0, copy(b); iters=iters, m=8, fix_bias=true).ω
end
nb=12
kf=[(i-1)<res÷2 ? i-1 : i-1-res for i in 1:res]; kh=[i-1 for i in 1:res÷2+1]
kg=Float64[sqrt(kh[a]^2+kf[b]^2+kf[c]^2) for a in 1:res÷2+1,b in 1:res,c in 1:res]
edges=range(0,maximum(kg)+1e-6,length=nb+1); bnd(k)=clamp(searchsortedlast(edges,k),1,nb); bmask=[bnd.(kg).==bb for bb in 1:nb]
bandproj(δ,bb)=(F=rfft(δ); F[.!bmask[bb]].=0; irfft(F,res))
kmid=[0.5*(edges[bb]+edges[bb+1])*2π/L for bb in 1:nb]
allrows=[]
for real in 1:2
    ωtrue=randn(MersenneTwister(real),res,res,res)
    probt=DI.InferenceProblem{T,typeof(gm2)}(gm2,W,mask,zeros(res,res,res),ntot,btrue,σb,λf)
    prob=DI.inject_mock(probt, ωtrue, btrue; ntot=ntot, seed=100+real); nobs=prob.n_obs
    t0=time(); ωfid=recon(gm2,btrue;counts=nobs,iters=50); @printf("[real %d] fiducial %.0fs\n",real,time()-t0); flush(stdout)
    ω3=recon(gm3,btrue;counts=nobs,iters=15,ω0=copy(ωfid))
    ωlin=recon(gm2,[btrue[1],0,0];counts=nobs,iters=15,ω0=copy(ωfid))
    ωhi=recon(gm2,[btrue[1]*1.2,btrue[2],btrue[3]];counts=nobs,iters=15,ω0=copy(ωfid))
    ωlo=recon(gm2,[btrue[1]*0.8,btrue[2],btrue[3]];counts=nobs,iters=15,ω0=copy(ωfid))
    K=6; boot=Vector{Array{T,3}}()
    for k in 1:K
        rng=MersenneTwister(700+10*real+k); nbc=similar(nobs); for i in eachindex(nobs); nbc[i]=T(DI._rand_poisson(rng,max(nobs[i],0.0))); end
        push!(boot, recon(gm2,btrue;counts=nbc,iters=15,ω0=copy(ωfid)))
    end
    r0=DI.cross_spectrum_r(ωfid,ωtrue;nbins=nb,boxsize=L).r
    fidb=[sqrt(mean(bandproj(ωfid,bb).^2)) for bb in 1:nb]
    statb=[(S=std(cat([bandproj(boot[k],bb) for k in 1:K]...;dims=4);dims=4); sqrt(mean(S.^2)))/fidb[bb] for bb in 1:nb]
    sL=[sqrt(mean(bandproj(ω3.-ωfid,bb).^2))/fidb[bb] for bb in 1:nb]
    sl=[sqrt(mean(bandproj(ωlin.-ωfid,bb).^2))/fidb[bb] for bb in 1:nb]
    sb1=[sqrt(mean(bandproj((ωhi.-ωlo)./2,bb).^2))/fidb[bb] for bb in 1:nb]
    push!(allrows,(r0,statb,sL,sl,sb1)); @printf("[real %d] done\n",real); flush(stdout)
end
rkm=mean([a[1] for a in allrows]); stm=mean([a[2] for a in allrows]); Lm=mean([a[3] for a in allrows]); lm=mean([a[4] for a in allrows]); b1m=mean([a[5] for a in allrows])
println("\nBUDGET at fixed 16e13 cubic cell (mean of 2 realizations):")
for bb in 1:nb
  @printf("ROW %2d k=%.3f rk=%+.2f stat=%.2f LPT=%.2f dropB2=%.2f b1pm=%.2f\n",bb,kmid[bb],rkm[bb],stm[bb],Lm[bb],lm[bb],b1m[bb])
end
println("DONE")
