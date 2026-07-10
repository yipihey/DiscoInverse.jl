using DiscoInverse, NPZ, FFTW, Statistics, Printf
const DI=DiscoInverse; T=Float32
COARSE="/zpool/nvme/data/tabel_scratch/graphGP-cosmology/data/desi/desi_highz_recon_r384.npz"
d=npzread(COARSE); ωc=T.(d["omega"]); L=Float64(d["boxsize"]); resc=Int(d["res"])
@printf("coarse DESI high-z field: %d³, box=%.0f Mpc/h (Δq=%.1f)\n", resc, L, L/resc)
φc = angle.(rfft(ωc))                                       # recover constrained coarse phases (193,384,384)
res_box = 892                                               # Δq = 9556/892 = 10.71 ≈ the fixed 16e13 cube
t0=time(); ωb = DI.refine_phases(φc, res_box; seed=202, fixed_amplitude=true); tr=time()-t0
@printf("\nfine-realize: %d³ (Δq=%.2f Mpc/h) in the %.0f Mpc/h box  [%.1fs, no forward/AD]\n",
        res_box, L/res_box, L, tr)
@printf("  ω_box: std=%.4f  mean=%.1e  finite=%s  (unit-variance fixed-amplitude ⇒ exact fiducial P(k))\n",
        std(ωb), mean(ωb), all(isfinite, ωb))
# verify the DESI constraint is embedded EXACTLY at matching physical k
Fb = rfft(Float32.(ωb)); i = DI._embed_indices(resc, res_box)
φb = angle.(Fb[1:resc÷2+1, i, i])
dφ = mod.(φb .- φc .+ Float32(π), 2f0π) .- Float32(π)
@printf("  embedded coarse phases vs source: median|Δφ|=%.2e rad, frac<0.1rad=%.3f (→ constraint carried)\n",
        median(abs.(dφ)), mean(abs.(dφ).<0.1f0))
@printf("  constrained band: %d³ of %d³ modes = %.1f%% (large scales = DESI constraint; rest fresh, fixed P(k))\n",
        resc, res_box, 100*(resc/res_box)^3)
out="/zpool/nvme/data/tabel_scratch/graphGP-cosmology/data/desi/desi_highz_finecube_r892.f32"
open(io->write(io, Float32.(ωb)), out, "w")
@printf("saved %d³ fine cube (%.1f GB raw f32) → %s\nDONE\n", res_box, res_box^3*4/2^30, out)
