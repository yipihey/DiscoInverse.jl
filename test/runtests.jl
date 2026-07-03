using Test
using DiscoInverse
using DiscoDJNative: nodal_density
using Random: MersenneTwister
using Statistics: mean, std, median

ad_ok = false
try
    @eval using Zygote, FiniteDifferences
    global ad_ok = true
catch
    global ad_ok = false
end

@testset "DiscoInverse" begin

    @testset "2nd-order Lagrangian bias (δ_L, s², w)" begin
        res = 16; L = 500.0
        c  = Cosmology("Planck18EEBAOSN"); pk = linear_power_spectrum(c)
        op  = ic_operator(res, L, pk; T=Float64)
        ops = bias_operators(res, L, 16.0)            # R = 16 Mpc/h Lagrangian smoothing
        ω   = randn(MersenneTwister(0), res, res, res)
        fphi = white_noise_to_fphi(op, ω)
        δL, s2 = bias_fields(fphi, ops)

        @test size(δL) == (res, res, res) && size(s2) == (res, res, res)
        @test abs(mean(δL)) < 1e-9            # δ_L has zero mean (DC removed)
        @test 0.05 < std(δL) < 5.0            # sensible linear-density amplitude
        @test all(s2 .>= 0)                   # tidal s² is nonnegative

        σ², s2m = bias_moments(δL, s2)               # fixed reference moments
        w = bias_weight(δL, s2, σ², s2m; b1=1.5, b2=0.5, bs2=0.2)
        @test size(w) == (res, res, res)
        @test isapprox(mean(w), 1.0; atol=1e-6)      # mean preserved (moments from this field)

        if ad_ok
            forward(x, b) = (f = white_noise_to_fphi(op, x);
                             (d, s) = bias_fields(f, ops);
                             bias_weight(d, s, σ², s2m; b1=b[1], b2=b[2], bs2=b[3]))
            loss(x, b) = sum(abs2, forward(x, b))
            b0 = [1.5, 0.5, 0.2]
            gω, gb = Zygote.gradient(loss, ω, b0)
            fdm = central_fdm(5, 1)
            for idx in ((2, 3, 4), (10, 10, 10))
                gfd = FiniteDifferences.grad(fdm, t -> (u = copy(ω); u[idx...] = t; loss(u, b0)), ω[idx...])[1]
                @test isapprox(gω[idx...], gfd; rtol=1e-5)
            end
            for k in 1:3
                gfd = FiniteDifferences.grad(fdm, t -> (bb = copy(b0); bb[k] = t; loss(ω, bb)), b0[k])[1]
                @test isapprox(gb[k], gfd; rtol=1e-5)
            end
        end
    end

    @testset "Galaxy field forward (ω,b → n_g on lightcone)" begin
        res = 8; L = 300.0
        c  = Cosmology("Planck18EEBAOSN"); pk = linear_power_spectrum(c)
        gm = galaxy_model(res, L, c, pk; R=20.0, observer=[-1400.0, L/2, L/2],
                          a_far=0.4, a_near=1.0, n_order=3, n_sub=1, rsd=true)
        ω = randn(MersenneTwister(2), res, res, res); b = [1.8, 0.4, 0.2]
        ng = galaxy_density(gm, ω, b)
        @test size(ng) == (res, res, res)
        @test isapprox(mean(ng), 1.0; atol=0.1)        # mass-conserving, w mean ≈ 1
        if ad_ok
            loss(w, bb) = sum(abs2, galaxy_density(gm, w, bb) .- 1.0)
            gω, gb = Zygote.gradient(loss, ω, b)
            fdm = central_fdm(5, 1)
            gfd = FiniteDifferences.grad(fdm, t -> (u = copy(ω); u[2,3,4] = t; loss(u, b)), ω[2,3,4])[1]
            @test isapprox(gω[2,3,4], gfd; rtol=1e-3)   # ∂/∂ω through crossing + sheet vs FD
            for k in 1:3
                gfd = FiniteDifferences.grad(fdm, t -> (bb = copy(b); bb[k] = t; loss(ω, bb)), b[k])[1]
                @test isapprox(gb[k], gfd; rtol=1e-5)
            end
        end

        # 1LPT (Zel'dovich) and 2LPT also run and are differentiable (fast paths).
        for nord in (1, 2)
            gmk = galaxy_model(res, L, c, pk; R=20.0, observer=[-1400.0, L/2, L/2],
                               a_far=0.4, a_near=1.0, n_order=nord, n_sub=1, rsd=true)
            ngk = galaxy_density(gmk, ω, b)
            @test size(ngk) == (res, res, res)
            @test isapprox(mean(ngk), 1.0; atol=0.1)
            if ad_ok
                lk(w) = sum(abs2, galaxy_density(gmk, w, b))
                gk = Zygote.gradient(lk, ω)[1]
                gfd = FiniteDifferences.grad(central_fdm(5, 1),
                          t -> (u = copy(ω); u[2,3,4] = t; lk(u)), ω[2,3,4])[1]
                @test isapprox(gk[2,3,4], gfd; rtol=1e-3)
            end
        end
    end

    @testset "Geometry + window (synthetic footprint)" begin
        cosmo = fiducial_cosmology()
        rng = MersenneTwister(0); N = 5000
        ra  = rand(rng, N) .* 20 .+ 100; dec = rand(rng, N) .* 20 .- 5; z = rand(rng, N) .* 0.15 .+ 0.45
        rnd = (ra=ra, dec=dec, z=z)
        geom = box_geometry(rnd, cosmo; res=32, pad_frac=0.15)
        @test geom.boxsize > 0
        @test isapprox(1 / geom.a_near - 1, 0.45; atol=0.01) && isapprox(1 / geom.a_far - 1, 0.60; atol=0.01)
        cart = radec_z_to_cartesian(ra, dec, z, cosmo)
        @test size(cart) == (N, 3)
        xb = embed(geom, cart)
        @test all(0 .<= xb .<= geom.boxsize)            # all galaxies inside the box
        W = survey_window(geom, rnd)
        @test isapprox(mean(W[W .> 0]), 1.0; atol=1e-6)  # normalized over footprint
        @test 0 < sum(W .> 0) < length(W)                # partial footprint
        ng = bin_galaxies(geom, ra[1:1000], dec[1:1000], z[1:1000])
        @test isapprox(sum(ng), 1000.0; rtol=1e-10)      # binning conserves count
    end

    @testset "Injection-recovery (field-level IC)" begin
        cosmo = fiducial_cosmology(); pk = linear_power_spectrum(cosmo)
        res = 12; L = 400.0; obs = [-1300.0, L/2, L/2]
        gm = galaxy_model(res, L, cosmo, pk; R=40.0, observer=obs, a_far=0.4, a_near=1.0,
                          n_order=3, n_sub=1, rsd=false)
        W = ones(res, res, res); mask = ones(res, res, res)
        ntot = 20.0 * res^3                              # ample signal (mean 20/cell)
        probt = InferenceProblem{Float64, typeof(gm)}(gm, W, mask, zeros(res, res, res),
                    ntot, [1.8, 0.0, 0.0], [5.0, 5.0, 5.0], 1e-6)
        ωtrue = randn(MersenneTwister(42), res, res, res); btrue = [1.8, 0.4, 0.2]
        prob = inject_mock(probt, ωtrue, btrue; ntot=ntot, seed=7)
        if ad_ok   # map_optimize needs Zygote (Optim is a hard dep)
            ω0 = zeros(res, res, res); b0 = [1.5, 0.0, 0.0]
            L0 = loss(prob, ω0, b0)
            rec = map_optimize(prob, ω0, b0; iters=60)
            @test rec.loss < L0                          # optimizer reduced the loss
            # large-scale initial-condition modes recovered (the BORG diagnostic);
            # high-k modes are data-uninformative so r(k) declines there.
            rk = cross_spectrum_r(rec.ω, ωtrue; nbins=5, boxsize=L)
            @test rk.r[1] > 0.8
            @test rk.r[2] > 0.5

            # Adam driver (the device-resident path the GPU optimizer uses) also reduces
            # the loss and recovers the large-scale modes.
            ra = adam_optimize(prob, ω0, b0; iters=80, lr=0.05)
            @test ra.loss < L0
            @test ra.history[end] < ra.history[1]
            @test cross_spectrum_r(ra.ω, ωtrue; nbins=5, boxsize=L).r[1] > 0.7
        end
    end

    @testset "Over-dispersed PROV likelihood" begin
        res = 8; L = 300.0
        c  = Cosmology("Planck18EEBAOSN"); pk = linear_power_spectrum(c)
        gm = galaxy_model(res, L, c, pk; R=20.0, observer=[-1400.0, L/2, L/2],
                          a_far=0.4, a_near=1.0, n_order=1, rsd=false)
        ω = randn(MersenneTwister(3), res, res, res); b = [1.5, 0.0, 0.0]
        W = ones(res, res, res); mask = ones(res, res, res)
        nobs = fill(5.0, res, res, res); dvar = fill(2.0, res, res, res)   # synthetic Var_miss
        pod = InferenceProblem{Float64, typeof(gm)}(gm, W, mask, nobs, sum(nobs), b, [5.0,5,5], 1e-6, dvar)
        ppo = InferenceProblem{Float64, typeof(gm)}(gm, W, mask, nobs, sum(nobs), b, [5.0,5,5], 1e-6)
        Lod = loss(pod, ω, b)
        @test isfinite(Lod) && Lod != loss(ppo, ω, b)     # over-dispersed branch active + distinct
        if ad_ok
            g = Zygote.gradient(w -> loss(pod, w, b), ω)[1]
            gfd = FiniteDifferences.grad(central_fdm(5,1), t -> (u=copy(ω); u[2,3,4]=t; loss(pod,u,b)), ω[2,3,4])[1]
            @test isapprox(g[2,3,4], gfd; rtol=1e-3)       # over-dispersed NLL is differentiable
        end
    end

    @testset "Joint (ω,b) HMC sampler" begin
        res = 10; L = 350.0
        c  = Cosmology("Planck18EEBAOSN"); pk = linear_power_spectrum(c)
        gm = galaxy_model(res, L, c, pk; R=40.0, observer=[-1300.0, L/2, L/2],
                          a_far=0.4, a_near=1.0, n_order=1, rsd=false)
        W = ones(res, res, res); mask = ones(res, res, res); ntot = 15.0 * res^3
        prob0 = InferenceProblem{Float64, typeof(gm)}(gm, W, mask, zeros(res,res,res), ntot, [1.0,0,0], [1.0,1,1], 1e-6)
        mock = inject_mock(prob0, randn(MersenneTwister(0), res, res, res), [1.2, 0.3, 0.1]; ntot=ntot, seed=1)
        if ad_ok
            s = hmc_sample(mock, zeros(res,res,res), [1.0,0,0]; nsamples=40, nwarmup=60, nleap=12, seed=0)
            @test size(s.b_samples) == (40, 3)
            @test 0.3 < s.accept < 1.0                       # leapfrog + dual-averaging healthy
            @test all(isfinite, s.b_mean) && all(s.b_std .< 5) # bounded posterior (NOT the MAP runaway)
            @test size(s.ω_mean) == (res, res, res)
        end
    end

    @testset "NUTS + convergence diagnostics" begin
        # R̂ / ESS on synthetic chains (deterministic)
        M = reduce(vcat, [randn(MersenneTwister(c), 300)' for c in 1:4])        # iid
        @test isapprox(rhat(M), 1.0; atol=0.05) && ess(M) > 700
        S = reduce(vcat, [(randn(MersenneTwister(c), 300) .+ 5c)' for c in 1:4]) # offset means
        @test rhat(S) > 2.0                                                     # flags non-convergence

        c  = Cosmology("Planck18EEBAOSN"); pk = linear_power_spectrum(c)
        res = 8; L = 300.0
        gm = galaxy_model(res, L, c, pk; R=40.0, observer=[-1300.0, L/2, L/2],
                          a_far=0.4, a_near=1.0, n_order=1, rsd=false)
        W = ones(res,res,res); mask = ones(res,res,res); ntot = 15.0 * res^3
        p0 = InferenceProblem{Float64, typeof(gm)}(gm, W, mask, zeros(res,res,res), ntot, [1.0,0,0], [1.0,1,1], 1e-6)
        mock = inject_mock(p0, randn(MersenneTwister(0), res, res, res), [1.2, 0.3, 0.1]; ntot=ntot, seed=1)
        if ad_ok
            s = nuts_sample(mock, zeros(res,res,res), [1.0,0,0]; nsamples=30, nwarmup=40, max_depth=6, seed=0)
            @test size(s.b_samples) == (30, 3)
            @test 0.4 < s.accept < 1.0                       # auto-tuned acceptance near target
            @test all(isfinite, s.b_mean) && all(s.b_std .< 5)
            @test all(1 .<= s.depths .<= 6)                  # tree depth within bound
        end
    end

    @testset "Grid-free sheet point-process likelihood (P4)" begin
        c  = fiducial_cosmology(); pk = linear_power_spectrum(c)
        res = 8; L = 300.0; obs = [-1300.0, L/2, L/2]
        gm = galaxy_model(res, L, c, pk; R=40.0, observer=obs, a_far=0.4, a_near=1.0, n_order=1, rsd=false)
        ωstar = randn(MersenneTwister(1), res,res,res); bstar = [1.8, 0.4, 0.2]
        pts = inject_mock_sheet(gm, ωstar, bstar, 25.0*res^3; seed=2)
        @test size(pts,1) > 1000
        prob = sheet_problem(gm, pts; b0=[1.5,0,0], σb=[5.,5,5])
        @test isfinite(loss(prob, zeros(res,res,res), [1.5,0,0]))
        @test loss(prob, ωstar, bstar) < loss(prob, zeros(res,res,res), bstar)   # well-posed: truth lower
        if ad_ok
            fdm = central_fdm(5,1)
            ω0 = zeros(res,res,res)                          # ∂/∂ω clean at the undeformed sheet
            gω = Zygote.gradient(w -> loss(prob, w, bstar), ω0)[1]
            for idx in ((3,4,5), (6,2,7))
                fd = FiniteDifferences.grad(fdm, t->(u=copy(ω0); u[idx...]=t; loss(prob,u,bstar)), ω0[idx...])[1]
                @test isapprox(gω[idx...], fd; rtol=1e-3)
            end
            gb = Zygote.gradient(b -> loss(prob, ωstar, b), bstar)[1]   # b only scales w_T (smooth)
            for k in 2:3
                fd = FiniteDifferences.grad(fdm, t->(bb=copy(bstar); bb[k]=t; loss(prob,ωstar,bb)), bstar[k])[1]
                @test isapprox(gb[k], fd; rtol=1e-3)
            end
        end
    end

    @testset "C⁰ density + F32 mixed-precision NUTS path (P6)" begin
        c  = fiducial_cosmology(); pk = linear_power_spectrum(c)
        res = 8; L = 300.0; obs = [-1300.0, L/2, L/2]
        mk(T) = galaxy_model(res, L, c, pk; R=40.0, observer=obs, a_far=0.4, a_near=1.0, n_order=1, rsd=false, T=T)
        gm = mk(Float64)
        ωstar = randn(MersenneTwister(1), res,res,res); bstar = [1.8, 0.4, 0.2]
        pts = inject_mock_sheet(gm, ωstar, bstar, 25.0*res^3; seed=2)
        prob = sheet_problem(gm, pts; b0=[1.5,0,0], σb=[5.,5,5], c0=true)
        @test loss(prob, ωstar, bstar) < loss(prob, zeros(res,res,res), bstar)   # C⁰ well-posed
        if ad_ok
            fdm = central_fdm(5,1); ω0 = zeros(res,res,res)
            gω = Zygote.gradient(w -> loss(prob, w, bstar), ω0)[1]                # C⁰ loss FD (nodal+interp path)
            for idx in ((3,4,5), (6,2,7))
                fd = FiniteDifferences.grad(fdm, t->(u=copy(ω0); u[idx...]=t; loss(prob,u,bstar)), ω0[idx...])[1]
                @test isapprox(gω[idx...], fd; rtol=1e-3)
            end
        end
        # prior VALUE accumulated in F64 even for an F32 field (the NUTS ΔH-critical bit)
        ωf = Float32.(ωstar)
        @test isapprox(DiscoInverse.gaussian_prior(ωf), DiscoInverse.gaussian_prior(Float64.(ωf)); rtol=1e-9)
        # mixed precision: F32 model + F64 leapfrog state → gradient returned in F64, tracks all-F64
        if ad_ok
            prob32 = sheet_problem(mk(Float32), Float32.(pts); b0=[1.5,0,0], σb=[5.,5,5], c0=true)
            _, gω32, gb32 = DiscoInverse._loss_grad(prob32, Float64.(ωstar), Float64.(bstar))
            @test eltype(gω32) === Float64 && eltype(gb32) === Float64
            g64 = Zygote.gradient(w -> loss(prob, w, bstar), ωstar)[1]
            cosθ = sum(vec(gω32) .* vec(g64)) / (sqrt(sum(abs2,gω32)) * sqrt(sum(abs2,g64)))
            @test cosθ > 0.999
        end
    end

    @testset "Fixed-amplitude phase field (Angulo–Pontzen, Stage-1)" begin
        res = 8
        φ  = 2π .* rand(MersenneTwister(1), res÷2+1, res, res)
        ω  = phase_field(φ)
        @test size(ω) == (res, res, res)
        @test isapprox(std(ω), 1.0; rtol=1e-6)                       # unit variance (per-call normalized)
        φ2 = 2π .* rand(MersenneTwister(2), res÷2+1, res, res)        # ANY phases → unit variance
        @test isapprox(std(phase_field(φ2)), 1.0; rtol=1e-6)
        if ad_ok                                                     # differentiable w.r.t. the phases
            g = Zygote.gradient(φv -> sum(phase_field(φv).^3), φ)[1]
            @test size(g) == size(φ) && all(isfinite, g)
        end
    end

    @testset "PROV per-galaxy weights (P7)" begin
        cat = EchoesCatalog([10.0,11,12,13], [0.0,1,2,3], [0.5,0.5,0.5,0.5], Int8[0,1,2,3])
        u = prov_weights(cat; soft=0.4)
        @test u == [1.0, 0.4, 0.4, 0.4]                 # PROV=0 observed (hard); PROV≥1 soft
        @test eltype(u) == Float64
    end

    @testset "Redshift-space distortions (sheet, P7)" begin
        c  = fiducial_cosmology(); pk = linear_power_spectrum(c)
        res = 8; L = 300.0; obs = [-1300.0, L/2, L/2]
        gmR = galaxy_model(res, L, c, pk; R=40.0, observer=obs, a_far=0.4, a_near=1.0, n_order=1, rsd=false)
        gmS = galaxy_model(res, L, c, pk; R=40.0, observer=obs, a_far=0.4, a_near=1.0, n_order=1, rsd=true)
        ωstar = randn(MersenneTwister(1), res,res,res); bstar = [1.5, 0.0, 0.0]
        pts = inject_mock_sheet(gmR, ωstar, bstar, 25.0*res^3; seed=2)
        prob = sheet_problem(gmR, pts; b0=[1.5,0,0], σb=[5.,5,5])
        ρr, _ = galaxy_density_sheet_c0(gmR, ωstar, bstar, prob.pts, prob.cl)
        ρs, _ = galaxy_density_sheet_c0(gmS, ωstar, bstar, prob.pts, prob.cl)
        @test !isapprox(ρs, ρr; rtol=1e-3)             # RSD shifts the sheet ⇒ LOS density anisotropy
        probS = sheet_problem(gmS, pts; b0=[1.5,0,0], σb=[5.,5,5])
        @test isfinite(loss(probS, zeros(res,res,res), [1.5,0,0]))
        if ad_ok                                        # the v_r path is differentiable w.r.t. ω
            fdm = central_fdm(5,1); ω0 = zeros(res,res,res)
            g = Zygote.gradient(w -> loss(probS, w, [1.5,0,0]), ω0)[1]
            for idx in ((3,4,5),(6,2,7))
                fd = FiniteDifferences.grad(fdm, t->(u=copy(ω0);u[idx...]=t;loss(probS,u,[1.5,0,0])), ω0[idx...])[1]
                @test isapprox(g[idx...], fd; rtol=1e-3)
            end
        end
    end

    @testset "Survey window in sheet normalization (P7)" begin
        c  = fiducial_cosmology(); pk = linear_power_spectrum(c)
        res = 8; L = 300.0; obs = [-1300.0, L/2, L/2]
        gm = galaxy_model(res, L, c, pk; R=40.0, observer=obs, a_far=0.4, a_near=1.0, n_order=1, rsd=false)
        ωstar = randn(MersenneTwister(1), res,res,res); bstar = [1.5, 0.0, 0.0]
        pts = inject_mock_sheet(gm, ωstar, bstar, 25.0*res^3; seed=2)
        win = zeros(res,res,res); win[2:6, 2:6, 2:6] .= 1.0            # a sub-box footprint
        p0 = sheet_problem(gm, pts; b0=[1.5,0,0], σb=[5.,5,5])           # no window
        p1 = sheet_problem(gm, pts; b0=[1.5,0,0], σb=[5.,5,5], window=ones(res,res,res))
        pW = sheet_problem(gm, pts; b0=[1.5,0,0], σb=[5.,5,5], window=win)
        ω0 = zeros(res,res,res)
        @test loss(p1, ω0, [1.5,0,0]) ≈ loss(p0, ω0, [1.5,0,0])         # window=ones ≡ no window
        @test isfinite(loss(pW, ω0, [1.5,0,0]))
        Z0 = galaxy_density_sheet_c0(gm, ω0, [1.5,0,0], pW.pts, pW.cl)[2]
        ZW = galaxy_density_sheet_c0(gm, ω0, [1.5,0,0], pW.pts, pW.cl; window=win)[2]
        @test ZW < Z0                                                   # window shrinks the normalization
        if ad_ok
            g = Zygote.gradient(w -> loss(pW, w, [1.5,0,0]), ω0)[1]
            @test all(isfinite, g)
        end
    end

    @testset "Quaia field-level redshift reconstruction (χ-param)" begin
        c = fiducial_cosmology(); pk = linear_power_spectrum(c)

        # redshift_prior: value matches the inline quadratic; gradient is the analytic (χ−χ_obs)/σ²
        χo = collect(1000.0:10.0:1200.0); σ = fill(50.0, length(χo))
        χp = χo .+ 5 .* randn(MersenneTwister(0), length(χo))
        @test redshift_prior(χp, χo, σ) ≈ 0.5 * sum(((χo .- χp) ./ σ) .^ 2)
        if ad_ok
            g = Zygote.gradient(cc -> redshift_prior(cc, χo, σ), χp)[1]
            @test isapprox(g, (χp .- χo) ./ σ .^ 2; rtol=1e-10)        # custom rrule == analytic
            fdm = central_fdm(5, 1)
            for i in (2, 11, 20)
                fd = FiniteDifferences.grad(fdm, t -> (u = copy(χp); u[i] = t; redshift_prior(u, χo, σ)), χp[i])[1]
                @test isapprox(g[i], fd; rtol=1e-6)
            end
        end

        # mock all-sky catalog → QuaiaProblem (box sized from the randoms, observer at centre)
        rng = MersenneTwister(7); N = 80
        ra  = 360 .* rand(rng, N); dec = rad2deg.(asin.(2 .* rand(rng, N) .- 1)); z = 0.3 .+ 0.5 .* rand(rng, N)
        rnd = (ra = 360 .* rand(rng, 4N), dec = rad2deg.(asin.(2 .* rand(rng, 4N) .- 1)), z = 0.3 .+ 0.5 .* rand(rng, 4N))
        res = 8; geom = box_geometry(rnd, c; res=res, pad_frac=0.2); L = geom.boxsize
        W   = survey_window(geom, rnd)
        gm  = galaxy_model(res, L, c, pk; R=max(2L/res, 80.0), observer=geom.observer,
                           a_far=geom.a_far, a_near=geom.a_near, n_order=1, rsd=false)
        prob = quaia_problem(QuaiaCatalog(ra, dec, z, fill(0.05, N)), geom, gm, W; b1=2.0)
        @test length(prob) == N
        @test all(prob.σχ .> 0)                                          # χ(z+σ_z) > χ(z)
        @test all(isapprox.(vec(sum(prob.nhat .^ 2; dims=2)), 1.0; atol=1e-10))   # unit sky directions

        # ∂L/∂χ — the new query-point path (quasar positions as free parameters)
        if ad_ok
            ω = randn(MersenneTwister(1), res, res, res)
            xg, wg = DiscoInverse._sheet_inputs(gm, ω, [2.0, 0, 0])
            ρv, _  = nodal_density(xg, DiscoInverse._apply_window(wg, W), res, L)
            dx = L / res; χ0 = copy(prob.χ_obs)
            f  = cc -> DiscoInverse._quaia_chi_loss(cc, xg, ρv, prob.nhat, prob.shift, prob.χ_obs, prob.σχ, prob.u, dx, res)
            g  = Zygote.gradient(f, χ0)[1]
            @test all(isfinite, g)
            rels = Float64[]
            for i in (5, 30, 60)
                ep = zeros(N); ep[i] = 1.0
                fd = (f(χ0 .+ ep) - f(χ0 .- ep)) / 2
                push!(rels, abs(fd - g[i]) / max(abs(fd), 1e-8))
            end
            @test median(rels) < 1e-2
        end

        # the χ-MAP machinery pulls an offset back toward the photo-z centre (prior alone, deterministic —
        # the FIELD pull magnitude is structure-dependent and validated on real data, not this coarse mock)
        χoff = prob.χ_obs .+ 40.0
        fp = cc -> redshift_prior(cc, prob.χ_obs, prob.σχ)
        cg, _, _ = DiscoInverse._lbfgs_generic(fp, cc -> Zygote.gradient(fp, cc)[1], copy(χoff); iters=20)
        @test mean(abs.(cg .- prob.χ_obs)) < mean(abs.(χoff .- prob.χ_obs))

        # the alternating-MAP driver runs end-to-end on the host → finite, valid zero-error catalog
        r = reconstruct_quaia(prob, 101; device=identity, rounds=1, phase_iters=5, chi_iters=5)
        @test length(r.χ) == N && length(r.z) == N && size(r.ω) == (res, res, res)
        @test all(isfinite, r.χ) && all(isfinite, r.z)
        @test all(zz -> 0 < zz < 5, r.z)                                # χ→z lands in the table range
        @test size(r.φ) == (res ÷ 2 + 1, res, res)                      # fixed-amplitude phases exposed
    end

    @testset "Quaia-constrained periodic IC box (coarse→fine)" begin
        c = fiducial_cosmology()
        rng = MersenneTwister(5); N = 40
        ra  = 360 .* rand(rng, N); dec = rad2deg.(asin.(2 .* rand(rng, N) .- 1)); z = 0.3 .+ 0.4 .* rand(rng, N)
        rnd = (ra = 360 .* rand(rng, 4N), dec = rad2deg.(asin.(2 .* rand(rng, 4N) .- 1)), z = 0.3 .+ 0.4 .* rand(rng, 4N))
        cart = radec_z_to_cartesian(rnd.ra, rnd.dec, rnd.z, c)
        extent = maximum(vec(maximum(cart; dims=1)) .- vec(minimum(cart; dims=1)))

        # explicit box length ≥ survey extent, survey centered; rejects too-small boxes
        g_auto = box_geometry(rnd, c; res=8)
        g_big  = box_geometry(rnd, c; res=8, boxsize=1.5 * extent)
        @test g_big.boxsize ≈ 1.5 * extent
        @test g_big.boxsize > g_auto.boxsize
        @test_throws ErrorException box_geometry(rnd, c; res=8, boxsize=0.5 * extent)

        # rfft coarse→fine index map (numpy fftfreq order): 4→8 sends freqs [0,1,-2,-1] to [1,2,7,8]
        @test DiscoInverse._embed_indices(4, 8) == [1, 2, 7, 8]

        # refine_phases: identity when res_box==res_constrain (whole grid embedded); finite + right
        # shape when finer (the box is built with NO forward at res_box — the decoupling)
        Nc = 8; φc = 2π .* rand(MersenneTwister(3), Nc ÷ 2 + 1, Nc, Nc)
        @test refine_phases(φc, Nc; seed=0) ≈ phase_field(φc)
        ωb = refine_phases(φc, 16; seed=1)
        @test size(ωb) == (16, 16, 16) && all(isfinite, ωb)

        # end-to-end driver: constrain at 8³, realize at 16³
        cat = QuaiaCatalog(ra, dec, z, fill(0.05, N))
        box = constrained_ic_box(cat, rnd, c; L_box=1.5 * extent, res_constrain=8, res_box=16,
                                 b1=2.0, seed=101, rounds=1, phase_iters=4, chi_iters=4)
        @test size(box.ω_box) == (16, 16, 16) && all(isfinite, box.ω_box)
        @test box.manifest["res_box"] == 16 && box.manifest["res_constrain"] == 8
        @test box.manifest["boxsize"] ≈ 1.5 * extent
        @test box.manifest["constrained_radius"] > 0
        @test size(box.φ_coarse) == (8 ÷ 2 + 1, 8, 8)                   # forward ran only at res_constrain
    end

    @testset "Photo-z calibration + radial-posterior ensemble" begin
        rng = MersenneTwister(3); N = 20000
        ztrue = 0.8 .+ 2.7 .* rand(rng, N)
        iscat = rand(rng, N) .< 0.2                                     # 20% catastrophic outliers
        x = ifelse.(iscat, 0.15 .* randn(rng, N), 0.006 .* randn(rng, N))
        zobs = ztrue .+ x .* (1 .+ ztrue)
        m = calibrate_photoz(zobs, ztrue)
        @test 0.6 < m.w[1] < 0.95                                        # core weight ~0.8 recovered
        @test m.σ[1] < 0.02 && m.σ[2] > 0.05                             # tight core + broad outlier
        # sampled posterior is calibrated against the (held-out) truth
        ens = radial_posterior_ensemble(zobs, m; K=200, seed=1)
        cp = coverage_pit(ens, ztrue)
        i68 = argmin(abs.(cp.levels .- 0.68)); i90 = argmin(abs.(cp.levels .- 0.90))
        @test abs(cp.coverage[i68] - 0.68) < 0.08
        @test abs(cp.coverage[i90] - 0.90) < 0.08
        # a naive over-confident ensemble (draws hugging z_obs) badly under-covers — the failure we fixed
        narrow = zobs .+ 0.001 .* randn(rng, N, 50)
        @test coverage_pit(narrow, ztrue).coverage[i68] < 0.4
        # spec-z fold-in pins those objects
        mask = falses(N); mask[1:1000] .= true
        ens2 = radial_posterior_ensemble(zobs, m; z_spec=ztrue, spec_mask=mask, K=50, seed=2)
        @test all(std(@view ens2[i, :]) < 0.01 for i in 1:1000)
    end

    @testset "Multi-tracer joint field reconstruction" begin
        c = fiducial_cosmology(); pk = linear_power_spectrum(c)
        res = 8; L = 300.0; obs = [-1300.0, L/2, L/2]
        gm = galaxy_model(res, L, c, pk; R=40.0, observer=obs, a_far=0.4, a_near=1.0, n_order=1, rsd=false)
        ωstar = randn(MersenneTwister(1), res, res, res)
        pts1 = inject_mock_sheet(gm, ωstar, [2.0, 0, 0], 20.0 * res^3; seed=2)
        pts2 = inject_mock_sheet(gm, ωstar, [1.4, 0, 0], 20.0 * res^3; seed=3)
        W = ones(res, res, res)
        mtp = multitracer_problem(gm, [tracer(gm, pts1; b1=2.0, window=W),
                                       tracer(gm, pts2; b1=1.4, window=W)])

        # refactor sanity: _sheet_inputs == _sheet_geometry + _sheet_weight
        xg, wg = DiscoInverse._sheet_inputs(gm, ωstar, [2.0, 0, 0])
        xg2, δL, s2 = DiscoInverse._sheet_geometry(gm, ωstar)
        @test xg ≈ xg2 && wg ≈ DiscoInverse._sheet_weight(gm, δL, s2, [2.0, 0, 0])

        φ0 = 2π .* rand(MersenneTwister(5), res ÷ 2 + 1, res, res)
        @test isfinite(multitracer_phase_loss(mtp, φ0))
        if ad_ok
            g = Zygote.gradient(φ -> multitracer_phase_loss(mtp, φ), φ0)[1]
            @test all(isfinite, g) && any(abs.(g) .> 0)
        end
        # the joint reconstruction runs and improves the multi-tracer constraint
        l0 = multitracer_phase_loss(mtp, φ0)
        r = reconstruct_joint_field(mtp, 5; iters=10)
        @test size(r.ω) == (res, res, res) && all(isfinite, r.ω)
        @test size(r.φ) == (res ÷ 2 + 1, res, res)
        @test multitracer_phase_loss(mtp, r.φ) < l0
    end

    @testset "CMB-lensing constraint (κ ray-march through the sheet)" begin
        c = fiducial_cosmology(); pk = linear_power_spectrum(c)
        rng = MersenneTwister(0); Nr = 300
        ra = 360 .* rand(rng, Nr); dec = rad2deg.(asin.(2 .* rand(rng, Nr) .- 1)); z = 0.1 .+ 0.9 .* rand(rng, Nr)
        res = 16; geom = box_geometry((ra=ra, dec=dec, z=z), c; res=res, pad_frac=0.1); L = geom.boxsize
        gm = galaxy_model(res, L, c, pk; R=max(2L/res, 80.0), observer=geom.observer,
                          a_far=geom.a_far, a_near=geom.a_near, n_order=1, rsd=false)
        ndir = 60; φg = π*(3 - sqrt(5))                                  # Fibonacci sky directions
        nhat = [ (i2=i-1; y=1-2i2/(ndir-1); r=sqrt(max(0,1-y^2)); k==1 ? r*cos(φg*i2) : k==2 ? y : r*sin(φg*i2))
                 for i in 1:ndir, k in 1:3 ]
        ω = randn(MersenneTwister(1), res, res, res)
        # the differentiable convergence forward
        lc0 = lensing_constraint(gm, geom, c, nhat, zeros(ndir); nshell=12)
        κ = kappa_map(lc0, gm, ω)
        @test length(κ) == ndir && all(isfinite, κ) && std(κ) > 0
        if ad_ok
            g = Zygote.gradient(w -> 0.5*sum(abs2, kappa_map(lc0, gm, w)), ω)[1]
            @test all(isfinite, g) && any(abs.(g) .> 0)                  # ∂κ/∂ω via existing rrules
        end
        # as a constraint inside a joint problem: toward a mock κ, added to a tracer term
        lc = lensing_constraint(gm, geom, c, nhat, κ; nshell=12)
        pts = inject_mock_sheet(gm, ω, [2.0, 0, 0], 15.0 * res^3; seed=3)
        mtp = multitracer_problem(gm, [tracer(gm, pts; b1=2.0, window=ones(res,res,res))]; lensing=lc)
        φ0 = 2π .* rand(MersenneTwister(2), res ÷ 2 + 1, res, res)
        @test isfinite(multitracer_phase_loss(mtp, φ0))
        if ad_ok
            gφ = Zygote.gradient(φ -> multitracer_phase_loss(mtp, φ), φ0)[1]
            @test all(isfinite, gφ) && any(abs.(gφ) .> 0)
        end
    end

    @testset "Peculiar-velocity constraint (CF4-style, sheet velocity)" begin
        c = fiducial_cosmology(); pk = linear_power_spectrum(c)
        rng = MersenneTwister(0); Nr = 300
        ra = 360 .* rand(rng, Nr); dec = rad2deg.(asin.(2 .* rand(rng, Nr) .- 1)); z = 0.1 .+ 0.9 .* rand(rng, Nr)
        res = 16; geom = box_geometry((ra=ra, dec=dec, z=z), c; res=res, pad_frac=0.1); L = geom.boxsize
        gm = galaxy_model(res, L, c, pk; R=max(2L/res, 80.0), observer=geom.observer,
                          a_far=geom.a_far, a_near=geom.a_near, n_order=2, rsd=false)
        ω = randn(MersenneTwister(1), res, res, res)
        shift = vec(geom.shift); χn = comoving_distance(c, geom.a_near); χf = comoving_distance(c, geom.a_far)
        Ng = 120; φg = π*(3 - sqrt(5))                                  # CF4-like tracers in the lightcone shell
        dirs = [ (i2=i-1; y=1-2i2/(Ng-1); r=sqrt(max(0,1-y^2)); k==1 ? r*cos(φg*i2) : k==2 ? y : r*sin(φg*i2))
                 for i in 1:Ng, k in 1:3 ]
        rr = χn .+ (χf - χn) .* rand(MersenneTwister(3), Ng); pts = zeros(Ng, 3)
        for p in 1:Ng; pts[p,:] = rr[p] .* dirs[p,:] .+ shift; end
        # the differentiable radial-velocity forward
        vc0 = velocity_constraint(gm, geom, c, pts, zeros(Ng); sigma_v=fill(150.0, Ng))
        vr = radial_velocity(vc0, gm, ω)
        @test length(vr) == Ng && all(isfinite, vr) && std(vr) > 0
        @test 5 < std(vr) < 5000                                       # physically-scaled km/s, not runaway
        if ad_ok
            g = Zygote.gradient(w -> 0.5*sum(abs2, radial_velocity(vc0, gm, w)), ω)[1]
            @test all(isfinite, g) && any(abs.(g) .> 0)                # ∂v_r/∂ω via the sheet velocity
        end
        # as a constraint toward mock v_obs = v_r(ω), velocity-only joint problem (empty tracer list)
        vc = velocity_constraint(gm, geom, c, pts, vr; sigma_v=fill(150.0, Ng), submean=false)
        mtp = multitracer_problem(gm, Tracer[]; velocity=vc)
        φ0 = 2π .* rand(MersenneTwister(2), res ÷ 2 + 1, res, res)
        @test isfinite(multitracer_phase_loss(mtp, φ0))
        if ad_ok
            gφ = Zygote.gradient(φ -> multitracer_phase_loss(mtp, φ), φ0)[1]
            @test all(isfinite, gφ) && any(abs.(gφ) .> 0)
        end
    end

    @testset "CF4 error model + Gaussian-ω posterior loss (HMC target)" begin
        c = fiducial_cosmology(); pk = linear_power_spectrum(c)
        rng = MersenneTwister(0); N = 200                          # synthetic CF4-like catalog (no file)
        ra = 360 .* rand(rng, N); dec = rad2deg.(asin.(2 .* rand(rng, N) .- 1)); dist = 5 .+ 95 .* rand(rng, N)
        e_dm = fill(0.4, N); vcmb = 75 .* dist .+ 300 .* randn(rng, N); ngal = ones(N)
        keep = dist .< 100.0                                       # (load_cf4_groups is just npzread + this filter)
        cat = CF4Catalog{Float64}(ra[keep], dec[keep], dist[keep], e_dm[keep], vcmb[keep], ngal[keep])
        @test length(cat) > 0
        pv = cf4_peculiar_velocity(cat)                            # Malmquist increases distances
        @test all(isfinite, pv.vpec) && all(pv.σv .> 0) && all(pv.dist .>= cat.dist) && pv.H0 > 0
        res = 16; geom = cf4_box_geometry(cat, c; res=res)
        gm = galaxy_model(res, geom.boxsize, c, pk; R=max(2geom.boxsize/res, 10.0), observer=geom.observer,
                          a_far=geom.a_far, a_near=geom.a_near, n_order=2, rsd=false)
        vc = cf4_velocity_constraint(gm, geom, c, cat)
        @test length(vc.v_obs) == length(cat)
        mtp = multitracer_problem(gm, Tracer[]; velocity=vc)
        ω = randn(MersenneTwister(1), res, res, res)
        L0 = loss(mtp, ω, zeros(3))                                # Gaussian-ω posterior loss (NUTS target)
        @test isfinite(L0) && L0 > 0                               # includes the ½‖ω‖² prior
        if ad_ok
            g = Zygote.gradient(w -> loss(mtp, w, zeros(3)), ω)[1]
            @test all(isfinite, g) && any(abs.(g) .> 0)
        end
    end

    @testset "Perturb-and-MAP constrained realizations (Wiener posterior)" begin
        c = fiducial_cosmology(); pk = linear_power_spectrum(c)
        rng = MersenneTwister(0); Nr = 200
        ra = 360 .* rand(rng, Nr); dec = rad2deg.(asin.(2 .* rand(rng, Nr) .- 1)); z = 0.02 .+ 0.1 .* rand(rng, Nr)
        res = 16; geom = box_geometry((ra=ra, dec=dec, z=z), c; res=res, pad_frac=0.15); L = geom.boxsize
        gm = galaxy_model(res, L, c, pk; R=max(2L/res, 30.0), observer=geom.observer,
                          a_far=geom.a_far, a_near=geom.a_near, n_order=2, rsd=false)
        ω_true = randn(MersenneTwister(42), res, res, res)          # a prior draw N(0,I)
        shift = vec(geom.shift); χn = comoving_distance(c, geom.a_near); χf = comoving_distance(c, geom.a_far)
        Ng = 80; φg = π*(3 - sqrt(5))
        dirs = [ (i2=i-1; y=1-2i2/(Ng-1); r=sqrt(max(0,1-y^2)); k==1 ? r*cos(φg*i2) : k==2 ? y : r*sin(φg*i2))
                 for i in 1:Ng, k in 1:3 ]
        rr = χn .+ (χf - χn) .* rand(MersenneTwister(3), Ng); pts = zeros(Ng, 3)
        for p in 1:Ng; pts[p,:] = rr[p] .* dirs[p,:] .+ shift; end
        vc0 = velocity_constraint(gm, geom, c, pts, zeros(Ng); sigma_v=fill(150.0, Ng))
        vobs = radial_velocity(vc0, gm, ω_true) .+ 150.0 .* randn(MersenneTwister(9), Ng)
        vc = velocity_constraint(gm, geom, c, pts, vobs; sigma_v=fill(150.0, Ng), submean=true)
        mtp = multitracer_problem(gm, Tracer[]; velocity=vc)
        ωwf = wiener_mean(mtp; iters=60)                            # posterior mean (Wiener filter)
        @test size(ωwf) == (res, res, res) && all(isfinite, ωwf)
        cr = constrained_realizations(mtp, 3; iters=60, seed=1)     # posterior samples
        @test size(cr.omega_mean) == (res, res, res) && length(cr.draws) == 3
        @test all(isfinite, cr.omega_mean) && all(isfinite, cr.omega_std) && mean(cr.omega_std) > 0
    end

end
