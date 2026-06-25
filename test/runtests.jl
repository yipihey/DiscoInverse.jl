using Test
using DiscoInverse
using Random: MersenneTwister
using Statistics: mean, std

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

end
