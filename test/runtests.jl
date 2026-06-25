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
    end

end
