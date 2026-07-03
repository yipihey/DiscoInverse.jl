# Field-level reconstruction from Cosmicflows-4 peculiar velocities.
#
# CF4 distances carry ~0.5-mag (lognormal) errors, so the per-group velocity noise σ_v ∝ distance
# reaches thousands of km/s beyond the Local Volume — the field is deeply noise-dominated.  The
# fixed-amplitude MAP overfits that noise; we sample the POSTERIOR (à la HAMLET) with NUTS instead.
#
# The CF4 groups .npz is staged from cf4_groups.fits (columns ra, dec, dist, e_dm, vcmb, ngal).
#
#   julia --project=. examples/cf4_reconstruction.jl

using DiscoInverse, DiscoDJNative, Statistics, Printf, Random

c  = fiducial_cosmology()
pk = linear_power_spectrum(c)

# 1. load CF4 groups (cut the noisy far tail; σ_v ∝ distance)
cat = load_cf4_groups("path/to/cf4_groups.npz"; dist_max=150.0)
@printf("%d CF4 groups; effective H0 = %.1f km/s/Mpc\n", length(cat), cf4_hubble(cat))

# 2. box + forward (observer at the box centre; z from Vcmb)
res  = 64
geom = cf4_box_geometry(cat, c; res=res)
gm   = galaxy_model(res, geom.boxsize, c, pk; R=max(2geom.boxsize/res, 10.0),
                    observer=geom.observer, a_far=geom.a_far, a_near=geom.a_near, n_order=2, rsd=false)

# 3. the peculiar-velocity constraint: Malmquist-corrected distances, lognormal-error σ_v, monopole
#    marginalized.  (Add galaxy `tracer`s and/or a `lensing_constraint` here to fuse them in one field.)
vc  = cf4_velocity_constraint(gm, geom, c, cat)          # malmquist=true, submean=true
mtp = multitracer_problem(gm, Tracer[]; velocity=vc)     # velocity-only; or pass tracers=[...], lensing=...

# 4. sample the POSTERIOR over the initial field (not MAP — CF4 is low-S/N).  ω_mean is the honest
#    reconstruction; ω_std is its per-voxel uncertainty; ω_draws are constrained realizations.
r = nuts_sample(mtp, randn(res, res, res), zeros(3); nsamples=200, nwarmup=200, max_depth=7, seed=1)
@printf("NUTS: accept %.2f, divergences %d, %d draws kept\n", r.accept, r.divergences, length(r.ω_draws))

# the model radial peculiar velocities of the posterior-mean field, at the CF4 groups
v_model = radial_velocity(vc, gm, r.ω_mean)
@printf("posterior-mean model v_r: std %.0f km/s over %d groups\n", std(v_model), length(v_model))
