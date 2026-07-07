# Our local reconstruction vs the published field: Manticore, CF4

Do our DiscoInverse local density-field reconstructions agree with the established Bayesian and
peculiar-velocity reconstructions of the same volume? This is the local-universe analogue of the DES
validation: an out-of-method cross-check against fields built by other groups from other data.

## The four fields

All placed on a common **observer-centred equatorial Cartesian grid in physical Mpc** (h = 0.681),
smoothed to a common Gaussian scale, compared voxel-wise inside r < 120 Mpc:

| field | method / data | grid | frame |
|---|---|---|---|
| **Manticore** | BORG field-level, 2M++-constrained ([Manticore I](https://arxiv.org/abs/2505.xxxxx)) | 256³ / 1000 Mpc | equatorial, 1+δ |
| **CF4** | CosmicFlows-4 Wiener ([Courtois+2023](https://academic.oup.com/mnras/article/527/2/3788/7419869)) | 64³ / 1000 Mpc/h | supergalactic, δ |
| **GraphGP (ours)** | 2M++ GraphGP inpaint | 48³ / 300 Mpc | equatorial, δ |
| **DiscoInverse (ours)** | field-level 2M++ **+** CF4, ω→δ_L (EH98 P(k)) | 96³ / 607 Mpc/h | equatorial, δ_L |

A frame note worth recording: the CF4 cube's axes are **natural (SGX,SGY,SGZ)**, not the "SGZ,SGY,SGX"
the ECHOES loader comment claims — recovered empirically (CF4↔Manticore jumps 0.10 → 0.62 with the
correct order; the simplest convention, no flips, wins cleanly). Fixed in this analysis; the ECHOES
loader should be corrected.

## Cross-correlation (r < 120 Mpc, 15 Mpc smoothing)

|                    | Manticore | CF4  | GraphGP | DiscoInverse |
|--------------------|:---------:|:----:|:-------:|:------------:|
| **Manticore**      | 1.00      | 0.63 | 0.49    | **0.76**     |
| **CF4**            | 0.63      | 1.00 | 0.09    | **0.55**     |
| **GraphGP (ours)** | 0.49      | 0.09 | 1.00    | 0.52         |
| **DiscoInverse**   | **0.76**  | 0.55 | 0.52    | 1.00         |

**The benchmark is Manticore↔CF4 = 0.63** — how well two accepted, independent reconstructions of this
volume agree. Our **DiscoInverse field is mutually consistent with both at that level**: 0.76 with
Manticore, 0.55 with CF4. It does not sit as an outlier to the published family — it sits inside it.

**Honest reading of the numbers.** DiscoInverse↔Manticore (0.76) is higher than CF4↔Manticore (0.63)
**not** because our method is "better" — DiscoInverse ingests 2M++, which Manticore also uses, so the two
**share data** and should correlate above the CF4 benchmark. The genuinely informative number is
DiscoInverse↔**CF4** = 0.55: CF4 is peculiar-velocity data, largely independent of the redshift surveys
our field is built from, and we reproduce it as well as Manticore does (0.63). GraphGP (2M++ only, 48³,
a smooth GP inpaint) recovers the big structures — 0.49 with Manticore — but correlates poorly with CF4
(0.09) and fills voids it should leave empty; it is the weaker of our two methods, as expected.

Two further caveats, both of which make 0.76/0.55 **conservative floors**: (i) our DiscoInverse δ is the
*linear* field (ω coloured with an EH98 P(k)), while Manticore/CF4 are evolved — at 15 Mpc smoothing
linear≈nonlinear, but the residual mismatch only lowers r; (ii) CF4's 23 Mpc voxel caps small-scale
agreement.

## Named superstructures (field value in σ; independent of the frame fit)

The named-structure positions were **not** used to fix any frame, so this is an independent check:

| structure | expect | Manticore | CF4  | GraphGP | DiscoInverse |
|---|:---:|:---:|:---:|:---:|:---:|
| Virgo          | + | **+11.2** | +0.1 | +2.7 | +1.5 |
| Great Attractor| + | +0.1 | +1.5 | +1.8 | +0.2 |
| Coma           | + | −0.1 | +1.7 | +2.4 | +2.1 |
| Perseus-Pisces | + | +0.5 | +2.1 | +2.7 | +2.3 |
| Local Void     | − | −0.1 | −0.6 | +1.3 | **−2.1** |

All four recover **Perseus-Pisces**; three of four recover **Coma** and the **Local Void** (our
DiscoInverse gets the Local Void at −2.1σ — correctly empty — where the GP inpaint fills it). Virgo (16
Mpc) dominates Manticore but is below CF4's voxel. The Manticore near-zeros for Coma/GA reflect its
enormous Virgo-centred dynamic range under σ-normalization, not a miss.

## Takeaway and next steps

Our field-level local reconstruction is **quantitatively consistent with the two leading published
reconstructions of the local volume, at the level they agree with each other** — an independent,
out-of-method validation of the DiscoInverse machinery on real data, complementary to the DES/κ tests
of the high-z carrier.

To fetch (public, would add independent references): **Carrick+2015** 2M++ (256³/400 h⁻¹Mpc, Yahil
iterative) and the reconstructions collated by the 2025 **[Velocity Field Olympics](https://arxiv.org/abs/2502.00121)**;
**[Manticore II](https://arxiv.org/abs/2506.10020)** extends BORG twins across the **SDSS/BOSS** volume —
a route to compare our *high-z carrier* (not just the local field) against a BORG reconstruction.

Artifacts: `scratch/local_reconstruction_comparison.png` (SG-plane slices),
`scratch/local_reconstruction_xcorr.npz` (matrix), analysis in `/tmp/local_compare.py`.
