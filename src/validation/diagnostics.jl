"""
Recovery diagnostics — the headline injection-recovery metrics.

`overall_correlation` is the scalar cosine between recovered and true ω;
`cross_spectrum_r(ω̂, ω*)` is the per-|k|-bin cross-correlation coefficient r(k) =
⟨ω̂ ω*⟩ / √(⟨|ω̂|²⟩⟨|ω*|²⟩) — the standard BORG diagnostic (→1 on large scales where the
data is informative, declining at high k).
"""

"""    overall_correlation(a, b) -> cosine similarity ⟨a·b⟩/√(‖a‖²‖b‖²)"""
overall_correlation(a::AbstractArray, b::AbstractArray) =
    sum(a .* b) / sqrt(sum(abs2, a) * sum(abs2, b))

"""
    cross_spectrum_r(ω1, ω2; nbins=8, boxsize=1.0) -> (; k, r, n_modes)

Per-|k|-bin cross-correlation coefficient of two real fields.
"""
function cross_spectrum_r(ω1::AbstractArray{T,3}, ω2::AbstractArray{T,3};
                          nbins::Int=8, boxsize::Real=1.0) where {T}
    res = size(ω1, 1)
    f1 = rfft(ω1, [3, 1, 2]); f2 = rfft(ω2, [3, 1, 2])
    dk = 2π / boxsize
    kf = T[(i <= res ÷ 2 ? i : i - res) for i in 0:res-1] .* dk
    kh = collect(T, 0:res÷2) .* dk
    kmag = sqrt.(reshape(kf, res, 1, 1).^2 .+ reshape(kf, 1, res, 1).^2 .+ reshape(kh, 1, 1, :).^2)
    kmax = maximum(kmag); kmin = dk
    edges = exp10.(range(log10(kmin), log10(kmax); length=nbins + 1))
    kcen = zeros(T, nbins); r = zeros(T, nbins); nm = zeros(Int, nbins)
    for b in 1:nbins
        m = (kmag .>= edges[b]) .& (kmag .< edges[b+1])
        n = count(m); nm[b] = n
        n == 0 && continue
        c12 = real(sum(f1[m] .* conj.(f2[m])))
        p1 = sum(abs2, f1[m]); p2 = sum(abs2, f2[m])
        kcen[b] = sum(kmag[m]) / n
        r[b] = c12 / sqrt(max(p1 * p2, eps(T)))
    end
    return (k=kcen, r=r, n_modes=nm)
end
