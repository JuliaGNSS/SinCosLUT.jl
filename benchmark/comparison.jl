# Reproduces the speed/accuracy comparison table in the README:
# SinCosLUT.jl vs FastSinCos.jl (float polynomial) vs
# FixedPointSinCosApproximations.jl (integer polynomial).
#
# Standalone script (NOT the AirspeedVelocity suite in benchmarks.jl). Run with:
#     julia benchmark/comparison.jl
#
# Measures kernel throughput (computing BOTH sin and cos for an array of inputs) and
# worst-case absolute error vs the true values, at AVX-512 and AVX2 widths. AVX2 rows
# force the AVX2-width path on the same hardware (Float32×8 / Int8×32). Numbers depend
# on the host CPU.

using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()   # installs all deps (SinCosLUT via its [sources] path entry)
using SinCosLUT, FastSinCos, FixedPointSinCosApproximations, SIMD, BenchmarkTools, Printf
using SinCosLUT: AVX512, AVX2
# generate_carrier! / cycles_per_sample are exported by both LUT and FixedPoint packages,
# so qualify them in the end-to-end carrier section below.
import SinCosLUT as SL
import FixedPointSinCosApproximations as FP

const L = 16384
const angles = Float32.((0:L-1) .* (2π * 7 / L))   # several cycles of continuous radians
mn(b) = minimum(b).time
ps(t) = round(t / L * 1000, digits = 1)

# ---- FastSinCos: loop fast_sincos over Vec{W,Float32} ----
fs_fill!(f, so, co, a, ::Val{W}) where {W} =
    (@inbounds for i in 1:W:length(a); l = VecRange{W}(i); s, c = f(a[l]); so[l] = s; co[l] = c; end)
fs_err(so, co, a) = maximum(max(abs(so[i] - sin(a[i])), abs(co[i] - cos(a[i]))) for i in eachindex(a))
function run_fs(f, W)
    s = zeros(Float32, L); c = zeros(Float32, L); fs_fill!(f, s, c, angles, Val(W))
    (ps(mn(@benchmark fs_fill!($f, $s, $c, $angles, $(Val(W))))), fs_err(s, c, angles))
end

# ---- SinCosLUT: round angle -> phase index, lookup_sincos! ----
function run_scl(T, steps, backend)
    tbl = SinCosTable(T; steps = steps); amp = Float64(typemax(T))
    idx = T.(mod.(round.(Int, angles ./ (2π) .* steps), steps))
    s = zeros(T, L); c = zeros(T, L)
    lookup_sincos!(s, c, idx, tbl; backend = backend)
    err = maximum(max(abs(s[i] / amp - sin(angles[i])), abs(c[i] / amp - cos(angles[i]))) for i in eachindex(angles))
    (ps(mn(@benchmark lookup_sincos!($s, $c, $idx, $tbl; backend = $backend))), err)
end

# ---- FixedPointSinCosApproximations: integer polynomial, N quarter-bits ----
fp_fill!(so, co, ph, ::Val{N}, ::Val{W}) where {N,W} =
    (@inbounds for i in 1:W:length(ph); l = VecRange{W}(i); s, c = fpsincos(ph[l], Val(N)); so[l] = s; co[l] = c; end)
function run_fp(T, N, W)
    amp = 2.0^N; ph = T.(round.(Int, angles .* (amp / (π / 2))))
    s = zeros(T, L); c = zeros(T, L); fp_fill!(s, c, ph, Val(N), Val(W))
    err = maximum(max(abs(s[i] / amp - sin(angles[i])), abs(c[i] / amp - cos(angles[i]))) for i in eachindex(angles))
    (ps(mn(@benchmark fp_fill!($s, $c, $ph, $(Val(N)), $(Val(W))))), err)
end

# ===== End-to-end carrier: phase GENERATION + sincos =====
# The kernel rows above feed pre-computed phases. Here each package generates the carrier
# itself — this is where FixedPoint's drift-free integer DDA (multiplicative-inverse phase
# init) shows up; the kernel rows are unaffected by it. All three run at the same
# normalised frequency.
const CARRIER_CYC = 0.01    # cycles/sample
true_sin(i) = sin(2π * CARRIER_CYC * (i - 1))
true_cos(i) = cos(2π * CARRIER_CYC * (i - 1))

function run_fp_carrier(T, N, W)
    s = Vector{T}(undef, L); c = Vector{T}(undef, L); amp = 2.0^N
    FP.generate_carrier!(s, c, Val(N), CARRIER_CYC; lanes = Val(W))
    err = maximum(max(abs(s[i] / amp - true_sin(i)), abs(c[i] / amp - true_cos(i))) for i in 1:L)
    (ps(mn(@benchmark FP.generate_carrier!($s, $c, $(Val(N)), $CARRIER_CYC; lanes = $(Val(W))))), err)
end

function run_scl_carrier(T, steps, backend)
    tbl = SL.SinCosTable(T; steps = steps); amp = Float64(typemax(T))
    s = Vector{T}(undef, L); c = Vector{T}(undef, L)
    SL.generate_carrier!(s, c, tbl, CARRIER_CYC; backend = backend)
    err = maximum(max(abs(s[i] / amp - true_sin(i)), abs(c[i] / amp - true_cos(i))) for i in 1:L)
    (ps(mn(@benchmark SL.generate_carrier!($s, $c, $tbl, $CARRIER_CYC; backend = $backend))), err)
end

# FastSinCos has no carrier API; generate the phase from the exact integer sample index
# each chunk (so the float phase doesn't drift — a plain `acc += step` accumulator would,
# the more so the longer the run / the narrower W) and call the kernel.
function fs_carrier!(re, im, f, ::Val{W}) where {W}
    step = Float32(2π * CARRIER_CYC)
    lane = Vec{W,Float32}(ntuple(j -> Float32(j - 1), Val(W)))
    @inbounds for i in 1:W:L
        phase = (Float32(i - 1) + lane) * step
        s, c = f(phase); l = VecRange{W}(i); im[l] = s; re[l] = c
    end
end
function run_fs_carrier(f, W)
    re = zeros(Float32, L); im = zeros(Float32, L); fs_carrier!(re, im, f, Val(W))
    err = maximum(max(abs(im[i] - true_sin(i)), abs(re[i] - true_cos(i))) for i in 1:L)
    (ps(mn(@benchmark fs_carrier!($re, $im, $f, $(Val(W))))), err)
end

row(nm, t, e) = @printf("  %-28s %8s %12.2e\n", nm, t, e)
@printf("%-30s %8s %12s\n", "kernel: sin & cos for L inputs", "ps/elem", "max abs err")

println("== AVX-512 ==")
for (nm, f) in (("FastSinCos u35", fast_sincos_u35), ("FastSinCos u100k", fast_sincos_u100k))
    t, e = run_fs(f, 16); row(nm, t, e)
end
for (nm, T, N, W) in (("FixedPoint Int32 Val14", Int32, 14, 16),
                      ("FixedPoint Int16 Val7", Int16, 7, 32),
                      ("FixedPoint Int32 Val8", Int32, 8, 16))
    t, e = run_fp(T, N, W); row(nm, t, e)
end
for (nm, T, st) in (("SinCosLUT Int8 steps=128", Int8, 128), ("SinCosLUT Int8 steps=64", Int8, 64))
    t, e = run_scl(T, st, AVX512()); row(nm, t, e)
end

println("== AVX2 ==")
for (nm, f) in (("FastSinCos u35", fast_sincos_u35), ("FastSinCos u100k", fast_sincos_u100k))
    t, e = run_fs(f, 8); row(nm, t, e)
end
for (nm, T, N, W) in (("FixedPoint Int16 Val7", Int16, 7, 16), ("FixedPoint Int32 Val8", Int32, 8, 8))
    t, e = run_fp(T, N, W); row(nm, t, e)
end
let (t, e) = run_scl(Int8, 64, AVX2()); row("SinCosLUT Int8 steps=64", t, e) end

@printf("\nend-to-end carrier (phase gen + sincos), %g cycles/sample, AVX-512\n", CARRIER_CYC)
let (t, e) = run_scl_carrier(Int8, 64, AVX512()); row("SinCosLUT Int8 steps=64", t, e) end
let (t, e) = run_scl_carrier(Int8, 128, AVX512()); row("SinCosLUT Int8 steps=128", t, e) end
for (nm, T, N, W) in (("FixedPoint Int16 Val7", Int16, 7, 32), ("FixedPoint Int32 Val13", Int32, 13, 16))
    t, e = run_fp_carrier(T, N, W); row(nm, t, e)
end
row("FastSinCos u100k (float ph)", run_fs_carrier(fast_sincos_u100k, 16)...)

@printf("\nend-to-end carrier (phase gen + sincos), %g cycles/sample, AVX2\n", CARRIER_CYC)
for (nm, T, N, W) in (("FixedPoint Int16 Val7", Int16, 7, 16), ("FixedPoint Int32 Val13", Int32, 13, 8))
    t, e = run_fp_carrier(T, N, W); row(nm, t, e)
end
row("FastSinCos u100k (float ph)", run_fs_carrier(fast_sincos_u100k, 8)...)
let (t, e) = run_scl_carrier(Int8, 64, AVX2()); row("SinCosLUT Int8 steps=64", t, e) end
