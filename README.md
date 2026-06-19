# SinCosLUT.jl

[![CI](https://github.com/JuliaGNSS/SinCosLUT.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/JuliaGNSS/SinCosLUT.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/JuliaGNSS/SinCosLUT.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JuliaGNSS/SinCosLUT.jl)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaGNSS.github.io/SinCosLUT.jl/dev)

Fast SIMD sine/cosine by **register-resident table lookup**. Each `sin`/`cos` is a
single hardware byte/word/dword permute over a table held entirely in vector
registers — no multiplies, no memory gathers, and (unlike floating-point range
reduction) **no input range limit**: the phase wraps for free via a bit-mask.

It is a deliberately *low-precision, very high-throughput* technique. Pick the
output element type to trade precision for speed:

| output `T` | lanes (AVX-512) | table entries / cycle | ~accuracy |
| ---------- | --------------- | --------------------- | --------- |
| `Int8`     | 64              | 64 or 128             | ~3–7 bit  |
| `Int16`    | 32              | 32 or 64              | finer amplitude |
| `Int32`    | 16              | 16 or 32              | finest amplitude |

A 512-bit register holds 64×`Int8` but only 32×`Int16` / 16×`Int32`, so wider
elements give more amplitude bits but fewer table entries (coarser phase) and
fewer lanes (lower throughput).

## Backends

The lookup primitive is chosen automatically from the CPU:

| backend  | instruction        | types        | notes |
| -------- | ------------------ | ------------ | ----- |
| AVX-512  | `vpermb`/`vpermw`/`vpermd` (+`vpermi2*`) | Int8/16/32 | fastest |
| AVX2     | `vpshufb` + blends | Int8 only    | 64-entry via 4-way split; slower |
| NEON     | `tbl` (`tbl4`)     | Int8 only    | AArch64; 16 lanes |
| portable | scalar             | Int8/16/32   | always available |

**SVE2 is not supported.** Julia cannot express LLVM scalable vectors
(`<vscale x N x T>`), see [JuliaLang/julia#40308](https://github.com/JuliaLang/julia/issues/40308).
NEON `tbl` runs natively on SVE hardware regardless. On AVX2/NEON, `Int16`/`Int32`
fall back to the portable backend (no word/dword table permute available there).

## Usage

```julia
using SinCosLUT

# 64-entry Int8 table (6-bit phase, full Int8 amplitude)
tbl = SinCosTable(Int8; steps = 64)

sins = zeros(Int8, 4096)
coss = zeros(Int8, 4096)

# Carrier of 0.002 cycles/sample (= normalised frequency f/fs), drift-free integer
# phase (no accumulation error even over very long runs):
generate_carrier!(sins, coss, tbl, 0.002)

# ...directly from a frequency and sampling frequency (keyword form):
generate_carrier!(sins, coss, tbl; frequency = 1000, sampling_frequency = 2e6)   # 1 kHz at 2 MHz
# ...or via the cycles_per_sample helper (= frequency / sampling_frequency):
generate_carrier!(sins, coss, tbl, cycles_per_sample(1000, 2e6))                 # → 0.0005

# ...or an exact rational phase step P/Q table-steps per sample:
generate_carrier!(sins, coss, tbl, 16, 125)      # 16/125 steps/sample

# optional initial carrier phase (default 0): Integer = table steps, Real = cycles
generate_carrier!(sins, coss, tbl, 0.002; phase = 16)     # 16 table steps  (¼ cycle, 64-step table)
generate_carrier!(sins, coss, tbl, 0.002; phase = 0.25)   # 0.25 cycles = 90°, same result

# Look up sin/cos for an arbitrary array of integer phase indices (taken mod steps):
phases = rand(Int8, 4096)
lookup_sincos!(sins, coss, phases, tbl)
```

### Array-free / fused generation (returns `Vec`s, like FastSinCos)

To avoid materialising a carrier array (and the cache traffic that comes with it),
iterate the carrier and consume each `(sin, cos)` SIMD `Vec` in registers. The
drift-free DDA lives in the (isbits) iteration state, so the loop allocates nothing:

```julia
# correlate a carrier against a signal without ever building the carrier array
function correlate(tbl, cps, signal)
    acc = 0; i = 1
    @inbounds for (s, c) in generate_carrier(tbl, cps, length(signal))   # s, c :: Vec{W,Int8}
        sg = signal[VecRange{length(s)}(i)]
        acc += sum(Vec{length(s),Int32}(s) * Vec{length(s),Int32}(sg))
        i += length(s)
    end
    acc
end
```

`generate_carrier(table, P, Q, nsamples)` / `generate_carrier(table, cycles_per_sample, nsamples)` yields
`nsamples ÷ W` chunks of `(sin, cos)::Tuple{Vec{W,T},Vec{W,T}}` (W = the backend's
SIMD width). The fused loop above is **0 allocations** and avoids the carrier
store/reload entirely.

For a trivial consumer (e.g. filling an array) the single-Vec iterator is
latency-bound on its one DDA carry chain. Use **`generate_carrier4`**, which yields four
`(sin, cos)` pairs per step from four interleaved DDA states — the carry chains
overlap and it reaches the full loop rate (~40 ps/elem at scale, matching
`generate_carrier!`), allocation-free:

```julia
i = 1
for ((s0,c0),(s1,c1),(s2,c2),(s3,c3)) in generate_carrier4(tbl, 0.002, length(sins))
    sins[VecRange{64}(i)]     = s0; coss[VecRange{64}(i)]     = c0
    sins[VecRange{64}(i+64)]  = s1; coss[VecRange{64}(i+64)]  = c1
    sins[VecRange{64}(i+128)] = s2; coss[VecRange{64}(i+128)] = c2
    sins[VecRange{64}(i+192)] = s3; coss[VecRange{64}(i+192)] = c3
    i += 256
end
```

Rule of thumb: `generate_carrier` when fusing into nontrivial work (it borrows your loop's
ILP); `generate_carrier4` (or `generate_carrier!`) when the per-sample work is light.

For the stateless, FastSinCos-style primitive (you supply the phase indices),
`prepare` builds the register-resident table once and returns a callable:

```julia
p = prepare(tbl)              # build table in registers once
s, c = p(idx::Vec)           # idx -> (sin, cos), like fast_sincos_*(::Vec)
```

Higher amplitude precision:

```julia
tbl16 = SinCosTable(Int16; steps = 64)   # 6-bit phase, 15-bit amplitude (vpermi2w)
sins = zeros(Int16, 4096); coss = zeros(Int16, 4096)
generate_carrier!(sins, coss, tbl16, 0.002)
```

You can force a backend (e.g. for testing or to avoid the runtime CPU check):

```julia
using SinCosLUT: AVX512, AVX2, Portable
generate_carrier!(sins, coss, tbl, 0.002; backend = Portable())
```

## Accuracy & range

- **Phase** is quantised to `steps` per cycle (`steps`-entry table): 64 entries ≈
  6-bit phase. Larger tables (`2*regsize(T)`, via `vpermi2`) double it.
- **Amplitude** is quantised to the output type's range (`Int8` ≈ 7-bit).
- **No range limit**: any phase value (positive, negative, huge) is reduced exactly
  by the index bit-mask — there is no Cody–Waite-style ceiling or precision loss at
  large arguments, unlike float approximations.

If you need more than a few bits, a polynomial approximation (e.g.
[FixedPointSinCosApproximations.jl](https://github.com/JuliaGNSS/FixedPointSinCosApproximations.jl)
or [FastSinCos.jl](https://github.com/JuliaGNSS/FastSinCos.jl)) is the better tool;
this package is for the regime where a tiny register-resident table and one permute
per result is the win.

## Comparison with FastSinCos.jl and FixedPointSinCosApproximations.jl

Three JuliaGNSS packages compute fast SIMD sin/cos with different speed–accuracy
trade-offs: this **lookup table**, the float minimax polynomial
[FastSinCos.jl](https://github.com/JuliaGNSS/FastSinCos.jl), and the integer minimax
polynomial
[FixedPointSinCosApproximations.jl](https://github.com/JuliaGNSS/FixedPointSinCosApproximations.jl).

Kernel throughput (computing **both** sin and cos for an array of inputs) and
worst-case absolute error vs the true values, measured on a Zen 5 core. `~bits` is
`-log2(max error)`. The `AVX2` block forces the AVX2-width path on the same hardware.

**AVX-512**

| method | ps/elem | max abs error | ~bits |
| ------ | ------: | ------------: | ----: |
| **SinCosLUT** Int8, steps=64  | **28**  | 5.2e-2 | ~4  |
| **SinCosLUT** Int8, steps=128 | **34**  | 2.8e-2 | ~5  |
| FixedPoint Int16 (Val 7)      | 103     | 1.3e-2 | ~6  |
| FastSinCos `u100k`            | 187     | 3.2e-4 | ~12 |
| FixedPoint Int32 (Val 8)      | 206     | 8.1e-3 | ~7  |
| FastSinCos `u35`              | 226     | 6.0e-8 | ~24 |
| FixedPoint Int32 (Val 14)     | 293     | 1.1e-4 | ~13 |

**AVX2** (no native cross-lane byte permute — `vpermb` is unavailable)

| method | ps/elem | max abs error | ~bits |
| ------ | ------: | ------------: | ----: |
| FixedPoint Int16 (Val 7) | 123 | 1.3e-2 | ~6  |
| FixedPoint Int32 (Val 8) | 246 | 8.1e-3 | ~7  |
| FastSinCos `u100k`       | 250 | 3.2e-4 | ~12 |
| FastSinCos `u35`         | 320 | 6.0e-8 | ~24 |
| **SinCosLUT** Int8, steps=64 | 346 | 5.2e-2 | ~4 |

**End-to-end carrier** (phase generation + sincos, 0.01 cycles/sample). The kernel rows
above feed *pre-computed* phases; here each package generates the carrier itself. This is
where FixedPoint's multiplicative-inverse phase work shows — the kernel rows are
unaffected by it. All three are drift-free: `generate_carrier!` advances the phase with an
exact integer DDA, and the FastSinCos row computes the Float32 phase from the exact sample
index (a plain `acc += step` accumulator is a little faster but drifts).

AVX-512:

| method | ps/elem | max abs error | ~bits |
| ------ | ------: | ------------: | ----: |
| **SinCosLUT** Int8, steps=64       | **70**  | 9.3e-2 | ~3  |
| FixedPoint Int16 (Val 7)           | **129** | 1.5e-2 | ~6  |
| FastSinCos `u100k` (Float32 phase) | 235     | 3.1e-4 | ~12 |
| FixedPoint Int32 (Val 13)          | 318     | 2.4e-4 | ~12 |

AVX2:

| method | ps/elem | max abs error | ~bits |
| ------ | ------: | ------------: | ----: |
| FixedPoint Int16 (Val 7)           | **143** | 1.5e-2 | ~6  |
| FastSinCos `u100k` (Float32 phase) | 325     | 3.1e-4 | ~12 |
| **SinCosLUT** Int8, steps=64       | 368     | 9.3e-2 | ~3  |
| FixedPoint Int32 (Val 13)          | 383     | 2.4e-4 | ~12 |

FixedPoint's drift-free DDA carrier is only marginally slower than its bare kernel (129 vs
103 ps on AVX-512) — integer phase generation is nearly free. At ~6-bit accuracy
FixedPoint Int16 is the fastest *polynomial* carrier on both AVX-512 (~1.8×) and AVX2
(~2.3×); at float-grade accuracy FastSinCos edges FixedPoint Int32. SinCosLUT is fastest
of all on AVX-512 but coarsest, and (as in the kernel rows) loses its edge on AVX2.

Takeaways:

- **On AVX-512, SinCosLUT is fastest by 3–7×** (one `vpermb` per result) — but only
  ~4–5 bits accurate, set by the table's phase resolution.
- **On AVX2, SinCosLUT loses its edge**: with no native byte permute the lookup is
  emulated with a four-way `vpshufb`+blend split and is *slower* than either
  polynomial package. On AVX2 (and Haswell), prefer FixedPoint or FastSinCos.
- Pick by accuracy: **≤5-bit on AVX-512/NEON → SinCosLUT**; **~6–13-bit integer →
  FixedPointSinCosApproximations**; **float-grade (12–24 bit) → FastSinCos**.

(Reproduce both tables with `julia benchmark/comparison.jl`.)

## Drift-free phase

`generate_carrier!` advances the phase with an exact integer DDA (`index = div(i·P, Q) mod
steps`), so the frequency never drifts — even over millions of samples — whereas a
binary fractional phase accumulator drifts whenever the frequency's denominator is
not a power of two.
