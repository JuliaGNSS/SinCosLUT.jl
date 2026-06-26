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
| AVX2     | `vpshufb` + blends | Int8 only    | 64-entry via 4-way split; ~3× slower than AVX-512 |
| NEON     | `tbl` (`tbl4`)     | Int8 only    | AArch64; 16 lanes |
| portable | scalar             | Int8/16/32   | always available |

**AVX2 (and NEON) is `Int8`-only.** AVX2's only register-resident table permute is
`vpshufb`, a *byte* shuffle — there is no word/dword permute (`vpermw`/`vpermd` are
AVX-512). So on AVX2/NEON the SIMD lookup is implemented for `Int8` (`steps = 64`) only;
`Int16`/`Int32` silently fall back to the scalar **portable** backend (correct, but not
vectorised). If you need more amplitude bits on an AVX2 host, prefer a polynomial package
(FixedPoint/FastSinCos) over a wider SinCosLUT element type.

**SVE2 is not supported.** Julia cannot express LLVM scalable vectors
(`<vscale x N x T>`), see [JuliaLang/julia#40308](https://github.com/JuliaLang/julia/issues/40308).
NEON `tbl` runs natively on SVE hardware regardless.

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

# Any frequency resolves on a uniform f_s/2^32 grid (≈0.001 Hz at 5 MHz) — arbitrary
# Doppler works with no dead-zone near DC and never errors:
generate_carrier!(sins, coss, tbl; frequency = 1234.567, sampling_frequency = 5e6)

# optional initial carrier phase (default 0): Integer = table steps, Real = cycles
generate_carrier!(sins, coss, tbl, 0.002; phase = 16)     # 16 table steps  (¼ cycle, 64-step table)
generate_carrier!(sins, coss, tbl, 0.002; phase = 0.25)   # 0.25 cycles = 90°, same result

# Look up sin/cos for an arbitrary array of integer phase indices (taken mod steps):
phases = rand(Int8, 4096)
lookup_sincos!(sins, coss, phases, tbl)
```

### Array-free / fused generation (returns `Vec`s, like FastSinCos)

To avoid materialising a carrier array (and the cache traffic that comes with it),
build a loop-invariant `carrier_engine` once and drive it with isbits `CarrierState`s
renewed by value each iteration — the drift-free DDA lives in the state, so the loop
allocates nothing and nothing ever escapes to the heap:

```julia
# correlate a carrier against a signal without ever building the carrier array
function correlate(tbl, cps, signal)
    eng = carrier_engine(tbl, cps)
    W = carrier_width(eng); st = carrier_state(eng); acc = 0; i = 1
    @inbounds for _ in 1:(length(signal) ÷ W)
        s, c = carrier_lookup(eng, st)             # s, c :: Vec{W,Int8}
        sg = signal[VecRange{W}(i)]
        acc += sum(Vec{W,Int32}(s) * Vec{W,Int32}(sg))
        i += W; st = carrier_advance(eng, st, 1)
    end
    acc
end
```

`carrier_engine(table, cycles_per_sample)` (or `; frequency, sampling_frequency`) builds
the engine; `carrier_lookup` returns the `(sin, cos)::Tuple{Vec{W,T},Vec{W,T}}` chunk at
the current phase (W = the backend's SIMD width) and `carrier_advance(eng, st, nchunks)`
steps the state forward. The fused loop above is **0 allocations** and avoids the carrier
store/reload entirely.

For a trivial consumer (e.g. filling an array) a single stream is latency-bound on its one
DDA carry chain. Interleave `K` independent streams instead — hold `K` states
`carrier_state(eng, (k-1)*W)` and advance each by `K` chunks per iteration — so the carry
chains overlap and it reaches the full loop rate (~40 ps/elem at scale, matching
`generate_carrier!`), allocation-free:

```julia
eng = carrier_engine(tbl, 0.002); W = carrier_width(eng)   # W = 64 on AVX-512
st0 = carrier_state(eng, 0);  st1 = carrier_state(eng, W)
st2 = carrier_state(eng, 2W); st3 = carrier_state(eng, 3W)
i = 1
for _ in 1:(length(sins) ÷ (4W))
    s0,c0 = carrier_lookup(eng, st0); s1,c1 = carrier_lookup(eng, st1)
    s2,c2 = carrier_lookup(eng, st2); s3,c3 = carrier_lookup(eng, st3)
    sins[VecRange{W}(i)]      = s0; coss[VecRange{W}(i)]      = c0
    sins[VecRange{W}(i+W)]    = s1; coss[VecRange{W}(i+W)]    = c1
    sins[VecRange{W}(i+2W)]   = s2; coss[VecRange{W}(i+2W)]   = c2
    sins[VecRange{W}(i+3W)]   = s3; coss[VecRange{W}(i+3W)]   = c3
    i += 4W
    st0 = carrier_advance(eng, st0, 4); st1 = carrier_advance(eng, st1, 4)
    st2 = carrier_advance(eng, st2, 4); st3 = carrier_advance(eng, st3, 4)
end
```

Rule of thumb: a single stream when fusing into nontrivial work (it borrows your loop's
ILP); a 4-way interleave (or `generate_carrier!`) when the per-sample work is light.

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
| **SinCosLUT** Int8, steps=64  | **13**  | 5.2e-2 | ~4  |
| **SinCosLUT** Int8, steps=128 | **13**  | 2.8e-2 | ~5  |
| FixedPoint Int16 (Val 7)      | 103     | 1.3e-2 | ~6  |
| FastSinCos `u100k`            | 187     | 3.2e-4 | ~12 |
| FixedPoint Int32 (Val 8)      | 206     | 8.1e-3 | ~7  |
| FastSinCos `u35`              | 226     | 6.0e-8 | ~24 |
| FixedPoint Int32 (Val 14)     | 293     | 1.1e-4 | ~13 |

**AVX2** (no native cross-lane byte permute — `vpermb` is unavailable; the lookup is
emulated with a four-way `vpshufb`+blend split)

| method | ps/elem | max abs error | ~bits |
| ------ | ------: | ------------: | ----: |
| **SinCosLUT** Int8, steps=64 | **43** | 5.2e-2 | ~4 |
| FixedPoint Int16 (Val 7) | 124 | 1.3e-2 | ~6  |
| FastSinCos `u100k`       | 251 | 3.2e-4 | ~12 |
| FixedPoint Int32 (Val 8) | 247 | 8.1e-3 | ~7  |
| FastSinCos `u35`         | 322 | 6.0e-8 | ~24 |

**End-to-end carrier** (phase generation + sincos, 0.01 cycles/sample). The kernel rows
above feed *pre-computed* phases; here each package generates the carrier itself. This is
where FixedPoint's multiplicative-inverse phase work shows — the kernel rows are
unaffected by it. SinCosLUT generates the phase with a UInt32 NCO accumulator (one add per
step, a single multiply to initialise), so its carrier stays close to its bare kernel
(≈26 vs ≈13 ps on AVX-512). All three are exact in phase: SinCosLUT's accumulator is
`n·freq_word mod 2^32` (integer, no rounding accumulation), and the FastSinCos row computes
the Float32 phase from the exact sample index (a plain `acc += step` float accumulator is a
little faster but drifts).

AVX-512:

| method | ps/elem | max abs error | ~bits |
| ------ | ------: | ------------: | ----: |
| **SinCosLUT** Int8, steps=64       | **26**  | 9.3e-2 | ~3  |
| **SinCosLUT** Int8, steps=128      | **26**  | 4.3e-2 | ~5  |
| FixedPoint Int16 (Val 7)           | 131     | 1.5e-2 | ~6  |
| FastSinCos `u100k` (Float32 phase) | 235     | 3.1e-4 | ~12 |
| FixedPoint Int32 (Val 13)          | 317     | 2.4e-4 | ~12 |

AVX2:

| method | ps/elem | max abs error | ~bits |
| ------ | ------: | ------------: | ----: |
| **SinCosLUT** Int8, steps=64       | **62**  | 9.3e-2 | ~3  |
| FixedPoint Int16 (Val 7)           | 142     | 1.5e-2 | ~6  |
| FastSinCos `u100k` (Float32 phase) | 326     | 3.1e-4 | ~12 |
| FixedPoint Int32 (Val 13)          | 383     | 2.4e-4 | ~12 |

FixedPoint's drift-free DDA carrier is only marginally slower than its bare kernel (131 vs
103 ps on AVX-512) — integer phase generation is nearly free. At ~6-bit accuracy
FixedPoint Int16 is the fastest *polynomial* carrier on both AVX-512 and AVX2; at
float-grade accuracy FastSinCos edges FixedPoint Int32. SinCosLUT is the fastest of all on
both AVX-512 and AVX2 — coarsest in accuracy, but the register-resident lookup beats every
polynomial on raw throughput. On AVX-512 the 128-entry table (two-register `vpermi2b`)
doubles phase resolution to ~5 bits at the same throughput as the 64-entry table, so it is
the better default there; AVX2's four-way `vpshufb` split supports only the 64-entry table.

Takeaways:

- **SinCosLUT is fastest on both AVX-512 and AVX2** (one `vpermb` per result on AVX-512;
  a four-way `vpshufb`+blend split on AVX2) — but only ~4–5 bits accurate, set by the
  table's phase resolution.
- **AVX2 is ~3× slower than AVX-512** here (half the lanes, plus the four-way emulation of
  the missing byte permute), but still the fastest option at its accuracy — it is no
  longer beaten by the polynomial packages.
- Pick by accuracy: **≤5-bit → SinCosLUT** (any of AVX-512/AVX2/NEON); **~6–13-bit integer
  → FixedPointSinCosApproximations**; **float-grade (12–24 bit) → FastSinCos**.

(Reproduce both tables with `julia benchmark/comparison.jl`.)

## Drift-free phase

`generate_carrier!` is a UInt32 NCO: the phase index for sample `n` is
`(freq_word · n) >> (32 − log2 steps)` with `freq_word = round(f/f_s · 2^32)`, evaluated in
exact integer arithmetic (the accumulator is `n · freq_word mod 2^32`). The phase therefore
accumulates **no** rounding error — even over billions of samples — unlike a floating-point
`acc += step` accumulator. The only residual is that the synthesised frequency is quantised
to the nearest `f_s / 2^32` (≈ 0.0006 Hz at 5 MHz), a uniform, dead-zone-free grid; that is
a fixed sub-milliHz frequency offset, not an accumulating drift.
