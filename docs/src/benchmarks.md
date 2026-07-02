```@meta
CurrentModule = SinCosLUT
```

# Benchmarks

These numbers compare SinCosLUT against the two polynomial JuliaGNSS packages — the
float minimax polynomial [FastSinCos.jl](https://github.com/JuliaGNSS/FastSinCos.jl) and
the integer minimax polynomial
[FixedPointSinCosApproximations.jl](https://github.com/JuliaGNSS/FixedPointSinCosApproximations.jl).
`ps/elem` is picoseconds per output element (computing **both** sin and cos);
`max abs error` is the worst-case absolute error vs the true values over several cycles;
`~bits` is `-log2(max error)`.

!!! note "Why these tables are not run during the docs build"
    Unlike the code examples elsewhere in this manual (which run on every build), the
    numbers below are *measurements* and depend on the host CPU — in particular on the
    presence of AVX-512 `vpermb`, which the shared CI runners that build these docs do
    not have. Running the suite there would produce slow, misleading figures. They are
    instead measured on fixed hardware and refreshed with each release. **Reproduce them
    on your own machine with** `julia benchmark/comparison.jl`.

Measured on an **AMD Ryzen AI 9 HX PRO 370** (Zen 5) with Julia 1.12. The `AVX2` blocks
force the AVX2-width path on the same hardware.

## Kernel throughput

Sin *and* cos for an array of **pre-computed** phase indices — the bare lookup/evaluate
cost, with no phase generation.

### AVX-512

| method | ps/elem | max abs error | ~bits |
| ------ | ------: | ------------: | ----: |
| **SinCosLUT** Int8, steps=128 | **10.1** | 2.8e-2 | ~5  |
| **SinCosLUT** Int8, steps=64  | **11.2** | 5.2e-2 | ~4  |
| FixedPoint Int16 (Val 7)      | 103      | 1.3e-2 | ~6  |
| FastSinCos `u100k`            | 187      | 3.2e-4 | ~12 |
| FixedPoint Int32 (Val 8)      | 207      | 8.1e-3 | ~7  |
| FastSinCos `u35`              | 229      | 6.0e-8 | ~24 |
| FixedPoint Int32 (Val 14)     | 293      | 1.1e-4 | ~13 |

### AVX2

No native cross-lane byte permute (`vpermb` is unavailable); the lookup is emulated with
a four-way `vpshufb`+blend split.

| method | ps/elem | max abs error | ~bits |
| ------ | ------: | ------------: | ----: |
| **SinCosLUT** Int8, steps=64 | **42.7** | 5.2e-2 | ~4  |
| FixedPoint Int16 (Val 7)     | 126      | 1.3e-2 | ~6  |
| FixedPoint Int32 (Val 8)     | 246      | 8.1e-3 | ~7  |
| FastSinCos `u100k`           | 254      | 3.2e-4 | ~12 |
| FastSinCos `u35`             | 336      | 6.0e-8 | ~24 |

## End-to-end carrier

Phase generation **plus** sincos, at 0.01 cycles/sample — each package generates the
carrier itself. This is where FixedPoint's multiplicative-inverse phase work shows; the
kernel rows above are unaffected by it. SinCosLUT generates the phase with a `UInt32` NCO
accumulator (one add per step), so its carrier stays close to its bare kernel (≈17 vs
≈10 ps on AVX-512). All three are exact in phase.

### AVX-512

| method | ps/elem | max abs error | ~bits |
| ------ | ------: | ------------: | ----: |
| **SinCosLUT** Int8, steps=128      | **17.3** | 4.3e-2 | ~5  |
| **SinCosLUT** Int8, steps=64       | **17.5** | 9.3e-2 | ~3  |
| FixedPoint Int16 (Val 7)           | 132      | 1.5e-2 | ~6  |
| FastSinCos `u100k` (Float32 phase) | 239      | 3.1e-4 | ~12 |
| FixedPoint Int32 (Val 13)          | 318      | 2.4e-4 | ~12 |

### AVX2

| method | ps/elem | max abs error | ~bits |
| ------ | ------: | ------------: | ----: |
| **SinCosLUT** Int8, steps=64       | **61.7** | 9.3e-2 | ~3  |
| FixedPoint Int16 (Val 7)           | 145      | 1.5e-2 | ~6  |
| FastSinCos `u100k` (Float32 phase) | 329      | 3.1e-4 | ~12 |
| FixedPoint Int32 (Val 13)          | 390      | 2.4e-4 | ~12 |

## Takeaways

- **SinCosLUT is fastest on both AVX-512 and AVX2** — one `vpermb` per result on
  AVX-512, a four-way `vpshufb`+blend split on AVX2 — but only ~4–5 bits accurate, set
  by the table's phase resolution.
- **AVX2 is ~3.5–4× slower than AVX-512** here (half the lanes, plus the four-way
  emulation of the missing byte permute), yet still the fastest option at its accuracy —
  it is no longer beaten by the polynomial packages.
- On AVX-512 the **128-entry table** (two-register `vpermi2b`) doubles phase resolution
  to ~5 bits at the same throughput as the 64-entry table, so it is the better default
  there; AVX2's four-way split supports only the 64-entry table.
- FixedPoint's drift-free DDA carrier is only marginally slower than its bare kernel
  (132 vs 103 ps on AVX-512) — integer phase generation is nearly free. At ~6-bit
  accuracy FixedPoint Int16 is the fastest *polynomial* carrier; at float-grade accuracy
  FastSinCos edges FixedPoint Int32.
- **Pick by accuracy**: ≤5-bit → SinCosLUT (AVX-512/AVX2/NEON); ~6–13-bit integer →
  FixedPointSinCosApproximations; float-grade (12–24-bit) → FastSinCos. See
  [Choosing a package](@ref).

## Continuous benchmarking

A separate [AirspeedVelocity.jl](https://github.com/MilesCranmer/AirspeedVelocity.jl)
suite in `benchmark/benchmarks.jl` tracks throughput regressions across revisions
(`benchpkg SinCosLUT --rev=dirty`). It covers `generate_carrier!` for each element type
and buffer size, the 1-bit `generate_carrier_signs!` path, the 4-way interleaved fill,
the fused single-stream reduction, and `lookup_sincos!`.
