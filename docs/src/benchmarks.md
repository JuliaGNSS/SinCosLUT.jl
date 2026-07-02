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
| **SinCosLUT** Int8, steps=64  | **11.7** | 5.2e-2 | ~4  |
| **SinCosLUT** Int8, steps=128 | **13.9** | 2.8e-2 | ~5  |
| FixedPoint Int16 (Val 7)      | 105      | 1.3e-2 | ~6  |
| FastSinCos `u100k`            | 190      | 3.2e-4 | ~12 |
| FixedPoint Int32 (Val 8)      | 209      | 8.1e-3 | ~7  |
| FastSinCos `u35`              | 229      | 6.0e-8 | ~24 |
| FixedPoint Int32 (Val 14)     | 297      | 1.1e-4 | ~13 |

### AVX2

No native cross-lane byte permute (`vpermb` is unavailable); the lookup is a half-table
`vpshufb`+blend split plus a `psignb` sign flip (exploiting the table's half-wave
anti-symmetry).

| method | ps/elem | max abs error | ~bits |
| ------ | ------: | ------------: | ----: |
| **SinCosLUT** Int8, steps=64 | **21.2** | 5.2e-2 | ~4  |
| FixedPoint Int16 (Val 7)     | 123      | 1.3e-2 | ~6  |
| FixedPoint Int32 (Val 8)     | 250      | 8.1e-3 | ~7  |
| FastSinCos `u100k`           | 250      | 3.2e-4 | ~12 |
| FastSinCos `u35`             | 327      | 6.0e-8 | ~24 |

## End-to-end carrier

Phase generation **plus** sincos, at 0.01 cycles/sample — each package generates the
carrier itself. This is where FixedPoint's multiplicative-inverse phase work shows; the
kernel rows above are unaffected by it. SinCosLUT generates the phase with a `UInt32` NCO
accumulator (one add per step), so its carrier stays close to its bare kernel (≈17 vs
≈10 ps on AVX-512). All three are exact in phase.

### AVX-512

| method | ps/elem | max abs error | ~bits |
| ------ | ------: | ------------: | ----: |
| **SinCosLUT** Int8, steps=64       | **16.8** | 9.3e-2 | ~3  |
| **SinCosLUT** Int8, steps=128      | **17.7** | 4.3e-2 | ~5  |
| FixedPoint Int16 (Val 7)           | 131      | 1.5e-2 | ~6  |
| FastSinCos `u100k` (Float32 phase) | 237      | 3.1e-4 | ~12 |
| FixedPoint Int32 (Val 13)          | 317      | 2.4e-4 | ~12 |

### AVX2

| method | ps/elem | max abs error | ~bits |
| ------ | ------: | ------------: | ----: |
| **SinCosLUT** Int8, steps=64       | **47.8** | 9.3e-2 | ~3  |
| FixedPoint Int16 (Val 7)           | 144      | 1.5e-2 | ~6  |
| FastSinCos `u100k` (Float32 phase) | 328      | 3.1e-4 | ~12 |
| FixedPoint Int32 (Val 13)          | 387      | 2.4e-4 | ~12 |

## Takeaways

- **SinCosLUT is fastest on both AVX-512 and AVX2** — one `vpermb` per result on
  AVX-512, a half-table `vpshufb`+blend split with a `psignb` sign flip on AVX2 — but
  only ~4–5 bits accurate, set by the table's phase resolution.
- **AVX2 is ~2–3× slower than AVX-512** here (half the lanes, plus the emulation of the
  missing byte permute), yet still the fastest option at its accuracy by a wide margin —
  the half-table split (v3.2) roughly doubled its kernel throughput over the earlier
  four-way split.
- On AVX-512 the **128-entry table** (two-register `vpermi2b`) doubles phase resolution
  to ~5 bits for a ~20% kernel premium — and no carrier premium, where the NCO hides
  it — so it is the better default there when the extra phase bit matters; the AVX2
  split supports only the 64-entry table.
- FixedPoint's drift-free DDA carrier is only marginally slower than its bare kernel
  (131 vs 105 ps on AVX-512) — integer phase generation is nearly free. At ~6-bit
  accuracy FixedPoint Int16 is the fastest *polynomial* carrier; at float-grade accuracy
  FastSinCos edges FixedPoint Int32.
- **NEON** (not in the x86 tables above): on the CI Apple M1 runner the Int8 carrier
  runs at ~70 ps/elem, and the bytewise-`tbl` Int16 backend (v3.2) is ~2.2× faster than
  the scalar fallback it replaces (~220 vs ~510 ps/elem).
- **Pick by accuracy**: ≤5-bit → SinCosLUT (AVX-512/AVX2/NEON); ~6–13-bit integer →
  FixedPointSinCosApproximations; float-grade (12–24-bit) → FastSinCos. See
  [Choosing a package](@ref).

## Continuous benchmarking

A separate [AirspeedVelocity.jl](https://github.com/MilesCranmer/AirspeedVelocity.jl)
suite in `benchmark/benchmarks.jl` tracks throughput regressions across revisions
(`benchpkg SinCosLUT --rev=dirty`). It covers `generate_carrier!` for each element type
and buffer size, the 1-bit `generate_carrier_signs!` path, the 4-way interleaved fill,
the fused single-stream reduction, and `lookup_sincos!`.
