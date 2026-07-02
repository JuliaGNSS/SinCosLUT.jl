# SinCosLUT.jl

[![CI](https://github.com/JuliaGNSS/SinCosLUT.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/JuliaGNSS/SinCosLUT.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/JuliaGNSS/SinCosLUT.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JuliaGNSS/SinCosLUT.jl)
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaGNSS.github.io/SinCosLUT.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaGNSS.github.io/SinCosLUT.jl/dev)

Fast SIMD sine/cosine by **register-resident table lookup**. Each `sin`/`cos` is a
single hardware byte/word/dword permute over a table held entirely in vector registers —
no multiplies, no memory gathers, and (unlike floating-point range reduction) **no input
range limit**: the phase wraps for free via a bit-mask.

It is a deliberately *low-precision, very high-throughput* technique for GNSS carrier
generation and bit-wise software correlators. Pick the output element type (`Int8` /
`Int16` / `Int32`) to trade precision for speed. On a Zen 5 core it generates a carrier
at **~17 ps/element** (AVX-512) — several times faster than the polynomial alternatives
(see the [benchmarks](https://JuliaGNSS.github.io/SinCosLUT.jl/dev/benchmarks/)).

## Installation

```julia
using Pkg
Pkg.add("SinCosLUT")
```

## Quick start

```julia
using SinCosLUT

tbl  = SinCosTable(Int8; steps = 64)          # 64-entry Int8 table
sins = zeros(Int8, 4096); coss = zeros(Int8, 4096)

# Drift-free carrier at 0.002 cycles/sample (= frequency / sampling_frequency):
generate_carrier!(sins, coss, tbl, 0.002)

# ...or straight from a frequency and sampling frequency:
generate_carrier!(sins, coss, tbl; frequency = 1000, sampling_frequency = 2e6)
```

## Documentation

Full, always-runnable documentation lives at
**[JuliaGNSS.github.io/SinCosLUT.jl/dev](https://JuliaGNSS.github.io/SinCosLUT.jl/dev)**
(every example is executed and checked when the docs are built):

- [**Usage guide**](https://JuliaGNSS.github.io/SinCosLUT.jl/dev/guide/) — carrier
  generation, frequency forms, initial phase, amplitude precision, backends,
  arbitrary-index lookup, and the 1-bit (hard-limited) carrier.
- [**Fused, array-free generation**](https://JuliaGNSS.github.io/SinCosLUT.jl/dev/fused/) —
  the allocation-free `carrier_engine` for fusing sin/cos straight into a correlation loop.
- [**Accuracy & drift-free phase**](https://JuliaGNSS.github.io/SinCosLUT.jl/dev/accuracy/) —
  what the quantisation buys you and why the integer NCO phase never drifts.
- [**Benchmarks**](https://JuliaGNSS.github.io/SinCosLUT.jl/dev/benchmarks/) — measured
  throughput and accuracy vs the polynomial packages.
- [**API reference**](https://JuliaGNSS.github.io/SinCosLUT.jl/dev/api/).

## Choosing a package

Three JuliaGNSS packages compute fast SIMD sin/cos at different points on the
speed/accuracy curve:

| accuracy needed | use |
| --------------- | --- |
| ≤ ~5-bit (very high throughput) | **SinCosLUT** (this package) |
| ~6–13-bit, integer output | [FixedPointSinCosApproximations.jl](https://github.com/JuliaGNSS/FixedPointSinCosApproximations.jl) |
| float-grade, 12–24-bit | [FastSinCos.jl](https://github.com/JuliaGNSS/FastSinCos.jl) |

> **Note** — the SIMD lookup runs on AVX-512 (all types), AVX2 (`Int8`), and NEON
> (`Int8` and `Int16`); other type/CPU combinations fall back to a correct scalar path.
> SVE2 is not supported. See the [usage guide](https://JuliaGNSS.github.io/SinCosLUT.jl/dev/guide/#Choosing-a-backend).
