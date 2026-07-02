```@meta
CurrentModule = SinCosLUT
```

# SinCosLUT.jl

Fast SIMD sine/cosine by **register-resident table lookup**. Each `sin`/`cos` is a
single hardware byte/word/dword permute over a table held entirely in vector
registers — no multiplies, no memory gathers, and (unlike floating-point range
reduction) **no input range limit**: the phase wraps for free via a bit-mask.

It is a deliberately *low-precision, very high-throughput* technique, built for GNSS
carrier generation and bit-wise software correlators. If you need more than a few bits
of accuracy, reach for a polynomial approximation instead
([FixedPointSinCosApproximations.jl](https://github.com/JuliaGNSS/FixedPointSinCosApproximations.jl)
or [FastSinCos.jl](https://github.com/JuliaGNSS/FastSinCos.jl)); see
[Choosing a package](@ref) and [Benchmarks](@ref).

## Installation

```julia
using Pkg
Pkg.add("SinCosLUT")
```

## The speed/precision trade

Pick the output element type to trade precision for speed. A 512-bit register holds
64×`Int8` but only 32×`Int16` / 16×`Int32`, so wider elements give more amplitude bits
but fewer table entries (coarser phase) and fewer lanes (lower throughput):

| output `T` | lanes (AVX-512) | table entries / cycle | ~accuracy |
| ---------- | --------------- | --------------------- | --------- |
| `Int8`     | 64              | 64 or 128             | ~3–7 bit  |
| `Int16`    | 32              | 32 or 64              | finer amplitude |
| `Int32`    | 16              | 16 or 32              | finest amplitude |

## Quick start

The examples on this page (and throughout the docs) are executed when the manual is
built, against the current source — if the API changes under them, the build fails, so
they cannot go stale.

```@example quickstart
using SinCosLUT

# 64-entry Int8 table (6-bit phase, full Int8 amplitude)
tbl = SinCosTable(Int8; steps = 64)

sins = zeros(Int8, 8)
coss = zeros(Int8, 8)

# Carrier of 0.05 cycles/sample, drift-free integer phase:
generate_carrier!(sins, coss, tbl, 0.05)
sins
```

```@example quickstart
coss
```

From here:

- **[Usage guide](@ref)** — carrier generation, frequency forms, initial phase,
  amplitude precision, backends, arbitrary-index lookup, and the 1-bit carrier.
- **[Fused, array-free generation](@ref)** — the allocation-free `carrier_engine` for
  fusing sin/cos straight into a correlation loop.
- **[Accuracy & drift-free phase](@ref)** — what the quantisation buys you and why the
  phase never drifts.
- **[Benchmarks](@ref)** — throughput and accuracy against the polynomial packages.
- **[API reference](@ref)** — every exported function.

!!! note "SVE2 is not supported"
    Julia cannot express LLVM scalable vectors (`<vscale x N x T>`), see
    [JuliaLang/julia#40308](https://github.com/JuliaLang/julia/issues/40308). NEON
    `tbl` runs natively on SVE hardware regardless.
