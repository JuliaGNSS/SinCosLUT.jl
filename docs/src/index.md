```@meta
CurrentModule = SinCosLUT
```

# SinCosLUT.jl

Fast SIMD sine/cosine by **register-resident table lookup**. Each `sin`/`cos` is a
single hardware byte/word/dword permute over a table held entirely in vector
registers (`vpermb`/`vpermw`/`vpermd` on AVX-512, `vpshufb` on AVX2, NEON `tbl` on
AArch64) — no multiplies, no memory gathers, and no input range limit (the phase
wraps for free via a bit-mask).

It is a deliberately *low-precision, very high-throughput* technique. Choose the
output element type to trade precision for speed:

| output `T` | lanes (AVX-512) | entries / cycle | ~accuracy |
| ---------- | --------------- | --------------- | --------- |
| `Int8`     | 64              | 64 or 128       | ~3–7 bit  |
| `Int16`    | 32              | 32 or 64        | finer amplitude |
| `Int32`    | 16              | 16 or 32        | finest amplitude |

See the [README](https://github.com/JuliaGNSS/SinCosLUT.jl) for backend details,
the accuracy/range discussion, and the drift-free phase generation.

!!! note "SVE2"
    SVE2 is not supported: Julia cannot express LLVM scalable vectors
    (`<vscale x N x T>`), see
    [JuliaLang/julia#40308](https://github.com/JuliaLang/julia/issues/40308). NEON
    `tbl` runs natively on SVE hardware regardless.

## Quick start

```julia
using SinCosLUT

tbl  = SinCosTable(Int8; steps = 64)
sins = zeros(Int8, 4096); coss = zeros(Int8, 4096)
generate_carrier!(sins, coss, tbl, 0.002)          # drift-free carrier, 0.002 cycles/sample
```

Array-free / fused (returns `Vec`s, like FastSinCos):

```julia
for (s, c) in CarrierIterator(tbl, 0.002, length(signal))   # s, c :: Vec{W,Int8}
    # consume in registers; no carrier array is allocated
end
```

## API

### Tables

```@docs
SinCosTable
```

### Carrier generation

```@docs
generate_carrier!
CarrierIterator
CarrierIterator4
cycles_per_sample
```

### Lookup primitives

```@docs
lookup_sincos!
prepare
```
