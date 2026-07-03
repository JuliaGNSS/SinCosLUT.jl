```@meta
CurrentModule = SinCosLUT
```

# Usage guide

Every code block below is run when the docs are built, so the printed outputs are the
real values produced by the current version.

## Generating a carrier

[`generate_carrier!`](@ref) fills a pair of pre-allocated `sin`/`cos` buffers with a
carrier at a given normalised frequency (cycles per sample). The output element type is
the table's element type.

```@example guide
using SinCosLUT

tbl  = SinCosTable(Int8; steps = 64)   # 64-entry Int8 table
sins = zeros(Int8, 16)
coss = zeros(Int8, 16)

generate_carrier!(sins, coss, tbl, 0.05)   # 0.05 cycles/sample
sins
```

## Specifying the frequency

The frequency can be given three equivalent ways. Use whichever matches the numbers you
already have:

```@example guide
sins = zeros(Int8, 16); coss = zeros(Int8, 16)

# 1. Directly as cycles/sample (= frequency / sampling_frequency):
generate_carrier!(sins, coss, tbl, 0.0005)

# 2. From a frequency and sampling frequency (keyword form):
generate_carrier!(sins, coss, tbl; frequency = 1000, sampling_frequency = 2e6)  # 1 kHz at 2 MHz

# 3. Via the cycles_per_sample helper:
generate_carrier!(sins, coss, tbl, cycles_per_sample(1000, 2e6))                # → 0.0005
nothing # hide
```

Any frequency resolves on a uniform `sampling_frequency / 2^32` grid (≈ 0.001 Hz at
5 MHz) — arbitrary Doppler works with no dead-zone near DC and never errors:

```@example guide
generate_carrier!(sins, coss, tbl; frequency = 1234.567, sampling_frequency = 5e6)
nothing # hide
```

See [Accuracy & drift-free phase](@ref) for what that grid means for precision.

## Initial phase

The `phase` keyword sets the starting carrier phase (default `0`). An `Integer` is
counted in **table steps**; a `Real` is counted in **cycles** — so on a 64-step table,
`16` steps and `0.25` cycles are the same quarter cycle:

```@example guide
s1 = zeros(Int8, 8); c1 = zeros(Int8, 8)
s2 = zeros(Int8, 8); c2 = zeros(Int8, 8)
generate_carrier!(s1, c1, tbl, 0.05; phase = 16)     # 16 table steps  (¼ cycle)
generate_carrier!(s2, c2, tbl, 0.05; phase = 0.25)   # 0.25 cycles = 90°, same result
(s1 == s2, c1 == c2)
```

## Higher amplitude precision

Switch the table's element type to `Int16` or `Int32` for finer amplitude quantisation
(at the cost of throughput and phase resolution — see [The speed/precision
trade](@ref)):

```@example guide
tbl16 = SinCosTable(Int16; steps = 64)   # 6-bit phase, 15-bit amplitude (vpermi2w on AVX-512)
s16 = zeros(Int16, 8); c16 = zeros(Int16, 8)
generate_carrier!(s16, c16, tbl16, 0.05)
s16
```

## Looking up arbitrary phase indices

If you already have integer phase indices (not a swept carrier), [`lookup_sincos!`](@ref)
maps each index (taken mod `steps`) to its table entry:

```@example guide
phases = Int8[0, 8, 16, 24, 32, 40, 48, 56]   # eighths of a cycle on a 64-step table
s = zeros(Int8, 8); c = zeros(Int8, 8)
lookup_sincos!(s, c, phases, tbl)
s
```

## Choosing a backend

The lookup primitive is picked automatically from the host CPU:

| backend  | instruction        | types        | notes |
| -------- | ------------------ | ------------ | ----- |
| AVX-512  | `vpermb`/`vpermw`/`vpermd` (+`vpermi2*`) | Int8/16/32 | fastest |
| AVX2     | `vpshufb` + blends | Int8 only    | 64-entry table, half-table split + sign flip |
| NEON     | `tbl` (`tbl4`)     | Int8, Int16  | AArch64; Int16 is looked up bytewise |
| portable | scalar             | Int8/16/32   | always available |

**AVX2 and NEON have only a *byte* table permute** (`vpshufb` / `tbl`) — the word/dword
permutes (`vpermw`/`vpermd`) are AVX-512. On AVX2 the SIMD lookup therefore exists for
`Int8` (`steps = 64`) only. On NEON, `Int16` (`steps = 32` or `64`) is additionally
served by looking the table up *bytewise* (each word index becomes its little-endian
byte pair). `Int32` — and `Int16` on AVX2 — fall back to the (correct, but scalar)
**portable** backend; at those accuracies, prefer a polynomial package over a wider
SinCosLUT element type on such hosts.

You can query the choice, or force one (e.g. for testing, or to skip the runtime CPU
check):

```@example guide
using SinCosLUT: AVX512, AVX2, Portable

backend_name(default_backend(Int8, 64))   # what will run here
```

!!! note "Restricted or multiversioned CPU targets"
    The x86 ISA backends are chosen only when the LLVM codegen target is `native` (the
    default). Under a restricted or multiversioned CPU target — `julia --cpu-target=…`, or
    `JULIA_CPU_TARGET=…`, which is how the official binaries build every pkgimage — the
    CPU-detected features need not match what LLVM can actually emit, so `default_backend`
    falls back to `Portable()` for correctness (emitting an ISA the target cannot legalise
    would abort codegen with an uncatchable `LLVM ERROR`). Set `JULIA_CPU_TARGET=native`
    to keep the AVX-512/AVX2 backends under a custom target.

The results are backend-independent — forcing the scalar path produces exactly the same
values, just computed without SIMD:

```@example guide
s_auto = zeros(Int8, 32); c_auto = zeros(Int8, 32)
s_port = zeros(Int8, 32); c_port = zeros(Int8, 32)
generate_carrier!(s_auto, c_auto, tbl, 0.05)
generate_carrier!(s_port, c_port, tbl, 0.05; backend = Portable())
(s_auto == s_port, c_auto == c_port)
```

## One-bit (hard-limited) carrier

For bit-wise ("bit-sliced") software correlators the carrier is quantised to a single
sign bit, so wipe-off becomes XOR and accumulation becomes popcount.
[`generate_carrier_signs!`](@ref) produces that directly — no table, no output-type
quantisation — by reading the sign straight off the same `UInt32` NCO
(`sign(sin) = MSB(acc)`, `sign(cos) = MSB(acc + ¼ cycle)`) and packing it into `UInt64`
words. Bit `j` of word `w` corresponds to sample `64w + j`; a **set bit means that
component is negative**.

```@example guide
n = 200
sin_signs = Vector{UInt64}(undef, cld(n, 64))
cos_signs = Vector{UInt64}(undef, cld(n, 64))
# same frequency forms as generate_carrier!
generate_carrier_signs!(sin_signs, cos_signs, n; frequency = 1234, sampling_frequency = 5e6)

# bit 0 of word 1 is the sign of sample 0:
first_sin_negative = (sin_signs[1] & 1) != 0
```

A 1-bit carrier is a square wave, so almost every 64-sample word is a single constant
run written with one store; only words straddling a sign flip are filled with a single
SIMD sign-mask. It therefore stays fast at any frequency.
