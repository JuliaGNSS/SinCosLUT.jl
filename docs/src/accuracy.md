```@meta
CurrentModule = SinCosLUT
```

# Accuracy & drift-free phase

SinCosLUT quantises in two independent dimensions. Understanding them tells you exactly
what to expect from the output.

## What is quantised

- **Phase** is quantised to `steps` entries per cycle. A 64-entry table gives ~6-bit
  phase resolution; the larger `2·regsize(T)` tables (via `vpermi2` on AVX-512) double
  it — Int8 → 128 entries, ~7-bit phase.
- **Amplitude** is quantised to the output type's range (`Int8` ≈ 7-bit, `Int16` ≈
  15-bit, `Int32` ≈ 31-bit).
- **No range limit**: any phase — positive, negative, or huge — is reduced *exactly* by
  the index bit-mask. There is no Cody–Waite-style ceiling and no precision loss at
  large arguments, unlike float approximations.

We can see the amplitude quantisation directly by comparing the normalised table output
to the true `sin`/`cos`:

```@example accuracy
using SinCosLUT

tbl   = SinCosTable(Int8; steps = 128)   # finest Int8 phase resolution
n     = 4096
sins  = zeros(Int8, n); coss = zeros(Int8, n)
cps   = 0.01
generate_carrier!(sins, coss, tbl, cps)

amp = Float64(typemax(Int8))
maxerr = maximum(1:n) do i
    φ = 2π * cps * (i - 1)
    max(abs(sins[i]/amp - sin(φ)), abs(coss[i]/amp - cos(φ)))
end
(maxerr, "≈ $(round(-log2(maxerr), digits=1)) bits")
```

For anything needing more than a few bits, a polynomial approximation is the better
tool — see [Choosing a package](@ref).

## Drift-free phase

[`generate_carrier!`](@ref) is a `UInt32` NCO (numerically-controlled oscillator). The
phase index for sample `n` is

```math
\left\lfloor \frac{\texttt{freq\_word} \cdot n}{2^{32 - \log_2 \texttt{steps}}} \right\rfloor,
\qquad \texttt{freq\_word} = \operatorname{round}\!\left(\frac{f}{f_s} \cdot 2^{32}\right)
```

evaluated in **exact integer arithmetic** — the accumulator is `freq_word·n mod 2^32`.
The phase therefore accumulates **no** rounding error, even over billions of samples,
unlike a floating-point `acc += step` accumulator that drifts a little more with every
add.

The only residual is that the synthesised frequency is quantised to the nearest
`f_s / 2^32` (≈ 0.0006 Hz at 5 MHz) — a uniform, dead-zone-free grid. That is a fixed
sub-milliHz frequency *offset*, not an accumulating drift.

The point is not that the accumulator hits some magic value, but that accumulating it
**incrementally** (one add per sample) gives the *same* answer as the exact closed form,
no matter how many samples you run. We can show that directly: replay `acc += step` for
`n` samples and compare to the closed form `step · n`, once with SinCosLUT's `UInt32`
frequency word and once with a naive `Float32` step.

```@example accuracy
cps       = 1234.5678 / 5_000_000              # cycles/sample, not nicely representable
freq_word = UInt32(round(Int64, cps * 4.294967296e9))
step_f32  = Float32(cps)

int_incremental(n) = (acc = UInt32(0); for _ in 1:n; acc += freq_word; end; acc)
f32_incremental(n) = (acc = 0.0f0;     for _ in 1:n; acc += step_f32;  end; acc)

int_closed(n) = freq_word * UInt32(n)          # exact, wraps mod 2^32
f32_closed(n) = step_f32 * Float32(n)

[(n = n,
  integer_drift        = Int(int_incremental(n)) - Int(int_closed(n)),   # accumulator units
  float32_drift_cycles = f32_incremental(n) - f32_closed(n))
 for n in (10^3, 10^5, 10^7)]
```

The integer NCO's `integer_drift` is exactly `0` at every horizon — the incremental sum
*is* the closed form, because `UInt32` addition is exact. The `Float32` accumulator, by
contrast, drifts further with every add: by `n = 10^7` it is off by tens of cycles.
That is the difference between a bounded frequency *offset* and an accumulating *drift*.

## Choosing a package

Three JuliaGNSS packages compute fast SIMD sin/cos at different points on the
speed/accuracy curve:

| accuracy needed | use |
| --------------- | --- |
| ≤ ~5-bit (very high throughput) | **SinCosLUT** (this package) |
| ~6–13-bit, integer output | [FixedPointSinCosApproximations.jl](https://github.com/JuliaGNSS/FixedPointSinCosApproximations.jl) |
| float-grade, 12–24-bit | [FastSinCos.jl](https://github.com/JuliaGNSS/FastSinCos.jl) |

The [Benchmarks](@ref) page quantifies the trade with measured throughput and error.
