```@meta
CurrentModule = SinCosLUT
```

# Fused, array-free generation

[`generate_carrier!`](@ref) is the right tool when you actually want a carrier array.
But if you are going to *consume* the carrier immediately — correlating it against a
signal, say — materialising it first wastes a store and the cache traffic that comes
with it. The value-based engine lets you pull `(sin, cos)` chunks straight into
registers and never touch memory.

Build a loop-invariant [`CarrierEngine`](@ref) once with [`carrier_engine`](@ref), then
drive it with an isbits [`CarrierState`](@ref) that you renew *by value* each iteration.
The drift-free NCO lives in the state, so the loop allocates nothing and nothing escapes
to the heap. It returns `SIMD.Vec`s, the same shape as `FastSinCos`.

## A fused correlation loop

```@example fused
using SinCosLUT, SIMD

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

tbl    = SinCosTable(Int8; steps = 64)
signal = rand(Int8, 4096)
correlate(tbl, 0.002, signal)
```

- [`carrier_engine(table, cycles_per_sample)`](@ref carrier_engine) (or `; frequency,
  sampling_frequency`) builds the engine.
- [`carrier_lookup`](@ref) returns the `(sin, cos)::Tuple{Vec{W,T},Vec{W,T}}` chunk at
  the current phase — `W` is the backend's SIMD width. It is a pure read.
- [`carrier_advance`](@ref)`(eng, st, nchunks)` steps the state forward by `nchunks`
  `W`-wide chunks, returning a new state.
- [`carrier_width`](@ref) is `W`.

The loop above allocates nothing beyond its inputs:

```@example fused
correlate(tbl, 0.002, signal)      # warm up / compile
@allocated correlate(tbl, 0.002, signal)
```

## Interleaving for full throughput

A single stream is latency-bound on its one NCO carry chain. When the per-sample work
is light (e.g. just filling an array), interleave `K` independent streams so the carry
chains overlap and the loop reaches its full rate. Hold `K` states positioned at
`carrier_state(eng, (k-1)*W)` and advance each by `K` chunks per iteration:

```@example fused
function fill4!(sins, coss, tbl, cps)
    eng = carrier_engine(tbl, cps); W = carrier_width(eng)
    st0 = carrier_state(eng, 0);  st1 = carrier_state(eng, W)
    st2 = carrier_state(eng, 2W); st3 = carrier_state(eng, 3W)
    i = 1
    @inbounds for _ in 1:(length(sins) ÷ (4W))
        s0,c0 = carrier_lookup(eng, st0); s1,c1 = carrier_lookup(eng, st1)
        s2,c2 = carrier_lookup(eng, st2); s3,c3 = carrier_lookup(eng, st3)
        sins[VecRange{W}(i)]    = s0; coss[VecRange{W}(i)]    = c0
        sins[VecRange{W}(i+W)]  = s1; coss[VecRange{W}(i+W)]  = c1
        sins[VecRange{W}(i+2W)] = s2; coss[VecRange{W}(i+2W)] = c2
        sins[VecRange{W}(i+3W)] = s3; coss[VecRange{W}(i+3W)] = c3
        i += 4W
        st0 = carrier_advance(eng, st0, 4); st1 = carrier_advance(eng, st1, 4)
        st2 = carrier_advance(eng, st2, 4); st3 = carrier_advance(eng, st3, 4)
    end
    sins, coss
end

W = carrier_width(carrier_engine(tbl, 0.002))
sins = zeros(Int8, 8W); coss = zeros(Int8, 8W)
fill4!(sins, coss, tbl, 0.002)
sins[1:8]
```

At scale this matches `generate_carrier!` (which is itself a 4-way interleave
internally).

!!! tip "How many streams?"
    Use a **single stream** when you fuse into nontrivial work — it borrows the ILP of
    your own loop. Use a **4-way interleave** (or just `generate_carrier!`) when the
    per-sample work is light.

## The stateless primitive

For the FastSinCos-style primitive where *you* supply the phase indices, [`prepare`](@ref)
builds the register-resident table once and returns a callable. It maps a `Vec` of phase
indices to `(sin, cos)` `Vec`s, so the input must be the backend's SIMD width `W` (as it
would be inside a `VecRange` loop):

```@example fused
p = prepare(tbl)                                   # build table in registers once
W = carrier_width(carrier_engine(tbl, 0.0))        # the backend's SIMD width
idx  = Vec{W,Int8}(ntuple(j -> Int8((j - 1) & 63), W))
s, c = p(idx)                                      # idx -> (sin, cos), like fast_sincos_*(::Vec)
(s[1], c[1])                                       # phase index 0 → (sin, cos) = (0, 127)
```
