```@meta
CurrentModule = SinCosLUT
```

# API reference

```@docs
SinCosLUT
```

```@index
```

## Tables

```@docs
SinCosTable
```

## Carrier generation

```@docs
generate_carrier!
generate_carrier_signs!
cycles_per_sample
```

## Arbitrary-index lookup

```@docs
lookup_sincos!
prepare
```

## Value-based carrier engine

Allocation-free, register-resident NCO carrier for fusing into a correlation loop: a
loop-invariant [`CarrierEngine`](@ref) plus an isbits [`CarrierState`](@ref) renewed by
value each iteration. One engine/state pair serves any interleave factor `K` — hold `K`
states `carrier_state(eng, (k-1)*W)` and advance each by `K` chunks per iteration. See
[Fused, array-free generation](@ref) for the full pattern.

```@docs
CarrierEngine
CarrierState
carrier_engine
carrier_state
carrier_lookup
carrier_advance
carrier_width
```

## Backends

```@docs
default_backend
backend_name
```
