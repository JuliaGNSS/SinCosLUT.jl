# Changelog

# [3.0.0](https://github.com/JuliaGNSS/SinCosLUT.jl/compare/v2.0.0...v3.0.0) (2026-06-26)


* feat!: value-based CarrierEngine/CarrierState replacing the iterators ([939714f](https://github.com/JuliaGNSS/SinCosLUT.jl/commit/939714f57b68d50df3c8504ca1fdf2ed78693bc2))


### BREAKING CHANGES

* `CarrierIterator` and `CarrierIterator4` are removed. Replace
`for (s,c) in CarrierIterator(table, fw, n)` with a `carrier_engine` + `carrier_state`
+ `carrier_lookup`/`carrier_advance` loop (see the iterate.jl module docstring).

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>

# [2.0.0](https://github.com/JuliaGNSS/SinCosLUT.jl/compare/v1.1.0...v2.0.0) (2026-06-25)


* refactor!: rename generate_carrier/4 -> CarrierIterator/CarrierIterator4 ([329088d](https://github.com/JuliaGNSS/SinCosLUT.jl/commit/329088d4c77c8229049bc2bdb3fe88b9ad9ae924))


### Performance Improvements

* single-stream SIMD tail in the carrier kernel ([c791907](https://github.com/JuliaGNSS/SinCosLUT.jl/commit/c7919079ecdb05fa4c453476adafc6436d311577))


### BREAKING CHANGES

* generate_carrier and generate_carrier4 are renamed to the constructors
CarrierIterator and CarrierIterator4. Replace `generate_carrier(table, …)` with
`CarrierIterator(table, …)` and `generate_carrier4(…)` with `CarrierIterator4(…)`.
`generate_carrier!` is unaffected.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>

# [1.1.0](https://github.com/JuliaGNSS/SinCosLUT.jl/compare/v1.0.1...v1.1.0) (2026-06-24)


### Features

* NCO phase-accumulator carrier — fine, dead-zone-free Doppler ([c2c1ffd](https://github.com/JuliaGNSS/SinCosLUT.jl/commit/c2c1ffd1462752713f5bc954927192a5211c1ae3))

## [1.0.1](https://github.com/JuliaGNSS/SinCosLUT.jl/compare/v1.0.0...v1.0.1) (2026-06-24)


### Performance Improvements

* SIMD-ize Int8 DDA init via vector mul-inverse ([c2c7558](https://github.com/JuliaGNSS/SinCosLUT.jl/commit/c2c75584152b0a69bbf0c501cd270eb45c6a6bce))
