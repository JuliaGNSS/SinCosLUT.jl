# Changelog

## [3.2.2](https://github.com/JuliaGNSS/SinCosLUT.jl/compare/v3.2.1...v3.2.2) (2026-07-07)


### Performance Improvements

* **neon:** addp-tree sign-mask in _sign_pack on aarch64 ([812e667](https://github.com/JuliaGNSS/SinCosLUT.jl/commit/812e6672fa13fe83b5e8860ad928de5dd352dd40))

## [3.2.1](https://github.com/JuliaGNSS/SinCosLUT.jl/compare/v3.2.0...v3.2.1) (2026-07-07)


### Performance Improvements

* **neon:** read the sign MSB directly, skipping the ±1-table permute ([090df84](https://github.com/JuliaGNSS/SinCosLUT.jl/commit/090df844f9e90200ca95b47d446d5765cfc908f3))

# [3.2.0](https://github.com/JuliaGNSS/SinCosLUT.jl/compare/v3.1.1...v3.2.0) (2026-07-03)


### Bug Fixes

* **avx512:** gate ISA backends on the codegen target, not CPUID ([#19](https://github.com/JuliaGNSS/SinCosLUT.jl/issues/19)) ([30ec08c](https://github.com/JuliaGNSS/SinCosLUT.jl/commit/30ec08c7b731b11ea91f060350adabcef9198119))


### Features

* warn once when a restricted CPU target demotes a SIMD-capable host ([f31c9e0](https://github.com/JuliaGNSS/SinCosLUT.jl/commit/f31c9e0f9e2dd83e539edc2bd7e111577961be24))

## [3.1.1](https://github.com/JuliaGNSS/SinCosLUT.jl/compare/v3.1.0...v3.1.1) (2026-07-02)


### Bug Fixes

* **avx2:** keep the value-based paths type-stable on Julia < 1.12 ([8a3241f](https://github.com/JuliaGNSS/SinCosLUT.jl/commit/8a3241f9372af42b8ae06b82381c5e70de9d246a))


### Performance Improvements

* **avx2:** half-table psignb lookup, packed index extraction, 1-stream fill ([70f94a5](https://github.com/JuliaGNSS/SinCosLUT.jl/commit/70f94a56de72390949cd766e096a1d54432d08cc))
* **avx512:** cheaper phase-index extraction (ternlog merge, vendor-tuned align, Int16 word-gather) ([15c59ae](https://github.com/JuliaGNSS/SinCosLUT.jl/commit/15c59ae176c46eb129399b2d54d8496e33ec5a11))
* feed permutes directly, skipping the Prepared functor's re-mask ([428fe7e](https://github.com/JuliaGNSS/SinCosLUT.jl/commit/428fe7e206cbc8c42abc7b851dccccaed56c0ddb))
* **neon:** Int16 NEON backend via byte-pair tbl, exact byte-shift index ([57d94c5](https://github.com/JuliaGNSS/SinCosLUT.jl/commit/57d94c5b6a0732a243867fc97b3fb8700395d059))
* **neon:** uzp2-chain phase-index extraction instead of tbl gathers ([134b3f1](https://github.com/JuliaGNSS/SinCosLUT.jl/commit/134b3f13a3aa48fa24e6788ed96ab32e1fe20932))

# [3.1.0](https://github.com/JuliaGNSS/SinCosLUT.jl/compare/v3.0.1...v3.1.0) (2026-07-01)


### Features

* 1-bit carrier generation (generate_carrier_signs!) ([381c4ba](https://github.com/JuliaGNSS/SinCosLUT.jl/commit/381c4bab33dd9cfe9a76495c0c1591c4d4d969ed)), closes [hi#frequency](https://github.com/hi/issues/frequency)


### Performance Improvements

* AVX2 and NEON fast paths for the sign-flip lookup ([6b081da](https://github.com/JuliaGNSS/SinCosLUT.jl/commit/6b081daf9c9614457ab5fb7da670b16d7d1c0e23))
* LUT-backed sign lookup for the sign-flip path (uniformly fastest) ([327ce00](https://github.com/JuliaGNSS/SinCosLUT.jl/commit/327ce0017e9c9ba56d26e132e71f1b1f45740323))
* pack sign flips with a SIMD sign-mask — fast at any frequency ([88b6456](https://github.com/JuliaGNSS/SinCosLUT.jl/commit/88b6456555626a06fed9f14ccbe42546705a9d49)), closes [hi#frequency](https://github.com/hi/issues/frequency)

## [3.0.1](https://github.com/JuliaGNSS/SinCosLUT.jl/compare/v3.0.0...v3.0.1) (2026-06-26)


### Performance Improvements

* **avx512:** byte-gather the carrier NCO index (7→3 shuffle-port µops) ([df07e68](https://github.com/JuliaGNSS/SinCosLUT.jl/commit/df07e68d543427fb7256396ce897ce2609d9eff9))
* **neon:** byte-gather the carrier NCO index via tbl4 ([e76826e](https://github.com/JuliaGNSS/SinCosLUT.jl/commit/e76826e965a5a8b3f5bce4973bb35d8e158ae34a))

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
