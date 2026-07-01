# One-bit (hard-limited) carrier generation.
#
# A 1-bit carrier keeps only the SIGN of sin/cos — the natural input to a bit-wise
# ("bit-sliced") software correlator, where the carrier wipe-off collapses to XOR and the
# accumulate to popcount. There is no table lookup here: the sign of the NCO output is an
# exact function of the phase accumulator's top bit, so we read it straight off the same
# UInt32 NCO that `generate_carrier!` uses (`acc[n] = freq_word·n + offset` mod 2³²):
#
#   sin(2π·φ) < 0  ⇔  φ mod 1 ∈ [½, 1)      ⇔  MSB(acc)              is set
#   cos(2π·φ) < 0  ⇔  φ mod 1 ∈ (¼, ¾)      ⇔  MSB(acc + ¼ cycle)    is set
#
# The signs are packed into `UInt64` words (bit `j` of word `w` ↔ sample `64w+j`), a set
# bit meaning that component is NEGATIVE (the ±1 hard-limited value is −1).
#
# Fast at ANY frequency via two per-word paths:
#   • constant run — when the whole 64-sample word has one sign (a low residual carrier is a
#     square wave with runs of thousands of samples), write it in ONE store;
#   • otherwise — a sign flip falls inside the word: extract the top bit of all 64 phase
#     accumulators at once with a single SIMD sign-mask (`vpmovd2m` / equivalent), O(1) per
#     word no matter how many flips it contains — no per-sample loop.

@inline _msb(x::UInt32) = (x & 0x80000000) != zero(UInt32)
# Low-`rem`-bits mask (all ones for a full 64-lane word). Top-level, not a local closure —
# a local closure here boxes on Julia 1.10 (≈32 B/call).
@inline _lowmask(rem::Int) = rem == 64 ? typemax(UInt64) : (UInt64(1) << rem) - UInt64(1)

# Lane ramp [0,1,…,63] as a compile-time-constant Vec (folded to a literal).
@inline @generated _iota_u32() = :(Vec{64,UInt32}($(Expr(:tuple, (UInt32(j) for j in 0:63)...))))

# Pack the sign bit (MSB) of 64 lanes into a `UInt64`: bit j set ⇔ lane j is negative as a signed
# integer. Portable LLVM (an `icmp` + `<64 x i1>→i64` bitcast); lowers to `vpmov{d,b}2m`+kmov on
# AVX-512, a compare/movemask sequence on AVX2/NEON, and scalar elsewhere — always correct.
@inline _msb_pack(v::Vec{64,UInt32}) = Base.llvmcall(("""
    define i64 @entry(<64 x i32> %v) #0 {
      %c = icmp slt <64 x i32> %v, zeroinitializer
      %m = bitcast <64 x i1> %c to i64
      ret i64 %m }
    attributes #0 = { alwaysinline }""", "entry"),
    UInt64, Tuple{NTuple{64,Base.VecElement{Int32}}}, reinterpret(Vec{64,Int32}, v).data)
@inline _sign_pack(v::Vec{64,Int8}) = Base.llvmcall(("""
    define i64 @entry(<64 x i8> %v) #0 {
      %c = icmp slt <64 x i8> %v, zeroinitializer
      %m = bitcast <64 x i1> %c to i64
      ret i64 %m }
    attributes #0 = { alwaysinline }""", "entry"),
    UInt64, Tuple{NTuple{64,Base.VecElement{Int8}}}, v.data)

# Quadrant-sign table (N=64): entry k is the SIGN of sin/cos in quadrant k, ±1 as Int8. Permuting
# it by the phase index recovers the exact MSB sign (the sign depends only on the index's top bit,
# which is the accumulator MSB — no zero-crossing rounding as a value table would have). The cos
# column bakes in the ¼-cycle offset, so one index feeds both. `_SIGN_PREP` materialises it in
# registers once (default backend const-folds). This is only USED on the AVX-512 (W=64) path below;
# on other widths the `_msb_pack` sign-mask is used, so the prepared table just sits unused there.
const _SIGN_TABLE = SinCosTable{Int8,64}(
    ntuple(k -> (k - 1) < 32 ? Int8(1) : Int8(-1), Val(64)),                    # sin<0 ⇔ k∈[32,64)
    ntuple(k -> (16 ≤ (k - 1) < 48) ? Int8(-1) : Int8(1), Val(64)))            # cos<0 ⇔ k∈[16,48)
const _SIGN_PREP = prepare(_SIGN_TABLE)

# Sign words (sin, cos) of one full 64-sample word that contains ≥ 1 flip. The SIMD backends use
# the package's own cheap index (byte-gather) + a permute of the ±1 quadrant table (both columns) +
# an Int8 sign-mask per SIMD chunk — the carrier's own permute machinery, at whatever width the
# backend runs (AVX-512: one 64-lane chunk; AVX2: 2×32; NEON: 4×16). Portable / any other backend
# uses the generic UInt32 sign-mask. All bit-identical (they all read the accumulator's top bit).
@inline _flip_words(base::UInt32, ramp::Vec{64,UInt32}) = _flip_words(_SIGN_PREP, base, ramp)

# Extract lanes [o, o+W) of a Vec{64} as a Vec{W}.
@inline _slice(v::Vec{64,T}, ::Val{o}, ::Val{W}) where {T,o,W} =
    shufflevector(v, Val(ntuple(i -> i - 1 + o, Val(W))))

@inline function _flip_words(prep::Prepared{Int8,64,<:AVX512}, base::UInt32, ramp::Vec{64,UInt32})
    idx = _phase_index(prep.backend, Vec{64,UInt32}(base) + ramp, Val(64), Int8)
    sv, cv = prep(idx)                                                          # quadrant ±1 (sin, cos)
    (_sign_pack(sv), _sign_pack(cv))
end
@inline _flip_words(::Prepared, base::UInt32, ramp::Vec{64,UInt32}) =           # Portable / fallback
    (_msb_pack(Vec{64,UInt32}(base) + ramp),
     _msb_pack(Vec{64,UInt32}(base + 0x40000000) + ramp))

# One W-lane chunk at accumulator offset `ramp_chunk`: (sin_bits, cos_bits) as UInt64 in lanes [0,W).
@inline function _flip_chunk(prep, backend, base::UInt32, ramp_chunk::Vec{W,UInt32}) where {W}
    s, c = prep(_phase_index(backend, Vec{W,UInt32}(base) + ramp_chunk, Val(64), Int8))
    (_sign_pack(s), _sign_pack(c))
end

@static if Sys.ARCH in (:x86_64, :i686)
    @inline _sign_pack(v::Vec{32,Int8}) = UInt64(Base.llvmcall(("""
        define i32 @entry(<32 x i8> %v) #0 {
          %c = icmp slt <32 x i8> %v, zeroinitializer
          %m = bitcast <32 x i1> %c to i32
          ret i32 %m }
        attributes #0 = { alwaysinline }""", "entry"),
        UInt32, Tuple{NTuple{32,Base.VecElement{Int8}}}, v.data))
    @inline function _flip_words(prep::Prepared{Int8,64,<:AVX2}, base::UInt32, ramp::Vec{64,UInt32})
        s0, c0 = _flip_chunk(prep, prep.backend, base, _slice(ramp, Val(0),  Val(32)))
        s1, c1 = _flip_chunk(prep, prep.backend, base, _slice(ramp, Val(32), Val(32)))
        (s0 | (s1 << 32), c0 | (c1 << 32))
    end
end

@static if Sys.ARCH === :aarch64
    @inline _sign_pack(v::Vec{16,Int8}) = UInt64(Base.llvmcall(("""
        define i16 @entry(<16 x i8> %v) #0 {
          %c = icmp slt <16 x i8> %v, zeroinitializer
          %m = bitcast <16 x i1> %c to i16
          ret i16 %m }
        attributes #0 = { alwaysinline }""", "entry"),
        UInt16, Tuple{NTuple{16,Base.VecElement{Int8}}}, v.data))
    @inline function _flip_words(prep::Prepared{Int8,64,<:Neon}, base::UInt32, ramp::Vec{64,UInt32})
        s0, c0 = _flip_chunk(prep, prep.backend, base, _slice(ramp, Val(0),  Val(16)))
        s1, c1 = _flip_chunk(prep, prep.backend, base, _slice(ramp, Val(16), Val(16)))
        s2, c2 = _flip_chunk(prep, prep.backend, base, _slice(ramp, Val(32), Val(16)))
        s3, c3 = _flip_chunk(prep, prep.backend, base, _slice(ramp, Val(48), Val(16)))
        (s0 | (s1 << 16) | (s2 << 32) | (s3 << 48),
         c0 | (c1 << 16) | (c2 << 32) | (c3 << 48))
    end
end

function _carrier_signs!(sin_signs::AbstractVector{UInt64}, cos_signs::AbstractVector{UInt64},
                         n::Int, fw::UInt32, off::UInt32)
    nwords = cld(n, 64)
    ramp = _iota_u32() * Vec{64,UInt32}(fw)    # lane j = j·freq_word (accumulator offsets within a word)
    step = fw * UInt32(64)                     # accumulator advance per 64-sample word (wraps mod 2³²)
    span = fw * UInt32(63)                     # advance across the 63 lanes within a word
    # ≤ 1 sign flip per word ⇔ the word advances < ½ cycle ⇔ 64·fw < 2³¹ ⇔ fw < 2²⁵. Test fw
    # directly (NOT the wrapped `step`, which loses the whole-cycle count for a fast carrier).
    single_flip = fw < 0x02000000
    base = off                                 # accumulator at sample 0 (lane 0) = off
    @inbounds for w in 1:nwords
        m = _lowmask(w == nwords ? n - (w - 1) * 64 : 64)              # valid-lane mask (tail may be < 64)
        bc = base + 0x40000000                                          # cos accumulator (+¼ cycle)
        sin_const = single_flip && _msb(base) == _msb(base + span)
        cos_const = single_flip && _msb(bc)   == _msb(bc + span)
        if sin_const & cos_const                                        # both constant → two stores
            sin_signs[w] = (_msb(base) ? typemax(UInt64) : zero(UInt64)) & m
            cos_signs[w] = (_msb(bc)   ? typemax(UInt64) : zero(UInt64)) & m
        else                                                            # a flip inside → one lookup
            sw, cw = _flip_words(base, ramp)
            sin_signs[w] = sw & m; cos_signs[w] = cw & m
        end
        base += step
    end
    sin_signs, cos_signs
end

"""
    generate_carrier_signs!(sin_signs, cos_signs, n, freq_word::Integer;      phase=0)
    generate_carrier_signs!(sin_signs, cos_signs, n, cycles_per_sample::Real; phase=0)
    generate_carrier_signs!(sin_signs, cos_signs, n; frequency, sampling_frequency, phase=0)

Generate a **1-bit (hard-limited) carrier**: pack the sign bits of the NCO sin/cos for `n`
samples into `sin_signs`/`cos_signs` (`UInt64`, each `≥ cld(n, 64)` words). Bit `j` of word
`w` corresponds to sample `64w + j` (0-based); a **set bit means that component is negative**
(i.e. its ±1 hard-limited value is `−1`). Bits past sample `n` in the last word are cleared.

This is the same NCO as [`generate_carrier!`](@ref) — a `UInt32` phase accumulator advanced by
a fixed frequency word each sample (`acc[n] = freq_word·n + offset` mod `2³²`) — but only the
sign is kept, read directly off the accumulator's top bit (`sign(sin) = MSB(acc)`,
`sign(cos) = MSB(acc + ¼ cycle)`), so there is no table and no output-type quantisation. The
frequency argument matches `generate_carrier!`: a raw `UInt32` `freq_word`, a
`cycles_per_sample::Real` (`freq_word = round(cps·2³²)`), or the `frequency`/`sampling_frequency`
keyword form. `phase` is the initial carrier phase in **cycles** (a `Real`; default 0).

Fast at any frequency: a 1-bit carrier is a square wave, so for a low residual carrier almost
every 64-sample word is a single constant run written with one store; where a sign flip falls
inside a word, the 64 signs are packed with a single SIMD sign-mask (no per-sample loop). This
is the carrier form a bit-wise correlator consumes, where wipe-off becomes XOR and accumulation
becomes popcount.

```julia
n = 5000
sin_signs = Vector{UInt64}(undef, cld(n, 64))
cos_signs = Vector{UInt64}(undef, cld(n, 64))
generate_carrier_signs!(sin_signs, cos_signs, n; frequency = 1234, sampling_frequency = 5e6)
```
"""
function generate_carrier_signs!(sin_signs::AbstractVector{UInt64}, cos_signs::AbstractVector{UInt64},
                                 n::Integer, freq_word::Integer; phase::Real = 0)
    nwords = cld(Int(n), 64)
    (length(sin_signs) ≥ nwords && length(cos_signs) ≥ nwords) ||
        throw(DimensionMismatch("sign buffers need ≥ cld(n, 64) = $nwords words"))
    (0 ≤ freq_word ≤ typemax(UInt32)) ||
        throw(ArgumentError("need 0 ≤ freq_word ≤ typemax(UInt32) = $(typemax(UInt32))"))
    _carrier_signs!(sin_signs, cos_signs, Int(n), UInt32(freq_word), _freq_word(phase))
end
function generate_carrier_signs!(sin_signs::AbstractVector{UInt64}, cos_signs::AbstractVector{UInt64},
                                 n::Integer, cycles_per_sample::Real; kw...)
    generate_carrier_signs!(sin_signs, cos_signs, n, _freq_word(cycles_per_sample); kw...)
end
function generate_carrier_signs!(sin_signs::AbstractVector{UInt64}, cos_signs::AbstractVector{UInt64},
                                 n::Integer; frequency::Real, sampling_frequency::Real, kw...)
    generate_carrier_signs!(sin_signs, cos_signs, n,
                            cycles_per_sample(frequency, sampling_frequency); kw...)
end
