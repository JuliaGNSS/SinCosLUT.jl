# Two-bit (sign + magnitude) carrier generation.
#
# A 2-bit carrier keeps the SIGN of sin/cos plus one MAGNITUDE bit per component — the
# standard 2-bit ("sign-magnitude") local carrier of GNSS hardware correlators (e.g. the
# GP2021's ±1/±2 octant waveform). The magnitude threshold sits on the octant boundary:
#
#   mag = 1  ⇔  |component| ≥ sin(π/4) = √2/2
#
# i.e. the magnitude bit is set exactly in the octants where that component is the larger
# of the two. Reconstructing `value = (sign ? -1 : 1) * (mag ? 2 : 1)` yields the classic
# eight-segment waveform sin ≈ {+1,+2,+2,+1,−1,−2,−2,−1} (the weight pair — ±1/±2, ±1/±3,
# … — is the consumer's choice; only the bits are produced here).
#
# Everything is read straight off the same UInt32 NCO as `generate_carrier_signs!`
# (`acc[n] = freq_word·n + offset` mod 2³²) — with this threshold all four bit-planes are
# exact functions of the accumulator's top THREE bits (the octant):
#
#   sign(sin) = MSB(acc)                              = acc₃₁
#   sign(cos) = MSB(acc + ¼·2³²)                      = acc₃₁ ⊕ acc₃₀
#   mag(sin)  = MSB(2·acc + ¼·2³²)                    = acc₃₀ ⊕ acc₂₉
#   mag(cos)  = MSB(2·(acc + ¼·2³²) + ¼·2³²)          = ¬ mag(sin)
#
# (adding the single bit ¼·2³² = 2³⁰ carries into bit 31 iff bit 30 is set, hence the
# XORs; |sin| ≥ √2/2 ⇔ φ mod ½ ∈ [⅛, ⅜) ⇔ MSB(2·acc + 2³⁰). And since with a 45°
# threshold exactly one of |sin|,|cos| exceeds it, the cos magnitude plane is the bitwise
# NOT of the sin magnitude plane — both are still written out for the consumer's
# convenience; the complement costs one scalar `~` per word.)
#
# The packing convention matches `generate_carrier_signs!`: bit `j` of word `w` ↔ sample
# `64w + j`; a set sign bit means NEGATIVE, a set magnitude bit means HIGH (|x| ≥ √2/2).
#
# Fast at any frequency via two per-FILL (not per-word) paths, split on the frequency word:
#   • fw < 2²⁴ — the magnitude square wave (which runs at TWICE the carrier frequency, so
#     it is always the first plane to flip: it needs 64·2·fw < 2³¹ where the sign planes
#     need 64·fw < 2³¹) can span whole 64-sample words. Per word: if all four planes are
#     constant, four plain stores; else one SIMD evaluation. Same shape as the 1-bit kernel.
#   • fw ≥ 2²⁴ — every word may contain a magnitude flip, so the constant-run test can
#     never win: run a branch-free SIMD loop instead (on x86 unrolled over independent NCO
#     streams, like the `generate_carrier!` kernel), so the per-word packs pipeline instead
#     of serialising on the word-at-a-time scalar bookkeeping.
# Both paths are O(1) per word no matter how many flips it contains.
#
# The per-word SIMD evaluation extracts the accumulator HIGH BYTE of all 64 lanes once —
# the `_phase_index` byte-gather on x86 (without its index-alignment tail, which the bit
# tests don't need), the `uzp2` chain on NEON — and the planes then fall out as byte
# adds/XORs of it per the identities above. No table and no permute beyond that gather;
# each plane costs one SIMD bit-extract:
#   • AVX-512: vpmovb2m / `vptestmb` (test one bit of each byte straight into a mask);
#   • AVX2: shift the bit into the byte MSB (adds) + `vpmovmskb`, 2 × 32-lane chunks;
#   • NEON: `cmlt`/`cmtst` + the positional-bitmask `addp` tree of `_sign_pack`;
#   • anywhere else: the generic `_msb_pack` UInt32 sign-mask (always correct).

# Empty-asm barrier pinning a UInt64 into a GPR. Used on the magnitude word before the
# `sin_mags`/`cos_mags` stores: without it LLVM computes `~mag` by re-running the vector
# bit test with an inverted predicate (a second vptestnmb per word on AVX-512, on the
# saturated vector pipes) instead of a 1-op scalar `not` of the value it already moved to
# a GPR for the store. Measured ≈15% on the whole fill (AVX-512, flip-dominated).
@inline _opaque_u64(x::UInt64) = Base.llvmcall(("""
    define i64 @entry(i64 %v) #0 {
      %r = call i64 asm "", "=r,0"(i64 %v)
      ret i64 %r }
    attributes #0 = { alwaysinline }""", "entry"), UInt64, Tuple{UInt64}, x)

# ---- per-word evaluation: (sin_signs, cos_signs, sin_mags) of one 64-sample word.
# cos_mags is NOT returned — it is the complement, applied wordwise by the callers.
#
# The ramp objects and word-quarter machinery are shared with the 1-bit kernel
# (`_bits_ramp`, `_hb64` — see signbits.jl); `_flip_words_sm(backend, base, ramp)`
# evaluates the word starting at accumulator `base`, `_sm_flip3(backend, acc)` from an
# already-built 64-lane accumulator.

@inline _flip_words_sm(backend::Backend, base::UInt32, ramp::Vec{64,UInt32}) =
    _sm_flip3(backend, Vec{64,UInt32}(base) + ramp)

# Generic fallback (Portable / anything else): three UInt32 sign-masks, always correct.
@inline function _sm_flip3(::Backend, acc::Vec{64,UInt32})
    (_msb_pack(acc), _msb_pack(acc + 0x40000000), _msb_pack(acc + acc + 0x40000000))
end

@static if Sys.ARCH in (:x86_64, :i686)
    # Pack one BIT of each byte into a UInt64: bit j set ⇔ v[j] & m[j] ≠ 0 (m is a
    # single-bit broadcast below, so this is a per-byte bit test). The and + icmp-ne +
    # bitcast pattern lowers to a single `vptestmb` + kmov on AVX-512BW — no shifting the
    # bit into the MSB first, which is what the AVX2 fallback below has to do.
    @inline _bit_pack(v::Vec{64,Int8}, m::Vec{64,Int8}) = Base.llvmcall(("""
        define i64 @entry(<64 x i8> %v, <64 x i8> %m) #0 {
          %a = and <64 x i8> %v, %m
          %c = icmp ne <64 x i8> %a, zeroinitializer
          %r = bitcast <64 x i1> %c to i64
          ret i64 %r }
        attributes #0 = { alwaysinline "target-features"="+avx512bw,+avx512f" }""", "entry"),
        UInt64, Tuple{NTuple{64,VecElement{Int8}},NTuple{64,VecElement{Int8}}}, v.data, m.data)

    # hb = the raw gathered accumulator high byte (bit 7 = acc₃₁, bit 6 = acc₃₀, bit 5 =
    # acc₂₉) — the `_phase_index` gather WITHOUT its index-alignment tail, which the bit
    # tests don't need. sin sign is its MSB (`_sign_pack` → vpmovb2m); cos sign and mag
    # are bits 7/6 of x = hb ⊕ (hb ≪ 1).
    @inline function _sm_flip3(::AVX512, acc::Vec{64,UInt32})
        hb = _msbyte64(acc)
        x = hb ⊻ (hb + hb)                             # bit k = hb[k] ⊕ hb[k-1]
        (_sign_pack(hb),                               # acc₃₁       = sign(sin)
         _sign_pack(x),                                # acc₃₁⊕acc₃₀ = sign(cos)
         _bit_pack(x, Vec{64,Int8}(0x40 % Int8)))      # acc₃₀⊕acc₂₉ = mag
    end

    # AVX2 has no per-byte bit-test-to-mask; move the wanted bit into the byte MSB with
    # adds (vpaddb is 1 µop; a <32 x i8> `shl` legalises to vpsllw+vpand) and vpmovmskb.
    # idx = top 6 accumulator bits, exact (idx₅ = acc₃₁, idx₄ = acc₃₀, idx₃ = acc₂₉).
    @inline function _sm_pack3_avx2(idx::Vec{32,Int8})
        t1 = idx + idx                            # idx ≪ 1
        x  = idx ⊻ t1
        x2 = x + x
        x4 = x2 + x2                              # x ≪ 2 : MSB = sign(cos)
        (_sign_pack(t1 + t1), _sign_pack(x4), _sign_pack(x4 + x4))
    end
    @inline function _sm_flip3(b::AVX2, acc::Vec{64,UInt32})
        i0 = _phase_index(b, _slice(acc, Val(0),  Val(32)), Val(64), Int8)
        i1 = _phase_index(b, _slice(acc, Val(32), Val(32)), Val(64), Int8)
        s0, c0, m0 = _sm_pack3_avx2(i0)
        s1, c1, m1 = _sm_pack3_avx2(i1)
        (s0 | (s1 << 32), c0 | (c1 << 32), m0 | (m1 << 32))
    end
end

@static if Sys.ARCH === :aarch64
    # `_sign_pack` with the sign compare replaced by a bit test: lane j contributes ⇔
    # v[j] & m[j] ≠ 0 (`cmtst` instead of `cmlt`), then the same positional-bitmask
    # `addp` tree (see signbits.jl for why the tree beats the generic bitcast lowering).
    @inline _bit_pack(v::Vec{64,Int8}, m::Vec{64,Int8}) = Base.llvmcall(("""
        declare <16 x i8> @llvm.aarch64.neon.addp.v16i8(<16 x i8>, <16 x i8>)
        define i64 @entry(<64 x i8> %v, <64 x i8> %m) #0 {
          %a = and <64 x i8> %v, %m
          %c = icmp ne <64 x i8> %a, zeroinitializer
          %s = sext <64 x i1> %c to <64 x i8>
          %v0 = shufflevector <64 x i8> %s, <64 x i8> undef, <16 x i32> <i32 0,i32 1,i32 2,i32 3,i32 4,i32 5,i32 6,i32 7,i32 8,i32 9,i32 10,i32 11,i32 12,i32 13,i32 14,i32 15>
          %v1 = shufflevector <64 x i8> %s, <64 x i8> undef, <16 x i32> <i32 16,i32 17,i32 18,i32 19,i32 20,i32 21,i32 22,i32 23,i32 24,i32 25,i32 26,i32 27,i32 28,i32 29,i32 30,i32 31>
          %v2 = shufflevector <64 x i8> %s, <64 x i8> undef, <16 x i32> <i32 32,i32 33,i32 34,i32 35,i32 36,i32 37,i32 38,i32 39,i32 40,i32 41,i32 42,i32 43,i32 44,i32 45,i32 46,i32 47>
          %v3 = shufflevector <64 x i8> %s, <64 x i8> undef, <16 x i32> <i32 48,i32 49,i32 50,i32 51,i32 52,i32 53,i32 54,i32 55,i32 56,i32 57,i32 58,i32 59,i32 60,i32 61,i32 62,i32 63>
          %bm0 = and <16 x i8> %v0, <i8 1,i8 2,i8 4,i8 8,i8 16,i8 32,i8 64,i8 -128,i8 1,i8 2,i8 4,i8 8,i8 16,i8 32,i8 64,i8 -128>
          %bm1 = and <16 x i8> %v1, <i8 1,i8 2,i8 4,i8 8,i8 16,i8 32,i8 64,i8 -128,i8 1,i8 2,i8 4,i8 8,i8 16,i8 32,i8 64,i8 -128>
          %bm2 = and <16 x i8> %v2, <i8 1,i8 2,i8 4,i8 8,i8 16,i8 32,i8 64,i8 -128,i8 1,i8 2,i8 4,i8 8,i8 16,i8 32,i8 64,i8 -128>
          %bm3 = and <16 x i8> %v3, <i8 1,i8 2,i8 4,i8 8,i8 16,i8 32,i8 64,i8 -128,i8 1,i8 2,i8 4,i8 8,i8 16,i8 32,i8 64,i8 -128>
          %p0 = call <16 x i8> @llvm.aarch64.neon.addp.v16i8(<16 x i8> %bm0, <16 x i8> %bm1)
          %p1 = call <16 x i8> @llvm.aarch64.neon.addp.v16i8(<16 x i8> %bm2, <16 x i8> %bm3)
          %p2 = call <16 x i8> @llvm.aarch64.neon.addp.v16i8(<16 x i8> %p0, <16 x i8> %p1)
          %p3 = call <16 x i8> @llvm.aarch64.neon.addp.v16i8(<16 x i8> %p2, <16 x i8> %p2)
          %r = bitcast <16 x i8> %p3 to <2 x i64>
          %lo = extractelement <2 x i64> %r, i32 0
          ret i64 %lo }
        attributes #0 = { alwaysinline "target-features"="+neon" }""", "entry"),
        UInt64, Tuple{NTuple{64,VecElement{Int8}},NTuple{64,VecElement{Int8}}}, v.data, m.data)

    # NEON works word-quarters (16 lanes) at a time via the shared `_hb64` (signbits.jl):
    # hb = accumulator high bytes (bit 7 = acc₃₁, bit 6 = acc₃₀, bit 5 = acc₂₉). Signs as
    # in the 1-bit NEON path: MSB(hb) and MSB(hb + 0x40) (the +¼-cycle carry). Magnitude:
    # y = hb + 0x20 has y₆ = acc₃₀ ⊕ acc₂₉ (the 0x20 carry into bit 6), read with a
    # single bit-6 test — no shift needed.
    @inline function _flip_words_sm(::Neon, base::UInt32, ramp::Tuple{Vec{16,UInt32},UInt32})
        r, q = ramp
        b1 = base + q; b2 = b1 + q; b3 = b2 + q
        _sm_flip3_neon(Vec{16,UInt32}(base) + r, Vec{16,UInt32}(b1) + r,
                       Vec{16,UInt32}(b2) + r, Vec{16,UInt32}(b3) + r)
    end
end

# ---- fw ≥ 2²⁴: branch-free SIMD loop over the full words, plus the masked partial tail.
# The generic (x86/portable) form keeps Vec{64,UInt32} NCO accumulators live and is
# unrolled over independent streams (hand-unrolled with plain locals, like kernel.jl — a
# tuple/closure formulation boxes): AVX-512 has 32 ZMM registers and a short per-word
# dependency chain, so 4 streams pipeline well; on AVX2 a 64-lane accumulator is already
# 8 of the 16 YMM registers, so it runs a single stream.
_sm_unroll(::Backend) = Val(1)
_sm_unroll(::AVX512)  = Val(4)

function _sm_fill_flips!(backend::Backend, sin_signs, cos_signs, sin_mags, cos_mags,
                         n::Int, nwords::Int, fw::UInt32, off::UInt32)
    rem = n - (nwords - 1) * 64                # valid lanes in the last word
    nfull = rem == 64 ? nwords : nwords - 1
    step = fw * UInt32(64)
    acc0 = Vec{64,UInt32}(off) + _iota_u32() * Vec{64,UInt32}(fw)
    acc = _sm_flips_loop!(sin_signs, cos_signs, sin_mags, cos_mags, nfull, acc0, step,
                          _sm_unroll(backend))
    @inbounds if nfull < nwords                # masked partial tail word
        m = _lowmask(rem)
        s, c, mg = _sm_flip3(backend, acc); mg = _opaque_u64(mg)
        sin_signs[nwords] = s & m; cos_signs[nwords] = c & m
        sin_mags[nwords] = mg & m; cos_mags[nwords] = ~mg & m
    end
end

function _sm_flips_loop!(sin_signs, cos_signs, sin_mags, cos_mags,
                         nfull::Int, acc1::Vec{64,UInt32}, step::UInt32, ::Val{4})
    step4 = step * UInt32(4)                    # wraps mod 2³², like the accumulator
    acc2 = acc1 + step; acc3 = acc2 + step; acc4 = acc3 + step
    w = 1
    @inbounds while w + 3 <= nfull
        s, c, m = _sm_flip3(_BITS_BACKEND, acc1); m = _opaque_u64(m)
        sin_signs[w] = s; cos_signs[w] = c; sin_mags[w] = m; cos_mags[w] = ~m
        s, c, m = _sm_flip3(_BITS_BACKEND, acc2); m = _opaque_u64(m)
        sin_signs[w+1] = s; cos_signs[w+1] = c; sin_mags[w+1] = m; cos_mags[w+1] = ~m
        s, c, m = _sm_flip3(_BITS_BACKEND, acc3); m = _opaque_u64(m)
        sin_signs[w+2] = s; cos_signs[w+2] = c; sin_mags[w+2] = m; cos_mags[w+2] = ~m
        s, c, m = _sm_flip3(_BITS_BACKEND, acc4); m = _opaque_u64(m)
        sin_signs[w+3] = s; cos_signs[w+3] = c; sin_mags[w+3] = m; cos_mags[w+3] = ~m
        acc1 += step4; acc2 += step4; acc3 += step4; acc4 += step4
        w += 4
    end
    @inbounds while w <= nfull                  # ≤ 3 leftover full words; acc1 tracks w
        s, c, m = _sm_flip3(_BITS_BACKEND, acc1); m = _opaque_u64(m)
        sin_signs[w] = s; cos_signs[w] = c; sin_mags[w] = m; cos_mags[w] = ~m
        acc1 += step; w += 1
    end
    acc1                                        # positioned at the (possibly partial) tail
end
function _sm_flips_loop!(sin_signs, cos_signs, sin_mags, cos_mags,
                         nfull::Int, acc1::Vec{64,UInt32}, step::UInt32, ::Val{1})
    @inbounds for w in 1:nfull
        s, c, m = _sm_flip3(_BITS_BACKEND, acc1); m = _opaque_u64(m)
        sin_signs[w] = s; cos_signs[w] = c; sin_mags[w] = m; cos_mags[w] = ~m
        acc1 += step
    end
    acc1
end

@static if Sys.ARCH === :aarch64
    # Planes of one word given its four quarter accumulators (16 lanes each — the
    # transient working set the register file can afford; see _flip_words_sm above).
    @inline function _sm_flip3_neon(a0::Vec{16,UInt32}, a1::Vec{16,UInt32},
                                    a2::Vec{16,UInt32}, a3::Vec{16,UInt32})
        hb = _hb64(a0, a1, a2, a3)
        hbu = reinterpret(Vec{64,UInt8}, hb)
        hbc = reinterpret(Vec{64,Int8}, hbu + UInt8(0x40))
        y   = reinterpret(Vec{64,Int8}, hbu + UInt8(0x20))
        (_sign_pack(hb), _sign_pack(hbc), _bit_pack(y, Vec{64,Int8}(0x40 % Int8)))
    end

    # NEON flip loop: one resident 16-lane accumulator (4 V regs) plus three broadcast
    # quarter offsets — each word's four quarter accumulators are transient, so nothing
    # spills (a resident 64-lane accumulator is 16 of the 32 V regs and forced ~26
    # spill loads/stores per word; measured ~12% slower even at unroll 1).
    function _sm_fill_flips!(backend::Neon, sin_signs, cos_signs, sin_mags, cos_mags,
                             n::Int, nwords::Int, fw::UInt32, off::UInt32)
        rem = n - (nwords - 1) * 64
        nfull = rem == 64 ? nwords : nwords - 1
        q = fw * UInt32(16)
        q1 = Vec{16,UInt32}(q); q2 = Vec{16,UInt32}(q + q); q3 = Vec{16,UInt32}(q + q + q)
        stepv = Vec{16,UInt32}(fw * UInt32(64))
        acc = Vec{16,UInt32}(off) + _lanes_u32(Val(16)) * fw
        @inbounds for w in 1:nfull
            s, c, m = _sm_flip3_neon(acc, acc + q1, acc + q2, acc + q3); m = _opaque_u64(m)
            sin_signs[w] = s; cos_signs[w] = c; sin_mags[w] = m; cos_mags[w] = ~m
            acc += stepv
        end
        @inbounds if nfull < nwords            # masked partial tail word
            m = _lowmask(rem)
            s, c, mg = _sm_flip3_neon(acc, acc + q1, acc + q2, acc + q3); mg = _opaque_u64(mg)
            sin_signs[nwords] = s & m; cos_signs[nwords] = c & m
            sin_mags[nwords] = mg & m; cos_mags[nwords] = ~mg & m
        end
    end
end

# ---- fw < 2²⁴: run-length regime, same word-at-a-time shape as the 1-bit kernel. Every
# word is guaranteed ≤ 1 flip per plane (64·2·fw < 2³¹), so a plane is constant across the
# word iff its accumulator MSB matches at the first and last lane.
function _sm_fill_runs!(backend::Backend, sin_signs, cos_signs, sin_mags, cos_mags,
                        n::Int, nwords::Int, fw::UInt32, off::UInt32)
    ramp  = _bits_ramp(backend, fw)
    step  = fw * UInt32(64)                    # accumulator advance per 64-sample word
    span  = fw * UInt32(63)                    # advance across the 63 lanes within a word
    span2 = span + span                        # same, on the doubled-phase magnitude NCO
    base = off
    @inbounds for w in 1:nwords
        m = _lowmask(w == nwords ? n - (w - 1) * 64 : 64)          # valid-lane mask (tail may be < 64)
        bc = base + 0x40000000                                      # cos accumulator (+¼ cycle)
        bm = base + base + 0x40000000                               # magnitude accumulator (2·acc + ¼ cycle)
        sin_const = _msb(base) == _msb(base + span)
        cos_const = _msb(bc)   == _msb(bc + span)
        mag_const = _msb(bm)   == _msb(bm + span2)
        if sin_const & cos_const & mag_const                        # all planes constant → four stores
            sin_signs[w] = (_msb(base) ? typemax(UInt64) : zero(UInt64)) & m
            cos_signs[w] = (_msb(bc)   ? typemax(UInt64) : zero(UInt64)) & m
            mw = _msb(bm) ? typemax(UInt64) : zero(UInt64)
            sin_mags[w] = mw & m; cos_mags[w] = ~mw & m
        else                                                        # a flip inside → one SIMD evaluation
            sw, cw, mw = _flip_words_sm(backend, base, ramp)
            mw = _opaque_u64(mw)
            sin_signs[w] = sw & m; cos_signs[w] = cw & m
            sin_mags[w]  = mw & m; cos_mags[w]  = ~mw & m
        end
        base += step
    end
end

# Whether the branch-free SIMD loop beats the constant-run loop even where runs exist.
# With FOUR planes a "constant" word still costs three run tests, a mask select, and four
# masked stores of word-at-a-time scalar bookkeeping — on AVX-512 that is MORE than the
# whole branch-free SIMD word (measured 23.1 vs 19.3 ps/element at a low fw on Zen 5), so
# there the flip loop runs unconditionally. Where SIMD word evaluation is expensive the
# run loop keeps its large low-frequency win (NEON: 105 vs 358 ps/element), and Portable
# (scalarised `_msb_pack`) depends on it entirely.
_sm_always_flips(::Backend) = false
_sm_always_flips(::AVX512)  = true

function _carrier_signs_mags!(sin_signs::AbstractVector{UInt64}, cos_signs::AbstractVector{UInt64},
                              sin_mags::AbstractVector{UInt64}, cos_mags::AbstractVector{UInt64},
                              n::Int, fw::UInt32, off::UInt32)
    nwords = cld(n, 64)
    # The magnitude plane flips at twice the carrier frequency, so IT sets the constant-run
    # guard: a word is flip-free only if 64·2·fw < 2³¹ ⇔ fw < 2²⁴. Test fw directly (NOT a
    # wrapped step, which loses the whole-cycle count for a fast carrier — see signbits.jl).
    if !_sm_always_flips(_BITS_BACKEND) && fw < 0x01000000
        _sm_fill_runs!(_BITS_BACKEND, sin_signs, cos_signs, sin_mags, cos_mags, n, nwords, fw, off)
    else                                       # every word may flip → branch-free SIMD loop
        _sm_fill_flips!(_BITS_BACKEND, sin_signs, cos_signs, sin_mags, cos_mags, n, nwords, fw, off)
    end
    sin_signs, cos_signs, sin_mags, cos_mags
end

"""
    generate_carrier_signs_mags!(sin_signs, cos_signs, sin_mags, cos_mags, n, freq_word::Integer;      phase=0)
    generate_carrier_signs_mags!(sin_signs, cos_signs, sin_mags, cos_mags, n, cycles_per_sample::Real; phase=0)
    generate_carrier_signs_mags!(sin_signs, cos_signs, sin_mags, cos_mags, n; frequency, sampling_frequency, phase=0)

Generate a **2-bit (sign + magnitude) carrier**: pack the sign and magnitude bits of the NCO
sin/cos for `n` samples into four bit-plane buffers (`UInt64`, each `≥ cld(n, 64)` words).
Bit `j` of word `w` corresponds to sample `64w + j` (0-based); a **set sign bit means that
component is negative**, a **set magnitude bit means it is large** — `|component| ≥ sin(π/4) =
√2/2`, the octant where that component is the larger of the two. Reconstructing
`value = (sign ? -1 : 1) * (mag ? 2 : 1)` gives the classic 2-bit octant waveform
`sin ≈ {+1,+2,+2,+1,−1,−2,−2,−1}` of hardware GNSS correlators (any weight pair — ±1/±2,
±1/±3, … — works; only the bits are produced). Bits past sample `n` in the last word are
cleared.

This is the same NCO as [`generate_carrier!`](@ref) and [`generate_carrier_signs!`](@ref) —
`acc[n] = freq_word·n + offset` mod `2³²` — and every bit is exact in the phase: `sign(sin) =
MSB(acc)`, `sign(cos) = MSB(acc + ¼·2³²)`, `mag(sin) = MSB(2·acc + ¼·2³²)`, and `mag(cos) =
!mag(sin)` (with the threshold on the 45° octant boundary exactly one component is large at a
time; both magnitude planes are still filled, for consumers that index them independently).
The frequency argument matches `generate_carrier!`: a raw `UInt32` `freq_word`, a
`cycles_per_sample::Real` (`freq_word = round(cps·2³²)`), or the `frequency`/`sampling_frequency`
keyword form. `phase` is the initial carrier phase in **cycles** (a `Real`; default 0).

Fast at any frequency, like the 1-bit kernel: at low frequencies words where all four planes
are constant (the magnitude square wave runs at twice the carrier frequency, so its runs are
half as long) are written with four plain stores; at high frequencies the fill runs a
branch-free SIMD loop — O(1) per word either way, no per-sample loop.

```julia
n = 5000
planes = [Vector{UInt64}(undef, cld(n, 64)) for _ in 1:4]
generate_carrier_signs_mags!(planes..., n; frequency = 1234, sampling_frequency = 5e6)
```
"""
function generate_carrier_signs_mags!(sin_signs::AbstractVector{UInt64}, cos_signs::AbstractVector{UInt64},
                                      sin_mags::AbstractVector{UInt64}, cos_mags::AbstractVector{UInt64},
                                      n::Integer, freq_word::Integer; phase::Real = 0)
    nwords = cld(Int(n), 64)
    (length(sin_signs) ≥ nwords && length(cos_signs) ≥ nwords &&
     length(sin_mags)  ≥ nwords && length(cos_mags)  ≥ nwords) ||
        throw(DimensionMismatch("bit-plane buffers need ≥ cld(n, 64) = $nwords words"))
    (0 ≤ freq_word ≤ typemax(UInt32)) ||
        throw(ArgumentError("need 0 ≤ freq_word ≤ typemax(UInt32) = $(typemax(UInt32))"))
    _carrier_signs_mags!(sin_signs, cos_signs, sin_mags, cos_mags,
                         Int(n), UInt32(freq_word), _freq_word(phase))
end
function generate_carrier_signs_mags!(sin_signs::AbstractVector{UInt64}, cos_signs::AbstractVector{UInt64},
                                      sin_mags::AbstractVector{UInt64}, cos_mags::AbstractVector{UInt64},
                                      n::Integer, cycles_per_sample::Real; kw...)
    generate_carrier_signs_mags!(sin_signs, cos_signs, sin_mags, cos_mags, n,
                                 _freq_word(cycles_per_sample); kw...)
end
function generate_carrier_signs_mags!(sin_signs::AbstractVector{UInt64}, cos_signs::AbstractVector{UInt64},
                                      sin_mags::AbstractVector{UInt64}, cos_mags::AbstractVector{UInt64},
                                      n::Integer; frequency::Real, sampling_frequency::Real, kw...)
    generate_carrier_signs_mags!(sin_signs, cos_signs, sin_mags, cos_mags, n,
                                 cycles_per_sample(frequency, sampling_frequency); kw...)
end
