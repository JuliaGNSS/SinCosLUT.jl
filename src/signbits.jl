# One-bit (hard-limited) carrier generation.
#
# A 1-bit carrier keeps only the SIGN of sin/cos — the natural input to a bit-wise
# ("bit-sliced") software correlator, where the carrier wipe-off collapses to XOR and the
# accumulate to popcount. There is no table lookup here: the sign of the NCO output is an
# exact function of the phase accumulator's top bits, so we read it straight off the same
# UInt32 NCO that `generate_carrier!` uses (`acc[n] = freq_word·n + offset` mod 2³²):
#
#   sin(2π·φ) < 0  ⇔  φ mod 1 ∈ [½, 1)      ⇔  MSB(acc)              is set  = acc₃₁
#   cos(2π·φ) < 0  ⇔  φ mod 1 ∈ (¼, ¾)      ⇔  MSB(acc + ¼ cycle)    is set  = acc₃₁ ⊕ acc₃₀
#
# (adding the single bit ¼·2³² = 2³⁰ carries into bit 31 iff bit 30 is set, hence the XOR.)
# The signs are packed into `UInt64` words (bit `j` of word `w` ↔ sample `64w+j`), a set
# bit meaning that component is NEGATIVE (the ±1 hard-limited value is −1).
#
# Fast at ANY frequency via two per-FILL paths, split on the frequency word:
#   • fw < 2²⁵ — every 64-sample word has ≤ 1 flip per component (64·fw < 2³¹), so a low
#     residual carrier is runs of constant words written in ONE store each; a word that
#     straddles a flip gets one SIMD evaluation.
#   • fw ≥ 2²⁵ — a flip can fall in every word, so the constant-run test can never win:
#     run a branch-free SIMD loop instead (on x86 unrolled over independent NCO streams,
#     like the `generate_carrier!` kernel), so the per-word packs pipeline instead of
#     serialising on the word-at-a-time scalar bookkeeping.
# Both are O(1) per word no matter how many flips it contains — no per-sample loop.
#
# The per-word SIMD evaluation extracts the accumulator HIGH BYTE of all 64 lanes once
# (bit 7 = acc₃₁, bit 6 = acc₃₀): the `_phase_index` byte-gather on x86 (without its
# index-alignment tail, which the bit tests don't need), the `uzp2` chain on NEON —
# evaluated in 16-lane quarters there, since a register-resident 64-lane accumulator is
# 16 of the 32 V registers and spills. Each component then costs ONE SIMD sign-mask:
# sin = MSB(hb), cos = MSB(hb ⊕ (hb ≪ 1)) (or MSB(hb + 0x40), same carry) — no table,
# no permute. The Portable fallback packs the UInt32 MSBs directly (`_msb_pack`).
# The 2-bit carrier (twobit.jl) builds on the same helpers, adding a magnitude plane.

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
# `_sign_pack(::Vec{64,Int8})`: pack bit 7 of 64 bytes into a UInt64. Same op, two lowerings.
# On x86/portable the `icmp` + `bitcast <64 x i1> to i64` is optimal (→ `vpmovb2m`+kmov on
# AVX-512, `vpmovmskb` on AVX2). On aarch64 that bitcast lowers to a per-16-lane `cmlt` + `and`
# + `addv` + GPR-move whose serial `addv`→GPR moves dominate; the aarch64 override below reduces
# with an `addp` (vpaddq) tree instead (one GPR move, no `addv`). Both bit-identical.
@static if Sys.ARCH !== :aarch64
    @inline _sign_pack(v::Vec{64,Int8}) = Base.llvmcall(("""
        define i64 @entry(<64 x i8> %v) #0 {
          %c = icmp slt <64 x i8> %v, zeroinitializer
          %m = bitcast <64 x i1> %c to i64
          ret i64 %m }
        attributes #0 = { alwaysinline }""", "entry"),
        UInt64, Tuple{NTuple{64,Base.VecElement{Int8}}}, v.data)
end

const _BITS_BACKEND = default_backend(Int8, 64)   # const → the dispatch below folds statically

# Extract lanes [o, o+W) of a Vec{64} as a Vec{W}.
@inline _slice(v::Vec{64,T}, ::Val{o}, ::Val{W}) where {T,o,W} =
    shufflevector(v, Val(ntuple(i -> i - 1 + o, Val(W))))

# ---- per-word evaluation: (sin_signs, cos_signs) of one 64-sample word.
# `_bits_ramp(backend, fw)` builds the backend's per-word lane-offset object once per
# fill; `_flip_words(backend, base, ramp)` evaluates the word at accumulator `base`;
# `_flip2(backend, acc)` evaluates from an already-built 64-lane accumulator (the
# branch-free x86 loops keep those live). NEON's quarter evaluation has no 64-lane
# accumulator form — see its `_flip_words` below.

@inline _bits_ramp(::Backend, fw::UInt32) = _iota_u32() * Vec{64,UInt32}(fw)
@inline _flip_words(backend::Backend, base::UInt32, ramp::Vec{64,UInt32}) =
    _flip2(backend, Vec{64,UInt32}(base) + ramp)

# Generic fallback (Portable / anything else): two UInt32 sign-masks, always correct.
@inline _flip2(::Backend, acc::Vec{64,UInt32}) =
    (_msb_pack(acc), _msb_pack(acc + 0x40000000))

# The sign planes are combined from RAW accumulator-bit planes: pack plane(acc₃₁) and
# plane(acc₃₀) with one SIMD bit-extract each, then sign(cos) = sign(sin) ⊕ plane(acc₃₀)
# as a 64-bit SCALAR xor — the scalar pipes idle next to the saturated vector pipes, so
# the xor is free and no byte add/xor is spent shaping a cos byte first. The 2-bit kernel
# (twobit.jl) extends the same scheme with plane(acc₂₉) for its magnitude.

@static if Sys.ARCH in (:x86_64, :i686)
    # Pack one BIT of each byte into a UInt64: bit j set ⇔ v[j] & m[j] ≠ 0 (m is a
    # single-bit broadcast, so this is a per-byte bit test). The and + icmp-ne + bitcast
    # pattern lowers to a single `vptestmb` + kmov on AVX-512BW — no shifting the bit
    # into the MSB first, which is what the AVX2 fallback below has to do.
    @inline _bit_pack(v::Vec{64,Int8}, m::Vec{64,Int8}) = Base.llvmcall(("""
        define i64 @entry(<64 x i8> %v, <64 x i8> %m) #0 {
          %a = and <64 x i8> %v, %m
          %c = icmp ne <64 x i8> %a, zeroinitializer
          %r = bitcast <64 x i1> %c to i64
          ret i64 %r }
        attributes #0 = { alwaysinline "target-features"="+avx512bw,+avx512f" }""", "entry"),
        UInt64, Tuple{NTuple{64,VecElement{Int8}},NTuple{64,VecElement{Int8}}}, v.data, m.data)

    # hb = the raw gathered accumulator high byte — the `_phase_index` gather WITHOUT its
    # index-alignment tail (bit 7 = acc₃₁, bit 6 = acc₃₀). One vpmovb2m + one vptestmb.
    @inline function _flip2(::AVX512, acc::Vec{64,UInt32})
        hb = _msbyte64(acc)
        s = _sign_pack(hb)
        (s, s ⊻ _bit_pack(hb, Vec{64,Int8}(0x40 % Int8)))
    end

    # AVX2: vpmovmskb reads the byte MSB, so move the wanted bits there with adds
    # (vpaddb is 1 µop; a <32 x i8> `shl` legalises to vpsllw+vpand), 2 × 32-lane chunks.
    # idx = top 6 accumulator bits, exact (idx₅ = acc₃₁, idx₄ = acc₃₀).
    @inline function _sb_pack2(idx::Vec{32,Int8})
        t2 = (idx + idx) + (idx + idx)               # idx ≪ 2 : MSB = acc₃₁
        (_sign_pack(t2), _sign_pack(t2 + t2))        # (plane(acc₃₁), plane(acc₃₀))
    end
    @inline function _flip2(b::AVX2, acc::Vec{64,UInt32})
        s0, a0 = _sb_pack2(_phase_index(b, _slice(acc, Val(0),  Val(32)), Val(64), Int8))
        s1, a1 = _sb_pack2(_phase_index(b, _slice(acc, Val(32), Val(32)), Val(64), Int8))
        s = s0 | (s1 << 32)
        (s, s ⊻ (a0 | (a1 << 32)))
    end

    @inline _sign_pack(v::Vec{32,Int8}) = UInt64(Base.llvmcall(("""
        define i32 @entry(<32 x i8> %v) #0 {
          %c = icmp slt <32 x i8> %v, zeroinitializer
          %m = bitcast <32 x i1> %c to i32
          ret i32 %m }
        attributes #0 = { alwaysinline }""", "entry"),
        UInt32, Tuple{NTuple{32,Base.VecElement{Int8}}}, v.data))
end

@static if Sys.ARCH === :aarch64
    # aarch64 `_sign_pack`: isolate each lane's sign bit with the {1,2,…,128} bitmask, then a
    # 4-deep `addp` (vpaddq) tree — one GPR move, no `addv` (see the x86/portable form above).
    # Measured ≈2.4× faster than the `bitcast` lowering on Cortex-A78AE. Bit-identical.
    @inline _sign_pack(v::Vec{64,Int8}) = Base.llvmcall(("""
        declare <16 x i8> @llvm.aarch64.neon.addp.v16i8(<16 x i8>, <16 x i8>)
        define i64 @entry(<64 x i8> %v) #0 {
          %c = icmp slt <64 x i8> %v, zeroinitializer
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
        attributes #0 = { alwaysinline }""", "entry"),
        UInt64, Tuple{NTuple{64,Base.VecElement{Int8}}}, v.data)

    # NEON works word-quarters (16 lanes) at a time: only a 16-lane ramp/accumulator stays
    # register-resident (4 V regs), each quarter's `uzp2` chain is transient. The 64-lane
    # concatenation is free (it only names the four 16-byte registers as one value).
    @inline _bits_ramp(::Neon, fw::UInt32) = (_lanes_u32(Val(16)) * fw, fw * UInt32(16))
    @inline _hb_quarter(acc::Vec{16,UInt32}) =                      # high byte of 16 lanes
        shufflevector(reinterpret(Vec{32,Int8}, _high_words(acc)), Val(ntuple(i -> 2i - 1, Val(16))))
    @inline _cat16(a::Vec{16,Int8}, b::Vec{16,Int8}) = shufflevector(a, b, Val(ntuple(i -> i - 1, Val(32))))
    @inline _cat32(a::Vec{32,Int8}, b::Vec{32,Int8}) = shufflevector(a, b, Val(ntuple(i -> i - 1, Val(64))))
    @inline _hb64(a0::Vec{16,UInt32}, a1::Vec{16,UInt32}, a2::Vec{16,UInt32}, a3::Vec{16,UInt32}) =
        _cat32(_cat16(_hb_quarter(a0), _hb_quarter(a1)), _cat16(_hb_quarter(a2), _hb_quarter(a3)))

    # Pack bit `b` of 64 bytes into a UInt64 via `sshl`: each lane's shift amount
    # (j mod 8) − b moves bit b straight onto the lane's POSITIONAL bit (arithmetic
    # right-shift smear only touches bits above position b, which the positional mask
    # discards), so one sshl + one and per register feed the same `addp` tree as
    # `_sign_pack` — no compare. (An and + icmp-ne formulation is NOT selected to
    # `cmtst`; LLVM legalises it as and + cmeq + bic, one op more per register.)
    @inline @generated _shift_vec(::Val{b}) where {b} =
        :(Vec{64,Int8}($(Expr(:tuple, (Int8(j % 8 - b) for j in 0:63)...))))
    @inline _shift_pack(v::Vec{64,Int8}, ::Val{b}) where {b} = _shift_pack_sshl(v, _shift_vec(Val(b)))
    @inline _shift_pack_sshl(v::Vec{64,Int8}, sh::Vec{64,Int8}) = Base.llvmcall(("""
        declare <16 x i8> @llvm.aarch64.neon.sshl.v16i8(<16 x i8>, <16 x i8>)
        declare <16 x i8> @llvm.aarch64.neon.addp.v16i8(<16 x i8>, <16 x i8>)
        define i64 @entry(<64 x i8> %v, <64 x i8> %sh) #0 {
          %v0 = shufflevector <64 x i8> %v, <64 x i8> undef, <16 x i32> <i32 0,i32 1,i32 2,i32 3,i32 4,i32 5,i32 6,i32 7,i32 8,i32 9,i32 10,i32 11,i32 12,i32 13,i32 14,i32 15>
          %v1 = shufflevector <64 x i8> %v, <64 x i8> undef, <16 x i32> <i32 16,i32 17,i32 18,i32 19,i32 20,i32 21,i32 22,i32 23,i32 24,i32 25,i32 26,i32 27,i32 28,i32 29,i32 30,i32 31>
          %v2 = shufflevector <64 x i8> %v, <64 x i8> undef, <16 x i32> <i32 32,i32 33,i32 34,i32 35,i32 36,i32 37,i32 38,i32 39,i32 40,i32 41,i32 42,i32 43,i32 44,i32 45,i32 46,i32 47>
          %v3 = shufflevector <64 x i8> %v, <64 x i8> undef, <16 x i32> <i32 48,i32 49,i32 50,i32 51,i32 52,i32 53,i32 54,i32 55,i32 56,i32 57,i32 58,i32 59,i32 60,i32 61,i32 62,i32 63>
          %s0 = shufflevector <64 x i8> %sh, <64 x i8> undef, <16 x i32> <i32 0,i32 1,i32 2,i32 3,i32 4,i32 5,i32 6,i32 7,i32 8,i32 9,i32 10,i32 11,i32 12,i32 13,i32 14,i32 15>
          %s1 = shufflevector <64 x i8> %sh, <64 x i8> undef, <16 x i32> <i32 16,i32 17,i32 18,i32 19,i32 20,i32 21,i32 22,i32 23,i32 24,i32 25,i32 26,i32 27,i32 28,i32 29,i32 30,i32 31>
          %s2 = shufflevector <64 x i8> %sh, <64 x i8> undef, <16 x i32> <i32 32,i32 33,i32 34,i32 35,i32 36,i32 37,i32 38,i32 39,i32 40,i32 41,i32 42,i32 43,i32 44,i32 45,i32 46,i32 47>
          %s3 = shufflevector <64 x i8> %sh, <64 x i8> undef, <16 x i32> <i32 48,i32 49,i32 50,i32 51,i32 52,i32 53,i32 54,i32 55,i32 56,i32 57,i32 58,i32 59,i32 60,i32 61,i32 62,i32 63>
          %t0 = call <16 x i8> @llvm.aarch64.neon.sshl.v16i8(<16 x i8> %v0, <16 x i8> %s0)
          %t1 = call <16 x i8> @llvm.aarch64.neon.sshl.v16i8(<16 x i8> %v1, <16 x i8> %s1)
          %t2 = call <16 x i8> @llvm.aarch64.neon.sshl.v16i8(<16 x i8> %v2, <16 x i8> %s2)
          %t3 = call <16 x i8> @llvm.aarch64.neon.sshl.v16i8(<16 x i8> %v3, <16 x i8> %s3)
          %bm0 = and <16 x i8> %t0, <i8 1,i8 2,i8 4,i8 8,i8 16,i8 32,i8 64,i8 -128,i8 1,i8 2,i8 4,i8 8,i8 16,i8 32,i8 64,i8 -128>
          %bm1 = and <16 x i8> %t1, <i8 1,i8 2,i8 4,i8 8,i8 16,i8 32,i8 64,i8 -128,i8 1,i8 2,i8 4,i8 8,i8 16,i8 32,i8 64,i8 -128>
          %bm2 = and <16 x i8> %t2, <i8 1,i8 2,i8 4,i8 8,i8 16,i8 32,i8 64,i8 -128,i8 1,i8 2,i8 4,i8 8,i8 16,i8 32,i8 64,i8 -128>
          %bm3 = and <16 x i8> %t3, <i8 1,i8 2,i8 4,i8 8,i8 16,i8 32,i8 64,i8 -128,i8 1,i8 2,i8 4,i8 8,i8 16,i8 32,i8 64,i8 -128>
          %p0 = call <16 x i8> @llvm.aarch64.neon.addp.v16i8(<16 x i8> %bm0, <16 x i8> %bm1)
          %p1 = call <16 x i8> @llvm.aarch64.neon.addp.v16i8(<16 x i8> %bm2, <16 x i8> %bm3)
          %p2 = call <16 x i8> @llvm.aarch64.neon.addp.v16i8(<16 x i8> %p0, <16 x i8> %p1)
          %p3 = call <16 x i8> @llvm.aarch64.neon.addp.v16i8(<16 x i8> %p2, <16 x i8> %p2)
          %r = bitcast <16 x i8> %p3 to <2 x i64>
          %lo = extractelement <2 x i64> %r, i32 0
          ret i64 %lo }
        attributes #0 = { alwaysinline "target-features"="+neon" }""", "entry"),
        UInt64, Tuple{NTuple{64,VecElement{Int8}},NTuple{64,VecElement{Int8}}}, v.data, sh.data)

    # hb bit 7 is the accumulator MSB (the sin sign), bit 6 is acc₃₀ — no byte add
    # shapes a cos byte; the ⊕ happens on the packed words in the caller's scalar xor.
    @inline function _flip2_neon(a0::Vec{16,UInt32}, a1::Vec{16,UInt32},
                                 a2::Vec{16,UInt32}, a3::Vec{16,UInt32})
        hb = _hb64(a0, a1, a2, a3)
        s = _sign_pack(hb)
        (s, s ⊻ _shift_pack(hb, Val(6)))               # plane(acc₃₀); cos = s ⊕ it
    end
    @inline function _flip_words(::Neon, base::UInt32, ramp::Tuple{Vec{16,UInt32},UInt32})
        r, q = ramp
        b1 = base + q; b2 = b1 + q; b3 = b2 + q
        _flip2_neon(Vec{16,UInt32}(base) + r, Vec{16,UInt32}(b1) + r,
                    Vec{16,UInt32}(b2) + r, Vec{16,UInt32}(b3) + r)
    end
end

# ---- fw ≥ 2²⁵: branch-free SIMD loop over the full words, plus the masked partial tail.
# The generic (x86/portable) form keeps Vec{64,UInt32} NCO accumulators live and is
# unrolled over independent streams (hand-unrolled with plain locals, like kernel.jl — a
# tuple/closure formulation boxes): AVX-512 has 32 ZMM registers and a short per-word
# dependency chain, so 4 streams pipeline well; on AVX2 a 64-lane accumulator is already
# 8 of the 16 YMM registers, so it runs a single stream.
_sb_unroll(::Backend) = Val(1)
_sb_unroll(::AVX512)  = Val(4)

function _signs_fill_flips!(backend::Backend, sin_signs, cos_signs,
                            n::Int, nwords::Int, fw::UInt32, off::UInt32)
    rem = n - (nwords - 1) * 64                # valid lanes in the last word
    nfull = rem == 64 ? nwords : nwords - 1
    step = fw * UInt32(64)
    acc0 = Vec{64,UInt32}(off) + _iota_u32() * Vec{64,UInt32}(fw)
    acc = _signs_flips_loop!(sin_signs, cos_signs, nfull, acc0, step, _sb_unroll(backend))
    @inbounds if nfull < nwords                # masked partial tail word
        m = _lowmask(rem)
        s, c = _flip2(backend, acc)
        sin_signs[nwords] = s & m; cos_signs[nwords] = c & m
    end
end

function _signs_flips_loop!(sin_signs, cos_signs,
                            nfull::Int, acc1::Vec{64,UInt32}, step::UInt32, ::Val{4})
    step4 = step * UInt32(4)                    # wraps mod 2³², like the accumulator
    acc2 = acc1 + step; acc3 = acc2 + step; acc4 = acc3 + step
    w = 1
    @inbounds while w + 3 <= nfull
        s, c = _flip2(_BITS_BACKEND, acc1); sin_signs[w]   = s; cos_signs[w]   = c
        s, c = _flip2(_BITS_BACKEND, acc2); sin_signs[w+1] = s; cos_signs[w+1] = c
        s, c = _flip2(_BITS_BACKEND, acc3); sin_signs[w+2] = s; cos_signs[w+2] = c
        s, c = _flip2(_BITS_BACKEND, acc4); sin_signs[w+3] = s; cos_signs[w+3] = c
        acc1 += step4; acc2 += step4; acc3 += step4; acc4 += step4
        w += 4
    end
    @inbounds while w <= nfull                  # ≤ 3 leftover full words; acc1 tracks w
        s, c = _flip2(_BITS_BACKEND, acc1); sin_signs[w] = s; cos_signs[w] = c
        acc1 += step; w += 1
    end
    acc1                                        # positioned at the (possibly partial) tail
end
function _signs_flips_loop!(sin_signs, cos_signs,
                            nfull::Int, acc1::Vec{64,UInt32}, step::UInt32, ::Val{1})
    @inbounds for w in 1:nfull
        s, c = _flip2(_BITS_BACKEND, acc1); sin_signs[w] = s; cos_signs[w] = c
        acc1 += step
    end
    acc1
end

@static if Sys.ARCH === :aarch64
    # NEON flip loop: one resident 16-lane accumulator plus three broadcast quarter
    # offsets — each word's four quarter accumulators are transient, nothing spills.
    function _signs_fill_flips!(::Neon, sin_signs, cos_signs,
                                n::Int, nwords::Int, fw::UInt32, off::UInt32)
        rem = n - (nwords - 1) * 64
        nfull = rem == 64 ? nwords : nwords - 1
        q = fw * UInt32(16)
        q1 = Vec{16,UInt32}(q); q2 = Vec{16,UInt32}(q + q); q3 = Vec{16,UInt32}(q + q + q)
        stepv = Vec{16,UInt32}(fw * UInt32(64))
        acc = Vec{16,UInt32}(off) + _lanes_u32(Val(16)) * fw
        @inbounds for w in 1:nfull
            s, c = _flip2_neon(acc, acc + q1, acc + q2, acc + q3)
            sin_signs[w] = s; cos_signs[w] = c
            acc += stepv
        end
        @inbounds if nfull < nwords            # masked partial tail word
            m = _lowmask(rem)
            s, c = _flip2_neon(acc, acc + q1, acc + q2, acc + q3)
            sin_signs[nwords] = s & m; cos_signs[nwords] = c & m
        end
    end
end

# ---- fw < 2²⁵: run-length regime, one word at a time. Every word has ≤ 1 flip per
# component (64·fw < 2³¹), so a component is constant across the word iff its accumulator
# MSB matches at the first and last lane; a low residual carrier is a square wave with
# runs of thousands of samples, making almost every word one constant store.
function _signs_fill_runs!(backend::Backend, sin_signs, cos_signs,
                           n::Int, nwords::Int, fw::UInt32, off::UInt32)
    ramp = _bits_ramp(backend, fw)
    step = fw * UInt32(64)                     # accumulator advance per 64-sample word
    span = fw * UInt32(63)                     # advance across the 63 lanes within a word
    base = off                                 # accumulator at sample 0 (lane 0) = off
    @inbounds for w in 1:nwords
        m = _lowmask(w == nwords ? n - (w - 1) * 64 : 64)              # valid-lane mask (tail may be < 64)
        bc = base + 0x40000000                                          # cos accumulator (+¼ cycle)
        sin_const = _msb(base) == _msb(base + span)
        cos_const = _msb(bc)   == _msb(bc + span)
        if sin_const & cos_const                                        # both constant → two stores
            sin_signs[w] = (_msb(base) ? typemax(UInt64) : zero(UInt64)) & m
            cos_signs[w] = (_msb(bc)   ? typemax(UInt64) : zero(UInt64)) & m
        else                                                            # a flip inside → one evaluation
            sw, cw = _flip_words(backend, base, ramp)
            sin_signs[w] = sw & m; cos_signs[w] = cw & m
        end
        base += step
    end
end

function _carrier_signs!(sin_signs::AbstractVector{UInt64}, cos_signs::AbstractVector{UInt64},
                         n::Int, fw::UInt32, off::UInt32)
    nwords = cld(n, 64)
    # ≤ 1 sign flip per word ⇔ the word advances < ½ cycle ⇔ 64·fw < 2³¹ ⇔ fw < 2²⁵. Test fw
    # directly (NOT the wrapped `step`, which loses the whole-cycle count for a fast carrier).
    if fw < 0x02000000
        _signs_fill_runs!(_BITS_BACKEND, sin_signs, cos_signs, n, nwords, fw, off)
    else                                       # a flip can fall in every word → branch-free loop
        _signs_fill_flips!(_BITS_BACKEND, sin_signs, cos_signs, n, nwords, fw, off)
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
every 64-sample word is a single constant run written with one store; where sign flips fall
inside words, each word's 64 signs are packed with a single SIMD sign-mask (no per-sample
loop). This is the carrier form a bit-wise correlator consumes, where wipe-off becomes XOR
and accumulation becomes popcount.

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
