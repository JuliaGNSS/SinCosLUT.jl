# AVX2 backend: vpshufb-based 64-entry Int8 lookup (no cross-lane byte permute on
# AVX2, so the table is looked up as in-lane 16-entry shuffles + blends).
# Int16/Int32 have no AVX2 word/dword table permute here → handled by the portable
# backend.
#
# Constructor-built tables are half-wave ANTI-SYMMETRIC — value[k+32] == -value[k]
# exactly (sin(θ+π) = -sin θ, and 2πk/64 angles admit no representable rounding ties by
# Niven's theorem, so rounding commutes with negation). `_prepare` verifies this once and
# returns a 4-register layout: each component becomes a 32-entry HALF-table lookup
# (2 pshufb + 1 blend on index bit 4) followed by one `psignb` driven by index bit 5 —
# instead of 4 pshufb + 3 blends over the full 64 entries. That halves both the pshufb
# count and the table registers (8 → 4), which matters doubly on AVX2: pshufb/pblendvb
# contend on the one shuffle port of older Intel cores, and only 16 YMM registers exist,
# so a smaller table footprint eliminates accumulator spills in the fill kernel.
# A hand-built table that violates the symmetry falls back to the original 8-register
# full-table layout (dispatch on the tuple size; the two `_prepare` return types are
# split by the function barrier in kernel.jl).
#
# `pshufb` architecturally reads only index bits [3:0] (byte select) and bit 7 (zeroing);
# bits [6:4] are IGNORED — so the in-package indices (always in [0, 63], bit 7 clear) need
# no `& 15` before the shuffle, and the half/sign selectors read bits 4/5 via shifts.
#
# Every llvmcall IR module below MUST carry `alwaysinline` on its entry function:
# the (module, "entry") form of llvmcall otherwise emits a real `call` (with a full
# register spill/reload around it) rather than the bare instruction. With several
# shuffle/blend ops per `_apply`, that call+spill overhead dominated — `alwaysinline`
# folds each wrapper back to a single instruction and is worth ~8× on this backend.

const _PSHUFB_IR = """
declare <32 x i8> @llvm.x86.avx2.pshuf.b(<32 x i8>, <32 x i8>)
define <32 x i8> @entry(<32 x i8> %t, <32 x i8> %i) #0 {
  %r = call <32 x i8> @llvm.x86.avx2.pshuf.b(<32 x i8> %t, <32 x i8> %i)
  ret <32 x i8> %r }
attributes #0 = { alwaysinline "target-features"="+avx2" }
"""
@inline _pshufb(table::Vec{32,Int8}, index::Vec{32,Int8}) =
    Vec{32,Int8}(Base.llvmcall((_PSHUFB_IR, "entry"), NTuple{32,VecElement{Int8}},
        Tuple{NTuple{32,VecElement{Int8}},NTuple{32,VecElement{Int8}}}, table.data, index.data))

const _PBLENDVB_IR = """
declare <32 x i8> @llvm.x86.avx2.pblendvb(<32 x i8>, <32 x i8>, <32 x i8>)
define <32 x i8> @entry(<32 x i8> %a, <32 x i8> %b, <32 x i8> %m) #0 {
  %r = call <32 x i8> @llvm.x86.avx2.pblendvb(<32 x i8> %a, <32 x i8> %b, <32 x i8> %m)
  ret <32 x i8> %r }
attributes #0 = { alwaysinline "target-features"="+avx2" }
"""
@inline _pblendvb(a::Vec{32,Int8}, b::Vec{32,Int8}, mask::Vec{32,Int8}) =
    Vec{32,Int8}(Base.llvmcall((_PBLENDVB_IR, "entry"), NTuple{32,VecElement{Int8}},
        Tuple{NTuple{32,VecElement{Int8}},NTuple{32,VecElement{Int8}},NTuple{32,VecElement{Int8}}},
        a.data, b.data, mask.data))

# psignb: r = b > 0 ? a : (b < 0 ? -a : 0). The sign vector below is never 0, so it is a
# pure conditional negate. Negation wraps (-(-128) = -128), matching Julia's Int8 `-`.
const _PSIGNB_IR = """
declare <32 x i8> @llvm.x86.avx2.psign.b(<32 x i8>, <32 x i8>)
define <32 x i8> @entry(<32 x i8> %a, <32 x i8> %b) #0 {
  %r = call <32 x i8> @llvm.x86.avx2.psign.b(<32 x i8> %a, <32 x i8> %b)
  ret <32 x i8> %r }
attributes #0 = { alwaysinline "target-features"="+avx2" }
"""
@inline _psignb(a::Vec{32,Int8}, b::Vec{32,Int8}) =
    Vec{32,Int8}(Base.llvmcall((_PSIGNB_IR, "entry"), NTuple{32,VecElement{Int8}},
        Tuple{NTuple{32,VecElement{Int8}},NTuple{32,VecElement{Int8}}}, a.data, b.data))

# 16-entry subtable `k` of a 64-entry table, broadcast into both 128-bit lanes
@inline _subtable(values::NTuple{64,Int8}, k) =
    Vec{32,Int8}(ntuple(j -> @inbounds(values[16k + ((j - 1) & 15) + 1]), Val(32)))

# half-wave anti-symmetry: value[k+32] == -value[k] with wrapping negation (=== psignb's)
_antisymmetric(values::NTuple{64,Int8}) = all(k -> values[k + 32] === -values[k], 1:32)

# prepare: 4-register half-table layout when the table is anti-symmetric (always true for
# constructor-built tables), else the 8-register full-table fallback. Value-dependent
# return TYPE — callers in kernel.jl hoist this behind a function barrier.
@inline function _prepare(::AVX2, table::SinCosTable{Int8,64})
    if _antisymmetric(table.sin) && _antisymmetric(table.cos)
        (_subtable(table.sin, 0), _subtable(table.sin, 1),
         _subtable(table.cos, 0), _subtable(table.cos, 1))
    else
        (_subtable(table.sin, 0), _subtable(table.sin, 1), _subtable(table.sin, 2), _subtable(table.sin, 3),
         _subtable(table.cos, 0), _subtable(table.cos, 1), _subtable(table.cos, 2), _subtable(table.cos, 3))
    end
end

# fast path: 32-entry half lookup (bit 4 selects the sub-table) + conditional negate on
# bit 5. The `| 1` keeps the psignb selector nonzero (index 0 would otherwise zero lane 0).
@inline function _apply(::AVX2, regs::NTuple{4,Vec{32,Int8}}, index::Vec{32,Int8})
    sel_bit4 = index << 3                    # bit 4 → sign bit for the half-table blend
    sign_sel = (index << 2) | Int8(1)        # bit 5 → sign bit for psignb; |1 avoids 0
    sin_vec = _psignb(_pblendvb(_pshufb(regs[1], index), _pshufb(regs[2], index), sel_bit4), sign_sel)
    cos_vec = _psignb(_pblendvb(_pshufb(regs[3], index), _pshufb(regs[4], index), sel_bit4), sign_sel)
    (sin_vec, cos_vec)
end

# fallback (asymmetric hand-built tables): full 64-entry lookup, bits 4/5 select via blends
@inline function _apply(::AVX2, regs::NTuple{8,Vec{32,Int8}}, index::Vec{32,Int8})
    sel_bit4 = index << 3; sel_bit5 = index << 2   # move bits 4/5 into the sign bit for blends
    sin_vec = _pblendvb(_pblendvb(_pshufb(regs[1], index), _pshufb(regs[2], index), sel_bit4),
                        _pblendvb(_pshufb(regs[3], index), _pshufb(regs[4], index), sel_bit4), sel_bit5)
    cos_vec = _pblendvb(_pblendvb(_pshufb(regs[5], index), _pshufb(regs[6], index), sel_bit4),
                        _pblendvb(_pshufb(regs[7], index), _pshufb(regs[8], index), sel_bit4), sel_bit5)
    (sin_vec, cos_vec)
end

_vwidth(::AVX2, ::Type{Int8}) = Val(32)

# Fast Int8 phase→index extraction (overrides the generic shift+convert in kernel.jl).
# The generic narrowing convert leaves LLVM to improvise (~3 extra lane fixups); this pins
# the minimal exact sequence: shift the four 8-dword quarters down to the index (values in
# [0, N) ≤ 63, so the unsigned-saturating packs are exact), pack dwords→words→bytes in-lane,
# and undo the two levels of 128-bit-lane interleaving with a single cross-lane vpermd —
# 4×vpsrld + 2×vpackusdw + 1×vpackuswb + 1×vpermd, no mask needed.
const _PACKUSDW_IR = """
declare <16 x i16> @llvm.x86.avx2.packusdw(<8 x i32>, <8 x i32>)
define <16 x i16> @entry(<8 x i32> %a, <8 x i32> %b) #0 {
  %r = call <16 x i16> @llvm.x86.avx2.packusdw(<8 x i32> %a, <8 x i32> %b)
  ret <16 x i16> %r }
attributes #0 = { alwaysinline "target-features"="+avx2" }
"""
@inline _packusdw(a::Vec{8,UInt32}, b::Vec{8,UInt32}) =
    Vec{16,UInt16}(Base.llvmcall((_PACKUSDW_IR, "entry"), NTuple{16,VecElement{UInt16}},
        Tuple{NTuple{8,VecElement{UInt32}},NTuple{8,VecElement{UInt32}}}, a.data, b.data))

const _PACKUSWB_IR = """
declare <32 x i8> @llvm.x86.avx2.packuswb(<16 x i16>, <16 x i16>)
define <32 x i8> @entry(<16 x i16> %a, <16 x i16> %b) #0 {
  %r = call <32 x i8> @llvm.x86.avx2.packuswb(<16 x i16> %a, <16 x i16> %b)
  ret <32 x i8> %r }
attributes #0 = { alwaysinline "target-features"="+avx2" }
"""
@inline _packuswb(a::Vec{16,UInt16}, b::Vec{16,UInt16}) =
    Vec{32,Int8}(Base.llvmcall((_PACKUSWB_IR, "entry"), NTuple{32,VecElement{Int8}},
        Tuple{NTuple{16,VecElement{UInt16}},NTuple{16,VecElement{UInt16}}}, a.data, b.data))

@inline _avx2_quarter(acc::Vec{32,UInt32}, ::Val{o}) where {o} =
    shufflevector(acc, Val(ntuple(i -> i - 1 + o, Val(8))))

@inline function _phase_index(::AVX2, acc::Vec{32,UInt32}, ::Val{N}, ::Type{Int8}) where {N}
    shift = UInt32(_index_shift(Val(N)))
    q0 = _avx2_quarter(acc, Val(0)) >> shift; q1 = _avx2_quarter(acc, Val(8)) >> shift
    q2 = _avx2_quarter(acc, Val(16)) >> shift; q3 = _avx2_quarter(acc, Val(24)) >> shift
    w01 = _packusdw(q0, q1)     # in-lane pack: lane0 = words of dwords [0-3, 8-11], …
    w23 = _packusdw(q2, q3)
    b = _packuswb(w01, w23)     # bytes; two in-lane packs leave dword-groups shuffled
    reinterpret(Vec{32,Int8},   # one cross-lane vpermd restores sample order
        shufflevector(reinterpret(Vec{8,UInt32}, b), Val((0, 4, 1, 5, 2, 6, 3, 7))))
end
