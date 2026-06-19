# AVX2 backend: vpshufb-based 64-entry Int8 lookup (no cross-lane byte permute on
# AVX2, so split the 6-bit index into four in-lane 16-entry shuffles + blends).
# Int16/Int32 have no AVX2 word/dword table permute here → handled by the portable
# backend.
#
# Every llvmcall IR module below MUST carry `alwaysinline` on its entry function:
# the (module, "entry") form of llvmcall otherwise emits a real `call` (with a full
# register spill/reload around it) rather than the bare instruction. With ~14
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

# 16-entry subtable `k` of a 64-entry table, broadcast into both 128-bit lanes
@inline _subtable(values::NTuple{64,Int8}, k) =
    Vec{32,Int8}(ntuple(j -> @inbounds(values[16k + ((j - 1) & 15) + 1]), Val(32)))

# prepare: the four 16-entry sub-tables (broadcast to both lanes) for sin and cos
@inline _prepare(::AVX2, table::SinCosTable{Int8,64}) =
    (_subtable(table.sin, 0), _subtable(table.sin, 1), _subtable(table.sin, 2), _subtable(table.sin, 3),
     _subtable(table.cos, 0), _subtable(table.cos, 1), _subtable(table.cos, 2), _subtable(table.cos, 3))

@inline function _apply(::AVX2, regs::NTuple{8,Vec{32,Int8}}, index::Vec{32,Int8})
    low = index & Int8(15)                 # low 4 bits select within each 16-entry sub-table
    sel_bit4 = index << 3; sel_bit5 = index << 2   # move bits 4/5 into the sign bit for blends
    sin_vec = _pblendvb(_pblendvb(_pshufb(regs[1], low), _pshufb(regs[2], low), sel_bit4),
                        _pblendvb(_pshufb(regs[3], low), _pshufb(regs[4], low), sel_bit4), sel_bit5)
    cos_vec = _pblendvb(_pblendvb(_pshufb(regs[5], low), _pshufb(regs[6], low), sel_bit4),
                        _pblendvb(_pshufb(regs[7], low), _pshufb(regs[8], low), sel_bit4), sel_bit5)
    (sin_vec, cos_vec)
end

_vwidth(::AVX2, ::Type{Int8}) = Val(32)
