# AArch64 NEON backend: `tbl4` is a register-resident byte-table permute (the
# direct analogue of x86 `vpermb`). Four 16-byte table registers cover a 64-entry
# table, 16 lanes wide; out-of-range indices return 0.
#
# Int16/Int32 are byte-only on NEON (no word/dword table permute), so they use the
# portable backend here. SVE2 would give a scalable-width `tbl`, but Julia cannot
# express scalable vectors (JuliaLang/julia#40308); NEON runs natively on SVE HW.
#
# NOTE: written against the LLVM aarch64 intrinsics; not executed on ARM hardware
# in development — validate on-device before relying on it.

const _TBL4_IR = """
declare <16 x i8> @llvm.aarch64.neon.tbl4(<16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>)
define <16 x i8> @entry(<16 x i8> %a, <16 x i8> %b, <16 x i8> %c, <16 x i8> %d, <16 x i8> %i) #0 {
  %r = call <16 x i8> @llvm.aarch64.neon.tbl4(<16 x i8> %a, <16 x i8> %b, <16 x i8> %c, <16 x i8> %d, <16 x i8> %i)
  ret <16 x i8> %r }
attributes #0 = { "target-features"="+neon" }
"""
@inline function _tbl4(part0::Vec{16,Int8}, part1::Vec{16,Int8}, part2::Vec{16,Int8},
                       part3::Vec{16,Int8}, index::Vec{16,Int8})
    V = NTuple{16,VecElement{Int8}}
    Vec{16,Int8}(Base.llvmcall((_TBL4_IR, "entry"), V, Tuple{V,V,V,V,V},
        part0.data, part1.data, part2.data, part3.data, index.data))
end

# 16-entry slice `k` of a 64-entry table (one of the four tbl4 table registers)
@inline _slice(values::NTuple{64,Int8}, k) = Vec{16,Int8}(ntuple(j -> @inbounds(values[16k + j]), Val(16)))

@inline _prepare(::Neon, table::SinCosTable{Int8,64}) =
    (_slice(table.sin, 0), _slice(table.sin, 1), _slice(table.sin, 2), _slice(table.sin, 3),
     _slice(table.cos, 0), _slice(table.cos, 1), _slice(table.cos, 2), _slice(table.cos, 3))

@inline _apply(::Neon, regs::NTuple{8,Vec{16,Int8}}, index::Vec{16,Int8}) =
    (_tbl4(regs[1], regs[2], regs[3], regs[4], index), _tbl4(regs[5], regs[6], regs[7], regs[8], index))

_vwidth(::Neon, ::Type{Int8}) = Val(16)
