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
attributes #0 = { alwaysinline "target-features"="+neon" }
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

# Fast Int8 phase→index extraction (overrides the generic shift+convert in kernel.jl),
# the NEON analogue of the AVX-512 byte-gather. `convert(Vec{16,UInt32} -> Int8)` narrows
# across the four Q registers the accumulator occupies (a uzp/xtn chain), and that narrowing
# is a large fraction of the per-element cost here since the lookup itself is only 2×tbl4.
# The table index is the top log2(N) bits of each lane, which sit entirely in byte 3 of the
# little-endian UInt32. A single `tbl4` over the accumulator's four 16-byte quarters gathers
# byte 3 of all 16 lanes (table index 4k+3) — tbl4 permutes bytes across all four source
# registers in one op, so unlike AVX-512 vpermi2b no merge is needed. Then right-align
# (index = byte3 >> (8-log2(N)) = byte3 >> (index_shift-24)) and mask to N-1. The byte shift
# can bleed a neighbour lane's low bits into bits ≥ log2(N); the `& (N-1)` clears them, so the
# result is exact. Out-of-range tbl4 indices return 0, but 4k+3 ∈ 3..63 is always in range.
@inline function _phase_index(::Neon, acc::Vec{16,UInt32}, ::Val{N}, ::Type{Int8}) where {N}
    b = reinterpret(Vec{64,Int8}, acc)
    q0 = shufflevector(b, Val(ntuple(i -> i - 1,  Val(16))))
    q1 = shufflevector(b, Val(ntuple(i -> i + 15, Val(16))))
    q2 = shufflevector(b, Val(ntuple(i -> i + 31, Val(16))))
    q3 = shufflevector(b, Val(ntuple(i -> i + 47, Val(16))))
    gather = Vec{16,Int8}(ntuple(k -> Int8(4 * (k - 1) + 3), Val(16)))   # byte 3 of dwords 0..15
    msbyte = _tbl4(q0, q1, q2, q3, gather)
    (msbyte >> UInt8(_index_shift(Val(N)) - 24)) & UInt8(N - 1)
end
