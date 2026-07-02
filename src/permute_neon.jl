# AArch64 NEON backend: `tbl` is a register-resident byte-table permute (the direct
# analogue of x86 `vpermb`). Four 16-byte table registers (`tbl4`) cover a 64-byte
# table, 16 lanes wide; out-of-range indices return 0.
#
# Int8 (steps = 64): one tbl4 per component.
# Int16 (steps = 32 or 64): looked up BYTEWISE — word index k becomes the byte pair
# (2k, 2k+1) via one word-domain multiply-add (k·514 + 256 ≡ little-endian bytes
# [2k, 2k+1]), and each output half is a tbl4 over the table's bytes. A 64-entry Int16
# table is 128 bytes — beyond one tbl4 — but `tbl` returning 0 for out-of-range indices
# makes the split exact for ARBITRARY tables: tbl4(first 64 B, b) | tbl4(second 64 B,
# b - 64); exactly one side is nonzero per lane. No symmetry assumption, type-stable.
# Int32 has no payoff at 4 lanes/register → portable backend.
#
# SVE2 would give a scalable-width `tbl`, but Julia cannot express scalable vectors
# (JuliaLang/julia#40308); NEON runs natively on SVE HW.
#
# NOTE: written against the LLVM aarch64 intrinsics; not executed on ARM hardware
# in development — validated by the CI test matrix on Apple M1 and Neoverse.

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

const _TBL2_IR = """
declare <16 x i8> @llvm.aarch64.neon.tbl2(<16 x i8>, <16 x i8>, <16 x i8>)
define <16 x i8> @entry(<16 x i8> %a, <16 x i8> %b, <16 x i8> %i) #0 {
  %r = call <16 x i8> @llvm.aarch64.neon.tbl2(<16 x i8> %a, <16 x i8> %b, <16 x i8> %i)
  ret <16 x i8> %r }
attributes #0 = { alwaysinline "target-features"="+neon" }
"""
@inline function _tbl2(part0::Vec{16,Int8}, part1::Vec{16,Int8}, index::Vec{16,Int8})
    V = NTuple{16,VecElement{Int8}}
    Vec{16,Int8}(Base.llvmcall((_TBL2_IR, "entry"), V, Tuple{V,V,V},
        part0.data, part1.data, index.data))
end

# ---- Int8 (steps = 64) ----

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
# registers in one op, so unlike AVX-512 vpermi2b no merge is needed. Then right-align:
# index = byte3 >> (8-log2(N)) = byte3 >> (index_shift-24). NEON has NATIVE byte shifts
# (`ushr.16b` — unlike x86, where byte shifts are emulated as word shifts and bleed
# neighbour bits), so an UNSIGNED shift zero-fills and the result is exact with no mask.
# Exactness matters here: tbl returns 0 for out-of-range indices, so junk bits would
# corrupt the lookup, not be ignored. Out-of-range tbl4 gather indices cannot occur
# (4k+3 ∈ 3..63).
@inline function _phase_index(::Neon, acc::Vec{16,UInt32}, ::Val{N}, ::Type{Int8}) where {N}
    b = reinterpret(Vec{64,Int8}, acc)
    q0 = shufflevector(b, Val(ntuple(i -> i - 1,  Val(16))))
    q1 = shufflevector(b, Val(ntuple(i -> i + 15, Val(16))))
    q2 = shufflevector(b, Val(ntuple(i -> i + 31, Val(16))))
    q3 = shufflevector(b, Val(ntuple(i -> i + 47, Val(16))))
    gather = Vec{16,Int8}(ntuple(k -> Int8(4 * (k - 1) + 3), Val(16)))   # byte 3 of dwords 0..15
    msbyte = _tbl4(q0, q1, q2, q3, gather)
    reinterpret(Vec{16,Int8},
        reinterpret(Vec{16,UInt8}, msbyte) >> UInt8(_index_shift(Val(N)) - 24))
end

# ---- Int16 (steps = 32 or 64): bytewise tbl over the table's bytes ----

# 16-BYTE slice `j` of an Int16 table (byte pairs little-endian: entry k → [low, high])
@inline _slice16(values::NTuple{N,Int16}, j) where {N} =
    Vec{16,Int8}(ntuple(i -> begin
        e = @inbounds values[8j + ((i - 1) >> 1) + 1]
        (reinterpret(UInt16, e) >> (8 * ((i - 1) & 1))) % Int8
    end, Val(16)))

# N=32: 64 B per component = one tbl4 set → 8 registers total
@inline _prepare(::Neon, table::SinCosTable{Int16,32}) =
    (ntuple(j -> _slice16(table.sin, j - 1), Val(4))...,
     ntuple(j -> _slice16(table.cos, j - 1), Val(4))...)
# N=64: 128 B per component = two tbl4 sets → 16 registers total
@inline _prepare(::Neon, table::SinCosTable{Int16,64}) =
    (ntuple(j -> _slice16(table.sin, j - 1), Val(8))...,
     ntuple(j -> _slice16(table.cos, j - 1), Val(8))...)

# word index k → little-endian byte-pair indices [2k, 2k+1] packed in the same word:
# 2k + ((2k+1) << 8) = 514k + 256. k ≤ 63 → ≤ 32638, no Int16 overflow.
@inline _byte_pair_index(index::Vec{16,Int16}) =
    reinterpret(Vec{32,Int8}, index * Int16(514) + Int16(256))
@inline _half16(v::Vec{32,Int8}, ::Val{o}) where {o} =
    shufflevector(v, Val(ntuple(i -> i - 1 + o, Val(16))))
@inline _cat_words(lo::Vec{16,Int8}, hi::Vec{16,Int8}) =
    reinterpret(Vec{16,Int16}, shufflevector(lo, hi, Val(ntuple(i -> i - 1, Val(32)))))

@inline function _apply(::Neon, regs::NTuple{8,Vec{16,Int8}}, index::Vec{16,Int16})   # N=32
    b = _byte_pair_index(index)
    blo = _half16(b, Val(0)); bhi = _half16(b, Val(16))
    (_cat_words(_tbl4(regs[1], regs[2], regs[3], regs[4], blo),
                _tbl4(regs[1], regs[2], regs[3], regs[4], bhi)),
     _cat_words(_tbl4(regs[5], regs[6], regs[7], regs[8], blo),
                _tbl4(regs[5], regs[6], regs[7], regs[8], bhi)))
end
@inline function _apply(::Neon, regs::NTuple{16,Vec{16,Int8}}, index::Vec{16,Int16})  # N=64
    b = _byte_pair_index(index)
    b2 = b - Int8(64)                     # second 64 B; <64 wraps out of range → tbl gives 0
    blo = _half16(b, Val(0)); bhi = _half16(b, Val(16))
    b2lo = _half16(b2, Val(0)); b2hi = _half16(b2, Val(16))
    sin_lo = _tbl4(regs[1], regs[2], regs[3], regs[4], blo) |
             _tbl4(regs[5], regs[6], regs[7], regs[8], b2lo)
    sin_hi = _tbl4(regs[1], regs[2], regs[3], regs[4], bhi) |
             _tbl4(regs[5], regs[6], regs[7], regs[8], b2hi)
    cos_lo = _tbl4(regs[9], regs[10], regs[11], regs[12], blo) |
             _tbl4(regs[13], regs[14], regs[15], regs[16], b2lo)
    cos_hi = _tbl4(regs[9], regs[10], regs[11], regs[12], bhi) |
             _tbl4(regs[13], regs[14], regs[15], regs[16], b2hi)
    (_cat_words(sin_lo, sin_hi), _cat_words(cos_lo, cos_hi))
end

_vwidth(::Neon, ::Type{Int16}) = Val(16)

# The N=64 tables occupy 16 registers; with the default 4 NCO streams (4×4 accumulator Q
# regs) that would exactly exhaust all 32 V registers and spill. 2 streams leave headroom.
_unroll(::Neon, ::Type{Int16}) = Val(2)

# Int16 phase→index extraction: the index is the top log2(N) ≤ 6 bits, entirely inside the
# HIGH word (bytes 2,3) of each little-endian dword lane. One tbl2 per accumulator half
# gathers those byte pairs, then a native 16-bit unsigned shift right-aligns — exact,
# nothing to mask.
@inline function _phase_index(::Neon, acc::Vec{16,UInt32}, ::Val{N}, ::Type{Int16}) where {N}
    b = reinterpret(Vec{64,Int8}, acc)
    q0 = shufflevector(b, Val(ntuple(i -> i - 1,  Val(16))))
    q1 = shufflevector(b, Val(ntuple(i -> i + 15, Val(16))))
    q2 = shufflevector(b, Val(ntuple(i -> i + 31, Val(16))))
    q3 = shufflevector(b, Val(ntuple(i -> i + 47, Val(16))))
    gather = Vec{16,Int8}(ntuple(i -> Int8(4 * ((i - 1) >> 1) + 2 + ((i - 1) & 1)), Val(16)))
    w_lo = _tbl2(q0, q1, gather)          # high words of dwords 0..7
    w_hi = _tbl2(q2, q3, gather)          # high words of dwords 8..15
    hw = reinterpret(Vec{16,UInt16}, shufflevector(w_lo, w_hi, Val(ntuple(i -> i - 1, Val(32)))))
    reinterpret(Vec{16,Int16}, hw >> UInt16(_index_shift(Val(N)) - 16))
end
