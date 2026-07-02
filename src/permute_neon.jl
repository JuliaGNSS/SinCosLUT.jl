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

# ---- Int8 (steps = 64) ----

# 16-entry slice `k` of a 64-entry table (one of the four tbl4 table registers)
@inline _slice(values::NTuple{64,Int8}, k) = Vec{16,Int8}(ntuple(j -> @inbounds(values[16k + j]), Val(16)))

@inline _prepare(::Neon, table::SinCosTable{Int8,64}) =
    (_slice(table.sin, 0), _slice(table.sin, 1), _slice(table.sin, 2), _slice(table.sin, 3),
     _slice(table.cos, 0), _slice(table.cos, 1), _slice(table.cos, 2), _slice(table.cos, 3))

@inline _apply(::Neon, regs::NTuple{8,Vec{16,Int8}}, index::Vec{16,Int8}) =
    (_tbl4(regs[1], regs[2], regs[3], regs[4], index), _tbl4(regs[5], regs[6], regs[7], regs[8], index))

_vwidth(::Neon, ::Type{Int8}) = Val(16)

# Fast phase→index extraction (overrides the generic shift+convert in kernel.jl). The
# table index is the top log2(N) bits of each accumulator lane, which sit entirely in the
# HIGH word / high byte of the little-endian UInt32. NEON's `uzp2` (unzip odd elements)
# extracts them in dedicated 1-µop steps: two uzp2.8h collapse the four accumulator Q regs
# to the 16 high words, one more uzp2.16b takes their high bytes — 3 µops, no gather
# constant. (An earlier revision gathered byte 3 with a `tbl4`, but TBL µop cost scales
# with the number of table registers — 4-source ≈ 4 µops on Apple cores — so the uzp2
# chain is strictly cheaper. The lookup itself stays on tbl4: for a full 64-entry table
# that IS the floor, and half-table tricks that pay on x86 lose here for the same
# linear-cost reason: 2×tbl2 ≈ 1×tbl4, plus the extra sign-fix ops.)
#
# The final right-align uses NEON's NATIVE byte shift (`ushr.16b` — unlike x86, where
# byte shifts are emulated as word shifts and bleed neighbour bits), so an UNSIGNED shift
# zero-fills and the result is exact with no mask. Exactness matters here: tbl returns 0
# for out-of-range indices, so junk bits would corrupt the lookup, not be ignored.

# high (odd) 16-bit halves of the 16 dword lanes — lowers to two uzp2.8h
@inline _high_words(acc::Vec{16,UInt32}) =
    shufflevector(reinterpret(Vec{32,Int16}, acc), Val(ntuple(i -> 2i - 1, Val(16))))

@inline function _phase_index(::Neon, acc::Vec{16,UInt32}, ::Val{N}, ::Type{Int8}) where {N}
    msbyte = shufflevector(reinterpret(Vec{32,Int8}, _high_words(acc)),
                           Val(ntuple(i -> 2i - 1, Val(16))))          # uzp2.16b: byte 3s
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
# HIGH word of each little-endian dword lane — two uzp2.8h (via _high_words) plus a native
# 16-bit unsigned shift. Exact, nothing to mask.
@inline _phase_index(::Neon, acc::Vec{16,UInt32}, ::Val{N}, ::Type{Int16}) where {N} =
    reinterpret(Vec{16,Int16},
        reinterpret(Vec{16,UInt16}, _high_words(acc)) >> UInt16(_index_shift(Val(N)) - 16))
