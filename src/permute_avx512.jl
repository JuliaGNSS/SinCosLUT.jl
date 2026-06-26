# AVX-512 backend: vpermb / vpermw / vpermd (single permute) and their
# two-register vpermi2 variants, generic over Int8 / Int16 / Int32.

@static if Sys.ARCH in (:x86_64, :i686)

# ---- CPU feature detection (cpuid leaf 7, subleaf 0) ----
@inline function _cpuid(leaf::UInt32, subleaf::UInt32)
    Base.llvmcall(
        """
        %s = call { i32, i32, i32, i32 } asm sideeffect "cpuid",
             "={ax},={bx},={cx},={dx},{ax},{cx},~{dirflag},~{fpsr},~{flags}"(i32 %0, i32 %1)
        %a = extractvalue { i32, i32, i32, i32 } %s, 0
        %b = extractvalue { i32, i32, i32, i32 } %s, 1
        %c = extractvalue { i32, i32, i32, i32 } %s, 2
        %d = extractvalue { i32, i32, i32, i32 } %s, 3
        %r0 = insertvalue [4 x i32] undef, i32 %a, 0
        %r1 = insertvalue [4 x i32] %r0, i32 %b, 1
        %r2 = insertvalue [4 x i32] %r1, i32 %c, 2
        %r3 = insertvalue [4 x i32] %r2, i32 %d, 3
        ret [4 x i32] %r3
        """,
        Tuple{UInt32,UInt32,UInt32,UInt32}, Tuple{UInt32,UInt32}, leaf, subleaf)
end
@inline _bit(x::UInt32, n) = (x >> n) & 0x1 == 0x1

# XCR0 (low 32 bits) via xgetbv with ecx=0 — tells us which vector state the OS has
# enabled. A CPUID feature bit alone is not enough: a hypervisor can expose AVX-512 in
# CPUID without enabling ZMM state, and then LLVM/Julia codegen cannot use it either.
@inline _xcr0() = Base.llvmcall(
    """
    %r = call i32 asm sideeffect "xgetbv", "={ax},{cx},~{dx},~{dirflag},~{fpsr},~{flags}"(i32 %0)
    ret i32 %r
    """, UInt32, Tuple{UInt32}, UInt32(0))

function _x86_features()
    _, _, ecx1, _ = _cpuid(UInt32(1), UInt32(0))     # leaf 1: OSXSAVE in ECX bit 27
    osxsave = _bit(ecx1, 27)
    xcr0 = osxsave ? _xcr0() : UInt32(0)
    avx_os    = osxsave && _bit(xcr0, 1) && _bit(xcr0, 2)                      # SSE + YMM state
    avx512_os = avx_os  && _bit(xcr0, 5) && _bit(xcr0, 6) && _bit(xcr0, 7)    # opmask + ZMM state
    _, ebx, ecx, _ = _cpuid(UInt32(7), UInt32(0))    # leaf 7: feature bits
    (avx2       = avx_os    && _bit(ebx, 5),
     avx512f    = avx512_os && _bit(ebx, 16),
     avx512bw   = avx512_os && _bit(ebx, 30),
     avx512vbmi = avx512_os && _bit(ecx, 1))
end

# Detect ONCE at precompile and bake into a const (like VectorizationBase.jl), so
# `default_backend` is a pure function of compile-time constants and folds to a
# concrete backend type — keeping default-backend iteration allocation-free. The
# const reflects the build machine; Julia keys pkgimages on the CPU target, so a
# different host re-precompiles and re-detects.
const HOST_FEATURES = _x86_features()

_llvmtype(::Type{Int8}) = "i8"; _llvmtype(::Type{Int16}) = "i16"; _llvmtype(::Type{Int32}) = "i32"
# Generate explicit @inline permute methods per element type. (A @generated
# function here does not inline and adds ~1 call per permute → ~7× slower.)
# The IR entry functions also carry `alwaysinline`: without it the (module,"entry")
# llvmcall form emits a real `call` + register spill instead of the bare permute
# (~2.5× slower here, far worse on the multi-op AVX2 path).
# LLVM 32-bit suffix: permvar→"si", vpermi2var→"d"; 8/16-bit share qi/hi.
for (Tt, L, et, pv, p2, feat) in (
        (Int8,  64, "i8",  "qi", "qi", "+avx512vbmi,+avx512bw,+avx512f"),
        (Int16, 32, "i16", "hi", "hi", "+avx512bw,+avx512f"),
        (Int32, 16, "i32", "si", "d",  "+avx512f"))
    VE = NTuple{L,VecElement{Tt}}
    pv_ir = """
    declare <$L x $et> @llvm.x86.avx512.permvar.$pv.512(<$L x $et>, <$L x $et>)
    define <$L x $et> @entry(<$L x $et> %0, <$L x $et> %1) #0 {
      %r = call <$L x $et> @llvm.x86.avx512.permvar.$pv.512(<$L x $et> %0, <$L x $et> %1)
      ret <$L x $et> %r }
    attributes #0 = { alwaysinline "target-features"="$feat" }
    """
    p2_ir = """
    declare <$L x $et> @llvm.x86.avx512.vpermi2var.$p2.512(<$L x $et>, <$L x $et>, <$L x $et>)
    define <$L x $et> @entry(<$L x $et> %0, <$L x $et> %1, <$L x $et> %2) #0 {
      %r = call <$L x $et> @llvm.x86.avx512.vpermi2var.$p2.512(<$L x $et> %0, <$L x $et> %1, <$L x $et> %2)
      ret <$L x $et> %r }
    attributes #0 = { alwaysinline "target-features"="$feat" }
    """
    @eval @inline _permvar(table::Vec{$L,$Tt}, index::Vec{$L,$Tt}) =
        Vec{$L,$Tt}(Base.llvmcall(($pv_ir, "entry"), $VE, Tuple{$VE,$VE}, table.data, index.data))
    @eval @inline _permi2(low::Vec{$L,$Tt}, index::Vec{$L,$Tt}, high::Vec{$L,$Tt}) =
        Vec{$L,$Tt}(Base.llvmcall(($p2_ir, "entry"), $VE, Tuple{$VE,$VE,$VE}, low.data, index.data, high.data))
end

@inline _firsthalf(t::NTuple{N,T}) where {N,T} = Vec{N ÷ 2,T}(ntuple(j -> @inbounds(t[j]), Val(N ÷ 2)))
@inline _secondhalf(t::NTuple{N,T}) where {N,T} = Vec{N ÷ 2,T}(ntuple(j -> @inbounds(t[N ÷ 2 + j]), Val(N ÷ 2)))

# Materialise the table into registers ONCE (call _prepare before the loop), then
# _apply just issues the permute(s). single permute when N == regsize(T),
# two-register vpermi2 when N == 2*regsize(T).
# dispatch the single/double choice on a Val so the return type is concrete
@inline _prepare(backend::AVX512, table::SinCosTable{T,N}) where {T,N} =
    _prepare(backend, table, Val(N == regsize(T)))
@inline _prepare(::AVX512, table::SinCosTable{T,N}, ::Val{true}) where {T,N} =
    (Vec{N,T}(table.sin), Vec{N,T}(table.cos))
@inline _prepare(::AVX512, table::SinCosTable{T,N}, ::Val{false}) where {T,N} =
    (_firsthalf(table.sin), _secondhalf(table.sin), _firsthalf(table.cos), _secondhalf(table.cos))

@inline _apply(::AVX512, regs::Tuple{Vec{L,T},Vec{L,T}}, index::Vec{L,T}) where {L,T} =
    (_permvar(regs[1], index), _permvar(regs[2], index))
@inline _apply(::AVX512, regs::NTuple{4,Vec{L,T}}, index::Vec{L,T}) where {L,T} =
    (_permi2(regs[1], index, regs[2]), _permi2(regs[3], index, regs[4]))

_vwidth(::AVX512, ::Type{T}) where T = Val(regsize(T))

# Fast Int8 phase→index extraction (overrides the generic shift+convert in kernel.jl).
# `convert(Vec{64,UInt32} -> Int8)` lowers to 4×vpmovdb + 3×inserts (7 shuffle-port µops),
# and the shuffle port is the bottleneck of both the value-based carrier loop and the fill.
# The table index is the top log2(N) bits of each lane, which sit entirely in byte 3 of the
# little-endian UInt32. Gather byte 3 of all 64 lanes with two `vpermi2b` over the accumulator's
# four byte-quarters (the quarter splits are free sub-register selects) plus one merge — 3
# shuffle-port µops — then right-align (index = byte3 >> (8-log2(N)) = byte3 >> (index_shift-24))
# and mask to N-1. The byte shift can bleed a neighbour lane's low bits into bits ≥ log2(N);
# the `& (N-1)` clears them, so the result is exact. The gather index selects vpermi2b's
# 0–63 = first operand, 64–127 = second; lanes 32–63 are don't-care (overwritten by the merge).
@inline function _phase_index(::AVX512, acc::Vec{64,UInt32}, ::Val{N}, ::Type{Int8}) where {N}
    b = reinterpret(Vec{256,Int8}, acc)
    q0 = shufflevector(b, Val(ntuple(i -> i - 1, Val(64))))
    q1 = shufflevector(b, Val(ntuple(i -> i + 63, Val(64))))
    q2 = shufflevector(b, Val(ntuple(i -> i + 127, Val(64))))
    q3 = shufflevector(b, Val(ntuple(i -> i + 191, Val(64))))
    gather = Vec{64,Int8}(ntuple(
        k -> k <= 16 ? Int8(4 * (k - 1) + 3) : (k <= 32 ? Int8(64 + 4 * (k - 17) + 3) : Int8(0)),
        Val(64)))
    lo = _permi2(q0, gather, q1)                         # lanes 0..31 = byte 3 of dwords 0..31
    hi = _permi2(q2, gather, q3)                         # lanes 0..31 = byte 3 of dwords 32..63
    msbyte = shufflevector(lo, hi, Val(ntuple(k -> k <= 32 ? k - 1 : 64 + (k - 33), Val(64))))
    reinterpret(Vec{64,Int8}, (msbyte >> UInt8(_index_shift(Val(N)) - 24)) & UInt8(N - 1))
end

end # @static x86
