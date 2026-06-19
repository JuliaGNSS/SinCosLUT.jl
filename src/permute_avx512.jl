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
    attributes #0 = { "target-features"="$feat" }
    """
    p2_ir = """
    declare <$L x $et> @llvm.x86.avx512.vpermi2var.$p2.512(<$L x $et>, <$L x $et>, <$L x $et>)
    define <$L x $et> @entry(<$L x $et> %0, <$L x $et> %1, <$L x $et> %2) #0 {
      %r = call <$L x $et> @llvm.x86.avx512.vpermi2var.$p2.512(<$L x $et> %0, <$L x $et> %1, <$L x $et> %2)
      ret <$L x $et> %r }
    attributes #0 = { "target-features"="$feat" }
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

end # @static x86
