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
    # leaf 0: vendor string ("GenuineIntel" in EBX,EDX,ECX). Intel AVX-512 cores execute all
    # byte permutes on a single shuffle port (port 5), so the phase-index extraction picks a
    # port-balanced instruction mix there; AMD double-pumps 512-bit ops across 256-bit pipes,
    # making TOTAL µops the binding constraint instead (see _phase_index below).
    _, ebx0, ecx0, edx0 = _cpuid(UInt32(0), UInt32(0))
    intel = ebx0 == 0x756e6547 && edx0 == 0x49656e69 && ecx0 == 0x6c65746e
    _, _, ecx1, _ = _cpuid(UInt32(1), UInt32(0))     # leaf 1: OSXSAVE in ECX bit 27
    osxsave = _bit(ecx1, 27)
    xcr0 = osxsave ? _xcr0() : UInt32(0)
    avx_os    = osxsave && _bit(xcr0, 1) && _bit(xcr0, 2)                      # SSE + YMM state
    avx512_os = avx_os  && _bit(xcr0, 5) && _bit(xcr0, 6) && _bit(xcr0, 7)    # opmask + ZMM state
    _, ebx, ecx, _ = _cpuid(UInt32(7), UInt32(0))    # leaf 7: feature bits
    (intel      = intel,
     avx2       = avx_os    && _bit(ebx, 5),
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

# CPUID (`HOST_FEATURES`) reports the *host*; the ISA we may legally emit is fixed by the
# LLVM *codegen target*. These diverge under a restricted or multiversioned CPU target
# (`--cpu-target=…` / `JULIA_CPU_TARGET=…`, the latter being how the official binaries build
# every pkgimage): a `generic`/`haswell` clone has no AVX-512 codegen target, yet CPUID on an
# AVX-512 host still reports AVX-512. If `default_backend` trusted CPUID there it would emit
# AVX-512 permutes into a clone that cannot legalise them, aborting codegen outright — an
# uncatchable `LLVM ERROR: couldn't allocate output register for constraint 'x'` (older LLVM)
# / `Do not know how to split the result of this operator!` (newer). So gate ISA-backend
# selection on the codegen target instead of CPUID: only when it is exactly `native` is the
# host the codegen target, so that CPUID matches what LLVM will emit. `JLOptions().cpu_target`
# is read here in the precompile worker, where `-C` has been set from `JULIA_CPU_TARGET`
# (see Base.loading `create_expr_cache`), so the const captures the clone's actual target.
# Any other target (a named CPU, or a multiversion string) falls back to `Portable()` — always
# correct, and the only ISA a baseline clone is guaranteed to be able to emit. Set
# `JULIA_CPU_TARGET=native` (the documented workaround) to opt back into the ISA backends.
# `CODEGEN_TARGET` keeps the baked target string for the `__init__` diagnostic (see SinCosLUT.jl).
const CODEGEN_TARGET = unsafe_string(Base.JLOptions().cpu_target)
const CODEGEN_IS_NATIVE = CODEGEN_TARGET == "native"

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
# and shuffle work is the bottleneck of both the value-based carrier loop and the fill.
# The table index is the top log2(N) bits of each lane, which sit entirely in byte 3 of the
# little-endian UInt32. Gather byte 3 of all 64 lanes with two `vpermi2b` over the accumulator's
# four byte-quarters (the quarter splits are free sub-register selects): the lo gather fills
# lanes 0–31 from dwords 0–31, the hi gather fills lanes 32–63 from dwords 32–63 (vpermi2b
# index 0–63 = first operand, 64–127 = second; each gather's other half is don't-care). The
# halves are merged with a `vpternlogq` bit-select — NOT a lane shuffle: the select mask is
# routed through an empty-asm barrier (`_opaque`), otherwise LLVM canonicalises the constant-
# mask select back into a 3rd shuffle-port op (vshufi64x2/vinserti64x4), re-creating the port-5
# bottleneck on Intel. ternlog runs on the plain vector-ALU ports, which have slack here.
#
# The gathered byte still holds the index at bits [7 : 8-log2(N)] and must be right-aligned.
# Two tails, chosen per vendor at precompile (HOST_FEATURES is const, so the branch folds):
#   • AMD & others (512-bit ops double-pumped over 256-bit pipes → TOTAL µops bind):
#     one `vpmultishiftqb` per-byte bit-field extract (VBMI, which this path requires anyway).
#     x86 has no byte-granular shift — a plain `>>` lowers to vpsrlw+vpand (2 ops) — so the
#     1-op multishift wins. Its field wraps neighbouring bits into index bits ≥ log2(N):
#     the RESULT IS NOT EXACT — junk in bits the consumers ignore (see contract below).
#   • Intel (full 512-bit units, all byte permutes contend on port 5 → port 5 binds):
#     vpsrlw+vpand — 2 µops, but on the ALU ports where there is slack; adding multishift
#     would put a 3rd op on port 5 instead. This tail is exact.
#
# CONTRACT: bits ≥ log2(N) of each returned index byte are UNSPECIFIED (junk on the AMD tail,
# zero on the Intel tail). Every consumer is safe: `_apply(::AVX512, …)` feeds vpermb (which
# architecturally ignores index bits 7:6; N=64) or vpermi2b (ignores bit 7; N=128), and the
# `Prepared` functor masks with `& (N-1)` before its permute. Do NOT use the raw result as an
# arithmetic index without masking.
# The explicit target-features matter: the asm 'x' constraint needs a ZMM register, and
# under a restricted --cpu-target (e.g. a generic sysimage) the base target has none —
# without the attribute LLVM fails with "couldn't allocate output register".
@inline _opaque(v::Vec{64,Int8}) = Vec{64,Int8}(Base.llvmcall(("""
    define <64 x i8> @entry(<64 x i8> %v) #0 {
      %r = call <64 x i8> asm "", "=x,0"(<64 x i8> %v)
      ret <64 x i8> %r }
    attributes #0 = { alwaysinline "target-features"="+avx512vbmi,+avx512bw,+avx512f" }""", "entry"),
    NTuple{64,VecElement{Int8}}, Tuple{NTuple{64,VecElement{Int8}}}, v.data))

# vpmultishiftqb: output byte j of each qword = the 8-bit field of the SOURCE qword starting
# at bit ctrl[j] (circular within the qword). One op ≡ per-byte `>> shift` with junk, not
# zeros, shifted into the top bits.
const _MULTISHIFT_IR = """
declare <64 x i8> @llvm.x86.avx512.pmultishift.qb.512(<64 x i8>, <64 x i8>)
define <64 x i8> @entry(<64 x i8> %c, <64 x i8> %v) #0 {
  %r = call <64 x i8> @llvm.x86.avx512.pmultishift.qb.512(<64 x i8> %c, <64 x i8> %v)
  ret <64 x i8> %r }
attributes #0 = { alwaysinline "target-features"="+avx512vbmi,+avx512bw,+avx512f" }
"""
@inline _multishift(ctrl::Vec{64,Int8}, v::Vec{64,Int8}) =
    Vec{64,Int8}(Base.llvmcall((_MULTISHIFT_IR, "entry"), NTuple{64,VecElement{Int8}},
        Tuple{NTuple{64,VecElement{Int8}},NTuple{64,VecElement{Int8}}}, ctrl.data, v.data))

@inline function _phase_index(::AVX512, acc::Vec{64,UInt32}, ::Val{N}, ::Type{Int8}) where {N}
    b = reinterpret(Vec{256,Int8}, acc)
    q0 = shufflevector(b, Val(ntuple(i -> i - 1, Val(64))))
    q1 = shufflevector(b, Val(ntuple(i -> i + 63, Val(64))))
    q2 = shufflevector(b, Val(ntuple(i -> i + 127, Val(64))))
    q3 = shufflevector(b, Val(ntuple(i -> i + 191, Val(64))))
    gather_lo = Vec{64,Int8}(ntuple(
        k -> k <= 16 ? Int8(4 * (k - 1) + 3) : (k <= 32 ? Int8(64 + 4 * (k - 17) + 3) : Int8(0)),
        Val(64)))
    gather_hi = Vec{64,Int8}(ntuple(
        k -> k <= 32 ? Int8(0) :
             (k <= 48 ? Int8(4 * (k - 33) + 3) : Int8(64 + 4 * (k - 49) + 3)),
        Val(64)))
    lo = _permi2(q0, gather_lo, q1)                      # lanes 0..31 = byte 3 of dwords 0..31
    hi = _permi2(q2, gather_hi, q3)                      # lanes 32..63 = byte 3 of dwords 32..63
    m = _opaque(Vec{64,Int8}(ntuple(k -> k <= 32 ? Int8(-1) : Int8(0), Val(64))))
    msbyte = (lo & m) | (hi & ~m)                        # one vpternlogq (vector-ALU, not shuffle)
    shift = _index_shift(Val(N)) - 24                    # byte3 >> shift right-aligns the index
    # plain `if` (not @static: the whole file is one @static block, so HOST_FEATURES does not
    # exist yet at macro-expansion time) — HOST_FEATURES is const, so the branch folds away.
    if HOST_FEATURES.intel
        reinterpret(Vec{64,Int8}, (msbyte >> UInt8(shift)) & UInt8(N - 1))
    else
        ctrl = Vec{64,Int8}(ntuple(i -> Int8(8 * ((i - 1) & 7) + shift), Val(64)))
        _multishift(ctrl, msbyte)
    end
end

# Fast Int16 phase→index extraction. The generic form costs 2×vpsrld + a narrowing
# `convert(Vec{32,UInt32} -> Int16)` (2×vpmovdw + insert — 3 shuffle µops). The index lives in
# the HIGH word of each dword (bits 31:16, and log2(N) ≤ 16 always), so a single `vpermi2w`
# gathers the high words of all 32 lanes across the accumulator's two halves, and one logical
# word shift right-aligns it — 2 ops total, and EXACT (a word shift zero-fills; nothing bleeds).
@inline function _phase_index(::AVX512, acc::Vec{32,UInt32}, ::Val{N}, ::Type{Int16}) where {N}
    w = reinterpret(Vec{64,Int16}, acc)
    h0 = shufflevector(w, Val(ntuple(i -> i - 1, Val(32))))    # words of dwords 0..15
    h1 = shufflevector(w, Val(ntuple(i -> i + 31, Val(32))))   # words of dwords 16..31
    gather = Vec{32,Int16}(ntuple(k -> Int16(2k - 1), Val(32)))  # word 1 (high) of each dword
    hiw = _permi2(h0, gather, h1)
    reinterpret(Vec{32,Int16},
        reinterpret(Vec{32,UInt16}, hiw) >> UInt16(_index_shift(Val(N)) - 16))
end

end # @static x86
