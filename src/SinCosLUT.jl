"""
    SinCosLUT

Fast SIMD sine/cosine via register-resident byte/word/dword table lookups.

The lookup is a single hardware permute over a table held in vector registers:
`vpermb`/`vpermw`/`vpermd` (AVX-512), `vpshufb` (AVX2, Int8 only), or NEON `tbl`
(AArch64, Int8 and Int16). Output element type is selectable — `Int8` (fastest,
~3-7 bit), `Int16`, or `Int32` (more amplitude precision, fewer lanes/entries).

Accuracy is a deliberate trade for speed: a 512-bit register holds 64×Int8 but
only 32×Int16 / 16×Int32, so wider elements give finer amplitude but coarser
phase resolution (fewer table entries) and lower throughput.

SVE2 is not supported: Julia cannot express LLVM scalable vectors
(`<vscale x N x T>`), see JuliaLang/julia#40308. NEON `tbl` runs natively on
SVE hardware regardless.
"""
module SinCosLUT

using SIMD

export SinCosTable, generate_carrier!, generate_carrier_signs!, generate_carrier_signs_mags!,
       lookup_sincos!, prepare,
       cycles_per_sample, default_backend, backend_name,
       CarrierEngine, CarrierState, carrier_engine, carrier_state, carrier_lookup,
       carrier_advance, carrier_width

# ---- backends ----
abstract type Backend end
struct AVX512  <: Backend end   # vpermb / vpermw / vpermd (+vpermi2*)
struct AVX2    <: Backend end   # vpshufb (Int8 only)
struct Neon    <: Backend end   # AArch64 NEON tbl (Int8, Int16)
struct Portable <: Backend end  # scalar fallback (any T)

"""
    backend_name(backend) -> String

Human-readable name of a `Backend` (`"AVX-512"`, `"AVX2"`, `"NEON"`, or `"portable"`).
Useful for reporting which lookup primitive [`default_backend`](@ref) selected.
"""
backend_name(::AVX512)   = "AVX-512"
backend_name(::AVX2)     = "AVX2"
backend_name(::Neon)     = "NEON"
backend_name(::Portable) = "portable"

# elements per 512-bit register for the AVX-512 permute family
regsize(::Type{Int8})  = 64
regsize(::Type{Int16}) = 32
regsize(::Type{Int32}) = 16

@inline _val(::Val{W}) where {W} = W   # extract the SIMD width from a Val

"""
    SinCosTable(T=Int8; steps=64, amplitude=typemax(T))

Build sin/cos lookup tables of `steps` entries per full cycle, output type `T`
(`Int8`/`Int16`/`Int32`). `steps` should be a power of two. For the AVX-512
backend `steps` must be `regsize(T)` (single permute) or `2*regsize(T)`
(two-register `vpermi2`): Int8 → 64 or 128, Int16 → 32 or 64, Int32 → 16 or 32.
Other sizes still work on the portable backend.
"""
struct SinCosTable{T,N}
    sin::NTuple{N,T}
    cos::NTuple{N,T}
end

function SinCosTable(::Type{T} = Int8; steps::Integer = 64,
                    amplitude::Real = typemax(T)) where {T<:Union{Int8,Int16,Int32}}
    _build_table(T, Val(Int(steps)), amplitude)
end
function _build_table(::Type{T}, ::Val{N}, A) where {T,N}
    SinCosTable{T,N}(ntuple(k -> round(T, A * sinpi(2 * (k - 1) / N)), Val(N)),
                     ntuple(k -> round(T, A * cospi(2 * (k - 1) / N)), Val(N)))
end

nsteps(::SinCosTable{T,N}) where {T,N} = N

include("permute_avx512.jl")
@static if Sys.ARCH in (:x86_64, :i686)
    include("permute_avx2.jl")
end
@static if Sys.ARCH === :aarch64
    include("permute_neon.jl")
end
include("kernel.jl")
include("iterate.jl")

# ---- backend selection ----
@static if Sys.ARCH in (:x86_64, :i686)
    function _avx512_supports(::Type{T}, steps, features) where T
        T === Int8  ? (features.avx512vbmi && (steps == 64 || steps == 128)) :
        T === Int16 ? (features.avx512bw  && (steps == 32 || steps == 64))  :
                      (features.avx512f   && (steps == 16 || steps == 32))
    end
    function default_backend(::Type{T}, steps::Integer) where T
        features = HOST_FEATURES   # const (detected at precompile) → foldable, type-stable
        # CODEGEN_IS_NATIVE guards the ISA branches: under a restricted/multiversioned codegen
        # target the CPUID features do not match what LLVM can emit, so fall back to Portable()
        # rather than bake un-legalisable AVX-512/AVX2 into a baseline clone (see permute_avx512.jl).
        (CODEGEN_IS_NATIVE && _avx512_supports(T, steps, features)) ? AVX512() :
        (CODEGEN_IS_NATIVE && T === Int8 && features.avx2 && steps == 64) ? AVX2() : Portable()
    end
elseif Sys.ARCH === :aarch64
    function default_backend(::Type{T}, steps::Integer) where T
        (T === Int8 && steps == 64)                    ? Neon() :
        (T === Int16 && (steps == 32 || steps == 64))  ? Neon() : Portable()
    end
else
    default_backend(::Type{T}, steps::Integer) where T = Portable()
end

"""
    default_backend(table::SinCosTable) -> Backend
    default_backend(T::Type, steps::Integer) -> Backend

The lookup backend automatically selected for output type `T` and table size `steps` on
the host CPU: `AVX512()`, `AVX2()`, or `Neon()` where the required SIMD permute is
available, otherwise `Portable()` (the always-correct scalar fallback). AVX2 supports
only `Int8` with `steps = 64`; NEON supports `Int8` (`steps = 64`) and `Int16`
(`steps = 32` or `64`); other combinations fall back to `Portable()`.

The x86 ISA backends (AVX-512/AVX2) are selected only when the LLVM codegen target is
`native`. Under a restricted or multiversioned CPU target (`--cpu-target=…` /
`JULIA_CPU_TARGET=…`, the latter being how the official binaries build every pkgimage) the
CPU-detected features need not match what LLVM can emit, so those backends fall back to
`Portable()` rather than bake un-legalisable ISA into a baseline clone (which would abort
codegen). Set `JULIA_CPU_TARGET=native` to keep the ISA backends under a custom target.

Pass the result (or any `Backend`) as the `backend` keyword of [`generate_carrier!`](@ref),
[`lookup_sincos!`](@ref), [`carrier_engine`](@ref), or [`prepare`](@ref) to override the
choice. See [`backend_name`](@ref) for a readable label.
"""
default_backend(table::SinCosTable{T,N}) where {T,N} = default_backend(T, N)

include("signbits.jl")   # after default_backend: its `_SIGN_PREP` const prepares a table at load
include("twobit.jl")     # 2-bit sign+magnitude carrier; shares the signbits.jl helpers

# Tell the user, once, if we had to demote an ISA-capable host to the scalar fallback because
# the codegen target was restricted (see `default_backend` / permute_avx512.jl). Only fires when
# it actually costs something — the CPU has SIMD (`avx2`) but the target could not emit it — so a
# genuinely portable/old host stays quiet. `maxlog=1` and `SINCOSLUT_QUIET=1` keep it from nagging
# (e.g. in a deliberately restricted / compiled-app build).
@static if Sys.ARCH in (:x86_64, :i686)
    function __init__()
        if !CODEGEN_IS_NATIVE && HOST_FEATURES.avx2 && get(ENV, "SINCOSLUT_QUIET", "") != "1"
            @warn """
            SinCosLUT is using the scalar (Portable) fallback: the LLVM CPU target is \
            `$(CODEGEN_TARGET)`, not `native`, so the AVX2/AVX-512 permute backends are \
            disabled even though this CPU supports them. Set `JULIA_CPU_TARGET=native` and \
            re-precompile to enable them (see `?default_backend`). Silence this warning with \
            `ENV["SINCOSLUT_QUIET"] = "1"`.""" maxlog = 1
        end
        return nothing
    end
end

end # module
