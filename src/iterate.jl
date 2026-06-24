# Array-free, allocation-free carrier generation.
#
# `prepare(table)` returns a callable mapping a phase-index `Vec` to (sin, cos) `Vec`s
# — the direct analogue of FastSinCos's `fast_sincos_*(::Vec)`, but backed by a
# register-resident table (built once).
#
# `generate_carrier(table, …, num_samples)` is an iterator yielding `(sin, cos)` `Vec`s
# with a textbook NCO phase accumulator (UInt32) carried in the (isbits) iteration
# state. Fuse it straight into your own loop — the carrier is never written to memory:
#
#     accumulator = zero(...)
#     for (sin_vec, cos_vec) in generate_carrier(table, 0.002, length(signal))
#         accumulator += dot_into_registers(sin_vec, cos_vec, ...)   # Vec{W,T}
#     end

# ---- stateless primitive: prepare once, then map Vec -> (Vec, Vec) ----
struct Prepared{T,N,Backend,TableRegisters}
    table_registers::TableRegisters
    backend::Backend
    index_mask::T
end
Prepared{T,N}(table_registers::R, backend::B, index_mask::T) where {T,N,B,R} =
    Prepared{T,N,B,R}(table_registers, backend, index_mask)

"""
    prepare(table; backend=default_backend(table)) -> callable

Materialise `table` into registers once and return a callable `p` such that
`p(phase_index::Vec) -> (sin::Vec, cos::Vec)` (indices taken mod `steps`). Analogous
to `FastSinCos`'s `fast_sincos_*`, but a table lookup.
"""
prepare(table::SinCosTable{T,N}; backend::Backend = default_backend(T, N)) where {T,N} =
    Prepared{T,N}(_prepare(backend, table), backend, T(N - 1))

@inline (p::Prepared{T,N})(phase_index::Vec{W,T}) where {T,N,W} =
    _apply(p.backend, p.table_registers, phase_index & p.index_mask)

# ---- stateful iterator: NCO phase accumulator, yields (sin, cos) Vecs ----
struct CarrierIterator{T,N,W,Prep}
    prepared::Prep
    acc_init::Vec{W,UInt32}
    step_advance::UInt32          # W * freq_word (advance per yielded chunk)
    num_chunks::Int
end

"""
    generate_carrier(table, freq_word::Integer,      num_samples; phase=0, backend=…)
    generate_carrier(table, cycles_per_sample::Real, num_samples; phase=0, backend=…)
    generate_carrier(table, num_samples; frequency, sampling_frequency, phase=0, backend=…)

Iterator over `num_samples ÷ W` chunks (W = SIMD width for the backend/type), each
yielding `(sin, cos)::Tuple{Vec{W,T},Vec{W,T}}`. The carrier is produced by a textbook
NCO (UInt32 phase accumulator advanced by `freq_word` per sample) held in the iteration
state — no carrier array is allocated, the phase is uniform/dead-zone-free and never
drifts. The low-level form takes a raw `freq_word` (UInt32, `0 ≤ freq_word ≤ 2^32-1`);
`cycles_per_sample::Real` converts as `round(cps·2^32)`; the keyword form derives it
from `frequency`/`sampling_frequency`. `phase` is the initial carrier phase (default 0):
an `Integer` is table steps, a `Real` is cycles. Any leftover `num_samples % W` tail is
not produced (handle it yourself if needed).
"""
function generate_carrier(table::SinCosTable{T,N}, freq_word::Integer,
                          num_samples::Integer; phase::Real = 0,
                          backend::Backend = default_backend(T, N)) where {T,N}
    (0 ≤ freq_word ≤ typemax(UInt32)) ||
        throw(ArgumentError("need 0 ≤ freq_word ≤ typemax(UInt32) = $(typemax(UInt32))"))
    _make_carrier(table, UInt32(freq_word), Int(num_samples),
                  _phase_steps(phase, N), backend, _vwidth(backend, T))
end
function generate_carrier(table::SinCosTable{T,N}, cycles_per_sample::Real, num_samples::Integer; kw...) where {T,N}
    generate_carrier(table, _freq_word(cycles_per_sample), num_samples; kw...)
end
function generate_carrier(table::SinCosTable{T,N}, num_samples::Integer;
                          frequency::Real, sampling_frequency::Real, kw...) where {T,N}
    generate_carrier(table, cycles_per_sample(frequency, sampling_frequency), num_samples; kw...)
end

function _make_carrier(table::SinCosTable{T,N}, freq_word::UInt32, num_samples, phase_offset,
                       backend, ::Val{W}) where {T,N,W}
    acc_offset = _acc_offset(phase_offset, Val(N))
    acc_init = _init_acc(Val(W), freq_word, acc_offset, 0)
    prepared = Prepared{T,N}(_prepare(backend, table), backend, T(N - 1))
    CarrierIterator{T,N,W,typeof(prepared)}(prepared, acc_init, freq_word * UInt32(W), num_samples ÷ W)
end

Base.length(it::CarrierIterator) = it.num_chunks
Base.IteratorSize(::Type{<:CarrierIterator}) = Base.HasLength()
Base.eltype(::Type{<:CarrierIterator{T,N,W}}) where {T,N,W} = Tuple{Vec{W,T},Vec{W,T}}

@inline function Base.iterate(it::CarrierIterator{T,N,W},
                              state = (it.acc_init, 0)) where {T,N,W}
    acc, chunk = state
    chunk >= it.num_chunks && return nothing
    idx = convert(Vec{W,T}, acc >> _index_shift(Val(N)))
    result = it.prepared(idx)                         # (sin, cos), mask applied inside
    (result, (acc + it.step_advance, chunk + 1))
end

# ---- 4-way interleaved iterator: yields 4 (sin,cos) pairs (4W samples) per step ----
# The four DDA carry chains are independent, so they overlap even when the consumer is
# trivial (e.g. array fill) — reaching the ~40 ps/elem loop rate at scale. Use the
# single-Vec `generate_carrier` when fusing into nontrivial work (it provides its own ILP).
struct CarrierIterator4{T,N,W,Prep}
    prepared::Prep
    acc1::Vec{W,UInt32}; acc2::Vec{W,UInt32}; acc3::Vec{W,UInt32}; acc4::Vec{W,UInt32}
    step_advance::UInt32          # 4W * freq_word (advance per yielded step)
    num_steps::Int
end

"""
    generate_carrier4(table, freq_word::Integer,      num_samples; phase=0, backend=…)
    generate_carrier4(table, cycles_per_sample::Real, num_samples; phase=0, backend=…)
    generate_carrier4(table, num_samples; frequency, sampling_frequency, phase=0, backend=…)

Like [`generate_carrier`](@ref) but yields a 4-tuple of `(sin, cos)` `Vec` pairs per
step (`4W` samples), running four interleaved NCO accumulators so the stores/lookups
overlap. Reaches the full loop throughput (~40 ps/elem) even for trivial consumers such
as array fill. **Destructure the 4-tuple in the loop header** —
`for ((s0,c0),(s1,c1),(s2,c2),(s3,c3)) in generate_carrier4(...)` — rather than
iterating it with an inner `for pair in quad` loop, which does not unroll and is much
slower. `phase` is the initial carrier phase (default 0): `Integer` = table steps,
`Real` = cycles. Produces `num_samples ÷ (4W)` steps; handle any tail yourself.
"""
function generate_carrier4(table::SinCosTable{T,N}, freq_word::Integer,
                           num_samples::Integer; phase::Real = 0,
                           backend::Backend = default_backend(T, N)) where {T,N}
    (0 ≤ freq_word ≤ typemax(UInt32)) ||
        throw(ArgumentError("need 0 ≤ freq_word ≤ typemax(UInt32) = $(typemax(UInt32))"))
    _make_carrier4(table, UInt32(freq_word), Int(num_samples),
                   _phase_steps(phase, N), backend, _vwidth(backend, T))
end
function generate_carrier4(table::SinCosTable{T,N}, cycles_per_sample::Real, num_samples::Integer; kw...) where {T,N}
    generate_carrier4(table, _freq_word(cycles_per_sample), num_samples; kw...)
end
function generate_carrier4(table::SinCosTable{T,N}, num_samples::Integer;
                           frequency::Real, sampling_frequency::Real, kw...) where {T,N}
    generate_carrier4(table, cycles_per_sample(frequency, sampling_frequency), num_samples; kw...)
end

function _make_carrier4(table::SinCosTable{T,N}, freq_word::UInt32, num_samples, phase_offset,
                        backend, ::Val{W}) where {T,N,W}
    acc_offset = _acc_offset(phase_offset, Val(N))
    acc1 = _init_acc(Val(W), freq_word, acc_offset, 0)
    acc2 = _init_acc(Val(W), freq_word, acc_offset, W)
    acc3 = _init_acc(Val(W), freq_word, acc_offset, 2W)
    acc4 = _init_acc(Val(W), freq_word, acc_offset, 3W)
    prepared = Prepared{T,N}(_prepare(backend, table), backend, T(N - 1))
    CarrierIterator4{T,N,W,typeof(prepared)}(prepared, acc1, acc2, acc3, acc4,
        freq_word * UInt32(4W), num_samples ÷ (4W))
end

Base.length(it::CarrierIterator4) = it.num_steps
Base.IteratorSize(::Type{<:CarrierIterator4}) = Base.HasLength()
Base.eltype(::Type{<:CarrierIterator4{T,N,W}}) where {T,N,W} = NTuple{4,Tuple{Vec{W,T},Vec{W,T}}}

@inline function Base.iterate(it::CarrierIterator4{T,N,W},
                              state = (it.acc1, it.acc2, it.acc3, it.acc4, 0)) where {T,N,W}
    acc1, acc2, acc3, acc4, step_count = state
    step_count >= it.num_steps && return nothing
    shift = _index_shift(Val(N))
    pair1 = it.prepared(convert(Vec{W,T}, acc1 >> shift)); pair2 = it.prepared(convert(Vec{W,T}, acc2 >> shift))
    pair3 = it.prepared(convert(Vec{W,T}, acc3 >> shift)); pair4 = it.prepared(convert(Vec{W,T}, acc4 >> shift))
    step_advance = it.step_advance
    ((pair1, pair2, pair3, pair4),
     (acc1 + step_advance, acc2 + step_advance, acc3 + step_advance, acc4 + step_advance, step_count + 1))
end
