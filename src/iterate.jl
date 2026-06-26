# Array-free, allocation-free carrier generation.
#
# `prepare(table)` returns a callable mapping a phase-index `Vec` to (sin, cos) `Vec`s
# — the direct analogue of FastSinCos's `fast_sincos_*(::Vec)`, but backed by a
# register-resident table (built once).
#
# For fused, allocation-free carrier wipe-off, use the value-based NCO API: build a
# loop-invariant `CarrierEngine` once, create one isbits `CarrierState` per interleaved
# stream, then per loop iteration look the (sin, cos) up and renew the state by value.
# Nothing is ever written to memory and nothing escapes to the heap:
#
#     eng = carrier_engine(table; frequency = 1000, sampling_frequency = 2e6)
#     st  = carrier_state(eng)                       # one stream, starting at sample 0
#     acc = zero(...)
#     for _ in 1:(length(signal) ÷ W)
#         sin_vec, cos_vec = carrier_lookup(eng, st)
#         acc += dot_into_registers(sin_vec, cos_vec, ...)
#         st = carrier_advance(eng, st, 1)           # next W-wide chunk
#     end
#
# The faster K-way *interleaved* loop (independent NCO carry chains overlap → full ILP)
# just holds K states `carrier_state(eng, (k-1)*W)` and advances each by `K` chunks per
# iteration — see `carrier_advance`. This replaces the old `CarrierIterator` (K=1) /
# `CarrierIterator4` (K=4) split with one engine that supports any interleave factor.

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

# ─────────────────────────────────────────────────────────────────────────────
# Value-based NCO carrier: a loop-invariant `CarrierEngine` (the register-resident table
# + the NCO frequency word) plus an isbits `CarrierState` (the phase accumulator) renewed
# by value every iteration. Both are isbits, so construction and stepping allocate nothing
# and never escape to the heap — fuse it straight into a correlation loop. One engine/state
# pair serves any interleave factor: `carrier_advance(eng, st, K)` steps a stream by K
# W-wide chunks, so a K-way interleaved loop holds K states and advances each by K.
# ─────────────────────────────────────────────────────────────────────────────

"""
    CarrierEngine

Immutable, loop-invariant NCO carrier engine built once by [`carrier_engine`](@ref): holds
the register-resident sin/cos table and the phase-accumulator frequency word. Pair it with
one [`CarrierState`](@ref) per interleaved stream and drive it with [`carrier_lookup`](@ref)
/ [`carrier_advance`](@ref). It is isbits, so it costs no allocation.
"""
struct CarrierEngine{T,N,W,Prep}
    prepared::Prep
    freq_word::UInt32
    # Loop-invariant per-lane phase offset `freq_word·[0,1,…,W-1]`. Added back at lookup
    # time so each `CarrierState` only has to carry the scalar phase base (see below).
    lane_offset::Vec{W,UInt32}
end

"""
    CarrierState{W}

Immutable, isbits NCO phase-accumulator state: a single scalar phase base (`UInt32`). The
per-lane accumulator is `base + eng.lane_offset`, reconstructed at [`carrier_lookup`](@ref)
time — so a K-way interleaved loop keeps only `K` scalars live instead of `K` wide vectors,
relieving register pressure in the fused inner loop. Created by [`carrier_state`](@ref) and
renewed by value each iteration via [`carrier_advance`](@ref). The `{W}` parameter is a
phantom that keeps a state bound to its engine's lane count. Holds no table data, so it stays
in registers.
"""
struct CarrierState{W}
    base::UInt32
end

"""
    carrier_engine(table, freq_word::Integer;      backend=…) -> CarrierEngine
    carrier_engine(table, cycles_per_sample::Real; backend=…) -> CarrierEngine
    carrier_engine(table; frequency, sampling_frequency, backend=…) -> CarrierEngine

Build the loop-invariant carrier engine for `table`. The NCO phase advances by a `UInt32`
frequency word per sample; the low-level form takes it raw (`0 ≤ freq_word ≤ 2^32-1`),
`cycles_per_sample::Real` converts as `round(cps·2^32)`, and the keyword form derives it from
`frequency`/`sampling_frequency`. The phase is uniform, dead-zone-free and never drifts.
"""
function carrier_engine(table::SinCosTable{T,N}, freq_word::Integer;
                        backend::Backend = default_backend(T, N)) where {T,N}
    (0 ≤ freq_word ≤ typemax(UInt32)) ||
        throw(ArgumentError("need 0 ≤ freq_word ≤ typemax(UInt32) = $(typemax(UInt32))"))
    _make_engine(table, UInt32(freq_word), backend, _vwidth(backend, T))
end
carrier_engine(table::SinCosTable, cps::Real; kw...) = carrier_engine(table, _freq_word(cps); kw...)
carrier_engine(table::SinCosTable; frequency::Real, sampling_frequency::Real, kw...) =
    carrier_engine(table, cycles_per_sample(frequency, sampling_frequency); kw...)

function _make_engine(table::SinCosTable{T,N}, freq_word::UInt32, backend, ::Val{W}) where {T,N,W}
    prepared = Prepared{T,N}(_prepare(backend, table), backend, T(N - 1))
    lane_offset = Vec{W,UInt32}(freq_word) * _lanes_u32(Val(W))
    CarrierEngine{T,N,W,typeof(prepared)}(prepared, freq_word, lane_offset)
end

"""
    carrier_state(eng::CarrierEngine, start::Integer = 0; phase = 0) -> CarrierState

Initial NCO state for a stream whose first sample is absolute index `start` (use `(k-1)·W`
for the k-th lane of a W-wide, K-way interleaved loop). `phase` is the initial carrier phase
added to every lane: an `Integer` is table steps, a `Real` is cycles.
"""
@inline function carrier_state(eng::CarrierEngine{T,N,W}, start::Integer = 0;
                               phase::Real = 0) where {T,N,W}
    acc_offset = _acc_offset(_phase_steps(phase, N), Val(N))
    # Lane 0 of `_init_acc`: base = freq_word·start + acc_offset (scalar, mod 2³²). The
    # per-lane `freq_word·j` term is folded into `eng.lane_offset` at lookup time.
    CarrierState{W}(eng.freq_word * UInt32(start) + acc_offset)
end

"""
    carrier_lookup(eng::CarrierEngine, st::CarrierState) -> (sin::Vec{W,T}, cos::Vec{W,T})

The `(sin, cos)` chunk at `st`'s current phase. Pure read — does not advance the state.
"""
@inline carrier_lookup(eng::CarrierEngine{T,N,W}, st::CarrierState{W}) where {T,N,W} =
    eng.prepared(convert(Vec{W,T}, (st.base + eng.lane_offset) >> _index_shift(Val(N))))

"""
    carrier_advance(eng::CarrierEngine, st::CarrierState, nchunks::Integer) -> CarrierState

Advance the stream by `nchunks` W-wide chunks (`nchunks·W` samples), returning a new
immutable state. A K-way interleaved loop advances every state by `K` each iteration.
"""
@inline carrier_advance(eng::CarrierEngine{T,N,W}, st::CarrierState{W}, nchunks::Integer) where {T,N,W} =
    CarrierState{W}(st.base + eng.freq_word * UInt32(Int(nchunks) * W))

"""
    carrier_width(eng::CarrierEngine) -> Int

SIMD lane count `W` of the engine (samples per chunk).
"""
@inline carrier_width(::CarrierEngine{T,N,W}) where {T,N,W} = W
