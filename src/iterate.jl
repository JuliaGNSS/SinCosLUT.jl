# Array-free, allocation-free carrier generation.
#
# `prepare(table)` returns a callable mapping a phase-index `Vec` to (sin, cos) `Vec`s
# — the direct analogue of FastSinCos's `fast_sincos_*(::Vec)`, but backed by a
# register-resident table (built once).
#
# `generate_carrier(table, …, num_samples)` is an iterator yielding `(sin, cos)` `Vec`s
# with a drift-free integer DDA carried in the (isbits) iteration state. Fuse it
# straight into your own loop — the carrier is never written to memory:
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

# ---- stateful iterator: drift-free phase, yields (sin, cos) Vecs ----
struct CarrierIterator{T,N,U,W,Prep}
    prepared::Prep
    phase_init::Vec{W,T}
    remainder_init::Vec{W,U}
    whole_step::T
    frac_step::U
    modulus::U
    num_chunks::Int
end

"""
    generate_carrier(table, step_numerator, step_denominator, num_samples; phase=0, backend=…)
    generate_carrier(table, cycles_per_sample::Real,          num_samples; phase=0, backend=…)
    generate_carrier(table, num_samples; frequency, sampling_frequency,    phase=0, backend=…)

Iterator over `num_samples ÷ W` chunks (W = SIMD width for the backend/type), each
yielding `(sin, cos)::Tuple{Vec{W,T},Vec{W,T}}`. Phase advances by an exact
`step_numerator / step_denominator` (or `cycles_per_sample·steps`) table-steps per
sample via a drift-free DDA held in the iteration state — no carrier array is
allocated. `phase` is the initial carrier phase (default 0): an `Integer` is table
steps, a `Real` is cycles. Requires `0 < step_denominator ≤ typemax(T)`. Any leftover
`num_samples % W` tail is not produced (handle it yourself if needed).
"""
function generate_carrier(table::SinCosTable{T,N}, step_numerator::Integer, step_denominator::Integer,
                          num_samples::Integer; phase::Real = 0,
                          backend::Backend = default_backend(T, N)) where {T,N}
    (0 < step_denominator ≤ typemax(T)) ||
        throw(ArgumentError("need 0 < step_denominator ≤ typemax($T) = $(typemax(T))"))
    _make_carrier(table, Int(step_numerator), Int(step_denominator), Int(num_samples),
                  _phase_steps(phase, N), backend, _vwidth(backend, T), unsigned(T))
end
function generate_carrier(table::SinCosTable{T,N}, cycles_per_sample::Real, num_samples::Integer; kw...) where {T,N}
    ratio = rationalize(Int, cycles_per_sample * N; tol = 1 / (1 << 20))
    generate_carrier(table, numerator(ratio), denominator(ratio), num_samples; kw...)
end
function generate_carrier(table::SinCosTable{T,N}, num_samples::Integer;
                          frequency::Real, sampling_frequency::Real, kw...) where {T,N}
    generate_carrier(table, cycles_per_sample(frequency, sampling_frequency), num_samples; kw...)
end

function _make_carrier(table::SinCosTable{T,N}, step_num, step_den, num_samples, phase_offset,
                       backend, ::Val{W}, ::Type{U}) where {T,N,W,U}
    den_inverse = Base.SignedMultiplicativeInverse(step_den)
    phase_init, remainder_init = _init_state(Val(W), T, U, step_num, den_inverse, step_den, 0, phase_offset)
    prepared = Prepared{T,N}(_prepare(backend, table), backend, T(N - 1))
    CarrierIterator{T,N,U,W,typeof(prepared)}(prepared, phase_init, remainder_init,
        div(W * step_num, step_den) % T, U(mod(W * step_num, step_den)), U(step_den), num_samples ÷ W)
end

Base.length(it::CarrierIterator) = it.num_chunks
Base.IteratorSize(::Type{<:CarrierIterator}) = Base.HasLength()
Base.eltype(::Type{<:CarrierIterator{T,N,U,W}}) where {T,N,U,W} = Tuple{Vec{W,T},Vec{W,T}}

@inline function Base.iterate(it::CarrierIterator{T,N,U,W},
                              state = (it.phase_init, it.remainder_init, 0)) where {T,N,U,W}
    phase, remainder, chunk = state
    chunk >= it.num_chunks && return nothing
    result = it.prepared(phase)                       # (sin, cos), mask applied inside
    remainder += it.frac_step
    carry = remainder >= it.modulus
    remainder = vifelse(carry, remainder - it.modulus, remainder)
    phase = vifelse(carry, phase + it.whole_step + one(T), phase + it.whole_step)
    (result, (phase, remainder, chunk + 1))
end

# ---- 4-way interleaved iterator: yields 4 (sin,cos) pairs (4W samples) per step ----
# The four DDA carry chains are independent, so they overlap even when the consumer is
# trivial (e.g. array fill) — reaching the ~40 ps/elem loop rate at scale. Use the
# single-Vec `generate_carrier` when fusing into nontrivial work (it provides its own ILP).
struct CarrierIterator4{T,N,U,W,Prep}
    prepared::Prep
    phase1::Vec{W,T}; phase2::Vec{W,T}; phase3::Vec{W,T}; phase4::Vec{W,T}
    rem1::Vec{W,U};   rem2::Vec{W,U};   rem3::Vec{W,U};   rem4::Vec{W,U}
    whole_step::T
    frac_step::U
    modulus::U
    num_steps::Int
end

"""
    generate_carrier4(table, step_numerator, step_denominator, num_samples; phase=0, backend=…)
    generate_carrier4(table, cycles_per_sample, num_samples;                phase=0, backend=…)
    generate_carrier4(table, num_samples; frequency, sampling_frequency,    phase=0, backend=…)

Like [`generate_carrier`](@ref) but yields a 4-tuple of `(sin, cos)` `Vec` pairs per
step (`4W` samples), running four interleaved DDA states so the carry chains overlap.
Reaches the full loop throughput (~40 ps/elem) even for trivial consumers such as
array fill. **Destructure the 4-tuple in the loop header** —
`for ((s0,c0),(s1,c1),(s2,c2),(s3,c3)) in generate_carrier4(...)` — rather than
iterating it with an inner `for pair in quad` loop, which does not unroll and is much
slower. `phase` is the initial carrier phase (default 0): `Integer` = table steps,
`Real` = cycles. Produces `num_samples ÷ (4W)` steps; handle any tail yourself.
"""
function generate_carrier4(table::SinCosTable{T,N}, step_numerator::Integer, step_denominator::Integer,
                           num_samples::Integer; phase::Real = 0,
                           backend::Backend = default_backend(T, N)) where {T,N}
    (0 < step_denominator ≤ typemax(T)) ||
        throw(ArgumentError("need 0 < step_denominator ≤ typemax($T) = $(typemax(T))"))
    _make_carrier4(table, Int(step_numerator), Int(step_denominator), Int(num_samples),
                   _phase_steps(phase, N), backend, _vwidth(backend, T), unsigned(T))
end
function generate_carrier4(table::SinCosTable{T,N}, cycles_per_sample::Real, num_samples::Integer; kw...) where {T,N}
    ratio = rationalize(Int, cycles_per_sample * N; tol = 1 / (1 << 20))
    generate_carrier4(table, numerator(ratio), denominator(ratio), num_samples; kw...)
end
function generate_carrier4(table::SinCosTable{T,N}, num_samples::Integer;
                           frequency::Real, sampling_frequency::Real, kw...) where {T,N}
    generate_carrier4(table, cycles_per_sample(frequency, sampling_frequency), num_samples; kw...)
end

function _make_carrier4(table::SinCosTable{T,N}, step_num, step_den, num_samples, phase_offset,
                        backend, ::Val{W}, ::Type{U}) where {T,N,W,U}
    den_inverse = Base.SignedMultiplicativeInverse(step_den)
    phase1, rem1 = _init_state(Val(W), T, U, step_num, den_inverse, step_den, 0,  phase_offset)
    phase2, rem2 = _init_state(Val(W), T, U, step_num, den_inverse, step_den, W,  phase_offset)
    phase3, rem3 = _init_state(Val(W), T, U, step_num, den_inverse, step_den, 2W, phase_offset)
    phase4, rem4 = _init_state(Val(W), T, U, step_num, den_inverse, step_den, 3W, phase_offset)
    prepared = Prepared{T,N}(_prepare(backend, table), backend, T(N - 1))
    CarrierIterator4{T,N,U,W,typeof(prepared)}(prepared, phase1, phase2, phase3, phase4,
        rem1, rem2, rem3, rem4,
        div(4W * step_num, step_den) % T, U(mod(4W * step_num, step_den)), U(step_den), num_samples ÷ (4W))
end

Base.length(it::CarrierIterator4) = it.num_steps
Base.IteratorSize(::Type{<:CarrierIterator4}) = Base.HasLength()
Base.eltype(::Type{<:CarrierIterator4{T,N,U,W}}) where {T,N,U,W} = NTuple{4,Tuple{Vec{W,T},Vec{W,T}}}

@inline function Base.iterate(it::CarrierIterator4{T,N,U,W},
                              state = (it.phase1, it.phase2, it.phase3, it.phase4,
                                       it.rem1, it.rem2, it.rem3, it.rem4, 0)) where {T,N,U,W}
    phase1, phase2, phase3, phase4, rem1, rem2, rem3, rem4, step_count = state
    step_count >= it.num_steps && return nothing
    pair1 = it.prepared(phase1); pair2 = it.prepared(phase2)
    pair3 = it.prepared(phase3); pair4 = it.prepared(phase4)
    whole_step = it.whole_step; frac_step = it.frac_step; modulus = it.modulus
    # advance into fresh variables (reassigning the loop-carried state in place pessimises codegen)
    acc1 = rem1 + frac_step; carry1 = acc1 >= modulus
    next_rem1   = vifelse(carry1, acc1 - modulus, acc1)
    next_phase1 = vifelse(carry1, phase1 + whole_step + one(T), phase1 + whole_step)
    acc2 = rem2 + frac_step; carry2 = acc2 >= modulus
    next_rem2   = vifelse(carry2, acc2 - modulus, acc2)
    next_phase2 = vifelse(carry2, phase2 + whole_step + one(T), phase2 + whole_step)
    acc3 = rem3 + frac_step; carry3 = acc3 >= modulus
    next_rem3   = vifelse(carry3, acc3 - modulus, acc3)
    next_phase3 = vifelse(carry3, phase3 + whole_step + one(T), phase3 + whole_step)
    acc4 = rem4 + frac_step; carry4 = acc4 >= modulus
    next_rem4   = vifelse(carry4, acc4 - modulus, acc4)
    next_phase4 = vifelse(carry4, phase4 + whole_step + one(T), phase4 + whole_step)
    ((pair1, pair2, pair3, pair4),
     (next_phase1, next_phase2, next_phase3, next_phase4, next_rem1, next_rem2, next_rem3, next_rem4, step_count + 1))
end
