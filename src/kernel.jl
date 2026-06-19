# Generic kernels. `generate_carrier!` produces a drift-free phase via an integer DDA
# (phase index = exact `div(i*step_num, step_den)`, so it never drifts) and looks up
# sin/cos. `lookup_sincos!` looks up sin/cos for a supplied array of phase indices.
#
# Type parameters used throughout: `T` = output element type, `N` = table steps per
# cycle, `U` = unsigned(T) (remainder type), `W` = SIMD lane count for the backend.

"""
    cycles_per_sample(frequency, sampling_frequency)

Normalised frequency `frequency / sampling_frequency` (cycles per sample) — pass the
result as the frequency argument of [`generate_carrier!`](@ref),
[`generate_carrier`](@ref) or [`generate_carrier4`](@ref), e.g.
`generate_carrier!(sin_out, cos_out, table, cycles_per_sample(1000, 2e6))`.
"""
cycles_per_sample(frequency, sampling_frequency) = frequency / sampling_frequency

# Convert a `phase` keyword to an integer phase offset in table steps:
#   Integer → table steps (exact);  Real → cycles (fraction of a full cycle).
@inline _phase_steps(phase::Integer, steps) = Int(phase)
@inline _phase_steps(phase::Real, steps) = round(Int, phase * steps)

# ===== generate_carrier! =====
"""
    generate_carrier!(sin_out, cos_out, table, step_numerator, step_denominator; phase=0, backend=…)
    generate_carrier!(sin_out, cos_out, table, cycles_per_sample::Real;          phase=0, backend=…)
    generate_carrier!(sin_out, cos_out, table; frequency, sampling_frequency,    phase=0, backend=…)

Fill `sin_out`/`cos_out` (element type `T`) with a carrier whose phase advances by an
exact `step_numerator / step_denominator` table-steps per sample (drift-free integer
DDA). `cycles_per_sample` is the normalised frequency `f/fs` (the carrier completes
that many cycles per sample, i.e. `out[n] ≈ cos(2π·cycles_per_sample·n)`); its step is
`cycles_per_sample * steps`, rationalised internally. The third form takes a
`frequency` and `sampling_frequency` directly (`cycles_per_sample = frequency /
sampling_frequency`). `phase` is the initial carrier phase, default 0: an `Integer` is
**table steps** (exact), a `Real` is **cycles**. Requires `0 < step_denominator ≤ typemax(T)`.
"""
function generate_carrier!(sin_out::AbstractVector{T}, cos_out::AbstractVector{T},
                           table::SinCosTable{T,N}, step_numerator::Integer, step_denominator::Integer;
                           phase::Real = 0, backend::Backend = default_backend(T, N)) where {T,N}
    length(sin_out) == length(cos_out) || throw(DimensionMismatch("sin/cos lengths differ"))
    (0 < step_denominator ≤ typemax(T)) ||
        throw(ArgumentError("need 0 < step_denominator ≤ typemax($T) = $(typemax(T))"))
    _generate!(sin_out, cos_out, table, Int(step_numerator), Int(step_denominator),
               _phase_steps(phase, N), backend)
end
function generate_carrier!(sin_out::AbstractVector{T}, cos_out::AbstractVector{T},
                           table::SinCosTable{T,N}, cycles_per_sample::Real; kw...) where {T,N}
    ratio = rationalize(Int, cycles_per_sample * N; tol = 1 / (1 << 20))
    generate_carrier!(sin_out, cos_out, table, numerator(ratio), denominator(ratio); kw...)
end

# Direct frequency / sampling-frequency form (keyword args avoid any ambiguity with the
# exact integer `step_numerator, step_denominator` method):
#   generate_carrier!(sin_out, cos_out, table; frequency = 1000, sampling_frequency = 2e6)
function generate_carrier!(sin_out::AbstractVector{T}, cos_out::AbstractVector{T},
                           table::SinCosTable{T,N};
                           frequency::Real, sampling_frequency::Real, kw...) where {T,N}
    generate_carrier!(sin_out, cos_out, table, cycles_per_sample(frequency, sampling_frequency); kw...)
end

# Initialise one DDA state of W lanes whose first lane is sample `start_sample`:
#   phase[j]     = div(step_num*(start_sample+j), step_den) + phase_offset
#   remainder[j] = mod(step_num*(start_sample+j), step_den),   j = 0..W-1
# Per-lane reductions use a multiplicative inverse (mul+shift, not idiv); they are
# independent across lanes so they pipeline (a serial Bresenham fill is slower).
@inline function _init_state(::Val{W}, ::Type{T}, ::Type{U},
                             step_num, den_inverse, step_den, start_sample, phase_offset) where {W,T,U}
    phase     = Vec{W,T}(ntuple(j -> (div(step_num * (start_sample + j - 1), den_inverse) + phase_offset) % T, Val(W)))
    remainder = Vec{W,U}(ntuple(j -> (n = step_num * (start_sample + j - 1); U(n - div(n, den_inverse) * step_den)), Val(W)))
    (phase, remainder)
end

# 4 independent DDA states run interleaved so their loop-carried carry chains
# (add→compare→blend→add) overlap instead of stalling. Hand-unrolled with plain
# locals — a tuple/closure formulation boxes the reassigned state and is ~100× slower.
function _generate!(sin_out, cos_out, table::SinCosTable{T,N},
                    step_num, step_den, phase_offset, backend::Union{AVX512,AVX2,Neon}) where {T,N}
    # pass unsigned(T) as a Type arg so U is a static parameter in the kernel
    # (binding `remainder_type = unsigned(T)` to a local makes Vec{W,…} type-unstable → boxing).
    _generate_simd!(sin_out, cos_out, table, step_num, step_den, phase_offset,
                    backend, _vwidth(backend, T), unsigned(T))
end
function _generate_simd!(sin_out, cos_out, table::SinCosTable{T,N},
                         step_num, step_den, phase_offset, backend, ::Val{W}, ::Type{U}) where {T,N,W,U}
    index_mask = T(N - 1)
    stride = 4W
    whole_step = div(stride * step_num, step_den) % T      # whole table-steps advanced per stride
    frac_step  = U(mod(stride * step_num, step_den))        # fractional remainder advanced per stride
    modulus    = U(step_den)
    den_inverse = Base.SignedMultiplicativeInverse(step_den)
    phase1, rem1 = _init_state(Val(W), T, U, step_num, den_inverse, step_den, 0,  phase_offset)
    phase2, rem2 = _init_state(Val(W), T, U, step_num, den_inverse, step_den, W,  phase_offset)
    phase3, rem3 = _init_state(Val(W), T, U, step_num, den_inverse, step_den, 2W, phase_offset)
    phase4, rem4 = _init_state(Val(W), T, U, step_num, den_inverse, step_den, 3W, phase_offset)
    prepared_table = _prepare(backend, table)              # materialise table in registers once
    num_samples = length(sin_out); chunk_start = 0
    @inbounds while chunk_start + stride <= num_samples
        # store each result immediately (keep ≤2 result vectors live → no spills);
        # the four lookups are still independent so out-of-order execution overlaps them.
        sin1, cos1 = _apply(backend, prepared_table, phase1 & index_mask)
        sin_out[VecRange{W}(chunk_start + 1)]      = sin1; cos_out[VecRange{W}(chunk_start + 1)]      = cos1
        sin2, cos2 = _apply(backend, prepared_table, phase2 & index_mask)
        sin_out[VecRange{W}(chunk_start + W + 1)]  = sin2; cos_out[VecRange{W}(chunk_start + W + 1)]  = cos2
        sin3, cos3 = _apply(backend, prepared_table, phase3 & index_mask)
        sin_out[VecRange{W}(chunk_start + 2W + 1)] = sin3; cos_out[VecRange{W}(chunk_start + 2W + 1)] = cos3
        sin4, cos4 = _apply(backend, prepared_table, phase4 & index_mask)
        sin_out[VecRange{W}(chunk_start + 3W + 1)] = sin4; cos_out[VecRange{W}(chunk_start + 3W + 1)] = cos4
        rem1 += frac_step; rem2 += frac_step; rem3 += frac_step; rem4 += frac_step
        carry1 = rem1 >= modulus; carry2 = rem2 >= modulus; carry3 = rem3 >= modulus; carry4 = rem4 >= modulus
        rem1 = vifelse(carry1, rem1 - modulus, rem1); rem2 = vifelse(carry2, rem2 - modulus, rem2)
        rem3 = vifelse(carry3, rem3 - modulus, rem3); rem4 = vifelse(carry4, rem4 - modulus, rem4)
        phase1 += whole_step; phase2 += whole_step; phase3 += whole_step; phase4 += whole_step
        phase1 = vifelse(carry1, phase1 + one(T), phase1); phase2 = vifelse(carry2, phase2 + one(T), phase2)
        phase3 = vifelse(carry3, phase3 + one(T), phase3); phase4 = vifelse(carry4, phase4 + one(T), phase4)
        chunk_start += stride
    end
    _generate_tail!(sin_out, cos_out, table, step_num, step_den, phase_offset, chunk_start + 1)
end

function _generate!(sin_out, cos_out, table::SinCosTable{T,N},
                    step_num, step_den, phase_offset, ::Portable) where {T,N}
    _generate_tail!(sin_out, cos_out, table, step_num, step_den, phase_offset, 1)
end
@inline function _generate_tail!(sin_out, cos_out, table::SinCosTable{T,N},
                                 step_num, step_den, phase_offset, sample) where {T,N}
    @inbounds while sample <= length(sin_out)
        index = mod(div(step_num * (sample - 1), step_den) + phase_offset, N)
        sin_out[sample] = table.sin[index + 1]; cos_out[sample] = table.cos[index + 1]
        sample += 1
    end
    sin_out, cos_out
end

# ===== lookup_sincos! (lookup from a phase-index array) =====
"""
    lookup_sincos!(sin_out, cos_out, phase_indices, table; backend=default_backend(table))

Look up sin/cos for each integer phase index in `phase_indices` (taken mod `steps`).
"""
function lookup_sincos!(sin_out::AbstractVector{T}, cos_out::AbstractVector{T},
                        phase_indices::AbstractVector{T}, table::SinCosTable{T,N};
                        backend::Backend = default_backend(T, N)) where {T,N}
    length(sin_out) == length(cos_out) == length(phase_indices) || throw(DimensionMismatch())
    _lookup!(sin_out, cos_out, phase_indices, table, backend)
end

function _lookup!(sin_out, cos_out, phase_indices, table::SinCosTable{T,N},
                  backend::Union{AVX512,AVX2,Neon}) where {T,N}
    _lookup_simd!(sin_out, cos_out, phase_indices, table, backend, _vwidth(backend, T))
end
function _lookup_simd!(sin_out, cos_out, phase_indices, table::SinCosTable{T,N},
                       backend, ::Val{W}) where {T,N,W}
    index_mask = T(N - 1); num_samples = length(phase_indices); sample = 1
    prepared_table = _prepare(backend, table)
    @inbounds while sample + W - 1 <= num_samples
        lane = VecRange{W}(sample)
        sin_vec, cos_vec = _apply(backend, prepared_table, phase_indices[lane] & index_mask)
        sin_out[lane] = sin_vec; cos_out[lane] = cos_vec
        sample += W
    end
    _lookup_tail!(sin_out, cos_out, phase_indices, table, sample)
end
function _lookup!(sin_out, cos_out, phase_indices, table::SinCosTable{T,N}, ::Portable) where {T,N}
    _lookup_tail!(sin_out, cos_out, phase_indices, table, 1)
end
@inline function _lookup_tail!(sin_out, cos_out, phase_indices, table::SinCosTable{T,N}, sample) where {T,N}
    @inbounds while sample <= length(phase_indices)
        index = Int(phase_indices[sample]) & (N - 1)
        sin_out[sample] = table.sin[index + 1]; cos_out[sample] = table.cos[index + 1]
        sample += 1
    end
    sin_out, cos_out
end

# ===== Portable iterator support =====
# The iterators (generate_carrier/generate_carrier4) are SIMD-first, but must still
# work on any CPU/type combo that resolves to the Portable backend (e.g. Int16/Int32
# on AVX2 or NEON). Use width 1 (Vec{1,T}) with a scalar table lookup — correct, just
# not vectorised. (The array kernels generate_carrier!/lookup_sincos! already have
# dedicated scalar Portable paths.)
_vwidth(::Portable, ::Type{T}) where {T} = Val(1)
@inline _prepare(::Portable, table::SinCosTable) = (table.sin, table.cos)
@inline function _apply(::Portable, tables, index::Vec{1,T}) where {T}
    k = Int(index[1]) + 1
    @inbounds (Vec{1,T}(tables[1][k]), Vec{1,T}(tables[2][k]))
end
