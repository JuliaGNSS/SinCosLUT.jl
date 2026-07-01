# One-bit (hard-limited) carrier generation.
#
# A 1-bit carrier keeps only the SIGN of sin/cos — the natural input to a bit-wise
# ("bit-sliced") software correlator, where the carrier wipe-off collapses to XOR and the
# accumulate to popcount. There is no table lookup here: the sign of the NCO output is an
# exact function of the phase accumulator's top bit, so we read it straight off the same
# UInt32 NCO that `generate_carrier!` uses (`acc[n] = freq_word·n + offset` mod 2³²):
#
#   sin(2π·φ) < 0  ⇔  φ mod 1 ∈ [½, 1)      ⇔  MSB(acc)              is set
#   cos(2π·φ) < 0  ⇔  φ mod 1 ∈ (¼, ¾)      ⇔  MSB(acc + ¼ cycle)    is set
#
# The signs are packed into `UInt64` words (bit `j` of word `w` ↔ sample `64w+j`), a set
# bit meaning that component is NEGATIVE (the ±1 hard-limited value is −1). Because a 1-bit
# carrier is a square wave, the sign is constant for `sampling_frequency/(2·frequency)`
# samples at a time: for a typical low residual carrier that is thousands of samples, so
# almost every 64-sample word is a single constant run written with ONE store. Only a word
# that straddles a sign flip is filled bit-by-bit. Work is therefore O(#flips)+O(#words)
# rather than O(#samples) with a per-sample table lookup and a sign extraction.

@inline _msb(x::UInt32) = (x & 0x80000000) != zero(UInt32)

# Sign bits of one 64-sample word. `b` is the accumulator at the word's first sample; the
# lanes are `b, b+fw, …, b+63·fw`. `rem` (≤64) is how many lanes are valid (the last word of
# a run may be short — the high bits stay 0). `single_flip` asserts the word spans < ½ cycle,
# so at most one sign flip occurs and equal endpoints ⇒ a constant run.
@inline function _sign_word(b::UInt32, fw::UInt32, span::UInt32, rem::Int, single_flip::Bool)
    if single_flip && rem == 64 && _msb(b) == _msb(b + span)
        return _msb(b) ? typemax(UInt64) : zero(UInt64)          # constant run — one store
    end
    word = zero(UInt64); a = b                                    # straddles a flip / short tail
    @inbounds for j in 0:(rem - 1)
        _msb(a) && (word |= UInt64(1) << j)
        a += fw
    end
    word
end

function _carrier_signs!(sin_signs::AbstractVector{UInt64}, cos_signs::AbstractVector{UInt64},
                         n::Int, fw::UInt32, off::UInt32)
    nwords = cld(n, 64)
    step = fw * UInt32(64)                 # accumulator advance per 64-sample word
    span = fw * UInt32(63)                 # advance across the 63 lanes within a word
    single_flip = step < 0x80000000        # < ½ cycle per word ⇒ ≤ 1 flip ⇒ run-fill is valid
    base = off                             # accumulator at sample 0 (lane 0) = off
    @inbounds for w in 1:nwords
        rem = w == nwords ? n - (w - 1) * 64 : 64                 # valid lanes (tail may be < 64)
        sin_signs[w] = _sign_word(base, fw, span, rem, single_flip)
        cos_signs[w] = _sign_word(base + 0x40000000, fw, span, rem, single_flip)  # +¼ cycle
        base += step
    end
    sin_signs, cos_signs
end

"""
    generate_carrier_signs!(sin_signs, cos_signs, n, freq_word::Integer;      phase=0)
    generate_carrier_signs!(sin_signs, cos_signs, n, cycles_per_sample::Real; phase=0)
    generate_carrier_signs!(sin_signs, cos_signs, n; frequency, sampling_frequency, phase=0)

Generate a **1-bit (hard-limited) carrier**: pack the sign bits of the NCO sin/cos for `n`
samples into `sin_signs`/`cos_signs` (`UInt64`, each `≥ cld(n, 64)` words). Bit `j` of word
`w` corresponds to sample `64w + j` (0-based); a **set bit means that component is negative**
(i.e. its ±1 hard-limited value is `−1`). Bits past sample `n` in the last word are cleared.

This is the same NCO as [`generate_carrier!`](@ref) — a `UInt32` phase accumulator advanced by
a fixed frequency word each sample (`acc[n] = freq_word·n + offset` mod `2³²`) — but only the
sign is kept, read directly off the accumulator's top bit (`sign(sin) = MSB(acc)`,
`sign(cos) = MSB(acc + ¼ cycle)`), so there is no table and no output-type quantisation. The
frequency argument matches `generate_carrier!`: a raw `UInt32` `freq_word`, a
`cycles_per_sample::Real` (`freq_word = round(cps·2³²)`), or the `frequency`/`sampling_frequency`
keyword form. `phase` is the initial carrier phase in **cycles** (a `Real`; default 0).

A 1-bit carrier is a square wave, so the sign is constant for `≈ sampling_frequency /
(2·frequency)` samples; almost every 64-sample word is a single constant run filled with one
store, and only words straddling a sign flip are filled bit-by-bit. This is the carrier form a
bit-wise correlator consumes, where wipe-off becomes XOR and accumulation becomes popcount.

```julia
n = 5000
sin_signs = Vector{UInt64}(undef, cld(n, 64))
cos_signs = Vector{UInt64}(undef, cld(n, 64))
generate_carrier_signs!(sin_signs, cos_signs, n; frequency = 1234, sampling_frequency = 5e6)
```
"""
function generate_carrier_signs!(sin_signs::AbstractVector{UInt64}, cos_signs::AbstractVector{UInt64},
                                 n::Integer, freq_word::Integer; phase::Real = 0)
    nwords = cld(Int(n), 64)
    (length(sin_signs) ≥ nwords && length(cos_signs) ≥ nwords) ||
        throw(DimensionMismatch("sign buffers need ≥ cld(n, 64) = $nwords words"))
    (0 ≤ freq_word ≤ typemax(UInt32)) ||
        throw(ArgumentError("need 0 ≤ freq_word ≤ typemax(UInt32) = $(typemax(UInt32))"))
    _carrier_signs!(sin_signs, cos_signs, Int(n), UInt32(freq_word), _freq_word(phase))
end
function generate_carrier_signs!(sin_signs::AbstractVector{UInt64}, cos_signs::AbstractVector{UInt64},
                                 n::Integer, cycles_per_sample::Real; kw...)
    generate_carrier_signs!(sin_signs, cos_signs, n, _freq_word(cycles_per_sample); kw...)
end
function generate_carrier_signs!(sin_signs::AbstractVector{UInt64}, cos_signs::AbstractVector{UInt64},
                                 n::Integer; frequency::Real, sampling_frequency::Real, kw...)
    generate_carrier_signs!(sin_signs, cos_signs, n,
                            cycles_per_sample(frequency, sampling_frequency); kw...)
end
