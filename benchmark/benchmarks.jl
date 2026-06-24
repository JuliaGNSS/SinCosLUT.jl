# Benchmark suite for AirspeedVelocity.jl (`benchpkg`).
# Run locally:  using AirspeedVelocity; or  benchpkg SinCosLUT --rev=dirty

using BenchmarkTools, SinCosLUT, SIMD

# Iterator constructors were renamed generate_carrier(4) → CarrierIterator(4) in v2. Use the
# v2 names when available, fall back to the pre-v2 names, so this one script runs against
# both the PR head and the (pre-rename) base in the AirspeedVelocity comparison.
const _CARRIER  = isdefined(SinCosLUT, :CarrierIterator)  ? SinCosLUT.CarrierIterator  : SinCosLUT.generate_carrier
const _CARRIER4 = isdefined(SinCosLUT, :CarrierIterator4) ? SinCosLUT.CarrierIterator4 : SinCosLUT.generate_carrier4

const SUITE = BenchmarkGroup()
# Three buffer sizes probe three regimes:
#   64k → steady-state throughput; the per-call DDA init is amortized away.
#    4k → short integration (≈ a GNSS 1 ms epoch); init/setup is a real fraction.
#    1k → very short integration; init/setup dominates, so init-path changes show
#         up most sharply here.
const SIZES = (("64k", 1 << 16), ("4k", 1 << 12), ("1k", 1 << 10))
const FREQ_WORD = 0x0a3d70a3   # NCO frequency word (≈0.005 cycles/sample)

# ---- array fill (NCO phase accumulator), per output element type and buffer size ----
SUITE["carrier!"] = BenchmarkGroup()
for (label, n) in SIZES, (T, steps) in ((Int8, 64), (Int16, 64), (Int32, 32))
    tbl = SinCosTable(T; steps = steps)
    s = zeros(T, n); c = zeros(T, n)
    SUITE["carrier!"]["$T/$label"] = @benchmarkable generate_carrier!($s, $c, $tbl, $FREQ_WORD)
end

# ---- 4-wide interleaved iterator filling arrays (Int8) — the ~40 ps/elem path ----
# Width is the backend's SIMD width (64 on AVX-512, 32 on AVX2, …), so read it off the
# yielded Vec rather than hard-coding it — keeps the suite runnable on AVX2-only hosts.
function _fill4!(sins, coss, tbl)
    i = 1
    @inbounds for q in _CARRIER4(tbl, FREQ_WORD, length(sins))
        for (sv, cv) in q
            W = length(sv)
            sins[VecRange{W}(i)] = sv; coss[VecRange{W}(i)] = cv; i += W
        end
    end
end
SUITE["carrier4_fill_Int8"] = BenchmarkGroup()
for (label, n) in SIZES
    let tbl = SinCosTable(Int8; steps = 64), s = zeros(Int8, n), c = zeros(Int8, n)
        SUITE["carrier4_fill_Int8"][label] = @benchmarkable _fill4!($s, $c, $tbl)
    end
end

# ---- fused, array-free reduction over the single-Vec iterator (Int8) ----
function _reduce(tbl, n)
    acc = 0
    @inbounds for (sv, _) in _CARRIER(tbl, FREQ_WORD, n)
        acc += sum(Vec{length(sv),Int32}(sv))
    end
    acc
end
SUITE["fused_reduce_Int8"] = BenchmarkGroup()
for (label, n) in SIZES
    let tbl = SinCosTable(Int8; steps = 64)
        SUITE["fused_reduce_Int8"][label] = @benchmarkable _reduce($tbl, $n)
    end
end

# ---- lookup from a supplied phase array (no DDA init) ----
SUITE["sincos_lut!_Int8"] = BenchmarkGroup()
for (label, n) in SIZES
    let tbl = SinCosTable(Int8; steps = 64), ph = rand(Int8, n), s = zeros(Int8, n), c = zeros(Int8, n)
        SUITE["sincos_lut!_Int8"][label] = @benchmarkable lookup_sincos!($s, $c, $ph, $tbl)
    end
end
