# Benchmark suite for AirspeedVelocity.jl (`benchpkg`).
# Run locally:  using AirspeedVelocity; or  benchpkg SinCosLUT --rev=dirty

using BenchmarkTools, SinCosLUT, SIMD

const SUITE = BenchmarkGroup()
# Three buffer sizes probe three regimes:
#   64k → steady-state throughput; the per-call DDA init is amortized away.
#    4k → short integration (≈ a GNSS 1 ms epoch); init/setup is a real fraction.
#    1k → very short integration; init/setup dominates, so init-path changes show
#         up most sharply here.
const SIZES = (("64k", 1 << 16), ("4k", 1 << 12), ("1k", 1 << 10))
const P, Q = 16, 125       # phase step P/Q steps per sample (≈0.002 cycles/sample at 64 steps)

# ---- array fill (drift-free DDA), per output element type and buffer size ----
SUITE["carrier!"] = BenchmarkGroup()
for (label, n) in SIZES, (T, steps) in ((Int8, 64), (Int16, 64), (Int32, 32))
    tbl = SinCosTable(T; steps = steps)
    s = zeros(T, n); c = zeros(T, n)
    SUITE["carrier!"]["$T/$label"] = @benchmarkable generate_carrier!($s, $c, $tbl, $P, $Q)
end

# ---- 4-wide interleaved iterator filling arrays (Int8) — the ~40 ps/elem path ----
# Width is the backend's SIMD width (64 on AVX-512, 32 on AVX2, …), so read it off the
# yielded Vec rather than hard-coding it — keeps the suite runnable on AVX2-only hosts.
function _fill4!(sins, coss, tbl)
    i = 1
    @inbounds for q in generate_carrier4(tbl, P, Q, length(sins))
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
    @inbounds for (sv, _) in generate_carrier(tbl, P, Q, n)
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
