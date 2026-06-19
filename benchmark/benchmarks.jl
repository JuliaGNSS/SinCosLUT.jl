# Benchmark suite for AirspeedVelocity.jl (`benchpkg`).
# Run locally:  using AirspeedVelocity; or  benchpkg SinCosLUT --rev=dirty

using BenchmarkTools, SinCosLUT, SIMD

const SUITE = BenchmarkGroup()
const L = 1 << 16          # samples per benchmark
const P, Q = 16, 125       # phase step P/Q steps per sample (≈0.002 cycles/sample at 64 steps)

# ---- array fill (drift-free DDA), per output element type ----
SUITE["carrier!"] = BenchmarkGroup()
for (T, steps) in ((Int8, 64), (Int16, 64), (Int32, 32))
    tbl = SinCosTable(T; steps = steps)
    s = zeros(T, L); c = zeros(T, L)
    SUITE["carrier!"][string(T)] = @benchmarkable generate_carrier!($s, $c, $tbl, $P, $Q)
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
let tbl = SinCosTable(Int8; steps = 64), s = zeros(Int8, L), c = zeros(Int8, L)
    SUITE["carrier4_fill_Int8"] = @benchmarkable _fill4!($s, $c, $tbl)
end

# ---- fused, array-free reduction over the single-Vec iterator (Int8) ----
function _reduce(tbl)
    acc = 0
    @inbounds for (sv, _) in generate_carrier(tbl, P, Q, L)
        acc += sum(Vec{length(sv),Int32}(sv))
    end
    acc
end
let tbl = SinCosTable(Int8; steps = 64)
    SUITE["fused_reduce_Int8"] = @benchmarkable _reduce($tbl)
end

# ---- lookup from a supplied phase array ----
let tbl = SinCosTable(Int8; steps = 64), ph = rand(Int8, L), s = zeros(Int8, L), c = zeros(Int8, L)
    SUITE["sincos_lut!_Int8"] = @benchmarkable lookup_sincos!($s, $c, $ph, $tbl)
end
