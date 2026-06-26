# Benchmark suite for AirspeedVelocity.jl (`benchpkg`).
# Run locally:  using AirspeedVelocity; or  benchpkg SinCosLUT --rev=dirty

using BenchmarkTools, SinCosLUT, SIMD

# The fused carrier path was rewritten from the `CarrierIterator`/`CarrierIterator4` iterators
# (≤ v2) to the value-based `carrier_engine`/`carrier_state`/… API (v3, breaking). benchpkg runs
# THIS (head) script against both the PR head and its base build, so the fused benchmarks below
# pick the API available on each rev — keyed identically so the comparison measures the rewrite's
# effect directly. Discriminate on the new `carrier_engine` function (absent on every pre-v3 rev).
const _HAS_ENGINE = isdefined(SinCosLUT, :carrier_engine)

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

# ---- 4-way interleaved fill (Int8) — the ~40 ps/elem path ----
# Width W is the backend's SIMD width (64 on AVX-512, 32 on AVX2, …). v3 reads it off the engine;
# the pre-v3 fallback reads it off the yielded Vec — keeps the suite runnable on AVX2-only hosts.
@static if _HAS_ENGINE
    function _fill4!(sins, coss, tbl)
        eng = carrier_engine(tbl, FREQ_WORD); W = carrier_width(eng)
        st1 = carrier_state(eng, 0); st2 = carrier_state(eng, W)
        st3 = carrier_state(eng, 2W); st4 = carrier_state(eng, 3W)
        @inbounds for step in 0:(length(sins) ÷ (4W) - 1)
            base = step * 4W + 1
            v1 = carrier_lookup(eng, st1); sins[VecRange{W}(base)]      = v1[1]; coss[VecRange{W}(base)]      = v1[2]
            v2 = carrier_lookup(eng, st2); sins[VecRange{W}(base + W)]  = v2[1]; coss[VecRange{W}(base + W)]  = v2[2]
            v3 = carrier_lookup(eng, st3); sins[VecRange{W}(base + 2W)] = v3[1]; coss[VecRange{W}(base + 2W)] = v3[2]
            v4 = carrier_lookup(eng, st4); sins[VecRange{W}(base + 3W)] = v4[1]; coss[VecRange{W}(base + 3W)] = v4[2]
            st1 = carrier_advance(eng, st1, 4); st2 = carrier_advance(eng, st2, 4)
            st3 = carrier_advance(eng, st3, 4); st4 = carrier_advance(eng, st4, 4)
        end
    end
else
    function _fill4!(sins, coss, tbl)   # pre-v3: the CarrierIterator4 iterator
        i = 1
        @inbounds for q in SinCosLUT.CarrierIterator4(tbl, FREQ_WORD, length(sins))
            for (sv, cv) in q
                W = length(sv)
                sins[VecRange{W}(i)] = sv; coss[VecRange{W}(i)] = cv; i += W
            end
        end
    end
end
SUITE["carrier4_fill_Int8"] = BenchmarkGroup()
for (label, n) in SIZES
    let tbl = SinCosTable(Int8; steps = 64), s = zeros(Int8, n), c = zeros(Int8, n)
        SUITE["carrier4_fill_Int8"][label] = @benchmarkable _fill4!($s, $c, $tbl)
    end
end

# ---- fused, array-free reduction over a single stream (Int8) ----
@static if _HAS_ENGINE
    function _reduce(tbl, n)
        eng = carrier_engine(tbl, FREQ_WORD); W = carrier_width(eng); st = carrier_state(eng)
        acc = 0
        @inbounds for _ in 1:(n ÷ W)
            sv, _ = carrier_lookup(eng, st)
            acc += sum(convert(Vec{W,Int32}, sv))
            st = carrier_advance(eng, st, 1)
        end
        acc
    end
else
    function _reduce(tbl, n)            # pre-v3: the single-Vec CarrierIterator iterator
        acc = 0
        @inbounds for (sv, _) in SinCosLUT.CarrierIterator(tbl, FREQ_WORD, n)
            acc += sum(Vec{length(sv),Int32}(sv))
        end
        acc
    end
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
