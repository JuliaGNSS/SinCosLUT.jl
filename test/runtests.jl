using SinCosLUT, Test, SIMD
using SinCosLUT: Portable, default_backend, backend_name, _phase_steps

# Ground-truth NCO index for sample n (0-based): top log2(N) bits of the UInt32 phase
# accumulator. acc[n] = freq_word*n + acc_offset (mod 2^32).
function ref_idx(freq_word::UInt32, N, n; phase = 0)
    shift = 32 - trailing_zeros(N)
    acc_offset = UInt32(mod(_phase_steps(phase, N), N) << shift)
    Int((freq_word * UInt32(n) + acc_offset) >> shift)
end

# Ground-truth carrier (independent of any SIMD path)
function ref_carrier(::Type{T}, N, freq_word::UInt32, L; phase = 0) where T
    tbl = SinCosTable(T; steps = N)
    s = zeros(T, L); c = zeros(T, L)
    for i in 1:L
        k = ref_idx(freq_word, N, i - 1; phase = phase)
        s[i] = tbl.sin[k + 1]; c[i] = tbl.cos[k + 1]
    end
    s, c
end

# NCO frequency word used across the byte-exact tests (≈0.005 cycles/sample).
const FW = 0x0a3d70a3

# Fused reduction over the value-based engine (width/type-agnostic via sum(s)).
function reduce_carrier(t, b, n)
    eng = carrier_engine(t, FW; backend = b); st = carrier_state(eng); a = 0
    for _ in 1:(n ÷ carrier_width(eng))
        s, _ = carrier_lookup(eng, st); a += sum(s); st = carrier_advance(eng, st, 1)
    end
    a
end
# Measure allocations INSIDE a barrier so `b` is concrete at the @allocated site.
# (A default-backend value is abstractly typed; on some Julia versions that boxes per
# element when measured directly. Specialising here gives a true 0.)
function alloc_of(t, b)
    reduce_carrier(t, b, 4096)            # compile
    @allocated reduce_carrier(t, b, 4096)
end

@testset "SinCosLUT" begin
    @testset "table construction" begin
        t = SinCosTable(Int8; steps = 64)
        @test length(t.sin) == 64 && length(t.cos) == 64
        @test t.cos[1] == round(Int8, 127)          # cos(0)
        @test abs(Int(t.sin[1])) <= 1               # sin(0)
        @test eltype(t.sin) == Int8
        @test eltype(SinCosTable(Int16).sin) == Int16
    end

    cases = ((Int8, (64, 128)), (Int16, (32, 64)), (Int32, (16, 32)))
    L = 1000
    # NCO frequency words to exercise: ≈100 Hz @ 5 MHz, a couple of fine Dopplers, and a
    # negative (receding) Doppler that wraps through two's complement.
    fws = (FW,
           UInt32(round(Int64, 1234.5 / 5e6 * 2.0^32)),
           UInt32(round(Int64, 12345.678 / 5e6 * 2.0^32)),
           unsafe_trunc(UInt32, round(Int64, -777.3 / 5e6 * 2.0^32)))  # negative Doppler

    @testset "$T steps=$N fw=$(repr(fw)) — active backend matches reference" for (T, Ns) in cases, N in Ns, fw in fws
        b = default_backend(T, N)
        tbl = SinCosTable(T; steps = N)
        s = zeros(T, L); c = zeros(T, L)
        generate_carrier!(s, c, tbl, fw; backend = b)
        sref, cref = ref_carrier(T, N, fw, L)
        @test s == sref
        @test c == cref
    end

    @testset "$T steps=$N fw=$(repr(fw)) — portable matches reference" for (T, Ns) in cases, N in Ns, fw in fws
        tbl = SinCosTable(T; steps = N)
        s = zeros(T, L); c = zeros(T, L)
        generate_carrier!(s, c, tbl, fw; backend = Portable())
        sref, cref = ref_carrier(T, N, fw, L)
        @test s == sref && c == cref
    end

    @testset "sincos_lut! matches carrier path" for (T, Ns) in cases, N in Ns
        tbl = SinCosTable(T; steps = N)
        phases = T[ (ref_idx(FW, N, i - 1) % T) for i in 1:L ]
        s = zeros(T, L); c = zeros(T, L)
        lookup_sincos!(s, c, phases, tbl)
        sref, cref = ref_carrier(T, N, FW, L)
        @test s == sref && c == cref
    end

    @testset "no drift over long run ($T, N=$N)" for (T, Ns) in cases, N in Ns
        tbl = SinCosTable(T; steps = N)
        L2 = 200_000
        s = zeros(T, L2); c = zeros(T, L2)
        generate_carrier!(s, c, tbl, FW)                 # default backend
        # phase index must stay exactly the NCO closed form for ALL i (no drift)
        ok = true
        for i in 1:L2
            k = ref_idx(FW, N, i - 1)
            (s[i] == tbl.sin[k + 1] && c[i] == tbl.cos[k + 1]) || (ok = false; break)
        end
        @test ok
    end

    @testset "fine Doppler is dead-zone-free" begin
        T = Int8; N = 64; fs = 5e6
        tbl = SinCosTable(T; steps = N)
        b = default_backend(T, N)
        out(f, len) = (s = zeros(T, len); c = zeros(T, len);
                       generate_carrier!(s, c, tbl; frequency = f, sampling_frequency = fs, backend = b); (s, c))
        s100,  c100  = out(100.0, 4096)
        s100b, c100b = out(100.001, 4096)   # a 1 mHz Doppler shift must change the output
        @test (s100, c100) != (s100b, c100b)
        @test !all(==(s100[1]), s100)       # not stuck at DC
        # a tiny 0.5 Hz Doppler still advances the NCO phase — given enough samples it must
        # leave DC (0.5 Hz crosses a 64-step index after ~5e6/(64·0.5) ≈ 156k samples).
        s05, c05 = out(0.5, 200_000)
        @test !(all(==(s05[1]), s05) && all(==(c05[1]), c05))
        # and the frequency word for 0.5 Hz is genuinely nonzero (no dead zone at the low end)
        @test SinCosLUT._freq_word(0.5 / fs) != 0
    end

    @testset "accuracy vs true sincos (Int16, N=64)" begin
        N = 64; T = Int16; A = typemax(Int16)
        tbl = SinCosTable(T; steps = N)
        L2 = 4096
        fw = UInt32(round(Int64, (1 / 13 / N) * 2.0^32))   # ≈ 1/13 table-steps per sample
        cps = fw / 2.0^32                                   # actual cycles/sample realised
        s = zeros(T, L2); c = zeros(T, L2)
        generate_carrier!(s, c, tbl, fw)
        maxerr = 0.0
        for i in 1:L2
            θ = 2pi * cps * (i - 1)
            maxerr = max(maxerr, abs(c[i] / A - cos(θ)))
        end
        @test maxerr < 0.11          # 64 phase steps, floored: error ≲ one step ≈ π/64 ≈ 0.098
    end

    @testset "engine (1-wide) == carrier! ($T, N=$N)" for (T, Ns) in cases, N in Ns
        tbl = SinCosTable(T; steps = N)
        sref, cref = ref_carrier(T, N, FW, L)
        eng = carrier_engine(tbl, FW); W = carrier_width(eng)
        nfull = (L ÷ W) * W
        s = zeros(T, L); c = zeros(T, L); st = carrier_state(eng); i = 1
        for _ in 1:(L ÷ W)
            sv, cv = carrier_lookup(eng, st)
            s[SIMD.VecRange{W}(i)] = sv; c[SIMD.VecRange{W}(i)] = cv; i += W
            st = carrier_advance(eng, st, 1)
        end
        @test s[1:nfull] == sref[1:nfull]
        @test c[1:nfull] == cref[1:nfull]
    end

    @testset "engine (4-way interleaved) == carrier! ($T, N=$N)" for (T, Ns) in cases, N in Ns
        tbl = SinCosTable(T; steps = N)
        sref, cref = ref_carrier(T, N, FW, L)
        eng = carrier_engine(tbl, FW); W = carrier_width(eng)
        nfull = (L ÷ (4W)) * (4W)
        s = zeros(T, L); c = zeros(T, L)
        st1 = carrier_state(eng, 0); st2 = carrier_state(eng, W)
        st3 = carrier_state(eng, 2W); st4 = carrier_state(eng, 3W)
        for step in 0:(L ÷ (4W) - 1)
            base = step * 4W + 1
            v1 = carrier_lookup(eng, st1); s[SIMD.VecRange{W}(base)]      = v1[1]; c[SIMD.VecRange{W}(base)]      = v1[2]
            v2 = carrier_lookup(eng, st2); s[SIMD.VecRange{W}(base + W)]  = v2[1]; c[SIMD.VecRange{W}(base + W)]  = v2[2]
            v3 = carrier_lookup(eng, st3); s[SIMD.VecRange{W}(base + 2W)] = v3[1]; c[SIMD.VecRange{W}(base + 2W)] = v3[2]
            v4 = carrier_lookup(eng, st4); s[SIMD.VecRange{W}(base + 3W)] = v4[1]; c[SIMD.VecRange{W}(base + 3W)] = v4[2]
            st1 = carrier_advance(eng, st1, 4); st2 = carrier_advance(eng, st2, 4)
            st3 = carrier_advance(eng, st3, 4); st4 = carrier_advance(eng, st4, 4)
        end
        @test s[1:nfull] == sref[1:nfull]
        @test c[1:nfull] == cref[1:nfull]
    end

    @testset "scalar base state ($T, N=$N)" for (T, Ns) in cases, N in Ns
        tbl = SinCosTable(T; steps = N)
        eng = carrier_engine(tbl, FW); W = carrier_width(eng)
        # State carries a single scalar phase base, not a W-wide vector.
        st = carrier_state(eng, 2W; phase = 5)
        @test st isa SinCosLUT.CarrierState{W}
        @test st.base isa UInt32
        @test sizeof(st) == sizeof(UInt32)
        # base + lane_offset reconstructs the old per-lane accumulator bit-for-bit.
        acc_offset = SinCosLUT._acc_offset(_phase_steps(5, N), Val(N))
        ref_acc = SinCosLUT._init_acc(Val(W), eng.freq_word, acc_offset, 2W)
        @test st.base + eng.lane_offset === ref_acc
        # advance stays bit-identical to advancing the full accumulator.
        st2 = carrier_advance(eng, st, 3)
        ref_acc2 = ref_acc + eng.freq_word * UInt32(3 * W)
        @test st2.base + eng.lane_offset === ref_acc2
    end

    @testset "starting phase offset ($T, N=$N)" for (T, Ns) in cases, N in Ns
        tbl = SinCosTable(T; steps = N)
        φ = 7
        s = zeros(T, L); c = zeros(T, L)
        generate_carrier!(s, c, tbl, FW; phase = φ)
        # output[i] must be the table at the NCO index with phase offset φ
        ok = true
        for i in 1:L
            k = ref_idx(FW, N, i - 1; phase = φ)
            (s[i] == tbl.sin[k + 1] && c[i] == tbl.cos[k + 1]) || (ok = false; break)
        end
        @test ok
        # phase = N wraps to phase = 0
        s0 = zeros(T, L); c0 = zeros(T, L); generate_carrier!(s0, c0, tbl, FW)
        sN = zeros(T, L); cN = zeros(T, L); generate_carrier!(sN, cN, tbl, FW; phase = N)
        @test s0 == sN && c0 == cN
        # Real phase in cycles: 0.25 cycles == N÷4 table steps
        sr = zeros(T, L); cr = zeros(T, L); generate_carrier!(sr, cr, tbl, FW; phase = 0.25)
        si = zeros(T, L); ci = zeros(T, L); generate_carrier!(si, ci, tbl, FW; phase = N ÷ 4)
        @test sr == si && cr == ci
    end

    @testset "cycles_per_sample helper" begin
        @test cycles_per_sample(1000, 2_000_000) == 0.0005
        tbl = SinCosTable(Int8; steps = 64)
        s1 = zeros(Int8, 512); c1 = zeros(Int8, 512)
        s2 = zeros(Int8, 512); c2 = zeros(Int8, 512)
        generate_carrier!(s1, c1, tbl, cycles_per_sample(1000, 2_000_000))
        generate_carrier!(s2, c2, tbl, 1000 / 2_000_000)
        @test s1 == s2 && c1 == c2
        # frequency / sampling_frequency keyword form
        s3 = zeros(Int8, 512); c3 = zeros(Int8, 512)
        generate_carrier!(s3, c3, tbl; frequency = 1000, sampling_frequency = 2_000_000)
        @test s1 == s3 && c1 == c3
    end

    @testset "iterator is allocation-free & prepare callable" begin
        tbl = SinCosTable(Int8; steps = 64)
        @testset "$(backend_name(b))" for b in (default_backend(Int8, 64), Portable())
            @test alloc_of(tbl, b) == 0                       # 0 allocations (measured in a barrier)
            # prepare callable: every lane at index 2 -> table[3], for the backend's width
            p = prepare(tbl; backend = b)
            s, c = p(SIMD.Vec(ntuple(_ -> Int8(2), SinCosLUT._vwidth(b, Int8))))
            @test all(==(tbl.sin[3]), Tuple(s)) && all(==(tbl.cos[3]), Tuple(c))
        end
    end

    println("default backends: ",
        join(["$T/$N→$(backend_name(default_backend(T,N)))" for (T,Ns) in cases for N in Ns], "  "))
end
