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

    # _phase_index extracts the table index (top log2(N) bits of each UInt32 NCO lane). The
    # AVX-512 Int8 path takes a byte-gather shortcut instead of a narrowing convert; check it
    # (and every other backend's generic form) against the shift+mask reference over the full
    # UInt32 range — including the high-bit patterns a real phase accumulator hits.
    @testset "_phase_index == shift+mask ($T, N=$N)" for (T, Ns) in cases, N in Ns
        b = default_backend(T, N)
        W = carrier_width(carrier_engine(SinCosTable(T; steps = N), FW; backend = b))
        shift = SinCosLUT._index_shift(Val(N))
        edge = (typemin(UInt32), typemax(UInt32), UInt32(0), UInt32(N - 1) << shift)
        accs = Iterators.flatten((
            (Vec{W,UInt32}(ntuple(_ -> rand(UInt32), W)) for _ in 1:200),
            (Vec{W,UInt32}(ntuple(_ -> rand(edge), W)) for _ in 1:50),
        ))
        ok = true
        for acc in accs
            idx = SinCosLUT._phase_index(b, acc, Val(N), T)
            for i in 1:W
                (Int(idx[i]) & (N - 1)) == Int((acc[i] >> UInt32(shift)) & UInt32(N - 1)) || (ok = false)
            end
        end
        @test ok
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

    @testset "one-bit carrier sign bits" begin
        using SinCosLUT: _freq_word
        # Ground-truth: bit = MSB of the same UInt32 NCO. acc[n] = fw*n + off (mod 2^32);
        # sin<0 ⇔ MSB(acc); cos<0 ⇔ MSB(acc + ¼ cycle).
        function ref_signs(n, fw::UInt32; phase = 0)
            off = _freq_word(phase)
            s = zeros(UInt64, cld(n, 64)); c = zeros(UInt64, cld(n, 64))
            for k in 1:n
                acc = fw * UInt32(k - 1) + off
                (acc & 0x80000000 != 0)               && (s[((k-1)>>6)+1] |= UInt64(1) << ((k-1)&63))
                ((acc+0x40000000) & 0x80000000 != 0)  && (c[((k-1)>>6)+1] |= UInt64(1) << ((k-1)&63))
            end
            s, c
        end
        # low freq (long runs → run-fill fast path) and high freq (a flip inside most words).
        # 0x4ccccccd (≈0.3 cyc/sample) is the trap case: 64·fw wraps mod 2³² to < 2³¹, so a guard
        # on the wrapped step would wrongly take the constant-run path — the fill path is only
        # valid when fw < 2²⁵. 0x02000001 is just past that boundary.
        @testset "n=$n fw=$(repr(fw))" for n in (5000, 20000, 64, 63, 130),
                                            fw in (0x0010c6f8, 0x01ffffff, 0x02000001,
                                                   0x0a3d70a3, 0x2aaaaaab, 0x4ccccccd)
            ns = cld(n, 64)
            s = Vector{UInt64}(undef, ns); c = Vector{UInt64}(undef, ns)
            generate_carrier_signs!(s, c, n, fw)
            rs, rc = ref_signs(n, fw)
            @test s == rs && c == rc
        end
        # bits past sample n in the final word are cleared (tail masking)
        let n = 130                       # 3 words, last word holds 2 valid bits
            s = fill(typemax(UInt64), 3); c = fill(typemax(UInt64), 3)
            generate_carrier_signs!(s, c, n, 0x0a3d70a3)
            @test (s[3] >> (n & 63)) == 0 && (c[3] >> (n & 63)) == 0
        end
        # the three frequency-argument forms agree, and `phase` shifts as expected
        let n = 5000, fs = 5_000_000, f = 1234
            s1 = Vector{UInt64}(undef, cld(n,64)); c1 = similar(s1)
            s2 = similar(s1); c2 = similar(s1); s3 = similar(s1); c3 = similar(s1)
            generate_carrier_signs!(s1, c1, n, _freq_word(cycles_per_sample(f, fs)))
            generate_carrier_signs!(s2, c2, n, cycles_per_sample(f, fs))
            generate_carrier_signs!(s3, c3, n; frequency = f, sampling_frequency = fs)
            @test s1 == s2 == s3 && c1 == c2 == c3
            # a ¼-cycle phase shift turns sin signs into cos signs
            sp = similar(s1); cp = similar(s1)
            generate_carrier_signs!(sp, cp, n, cycles_per_sample(f, fs); phase = 0.25)
            @test sp == c1
        end
        # allocation-free fill and argument validation. Measure through a barrier that discards the
        # return: the fill is 0-alloc, but the returned `(sin, cos)` tuple is only elided by the
        # optimizer on Julia ≥ 1.11, so measuring the call directly counts it on 1.10.
        _alloc_probe(s, c, n, fw) = (generate_carrier_signs!(s, c, n, fw); nothing)
        let n = 4096, s = Vector{UInt64}(undef, 64), c = Vector{UInt64}(undef, 64)
            _alloc_probe(s, c, n, 0x0a3d70a3)                     # compile
            @test (@allocated _alloc_probe(s, c, n, 0x0a3d70a3)) == 0
            @test_throws DimensionMismatch generate_carrier_signs!(zeros(UInt64,1), c, n, 0x1)
            @test_throws ArgumentError generate_carrier_signs!(s, c, n, typemax(UInt32) + 1)
        end
        # Cross-backend: the SIMD-chunked flip path (AVX2 2×32; the AVX-512 64-lane and NEON 4×16
        # paths are the default backend, covered by the ref_signs tests above on their CI) must match
        # the generic UInt32 sign-mask (Portable fallback) bit-for-bit. Forced backends, so this runs
        # wherever the ISA is present regardless of the default choice.
        @static if Sys.ARCH in (:x86_64, :i686)
            if SinCosLUT.HOST_FEATURES.avx2
                using SinCosLUT: _SIGN_TABLE, _flip_words, _iota_u32, AVX2, Portable, prepare
                pp = prepare(_SIGN_TABLE; backend = Portable())
                pa = prepare(_SIGN_TABLE; backend = AVX2())
                @testset "AVX2 flip path fw=$(repr(fw))" for fw in
                        (0x0a3d70a3, 0x2aaaaaab, 0x4ccccccd, 0x00012345, 0x7ffffff0)
                    ramp = _iota_u32() * SIMD.Vec{64,UInt32}(fw)
                    @test all(base -> _flip_words(pa, base, ramp) == _flip_words(pp, base, ramp),
                              UInt32.((0x0, 0x12345678, 0x80000000, 0xabcdef01)))
                end
            end
        end
    end

    println("default backends: ",
        join(["$T/$N→$(backend_name(default_backend(T,N)))" for (T,Ns) in cases for N in Ns], "  "))
end
