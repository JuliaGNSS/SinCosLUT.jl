using SinCosLUT, Test, SIMD
using SinCosLUT: Portable, default_backend, backend_name

# Ground-truth lookup (independent of any SIMD path)
function ref_carrier(::Type{T}, N, P, Q, L) where T
    tbl = SinCosTable(T; steps = N)
    s = zeros(T, L); c = zeros(T, L)
    for i in 1:L
        k = mod(div(P * (i - 1), Q), N)
        s[i] = tbl.sin[k + 1]; c[i] = tbl.cos[k + 1]
    end
    s, c
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
    P, Q = 7, 50            # exact rational step 7/50 table-steps per sample

    @testset "$T steps=$N — active backend matches reference" for (T, Ns) in cases, N in Ns
        b = default_backend(T, N)
        tbl = SinCosTable(T; steps = N)
        s = zeros(T, L); c = zeros(T, L)
        generate_carrier!(s, c, tbl, P, Q; backend = b)
        sref, cref = ref_carrier(T, N, P, Q, L)
        @test s == sref
        @test c == cref
    end

    @testset "$T steps=$N — portable matches reference" for (T, Ns) in cases, N in Ns
        tbl = SinCosTable(T; steps = N)
        s = zeros(T, L); c = zeros(T, L)
        generate_carrier!(s, c, tbl, P, Q; backend = Portable())
        sref, cref = ref_carrier(T, N, P, Q, L)
        @test s == sref && c == cref
    end

    @testset "sincos_lut! matches carrier path" for (T, Ns) in cases, N in Ns
        tbl = SinCosTable(T; steps = N)
        phases = T[ (div(P * (i - 1), Q) % T) for i in 1:L ]
        s = zeros(T, L); c = zeros(T, L)
        lookup_sincos!(s, c, phases, tbl)
        sref, cref = ref_carrier(T, N, P, Q, L)
        @test s == sref && c == cref
    end

    @testset "drift-free over long run ($T, N=$N)" for (T, Ns) in cases, N in Ns
        tbl = SinCosTable(T; steps = N)
        L2 = 200_000
        s = zeros(T, L2); c = zeros(T, L2)
        generate_carrier!(s, c, tbl, P, Q)               # default backend
        # phase index must stay exactly div(i*P,Q) mod N for ALL i (no drift)
        ok = true
        for i in 1:L2
            k = mod(div(P * (i - 1), Q), N)
            (s[i] == tbl.sin[k + 1] && c[i] == tbl.cos[k + 1]) || (ok = false; break)
        end
        @test ok
    end

    @testset "accuracy vs true sincos (Int16, N=64)" begin
        N = 64; T = Int16; A = typemax(Int16)
        tbl = SinCosTable(T; steps = N)
        L2 = 4096; P, Q = 1, 13
        s = zeros(T, L2); c = zeros(T, L2)
        generate_carrier!(s, c, tbl, P, Q)
        maxerr = 0.0
        for i in 1:L2
            θ = 2pi * (P * (i - 1) / Q) / N
            maxerr = max(maxerr, abs(c[i] / A - cos(θ)))
        end
        @test maxerr < 0.11          # 64 phase steps, floored: error ≲ one step ≈ π/64 ≈ 0.098
    end

    @testset "iterator == carrier! ($T, N=$N)" for (T, Ns) in cases, N in Ns
        tbl = SinCosTable(T; steps = N)
        sref, cref = ref_carrier(T, N, P, Q, L)
        W = SinCosLUT._val(SinCosLUT._vwidth(default_backend(T, N), T))
        nfull = (L ÷ W) * W
        s = zeros(T, L); c = zeros(T, L); i = 1
        for (sv, cv) in generate_carrier(tbl, P, Q, L)
            s[SIMD.VecRange{W}(i)] = sv; c[SIMD.VecRange{W}(i)] = cv; i += W
        end
        @test s[1:nfull] == sref[1:nfull]
        @test c[1:nfull] == cref[1:nfull]
    end

    @testset "carrier4 == carrier! ($T, N=$N)" for (T, Ns) in cases, N in Ns
        tbl = SinCosTable(T; steps = N)
        sref, cref = ref_carrier(T, N, P, Q, L)
        W = SinCosLUT._val(SinCosLUT._vwidth(default_backend(T, N), T))
        nfull = (L ÷ (4W)) * (4W)
        s = zeros(T, L); c = zeros(T, L); i = 1
        for quad in generate_carrier4(tbl, P, Q, L)
            for (sv, cv) in quad
                s[SIMD.VecRange{W}(i)] = sv; c[SIMD.VecRange{W}(i)] = cv; i += W
            end
        end
        @test s[1:nfull] == sref[1:nfull]
        @test c[1:nfull] == cref[1:nfull]
    end

    @testset "starting phase offset ($T, N=$N)" for (T, Ns) in cases, N in Ns
        tbl = SinCosTable(T; steps = N)
        φ = 7
        s = zeros(T, L); c = zeros(T, L)
        generate_carrier!(s, c, tbl, P, Q; phase = φ)
        # output[i] must be the table at (div(i*P,Q) + φ) mod N
        ok = true
        for i in 1:L
            k = mod(div(P * (i - 1), Q) + φ, N)
            (s[i] == tbl.sin[k + 1] && c[i] == tbl.cos[k + 1]) || (ok = false; break)
        end
        @test ok
        # phase = N wraps to phase = 0
        s0 = zeros(T, L); c0 = zeros(T, L); generate_carrier!(s0, c0, tbl, P, Q)
        sN = zeros(T, L); cN = zeros(T, L); generate_carrier!(sN, cN, tbl, P, Q; phase = N)
        @test s0 == sN && c0 == cN
        # Real phase in cycles: 0.25 cycles == N÷4 table steps
        sr = zeros(T, L); cr = zeros(T, L); generate_carrier!(sr, cr, tbl, P, Q; phase = 0.25)
        si = zeros(T, L); ci = zeros(T, L); generate_carrier!(si, ci, tbl, P, Q; phase = N ÷ 4)
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
        # width-agnostic reduction (W differs per backend: 64/32/16/1). Pass an explicit,
        # concrete backend so the iterator type is concrete and iteration allocates nothing
        # on every Julia version. (With the *default* backend, `default_backend` returns an
        # abstract Union that doesn't always const-fold, so iteration would box per element.)
        reduce_carrier(t, b, n) = (a = 0; for (s, _) in generate_carrier(t, 16, 125, n; backend = b); a += sum(s); end; a)
        @testset "alloc-free with $(backend_name(b))" for b in (default_backend(Int8, 64), Portable())
            reduce_carrier(tbl, b, 4096)                       # warm (compile)
            @test (@allocated reduce_carrier(tbl, b, 4096)) == 0
            # prepare callable: every lane at index 2 -> table[3], for the backend's width
            p = prepare(tbl; backend = b)
            s, c = p(SIMD.Vec(ntuple(_ -> Int8(2), SinCosLUT._vwidth(b, Int8))))
            @test all(==(tbl.sin[3]), Tuple(s)) && all(==(tbl.cos[3]), Tuple(c))
        end
    end

    println("default backends: ",
        join(["$T/$N→$(backend_name(default_backend(T,N)))" for (T,Ns) in cases for N in Ns], "  "))
end
