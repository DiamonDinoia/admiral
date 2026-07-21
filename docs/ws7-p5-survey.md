# WS7 P5 algorithm survey (2026-07-09, web/GitHub agent, sources cited inline)

Measured limits: (a) register ceiling (r16 f64 W=8 = 32 live batches), (b) port-5
shuffle/gather pressure, (c) L2 depth-first for 256KB-4MB band, (d) gather-in
tolerable / scatter-out terrible.

## Verdict map

| Idea | Limit | Actionable? |
|---|---|---|
| genfft Belady/MIN DAG scheduling for codelets | (a) | Yes — validates register-budget-derived scheduling (from-scratch (b)); radix-8 is the natural clean bound at 32 regs |
| FFTS JIT codelets (Blake) | (b),(c) | Mechanism = exact-width codegen; yafft already gets this via per-march compile; no new lever |
| PFFFT lane-planar (SoA-by-lane) layout | (b),(d) | yafft is already SoA re/im; confirms no-scatter designs |
| muFFT/otfft Stockham autosort | — | NEGATIVE for (c): doubles working set at the L2 boundary; avoid |
| KFR | — | No public mechanism detail |
| vkFFT two-level N=N1xN2, inner N2 sized to fast memory | (c) | **Most actionable**: CPU analogue = four-step with N2*16B <= L2/2, threshold ~8192-16384, NOT FFTW's 2^20 |
| SPIRAL / Shortest-Path FFT (arXiv 2604.04311) | (a),(b),(c) | Context-aware Dijkstra with MEASURED edge weights (pass cost depends on predecessor) — formalizes what the in-chain refits discovered empirically |
| Conjugate-pair split-radix (Johnson&Frigo 2007) | (a) | MARGINAL/NEGATIVE under SIMD (2014 SIMD paper: permute overhead eats the 6% flop saving at AVX W=8) — downgrade P5 track (a) for wide ISAs; possibly still relevant for v2 SSE small-N vs FFTW |
| ducc0 (opponent) | — | f64 SIMD capped at W=4 even on AVX-512; iterative full sweeps, no L2 blocking at 16k-64k; Bluestein for big primes. The 16k-64k band is ITS structural gap too — whoever blocks for L2 first wins the band |

## Priority feed into WS7

1. P5(c) L2-depth-first blocked driver is THE structural play for f64 16384/32768
   (and possibly 15120): ducc0 doesn't do it, FFTW only above 2^20.
2. From-scratch (b) register-budget DP constraint is validated by genfft/SPIRAL
   practice (admission rule: 2r + twiddle-live <= regs).
3. Split-radix chiplet demoted for v3/v4; keep only as a v2 SSE small-N candidate
   vs FFTW codelets.
4. Do NOT try Stockham ping-pong in the L2 band.
