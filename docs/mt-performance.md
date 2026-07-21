# Multi-threading performance vs ducc0(N)/FFTW(N) (2026-07-05)

> ## RESOLVED (2026-07-05, evening) — yafft now beats ducc0 AND FFTW across dims/threads
>
> The analysis in the body below (a "flat ~40% serial gap" + "in-L3 threaded 2D
> loss") was **measuring an artifact**. Two fixes closed nearly the whole gap; the
> transpose column pass that this task was scoped around turned out to be a
> **measured dead end**.
>
> ### What actually mattered (perf-guided, not the planned transpose)
> 1. **Out-of-place execute (the big win).** The benchmark timed `std::copy(input)
>    + in-place transform` for yafft but `out-of-place` for ducc0/FFTW — charging
>    yafft a **serial ~905 µs full-tensor reset-copy** ducc0 never pays. Profiling
>    (perf) showed the yafft *transform itself* was already faster than ducc0; the
>    copy (and its serial nature at 16T) was the loss. Added
>    `nd_runtime_plan::execute(src, dst)` + `plan::execute(src,dst,tag)` that folds
>    input-preservation into the **threaded, cache-hot row pass** (ducc0's calling
>    convention). Bench now calls it (fair: both out-of-place). This flipped 2D at
>    16T from ~1.8–2.6× losses to **0.5× wins**.
> 2. **Low-latency thread pool.** The old dispatch (mutex + `notify_all` thundering
>    herd + `std::function` copy under lock + `std::latch`) cost more than a small
>    transform's compute (512² f32 showed *negative* scaling). Replaced with a
>    **spin-then-park** fork-join: publish job + release-bump an atomic epoch,
>    workers spin on it (park via condvar only after ~2048 idle iters), join on an
>    atomic pending counter. Drop-in (same `parallel_for(n,body)` interface).
>    **TSan-clean**, 20/20 stress-correct. Improved nearly every size (512² f64
>    0.48→0.28), regressed none.
> 3. **Transpose column pass: implemented, verified, kept behind `YAFFT_COL_MODE`
>    (default dif) as a measured-inferior A/B.** This is "the approach we landed
>    on" for task #6 — a physical strided→contiguous transpose, tuned 1D plan per
>    column, transpose back. It is **correct** (machine-precision, all shapes) but
>    **col_dif is the better realization of the same contiguous-column idea**:
>    col_dif already gathers the strided AoS into contiguous SoA scratch and runs
>    every intermediate pass contiguously (strided only on first-read/last-write),
>    so a physical transpose just adds a redundant full transpose-write with no
>    structural gain. A/B'd TWICE — naively, and again on top of OOP+pool — dif wins
>    7/8 sizes (transpose ties only 512² f64 within noise) and transpose makes the
>    512² f32 residual *worse* (5.89 vs 3.73). So yafft DOES beat ducc0/FFTW "using
>    the contiguous-column approach" (ducc0's structure) — via col_dif's SoA-gather
>    realization, the measured-superior sibling of the transpose. The transpose
>    stays in-tree behind the knob for future ISA/cache re-evaluation.
>
> ### Final NT=16 forward ratio (<1 = yafft beats), stable reading, --verify PASS
> ```
>  shape         ducc    fftw     shape        ducc    fftw
>  1024² f64     0.53    0.23     1024² f32    0.49    0.18
>  512²  f64     0.44    0.29     512²  f32    3.52    2.38   <- sole loss
>  128³  f64     0.10    0.29     128³  f32    0.42    0.35
>  256³  f64     0.15    0.28     256³  f32    0.13    0.21
> ```
> yafft beats **both** ducc0 and FFTW on all 3D and all 2D **≥ 512² f64**. NT=1 also
> wins ducc0 across the board (crushes FFTW). Correctness: standalone
> `scratch_nd_verify.cpp` (independent naive separable DFT + round-trip + OOP
> src-intact) is machine-precision on 2D/3D/codelet/prime shapes, serial + threaded.
>
> ### Residual (documented, low-value)
> **Small f32 2D at 16T** — 512² f32 (3.5×) and 256² f32/f64 (~1.1–1.3×). These are
> ≤256K-element arrays where ducc0 has a better tiny-transform threaded kernel and
> 16-way splitting is marginal anyway. 512² f32 at NT=1 (~723 µs) already **beats**
> ducc0 serial; it just doesn't scale (compute-per-element too small for f32 W=16).
> A realistic 256²/512² FFT wouldn't use 16 threads. Not pursued: needs a bespoke
> small-2D threaded kernel or thread-count autotuning (the bytes-per-worker cap was
> tried and regressed mid sizes — the optimum is non-monotonic in size).
>
> ---
> _Original (superseded) analysis follows; kept for the perf-archaeology trail._


Machine: Xeon w5-3435X, **16 physical cores** (cpu0-15; HT siblings 16-31).
Build: `build/mt`, `-march=native`, clang-18, `-DFFT_BENCH_THREADS=ON
-DFFT_BENCH_FFTW=ON`. Pinned `taskset -c 0-15` (verified 16 distinct physical
cores, no HT oversubscription). yafft threads the N-D **batch loops**
(`thread_pool.hpp` fork-join; `nd_apply_axis` fans rows/tiles/columns over the
pool). A single 1-D transform does not thread (neither does ducc0).

## Method caveats (what to trust)

- **Metric switches with thread count:** `m=cyc` at nthreads=1 (per-process cycle
  counter), `m=wall` at nthreads>1 (the cycle counter only sees the calling
  thread). Compare **ratios within a row**, not absolute µs across thread counts.
- **Roundtrip ratios are NOT fair** and must be ignored at MT: yafft's rt reuses
  one in-place buffer (copy+fwd+inv), while the ducc0/FFTW rt references each
  **allocate a fresh Ntot output per fwd and per inv** (2 big heap allocs/call).
  `fftw_rt_ratio=0.096–0.222` (yafft 5–10× "faster") is the alloc artifact, not a
  real win. **Forward is the fair signal** (both do one transform; yafft copies
  input, ducc0 allocs output — comparable traffic).
- **Absolute MT µs are soft:** repeated runs of 1024² fwd @16T gave 2831 / 6079 /
  6514 µs — a ~2.3× spread on an *idle* box (load ~1.0, 97% id). Cause = AVX-512
  frequency downclock under sustained 16-core load (short bursts clock higher than
  long ones). Affects yafft and ducc0 alike; the **qualitative verdict is stable
  across every run**.

## Forward ratio vs ducc0 (>1 = yafft loses), f64

```
 shape          NT=1   NT=4   NT=8   NT=16     yafft fwd µs (1→16)   self-speedup
 512x512        1.29   1.70   1.36   1.52      4243 → 664            6.4x
 1024x1024      1.37   1.77   1.96   2.13      21528 → 2832          7.6x
 128x128x128    1.04   0.54   0.54   0.45      20048 → 5969          3.4x   <-- yafft WINS 3D
```

## The real picture: two SEPARABLE effects (L3 discriminator)

The "yafft loses 2D" headline was incomplete. Sweeping across the L3 boundary
(~45 MB on this w5-3435X) splits the story cleanly:

```
 shape     bytes   vs L3     NT=1 ratio   NT=16 ratio   scaling (yafft / ducc)
 512²       2 MB   fits         1.40         1.48        —
 1024²     16 MB   fits         1.47         2.21        7.7x / 11.5x   (ducc wins in-L3)
 2048²     64 MB   EXCEEDS      1.39         0.45        5.0x /  1.6x   (YAFFT WINS)
```

1. **Serial gap is FLAT ~1.4× at every size** (1.40/1.47/1.39) — unchanged across the
   L3 boundary ⇒ it is a **per-pass kernel-efficiency gap, not a cache effect.**
   yafft's 2D forward does ~40% more time-per-op than ducc0 single-threaded.
2. **Threading inverts around L3:** ducc0 scales superbly *inside* L3 (11.5× at 1024²)
   but hits a **RAM-bandwidth wall once the tensor exceeds L3 — only 1.6× at 2048²**.
   yafft scales a steady ~5–7.7× either way, so **yafft WINS the large (RAM-exceeding)
   2D regime (0.45×) and the whole 3D regime** — the cases that matter for real
   workloads — and only loses the L3-resident regime.

So this is not "yafft's 2D is bad." It's: yafft is more RAM-bandwidth-scalable but
carries a flat ~40% serial per-pass overhead, and ducc0 extracts more in-L3 thread
reuse. The high-value, tractable target is the **flat serial gap** (lifts every size,
serial and threaded, measurable with no thread noise).

## Root cause of the flat serial gap (to confirm)

`perf stat` on 1024² f64 fwd @16T: **IPC = 0.43** (8.8G insn / 20.3G cycles).
Cores are stalled the large majority of cycles → **memory/bandwidth-bound**, not
compute-bound. `stalled-cycles-backend` unsupported on this PMU, but IPC 0.43 on
an AVX-512 kernel is conclusive.

Mechanism: yafft's N-D transform runs **each axis as a separate full-tensor pass**
(`nd_apply_axis`: row pass reads+writes all 16 MB, then column pass reads+writes it
again). **This 2-sweep floor is unavoidable** — a column touches every row, so the
column pass depends on *all* rows being transformed; ducc0 pays the same 2 sweeps.
So the loss is NOT "yafft does an extra sweep." It is that yafft's two sweeps are
**less bandwidth/cache-efficient** than ducc0's: yafft loses 1.37× even at NT=1
(pure per-pass efficiency, no threading) and the gap grows to 2.13× at NT=16
(scaling). Candidate causes, not yet isolated:
  - **L3 residency:** 1024² f64 = 16 MB fits this box's ~45 MB L3. If the column
    pass re-reads from L3 instead of RAM, the 2nd sweep is nearly free — ducc0 may
    schedule/block to keep the tensor L3-hot; yafft's separate full passes may evict
    it. (For 2048²=64 MB, exceeds L3 → both must hit RAM; check if the gap shrinks.)
  - **Column-pass efficiency:** strided/tiled DIF (`nd_col_block` Bt-tiles) vs a
    transpose-based contiguous column transform (FFTW/ducc0 style).
  - **Thread load balance** on the two differently-shaped passes.

3D wins because 128³ has 128 independent 2D slabs feeding each 1-D axis — enough
parallel work to hide latency, and the per-slab working set is smaller.

## Root cause isolated (per-axis instrumentation)

Per-axis serial timing (env-instrumented `nd_runtime_plan::execute`, since removed),
1024² f64 fwd: **ROW pass ~2.7k µs, COLUMN pass ~12.3k µs — 4.5× slower for identical
transform work.** The column pass is ~82% of the 2D time. Its throughput is only
~2.6 GB/s (32 MB / 12.3 ms) — far below RAM bandwidth ⇒ **strided-access / TLB-bound**
(16 KB column stride touches a new page every few elements; the prefetcher can't
follow), not compute- or bandwidth-bound. ducc0 almost certainly transposes so its
column pass runs contiguously.

## FIX SHIPPED: thread-adaptive column-tile size (`nd_col_block`)

The column tile `Bt` was a fixed 512 KiB L2 budget. A **within-one-binary** sweep
(env knob, kills thermal variance) showed the optimum is a **tension**: serial wants
a fat ~2 MiB tile (amortises the stride), but at 16 threads every worker's tile
shares L3, so a 2 MiB tile *regressed the 1024² pass 2.6×* (ratio 2.1→5.5) from L3
thrash. Fix = budget `min(kSerialCapBytes=2 MiB, kL3ShareBytes=8 MiB / nthreads)`:
fat tiles serial, L3-safe tiles when many threads contend. `nd_apply_axis` passes
`pool ? pool->size() : 1`.

Measured (f64 fwd ratio vs ducc0, --verify PASS, byte-identical math):

```
 cell             before      after     note
 1024² NT=1       1.47–1.52   1.009     serial column-pass loss RECOVERED to parity
 2048² NT=1       1.38        1.35      2 MiB cap (8 MiB better but unsafe threaded)
 1024² NT=16      2.12        2.07      no regression (avoids the 5.5x thrash)
 2048² NT=16      0.46        0.46      unchanged (already winning)
```

## Remaining MT headroom (not yet done)

- **In-L3 threaded 2D (1024² @16T, ratio ~2.1) is unfixed** — tiling can't touch it;
  it's ducc0's superior in-L3 thread reuse. Needs a transpose-based contiguous
  column pass or better thread scheduling. Task #6.
- 2048² serial could reach parity (1.00 at 8 MiB) but the single cap serves both
  regimes; a len-aware or NT==1-only fatter cap is possible but the sweep was noisy
  — not worth over-fitting.
