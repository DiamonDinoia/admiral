# Pre-squash commit history (archived 2026-07-21)

The git history was squashed to a single commit on 2026-07-21. This file
preserves the full log (subjects + bodies) of the 142 commits leading up to
the Admiral rebrand, so the perf-campaign reasoning is not lost. Post-rebrand
work is re-squashed into the single root commit as it lands; those commit
messages are archived here too (newest first).

## 559a1fd — 2026-07-21

**refactor: update gen_base_cost_table.py emitted paths to admiral**

Follow the fft->admiral tree rename in the generator's output strings:
docstring, OUTPUT path, and the emitted namespace (fft::detail ->
admiral::detail). Regenerating now reproduces the committed header
verbatim.

## 91a1cb5 — 2026-07-21

**perf(plan): route Good-Thomas for small-N band on AVX2**

The butterfly refactor made the GT PFA kernel emit arms via callback,
reusing temporaries, so the old register-fit gate
(2*ceil(N/W)+2*max_factor+4 <= regs) is a false negative on AVX2's 16
registers. --base-cost (correctness-verified, paired-interleaved,
min-over-rounds) shows GT winning the small-N band:

  v3 f64 {10,12,15,20,24,30}  v3 f32 {10,12,15,20,30}  -> good_thomas

GT loses at N>=40 (r8-at-W4 spills at 40; PFA permute overhead at 60),
so the cost table routes it only where measured to win. v4/v2 unchanged.

- good_thomas_eligible: add the AVX2 band (N<=32 && W>=4) beside the
  register-fit path; drop PrecMask from the desc (cost table is now the
  sole precision authority)
- [[maybe_unused]] on the four mask tables: degenerate factors (N2==1)
  collapse a stage, leaving its table read only in a pruned constexpr
  index (fixes -Werror=unused-but-set-variable)
- regenerate base_cost_table.hpp + refresh base_cost_v3.txt receipts
- generator asserts only v2 (SSE) is GT-free; v3 now carries GT rows

Follow-up (not shipped): GT-as-composite-leaf / bigger-GT-block probed and
ruled a NO-GO. GT's radix set {2,3,4,5,8} caps the largest 3-factor coprime
product at 120; measured N=120 loses to iterative_dif 1.97x (f64) / 2.52x
(f32) — the register-resident kernel spills past N~30. A strided batched leaf
would hit the O(N) gather/scatter feeding wall instead. Not worth building.

## ffb1ec7 — 2026-07-21

**refactor(good_thomas): reuse generic recursive butterfly.hpp kernels**

good_thomas hand-rolled dft2/3/4/5/8 (+ GoodThomasDft{3,5,8}C constants) that
duplicate the math the DIF path already generates recursively in butterfly.hpp:
radix_sym_dft (symmetric conjugate-pair folding for odd radix) and
pow2_dif_butterfly (recursive split-radix Cooley-Tukey for pow2, trivial-twiddle
elided at compile time). good_thomas_apply_dft now gathers the arm slots into a
register buffer, runs dif_butterfly<T,true,IP>, and scatters the outputs back;
PFA is forward-only (inverse via conjugation).

Validated:
- Numeric equivalence vs the old hand kernels: pow2 (2,4,8) bit-identical,
  primes (3,5) within 1 ULP (summation order) — both precisions.
- asm @ AVX-512 (GT's only eligible ISA; the 16-reg fit gate excludes it on
  AVX2): zero new spills; identical FMA/mul for pow2; FEWER FMAs for the primes
  (radix-3 6->4, radix-5 24->16 — the hand dft5 under-exploited conjugate
  folding). Only cost is a few reg-reg gather moves.

-132 LOC. Single source of truth for the small-radix butterfly math.
149/149 tests pass.

## 03510c5 — 2026-07-21

**refactor: rebrand fft -> Admiral; minimal typed C++ API; deps to benchmarks**

Full rebrand of the library to "Admiral" (red-admiral butterfly; the FFT
kernel is the "butterfly").

C++ API (namespace admiral, <admiral/admiral.hpp>):
- 5 free names: forward, inverse, plan, plan_r2c, error.
- N-D one-shots merged into forward()/inverse() overloads (rank-1 shape = 1D).
- r2c/c2r merged into forward()/inverse() — real T* vs complex<T>* pointer
  types disambiguate; no _r2c/_c2r free functions.

C ABI (<admiral/admiral.h>) — FINUFFT convention:
- adm_ = double / precision-agnostic, admf_ = float.
- C keeps explicit _nd / _r2c / _c2r names (no overloading in C).
- adm_complex / admf_complex plain structs (no C99 _Complex layer).

Macros / CMake knobs: FFT_ -> ADM_ (FFTW_/fftw_ ABI symbols untouched).
Targets admiral / admiral_c / admiral_fftw; project admiral; find_package(admiral).

Dependencies: ducc0 and FFTW are benchmark references only (gated on
ADM_BUILD_BENCHMARKS / ADM_BENCH_FFTW), no longer library or test deps.
Tests are self-contained: analytical single-tone / DC references + round-trip
identities; Catch2 is the only test dependency. Library-only build fetches
just xsimd + poet.

README: Admiral branding + public-domain (USFWS) red-admiral photo.

149/149 tests pass.

## a364483 — 2026-07-16

**perf(plan): re-measure v4 base costs (median-of-3); route f32 N=20 to codelet**

Re-attested the v4 (AVX-512) base_cost_table on Sapphire Rapids at
-march=x86-64-v4, median of 3 independent sweeps (rounds=15 reps=12, pinned).

The b5afcff engine rework (compile-time direction + single-instantiation) sped
up the codelet path enough to overtake neighbours at the crossover:
  - f32 N=20: good_thomas -> codelet  (codelet 127.7 -> 72.8 cyc since Jul-10;
    good_thomas held ~92). Robust across all 3 runs, +26% margin.
  - f64 N=18, f32 N=21/55, f64 N=55: iterative_dif -> codelet; f64 N=45 the
    reverse. All confirmed by all-3-run agreement.
Good-Thomas is otherwise unchanged: 9 of 10 cells preserved. The borderline
cells (f64 N=20 74.4 vs 74.9, f32 N=15 86.4 vs 88.1) stay good_thomas via the
3% margin fallback. v3/v2 tables byte-identical (good_thomas is v4-only).

Also fix the generator: it emitted `static constexpr` locals, ill-formed in a
constexpr function before C++23 (project is C++20). 187b1c2 hand-patched the
header to plain constexpr but never updated the script, so re-running it
produced a non-building header. Now emits plain constexpr and reproduces the
committed style.

149/149 ctest at x86-64-v4; alderlake builds clean (route inert there).
decomp-report MISMATCH 2..2048 stays 0.

## 7c1956f — 2026-07-16

**fix(bench): score good_thomas/catalog extras from measured table in decomp-report**

decomp-report recomputed the model-optimal route from analytic candidates
(codelet_cost_cyc, dif_model_cost, ...) that omit good_thomas and read the
codelet cost array out of range past N=64. It therefore flagged every
table-routed cell (all good_thomas sizes + the 120 catalog extra) as MISMATCH
even though the planner routes those from the measured base_cost_table, which
is authoritative there.

Use base_cost_for<T>(N) as the optimal where it has an entry; add a meas column
with its cycle cost. --route-ab-dif confirms good_thomas beats forced
iterative_dif 2-3x on these sizes, so the route was correct all along.

MISMATCH over 2..2048 both precisions: 53 -> 0. Benchmark-only; no library or
routing change.

## 65dafbf — 2026-07-16

**refactor: single-source the four-step batched split table**

fsb_split_for's switch(N) and four_step_batched_dispatch's switch(N1,N2)
hand-maintained the same list of W=8 (N1,N2) splits (128-768) in two
places. Collapse to one constexpr fsb_splits table consumed by a lookup
and a capture-free fsb_dispatch_pack fold-expander, so adding a size is a
one-line edit both consumers pick up.

fsb_dispatch_pack is a free function (not a capturing IILE lambda, which
materializes a stack closure - see the dispatch-closure trap). asm-gated:
hot kernels byte-identical, delta confined to the cold dispatch path;
ctest 149/149.

## 8971e5a — 2026-07-16

**refactor: collapse good_thomas Stage A/B/C butterfly ladder**

The three PFA stages each carried a verbatim copy of the same 5-arm
if-constexpr dispatch (N==2/3/4/5/8 -> good_thomas_dftK), differing only
in radix, arm stride, and register buffer. Factor the ladder into one
good_thomas_apply_dft<Radix,B,Z> helper (Radix==1 is a no-op identity
axis); the three sites become single calls. Net -33 LOC.

asm-gated: fft_engine.cpp .s at -march=x86-64-v4 -O3 (LTO stripped),
before vs after -> asmdiff IDENTICAL (normalized), 56 good_thomas_
execute_impl symbols both. ctest 149/149 at v4 and alderlake.

The helper takes std::array<Batch,S> (not gt_array) for the same
S-deduction reason as good_thomas_gather.

## 953fa90 — 2026-07-16

**fix: restore good_thomas_gather S deduction (unbreak AVX-512 build)**

good_thomas is a live route on AVX-512 (v4/native) but is ineligible on
AVX2 (register-fit), so it is only instantiated in the v4/AVX-512 build —
which the shipping alderlake build never exercises. b5afcff swapped the
good_thomas_gather dispatcher parameter from std::array<Batch,S> to
gt_array<Batch,S>; since gt_array<Elem,Extent> = std::array<Elem,
static_cast<size_t>(Extent)>, the static_cast puts the extent in a
non-deduced context, so S can no longer be deduced and every v4/W=16/W=8
instantiation failed to compile ("couldn't infer template argument S").
This is the documented "good_thomas W=16 gap" — a real regression masked
by the ISA that instantiates it not being in the shipping build.

Revert the parameter to std::array<Batch,S> (identical type, deducible
form). v4/AVX-512 now compiles; ctest 149/149 at both x86-64-v4 (first
good_thomas correctness confirmation on AVX-512 since b5afcff) and
alderlake (inert there).

## d18ffa6 — 2026-07-15

**feat: C/FFTW interfaces, CMake install/export, and internals dedup**

Public interfaces:
- FFTW-compatible layer (fftw3.h + fftw_compat.cpp) and C interface
  built as self-contained SHARED libs baking $<TARGET_OBJECTS:fft_engine>
  with hidden visibility (only C symbols exported).
- Optional ducc0-style `fct` output-scale parameter (normalization stays
  opt-in, no implicit 1/N).

Packaging:
- CMake install/export: only fft::fft_c and fft::fftw are exported
  (header-only fft::fft would drag xsimd/poet into the export set);
  EXPORT_NAME maps fft_fftw -> fft::fftw. fftConfig.cmake.in + version
  file with SameMajorVersion. README find_package usage updated.

Internals (perf-verified, asm-gated):
- four_step_large: fold W_col/W_ into a single Wv (engine .o byte-identical).
- Share Bluestein cost model as constexpr bluestein_model_cost in math.hpp
  (countr_zero for log2); only the cold cost functions change and shrink,
  every hot symbol byte-identical, routing bit-identical.
- good_thomas: collapse Stage A/B/C consteval mask-table generators into one
  builder taking a mapping lambda (engine .o byte-identical, consteval).
- Comment/dead-code cleanup across detail/ headers (no codegen change).

All 149 tests pass. No hot-path regression by construction.

## 36fee33 — 2026-07-14

**docs: record single-1D-transform-MT verdict as measured NO-GO**

Replace the "Phase-4 deferred" roadmap prose with the measured reason:
large-N iterative_dif is DRAM-bandwidth-bound (32 MiB f32 = 47.5% backend-
bound at 73% LLC-miss, no thread scaling), small-N is barrier-dominated.
Also drop a doubled comment marker in thread_pool.

## 6a9a935 — 2026-07-14

**test: gate serial-vs-threaded on analytical FFT rounding bound**

nthreads=1 vs 4 is not bit-identical under -ffast-math: threading regroups a
SIMD pass's FMAs (chunk vs full sweep), perturbing the last bit. Same math,
two valid FFTs. Replace REQUIRE(a==b) with a normwise-L2 tolerance
16*u*log2(N) derived from Higham (Accuracy & Stability, Thm 24.2): each path
obeys ||fl(y)-y||/||y|| <= c*u*log2(N), so their difference <= 2c*u*log2(N).
Forecast validated: {512,513} f64 predicts ~1.6e-14, measured 1.4e-14.

## b5afcff — 2026-07-14

**feat: threading, compile-time direction, and single-instantiation engine**

Threading (#1): resolve_nthreads (0 -> min(hw,16)) and an nthreads param on
every one-shot 1-D/N-D surface + C API *_threaded; plan-owned jthread pool
splits the N-D batch/row/column loops; r2c odd-N row loop threaded.

Compile-time direction (#2): lift is_forward to a template once at the top of
execute(); every route is Forward-templated so the hot path carries no runtime
direction branch. good_thomas/codelet_dispatch call sites updated.

Control-flow (#3): fused-pair ladder -> poet::dispatch; shared
emplace_route_state().

Folded-conj inverse: inv(x) = conj(fwd(conj(x))), only the forward kernel is
instantiated (eps-correct, pow2 bit-identical) -> smaller .a, lower TU memory.

Extern-template engine: out-of-line plan_impl/nd_runtime_plan heavy members
and instantiate them once in src/fft_engine.cpp (OBJECT lib propagated via
target_sources); public API stays inline+thin. Consumer-TU compile peak drops
~3.1 GiB -> ~0.3-0.4 GiB. Hot code is one COMDAT def, byte-identical; the only
runtime delta is one non-inlined call per whole transform. nd_real_plan is
kept inline by design.

good_thomas -Werror: single gt_array<Elem,int> alias + C-array bound casts,
pragma removed.

PCH (xsimd/poet) on fft_codelets and fft_engine.

135/135 ctest, --verify all PASS.

## e7563a6 — 2026-07-14

**chore: repoint xsimd pin to upstream master a4b9113 (#1379 merged)**

The avx256 plain-move lowering (633553e) plus its AVX-512 extension now
live on xtensor-stack/xsimd master; the DiamonDinoia fork is retired.

## 52fba3b — 2026-07-12

**fix: build benchmark under clang fast-math with FFTW reference**

Enabling the FFTW reference (-DFFT_BENCH_FFTW=ON) compiles the FFTW path and
the f32 compare/base-cost templates that were never instantiated before, and
clang-18 rejected them under the project's own -ffast-math -Werror profile:

- -ffinite-math-only makes infinity()/isinf/isfinite/quiet_NaN UB, and clang's
  -Wnan-infinity-disabled flags every use. Replace the 4 sites with the large
  finite max() sentinel already used elsewhere in this file (see the
  choose_four_step_split cost comment).
- -Wdouble-promotion on std::complex<double>(cfloat.real(), .imag()) in the
  accuracy gate: wrap each component in static_cast<double>.

No behavior change (sentinels are only compared, never arithmetic). Verified:
full yafft-vs-ducc0-vs-FFTW sweep builds and runs clean on clang-18 v4.

## 30496d8 — 2026-07-12

**test: add coverage suite for routed sizes, large-align, and plan edges**

Add test_fft_coverage.cpp (batched four-step routed sizes, Rader prime with
four-step inner convolution, and other under-covered route paths) and
test_fft_large_align.cpp (large-N four_step_large alignment cases), registered
in test/CMakeLists.txt.

Harden test_fft_factor_plan: the additive-DP optimality invariant does not
hold for pow2 N in the fusion band (the planner minimizes a richer
fusion-discounted cost there), so skip those; and use max() rather than
infinity as the min-search sentinel since infinity is UB under -ffast-math.

## c184538 — 2026-07-12

**refactor: remove structurally-unreachable direct O(N^2) route**

The direct DFT route was dead for every N. select_route returns codelet for
any N in the catalog, and the catalog covers all of [2..64]; the later
is_small_direct guard requires N <= 64, so it could never be reached (dead
after the always-taken codelet return, and false for N > 64). Every
non-7-smooth N <= 64 already routes to a codelet Rader chiplet.

Remove the whole engine: direct.hpp (O(N^2) direct_dft_twiddles / execute),
its route_kind::direct / direct_state / construct / route_name / execute_route
in plan.hpp, is_small_direct + DIRECT_DFT_THRESHOLD in math.hpp, and the
direct.hpp include in kernels.hpp. Behavior-preserving (route never selected);
drops the header from every fft.hpp consumer TU.

## 493f233 — 2026-07-12

**fix: guard store-align peel against W < LANE column drop**

aos_store_align_peel's head stores the peel window [0, peel) from a single
W-wide block, so it requires peel <= W. peel ranges over [0, LANE-1], which
only stays within W when W >= LANE. On narrow ISAs where W < LANE (SSE2 f64
W=2, SSE f32 W=4 vs LANE 4/8) a peel of LANE-1 exceeded W and the single head
block silently dropped columns [W, peel), leaving them with stale pre-pass
values in dif_col_pass_last. Sanitizer-invisible; alignment/heap-layout
dependent. Disable the (perf-irrelevant on those baselines) cache-line peel
when W < LANE.

## 187b1c2 — 2026-07-12

**refactor: silence -Wsign-conversion in good_thomas mask builders**

The consteval Masks{1..8} builders subscript std::array<int,W> with a signed
loop index and (U)-cast the results, both of which trip -Wsign-conversion under
the -Werror profile. Add a constexpr gt_at(arr, i) helper that casts the index
to size_t once (all indices are provably in [0,W)), and switch the C-casts to
static_cast<U>. Behavior-preserving — same compile-time mask tables.

Also drop the redundant `static` from the function-local constexpr base-cost
tables; they are already immutable constexpr, static only added per-instantiation
storage.

## 6bcadac — 2026-07-12

**build: enable -Wsuggest-override and treat Catch2 as SYSTEM**

-Wsuggest-override is supported by clang-18 (JT lists it under GCC only);
add it to the clang warning set. Mark the Catch2 / Catch2WithMain interface
include dirs SYSTEM so their internal float->double matcher conversions do
not trip the project -Werror -Wconversion profile, same treatment as
xsimd/poet.

## 6ac3e17 — 2026-07-11

**Merge: fat col tile for four_step_large (1.2-1.5x, DRAM traffic down)**

Root-caused the large-N four_step_large DRAM-traffic gap to a tile-starved
col-DFT and fixed it with a route-specific fat-tile rule. Both goals met:
DRAM traffic down 12-16% AND wall-clock 1.17-1.52x across both ISAs, ST+MT,
all routed sizes. See commit for measurements and correctness gates.

## 4988749 — 2026-07-11

**perf(four_step_large): fat col tile (Bt 8→64) — traffic down + 1.2-1.5x**

The col-DFT dominates four_step_large DRAM traffic (per-phase IMC: 1294 of
2470 MB/it at 16M f64, read-heavy) and was tile-starved: it reused
nd_col_block, whose L2 budget caps Bt=8 for the large square factors here
(n2≈√N=4096). That budget is right for the ND column pass (which contends
with sibling per-thread working sets) but wrong for this route — it STREAMS
DRAM and its SoA scratch belongs in L3. The binding cost is per-tile overhead
(reloading the length-n2 twiddle set + poet::dispatch over n1/Bt=512 tiles),
so the optimum is ~64 columns, N-independent over a 16x range (column-count-
bound, not scratch-bound).

Give the col pass its own rule: Bt = min(n1, 64, L3/3 scratch cap), W-aligned
— a single amortization constant with a cache-derived ceiling (not a per-size
table). ND path (nd_col_block) untouched.

Measured, role-swapped min-of-N vs shipped Bt=8:
  ST v4: 1M 1.41x / 1.99M 1.19x / 4M 1.45x / 16M 1.36x
  ST v3: 1M 1.32x / 1.99M 1.18x / 4M 1.32x / 16M 1.17x
  MT [16,1M] 16 threads: v4 1.52x, v3 1.32x (fat tile does not thrash L3 —
  the case is DRAM-bound, the traffic cut outweighs L3 contention)
  DRAM traffic (col-DFT) down 12-16%.

Wins all cases both ISAs, ST+MT, with traffic reduced — reducing DRAM traffic
AND improving wall-clock together, which the earlier write-side levers
(bandbuf, NT stores) could not.

ctest native 116/116 (incl. ND four_step + nthreads 1-vs-N bit-identity);
v3 four_step accuracy machine precision (fwd_vs_dif ≤8e-16, roundtrip ≤1.9e-15).

## c47232b — 2026-07-11

**perf(four_step_large): fuse transpose into row band above L3 + j-outer reorder**

The final transpose out[k1*N2+k2]=W[k2*N1+k1] ran as a separate full-array pass
after the row loop. At N>L3 (4M/16M f64) it re-read all of W from DRAM (W is 64MB
> 45MB L3), showing up as ~19% self in perf. Two changes, both memory-locality
(the WxW register kernel instructions are byte-identical):

1. j-outer transpose loop order (four_step_transpose_band): iterate the output-
   column block outermost so each output row is written contiguously across the
   full i-range, instead of the old i0/j0 BxB tiling. Wins even when non-fused.
2. Fuse the band transpose into the row loop when W exceeds L3: transpose each
   W-aligned band of `four_step_tblock` rows right after the row-FFT computes it,
   while still hot in L2 -> kills the DRAM re-read. Cache-derived gate
   (N*sizeof(complex) > L3), not a size table: below L3 the separate pass re-reads
   W from L3 for free and fusing only adds L2 pressure (it regressed 1M f64 AVX2
   1.4% ungated).

Paired role-swapped A/B (taskset -c 2, min-of-N, interleaved fused-vs-baseline):
routed f64 wins BOTH ISAs -- v4 1M 1.06x / 1.99M 1.08x / 4M 1.05x / 16M 1.03x;
v3 1M 1.07x / 1.99M 1.09x / 4M 1.04x / 16M 1.02x. f32 and sub-12MiB f64 unchanged
(not routed). ctest 116/116; ND [8,1M] MT nthreads 1-vs-8 bit-identical; accuracy
machine precision (fwd_vs_dif 7-9.5e-16, roundtrip <=2.7e-15) incl. non-pow2.

## 47c05bc — 2026-07-11

**perf(four_step_large): fuse copy-in into first col pass (1.02-1.23x)**

The route staged the const transform input into its thread_local work buffer
with std::copy(in, in+N, W) before the inner column DFTs -- a full-array
read+write (memmove) that perf put at ~9% of the transform and that duplicates
the read the first col pass performs anyway.

col_dif_execute_ws only touches its AoS buffer in the first pass (gather) and
last pass (scatter); all intermediates ping-pong SoA scratch. So add a defaulted
first_src param: the first pass reads its strided AoS input from first_src while
the last pass still writes data. four_step_large passes first_src=in (per column
block), so the first gather reads the input directly and the memmove is gone.
All ND callers use the default (nullptr -> in-place) and are unchanged. In-place
(in==out) stays correct: every in read happens in the col loop before the
transpose writes out (same decoupling the copy gave).

Paired A/B (role-swap, taskset -c 2, 7 runs): v4 f64 1M 1.23x / 1.99M 1.04x /
4M 1.04x / 16M 1.02x; v3 f64 1M 1.18x / 4M 1.04x / 16M 1.02x. Largest at 1M
(16 MiB) where the copy's 32 MB traffic was the biggest relative share; smaller
at 16M (pass traffic dominates). No regression any size/ISA. ctest 116/116;
fslcheck machine precision (5-9.5e-16); ND [8,1M] in-place MT bit-identical.

## 2129f5a — 2026-07-11

**perf(four_step_large): vectorized WxW register transpose (1.04-1.14x)**

The final four-step transpose out[j*n2+i]=W[i*n1+j] was a blocked (B=16)
scalar scatter: each 16 B complex stored into a fresh 64 B line -> partial-
line RFO stores, ~33% of the transform (perf self, execute()) and the route's
latency bottleneck after the DRAM-traffic cut.

Replace with a blocked WxW register transpose (W = native batch width),
expressed entirely on xsimd batches (no raw intrinsics): deinterleave W source
rows into planar re/im, xsimd::transpose both planes (same primitive as the DIF
subgroup transpose), interleave back as W output rows -> each strided scalar
scatter becomes W contiguous full-vector stores. Width-parametric; scalar
remainder for the W-ragged right/bottom edges. B=32 (cache-block) picked by a
standalone transpose sweep (4x4/2x2 micro-kernels, B in {16,32,64}).

Transpose microbench (square, vs scalar B=16): AVX-512 1.24x (16M)..1.80x (1M),
AVX2 1.25..1.49x. End-to-end paired A/B (role-swapped, taskset -c 2, 7 runs):
  v4 f64  1M 1.14x / 1.99M 1.08x / 4M 1.06x / 16M 1.04x
  v3 f64  1M 1.11x / 1.99M 1.08x / 4M 1.08x / 16M 1.05x
No regression on any size/ISA. Codegen audit: vshuff64x2 + vpermt2pd + vmovupd
(vectorized, no scalar bulk). Transpose is a permutation -> accuracy unchanged
(fwd 5-9.5e-16). ctest 116/116 native; ND [8,1M] MT bit-identical.

## 70ed2f8 — 2026-07-11

**fix(four_step_large): split-twiddle twist for machine-precision four-step**

The incremental twist (cur *= W_N^{k2} per column) drifted to ~5e-14 fwd /
1.6e-13 roundtrip over n1 steps — ~60x above the intrinsic FFT floor, not
machine precision. Replace with a classic two-table split twiddle:

  q = n1*k2 accumulated as an EXACT integer
  W_N^{n1 k2} = hitab[q>>logM] * lotab[q&(M-1)],  M = 2^ceil(log2 sqrt N)

Each table entry is a single exact sincos, so the twist is one rounded
complex mul — no accumulated drift — at O(sqrt N) memory (128 KB at 16M).
qhi/qlo are a shift and a mask (M is a power of two).

Accuracy fwd_vs_dif 5e-14 -> 5-9.5e-16, roundtrip 1.6e-13 -> 1.3-2.7e-15
(both v3 and v4), now at the pure-iterative_dif floor. Perf unchanged
(twist is O(N) vs the O(N log N) FFT): v4 f64 1M 1.49x / 4M 1.39x /
16M 1.26x / 1.99M 1.58x. ctest 116/116 native; ND [8,1M] MT bit-identical.

## 9b3d82d — 2026-07-10

**perf(plan): large-N four-step route for DRAM-bound f64 (1.25-1.5x)**

Large 1D f64 FFTs (>12 MiB, e.g. 1M-16M) were DRAM-bandwidth-bound: the flat
iterative_dif pass chain read+writes the whole array ~log_radix(N) times
(measured 4M f64 = 1.71 GB/transform = 13.3x the r+w minimum, pinned at the
14.6 GB/s single-core DRAM ceiling).

New four_step_large route factors N = n1*n2 with both leaves cache-resident and
reuses the ND row-column machinery: strided inner col-DFTs (col_dif_execute_ws
+ nd_col_block tiling) -> fused incremental twist W_N^{n1 k2} -> contiguous
outer row-DFTs (iterative_dif) -> one cache-blocked transpose. No new
butterfly/kernel code. The blocked transpose is essential: a naive element-wise
strided scatter lost everywhere (page/TLB thrash); tiling it (B=16) flipped to
wins.

Measured (role-swapped min-of-N, pinned, box idle):
  - DRAM traffic 1.71 GB -> 512 MB per transform (3.3x less; perf cache-misses).
  - v4: 1M 1.37x, 4M 1.37x, 16M 1.25x, 1.99M 1.50x vs flat iterative_dif.
  - v3: 1M 1.25x, 4M 1.43x, 16M 1.31x, 1.99M 1.55x.
  - Now latency/IPC-bound (0.84 IPC, strided gather + transpose), not BW-bound.

Twist is the O(N2) per-row ratio W_N^{k2} rotated incrementally (O(N)->O(N2)
plan memory: a 16M plan's twist drops 256 MB -> 32 KB); f64-only route so the
drift over n1<=4096 steps (~5e-14) stays far under the 1e-11 accuracy gate.

Working buffer is thread_local + reused: MT-safe (the ND row driver shares one
plan_impl across worker threads and calls execute() concurrently on distinct
rows, nd_plan.hpp:150 -- a mutable member would race; a fresh per-call alloc
re-faults 64-256 MB every transform and erased the win at 4M/16M). ND [8,1M]
f64 nthreads 1-vs-8 is bit-identical.

Gate is f64-only + array >12 MiB (a precision fact, not a per-size override:
f32 reaches ~2x further per byte in cache so its iterative_dif stays ahead
until N where the four-step gain is within noise -- measured sweep confirms
f32 loses/ties across 2-128 MiB; same shape as the f32-only four_step_batched).

cache_bytes/cpu_cache/nd_col_block moved nd_plan.hpp -> dif_col_driver.hpp
(their natural home, next to col_dif_execute_ws) to avoid a circular include.

Correctness: ctest 116/116 (v4); four_step_large vs iterative_dif machine-
precision fwd + round-trip (v3+v4, pow2 and non-pow2 smooth N); ND MT
bit-identity.

## 4572948 — 2026-07-10

**perf(dif): cache-line store-align peel in dif_col_pass_last**

The ND forward bottleneck is the last column pass scattering W-complex AoS
blocks to `data + p*axis_stride + c`. A std::vector is only 16B-aligned, so
every 64B wide store straddles a cache line; the split-store penalty dominates
the whole ND transform.

Peel the misaligned head columns so the vectorized bulk stores land on 64B
boundaries (aos_store_align_peel, gated on axis_stride % LANE == 0 so every
output row shares one peel). The head and ragged tail are written by a new
aos_interleave_window masked store (one `vmovupd {%k}` per half, lanes [m0,m1))
— each output column is touched exactly once (no re-stores) and stays on the
IDENTICAL vector arithmetic, preserving nthreads=1-vs-N bit identity (a scalar
butterfly contracts FMAs differently). An earlier overlapping-full-block cut
won ST but regressed MT via redundant re-stores; the windowed store fixes both.

Role-swapped min-of-6, 1000^2 forward, w5-3435X, one core free. peel vs the
pre-peel baseline improves all 8 cells: f64 v4 ST +8.0% / MT +14.4%; f64 v3
ST +11.5% / MT +12.2%; f32 v4 ST +5.3% / MT +3.1%; f32 v3 ST +7.7% / MT +2.9%.
Beats ducc0 on 7/8 cells. ASM: bulk = plain unmasked vmovupd; head/tail =
vmovupd {%k1}; no scalar fallback, no double-store. ctest 116/116.

## bfdff0a — 2026-07-10

**refactor(xsimd): adopt AVX-512 constant-mask plain-move branch + strip .value casts**

Repoint xsimd to fork branch avx-avx512-constant-mask-plain-moves (2ff7cb4),
which extends the avx256 plain-move lowering (633553e) to AVX-512: compile-time
batch_bool_constant load/store lower to plain moves (not vmaskmov/kmask) on
AVX-512 too. Relevant on this box where -march=native is W=16/8.

Adopting it exposed a latent OOB in codelet.hpp: the cofactor pad loop
static_for<0, Mp-M> is zero-trip when Mp==M, but poet::static_for's
is_invocable_v probe still instantiates the body with P=0, so ovr[M] on a V[M]
is a compile-time -Warray-bounds (the branch's more constexpr-friendly V ctor
let clang fold the index it previously couldn't). Guard with
if constexpr (M + P < Mp) — general, helps every arch.

Simplify all 37 static_cast<std::size_t>(X.value) -> bare X across
butterfly/vecpass/dif_passes/dif_col_pass: integral_constant converts
implicitly and the values are known non-negative constants, so the
-Wsign-conversion carve-out holds.

Profile-gated (and rejected) a masked-tail rewrite of the runtime dif_col_pass
c-tails: profile shows those tails are cold (0% on aligned/pow2 ND; top-20 hot
insns all zmm even on unaligned 252x252 f64). 116/116 tests pass.

## fe51ce6 — 2026-07-10

**docs: correct README to C++20 and gitignore scratch_* files**

README claimed C++17 but CMakeLists sets CMAKE_CXX_STANDARD 20 REQUIRED;
a gcc-8 contributor would hit a confusing failure. .gitignore already
covered scratchpad/ and *.log but not the scratch_* session files
cluttering the working tree.

## 9269c22 — 2026-07-10

**perf(plan): BASECOST refresh + WS8 acceptance receipts**

Refresh the BASECOST epoch (sizes 2..64 + 120, both precisions, per ISA)
at the post-WS8 HEAD; regenerate base_cost_table.hpp. Acceptance receipts
(FFT_BENCH_FFTW_MEASURE=1 --fftw-ab, per-ISA rt tables, WS8 P0 census):
vs the fair ISA-matched opponent ducc0, yafft wins every tier-1 cell
(~1.2-3.6x); vs in-process MEASURE-FFTW, v4 wins outright at 360 f64
(0.79), 720 f32 (0.62), 2520 f64 (0.92) with the pow2 store-bound band
closed to ~1.07. 60/120 FLOP deficit is the remaining open lever.

Full per-candidate A/B and profiling receipts for the whole WS4-WS8
workstream are preserved on the archive/ws4-ws8-receipts branch.

## b2f9090 — 2026-07-10

**perf(dif): eliminate dif_pass_last scalar tail via constant-mask remainder**

Replace the auto-vectorized scalar tail (setb-storm alias versioning) with
a remainder strategy picked once per call: l1<W -> compile-time partial
chiplet (dead rows zero-inited, prefix constant-mask stores); rem>=W/2 ->
overlapped full-width block through the hot loop body; 0<rem<W/2 ->
NOINLINE-outlined scalar rows. The block body is a free function template,
not a lambda, so poet::dispatch doesn't materialize a closure the hot loop
reloads through (28 pointer reloads -> 4). aos_interleave_prefix stores the
first R of W planar complex with prefix constant masks; xsimd pinned to
fork branch 633553e so those lower to plain moves, not vmaskmov.
Paired-interleaved receipts: v3 360 f32 0.95, v4 720 f32 0.62.

## 36ed79d — 2026-07-10

**perf(dif): WS8 copy-free OOP, tiny-N codelets, col-major L1 passes, fusion**

Copy-free out-of-place execute on every route; good_thomas entry/exit
aliasing-templated so OOP no longer memmoves. Tiny-N: odd-cofactor peel +
vector combine tails + a 120 catalog extra; measured 120 f64 W=8 order
[8,15]. Column-major L1-resident passes for radix-4 (16-reg ISAs) extended
to radices 2/3/5 at W>=4; row-space pre-split last pass on 32-reg ISAs.
Admit middle-pass fusion at N=4096 (16-reg) and N=2048 (f64, 16-reg).

## 5436347 — 2026-07-10

**perf(planner): measured terminals, register restage, merged radix-9/15**

f64 W=8 16384 measured [16,16]+64 terminal (1.42 -> 0.98 vs ducc0).
Two-sweep restage for register-ceiling DIF butterflies. Merged radix-9/15
passes on 32-reg ISAs (15120 f64 beats ducc0). ND: parallel balance cap on
column tiles closes the full ND/MT matrix. Dead pow2_dif_butterfly_ld
removed. WS7 P2/P3/P4/P5 receipts: DP chains already measured-best, the
15120 residual is odd-radix kernel compute, split-radix chiplet is the sole
open item.

## 718d8fd — 2026-07-10

**feat(planner): DP base-kernel terminal + measured pow2 overrides**

Lane-packed DP terminal + enumerator terminal edge (dormant by receipt).
Fuse terminal output stores into the kernel_batched last combine. Measured
f64 W=8 4096 order [8,8,8,8] (vs ducc0 0.81 -> 0.70). Bench harness gains
--factors-ab with a +k codelet terminal on either side.

## 8e08f8d — 2026-07-10

**perf(kernels): Good-Thomas PFA route + measured base-cost table**

Generic Good-Thomas prime-factor route (N=60 beats FFTW both precisions),
extended to small N (10,12,15,20,24,30,40) with per-cell receipts. A
measured base-cost winner table replaces hand-maintained small-N routing;
DP base-kernel terminal mechanism lands dormant (receipt-gated). Small-ido
kernel replaced with sized-batch ISA mixing; dead radix-8 small-ido arm
removed. Planner internals refactored: pfa_catalog descriptor pack, variant
route state + visit dispatch (retired vecpass deleted), pfa -> good_thomas
rename. Deps: xsimd PR#1376 (constant-mask load fixes), poet Unroll==1
no-unroll contract.

## 6189ca0 — 2026-07-10

**perf(planner): wide-radix DP + width-adaptive dif_pass_last**

Admit radices 16/32 in the DP everywhere (retires vecpass + fsb at W=16)
and re-fit the f32 W=16 r5/r8 valley/last cells in-chain so smooth-size
chains flip to a trailing-8. Run dif_pass_last at W' = min(W, bit_ceil(IP))
so small-IP and l1<W last passes stay vectorized. In-chain refit of the
large-N cost constants flips f64 65536/262144 vs ducc0.

## 7593420 — 2026-07-10

**perf(planner): measured ISA-keyed cost model + 3-ISA acceptance**

Replace the hand-maintained routing heuristics with per-ISA measured cost
tables (v2/v3/v4) plus an analytical fallback, and width-key the
vecpass/fsb routes from role-swapped A/B. Unlocks the M=8 flat-leaf path.
Foundation: 1-permute AoS<->SoA DIF-boundary swizzle and the xsimd pin
(SSE2 native 3-of-4 float mask, 50d20bc). v2/v3/v4 acceptance closed
against ducc0.

## b43f026 — 2026-07-05

**perf(nd): cache-size-driven tile budget + thread-pool false-sharing fix**

Replace the hardcoded column-tile constants (2 MiB serial / 8 MiB L3 slice) with
values derived from the machine's real cache geometry, probed ONCE via a
function-local static (cpu_cache) and reused -- geometry is a property of the
machine, not the transform. OS-portable query per target: sysconf on Linux,
sysctlbyname on macOS/BSD, measured fallback elsewhere. Budget = min(L2, L3/6 /
nthreads); on this box (L2=2 MiB, L3=45 MiB) that reproduces the tuned 2 MiB /
~8 MiB sweet spots, and other machines scale automatically. No runtime ISA
dispatch -- this only sizes cache tiles; SIMD width stays compile-time fixed.

thread_pool: epoch_ (spin-read by every worker) and pending_ (fetch_sub'd by
every worker + the caller's join spin) were adjacent 8-byte atomics sharing one
cache line, so each pending_ RMW invalidated the line every spinner reads --
false sharing. alignas(64) splits the read-mostly dispatch line (job_/epoch_/
stopping_) from the churned pending_ counter, mutex on its own line.

Verified: build clean; ND DFT reference 14/14 PASS ~1e-15 (serial+MT); compare-nd
16T ratios within run-to-run variance of baseline (best-of-3 512^2/1024^2 f64
matches pre-change), still beating ducc0 and FFTW.

## 1574aeb — 2026-07-05

**perf(nd): out-of-place execute + low-latency thread pool; audit-driven cleanup**

MT campaign result: N-D forward now beats ducc0 and FFTW across all 3D and
2D >= 512^2 f64 at 16T (e.g. 1024^2 f64 0.43/0.18, 128^3 f64 0.14/0.43 vs
ducc0/FFTW forward ratios).

- nd_runtime_plan::execute(src,dst): out-of-place path so the innermost row
  pass copies+transforms cache-hot, instead of the benchmark charging yafft a
  serial full-tensor reset-copy that ducc0/FFTW never pay (the real 16T loss).
- thread_pool: spin-then-park fork-join replacing mutex + notify_all + latch;
  drop-in parallel_for, TSan-clean, 20/20 stress-correct.
- dif: plan-time fusion schedule + packed fused2 twiddle stream (four_step,
  dif_passes, vecpass); sweep-aware pow2 chain enumeration.

Audit cleanup (default execution path byte-identical, verified):
- drop the YAFFT_COL_MODE transpose A/B column pass -- measured slower at every
  size; col_dif already performs the fused strided<->contiguous transform.
- outer dif axes no longer construct an unused per-column plan_impl at plan time.
- remove dead both_tag, rader_plan is_forward field, unreachable size==0 guard.
- thread_pool stopping_ store release (pairs with workers' acquire load).

Verified: build clean; independent ND DFT reference 14/14 PASS ~1e-15
(serial+MT, fwd/roundtrip/out-of-place/src-intact); compare-nd ratios match
baseline (no regression).

## 1deae99 — 2026-07-04

**docs: band-B campaign close-out — W4 route re-validated, MEASURE headline, causes/mitigations**

route-ab on the post-W3 binary: fsb keeps f32 32768/65536 (dif rt still
1.21-1.24x slower; the new chain narrowed fwd to 1.03-1.07). MEASURE
headline rounds=15: f64 fwd 1.31-1.63, f32 1.48-1.77 across band B;
262144/1M skipped (MEASURE planning + rt gate exceed 45min). Residual gap
attributed to genfft-style cross-butterfly scheduling on a latency-bound
kernel; remaining levers AVX-512-gated (U=2 pipelined_for, r16) or a
codelet-scheduler project.

## 1fa152d — 2026-07-04

**perf(build): drop -funroll-loops — hot kernels are flag-invariant**

User-directed hot-loop unroll audit (full per-symbol tables in
docs/bridge-evidence/unroll-audit/): per-TU recompiles at exact production
flags with and without -funroll-loops show every perf-critical kernel
(dif_pass r4/r5/r7/r8, all fused2 pairs, fused3, every codelet_apply,
col_dif r4/r8, transpose packs, recomb) emits byte-identical code either
way — their unrolling is already explicit via templates/poet::static_for.
Where the flag did act it only ADDED code (dif r2/r3 +15-111%, vecpass f32
+30% instrs for +13 FMAs, r2c f64 WxW pack +380%, four_step_batched f32
+11%) with zero spill-count changes.

Alternated taskset+perf-stat cycle A/Bs (flag-on vs flag-off binaries, FFTW
as constant shared .so; 5-size basket 1260,2048,2520,4032,5040 and
r3-focused 1260,2520,5040; 3-4 pairs; both precisions): all deltas <1.2%
inside 2-7% spreads with inconsistent sign -> neutral. The 2026-06
"+5-28% f32 regression without the flag" was pre-fusion kernels and does
not reproduce. Codegen is now fully source-determined. ctest 114/114.

## 6b03c08 — 2026-07-04

**perf(planner): sweep-aware exhaustive pow2 chain enumeration for the fusion band**

enumerate_pow2_dif_plan<T>: for pow2 N >= kDifFuseMinN (8192) score every
ordered {4,8}-composition of lg N by sum(dif_stage_cost) + 0.8*N*sweeps,
sweeps counted from dif_fusion_schedule (the row driver's real fused2/fused3
gates). Alpha calibrated on the 7-point interleaved factor sweep; every alpha
in [0.4,1.45] reproduces all binding verdicts. dif_fuse/dif_fusion_schedule
moved above the planner (unchanged) so the enumerator can simulate sweeps.

Only chain change vs the DP: 32768 both precisions -> {4,4,4,8,8,8}.
Interleaved cyc A/B: f64 fwd 0.933 rt 0.955 (A/ducc 0.930, beats ducc0),
f32 fwd 0.939 rt 0.944 (f32 not in calibration set - independent confirm).
ctest 114/114.

The 2048/4096 measured overrides stay: re-A/B'd under the fused+merge driver
(0.904-0.952, robust wins, not ties). Below the fusion band the sweep term is
order-invariant while those wins are pure pass-ORDER working-set effects;
alpha >= 2.0 (needed to force f32 4096 {8,8,8,8}) breaks the calibrated f64
65536 non-flip. Evidence: docs/bridge-evidence/w3-chain-ab-2026-07-04.txt.

## 17a3e16 — 2026-07-04

**docs: W1a/b + W2 verdicts, cross-binary PMU protocol finding, W1 evidence**

W1a shipped perf-neutral (spills sat in the latency shadow); W1b prefetch
NO-GO; cross-binary tsc ratios at band B carry >±30% layout luck — verdicts
gate on pinned perf-stat total-cycle deltas from now on.

## 00a16c7 — 2026-07-04

**refactor(dif): plan-time fusion schedule + packed fused2 twiddle stream**

dif_fusion_schedule<T> (twiddles.hpp) is now the single source of truth for
which middle passes fuse: the row driver executes the schedule with no runtime
gate re-checks, and build_dif_twiddle_set chooses each pass's representation
from it — fused2 pairs store BOTH layers as one consumption-ordered packed
stream (single advancing pointer, compile-time load offsets) and drop their
plain tables, so twiddle footprint is unchanged. N-D column-driver sets opt
out via fuse_packed=false (col_dif reads plain tables, never fuses). The
plain-twiddle fused2 kernel and the FFT_TWPACK/FFT_PREFETCH experiment knobs
are deleted (prefetch measured no-effect: HW prefetchers cover the streams).

ASM (objdump census, f64 fwd 4,4 hot inner loops): 74 -> 59 instructions,
4 -> 0 GPR spill-reloads, 0 YMM spills, llvm-mca alderlake RThr 11.0 -> 10.3.
Whole-plan PMU cycles are UNCHANGED (+-0.05% f64, +0.002% f32 across band B,
taskset -c 2, --fftw-ab rounds=15): the kernel is latency-bound and the spill
movs sat in the dependency-stall shadow. Shipped as an enabler: frees ~15
uops/iter and GPR headroom for the U=2 double-pump (W1c), and the plan-time
schedule is the prerequisite for the sweep-aware pow2 enumerator (W3).

ctest 114/114.

## c1e18db — 2026-07-04

**docs: band-B bridge plan + preserved evidence; W2 Linzer-Feig resolved NO-GO**

Bridge plan committed as knowledge base with execution state. W2 resolved
NO-GO by paper derivation (arXiv:2604.00567 + Linzer-Feig 1993): the 6-FMA
saving needs the DIT shared-twiddle pair form; DIF standalone rotations are
op- and latency-identical (29=29 FP ops, 8-cycle chain, P5 unchanged), and
dual-select accuracy gains are FP16-only. Session-scratchpad evidence
(perf attribution, hot tables, fused2 asm/mca, factor sweep) preserved
under docs/bridge-evidence/.

## 25998d5 — 2026-07-04

**fix(bench): cap O(N^2) reference gate at 64K, round-trip gate above**

The naive reference DFT gate made --compare grind for hours at N>=262144
(~10^12 ops, computed per size). Above kNaiveRefMaxN=65536 the accuracy
gate is now a forward+inverse round-trip against the input. Also hoists
constexpr N into lambda-local n in the route-ab factories (make_shared
forwarding odr-used the enclosing constexpr).

## 997a4a1 — 2026-07-04

**perf(dif): fused 2/3-pass L1 tiles, f32 four-step route, folded inverse scale**

- dif_pass_fused2<IP1,IP2> gates for (4,4)/(4,8)/(8,4)/(8,8) and fused3
  (4,4,4) in iterative_dif_execute_ws; measured policy: fused3 only f64
  N<=16384, fused2 pairs otherwise (fuse-knob A/B, ESTIMATE anchor)
- four_step_batched f32 routes 32768=(128,256), 65536=(256,256) with
  plan-owned heap scratch above FSB_MAX_N (route-ab: fwd 1.15/1.06x)
- Scale template param folds 1/N into dif_pass_last / dif_col_pass_last
  stores; removes the separate inverse-scale sweep on dif routes
- ctest 114/114

## 271fcc5 — 2026-07-02

**perf(codelet): batched M=4 cofactor route + spill-aware combine-lambda inline (small-N band)**

Two scoped band-A fixes, bisected + gated on interleaved --fftw-ab
(deterministic ESTIMATE anchor, 15 rounds, 2 alternated pairs):

- cofactor_simd_profitable: accept power-of-two M == 4 when the cofactor
  batch is full (Wc == R). kernel<16>'s recursion was the audit's 'fully
  scalar' defect (72 scalar FP, 0 vector); routing through
  kernel_batched<4> vectorizes it. N=16 f64 fwd 1.15->0.90, f32 1.42->0.91,
  f32 32 fwd 0.90->0.73 vs FFTW -- all now beat FFTW-ESTIMATE. M == 8
  measured a LOSS (f32 64 +8-10%: 16 live YMM = whole AVX2 file) -> excluded.

- radix_butterfly_v: force-inline the bfly_chunk lambda only while the
  combine's live set ~fits the file (2R+10 <= regs+2, i.e. R<=4 on AVX2).
  Kills the call + push rbp in the hot loop: f64 64 -17%, f32 48 -13/-16%.
  Unconditional inline regressed f64 32 +7-9% (radix-8 combine spill merge).

NOT shipped (measured losses, bisected): explicit zip-ladder AoS<->SoA
swizzle in codelet_apply (f32 16 +76%, f64 32 +24% -- the 'boundary
shuffle' pile was already the compiler's auto-vectorization; P1c NO-GO).

ctest 114/114.

## b54e533 — 2026-07-02

**docs: beat-FFTW gap audit 2026-07 + fixed-march arch-level audit**


## 6f3086f — 2026-07-02

**fix(vecpass): gate vpass_forward instantiation to W in {4,8}**

The cross-lane combine only exists for W=4/8 (static_assert); guard with
if-constexpr so narrow-lane builds (f64 W=2 on SSE) compile. The route is
never selected there (vecpass_supported<double> false on SSE).

## 0baa011 — 2026-07-02

**feat(bench): interleaved --fftw-ab baseline + FFTW_MEASURE env switch**

- --fftw-ab: order-alternating fft<->FFTW A/B (like --factors-ab) so the
  TSC frequency-warmup order bias cancels; median ratio + spread over rounds.
- FFT_BENCH_FFTW_MEASURE=1 flips FFTW planning ESTIMATE->MEASURE (its tuned
  ceiling) without recompiling.
- FFT_USE_NATIVE_ARCH now opt-in for the benchmark target (fixed-march A/B
  builds from the arch audit).

## 821f652 — 2026-07-02

**feat(threads): opt-in plan-owned multithreading + threaded benchmarks**

Add a cached fork-join worker pool to the N-D engine and fair multi-thread
benchmarks vs ducc0/FFTW. Default nthreads=1 is the byte-identical serial path
(zero threads spawned) so the tuned single-thread engine is untouched.

- detail/thread_pool.hpp: hand-rolled fork-join pool. nthreads-1 std::jthread
  workers park on a condition_variable; parallel_for splits [0,n) into nthreads
  contiguous chunks, runs the last on the caller, joins via std::latch,
  first-exception-wins. body(begin,end,tid) (ChunkBody concept) so callers
  allocate per-chunk scratch once. Free parallel_for(pool*,...) overload runs
  serial-inline when pool==null or below a size gate.

- nd_plan.hpp: thread thread_pool* through nd_apply_axis/execute; parallelize
  the row pass, the batched column-DIF pass (flattened over slab x column-tile),
  and the scalar-fallback column pass, each with per-chunk scratch.

- real_fft.hpp: per-chunk tile scratch for the r2c/c2r batched tile loop when
  threaded (serial keeps the cached plan-owned member).

- fft.hpp: plan<T>/plan_r2c<T> gain an nthreads ctor param and own a
  unique_ptr<thread_pool> (not optional: workers capture the pool's `this`, and
  the plan must stay movable). Single large 1-D transform stays serial.

- benchmarks: --nthreads=N on --compare/--compare-nd/--compare-2d; ducc0 + FFTW
  threaded; metric forced to wall-clock when nthreads>1. option(FFT_BENCH_THREADS)
  links Threads into ducc0 and the fftw3_threads libs (found by name -- no .pc).
  Threads::Threads on the fft INTERFACE target (jthread in header).

- test_fft_threads.cpp: nthreads=1 vs 4 output is bit-identical across
  1D/2D/3D/r2c. 114/114 ctest; Release/NDEBUG/-Werror clean; f32 3.2-4.7x
  speedup on 6 cores (1024^2, 256^3).

## d588c1a — 2026-07-02

**docs(nd): Phase-4 — small-inner ducc0 gap NO-GO (chiplet/B-major/AoS all AVX-512-only)**

Four asm-grounded opus spikes closed the small-inner column-pass gap vs ducc0
(256x16/1024x64/60^2 f32). All NO-GO on AVX2:

- radix-8 spill wall is LAYOUT-INDEPENDENT: 8 arms x 2 (re/im) = 16 YMM = the
  whole AVX2 register file before twiddles. vpass_one<8,float> = 19 spills.
- fused-radix-8 chiplet: our radix-4 col pass is already 0-spill; loss is
  pass-count/dispatch (23% FP util), not spills.
- depth-first gather driver: extra gather/scatter sweeps -> 1.22->1.38 WORSE.
- B-major via vecpass: vpass_forward mixes lanes (wrong math) + still 19 spills.
- AoS-native: same geometry (B=8 arm=2 YMM); B=4 halves parallelism + adds
  vpermilps/twiddle on the dispatch-bound kernel.

FMA/arithmetic confirmed already-optimal everywhere. col_dif_execute_ws is the
AVX2 Pareto frontier (already B-major, 0-spill radix-4, folded AoS<->SoA). The
residual is AVX-512-only (32 ZMM -> radix-8 0-spill). Accepted as AVX2 frontier.

Also: gitignore scratchpad/ (session temp).

## 93a21b5 — 2026-07-01

**docs(nd): Phase-3 Sprint 3 — FFTW cube gap attribution + NO-GO**

64^3 (1.39x): PRIOR CONFIRMED — gap is 100% the 1D N=64 row codelet (col
passes beat FFTW 0.95x); genfft/AVX-512 frontier, no AVX2 lever.
8^4 (1.93x): PRIOR REFUTED — 3 N-D mechanisms (scalar ido=1 row serialized
512x; col2 inner=W slab fragmentation; col-driver AoS/SoA structural). A
transpose-batch lever tops out ~1.30x (still loses FFTW, niche shapes) →
deferred NO-GO. Robust A/B, perf+asm attributed.

## 9022409 — 2026-07-01

**perf(nd): Phase-3 residual close — radix-4 small-inner lever + r2c/c2r transpose pack**

Sprint 1 (c2c small-inner column, f32 pow2):
- make_nd_axis_state takes per-axis `inner`; force a spill-free radix-4-only
  factorization for narrow-inner pow2 f32 outer axes (radix-8 B-vec spills ~17
  YMM on AVX2). 256x16 1.24->1.21, 1024x64 1.34->1.28, 8^4 0.91->0.69; no
  regression; f64 untouched. <=1.00 parity needs a B-major fused multipass
  rewrite (out of scope) — documented residual. The ido-vectorized driver was
  built and measured a NO-GO (256x16 -> 1.54: pow2 ido shrinks per pass so only
  the first pass vectorizes) — reverted and deleted.

Sprint 2 (r2c/c2r forward, real_fft.hpp only, NO new TU):
- Replace the scalar-lane pack/unpack gather/scatter in r2c_even_tile /
  c2r_even_tile with in-register WxW xsimd::transpose tiles (H=W/2 j/k per
  transpose; scalar <H tail + k=M Nyquist bin). Fold the per-k `k%M` scalar div
  (16k/call) into a branchless block path. c2r input reads the reversed in[M-k]
  block. Inner compile-time loops use poet::static_for.
- Attribution (perf+asm): the f32-only gap was ~59% the pack/unpack transpose,
  not the multipass/column-axis; recombine is already a fused O(M) L1 post-pass
  so a per-M recombination codelet TU is a NO-GO.
- r2c fwd f32 256^2 1.24->0.91 / 512^2 1.02 / 1024^2 0.99 / 64^3 0.79; rt f32
  all <=1.01; f64 fwd <=1.03, rt wins. Beat-or-parity vs ducc0 both prec, fwd+rt;
  FFTW 0.59-0.90. asm float r2c_even_tile: 0 spills, 0 div.

All gated on the robust cycle-true A/B (identity control ~1.000). ctest 110/110.
docs/nd-perf-frontier.md updated (Phase-3 Sprints 1+2, kept/pruned/NO-GO).

## 6dca9a6 — 2026-07-01

**feat(nd): batched real-FFT + trustworthy robust A/B (N-D Phase 1+2)**

N-D FFT feature: row-column engine (nd_plan), r2c/c2r via half-length trick,
column-pass cache blocking, and the Phase-2 batched real-FFT.

WS-A: --compare-nd [--r2c] --robust engine A/B (role-swap + sqrt(mAB/mBA)
bias cancel, cycle-invariant, mandatory identity control ~1.000). Retires the
false sequential-nb_measure f32 r2c artifact.

WS-B: real_fft_plan even path rebuilt as a row-batched, ISA-parametric
transform — pack W real rows into SIMD lanes, inner size-M c2c once per W-row
tile via vp::multipass_run, recombination V-wide across rows. Scratch is one
plan-owned ping-pong block (per-tile malloc was the killer). r2c ours/ducc0:
fwd 2-3.7x -> 0.97-1.24, c2r beats ducc0 broadly (0.80-1.04), beats FFTW on
pow2 squares. 108/108 ctest + W-tail cases.

WS-C: c2c small-inner residual attributed (strided column-pass SoA repack;
16x256 wins 0.70 vs 256x16 loses 1.22) -> NO-GO (transpose lever marginal).

Docs: docs/nd-perf-frontier.md Phase 2 section (methodology, before/after,
kept/pruned, do-not-retry).

## 162d144 — 2026-06-30

**perf(twiddles): f64 odd-radix reorder rule subsumes override exceptions**

Post-DP reorder_odd_radices_to_extremes<T>: the DP's additive per-pass cost
sorts small radices first, clustering >=2 expensive odd radices (5/7/11) at the
small-ido scalar valley (1<ido<W) where they spill. Fix = move them to the
extremes (largest ido first, ido=1 last; 3s/pow2 middle). Scoped to f64,
no-radix-8, >=2 expensive odds (322-size role-swapped cyc A/B: 33 win / 42 tie /
3 marginal lose <=1.04). f32 wants the opposite (W=8 economics), radix-8 and
single-odd are DP-optimal -> scoped out. Converts ~33 f64 composites from
DP-losers to wins with no per-size entries (many now beat ducc0: 2058 0.66,
1575 0.80, 945 0.92, 4410 0.75, 2352 0.94).

Residual measured entries that beat the generic rule (second-order placement):
f64 2700/7056, f32 5292/7560. 3780 entry deleted (rule reproduces it verbatim).
ctest 96/96, accuracy clean, f64 byte-identical outside the firing class.

## c54f39e — 2026-06-30

**perf(twiddles): f32 pow2 ido-ordering overrides (2048, 4096); revert op-count rebuild**

The closed-form op-count rebuild of dif_stage_cost was reverted: its
"byte-identical, ctest 96/96" verification used a piped `diff | grep` that
silently returned 0, masking that it shredded 195/310 f64 and 149/310 f32 plans
into radix-2/3 chains (op-count DP minimizes arithmetic => maximizes passes, but
FFT cost is per-pass memory traffic => minimizes passes). Restored the known-good
fitted model + override table.

Measured (role-swapped cycle-true A/B, rounds=15x2, pinned core 0) that the
per-pass-memory / regime-dispatch hypothesis is refuted: a clean per-pass model
provably cannot reproduce the pow2 ladder (512 vs 1024 give contradictory
constraints); "maximize radix-8" regresses 1024/256/128 up to 2.2x; the real pow2
wins are pass-ORDER choices (2048 8-8-4-8 is 2x faster than the same multiset
8-8-8-4 in f32) that no per-stage cost can rank -> they belong in the residual
table, not a separate decomposer.

Shipped the only un-captured measured money: 2 f32 pow2 override entries
- 2048 -> 8-8-4-8 (+4.6% vs DP 4-4-4-4-8)
- 4096 -> 8-8-8-8 (+8% vs DP 4-4-4-8-8)
f64 512/4096 measured a TIE (omitted per the maintenance rule); f32 512's 8-8-8
win is real but production routes 512 f32 to four_step_batched, so an override
there would never fire.

Tooling: --cost-audit diagnostic (closed-form DP pick + optional A/B vs a
candidate ordering). Docs: corrected decomposition-planner.md (the false
byte-identical claim + the count-vs-order verdict). ctest 96/96.

## e76c6df — 2026-06-30

**fix(bench): trustworthy in-process A/B + honor dif_override route; audit override table**

Benchmark baseline was untrustworthy for factor-ordering A/B. Two bugs:

- plan_impl chose route via select_route(size) and ignored a passed
  dif_override, so --factors on any non-iterative_dif size silently measured
  the DEFAULT engine while labelling output with the requested factorization.
  An explicit override now forces iterative_dif. (This false baseline is what
  made the 448-f32 override read as a 2-3% win when it is a 57% regression and
  dead code — 448 f32 routes to four_step_batched.)

- Cross-process --factors comparison + per-object heap-layout bias (~2-3%,
  follows the allocated-first plan object; not frequency — cpucycles is
  frequency-invariant). New --factors-ab runs both plans in ONE process,
  role-swapped (sqrt(M_ab/M_ba) cancels the layout advantage), geomean over
  rounds, with an identity control (A==B must read ~1.000) as the health gate.

Audited every measured_dif_factor_plan override with the de-biased tool
(rounds>=15). The overrides are re-orderings the additive per-stage DP cannot
express; an offline sweep (dp_tune) of a smooth lanes/ido odd-radix term never
reproduces them and rewrites ~70% of factorizations, and the optimum is
cross-stage (5040 wants radix-7 last vs 1260/2520 radix-5 first). Kept the 6
measured wins (1260/1500/5040/2048/48 f64, 2520 f32), each documented with its
A/B number and the re-audit command; deleted 4 unjustified entries (2520-f64,
336, 6480 ties; 448-f32 dead/harmful). ctest 96/96, no ducc0 regressions.

## a2ee792 — 2026-06-30

**perf(twiddles): f32 ido-aware DIF scheduling — ties/beats FFTW on ~25 composites**

Scope the dif_stage_cost last-pass / small-ido surcharge to the odd radices
(5/7/11) for f32 only. Asm census (docs/beat-fftw-plan.md §2b/§7) shows only odd
radices spill in the ido-SoA shuffle dance, while pow2 radix-8 at ido<=lanes is
its cheapest placement (no twiddles, unit stride, W=8 lanes full). The DP
factorizer now places radix-8 innermost on f32.

Measured (cyc-true, pinned taskset -c 0, 7x paired vs the in-bench FFTW gate;
ctest 96/96):
- 3136 f32 roundtrip 0.945 — beats FFTW; forward 2.24 -> 1.00 (parity)
- 5040 1.68 -> 1.27, 2352 -> 1.07, 3920 -> 1.06, 5488 -> ~1.0; ~25 f32 composite
  sizes improved to near-parity. vs ducc0 several flip to wins (5040 1.065->0.798).

f64 (W=4) ido economics differ and regress with radix-8-last (112 +11%, 1792 +5%),
so f64 keeps the blanket radix>4 surcharge — factorizations byte-identical to
before (zero f64 risk). One f32 counter-override (448 -> {4,4,4,7}): the lone f32
size where radix-8-last loses (~3%).

Docs: beat-fftw-plan.md adds the Phase-0 asm grounding (chiplet-viability audit:
radix<=8 spill-free on AVX-512, radix-16/32 still spill -> chiplet path is AVX-512
+ genfft-scheduling gated) plus the Phase-1 A/B tables; pow2-fftw-codelet-frontier.md
the prior pow2 chiplet frontier analysis.

## e513701 — 2026-06-29

**docs: pow2 FFTW-gap Phase 0 attribution + GO decision**


## b76c623 — 2026-06-29

**feat(bench): optional in-bench FFTW reference (-DFFT_BENCH_FFTW)**

Add an FFTW path to fft_benchmark alongside ducc0: a reusable fftw_c2c<T>
holder building FFTW_MEASURE plans once and reusing them across timed reps
(fair vs ducc0's plan cache / our plan-reuse), wired into compare_min_of_n
with the existing accuracy gate and cyc-based ratio. Gated behind a new
CMake option FFT_BENCH_FFTW (pkg-config fftw3/fftw3f -> PkgConfig::FFTW).

Gives a fresh same-machine us-vs-FFTW A/B, replacing external/remembered
numbers, and is the reference gate for the pow2 fused-codelet campaign.

## df8e05d — 2026-06-29

**perf(codelet): flat register-resident pow2 leaf + poet index cleanup**


## ddf9c4e — 2026-06-29

**docs: add cache-blocking / decomposition-planner / pow2-compiler frontier notes**


## 9bc5fcc — 2026-06-29

**chore: pin xsimd to master**


## 5cc863d — 2026-06-26

**perf(route): ido-ordering overrides — f32 2520 beats FFTW; f64 {1260,1500,2520,5040} narrow the gap**

Expensive odd radix (5/7) FIRST at the largest ido beats the DP's additive-cost
ordering: the cost model misses the ido-dependent odd-radix penalty (a radix-5
butterfly is far cheaper batched over ido=504 than ido=56). Found via a
factor-plan override search, verified with 2-plan interleaved cycle-true A/B
(candidate-vs-DP, immune to FFTW's plan variance + freq drift), accuracy-gated.

- 2520 {5,3,3,7,8} (both precisions): f32 fwd 0.97x BEATS FFTW single-thread
  MEASURE, robust 7/7 independent plans; f64 fwd parity. CA/DP f64 .988/.976,
  f32 .948.
- f64-ONLY (the W=8 f32 ido economics differ — these orderings regress f32
  36-60%, so f32 stays on the DP): 1260 {5,3,3,7,4} CA/DP .955/.976;
  1500 {5,5,3,5,4} .86/.87 (~14%); 5040 {5,4,4,3,3,7} .94/.94. Narrow the FFTW
  gap and widen the ducc0 margin (don't beat FFTW on f64).

7560 DP {3,3,3,5,7,8} already optimal — no entry (its sequential-scan "win" was
measurement drift; interleaved A/B showed all candidates ~= DP). ctest 96/96.

Also: FFT_LAMBDA_ALWAYS_INLINE portability macro (token-identical to
__attribute__((always_inline)) -> byte-identical codegen) + dead <bit> include.

## 4139acb — 2026-06-26

**perf(simd): plain complex-mul over xsimd::fnma (fma3 miss on avxvnni)**

-march=native selects the avxvnni arch on Meteor/Alder Lake, for which xsimd
registers no FMA3 kernel (only fma3<avx2> does). xsimd::fnma/fnms therefore
fall back to the generic neg(vxorpd)+vfmadd form, paying a wasted sign-flip on
the twiddle-mul critical path -- the hottest single instruction in dif_pass<4>
(6.06%+3.40% of 4096 f64 cycles per addr2line).

Replace the 12 SoA complex-mul sites (dif_pass x3, dif_col_pass x2, vecpass x2,
four_step x1, codelet x4) with plain (owr*sr - owi*si): under -ffast-math the
compiler contracts this to a single vfnmadd231pd. Bit-identical rounding, less
code, matches what the scalar tails already do. xsimd::fma (x*y+z) is left
untouched -- it already lowers to one vfmadd.

Codegen: dif_pass<4> vxorpd 8->2, <8> 20->6; symbols smaller. Paired-cyc A/B
(taskset -c 0): 2048 f64 0.970, 1260 0.985; 4096/8192/5040 neutral (40%
memory-bound, not core). ctest 96/96, verify all sizes PASS.

## d456341 — 2026-06-24

**perf(route): precision-split codelet→iterative_dif for {9,15,21,25,33,35,49,50,55,63} f64 + {10,14} f32**

These catalog sizes beat their straight-line codelet on one precision only.
Interleaved cycle-true A/B (min-of-5 rounds x reps=15, pinned core 0, ducc0
common reference): every entry negative on BOTH fwd and rt, clear of the
small-N noise floor.

  f64 -> dif: 9(-42/-26) 15(-21/-22) 21(-26/-20) 25(-15/-9) 33(-19/-12)
              35(-28/-25) 49(-29/-19) 50(-19/-13) 55(-33/-18) 63(-15/-13)
  f32 -> dif: 10(-16/-7) 14(-18/-15)

f64's wider 8-element radix passes spill the big codelet harder than f32's,
so most wins are f64-side. All stay well under ducc0. Extends ff2b09c
(both-precision set). Full --verify PASS, codelet test 73800 assertions pass.

## ff2b09c — 2026-06-24

**perf(route): send {22,24,40,42,48,56,60} to iterative_dif over codelet**

Measured (pinned cycle-true A/B, min-of-15, both precisions) that the
straight-line catalog codelet is a pessimization for these highly-composite
sizes: the large unrolled kernel spills, while the iterative_dif radix chain
batches the radix passes over the whole transform. Routing them to dif is
15-48% faster on BOTH f32 and f64 and still well under ducc0:

  f64  N=24 0.70->0.42  N=40 0.71->0.45  N=48 0.84->0.53
       N=56 0.77->0.50  N=60 0.85->0.57  (ratio vs ducc0, lower=faster)
  f32  N=24 0.77->0.42  N=40 0.63->0.45  N=48 0.86->0.55  N=56 0.74->0.45

Generalizes the prior hand-found f32 N=60 special-case (now precision-agnostic;
f64 N=60 wins by the same margin). Full --verify passes; control catalog sizes
(12,18,36,54,64,128) keep the codelet route and are unchanged.

## ddb43d5 — 2026-06-24

**refactor: single-source dif_radix_set + T-derived scratch padding**

- dif_radix_set defined once in twiddles.hpp; dif_driver/vecpass derive
  from it (vecpass switch -> poet::dispatch over the shared sequence)
- gate f32-only routes (four_step_batched, vecpass) under
  if constexpr(sizeof(T)==4) so f64 strips them at compile time
- span_stride: derive cache-critical modulus from sizeof(T) (256 f64 /
  512 f32) instead of hardcoded 256; pin frame budget + bounds via static_assert

96/96 relwithdebinfo (strict -Werror -DNDEBUG), measured-neutral.

## 01b9190 — 2026-06-24

**perf(f32): vecpass W=8 route + no-init scratch heap**

Two validated wins (paired-interleaved cycle-true A/B, taskset -c 2):

- soa_scratch heap: std::vector<T> -> make_unique_for_overwrite<T[]>,
  dropping the per-execute value-init memset (FFT overwrites scratch
  before reading). Codegen audit: heap path no longer calls memset.
  Broad both-precision win for N>4096; f64 8192 1.08->0.96 (flips to a
  win), 5040 1.10->0.94 vs ducc0. N<=4096 stack path byte-identical.

- vecpass route: lane-batched W=8 f32 transform (new vecpass.hpp,
  Forward-templated for inverse). Explicit measured allowlist
  {4032,15120,20160} where the 2x SIMD-width edge over ducc0's
  hardcoded vlen=4 beats both ducc0 and iterative_dif fwd+rt
  (0.87-0.91). Per-execute working scratch is a single over-aligned
  V[] block (6 scattered mallocs regressed 15120; T[] reinterpret
  faults on aligned V stores). Routes before iterative_dif.

bench: --vpass probe now includes the tracked header (was the
gitignored report/ copy). 96/96 ctest, --verify --prec=both PASS,
inverse roundtrip PASS.

## 975e9d7 — 2026-06-23

**docs(dif): drop stale dif_pass_fused references (removed in 7e6fdf0)**


## 70cb0b7 — 2026-06-23

**perf(dif): masked-transpose load for IP<W last pass (f32 trio)**

Extend the dif_pass_last section-vec load to IP<W: masked-load W b-rows
of IP arms (lane<IP compile-time mask -> 256-bit vmaskmovps, no over-read
past the row/span, IP<W can't form an in-row overlap tile) + one WxW
transpose. Gated 2*IP>W so the low-half-mask case (f64 IP=2, f32 IP<=4)
stays on the scalar gather (xsimd routes a low-half mask to a slow scalar
stack-buffer load). Kills the f32 IP=5/7 last-pass scalar gather.

Also fix the kept scalar-gather buffer alignment to the xsimd contract
batch::arch_type::alignment() (the value load_aligned asserts) instead of
alignof(batch_t).

asm (f32 dif_pass_last<7>): 16 vmaskmovps + WxW transpose, gather pile
gone, symbol 0xa6a->0x8d5. Paired cyc A/B vs Phase 1: f32 1260 1.058->
1.023, 1500 1.087->1.046, 5040 1.179->1.167, 8192 1.193->1.168; f64
neutral (last passes IP>=W untouched), no guard regression. 96/96 ctest
strict -Werror -DNDEBUG.

## a3872c9 — 2026-06-23

**perf(dif): section-vectorize ido==1 last pass via transpose-load**

Replace the strided scalar gather in dif_pass_last (the p5-bound
vmovhpd/vinsertf128/vperm2f128 pile, ~56 instrs) with a contiguous load
+ in-register xsimd::transpose for IP>=W. Overlapping in-row WxW tiles
(offsets 0,W,...,IP-W) never read past arm IP-1 of a row, so no over-read
past the planar span (scratch only pads when N%256==0). IP<W keeps the
small scalar gather.

asm (alderlake f64 dif_pass_last<7>): vmovhpd 29->0, vmovsd 45->21,
symbol 0x879->0x776. Paired cyc A/B vs ducc0: 1260 f64 1.100->1.077,
5040 1.337->1.314, all f64 pow2 improved (4096 .879->.831,
8192 1.433->1.415), no regression. 96/96 ctest strict -Werror -DNDEBUG.

## 30cfd7c — 2026-06-23

**fix: mark ido [[maybe_unused]] in assert-only last/fused DIF passes**

The tier-1 cleanup replaced 'if (ido==1)' with 'assert(ido==1)' in
dif_pass_last / dif_col_pass_last. assert vanishes under -DNDEBUG, so ido
became an unused parameter -> -Werror=unused-parameter in the strict
release build (warnings-as-errors). Debug compiled (assert live).

## a84a453 — 2026-06-23

**fix(scratch): restore user-provided soa_scratch M() to suppress per-execute memset**

The Alloc-removal refactor changed M's ctor to '= default' + value-init
(m{}), which zero-initializes the entire 128 KB SBO stack_buf on every
execute() — 48% of 512 f64 runtime in __memset. A user-provided M() {}
suppresses the zero-init (master behavior). Restores parity:
512 f64 fwd 2.55 -> 0.95, 720 f64 1.97 -> 1.04 vs ducc0.

## d3be664 — 2026-06-23

**refactor: tier 2 API/ABI cleanup + restore nd zero-size guard**

Unified plan<T> N-D rework, c_api slim, header trims, drop dead
nd_dispatch.hpp. Restores the size>0 trust-boundary guard in
nd_runtime_plan that the 1D->N-D unification dropped (test_fft_plan
'Zero size plan'). gitignore generated graphify-out/ artifacts.

## 7e6fdf0 — 2026-06-23

**refactor: tier 1 dead-code cleanup — fused pass, stdlib wrappers, invoke structs, switch**

Items completed:
- Delete dif_pass_fused/dif_pass_fused_invoke; single-pass n_passes==1 in
  iterative_dif_execute_ws now uses dif_pass_first + plain re-interleave
- Delete iterative_dif_execute stateless wrapper; update test_fft_iterative
  to call iterative_dif_execute_ws directly with local scratch/twiddles
- Extract shared rader_gpow_table<P>/rader_ginvpow_table<P> consteval helpers;
  rader_apply and rader_apply_batched both use them (no duplicated lambdas)
- Item 4 SKIPPED: poet::dispatch requires template<int IP> operator(), not a
  generic lambda (auto IP_c form); the invoke structs are load-bearing
- plan.hpp: extract apply_inverse_scale() helper; all 5 per-route inverse
  loops call it instead of repeating the for loop
- Replace next_power_of_2 with std::bit_ceil, is_power_of_2 with
  std::has_single_bit at all call sites; delete wrappers from math.hpp;
  update test_fft.cpp to test std functions directly
- dif_pass_last: replace if(ido==1) guard with assert, de-nest body;
  dif_col_pass_fused: remove dead ido>1 runtime branches (invariant: ido==1)
- plan_impl ctor and execute(): convert if/else-if chains to switch on route_kind

## f6417f0 — 2026-06-23

**perf(dif): overlapping final full-width block replaces fat scalar tails**

The a-vectorized DIF pass (dif_pass, dif_pass_first) ran a scalar tail for the
ido%W leftover columns. ducc0 keeps such passes full-width via group batching,
so the tail was a measured chunk of the gap on the f64 smooth-composite losers.

Replace the scalar tail with one overlapping full-width butterfly at column
ido-W, gated to tail >= W/2. Output column a depends only on input column a, so
the overlap with the previous block recomputes bit-identical values: no mask, no
over-read past the ido-run. The W/2 gate is load-bearing -- a thin tail (size 1,
e.g. f64 ido=5/25/125 of N=1500) would replace one cheap scalar element with a
near-fully-redundant W-wide butterfly and regress; only a fat tail (f64 ido=7
radix-5 / ido=35 radix-4 of N=1260/5040, tail 3 of W=4) wins.

The U==1 hot loop is now driven through a single always_inline do_batch lambda
shared with the overlap; the attribute is mandatory (a shared out-of-line copy
regressed every size 30-40%).

Measured (155H P-core, taskset, cyc, 11-round paired A/B): 1260 f64 1.139->1.072
(-5.9% fwd, -7.0% rt), 5040 f64 1.363->1.313 (-3.7%), 1500 f32 -2.9%, 1500 f64
neutral. Accuracy unchanged (L2 vs reference, all sizes PASS). text +0.4%.

## f28386a — 2026-06-22

**refactor(dif): data-driven small-ido dispatch (predicate + allow-list + fold)**

Replace the hardcoded `if constexpr (W==8 && IP==5) { switch(ido) }` gate in
dif_pass with three findable, extensible pieces:
  - small_ido_eligible<T,IP>()  consteval eligibility predicate
  - small_ido_set<IP>           integer_sequence allow-list (the ONLY place
                                controlling which (IP,IDO) get instantiated)
  - small_ido_try<...>          fold over the pack with runtime fall-through
                                (ido 2/3/4/6 fall through to the generic loop;
                                not poet::dispatch, which throws on no-match,
                                nor static_for, which would instantiate 2..7)

dif_pass_small_ido body is untouched. Adding a future radix is one extra
small_ido_set specialization; an empty set dead-strips the fold. No new
template instantiations.

Behavior-preserving (asm-verified): inlined small-ido SIMD body byte-equivalent
(identical vmaskmovps/vunpck/vperm2f128/FMA counts), dispatch shrank 12 insns,
text -192B, small_ido_try fully inlined (no standalone symbol). Accuracy PASS
both precisions; paired cyc A/B (taskset -c 0, 21 rounds) shows targets
1260/1500/5040 at parity-or-faster and guards 720/2520/7560 within +-1%.

## aca3eda — 2026-06-22

**perf(dif): transpose lane-over-b for f32 small-ido radix-5 pass**

The odd-radix smooth-composite losers (N=1260/1500/5040) are dominated by
dif_pass<5> running at ido<W: the factorizer emits r5 at ido=7/5, and at
ido<W=8 the a-vectorized loop fills zero SIMD lanes so the radix-5 butterfly
runs fully scalar (30-40% of runtime).

Add dif_pass_small_ido: vectorize over the group dim b instead of a. Pack W
consecutive b into lanes (output twiddle depends only on a -> broadcast);
masked-load the contiguous ido-run -> xsimd::transpose -> full-width radix-5
butterfly -> transpose back -> masked-store. Compile-time IDO mask keeps every
access exactly within the ido-run (no over-read, no scratch-layout change, no
staging buffer). The l1%W tail is folded into the transpose path, not scalar.

Scalar gather/scatter was tried first and abandoned (net-negative: only converts
scalar->SIMD butterflies without cutting memory traffic; f64 +19/+41%, f32 5040
+7%). In-register transpose is the Phase-1-verified primitive (~3.9x faster f32).

Gated to f32 (W==8), IP==5, ido in {5,7} -- the only (radix,ido) the factorizer
emits for these sizes. Instantiating unused odd radices/idos bloats the TU and
perturbs other sizes' code layout (~3-5% on 7560/512 before trimming).

Measured (155H, taskset -c 0, cyc, 13-round paired A/B vs master):
  1260 fwd 0.778 (-22%, ducc0 1.32->1.04)
  1500 fwd 0.786 (-21%, ducc0 1.42->1.10)
  5040 fwd 0.814 (-19%, ducc0 1.31->1.17)
No regression elsewhere; --verify PASS both precisions. asm-audited: transpose +
vmaskmovps + full-width vfmadd231ps confirmed. Still loses ducc0 (residual =
staging spills + ducc0's hand-tuned vecpass); f64 untouched (ido>W, a-loop
already vectorizes).

## b21ef69 — 2026-06-22

**feat(plan): allocator template param + f32 batched four-step route**

Allocator-aware plans: add Alloc template parameter (default std::allocator)
to plan_impl / soa_scratch / forward_plan / inverse_plan. The default path is
behaviour-identical to before (A/B parity, both precisions); a caller that wants
to avoid the per-execute large-N malloc supplies a pooling/arena allocator
explicitly. soa_scratch's M gets an explicit ctor so stack_buf stays
uninitialized (no per-alloc memset).

f32 batched four-step route (from the decomposition campaign): four_step_batched
wired into select_route for the measured f32 win class {128,256,384,448,512,640,
768} via a measured split table; 448 (8x56) is the standout (+34% vs ducc0).
Harness: bench_fft.cpp catalog-factor sweep uses the cycle metric.

Scratch verdict (docs/decomposition-frontier.md): plan-owned reused buffer gives
+13% forward on 5040/7560/8192 but regresses the roundtrip (two resident buffers
double the rt working set -> L2 blowout); thread_local fixes it but is rejected
as hidden global state; unaligning the <=4096 path regresses 16-23% (alignas(64)
on SIMD scratch is load-bearing). Net: keep per-execute scratch, expose allocator.

Verified: --verify --prec=both PASS across the campaign size set; custom-allocator
self-check (report/probe_alloc.cpp) shows the allocator is routed through with a
correct round-trip (L2=5e-16).

## 63b5284 — 2026-06-22

**perf(codelet): f32 Rader→scalar, precision-aware N=54 radix, N=32 radix-8 leaf**

Codelet optimization campaign (sizes 2-64 vs ducc0), measurement-gated with
drift-free per-size-interleaved paired --compare. Catalog now has 0/63 losses
on both precisions (fwd+rt). Full ledger + per-size scoreboard with throughput
in docs/codelet-optimization-frontier.md.

WS1: f32 r=2 Rader-prime cofactor now takes the low-half-mask guard in
cofactor_simd_profitable, routing to scalar rader_apply<M> (full-ymm inner conv)
instead of the 2-idle-lane SSE batch. +12..44% on N=26,34,46,58,62 and nested
47,59; N=38 (M=19) -8.4% accepted (no clean structural predictor, still beats ducc0).

WS2: precision-aware codelet_radix_for<T> at the kernel<N,T> leaf. N=54 f32 peels
r=6 (cofactor-SIMD, 6/8 lanes): 1.11x LOSE -> 0.52x WIN, the catalog's last loss.
f64 keeps r=2 (native batch 4 < 6) and is unchanged.

WS4: codelet_radix(32)=8. Radix-8 leaf is +41%/+35% (f32/f64) despite ~2x spills
(peak_live 26>16); the better arithmetic structure wins on this wide OoO core.
N=64 (M=8) stays r=4 (register-starved, -25%).

Do-not-retry, documented (codelet.hpp WS3, butterfly.hpp WS5): Rader buffer-alias/
packed-gather is perf-neutral (buffers already L1-resident); small scalar leaves
(5,6,7,8,11) are latency-bound and only batching across transforms helps.

## 3d8d9fc — 2026-06-20

**perf(codelet): cofactor-SIMD scatter transpose + generalize to odd-composite M**

Two shipped wins on the terminal kernel<N> cofactor-SIMD path (N=r*M, r<=W),
verified by paired pinned --compare (taskset -c 0, CPU-cycle metric) vs master.

B3 — cofactor scatter via xsimd::transpose:
  Replace the N scalar output stores (yre[q*M+k]=...) with a Wc x Wc in-register
  transpose (vunpck/vperm, not vmaskmov) per tile + per-column store. N=52 f64
  -11.4% fwd; cofactor group (26..62) geomean -2.1%; 0 LOSE.

B5 — generalize cofactor-SIMD beyond Rader-prime M to odd composite / small odd
  prime M (kernel<N> now calls the generic kernel_batched<M>; for Rader-prime M it
  forwards to rader_apply_batched, so 26..62 codegen is unchanged). Gated by a
  measured, mechanism-correct predicate cofactor_simd_profitable<T,R>(M):
    - is_rader_prime(M): unconditional (validated set)
    - else odd M, by batch width Wc=cofactor_batch_width<T,R>:
        full (Wc==R) -> yes; low-half mask 2R<=Wc (f32 R=2, xsimd scalar fallback)
        -> no; 128-bit hw-masked load -> yes; 256-bit masked -> R*M>=27.
  Newly-enabled odd composites win 20-52% (N=12,18,20,27,28,33,36,44,45,55,63...);
  affected-subset geomean -12.3% fwd vs master, full catalog -4.3%, 0 new LOSE.

Caveat: N=9/15 f64 regress ~25-35% — NOT algorithmic (their cofactor path is gated
off; scalar path taken) but ThinLTO global-inlining churn from the new kernel_batched
instantiations; both still beat ducc0. A noinline-boundary mitigation was tried and
rejected (didn't fix it, erased small wins). Rejected as net-negative on this uarch:
B1 (masked combine tail), B2 (Rader pointwise overlap tail), B4 (AoS swizzle boundary);
B6 not pursued (same tail-micro-opt class). Analysis in report/ (gitignored).

Correctness: --verify --prec=both 0 FAIL.

## 8237b83 — 2026-06-19

**feat: Rader large-prime route + unified recursive cost model; wire planner**

rader.hpp: isolated primes p>64 (above the codelet catalog) executed via Rader's
algorithm — a length-(p-1) cyclic convolution run by the existing kernel paths
(codelet for p-1<=64, iterative_dif for 11-smooth p-1, four-step for two-<=64-
factor p-1). This is a COMPOSITION OF CODELET KERNELS over the composite p-1, with
no chirp-z zero-padding; the runtime analogue of the compile-time rader_apply<P>.
estimated_plan_cost() is a unified recursive cost model spanning codelet /
iterative_dif / four-step / Rader / Bluestein; rader_beats_bluestein() gates the
route so primes with a dear p-1 transform (79/83/127/191/251/283) correctly keep
Bluestein. asm: run_inner is 0-spill and dispatches into the audited kernels.

plan.hpp: route_kind::four_step + route_kind::rader wired into select_route,
both cost-gated (no regression). ~32/38 primes in 65-300 now beat Bluestein
22-51%. All 96 ctests pass; --verify both precisions (f64 ~1e-15, f32 ~1e-7).

For the residual primes whose p-1 has a prime factor >64 (167/179/227/263/269/
293), recursive Rader was evaluated and rejected: the validated cost model shows
it 2-5x slower than Bluestein, and Bluestein is itself a kernel composition (its
pow2_fft dispatches to iterative_dif/codelet) — so Bluestein is the optimal route.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>

## e1e2fb9 — 2026-06-19

**feat: four-step route for non-smooth composites N>64 (cost-gated codelet composition)**

four_step.hpp: two-factor Cooley-Tukey (N = N1*N2, both factors <=64 catalog
leaves) executed as a COMPOSITION OF spill-free codelet kernels + a twiddle twist
between the two leaf passes, instead of Bluestein's chirp-z padding. Adds a
calibrated cost model (codelet_cost_cyc[] measured leaf table, four_step_cost,
choose_four_step_split now cost-minimal) and four_step_beats_bluestein() — the
gate reproduces all 8 measured win/loss outcomes (143/187/289/338 win; 169/209/
221/247 keep Bluestein). ~299 sizes <=1024 flip to the codelet composition.

test_fft_codelet.cpp: four-step vs reference DFT + reference-free roundtrip
coverage (16x16..32x64, both precisions).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>

## e15aad0 — 2026-06-19

**test(bench): L2 accuracy gate on every timed plan + ducc0 --compare harness**

bench_fft.cpp: --verify [--prec][--sizes][--tol] checks the L2 error vs a
reference DFT for every planned size; --compare reports the per-size cycle ratio
vs ducc0 behind the same accuracy gate, so an un-run / mis-planned transform can
no longer score a phantom win. Pinned-core cycle metric.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>

## 7278c14 — 2026-06-19

**feat: ISA-adaptive cofactor-SIMD batched codelets + number-theory helpers**

- codelet.hpp: kernel_batched / rader_apply_batched templated on the SIMD batch
  type V; cofactor_batch_width() picks the narrowest ISA register >= the cofactor
  count r (make_sized_batch_t / is_void), with a compile-time masked load when
  r<W. Vectorizes the prime/Rader sub-transforms across the r==W cofactor copies
  (flips the f32 N=52 / N=26 losses; all prime composites 26..62 beat ducc0).
- ct_math.hpp: ct_powmod / ct_primitive_root / ct_is_prime (constexpr, also
  usable at runtime) for the Rader routes.
- macros.hpp / undef_macros.hpp: FFT_NOINLINE leaf-boundary macro.
- Dependencies.cmake: pin xsimd 14.2.0 (compile-time mask + make_sized_batch).
- src/CMakeLists.txt: codelet-catalog build knobs.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>

## e0ddcef — 2026-06-18

**perf: route codelet combine through unified dif_butterfly**

Replace bfly_chunk's naive O(R^2) DFT matrix with a call to the unified
dif_butterfly<T,Forward,R,V> — the same kernel the iterative-DIF path uses:
odd R routes through the low-multiply symmetric kernel (radix_sym_dft), pow2
R through the split-radix kernel (pow2_dif_butterfly), the rest through the
naive matrix. Every catalog codelet now inherits the symmetric multiply-halving
and split-radix twiddle elision.

Rewrite radix_butterfly_v with a spill-aware unroll: U = dif_pass_unroll<R>()
(registers / peak-live) driving poet::dynamic_for<U> over full W-wide batches
plus a scalar tail. U=1 on AVX2 (an R-input combine's ~2R+10 live batches
already exceed 16 YMM); scales to U>1 on AVX-512. Also sweep the codebase to
pointer-arithmetic form (ptr + i, not &ptr[i]).

Result (cyc vs ducc0, relwithdebinfo, taskset -c 0, reps=15): eliminates the
only LOSE (N=22 1.045 -> 0.475) and gives +25-75% on odd/prime sizes
7/11/13/14/17/21/28/30 and N=4 (split-radix). 37/37 tests pass.

Phase A2 (per-N measured factorization) was investigated and rejected: the
iterative factor-sweep proxy does not predict codelet-path cycles, and small
composite codelets carry ~+-10% bench noise. See report/PHASE_CODELET_BASELINE.md.

## d3e8365 — 2026-06-08

**refactor: one generic split-radix DIF butterfly (pow2_dif_butterfly<IP>)**

Replace the hand-written radix4_dft / radix8_dft kernels with a single
recursive split-radix DIF template that generates the optimal butterfly for
any power-of-two radix, and route all pow2 IP>=4 through it. Adds:

- apply_stage_twiddle<T,Forward,IP,N,V>: per-stage W_IP^{+-N} with compile-time
  elision — axis twiddles (1,-1,+-i) cost 0 mults, diagonal (|c|=|s|=1/sqrt2)
  factor to 2 mults, general rotation 4 mults. Keys on exact ct_sincos_turns
  values, so radix-4 stays multiply-free and radix-8 keeps exactly 4 mults.
- pow2_dif_butterfly<T,Forward,IP,V>: radix-2 base (sum/diff), recursive even
  (sums) / odd (twiddled diffs) halves in separate scopes to bound peak live
  batches by poet::vector_register_count(). Emit order {0,4,2,6,1,5,3,7} for
  IP=8, term-for-term identical to the old radix8_dft; enables 16/32/... for
  free (planner radix set unchanged, so they stay dormant).

Verified: debug + ASan/UBSan ctest 94/94 pass; codegen A/B vs HEAD shows
identical vmulpd/FMA counts (radix-4: 0; radix-8: 2 mul + 4 FMA under
-ffast-math contraction); cycle A/B (pinned core 0, best of 5) within +-2% on
every radix-4 and pure-pow8 size, several faster.

## 556d364 — 2026-06-08

**feat: N-D FFT (row-column algorithm) with 1D/2D/3D + C API**

Extend the strictly-1D library to N-D via batched 1D transforms per axis.
The chiplet/codelet/butterfly layer stays 1D and unchanged; only per-axis
addressing is new.

Engine (detail/):
- nd_plan.hpp: nd_runtime_plan<T> — runtime-rank, per-axis state vector. The
  innermost axis reuses plan_impl::execute per row (full 1D SIMD); smooth outer
  axes take a batched SIMD DIF column pass; non-smooth outer axes fall back to
  scalar gather -> plan_impl -> scatter. Per-axis 1/L inverse scale composes to
  1/Ntot.
- dif_col_pass.hpp / dif_col_driver.hpp: batched (column) Gentleman-Sande DIF,
  vectorized over the contiguous trailing lane with broadcast twiddles; reuses
  dif_butterfly and the aos swizzles.
- nd_dispatch.hpp: thin runtime-rank one-shot wrapper.

Public C++ API (fft.hpp): unified plan<T> for any rank (runtime shape, no Dim
template, two nd_runtime_plan by value); free forward_nd/inverse_nd one-shots.
Existing 1D API preserved exactly.

C API (fft.h / c_api.cpp): fft_{forward,inverse}_nd_{float,double} one-shots and
fft_plan_nd_{float,double} reusable bidirectional plans (reuse the BOTH_* execute
path). Shape validation rejects null/zero-rank/zero-extent.

Also completes two partial designated initializers (plan.hpp, bluestein.hpp) that
tripped GCC -Werror=missing-field-initializers in template context.

Tests: test_fft_nd.cpp (2D vs ducc0 across catalog/7-smooth/fallback routes,
round-trip, Parseval, 1D-matches-legacy, 3D smoke, runtime dispatch, unified
plan) and N-D cases in test_fft_c.cpp. 94/94 pass under debug -Werror and
debug-asan (0 sanitizer reports). Benchmark: --compare-2d mode vs ducc0 2D.

## 0127d77 — 2026-06-08

**Merge refactor/detail-rename-and-idiomatic-sweep: header rename, nodiscard/unlikely, struct M m**

- Drop redundant fft_ prefix from detail/ headers; #pragma once throughout
- [[nodiscard]] on public API + pure queries/builders (26 sites)
- [[unlikely]] on error/degenerate guards (8 sites; not in inner loops)
- Group instance state into internal 'struct M { } m;' (no trailing-_ members)
- Include hygiene in public fft.hpp

All steps: relwithdebinfo -Werror clean, 71/71 ctest pass.

## a9be2c6 — 2026-06-08

**refactor: fix include hygiene in public fft.hpp**

Drop unused <optional> (std::optional is used in detail/plan.hpp, not here) and
add <algorithm> for the std::copy used in the out-of-place staging path (was
relying on a transitive include). Sort includes.

Phase 3 (copy/auto) audit otherwise found nothing to change: no range-for is
by-value (reads use const auto&, in-place mutation uses auto&), and heavy types
are only ever returned by value (move/NRVO) or passed by ref. Build clean;
71/71 ctest pass.

## be78539 — 2026-06-08

**refactor: group instance state into an internal 'struct M { } m;'**

Replace trailing-underscore data members with a single internal aggregate m
across the stateful types (plan_impl, forward_plan/inverse_plan/plan,
bluestein_plan, soa_scratch). Fields are now m.size, m.route, m.chirp, ... -
clean undecorated names with no _ suffix, and no collision with the size()/
is_forward() getters (which a same-named member would forbid in C++).
Constructors use C++20 designated initializers (m{.size = size, ...}).

soa_scratch intentionally leaves m DEFAULT-initialized (not designated-init) so
m.stack_buf stays uninitialized raw bytes - aggregate-init would zero-fill the
~128KB stack buffer and reintroduce the per-alloc memset the SBO path avoids.

Perf-neutral by construction: identical member set/order (just nested in M) =
identical layout and codegen. Build (relwithdebinfo, -Werror) clean; 71/71 ctest
pass.

## ab40792 — 2026-06-08

**perf: [[unlikely]] on error/degenerate guards in forward/inverse and plan_impl**

Annotate the cold guard branches (size-mismatch throw, empty-size early return,
N==0/N==1 short-circuits, plan-size-zero throw) so the optimizer keeps them off
the hot path. Hints are confined to once-per-call guards outside any inner loop.

Perf: neutral within measurement resolution. The benchmark harness noise floor
is ~+/-25% in this environment (an A/A control -- the SAME binary run twice --
showed flops/cycle swings up to +26.7%, driven by editor/clangd reindexing
contention). That swamps any signal from guard-branch hints, which mechanically
cannot affect inner-loop codegen. 71/71 ctest pass; -Werror clean.

## adf3213 — 2026-06-08

**feat: [[nodiscard]] on public API + pure queries; pragma once in public headers**

Public API (fft.hpp): [[nodiscard]] on make_plan factories, plan/forward_plan/
inverse_plan constructors (catches a discarded plan temporary), and all size()
getters. Internal pure queries/builders: is_power_of_2, next_power_of_2,
is_codelet_supported/_catalog, is_small_direct (math.hpp); dif_stage_cost,
build_dif_factor_plan, measured_dif_factor_plan, build_dif_twiddle_set,
dif_factor_plan::operator[] (twiddles.hpp); ct_sincos_small/_turns,
smallest_radix, codelet_radix (ct_math.hpp); build_direct_dft_twiddles
(direct.hpp); plan_impl::size/is_forward (plan.hpp).

Also convert public fft.hpp/fft.h to #pragma once (matches detail/ convention)
and fix stale old-filename references in ct_math.hpp comments.

Build (relwithdebinfo, -Werror) clean; 71/71 ctest pass.

## 49fd94b — 2026-06-08

**refactor: drop redundant fft_ prefix from detail/ headers; use #pragma once**

Rename include/fft/detail/fft_*.hpp -> *.hpp (prefix is redundant under the
fft/detail/ path), plus the generated fft_codelet_max.hpp -> codelet_max.hpp
and src/fft_c.cpp -> src/c_api.cpp. Update all #include references (sibling,
detail/, fft/detail/ forms), the generated codelet template/dispatch, and
src/CMakeLists.txt. Replace the inconsistent _H/_HPP include-guard mix with
#pragma once across all detail/ headers (macros.hpp keeps its deliberate
re-include #error guard). Fix stale macros.h comment references.

Build (relwithdebinfo, full catalog, -Werror) clean; 71/71 ctest pass.

## 91b852c — 2026-06-08

**docs: spec for detail/ header rename + idiomatic C++ sweep**


## 9efc9f2 — 2026-06-08

**Merge refactor/hpp-convention-and-flops: .hpp convention, header split, flops report**

7-phase behavior-preserving refactor (output bit-identical, perf within noise,
71/71 tests) plus the additive benchmark FLOPS/%peak report.

## 91962da — 2026-06-08

**refactor: anonymous-namespace symbol hygiene in TUs (Phase 7)**

Behavior-preserving; output bit-identical, 71/71 tests pass.

- bench_fft.cpp: all file-scope helpers (ducc0 wrappers, time_execution,
  nb_measure, parse_*/enumerate_*/make_*, factor_sweep_*, benchmark_size,
  report helpers, compare_min_of_n, peak_flops_per_cycle, fft_flops) wrapped in
  an anonymous namespace; only main() keeps external linkage.
- fft_c.cpp: the internal to_cpp_span<T> helper moved into an anonymous
  namespace (the C API entry points keep external "C" linkage; fft_error_string
  stays public per fft.h).
- codelet_apply.hpp left as-is: codelet_apply<N,T,Forward> needs external
  linkage (extern-template instantiated across the codelet TUs).

## 5e12672 — 2026-06-08

**feat(bench): report measured FLOPS vs theoretical peak (Phase 6)**

Additive — library output unchanged, 71/71 tests pass.

The --compare CMP line now reports forward-transform throughput for both fft and
ducc0:
- GFLOPS = flops/(us*1e3): wall-clock, familiar units.
- flops/cycle (+ %peak): frequency-invariant, consistent with the cyc ratio
  (printed only when perf counters are available).

flops = 5*N*log2(N) (FFTW/ducc convention; reporting only, not the win/lose
ratio). peak_flops_per_cycle<T>() is consteval = 2 FMA units * batch<T>::size
lanes * 2 flops/FMA = 16 (f64) / 32 (f32) on this AVX2 P-core.

## fee36be — 2026-06-08

**refactor: rename C++ detail headers .h -> .hpp (Phase 5)**

Mechanical, history-preserving (git mv); output bit-identical, 71/71 tests pass.

C++ headers use .hpp, C headers use .h. Renames:
- include/fft/detail/{fft_butterfly,fft_codelet,fft_dif_driver,fft_dif_passes,
  fft_direct,fft_kernels,fft_math,fft_plan,fft_twiddles,macros,undef_macros}.h
  -> .hpp
- src/codelet_apply.h -> .hpp
- src/fft_codelet_max.h.in -> fft_codelet_max.hpp.in (configure_file now emits
  fft/detail/fft_codelet_max.hpp; fft_math.hpp includes the .hpp)

Kept as-is: include/fft/fft.h (C API, extern "C"), include/fft/fft.hpp, and the
Phase 3/4 headers already created as .hpp. Every #include edge (incl. the three
test TUs), the CMake configure_file output path, and the .in self-reference
comments updated together.

## 96a40ed — 2026-06-08

**refactor: split scratch/swizzle/bluestein; thin fft_plan shell (Phase 4)**

Behavior-preserving; output bit-identical, 71/71 tests pass.

- fft_scratch.hpp: SBO_MAX/SBO_PAD + soa_scratch<T,K> (moved from fft_plan.h).
- fft_simd_swizzle.hpp: aos_deinterleave/aos_interleave (moved from
  fft_dif_passes.h, which now includes it).
- fft_bluestein.hpp: Bluestein extracted into a self-contained bluestein_plan<T>
  (precompute in ctor, execute() applies 1/N for inverse, private pow2_fft).
- fft_plan.h is now a thin routing shell: it holds the active route's twiddles
  plus a std::optional<bluestein_plan<T>> composed only on the bluestein route
  (was 5 always-present Bluestein members for "6 members, 1 active"). Includes
  only what it routes to; drops the fft_kernels.h umbrella and the duplicate
  fft_math include. fft_kernels.h stays as a tests/codelet-lib umbrella.

## e4d0084 — 2026-06-08

**refactor: extract fft_ct_math.hpp; cut kernel<N> from consumer TUs (Phase 3)**

Behavior-preserving; output bit-identical, 71/71 tests pass.

New include/fft/detail/fft_ct_math.hpp holds the compile-time math shared by both
the codelet kernel<N> and the runtime DIF chain: ct_sincos_t / ct_sincos_small /
ct_sincos_turns (consteval) and smallest_radix / codelet_radix (constexpr).

- fft_codelet.h now includes fft_ct_math.hpp (kernel<N>/make_twiddle_table stay).
- fft_butterfly.h, fft_twiddles.h redirect to fft_ct_math.hpp (they only need
  ct-math), no longer pulling the heavy kernel<N> template.
- fft_dif_passes.h dropped its fft_codelet.h include outright — it referenced
  nothing from it (the "last/fused fallback" use is gone).

Effect: fft_codelet.h (and kernel<N>) is now reachable only from the compiled
fft_codelets library and the two tests that exercise kernel<N> directly
(test_fft_codelet, test_fft_iterative). fft.hpp / fft_plan.h consumers (benchmark,
fft_c, plan tests) no longer instantiate kernel<N>.

## cc3586f — 2026-06-08

**refactor: constexpr/consteval upgrades (Phase 2)**

Behavior-preserving; output bit-identical, 71/71 tests pass.

- portable_trig::{reduced_sincos, sincos_reduced_turns, sincos_turns x2} ->
  constexpr (pure FP + switch; they have runtime callers in twiddle generation,
  so constexpr not consteval). Enables future compile-time twiddle use.
- dif_stage_cost / measured_dif_factor_plan -> constexpr (runtime plan-size
  callers, so constexpr).
- ct_sincos_small / ct_sincos_turns: constexpr -> consteval. Every call site is
  an immediate invocation that folds a compile-time twiddle (kernel<N> /
  butterfly matrix); none is emitted as runtime code. Forcing compile-time
  evaluation keeps them out of runtime codegen and speeds builds.

## 78f1014 — 2026-06-08

**refactor: remove dead code + fix stale comments (Phase 1)**

Behavior-preserving cleanup; output bit-identical, 71/71 tests pass.

Dead code:
- delete direct_dft_execute (zero callers; _ws variant is the only one used)
- delete log2_pow2 (zero callers; superseded by std::countr_zero)
- drop the unreachable ido>1 else branch in dif_pass_last (last pass is always
  ido==1); twre/twim become [[maybe_unused]]

Stale comments:
- fft.hpp: drop the removed "four-step" route from the dispatch comment
- fft_math.h: codelets are a STATIC (not OBJECT) lib; drop the dangling
  kernel_codelet_execute reference
- fft_dif_driver.h: radix-8 preference lives in build_dif_factor_plan's cost
  model, not a "dif_pass_radix"
- fft_butterfly.h / fft_dif_passes.h: trim "content-frozen" process scaffolding
  and the report/ path to a one-line bench-gated note
- fft_plan.h / bench_fft.cpp: drop "Phase 0" tags; fix the "min-of-N" reference
- CMakeLists.txt: fix the fft_c.h -> fft.h C-API path
- iterative_dif_execute / dif_pass_fused: note they are production-dead but
  exercised by test_fft_iterative

measured_dif_factor_plan (f64 2048 {8,8,4,8}): kept. Pinned cycle-true A/B vs the
DP plan {4,4,4,4,8} shows the override is ~17% faster on the forward transform
(fwd ratio vs ducc0 0.95 vs 1.12); documented with a measured-justification comment.

## 01a00d5 — 2026-06-08

**perf: anti-alias scratch padding fixes f64 N=4096 L1 conflict-miss**

Pad each of the 4 planar soa_scratch spans by SBO_PAD=16 elements when n%256==0
so their base addresses no longer collide in the same L1 cache sets (the re/im
and ping/pong buffers were 32 KB apart for n=4096 f64 -> conflict misses). This
nearly halves the f64 4096 cycle count: forward ducc0 ratio 1.30 -> 0.78 and
round-trip -> 0.85, turning the only real f64 loss into a clear win. SBO stack
buffer grows +512 B; no new allocation, user data untouched. ducc0's trick.

## 505cc97 — 2026-06-08

**perf(bench): --factors/--factor-sweep harness + CPU-cycle compare metric**

Add a benchmark-only DIF factor-order sweep (--factor-sweep) and a --factors=
override for the paired ducc0 --compare gate. Switch the compare metric from
wall-clock to per-process CPU cycles (nanobench Measure::cpucycles): frequency-
and contention-invariant, so a powersave governor and a hybrid P/E core that
migrates the thread can no longer skew the ratio. Output tags m=cyc|wall and the
ratio uses cycles when perf counters are available, else falls back to wall.

## 1b7edd7 — 2026-06-08

**feat: route_kind plan dispatch + factor-plan override; factor-plan test**

Replace plan_impl's bool route flags with a single route_kind enum chosen by
select_route() (codelet / iterative_dif / direct / bluestein), and thread an
optional dif_factor_plan override into the iterative-DIF twiddle build. Add
test_fft_factor_plan.cpp asserting planner optimality / factorization choices.

## 54767f2 — 2026-06-08

**feat: DIF factor-plan cost model (DP over radices) + radix-4 kernel**

Add dif_candidate_radices/dif_stage_cost/build_dif_factor_plan: a DP that picks
the minimum-cost radix factorization for the iterative DIF driver, with a cost
model penalising register-spilling radices (radix-8 only cheap for n<=512, etc.)
and a measured_dif_factor_plan override hook. build_dif_twiddle_set gains an
optional plan override. Add a dedicated radix4_dft butterfly (IP==4) to keep the
common pow2 pass off the generic constant-matrix path.

## b0472ad — 2026-06-08

**feat: explicit generated codelet catalog + codelet_radix(60)=5**

Replace the contiguous [2, CODELET_CATALOG_MAX] codelet range with an explicit
CMake-generated size list (CODELET_CATALOG_SIZES, with measured exclusions), one
straight-line kernel<N> TU per selected size. is_codelet_catalog now membership-
tests that list; fft_codelet_max.h.in is the single source of truth shared with
the per-N TU generation. codelet_radix(N) special-cases N=60 -> radix 5.

## 5c9459e — 2026-06-05

**perf: swizzle-based AoS boundary (de)interleave, kill .get() scatter**

Per dynamic perf-record on 1024/2048/4096 (f64 fwd), the fused DIF boundary
passes are 34-49% of cycles -- dif_pass_last<8> alone 31% at N=1024. They used
per-lane scalar .real()/.imag() gather and a runtime-index batch::get(lane)
scatter (the only .get() in the tree), lowering to a vpermpd/vunpck/mov pile
(~675 mov, 28 vpermpd). But the AoS data touched per j/k is W *contiguous*
complex, so the conversion is a perfect-shuffle, not a scatter.

Replace with hand-written batch<T> swizzle (double interface, not load_complex;
compile-time-mask zip only):
  - aos_interleave: zip_lo/zip_hi -> 2 contiguous stores (one pair, any W).
  - aos_deinterleave: log2(W) rounds of (zip_lo,zip_hi), the inverse perfect
    shuffle. Verified bit-exact for f32 (W=8) and f64 (W=4).
zip_lo/zip_hi lower to vunpck + vperm2f128 with IMMEDIATE masks (fast), unlike
runtime-index get() or AVX2 scatter (no hw op, emulated).

Measured (perf stat instr/call, load-independent; paired-interleaved wallclock,
core 2, 9 runs):
  instr/call:  f64 -3.2..-6.4%,  f32 -12.6..-21.6% (W=8 saves the wider lane loop)
  fwd wallclock: f64 -1..-6%,    f32 -15..-22%
  vs ducc0: flips f64-2048 (1.015->0.988) and f32-4096 (1.022->0.858) to wins;
  improves 512/1024/4096/2520 both precisions. f64-128 +1.8% wallclock is noise
  (instr/call dropped -3.2%).

69/69 ctest; libfft_codelets.a unchanged (214456 symbols).

## 9e83318 — 2026-06-05

**perf: selective radix-8 DIF for pure power-of-8 <=512 (fixes N=512)**

Add a fast radix-2^3 split butterfly (radix8_dft: all non-trivial rotations
folded into W8^1/W8^3, ~4 real mults; each DFT4 output emitted as soon as its
lane is ready to cap the live YMM set) and route IP==8 through it in
dif_butterfly; add 8 to dif_radix_set.

dif_pass_radix prefers radix-8 ONLY for a pure power-of-8 cofactor that stays
L1-resident (n<=512), i.e. N=512=[8,8,8] -> measured -29..-40% on both float
and double (fft ratio <=1.0x ducc, reproduced across runs). Outside that
regime radix-8 loses and we fall back to smallest_radix: a trailing radix-2/4
pass (1024=[8,8,8,2], 256=[8,8,4]) is a near-empty full-array sweep, and a
large pure-pow8 (4096=8^4, 64KB > 48KB L1d) is memory-bound and register-
starved (the f32/f64 delta even flips sign). So the non-512 watch set is
bit-identical to baseline by construction.

libfft_codelets.a unchanged (214456 symbols); 69/69 ctest.

## 23d1a92 — 2026-06-05

**perf(bench): nanobench-backed adaptive --compare (shorter + self-stabilizing)**

Replace the hand-rolled chrono min-of-N loop (fixed 200k/50k/8k/2k inner
counts) with nanobench. nb_measure returns the best-epoch per-call time
(min-of-N semantics) plus MdAPE as a stability flag, and self-stabilizes by
doubling minEpochTime (1->64ms) while err>5% instead of requiring a quiet
machine. Default auto-tunes (inner=0); --inner still forces a fixed count.
The CMP line now prints err=NN% and flags <== UNSTABLE. Full 4-size run ~0.18s.

## 0c526c3 — 2026-06-05

**perf: restrict-qualify DIF pass buffers; address via ptr+offset**

The DIF pass buffers never alias within a call: cc (source) and ch (dest)
ping-pong between two distinct scratch halves, the twiddle table is separate,
and the AoS `data` pointer is distinct from the planar SoA scratch. Mark the
pointer parameters of dif_pass / dif_pass_first / dif_pass_last / dif_pass_fused
(and their invoke functors) FFT_RESTRICT so the vectorizer need not reload
across stores it can prove independent.

Also rewrite the SIMD load/store addresses from &arr[expr] to arr + (expr)
(28 sites) — same address, reads cleaner with restrict-qualified pointers.

Macro hygiene: fft_butterfly.h and fft_codelet.h each include+undef macros.h in
their own scope, so re-include macros.h here (FFT_RESTRICT) and undef at EOF.

libfft_codelets.a unaffected (codelets use kernel<N>, not these passes) — symbol
count 214456. 69/69 ctest green. min-of-N (best-epoch) timings unchanged within
noise; 60 f64 fwd 1.01->0.93.

## e92f61c — 2026-06-05

**feat: route 11-smooth sizes through radix-11 DIF (fix 121, drop Bluestein)**

121 = 11^2 was the worst loser in both precisions (baseline min-of-N: f64 fwd
1.92x/rt 1.60x, f32 fwd 1.64x/rt 1.83x): it failed the 7-smooth gate and fell to
Bluestein, padding to 256 — three radix-256 DIFs for a 121-point transform.

radix_sym_dft / dif_butterfly are already generic over any odd radix, so this is
a pure routing change, no new kernel code:
- smallest_radix(): N % 11 == 0 -> 11 (before the prime-to-set fallback). For
  N <= 64 the smallest factor is never 11 except N==11 itself (returns 11 either
  way), so codelet bodies are unchanged — libfft_codelets.a symbol count stays
  214456.
- is_codelet_supported(): 11-smooth (factors in {2,3,5,7,11}); fft_plan routing
  picks up the iterative-DIF path automatically.
- fft_dif_driver: 4 poet dispatch sites use an explicit value list
  std::integer_sequence<int,2,3,4,5,7,11> (dif_radix_set) instead of
  inclusive_range<2,7> — adds 11, skips the never-emitted 6/8/9/10.

dif_pass_unroll<11> -> U=1 (peak-live 32 > 16 YMM), correct.

Result (min-of-N, taskset -c2): 121 f64 fwd 1.92->0.55, rt 1.60->0.60; f32 fwd
1.64->0.50. f32 round-trip 1.83->1.08 (still marginally over: the radix-11
butterfly's 32-batch live set spills on AVX2 — a kernel-quality residual,
tracked for the butterfly-pressure work, not a routing defect).

Tests: +1 iterative 11-smooth case (11,22,33,55,77,99,110,121,154,242 vs
kernel<N> oracle); 69/69 ctest green. Watch set unchanged (no regression).

## 9a36c95 — 2026-06-05

**refactor: std::numbers constants, constexpr predicates, min-of-N bench gate**

Use std::numbers in place of hand-rolled literals (per user request):
- fft_codelet.h: drop ct_pi<T> template (using std::numbers::pi); inv_sqrt2 =
  std::numbers::sqrt2 / 2.0 (division by 2 is exact -> bit-identical to the old
  literal). Verified: non-LTO .text + .rodata.cst twiddle pools byte-identical
  old vs new for codelet_5/8/31; libfft_codelets.a symbol count unchanged
  (214456). Frozen-codelet invariant preserved.
- portable_trig.hpp: using std::numbers::pi instead of a local pi_const.

constexpr-ify the pure integer routing predicates in fft_math.h
(next_power_of_2/is_power_of_2/log2_pow2/is_codelet_supported/
is_codelet_catalog/is_small_direct); constexpr implies inline.

benchmark: add paired min-of-N compare mode (--compare [--prec] [--reps]
[--inner] [--sizes]) — Phase C authoritative gate. Builds plans once, times
our execute() and ducc0 c2c over >=7 epochs, reports the minimum per-call
(fwd + round-trip, per precision) and the fft/ducc0 ratio.

68/68 ctest green (Release).

## 0718901 — 2026-06-05

**feat: float as a first-class citizen (test + benchmark parity)**

Tests: parameterize the double-only suites on T via Catch2 TEMPLATE_TEST_CASE
(test_fft.cpp, test_fft_analytical.cpp, and the plan round-trip cases in
test_fft_plan.cpp now run float+double). Add float+double C-API round-trip size
sweeps (pow2/prime/composite/prime-power/Bluestein) to test_fft_c.cpp. Replace
flat 1e-10/1e-6f tolerances with the relative budget eps*sqrt(N)*(log2N+1)*64
scaled by data magnitude, meaningful for both precisions. 45 -> 68 ctest cases,
all green.

Benchmark: templatize ducc0 wrappers, benchmark_size, and profile_single_size on
T; run the full sweep for float AND double and report per-precision summaries +
per-precision ducc0-faster lists (so a regression in one precision is never
masked by the other). Add --prec=float to the --size profiling mode.

Adopt nanobench (CPM) for timing: median wall-clock per call over auto-tuned
epochs with warmup, replacing the hand-rolled fixed-iteration mean loop. Add the
ducc0 c2c<float> instantiation TU (fft_inst1.cc) so float races the same entry
point; mark nanobench headers SYSTEM.

Use std where C++20 offers it: std::numbers::pi / pi_v<T> replace the hand-rolled
pi/pi_catalog literals in the test suites. (fft_math.h already uses std::bit_ceil
/has_single_bit/countr_zero.)

Cleanups: clarify the SBO_MAX stack-size comment as K*SBO_MAX*sizeof(T) (128 KB
double / 64 KB float), and document that ct_sincos_t is intentionally double
(twiddles computed in double, cast to T) — comment-only, codelet object library
byte-identical (symbol count 213950 unchanged).

## dedec7d — 2026-06-05

**refactor: dedup dif_pass_fused hot path via dif_butterfly**

The single-pass (N == one radix) path re-derived the radix-IP DFT matrix
inline. Replace with a dif_butterfly call so the fused path shares the
unified butterfly body — routing odd IP (3,5,7) through the low-multiply
symmetric kernel instead of the naive matrix. Output twiddle + AoS
scatter per k preserved (identity when ido==1, which the fused case
always is).

Paired bench-gate (taskset -c 2, min of 9 reps, idle) on
N in {2,3,4,5,7,8,9,16,32}: every size at or below baseline (7,8,16,32
improved 2-5%); no regression. 45/45 ctest green.

## cc26bdf — 2026-06-05

**refactor: dedup dif_pass_last ido>1 fallback via dif_butterfly**

The cold ido>1 branch of dif_pass_last (never reached by the iterative
driver, whose last pass always has ido==1) re-derived the DFT matrix by
hand. Replace it with a dif_butterfly call so it shares the single
butterfly body with dif_pass; scalar over (b, a) with output twiddle and
AoS scatter per k. Pure dedup of a cold path; 45/45 ctest green.

## 4016010 — 2026-06-05

**refactor: split fft_kernels.h into focused headers; keep a shim**

Pure code move, no body edits. fft_kernels.h is now a ~20-line aggregation
shim including six focused headers:
  fft_math.h       size predicates + codelet_dispatch declaration
  fft_twiddles.h   dif_twiddle_set + build_dif_twiddle_set
  fft_butterfly.h  radix_sym_dft / dif_butterfly / dif_pass_unroll (own macros bracket)
  fft_dif_passes.h dif_pass[_first/_last/_fused] + dispatch functors
  fft_dif_driver.h iterative_dif_execute[_ws]
  fft_direct.h     direct DFT (small non-7-smooth fallback)

Verified perf-neutral the strong way: an -S probe forcing every hot DIF body
(all radices, float+double, both directions) is BYTE-IDENTICAL before/after
(md5 match, 0 diff lines). The fft_codelets object library is unchanged
(nm --print-size identical) since it never included fft_kernels.h. 45/45 ctest.

## 0259fe7 — 2026-06-05

**refactor: remove fft_impl.h; consumers include fft_kernels.h/fft_math.h directly**

fft_impl.h had become a thin re-export after its dead algorithms were removed.
Delete it: fft_plan.h now includes fft_kernels.h + fft_math.h directly, and
fft.hpp drops the fft_impl.h include.

## 331e4d2 — 2026-06-05

**refactor: delete dead four-step machinery**

is_four_step_ was never set true, so the four-step path (four_step_twiddles,
build_four_step_twiddles, choose_four_step_factors, transpose_blocked,
four_step_execute_ws and plan_impl's fs_*/execute_four_step) was unreachable.
A clean A/B had shown four-step loses to the iterative DIF on every size once
the per-call sincos tax was removed. Remove it entirely.

## 9792559 — 2026-06-05

**refactor: delete dead recursive-DIT path; move size predicates to fft_math.h**

twiddle_set/build_twiddle_set, butterfly_invoke, mixed_radix_rec, and
codelet_execute[_ws] implemented the old recursive Cooley-Tukey DIT, which
production no longer uses (everything 7-smooth routes through the iterative
DIF driver, catalog sizes through codelet_dispatch). Remove them.

Move the size-class predicates is_codelet_supported / is_codelet_catalog /
is_small_direct (and DIRECT_DFT_THRESHOLD) into fft_math.h, the shared base
header, so fft_kernels.h is left with only the live kernel machinery.

## e7b996f — 2026-06-05

**refactor: delete dead radix-2/Bluestein span algorithms from fft_impl.h**

The span-based fft_radix2_span/fft_bluestein_span and the
fft_forward/inverse_dispatch + fft_forward/inverse_inplace helpers were
dead — production routes everything through plan_impl (fft_plan.h). Remove
them along with bit_reverse and pi_v. Move the integer utilities
(next_power_of_2/is_power_of_2/log2_pow2) into a new fft_math.h; fft_impl.h
becomes a thin transitional re-export. Drop the now-dead bit_reverse test.

## 522443e — 2026-06-05

**perf: register/SIMD-width-derived unroll policy for the DIF a-loop**

Make the vectorized DIF a-loop unroll factor an explicit consteval function of
the architectural vector register file: dif_pass_unroll<IP>() = registers /
peak_live, with registers = poet::vector_register_count() (16 YMM on AVX2, 32
ZMM on AVX-512) and peak_live = 2*IP+10 an empirically calibrated upper bound on
the symmetric kernel's live batch set (x0 + the 4*(IP-1)/2 pair sums/diffs + 4
accumulators + twiddle/output temps). The factor caps unrolling so the working
set still fits the register file; U==1 keeps the proven single-batch loop body
verbatim (the unrolled lambda lives only in the if constexpr (U>1) branch), so
on AVX2 the shipped dif_pass<5>/<7> codegen is instruction-for-instruction
identical to before (verified via -S, modulo label numbers) -- zero regression.

Applied to the two a-axis passes (dif_pass, dif_pass_first); dif_pass_last
vectorizes over b and is left as-is.

Why U=1 is correct here, not just conservative -- measured two ways:
- llvm-mca (alderlake) on the shipped hot loops: radix-5 5.54 uOps/cyc, radix-7
  5.85 of a 6-wide machine; both dispatch-saturated and FMA-port (p0/p1) bound.
  All stack traffic lands on idle load/store ports, fully overlapped.
- Direct paired bench, forced U=2 vs U=1 (taskset -c 2, 7 runs): U=2 is 23-51%
  SLOWER on every throughput-bound size (720 -50.8%, 1024 -46.2%, 360 -38.8%,
  1000 -35.6%, 4096 -26.2%, 210 -24.9%, 2520 -23.2%). The loops are already
  saturated, so a larger U buys no pipeline fill and pays full freight in spills
  + register pressure. peak_live is calibrated so radix-5/7 also stay U=1 on
  AVX-512 (the measured W=8 zero-spill point); a smaller constant spilled there.

calibration + evidence recorded in report/PHASE1_OPTIMALITY.md. 45/45 ctest
Release + 45/45 ASan.

## 895c427 — 2026-06-05

**perf: low-multiply symmetric radix-5/7 DIF butterflies (Phase 1)**

The radix-5/7 iterative-DIF passes were the largest remaining gap vs ducc0
(1000@1.58x, 2520@1.42x, 720@1.21x, 360@1.13x, 210@1.16x). simdref + llvm-mca
(alderlake) showed these passes are bound by the FMA port (p01), not by stack
spills -- the held inputs spill at cpi-0.5 onto otherwise-idle store ports and
overlap fully with the FMAs. The lever is fewer multiplies, not fewer spills
(an output-at-a-time reload variant removed every spill yet ran ~18% slower).

Exploit the conjugate symmetry W_IP^{(IP-m)k} = conj(W_IP^{mk}): pair input j
with IP-j so each output reuses the pair sum a_m = x_m+x_{IP-m} and difference
d_m = x_m-x_{IP-m}, halving the real multiplies. The kernel (radix_sym_dft) is
fully generic in odd IP via static_for -- no per-radix hand derivation -- and
emits each un-twiddled output the moment it is ready so the caller applies the
output twiddle and stores while only x0 + the pair sums/diffs stay live.

All three DIF pass variants (dif_pass / dif_pass_first / dif_pass_last) now
route through one dif_butterfly helper (odd IP -> symmetric kernel, even IP ->
naive matrix); this consolidates the previously-duplicated butterfly arithmetic
(seeds Phase Z2). The dif_pass family is codelet-isolated, so this cannot
regress the compiled kernel<N> codelets (N<=64), verified: kernel<60> machine
code is byte-identical before/after.

Measured (llvm-mca alderlake, W=4 block): radix-5 RThroughput 35.8->24.3 (-32%),
radix-7 79.3->44.2 (-44%). Paired-interleaved wallclock (taskset -c 2, 7 runs,
plan-reuse): 210 +36.7%, 1000 +27.4%, 2520 +23.8%, 720 +18.1%, 360 +15.5%.
Guards neutral: 1024 +0.9%, 4096 +1.5%, 60 +1.0% (15-run, identical codelet).
Correctness: 45/45 ctest Release + 45/45 ASan; bit-exact vs kernel<N> oracle.

## dd19852 — 2026-06-05

**perf: keep iterative-DIF scratch on stack up to N=4096 (raise SBO_MAX)**

Phase 0 profiling showed the per-execute soa_scratch<T,4> heap_.resize(4*N)
(N > SBO_MAX=1024) cost ~12.7% of N=4096 forward cycles (libc memset +
malloc/free), since the scratch is rebuilt every execute(). Raising
SBO_MAX 1024->4096 keeps the 128 KB (K=4, double) scratch on the stack for
all benchmarked pow2/7-smooth sizes (2048, 2520, 4096), respecting the
no-mutable-scratch decision rather than reintroducing plan-owned buffers.

Mechanism: perf confirms memset/malloc/free fully gone at N=4096.
Perf (paired-interleaved, taskset -c 2, P-core): forward per-call
  N=2048 +13.9%, N=2520 +8.1% (14 runs), N=4096 +26.5%.
Correctness: ctest Release 45/45, ASan 45/45; bit-exact.

## 55fc096 — 2026-06-05

**perf(bench): add --size/--iters single-size profiling mode**

Enables single-size instruction-level profiling (forward FFT in a tight
loop, no ducc0/report machinery) for the simdref/perf optimization
campaign. No effect on the default full benchmark path.

## 08eee6a — 2026-06-04

**perf: route pow2/7-smooth through iterative DIF, not four-step (beat ducc0 on average)**

With the per-call sincos tax gone (plan-owned twiddles), a clean A/B (taskset -c 2,
5 runs each) finally measures four-step's true value: it LOSES to the iterative DIF
on every affected size. Four-step's three blocked transposes + a per-sub-transform
AoS<->SoA deinterleave + a runtime codelet dispatch cost more than one
fused-deinterleave lane-over-columns sweep.

  size  four-step  iterative        size  four-step  iterative
   256     2.10       0.66          1024     3.04       0.97
   128     2.30       0.83           720     3.67       1.21
    90     2.63       0.80           360     3.22       1.13

Stop selecting four-step in the planner; pow2 / 7-smooth > 64 now route to the
iterative DIF. Overall fwd ratio vs ducc0 (7-run medians): 1.34 -> 0.79, i.e. we
now beat ducc0 on average; rt 0.81. Only 12 of 41 sizes remain > 1.0, mostly
Bluestein (121, 251) and large pow2 (2048, 4096).

The four-step machinery (four_step_execute_ws, build_four_step_twiddles,
transpose_blocked, choose_four_step_factors) is now dead; kept pending Phase Z
cleanup or a future multi-level DP planner that may reuse the blocked transpose.

45/45 ctest (full-catalog Release) + 45/45 ASan (reduced), all clean.

## ef06746 — 2026-06-04

**perf: eliminate per-call sincos tax via fast turn-reduction sincos + plan-owned twiddles (Phase A+B)**

perf record showed 38.5% of benchmark runtime in __sincos_fma: the runtime
twiddle builders used libm cos/sin, and the public free path
(fft::forward -> fft_forward_inplace) rebuilt all twiddles on every call
while ducc0 caches its c2c plans. Two changes remove the tax:

1. Vendor a minimal, self-contained portable_trig.hpp (trimmed from
   DiamonDinoia/polyfit: dropped polyfit/poet macro deps, kept the
   well-conditioned 6-coeff Horner kernel; poet::static_for for the unroll).
   Twiddles are exact rational turns exp(+/- 2*pi*i*num/den), so
   sincos_turns<Forward>(num, den) reduces num mod den and rounds to the
   nearest quadrant in exact integer arithmetic, evaluating the polynomial on a
   residual in [-pi/4, pi/4] -- faster and more accurate than a generic large-
   angle sincos, and with the direction folded into the template param so call
   sites need no casts. Replaces libm in all five runtime builders
   (build_twiddle_set, build_dif_twiddle_set, build_four_step_twiddles,
   build_direct_dft_twiddles, Bluestein chirp). Codelet (<=64) twiddles stay
   compile-time via ct_sincos_turns.

2. Route the free one-shot API through plan_impl so it gets the same
   codelet/four-step/iterative/Bluestein dispatch, and build the benchmark plan
   once per size (timing execute() only) -- the fair comparison vs ducc0's
   cached plan, with zero trig per call. No thread_local cache; the plan owns
   its twiddles.

Result (taskset -c 2, 7-run medians, fwd ratio vs ducc0): overall 2.63 -> 1.34
(-49%). All sizes improved, none regressed; primes 11/13/17 and pow2 8-64 now
beat ducc0. perf confirms __sincos_fma is gone from the hot path. 45/45 ctest
(full-catalog Release) + 45/45 ASan (reduced catalog), all clean.

## 6a73eda — 2026-06-04

**perf: four-step Cooley-Tukey for large composite N via codelet sub-transforms (Phase 3)**

For N = N1*N2 with both factors in the codelet catalog [2,64] (large pow2 and
7-smooth composites), replace the long radix-2..7 Gentleman-Sande sweep with a
four-step decomposition: a cache-blocked transpose, N2 contiguous column-DFTs of
length N1 (codelet), an inter-stage twiddle fused into the second transpose, N1
contiguous row-DFTs of length N2 (codelet), and a final transpose to natural
order. All sub-DFTs are straight-line codelets; every codelet stage runs over
CONTIGUOUS rows (the blocked transpose removes the strided-gather cache thrash of
a naive four-step).

Routing is gated to the iterative-DIF domain (pow2 / 7-smooth): non-7-smooth
composites such as 121=11*11 are left on Bluestein, which measured faster for
them (their codelet factors are O(P^2) direct DFTs).

Measured (taskset -c 2, medians vs Phase 2): large pow2 512 4.16->4.06x and
4096 3.02->2.94x (~+2.5%, consistent across runs); 720/1000/1024/2048 neutral
within the ~7% run-to-run noise; 121/127/251 unchanged (Bluestein). No regression.
The cache-blocking advantage scales further for N >> 4096 (beyond the current
benchmark, which fits L2). ctest 45/45 (added a four-step forward-vs-reference
DFT check on 100/128/144/256, plus round-trips).

## 7cd9b5f — 2026-06-04

**perf: route Bluestein inner pow2 transforms through the vectorized FFT (Phase 2)**

The Bluestein path (plan_impl, used for N>64 that are non-pow2 and non-7-smooth)
performed its three padded-size pow2 transforms with fft_radix2_span — a scalar,
AoS, cos/sin-per-butterfly radix-2 with bit-reversal. Replace them with the
production vectorized path via a new bluestein_pow2_fft() helper: codelet_dispatch
for padded sizes <= catalog, else iterative_dif_execute_ws with a dif_twiddle_set
built for padded_size_. Both a forward and an inverse twiddle set are precomputed
(the inner convolution is always fwd(a), fwd(kernel), inv(a) regardless of the
outer direction). Normalization matches the old path: forward un-normalized,
inverse scaled by 1/padded_size_.

Measured (taskset -c 2, 7-run medians, fwd ratio vs ducc0, A/B vs scalar baseline):
N=67 9.94->9.54x (+4%), 121 33.43->30.63x (+9%), 127 14.70->13.34x (+10%),
251 17.85->16.46x (+8%). No regression on other sizes (Bluestein path only). The
win compounds with Phase 3 (faster pow2 => faster Bluestein) and unblocks Phase 5.

Tests: add primes 127, 251 (Bluestein path) to the non-catalog round-trip checks
(double + float). Benchmark: add a Bluestein section (67, 121, 127, 251). ctest
44/44 debug+ASan.

## b93c02c — 2026-06-04

**perf: broaden codelet catalog to all N in [2,64] via compiled codelet library (Phase 1b)**

Wire the compile-time kernel<N> straight-line codelet into production for every
size in [2,64] (previously only the 7-smooth subset, Phase 1a). The heavy per-N
kernel<N> bodies (prime radices degenerate to P^2 straight-line DFTs, x4
{float,double}x{fwd,inv}) are compiled ONCE each into a new STATIC fft_codelets
library, one small TU per N, so consumer TUs see only the narrow
codelet_dispatch<T> declaration + extern template and instantiate nothing heavy.

Build split:
- src/codelet_instance.cpp.in -> per-N TU (CMake foreach, no preprocessor macros)
- src/codelet_apply.h: deinterleave -> kernel<N>::apply -> reinterleave wrapper
- src/codelet_dispatch.cpp.in: poet::dispatch over inclusive_range<2,MAX>
- STATIC (not OBJECT) lib: fft_tests links both fft::fft and fft::fft_c; an
  OBJECT lib would double-define codelet_dispatch in that one executable.
- Hidden visibility project-wide (-fvisibility=hidden) + FFT_C_API re-export of
  the public C API; ccache as compiler launcher.

Dev-build knob: FFT_CODELET_MAX cache var generates fft_codelet_max.h, the single
source of truth for both the per-N TU generation and CODELET_CATALOG_MAX in
fft_kernels.h, so the dispatch range can never reference an uncompiled codelet.
Lower it (e.g. -DFFT_CODELET_MAX=8) for fast dev iteration; sizes above the cap
fall through to the runtime mixed-radix path (correct, just slower).

Tests: split into one executable per source for parallel compile + targeted
relink; catalog correctness now loops N=2..64.

Measured (taskset -c 2, 7-run medians, fwd ratio vs ducc0 nthreads=1):
small primes 11: 3.85->0.76, 13: 2.94->0.61, 17: 3.58->0.83, 31: 3.79->1.16
(-69%..-80%, three now beat ducc0). No regression on previously-optimized catalog
sizes (64: 0.84->0.73, 48: 0.95->0.90); N>64 path untouched. ctest 44/44 green
(reduced-catalog debug+ASan; full-catalog Release).

## 226fd6d — 2026-06-04

**perf: route small N through compile-time kernel<N> codelet (Phase 1a)**

Wire the generic straight-line codelet kernel<N> into production via a poet
runtime->compile-time dispatch over a fixed catalog (7-smooth N in [2,64], 35
sizes). For catalog sizes the public API now executes kernel<N>::apply directly
(compile-time twiddles in .rodata, no ping-pong, no per-pass runtime dispatch)
instead of the iterative DIF driver; larger sizes still fall back to iterative
DIF / direct-DFT / Bluestein.

- fft_kernels.h: is_codelet_catalog + codelet_catalog_seq (static_assert 7-smooth
  guard) + kernel_invoke functor + kernel_codelet_execute[_ws].
- fft_impl.h: catalog branch first in all 4 dispatch functions.
- fft_plan.h: plan_impl routes catalog -> execute_codelet; gates iterative-DIF
  and small-direct precompute behind !is_codelet_catalog.
- test_fft_plan.cpp: public-API correctness (catalog vs direct DFT + round-trip;
  non-catalog round-trip-only).

Measured (taskset -c 2, 7-run median fwd ratio vs ducc0): overall 2.63 -> 1.97
(-25%). Catalog sizes improve 56-81%, most now beat ducc0 (e.g. pow2 8: 1.54->
0.35, 64: 2.48->0.84; 7-smooth 12: 1.61->0.42, 48: 2.91->0.95). 44/44 tests
pass in debug and debug-asan.

## 94869ec — 2026-06-04

**bench: add N=2048,4096 pow2 sizes for large-N perf tracking (Phase 0)**


## 0c9e043 — 2026-06-03

**perf: lane-over-columns vectorization for ido==1 passes (Phase 2b)**

The last DIF pass (always ido==1) was scalar over columns; now it packs W
consecutive column indices into batch<T> lanes and runs the generic radix
butterfly full-width, scalar tail for l1 % W. dif_pass_first also vectorized
over the a dimension. No boundary transpose needed (AoS scatter is contiguous).
Generic split re/im batch butterfly kept; no hand-specialized radices.

Serial paired bench (taskset -c 2, 7 runs, median fwd ratio vs ducc0):
  category     pre(45e3e5b)  p2b
  Power-of-2   2.52          2.38   -5.8%
  Prime        2.47          2.42   -2.3%
  Composite    2.12          2.03   -4.6%
  Mixed-radix  3.73          3.42   -8.2%
  ALL          2.77          2.60   -5.9%
Small N (2,4,5,7) now <1.0x (beats ducc0). 42/42 ctest (debug+ASan), bit-exact.

## 45e3e5b — 2026-06-03

**perf: fuse AoS<->SoA deinterleave into first/last DIF pass (Phase 2a)**

Eliminates the two standalone full-array scalar copy loops per execute():
- dif_pass_first reads std::complex directly into SoA (was: deinterleave loop)
- dif_pass_last writes SoA results straight to std::complex (was: reinterleave)
- dif_pass_fused handles the single-pass case (N=small radix) fully in AoS
Intermediate passes unchanged (SoA ping-pong). Recursive kernel<N> oracle intact.

Serial paired bench (taskset -c 2, 7 runs, median fwd ratio vs ducc0):
  category     pre(ed384d4)  p2a
  Power-of-2   2.63          2.57   -2.5%
  Prime        2.54          2.50   -1.6%
  Composite    2.36          2.15   -9.0%
  Mixed-radix  4.21          3.83   -8.8%
  ALL          3.00          2.82   -6.0%

42/42 ctest (debug + ASan), bit-exact vs oracle.

## ed384d4 — 2026-06-03

**refactor: drop mutable scratch; local hybrid-SBO buffers in execute()**

Removes all mutable + preallocated scratch members from plan_impl. execute()
stays const and allocates working scratch locally via soa_scratch<T,K>: a
single stack std::array for N<=1024 (32KB budget for double, K=4), heap
std::vector above. SBO branch lives once in the helper; call sites don't
branch on N. Precomputed twiddles remain const members built in the ctor.

Public API + results identical. 42/42 ctest (debug + ASan). Serial bench
neutral (avg fwd ratio 2.95 vs 2.99, within noise).

## 5f7dff5 — 2026-06-03

**feat: iterative DIF Gentleman-Sande driver as core (Sprint-1 D, beats A)**

Adopts the iterative DIF pass-chain driver (pocketfft-style CC/CH layout,
output-twiddle Gentleman-Sande, batch<T> over contiguous ido dim + scalar
tail, plan-owned ping-pong SoA scratch) as the mixed-radix + pow2 core,
replacing per-call recursion. Recursive kernel<N> kept as bit-exact oracle.

Serial paired bench (taskset -c 2, 7 runs, median, fwd ratio vs ducc0):
  category     base   A(recursive+r8)  D(iterative)
  Power-of-2   5.22   3.69             2.67
  Composite    2.98   2.88             2.38
  Mixed-radix  5.25   5.40             4.13
  Prime        2.57   2.55             2.53
  ALL          4.19   3.77             2.99
D wins every category; biggest gains on large pow2 (1024: 13.4->4.1x).

42/42 ctest green (debug + ASan); +2 iterative-vs-oracle tests (47008 asserts).

## b0e464d — 2026-06-03

**feat: SIMD codelet + small-prime direct DFT baseline (Sprint-1 B integrated)**

Establishes the feat baseline for swarm orchestration:
- generic SIMD codelet (SoA split re/im xsimd::batch<T>, compile-time twiddles)
- small-prime direct DFT (N<=64 non-7-smooth) replacing Bluestein (exp B, KEEP)
- plan-owned preallocated scratch, allocation-free execute()
- 40/40 ctest green (debug + ASan)

## 0dbfd42 — 2026-01-28

**using radix2 directly when possible**


## f8d5b81 — 2026-01-28

**Added initial implementation and benchmarks**


## a561323 — 2026-01-28

**Initial commit**

