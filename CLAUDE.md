# admiral

FFT library (compiled engine behind extern-template boundaries) with C++
and C APIs plus an FFTW shim. `README.md` is the reference; this file only lists
what an agent gets wrong first.

## Language standard

C++20 by default. `-DADM_CXX_STANDARD=17` builds the same kernels at C++17; 23 also
builds. The seam is `include/admiral/detail/cxx_compat.hpp`: a `span` polyfill,
`ADM_CONSTEVAL`, `ADM_UNLIKELY`, `ADM_UNREACHABLE`, bit ops, `numbers`, `type_identity`
(the struct indirection is what makes the alias a non-deduced context; a bare alias is
not) and `const_find` (`std::find` goes constexpr only in C++20).

Constraint machinery keeps an `#if ADM_CXX20` pair, C++20 arm first, so deleting the
`#else` half leaves ordinary C++20: the `precision` and `ChunkBody` concepts, `run_tiles`'
`requires std::invocable`, and the two `requires`-expressions in `twiddles.hpp`. Both arms
constrain, never assert: a concept and `std::enable_if_t` each REMOVE the overload, while
a `static_assert` in the body makes the same mistake a hard error instead of a
substitution failure. The C++17 arm spells the constraint as `precision_void_t<T>` in a
public function's return type, a defaulted template parameter internally, or a `void_t`
member detection.

The four public class templates are the exception: they carry a one-line `static_assert`
in both standards. Constraining their heads means constraining all 34 out-of-class member
definitions in `src/cpp_api.hpp` to match, and the assert names the offending T where the
concept would only say the constraint failed.

Everything else carries ONE spelling, the one valid in both standards, because the C++17
form is also idiomatic C++20 and a second arm would only rot. A templated lambda becomes
a stateless struct with a member-template call operator, an `auto` parameter becomes a
named template parameter, and Good-Thomas's gather keeps its masks compile-time through
generator types keyed by integral NTTPs (class-type NTTPs and xsimd's array
`make_batch_constant` are C++20-only).

Verify a change to the seam with a per-object function-symbol diff: `nm --defined-only
-S`, comparing the multiset of symbol sizes. Do NOT use `objcopy -O binary
--only-section=.text`. Every template instantiation lands in its own COMDAT
`.text._Z...` section, which that filter skips, so a `.text`-only diff never looks at a
single kernel. At Release/x86-64/gcc 14.2 over the 225 objects of a full build:

- 165 objects carry a bit-identical size multiset, and every seam wrapper inlines away:
  `admiral::span`'s members, `const_find`, the bit ops, `make_unique_for_overwrite` and
  `cmp_less` have ZERO out-of-line definitions in either standard's library.
- The instantiation TUs SHRINK, because naming a closure type deduplicates what a
  templated lambda instantiated once per call site: `vpass_invoke` symbols fall from 120
  to 60, `inst_dif_f_{fwd,inv}` by 23%, `inst_real_{f,d}` by 11%.
- The codelets GROW, and the seam is not the cause. Master plus the `asm volatile`
  alignment barrier alone reproduces the shipped sizes byte-for-byte on codelet_6,
  codelet_12 and codelet_16, so the barrier owns the whole delta.

Anything else that moves is a regression to explain.

The two builds are NOT link-compatible: `admiral::span` is `std::span` at C++20 and the
polyfill at C++17.

## Alignment hazard that's already handled

codelet.hpp's cofactor `Wc == r` fast arm reinterprets scalar pointers as aligned batch
arrays behind a runtime check. Reading a misaligned-reinterpreted `V*` is UB the moment
it executes. The check makes that conditional dynamically, but a hoisting compiler can
still pull the direct loads above it. The `asm volatile("" ::: "memory")` immediately
before the `kb_aligned` computation is what stops the hoist, and it is load-bearing
today, not insurance against a future layout change: delete it and gcc 13.3 release at
v3 segfaults in `codelet kernel<N> matches reference DFT` and `...::apply_sink...`,
while the same tree with the barrier passes 295/295. gcc 14.2 is clean either way, so
a 14.2-only check cannot see this. Re-verify on 13.3 before touching that arm. The
barrier is not free: it constrains scheduling around the test, which grows
`codelet_apply<16u, float, true>` from 625 to 700 bytes and moves every codelet object
by 1-2% at v3/gcc 14.2.

## Build

- `module load gcc/14.2.0`. The system g++ is 8.5 and cannot compile C++20.
- `unset NINJA_STATUS` before any `cmake --build`. A placeholder this ninja
  rejects aborts the build with an error unrelated to the code; `scripts/validate.sh`
  already does it.
- Seven usable presets over a shared `base`: `debug`, `asan`, `tsan`, `coverage`,
  `dev`, `release`, `relwithdebinfo`. Benchmark only `release`.
- `scripts/validate.sh [isa|compilers|catalog|sanitize|valgrind]` is the one entry
  point for "does it build clean and pass everywhere". Every arm asserts against
  `compile_commands.json` that the flags it asked for actually arrived. CMake accepts
  an unknown `-D` name in silence, which is how a previous sweep ran four "sanitizer"
  builds carrying no `-fsanitize` at all.
- `coverage` is clang source-based coverage (`-fprofile-instr-generate
  -fcoverage-mapping`, -O1; report with llvm-profdata/llvm-cov). Do not resurrect
  the gcov preset: gcc's --coverage build here is ASSEMBLER-bound (one TU took
  1 h 50 m in single-threaded GNU `as`), clang's instrumented build is an ordinary
  compile.

## Precisions

`float` and `double` reach the SIMD engine. `long double` reaches
`include/admiral/detail/scalar_fft.hpp` instead, because no ISA has 80-bit lanes, and
it covers `plan`, `plan_r2c` and the one-shots only; `axis_plan` and `plan_r2r` stay
float and double. The two gates are `detail::is_precision_v` and
`detail::is_simd_precision_v` in the compat seam.

The scalar backend carries NO second copy of the radix math. `butterfly.hpp` is
V-generic: the same text is the SIMD kernel at `V = xsimd::batch<T>` and the scalar
kernel at `V = T`. An edit to a butterfly moves both, so a long double accuracy test
can fail on a change that looks SIMD-only. `ct_math.hpp` folds its twiddles at
`ct_real_t<T>`, which is long double past double, because a double constant caps a
transform at 2^-53.

## SIMD and dispatch

- All vector code goes through xsimd, all dispatch and unrolling through poet.
  There are zero raw `_mm*_` intrinsics in `include/` or `src/`; keep it that way.
  `xsimd::batch<std::complex<T>>` is banned. Use split re/im.
- `alignas` takes the arch alignment (`batch_t::arch_type::alignment()`), never a
  literal. The single non-arch constant, `kCacheLine` in `cache.hpp`, is documented.
- Never `(void)x` to silence a warning. Delete the dead thing or use
  `[[maybe_unused]]`.

## Compile-time architecture

Each engine is reached through a `forward` trampoline that turns one runtime bool into
`<Forward>` leaves; the scale factor is a runtime argument folded into the last pass,
not a template axis. The trampoline is the *only* place its two leaves are named, so
one `extern template` per precision keeps a whole engine tree out of every TU that
merely routes to it (`src/inst_*.cpp`, one TU per engine per direction; the comment in
`src/CMakeLists.txt` records which splits paid off). Do not mark such a
trampoline always-inline. That pastes both arms, engine included, into every call
site.

## Measuring

- Wall time cannot resolve small effects here: within-arm spread reaches 17 % on the
  compile side and 1.17× on the run side. Use differenced retired counters, two
  rep counts subtracted, and rotate arm order.
- Static asm cannot count executed work. In a hand-versioned function the object holds
  every version while the run executes one, so summed spill counts across versions are
  not comparable. Frames here are realigned to `%rbp`, so a `%rsp` spill regex reads a
  false zero.
- `.clang-format` is `ColumnLimit: 100`, but the tree is NOT format-clean: 188 lines
  exceed it (90 `include/`, 65 `benchmark/`, 26 `test/`, 7 `src/`). Re-measure rather than
  trusting that number: `find <dir> -type f \( -name '*.hpp' -o -name '*.cpp' -o -name
  '*.h' \) -print0 | xargs -0 awk 'length($0)>100' | wc -l`. Never run `clang-format -i`
  over a whole file. It rewrites unrelated hand-tuned layout. Format only the lines the change
  touches, and do not add new violations.

## Cost model

The routing cost model header (`include/admiral/detail/base_cost_model.hpp`) is
GENERATED, not handwritten: `cmake -DADM_FIT_COST_MODEL=ON`, targets
`admiral_cost_sweep` (re-measure this build, ~2 min) then `admiral_cost_model`
(refit from all receipts in `ADM_COST_MODEL_DATA`). The fitter is C++20 stdlib-only
(`tools/fit_cost_model.cpp`), with no Python. The refit overwrites the tracked header
by default; dry-run with `-DADM_COST_MODEL_OUT=<path>`.

It is COEFFICIENTS ONLY: 38 of them, one sparse lasso per form, pooled over
every swept build. There is no per-(machine, size) correction table: there used to
be 316 rows, and pricing each form off what the engine actually runs (the measured
per-precision leaf table, `bluestein_choose_pad`, length-(p−1) Rader) beat them
outright: 6.10% → 4.16% mean route regret on all 16 swept builds (same metric, rows
off both sides), 0.80 geomean runtime on the 77 cells whose route moved, and 0.95
plan time inside 2..512 because nothing walks a row table any more. 12 of 8190 cells
are worse at `effort::estimate` (worst 3.4x, N=67 f32); `effort::automatic` recovers
all 12 to 1.02 geomean with zero cells above their own noise. That is the trade the
rows were hiding. A table keyed to machines someone happened to sweep cannot race
candidates at plan time, and this model does not have to.

So onboarding a machine is just: sweep it (lands
`base_cost_<compiler>_<arch>_<flags+uarch hash>.txt`), refit over all receipts.
Nothing is keyed to silicon, so nothing has to be held fixed. The CANONICAL receipt
set is not in the repo (2e499e0 policy); it lives in ~/repos/memory/admiral/bench-results/
(the project's name, not this checkout's). Pointing `--data` straight at that
directory does NOT reproduce the tracked header: the fitter's walk is
`directory_iterator`, not recursive, so the znver2 receipt in its dated subdirectory
is skipped and every coefficient moves. The set that regenerates the tracked header
byte-for-byte is the 8 top-level receipts (the 6 legacy `base_cost_{gcc,clang}_v*`,
the SPR `e18facfb` and the clang-19 `c6aad4a5`) plus
`2026-08-06-znver2/base_cost_gnu14_x86_64_4a23d8b9.txt`, flattened into one
directory; `base_cost_gnu14_x86_64_9a65e7f8.txt` and znver4 are deliberately out (no
`uarch=znver4` row is fitted).

### The COMPILER is a model feature; the machine is not

One feature slot is keyed to the compiler. Build identity reaches the model nowhere
else except `log2(W)`, `log(bytes)` and the `regs` argument `chain_work` takes.
`regs` is collinear with `W` across every swept row, so it is not separately
identifiable. The compiler slot is there because of a paired sweep: one host, one
tree, one kernel vintage, back to back under one lock. That sweep reads the
`four_step`-versus-`iterative_dif` contrast 1.25x apart between gcc and clang, and
that pair is the dominant route below 512. The slot alone takes pooled route regret
from 66.1 to 56.7 over the 16 rows shared with the previous header, and 56.5 as
shipped (the slot plus the fresh clang-19 receipt), halving both worst rows
(clang-18 f32 W=4: 9.5 → 4.8; clang-18 f64 W=8: 10.3 → 5.8). Scored against each
receipt's MEASURED forms at AVX-512, it moves 37 cells under clang for a large net win
(f64 regret 2.34% → 1.45%, f32 2.39% → 1.36%, best cells 2.7x and 1.64x) and 10 cells
under gcc, where f32 improves slightly and f64 does not (0.97% → 1.13%). The two
compilers move the same pairs in OPPOSITE directions at equal width. No width or
machine key can express that. An unswept compiler (icx, msvc) takes the gcc branch.

The gcc f64 cost is real and measured: at `effort::estimate` three cells regress beyond
their own round-to-round spread (N=91 1.25x, N=169 1.30x, N=366 1.19x, all
`four_step` → `bluestein`), against one 1.23x gain at f32 N=71. `effort::automatic`
recovers all three to 0.99/1.05/1.00. Same trade the 316 correction rows bought, one
order milder.

Do NOT read this as licence for a uarch slot. Adding the archived znver4 receipts
COSTS regret (65.3 → 69.5 summed over the same 16 rows) and the archive confounds
machine with kernel vintage: gcc-14 at the same (W, regs) reads that same contrast
1.37x apart between the `gcc_v4` archive and a fresh sweep on this host. The compiler
is identifiable because gcc and clang were swept at matched width and matched
vintage; the machine is not.

A refit is only comparable inside ONE tree. The fitter prices each form off
`math.hpp`'s leaf tables through the shared helpers, so the same receipts fitted on
two trees give different coefficients. Reproduce the tracked header on the tree the
change starts from before believing anything about the tree it ends on.

Only the two SPR receipts (`base_cost_gnu14_x86_64_e18facfb.txt` and the clang-19
`base_cost_clang19_x86_64_c6aad4a5.txt`) were swept against the kernels that ship.
The clang/gcc v2-v4 and znver2 receipts predate the flat-prime, Meteor Lake and
znver2-gap integration, and re-sweeping epyc7742 and epyc9474 is the follow-up that
closes it.

### VINTAGE dominates the residual, so score a new feature on the FRESH rows

Four pairs of rows carry identical model inputs and differ only in receipt vintage
(`major` and `uarch` are not features, so each pair scores identically). The stale row
of each pair carries 2.7x the regret: 1.9/1.5, 4.0/1.0, 1.8/1.1, 5.8/1.4. Nothing
coefficient-shaped reaches that. The model already fits the fresh rows to 1.0-1.5%,
which is the sweep's own repeatability.

So a pooled regret sum is a majority vote by receipt count, and stale rows outnumber
fresh ones 14 to 4. A bluestein pad slot (`log2(pad2/M)`, the smooth-pad saving over
`bit_ceil`) was rejected on exactly this: it took the pooled sum 56.5 → 49.9, gained
on stale rows only, and lost on all four fresh ones, re-routing nine cells on the
shipped build for 3 wins and 5 losses (worst 1.74x, f32 N=251). A junk n-keyed control
gains exactly zero, so the metric is sound. The DATA is what is mixed. Split regret by
vintage before believing a new slot, and confirm on the shipped build with a route diff
filtered to the receipt's MEASURED forms.

### The leaf-cost table is measured, and its absolute scale decides routes

`codelet_cost_cyc_f64` / `_f32` / `_extra` in `include/admiral/detail/math.hpp` hold
absolute measured cycles, not relative weights. `iterative_dif` carries no leaf term,
so codelet-vs-`iterative_dif` turns on their magnitude and not only on intra-table
ratios. A sweep on another machine reads a near-constant multiple of the tracked
values (1.78 geomean, spread 1.26-3.4 on znver2), so importing another host's numbers
RE-FRAMES the table instead of refreshing it. Never import a leaf-table hunk.

Refresh protocol, after a kernel change re-prices a leaf:

1. Sweep the pre-change and post-change builds on ONE host, `--codelet-sweep
   --prec=both`, 7 sweeps, arm order alternating, pinned. `--codelet-sweep` emits no
   header row.
2. Take per-entry medians. A single sweep moves an entry by ~46%.
3. Move only entries whose RATIO between the two builds departs from 1 beyond their
   own spread. The ratio cancels the frame; an absolute reading does not.
4. Confirm every mover against a codegen diff: `nm --defined-only -S` over both
   `libadmiral.so`, `codelet_apply` specialisations. An entry whose symbol size is
   byte-identical did not change, whatever the timer says. That check killed 10 of 27
   candidate movers in the 2026-08 integration.
5. A new entry (a catalog addition, a new extras row) has no ratio. Price it from the
   post-change sweep rescaled by the frame factor measured over the codegen-IDENTICAL
   leaves only.

`gate_leaf_cyc_ref` STAYS FROZEN. It is the reference the compile-time `N > 512` gates
were fitted against, and refreshing it de-calibrated that band by 1.63 geomean once.

### The four_step_large admission lines are hand-fit, and they are measured crossovers

The `kLargeRoute*` constants in `include/admiral/detail/four_step_large.hpp` are the one
family of hand-fit numbers the cost model does not cover: the model's domain stops below
this band, so these lines are read off crossovers instead of fitted. A threshold and the
quantity it was fitted against are one artifact. Change either and BOTH have to be
re-derived, together, in the same run.

Re-derive by A/B-ing `four_step_large` against the serial DIF chain across the byte range,
one arm per route, and reading the crossover, not by nudging a constant until a cell
passes. The serial f32 line is a WINDOW (the DIF chain wins again from 64 MiB), so a sweep
that stops at 32 MiB reads only its lower edge. The threaded line is a budget over
`nthreads` clamped by a floor, so it needs a thread sweep and not a single thread count;
its floor is where the DIF stream stops being L2-resident per pass, which moves with the
host's L2. `test/` pins the threaded line against `large_route_threaded_bytes`, so a
re-derivation updates the test in the same commit.
