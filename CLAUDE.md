# admiral

FFT library (compiled engine behind extern-template boundaries) with C++ and C APIs plus an FFTW shim. `README.md` is the reference. This file lists what an agent gets wrong first.

## Language standard

C++20 by default. `-DADM_CXX_STANDARD=17` builds the same kernels at C++17. 23 also builds. The seam is `include/admiral/detail/cxx_compat.hpp`. It holds a `span` polyfill, `ADM_CONSTEVAL`, `ADM_UNLIKELY`, `ADM_UNREACHABLE`, and bit ops. It also holds `numbers`, `type_identity` (struct indirection makes the alias a non-deduced context), and `const_find` (`std::find` is constexpr only in C++20).

Keep one spelling valid in both standards where possible: a templated lambda becomes a stateless struct with a member-template call operator, an `auto` parameter becomes a named template parameter, and Good-Thomas's gather keeps its masks compile-time through generator types keyed by integral NTTPs (class-type NTTPs and xsimd's array `make_batch_constant` are C++20-only). Constraint machinery keeps an `#if ADM_CXX20` pair, C++20 arm first: the `precision` and `ChunkBody` concepts, `run_tiles`' `requires std::invocable`, and the two `requires`-expressions in `twiddles.hpp`. Both arms constrain, never assert: a concept and `std::enable_if_t` each remove the overload, and a `static_assert` in the body makes the same mistake a hard error, not a substitution failure. The C++17 arm spells the constraint as `precision_void_t<T>` in a public function's return type, a defaulted template parameter internally, or a `void_t` member detection.

The five public class templates are the exception: they carry a one-line `static_assert` in both standards. Constraining their heads means constraining all 41 out-of-class member definitions in `src/cpp_api.hpp` to match. The assert names the offending T; a concept would only say the constraint failed.

Use `IC<expr>{}` to read an outer `static_for` index inside a nested lambda. cl rejects the captured index in a constant expression (C2131, cl 19.44 and 19.51). A snapshot local swaps the capture from the index to that local and moves gcc 14.2's inliner and a bit under `-ffast-math` (digest case 107). A runtime read needs nothing: name the index. Two sites hold a snapshot: `codelet_many_block` (`IC<B * W>{}` moves the inliner) and `granule_row_tile`'s `qb` (the snapshot is also the narrowing from `ptrdiff_t`).

Verify a seam change with `nm --defined-only -S`, comparing the multiset of symbol sizes. Do not use `objcopy -O binary --only-section=.text`: every template instantiation lands in its own COMDAT `.text._Z...` section, which that filter skips. At Release/x86-64/gcc 14.2 over 225 objects, 165 are bit-identical. The seam wrappers inline away. The instantiation TUs shrink. The codelets grow because master plus the `asm volatile` barrier reproduces shipped sizes on codelet_6, codelet_12 and codelet_16. Anything else that moves is a regression to explain.

The two builds are NOT link-compatible: `admiral::span` is `std::span` at C++20 and the polyfill at C++17.

## Auto thread count

`resolve_nthreads(n, total, K, w_ns, cls)` in `thread_pool.hpp` selects pool width at every public entry point, resolved inside the engine constructors where the route is known (`route_plan` in `plan.hpp` for 1-D, the `nd_runtime_plan` ctor for N-D, the axis/strides split in `cpp_api.hpp`). The law: argmin over pow2 nt <= P of `W/min(nt, knee_eff) + K * Dhat(nt, gap^)`, with gap^ the self-consistent per-dispatch gap (two fixed-point iterations) and the pocket rider (a hot gap < 30 us at >= C0/2 prices Dhat at >= 100 us). knee_eff is the class's two-tier knee: home-domain width below the f64-array gate, additive-cap tier past it. Serial below 2^15 elements or at K == 0. K = 5 (1-D four_step_large) / rank (N-D batch loops) / 1, 2 with the strides slab. W = `line_work_cyc` (leaf-priced <= 64, n log2 n octave stream beyond) at measured core frequency (`core_cyc_per_ns`: 3-cycle IMUL latency chain, not rdtsc. The TSC ticks at base clock and miscalibrates W by the boost ratio. The chain is a once-per-process ~1 ms probe cached in a static. It is the tree's one live-measurement exception to the formula-only policy, because the boost/base ratio differs per host). Dhat is the 12x6 wake-probe grid per host class keyed by (C0=soc cores, P): icelake 2x32, rome 2x64, genoa 2x48. Unknown hosts take the rome row. Re-fit: `mt_scaling --mode=probe`, add a row. Pow2 quantization stays: off-divisor counts load-imbalance static chunks by 2-4x.

`nthreads=0` timing on 2P nodes is bistable at 1-D 2^15-2^18 through the `effort::measure` race. Do not fit to a single run there.

## Alignment barrier

codelet.hpp's cofactor `Wc == r` fast arm reinterprets scalar pointers as aligned batch arrays behind a runtime check. The `asm volatile("" ::: "memory")` immediately before `kb_aligned` stops the compiler from hoisting direct loads above the check. Delete it and gcc 13.3 release at v3 segfaults in `codelet kernel<N> matches reference DFT` and `...::apply_sink...`. The same tree with the barrier passes 295/295. gcc 14.2 is clean either way, so re-verify on 13.3 before touching that arm. The barrier grows `codelet_apply<16u, float, true>` from 625 to 700 bytes and moves every codelet object by 1-2% at v3/gcc 14.2.

**ubsan will not catch it.** `-fsanitize=undefined` does not instrument the arm's `kernel_batched<...>::apply(reinterpret_cast<const V*>(xre), ...)` shape: a misaligned `const V a = p[0];` segfaults with no diagnostic, and `objdump -d` finds zero `__ubsan_handle_type_mismatch` sites in that function. ubsan instruments scalar loads and reference binding, not a whole-object copy through a dereferenced pointer. Measured clang 20.1.8, `-O1 -march=x86-64-v3`, 2026-09-13.

The barrier is the guard and a segfault is the detector. Gate a change here on gcc 13.3 at v3, not on a sanitizer run.

## Bitwise stability of the column engine

The col chain clones its store-policy lambdas per inline context. Under `-ffast-math` the compiler contracts and reassociates one butterfly differently per clone. Transform bits depend on data alignment class. The pin is in `src/CMakeLists.txt`, on the four `inst_col_*` TUs only: `-ffp-contract=on -fno-associative-math`, guarded on the compiler id because the spellings differ and an unknown `-f` name is a hard error on clang. The 1-D engine is NOT alignment-invariant, and the col pin does not cover it. At f64/gcc 14.2 native its bits depended on the address mod 64 at lengths 256, 384, 500, 512 and 1024 (64, 128, 192 and 1000 invariant), through plan, strides_plan and axis_plan's last axis. `-DADM_PIN_1D=ON` reduces this but does not remove it. dif_pass_last's output-alignment peel (`aos_store_align_peel`) is one source: master skips it at `l1 <= 4W` (a8f0c72), so above 4W rows it still makes bits alignment-dependent. The peel pays one full-price butterfly block per call to line-align the targets of stores that are unaligned forms anyway, so at four row blocks or fewer the block costs more than the split-store traffic it avoids. A change here needs a pass-level A/B at both 4W and 8W blocks with a misaligned output buffer: aligned buffers run peel=0 on both arms and see nothing. Keep both buffers of any bit-exact test that touches a 1-D or last-axis transform in one alignment class (two halves of one buffer, split at a multiple of 64 bytes).

`test/transforms/test_strides.cpp`'s "bit-identical across alignment classes" case is the check. `e2_len_cap()` sets the length at which it fails: 32 below 2 MiB of L3 per physical core, 64 above. Raising the cap moves 16 lengths from chain to codelet. They are: 33 35 36 40 42 44 45 48 49 50 54 55 56 60 63 64. The case tries {20, 60, 81, 96, 192, 256, 1024}. Exactly one, 60, is in that set.

At cap 32 the lowest detecting length is 33 (gcc) or 35 f32 / 33 f64 (clang). At cap 64 it is 66. Nothing at or below the cap detects, so the pin protects the col pass, not the leaf codelet. `admiral_codelets` needs no pin. Length 20 does not detect on this host. 96, 192, 256 detect under gcc only. 1024 detects nowhere. Every failure goes through `axis_plan`.

Len 81 is load-bearing: it is not in `CODELET_CATALOG_SIZES` and exceeds `kFourStepLeafMax`, so no cap can route it to the codelet. It cannot change sides. 96 does not detect under clang at any cap or ISA. Without 81, a clang host with >= 2 MiB L3 per physical core runs this gate with nothing behind it.

OPEN: a few lengths still change bits with the pin ON. All of them carry a prime factor above 11: 191, 226, 247, 257, 289, 291, 292, 293. They run the 1-D engine, not the col chain. A breakpoint on `col_dif_execute_ws<T,Forward>` counts zero hits at every one. No measurement covers Bluestein and Rader alignment sensitivity. No test looks at it.

Both flags are load-bearing. The 2x2 at v4/gcc 14.2: no pin fails, `-fno-associative-math` alone fails, `-ffp-contract=on` alone fails, the pair passes. `-fno-associative-math` and `-fno-tree-reassoc` are interchangeable here (geomean 1.0027 over 10 col cells), so the portable `-fno-associative-math` wins. `-ftree-reassoc` stays enabled. gcc gates FP reassociation on `flag_associative_math`, and nothing tests that gate.

The pin costs: FMAs in `inst_col_f_fwd.cpp.o` fall from 21110 to 3029, total FP instructions rise 20%, costing about 0.9% geomean wall clock and at most 3% on any cell. `inst_dif_f_fwd.cpp.o` is byte-identical, the control that the pin reaches those four TUs and nothing else.

A second invariant sits next to the flag pin: `strides_plan` bits must not depend on the OUTPUT layout. `choose_line_route` in `apply_lines_strided_oop` reads `src_line` only. `strides_state` builds axis states from `istride` only. Pricing the destination stride into either flips the route on wide output strides. The "bits do not depend on the output layout" case then fails at len 64 nbatch 2 out (8192, 1).

## N-D engine

- Rank >= 3, serial: `execute_nd` runs the last two axes plane by plane when one plane fits L2 and the array does not (`fuse_planes`). Below the gate the unfused chain is L2-resident. Above it the plane spills. Measured on SPR: 256^3 0.885, 64^3 0.974.
- Transposed route, rows of one page or more (`inner * 16 >= 4096`): the strip is one page of columns capped at 3/4 of L3. The line pitch is `len + W` when `len` is a page multiple (4K aliasing). The gate is `col_budget_block < 2W`. The step5a widening to `max(16, 2W)` predates the 2048 B col floor. With the floor in, both measurable W=4 flips favor col_dif (rome 2d_1024 block 10: 0.86, SPR-v3 4096^2 block 10, Bt floored to 128: 0.92). Where W < 8 the change restores the pre-step5a gate.
- col_dif tile row: `nd_col_block` floors `Bt` at 2048 B of contiguous run (`kColDifMinRowBytes`) when the array exceeds L3. The floor pays for the per-row cost only from DRAM. Floor alone loses 2d_1024 1.08-1.23x.
- A default-off arm must leave master's TEXT. `const std::size_t pitch = len;` alone shifts gcc 14.2's inline budget and breaks the `nm -S` size-multiset identity.
- Gate a code path off by gating its VALUE, not by restructuring the loops around it. A restructured loop changed gcc's codegen of a block loop on a path where neither arm peels; `peel = l1 > 4W ? f() : 0` kept master's codegen. Time a cell whose executed path is unchanged at every ISA/compiler anyway.

## Build

- `module load gcc/14.2.0`. The system g++ is 8.5 and cannot compile C++20.
- `unset NINJA_STATUS` before any `cmake --build`. A placeholder this ninja rejects aborts the build with an unrelated error. `scripts/validate.sh` already does it.
- Seven presets over `base`: `debug`, `asan`, `tsan`, `coverage`, `dev`, `release`, `relwithdebinfo`. Benchmark only `release`.
- `release` sets `ADM_BUILD_TESTS=OFF`. Configure with `-DADM_BUILD_TESTS=ON` for the full suite (~414 tests). `debug`, `dev` and `relwithdebinfo` set it ON.
- For a benchmark-versus-driver A/B, rebuild `admiral` and `admiral_benchmark` together: `admiral_benchmark` links `libadmiral_internal.so` while the public `plan<T>` path embeds its own engine copies in `libadmiral.so`, and `cmake --build --target admiral_benchmark` does not relink `libadmiral`. A kernel change measured through the pass bench reads 1.000 at transform level when `libadmiral.so` is stale.
- `scripts/validate.sh [isa|compilers|catalog|digest|cxx17|sanitize|valgrind|tidy|cppcheck|receipt-coupling|granule-admission]` is the one entry point. Every arm asserts against `compile_commands.json` that the flags arrived. CMake accepts an unknown `-D` in silence.
- `tidy` and `cppcheck` run as compiler launchers (`ADM_ENABLE_CLANG_TIDY`, `ADM_ENABLE_CPPCHECK` in `cmake/StaticAnalysis.cmake`), so a finding is a build failure over every library TU.
- `.clang-tidy` writes `Checks` as a YAML sequence, never folded. A folded scalar collapses to one line, so a `#` rationale becomes glob text and swallows the next entry. Nine checks were off that way. The launcher passes `--config-file` explicitly because generated codelet TUs live in the build tree and the upward search can miss the repo config. `scripts/static_analysis_control.sh` checks both. Every name in `.clang-tidy` must appear in `--list-checks`. Each analyser must be silent on a clean file and report on `test/static_analysis/control.cpp`.
- `coverage` is clang source-based coverage (`-fprofile-instr-generate -fcoverage-mapping`). Do not resurrect gcov: gcc's `--coverage` builds are assembler-bound (one TU took 1 h 50 m in `as`), clang's instrumented build is an ordinary compile.

## Precisions

`float` and `double` reach the SIMD engine. `long double` reaches `include/admiral/detail/scalar_fft.hpp` because no ISA carries lanes that wide. It covers `plan`, `plan_r2c` and the one-shots only. `axis_plan`, `strides_plan` and `plan_r2r` stay float and double. Gates are `detail::is_precision_v` and `detail::is_simd_precision_v`.

`long double` is three types: x87 double-extended on x86-64 Linux (`digits` 64), IEEE binary128 on aarch64 Linux (113), plain `double` on arm64 macOS and MSVC (53). A width can be narrower than the carrier expects, not only wider: `test_cxx_compat.cpp`'s `pi_v` case asks for distinct low bits where they exist and equality where they do not. Do not truncate a series at a fixed term count. `ct_sincos_small` runs until both tails fall under `eps(F)/2`. That is 9 iterations at 53 bits, 10 at 64, and 14 or 15 at 113. A 12-term fold missed by 437 eps at den 7 and 16000 at den 64. `std::numeric_limits<F>::digits` is the only honest way to ask.

The scalar backend carries no second copy of the radix math. `butterfly.hpp` is V-generic: the same text is the SIMD kernel at `V = xsimd::batch<T>` and the scalar kernel at `V = T`. An edit to a butterfly moves both, so a long double accuracy test can fail on a change that looks SIMD-only. `ct_math.hpp` folds twiddles at `ct_real_t<T>`, long double past double, because a double constant caps a transform at 2^-53.

## SIMD and dispatch

- All vector code goes through xsimd, all dispatch and unrolling through poet. Zero raw `_mm*_` intrinsics in `include/` or `src/`. Keep it that way. Do not use `xsimd::batch<std::complex<T>>`. Use split re/im.
- `alignas` takes the arch alignment (`batch_t::arch_type::alignment()`), never a literal. The single non-arch constant, `kCacheLine` in `cache.hpp`, has a comment that documents it.
- Never `(void)x` to silence a warning. Delete the dead thing or use `[[maybe_unused]]`.

## Compile-time architecture

A `forward` trampoline reaches each engine and turns one runtime bool into `<Forward>` leaves. The scale factor is a runtime argument folded into the last pass. The trampoline is the only place that names its two leaves. So one `extern template` per precision keeps a whole engine tree out of every TU that merely routes to it. The tree lives in `src/inst_*.cpp`, one TU per engine per direction. Whether a split pays is per-TU and measured: `inst_col_*` and `inst_dif_*` split per direction with byte-identical leaf codegen. `inst_plan_*` does not because its two route trees share sub-instantiations. `inst_gt_*` is an engine split. `inst_dif_thunks_*` hosts direction-free families once each. Do not mark such a trampoline always-inline. That pastes both arms into every call site.

## Measuring

- Wall time cannot resolve small effects here: within-arm spread reaches 17% on the compile side and 1.17× on the run side. Use differenced retired counters, two rep counts subtracted, and rotate arm order.
- Static asm cannot count executed work. In a hand-versioned function the object holds every version while the run executes one, so summed spill counts across versions are not comparable. The build realigns frames here to `%rbp`, so a `%rsp` spill regex reads a false zero.
- Whole-transform A/B drivers must not draw the working buffer from the default heap: two binary images of identical driver code put the first `std::vector` chunk at different 64B phases, and an alignment-sensitive path flips arms at random. Use `posix_memalign` and add the offset in code (al64/al16) so both arms hold one alignment.
- Judge a candidate's digest movers against master's movers, not against zero, and at the golden ISA (x86-64-v3): master's shipped arm already moves cases off golden, and a native build moves a different set. Build master and the candidate at the final commit with identical flags, run `admiral_digest` on each, and diff with `scripts/check_digest.sh master.txt cand.txt`. Re-run after every reshaping of the change; a pair run on an earlier shape does not cover the final symbol sizes.
- `.clang-format` is `ColumnLimit: 100`, but the tree is not format-clean: 188 lines exceed it. Re-measure that number. Do not trust it. Never run `clang-format -i` over a whole file. It rewrites unrelated hand-tuned layout. Format only the lines the change touches, and do not add new violations.

## Cost model

The routing cost model header (`include/admiral/detail/base_cost_model.hpp`) comes from a generator, not from hand edits: `cmake -DADM_FIT_COST_MODEL=ON`, targets `admiral_cost_sweep` then `admiral_cost_model` (refit from `ADM_COST_MODEL_DATA`). The fitter is C++20 stdlib-only (`tools/fit_cost_model.cpp`), no Python. Dry-run with `-DADM_COST_MODEL_OUT=<path>`.

It is coefficients only: 38 of them, one sparse lasso per form, pooled over every swept build. There is no per-(machine, size) table. Pricing each form off what the engine runs (the measured leaf table, `bluestein_choose_pad`, Rader) beat 316 rows outright: 6.10% to 4.16% mean route regret on all 16 swept builds, 0.80 geomean runtime on the 77 cells whose route moved, and 0.95 plan time inside 2..512. 12 of 8190 cells are worse at `effort::estimate`. `effort::automatic` recovers all 12 to 1.02 geomean.

Onboarding a machine: sweep it, refit over all receipts. The canonical receipt set is not in the repo. It lives in `~/repos/memory/admiral/bench-results/`. Pointing `--data` at that directory does not reproduce the tracked header because the fitter's walk is `directory_iterator`, not recursive. The fitter's walk skips the znver2 receipt in its dated subdirectory. The set that regenerates the tracked header byte-for-byte is the 8 top-level receipts plus `2026-08-06-znver2/base_cost_gnu14_x86_64_4a23d8b9.txt`, flattened into one directory. It excludes `base_cost_gnu14_x86_64_9a65e7f8.txt` and znver4, because the fitter has no znver4 row.

### The compiler is a model feature. The machine is not

One feature slot keys on the compiler because a paired sweep ran gcc and clang at matched width and vintage. The `four_step`-versus-`iterative_dif` contrast reads 1.25x apart between them below 512. The slot takes pooled route regret from 66.1 to 56.7 over the shared 16 rows. An unswept compiler takes the gcc branch.

The gcc f64 cost is real: at `effort::estimate` three cells regress beyond their own spread (N=91 1.25x, N=169 1.30x, N=366 1.19x, all `four_step` to `bluestein`), recovered by `effort::automatic` to 0.99/1.05/1.00.

Do not add a uarch slot. Adding the archived znver4 receipts costs regret (65.3 to 69.5), and the archive confounds machine with kernel vintage: gcc-14 at the same (W, regs) reads the same contrast 1.37x apart between the `gcc_v4` archive and a fresh sweep.

A refit is only comparable inside one tree. The fitter prices each form off `math.hpp`'s leaf tables, so the same receipts fitted on two trees give different coefficients. Reproduce the tracked header on the tree the change starts from before believing anything about the tree it ends on.

Only the two SPR receipts were swept against the kernels that ship. The v2-v4 and znver2 receipts predate the flat-prime, Meteor Lake and znver2-gap integration. Re-sweeping epyc7742 and epyc9474 is the follow-up.

### VINTAGE dominates the residual

Four pairs of rows carry identical model inputs and differ only in vintage. The stale row of each pair carries 2.7x the regret. The model fits the fresh rows to 1.0-1.5%, the sweep's own repeatability. Stale rows outnumber fresh 14 to 4. A bluestein pad slot gained on stale rows only and lost on all four fresh ones, re-routing nine cells for 3 wins and 5 losses. Split regret by vintage before believing a new slot, and confirm on the shipped build with a route diff filtered to the receipt's measured forms.

### The leaf-cost table holds absolute measurements

`codelet_cost_cyc_f64` / `_f32` / `_extra` in `include/admiral/detail/math.hpp` hold absolute measured cycles. A sweep on another machine reads a near-constant multiple (1.78 geomean, spread 1.26-3.4 on znver2), so importing another host's numbers re-frames the table instead of refreshing it. Never import a leaf-table hunk.

Refresh after a kernel change re-prices a leaf:

1. Sweep pre-change and post-change builds on one host, `--codelet-sweep --prec=both`, 7 sweeps, arm order alternating, pinned.
2. Take per-entry medians. A single sweep moves an entry by ~46%.
3. Move only entries whose ratio departs from 1 beyond their own spread. The ratio cancels the frame.
4. Confirm every mover against a codegen diff: `nm --defined-only -S` over both `libadmiral.so`, `codelet_apply` specialisations. An entry whose symbol size is byte-identical did not change. That check killed 10 of 27 candidate movers in 2026-08.
5. A new entry has no ratio. Price it from the post-change sweep rescaled by the frame factor measured over codegen-identical leaves only.

`gate_leaf_cyc_ref` stays frozen. It was the fit reference for the compile-time `N > 512` gates. Refreshing it de-calibrated that band by 1.63 geomean once.

### Runtime probes set the four_step_large admission lines

Past the cost model's domain, the `kLargeRoute*` machinery in `include/admiral/detail/four_step_large.hpp` decides admission. The other runtime-probed family is the E2 col-batch gate `kE2Len64MinL3PerCoreBytes` in `include/admiral/detail/nd_plan.hpp`, re-derived per host class. Receipt: `admiral/beat-standings/wi2c-large1d.md`, 2026-09-15.

The running host measures the serial f64/f32 lower edges. A lazy once-per-process-per-precision probe (`plan_impl<T>::probe_large_route_serial`) races forced dif against forced four_step_large serial plans. It uses a 3-rung ladder around the prior and caches the crossover. It engages only above the probe floor (2 MiB f64 / 4 MiB f32). The shipped constants are the ladder's prior and the full fallback. A new host class needs nothing. The probe reads it, and only the receipted wave process moves the fallback constants. Tests never see probe output: `set_large_route_serial_override` and its RAII scope inject the line, and the seam's storage sits in `src/large_route_probe.cpp`.

Nothing caps the f32 serial window now: four_step wins at each rung beyond it through 256 MiB. The threaded line (unreachable at nt=1, `nthreads > 1` guard in `plan.hpp`) is an element-keyed floor (`kLargeRouteThreadFloorF64Bytes` / `...F32Bytes`), flat to the knees. The knees are `kLargeRouteThreadKneeF64Nt` 32 and `...F32Nt` 8. Beyond the knees the line rises linearly to `kLargeRouteThreadCapBytes`, with pool width P as the anchor and nt=2 f32 at the cap outright. Named exceptions the law accepts: genoa's nt=8 f32 notch and icelake interpolation/cap compromise rungs. Past the cost model's domain the race re-checks the line behind `kMeasureRejectRatio` (1.25). Only the subset whose wrongness clears the bar recovers: the notch's 1.756x and the 1.35x interpolation rung do, the 1.06-1.20x compromise rungs do not.

`test/` pins the constants, the law's shape, and the seam.
