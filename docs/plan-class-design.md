# Plan class structure — clean + performant + ISA-parametric (2026-07-05)

## What's actually there today

`plan_impl<T>` (`detail/plan.hpp`) is already a thin routing shell: `select_route()`
constexpr-picks one of 8 routes at construction, precomputes only that route's
twiddles, `execute()` switches on it. That part is fine and doesn't need a rewrite.

Two real smells, both tied to the ISA goal:

1. **`select_route` is precision-forked, not width-parametric.** `if constexpr
   (sizeof(T)==4/8)` blocks with hand-found size lists (`{9,15,21,25,33,…}` f64,
   `{10,14}` f32, the f32-only vecpass/four_step_batched gates). This is the
   de-spaghetti target and the thing that will *not* route correctly on ARM/other
   widths — it's keyed to precision, but the real driver is lane width `W`.

2. **Fat plan state.** `M` always holds `dif_tw`, `dtw`, `fs_split`, `fs_tw` as
   live members + 4 `optional`s, though exactly one route is ever active. Minor;
   the optionals already lazy-compose the heavy states.

## The mechanism reality-check (verified in poet sources)

- `poet::dispatch(functor, dispatch_param<seq>{val}, …)` = runtime int → compile-time
  int, via a jump table. Correct tool for the **last mile** of a chiplet: turn a
  runtime `instruction_set` value into a compile-time arch tag.
- `poet::detect_instruction_set()` is **`consteval`** — it returns the *build-time*
  ISA from `__AVX512F__`/`__AVX2__`/… macros. **Not** runtime CPUID.
- ⇒ A true chiplet (one fat binary, runtime-pick best ISA) is **not free**: it needs
  (a) runtime detection (xsimd `available_architectures()` / CPUID), (b) each kernel
  TU compiled once per `-march`, (c) `poet::dispatch` over the runtime ISA into the
  right TU. That's real build-system + code work.

**And it buys portability, not speed.** A chiplet binary is at best equal to a
correctly `-march`'d build on the same host — usually slightly worse (indirect
call + no cross-TU inlining at the dispatch seam). The campaign's *performance*
goal is served by the per-ISA parametric fixes (valley ✓, f64 vecpass, split
retune), which a per-`-march` build already gets. The chiplet is a *deployment*
feature for shipping one binary to heterogeneous hosts.

## The clean design (unifies per-ISA build AND future chiplet, zero dup)

Parametrize the plan on a compile-time **arch tag**, defaulted to the build-time ISA:

```cpp
template<class T, poet::instruction_set ISA = poet::detect_instruction_set()>
class plan_impl {
    static constexpr std::size_t W = xsimd::batch<T, xsimd_arch_of<ISA>>::size;
    // route table is a function of (T, W) — no sizeof(T) forks
    static constexpr route_kind select_route(std::size_t n) { /* uses W */ }
    …
};
```

- **Per-ISA build (today):** default arg = build-time ISA. Identical binaries to
  now, but routing is width-driven → ARM/SSE/AVX-512 all route correctly with no
  new precision forks. This is the simplify + parametric win.
- **Chiplet (opt-in, later):** `poet::dispatch` over the runtime ISA constructs
  `plan_impl<T, ISA_rt>`. Same class, no second code path.

Jason Turner / Core-Guidelines principles this already or newly honors: constexpr/
consteval-all-the-things (route table computed at compile time — "the fastest code
is the code that never runs at runtime"), `enum class` tags, `if constexpr` over
SFINAE, free functions + minimal const state, values over mutation.

## Decision (2026-07-05): NO CPUID/chiplet dispatch

Confirmed with the user: **do not build the runtime chiplet.** It buys portability,
not speed, and per-`-march` builds already get the per-ISA kernels. The arch-tag
template default (`ISA = poet::detect_instruction_set()`) is still worth keeping as
the vehicle for width-parametric routing — but never dispatched at runtime.

## Sequencing

1. **`select_route` de-spaghetti — DONE (2026-07-05).** The three inline precision-
   forked size-switches are extracted into one named `constexpr dif_beats_codelet(n)`;
   the router now reads as a clean first-match priority ladder. Behavior-preserving
   (`--verify` PASS across dif-wins/codelet/vecpass/four_step/rader/direct/bluestein/
   pow2). NOTE: the size-lists are *empirical measured win-sets*, precision-specific
   facts — NOT a width formula — so the `sizeof(T)` fork correctly stays as compile-
   time `if constexpr`. "Width-parametric" applies only where a real width property
   drives the choice (the f32-only vecpass/four_step_batched gates, already so).
2. **Multi-threading performance** (NEW priority): measure yafft's threaded plan vs
   ducc0(N)/FFTW(N) via `-DFFT_BENCH_THREADS=ON` + `--nthreads=N`. Tier2/tier3
   (L3/RAM-exceeding) sizes where threading actually pays.
3. **Per-ISA kernel fixes:** f64 vecpass at W=8, split/`dif_stage_cost` retune.
