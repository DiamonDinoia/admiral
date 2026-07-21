# Design: `detail/` header rename + idiomatic C++ sweep

Date: 2026-06-08
Branch: `refactor/detail-rename-and-idiomatic-sweep`

## Context

The build is already C++20 with a strict, *enforced* warning set
(`FFT_ENABLE_WARNINGS_AS_ERRORS` defaults ON): `-Wall -Wextra -Wconversion
-Wsign-conversion -Wdouble-promotion -Wold-style-cast -Wuseless-cast -Wshadow`
and more (`cmake/CompilerWarnings.cmake`). So "use `-Wdouble-promotion` and so
on" is done *and* enforced — the codebase compiles clean under it by
construction. `constexpr`/`consteval`/`static` are already heavily used
(twiddles, codelet catalog, register-count API).

What is missing / requested:
- `[[nodiscard]]`: **0** occurrences.
- `[[likely]]`/`[[unlikely]]`: **0** occurrences.
- `auto` used "as much as possible" *without* introducing copies — real intent:
  deduce types but never silently copy (`const auto&` / `auto&&` /
  `decltype(auto)`, not bare `auto x = expr`).
- `detail/` headers carry a redundant `fft_` prefix given they live in
  `include/fft/detail/`.

## Verification harness (all phases)

- Preset: `relwithdebinfo` (tests ON, benchmarks ON, warnings-as-errors ON,
  optimized + `-ffast-math`, full codelet catalog). Same preset throughout so
  perf comparisons are apples-to-apples.
- Build dir: `build/relwithdebinfo`. All builds/benches run in background +
  Monitor (per CLAUDE.md long-running-task rule).
- Correctness: `ctest` + accuracy-vs-reference (never byte-identity — fast-math).
- Perf gate (Phases 2–4): benchmark pinned `taskset -c 0` (P-core), compare
  **CPU cycles** not wall-clock. **Any** cycle regression on a hot size → the
  change is reverted and reported with numbers. No "looks fine" without data.

## Phase 0 — Rename `detail/` headers, drop `fft_` prefix (mechanical)

`git mv` 13 hand-written headers + 1 generated header template:

| old | new |
|-----|-----|
| `fft_bluestein.hpp` | `bluestein.hpp` |
| `fft_butterfly.hpp` | `butterfly.hpp` |
| `fft_codelet.hpp` | `codelet.hpp` |
| `fft_ct_math.hpp` | `ct_math.hpp` |
| `fft_dif_driver.hpp` | `dif_driver.hpp` |
| `fft_dif_passes.hpp` | `dif_passes.hpp` |
| `fft_direct.hpp` | `direct.hpp` |
| `fft_kernels.hpp` | `kernels.hpp` |
| `fft_math.hpp` | `math.hpp` |
| `fft_plan.hpp` | `plan.hpp` |
| `fft_scratch.hpp` | `scratch.hpp` |
| `fft_simd_swizzle.hpp` | `simd_swizzle.hpp` |
| `fft_twiddles.hpp` | `twiddles.hpp` |
| `src/fft_codelet_max.hpp.in` → generated `fft/detail/fft_codelet_max.hpp` | `codelet_max.hpp.in` → `fft/detail/codelet_max.hpp` |

Plus the folded-in optional: `src/fft_c.cpp` → `src/c_api.cpp`.

- Update every `#include` ref (sibling, `detail/`, `fft/detail/` forms) across
  `include/`, `src/` (incl. generated `*.in` templates), `test/`, `benchmark/`,
  and `src/CMakeLists.txt` (configure_file output path + `fft_c.cpp` source).
- Convert all `detail/` headers to `#pragma once` (consistent with the existing
  `macros.hpp`/`undef_macros.hpp`; replaces the inconsistent `_H`/`_HPP` guard
  mix left over from the Phase-5 `.h`→`.hpp` rename). `macros.hpp` keeps its
  deliberate re-include `#error` pattern.
- Out of scope: public `fft/fft.hpp` & `fft/fft.h` (prefix matches the lib, not
  redundant).
- Verify: clean `relwithdebinfo` build + `ctest`, then `graphify update .`.

## Phase 1 — `[[nodiscard]]` on the API (zero perf risk)

- Public (`include/fft/fft.hpp`): `make_plan` factory overloads, all `size()`
  getters, and the plan constructors (`forward_plan`/`inverse_plan`/`plan` —
  catches a discarded `forward_plan<float>(1024);`). `forward`/`inverse` are
  `void` → n/a.
- Internal: pure value-returning queries (factor-plan builders, `good_size`,
  register-count helpers).

## Phase 2 — surgical `[[likely]]`/`[[unlikely]]` (measured)

- `[[unlikely]]` on the size-mismatch throw + empty-size guards in
  `forward`/`inverse`; cold dispatch fallbacks (Bluestein / non-pow2).
- **Not** inside vectorized butterfly inner loops — those are register/spill
  bound (see memory `phase1-symmetric-radix-optimal`); hints there can hurt.
  Any in-loop hint must show a cycle win under asm-analysis or it is dropped.

## Phase 3 — copy/auto audit ("deduce types, never silently copy")

- `for (auto x : …)` → `const auto&`; value returns that should be `const&`;
  `decltype(auto)` where the exact forwarded type matters. Each hot-path change
  gated on the perf harness above.

## Phase 4 — `constexpr`/`consteval` expansion audit

- Hunt remaining runtime-computed-but-compile-time-known values. Honest
  expectation: mostly done; report what is actually left rather than force-fit.

## Commit strategy

One commit per phase (`refactor:`/`feat:`), matching the existing
`Phase N` convention. Perf-relevant phases include before/after cycle numbers in
the commit body.
