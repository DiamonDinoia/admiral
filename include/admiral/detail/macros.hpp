// Compiler-portability macros for the admiral detail headers. Every detail
// header that includes `macros.hpp` must end with `#include "undef_macros.hpp"`,
// so the macros never leak (POET convention). Not guarded: the undef removes
// the symbols, so re-inclusion is valid.

#if defined(ADM_DETAIL_MACROS_ACTIVE)
#error "admiral/detail/macros.hpp included twice without undef_macros.hpp in between"
#endif
#define ADM_DETAIL_MACROS_ACTIVE 1

// `ADM_ALWAYS_INLINE` forces inlining regardless of compiler heuristics, so the
// packed SIMD codelet lands in the stage loop. Without the attribute, GCC emits
// an out-of-line call and passes batches through memory.
#if defined(_MSC_VER) && !defined(__clang__)
#define ADM_ALWAYS_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
#define ADM_ALWAYS_INLINE inline __attribute__((always_inline))
#else
#define ADM_ALWAYS_INLINE inline
#endif

// `ADM_LAMBDA_ALWAYS_INLINE` force-inlines a lambda's call operator, in the
// lambda-attribute position: `[&](args) ADM_LAMBDA_ALWAYS_INLINE {}`.
// `POET_ALWAYS_INLINE_LAMBDA` is undef'd on poet exit, so admiral carries its
// own.
#if defined(_MSC_VER) && !defined(__clang__)
#define ADM_LAMBDA_ALWAYS_INLINE [[msvc::forceinline]]
#elif defined(__GNUC__) || defined(__clang__)
#define ADM_LAMBDA_ALWAYS_INLINE __attribute__((always_inline))
#else
#define ADM_LAMBDA_ALWAYS_INLINE
#endif

// `ADM_FLATTEN` inlines every call the function makes, recursively. Without
// the attribute, GCC's IPA-CP outlines the emit lambda as a `.constprop`
// clone. The pass hot loop then pays a call, an AVX->SSE `vzeroupper` and a
// spill/reload. clang already inlines the emit lambdas.
#if defined(__GNUC__) || defined(__clang__)
#define ADM_FLATTEN __attribute__((flatten))
#else
#define ADM_FLATTEN
#endif

// `ADM_NOINLINE` is a regalloc barrier between `kernel<N>` and `kernel<M=N/r>`
// recursion levels, and between large-N drivers and leaf codelets. Each callee
// gets its own regalloc, so deep spills never accumulate into the caller's
// live set. No portable mid-function regalloc reset exists, so `ADM_NOINLINE`
// is the only reliable control. Apply `ADM_NOINLINE` only above the
// register-pressure threshold; small leaves pay more in call overhead
// (`kernel_should_noinline`).
#if defined(_MSC_VER) && !defined(__clang__)
#define ADM_NOINLINE __declspec(noinline)
#elif defined(__GNUC__) || defined(__clang__)
#define ADM_NOINLINE __attribute__((noinline))
#else
#define ADM_NOINLINE
#endif

// `ADM_COLD` marks a function as rarely executed: gcc/clang sink the function
// to `.text.unlikely`, out of the hot path's I-cache. Layout-only; hot code
// stays byte-identical.
#if defined(__GNUC__) || defined(__clang__)
#define ADM_COLD __attribute__((cold))
#else
#define ADM_COLD
#endif

// `ADM_RESTRICT` is the non-aliasing pointer qualifier: planar re/im and
// twiddle arrays never overlap, so the vectorizer drops its alias checks.
#if defined(_MSC_VER) && !defined(__clang__)
#define ADM_RESTRICT __restrict
#elif defined(__GNUC__) || defined(__clang__)
#define ADM_RESTRICT __restrict__
#else
#define ADM_RESTRICT
#endif
