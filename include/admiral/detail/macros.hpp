// Compiler-portability macros for the fft detail headers.
//
// Define/undef paired with undef_macros.hpp (POET convention): each detail header
// that includes this must end with #include "undef_macros.hpp" to prevent leaking.
// Not include-guarded: the undef removes these symbols, so re-inclusion is valid.

#if defined(ADM_DETAIL_MACROS_ACTIVE)
#error "admiral/detail/macros.hpp included twice without undef_macros.hpp in between"
#endif
#define ADM_DETAIL_MACROS_ACTIVE 1

// ----------------------------------------------------------------------------
// ADM_ALWAYS_INLINE — force inlining regardless of compiler heuristics. Required
// on SIMD codelets so the packed body lands in the stage loop (without it GCC
// emits an out-of-line call and passes batches through memory).
// ----------------------------------------------------------------------------
#if defined(_MSC_VER) && !defined(__clang__)
#define ADM_ALWAYS_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
#define ADM_ALWAYS_INLINE inline __attribute__((always_inline))
#else
#define ADM_ALWAYS_INLINE inline
#endif

// ----------------------------------------------------------------------------
// ADM_LAMBDA_ALWAYS_INLINE — force-inline a lambda's call operator. Same intent
// as ADM_ALWAYS_INLINE but for the lambda-attribute position (after the parameter
// list): `[&](args) ADM_LAMBDA_ALWAYS_INLINE { ... }`. Used on hot per-block
// kernels (do_batch, per_b, load_tile). POET_ALWAYS_INLINE_LAMBDA is undef'd on
// poet exit, so we carry our own.
// ----------------------------------------------------------------------------
#if defined(_MSC_VER) && !defined(__clang__)
#define ADM_LAMBDA_ALWAYS_INLINE [[msvc::forceinline]]
#elif defined(__GNUC__) || defined(__clang__)
#define ADM_LAMBDA_ALWAYS_INLINE __attribute__((always_inline))
#else
#define ADM_LAMBDA_ALWAYS_INLINE
#endif

// ----------------------------------------------------------------------------
// ADM_FLATTEN — inline every call made within this function, recursively, so
// GCC's IPA-CP cannot outline the emit lambda as a .constprop clone (which forces
// a call + AVX->SSE vzeroupper + spill/reload in the pass hot loop). clang already
// inlines these; flatten makes GCC match. No effect on the function's own linkage.
// ----------------------------------------------------------------------------
#if defined(__GNUC__) || defined(__clang__)
#define ADM_FLATTEN __attribute__((flatten))
#else
#define ADM_FLATTEN
#endif

// ----------------------------------------------------------------------------
// ADM_NOINLINE — forbid inlining into the caller. Provides a regalloc barrier
// between kernel<N> and kernel<M=N/r> recursion levels and between large-N
// drivers and leaf codelets: each callee gets its own regalloc, so deep spills
// don't accumulate into the caller's live set. No portable mid-function regalloc
// reset exists (asm memory clobber is a memory barrier, not a regalloc reset);
// noinline is the reliable lever. Apply only above the register-pressure
// threshold — small leaves pay more in call overhead (see kernel_should_noinline).
// ----------------------------------------------------------------------------
#if defined(_MSC_VER) && !defined(__clang__)
#define ADM_NOINLINE __declspec(noinline)
#elif defined(__GNUC__) || defined(__clang__)
#define ADM_NOINLINE __attribute__((noinline))
#else
#define ADM_NOINLINE
#endif

// ADM_COLD — mark a function as rarely executed. gcc/clang sink it to
// .text.unlikely, out of the hot path's I-cache footprint, and bias callers'
// branches predicted-not-taken. Layout-only: hot code stays byte-identical.
#if defined(__GNUC__) || defined(__clang__)
#define ADM_COLD __attribute__((cold))
#else
#define ADM_COLD
#endif

// ----------------------------------------------------------------------------
// ADM_RESTRICT — non-aliasing pointer qualifier (proves planar re/im and twiddle
// arrays don't overlap, enabling vectorizer optimizations).
// ----------------------------------------------------------------------------
#if defined(_MSC_VER) && !defined(__clang__)
#define ADM_RESTRICT __restrict
#elif defined(__GNUC__) || defined(__clang__)
#define ADM_RESTRICT __restrict__
#else
#define ADM_RESTRICT
#endif
