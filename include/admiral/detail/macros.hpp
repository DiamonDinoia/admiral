// Compiler-portability macros for the fft detail headers.
//
// Mirrors POET's macros.hpp / undef_macros.hpp convention: define here, use in
// the detail headers, and #include "undef_macros.hpp" at the end of each header
// that included this one so the macros never leak into user translation units.
//
// Intentionally NOT include-guarded against re-definition the usual way: the
// paired undef header removes these symbols, so a header may include this one,
// use the macros, and undef them, and a later header can include it again.

#if defined(ADM_DETAIL_MACROS_ACTIVE)
#error "admiral/detail/macros.hpp included twice without undef_macros.hpp in between"
#endif
#define ADM_DETAIL_MACROS_ACTIVE 1

// ----------------------------------------------------------------------------
// ADM_ALWAYS_INLINE — force inlining regardless of compiler heuristics. Used on
// the SIMD codelets so the packed batch body lands directly in the stage loop
// (verified necessary via simdref: without it GCC emits an out-of-line call and
// passes batches through memory).
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
// list, no leading `inline`): `[&](args) ADM_LAMBDA_ALWAYS_INLINE { ... }`. Used
// on the hot per-block/per-group kernels (do_batch, per_b, load_tile) so the
// dedup'd body lands directly in the stage loop instead of an out-of-line call.
// Poet has POET_ALWAYS_INLINE_LAMBDA but <poet/poet.hpp> #undef's it on the way
// out, so we carry our own with the identical per-compiler form.
// ----------------------------------------------------------------------------
#if defined(_MSC_VER) && !defined(__clang__)
#define ADM_LAMBDA_ALWAYS_INLINE [[msvc::forceinline]]
#elif defined(__GNUC__) || defined(__clang__)
#define ADM_LAMBDA_ALWAYS_INLINE __attribute__((always_inline))
#else
#define ADM_LAMBDA_ALWAYS_INLINE
#endif

// ----------------------------------------------------------------------------
// ADM_NOINLINE — forbid inlining of this function into its caller. Used to put a
// register-allocation BARRIER between recursion levels of the codelet (kernel<N>
// calling kernel<M=N/r>) and between a large-N driver and its leaf codelets:
// each callee then gets its OWN register allocation, so a deep level's spills do
// not accumulate into the caller's live set. There is no portable construct that
// resets the allocator mid-function (an asm memory clobber is a memory barrier,
// not a regalloc reset), so a noinline call boundary is the reliable lever. Apply
// only ABOVE a register-pressure threshold — at small leaf sizes the call
// overhead erases the win (see codelet.hpp kernel_should_noinline).
// ----------------------------------------------------------------------------
#if defined(_MSC_VER) && !defined(__clang__)
#define ADM_NOINLINE __declspec(noinline)
#elif defined(__GNUC__) || defined(__clang__)
#define ADM_NOINLINE __attribute__((noinline))
#else
#define ADM_NOINLINE
#endif

// ----------------------------------------------------------------------------
// ADM_RESTRICT — non-aliasing pointer qualifier (helps the vectorizer prove the
// planar real/imag and twiddle arrays don't overlap).
// ----------------------------------------------------------------------------
#if defined(_MSC_VER) && !defined(__clang__)
#define ADM_RESTRICT __restrict
#elif defined(__GNUC__) || defined(__clang__)
#define ADM_RESTRICT __restrict__
#else
#define ADM_RESTRICT
#endif

// ----------------------------------------------------------------------------
// ADM_ASSUME(expr) — promise `expr` holds at this point. The runtime-condition
// analog of `if constexpr` dead-code elimination: lets the compiler delete code
// paths that contradict it (e.g. the ido==1 branch of a middle pass that the
// driver never reaches with ido==1). UB if violated — use only for invariants
// proven at the call site, never as a substitute for validation.
// ----------------------------------------------------------------------------
#if defined(__clang__)
#define ADM_ASSUME(x) __builtin_assume(x)
#elif defined(__GNUC__)
#define ADM_ASSUME(x) do { if (!(x)) __builtin_unreachable(); } while (0)
#else
#define ADM_ASSUME(x) ((void)0)
#endif

// ----------------------------------------------------------------------------
// ADM_LIKELY/ADM_UNLIKELY — branch-probability hints. Steer block layout so the
// hot path stays fall-through/contiguous and cold edges (rare scalar tails) sink
// out of line, tightening the hot loop's I-cache footprint.
// ----------------------------------------------------------------------------
#if defined(__GNUC__) || defined(__clang__)
#define ADM_LIKELY(x)   __builtin_expect(!!(x), 1)
#define ADM_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#define ADM_LIKELY(x)   (x)
#define ADM_UNLIKELY(x) (x)
#endif

// ----------------------------------------------------------------------------
// ADM_HOT/ADM_COLD — function placement hints. hot keeps the pass kernels in a
// contiguous hot text section; cold marks rarely-run helpers so they neither
// pollute the hot section nor pull in aggressive inlining.
// ----------------------------------------------------------------------------
#if defined(__GNUC__) || defined(__clang__)
#define ADM_HOT  __attribute__((hot))
#define ADM_COLD __attribute__((cold))
#else
#define ADM_HOT
#define ADM_COLD
#endif
