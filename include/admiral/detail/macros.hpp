
#if defined(ADM_DETAIL_MACROS_ACTIVE)
#error "admiral/detail/macros.hpp included twice without undef_macros.hpp in between"
#endif
#define ADM_DETAIL_MACROS_ACTIVE 1

#if defined(_MSC_VER) && !defined(__clang__)
#define ADM_ALWAYS_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
#define ADM_ALWAYS_INLINE inline __attribute__((always_inline))
#else
#define ADM_ALWAYS_INLINE inline
#endif

#if defined(_MSC_VER) && !defined(__clang__)
#define ADM_LAMBDA_ALWAYS_INLINE [[msvc::forceinline]]
#elif defined(__GNUC__) || defined(__clang__)
#define ADM_LAMBDA_ALWAYS_INLINE __attribute__((always_inline))
#else
#define ADM_LAMBDA_ALWAYS_INLINE
#endif

#if defined(__GNUC__) || defined(__clang__)
#define ADM_FLATTEN __attribute__((flatten))
#else
#define ADM_FLATTEN
#endif

#if defined(_MSC_VER) && !defined(__clang__)
#define ADM_NOINLINE __declspec(noinline)
#elif defined(__GNUC__) || defined(__clang__)
#define ADM_NOINLINE __attribute__((noinline))
#else
#define ADM_NOINLINE
#endif

#if defined(__GNUC__) || defined(__clang__)
#define ADM_COLD __attribute__((cold))
#else
#define ADM_COLD
#endif

// ADM_COLD/ADM_NOINLINE mark error arms and once-per-process probes only; a mark re-partitions
// every clone in its TU, so a hot sibling rules the site out (r6: thread_pool lifecycle marks).

// ADM_HOT marks per-call dispatch hubs; add a site only with a counter A/B at parity or better.
#if defined(__GNUC__) || defined(__clang__)
#define ADM_HOT __attribute__((hot))
#else
#define ADM_HOT
#endif

#if defined(_MSC_VER) && !defined(__clang__)
#define ADM_RESTRICT __restrict
#elif defined(__GNUC__) || defined(__clang__)
#define ADM_RESTRICT __restrict__
#else
#define ADM_RESTRICT
#endif
