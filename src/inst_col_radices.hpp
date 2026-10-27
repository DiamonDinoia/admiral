#pragma once

#include "admiral/detail/dif_col_driver.hpp"

#include <type_traits>
#include <utility>

// dif_radix_set and its staged subset per target, after poet's register count. Radix 2 stays in
// the col_dif_execute_ws TU: clang inlines its passes into the dispatch thunk.
#if defined(__AVX512F__) || defined(_M_ARM64) || defined(__ARM_NEON) || defined(__ARM_NEON__) ||   \
    defined(__ALTIVEC__) || defined(__mips_msa)
    #define ADM_COL_RADICES(X) X(3) X(4) X(5) X(7) X(8) X(11) X(16) X(32) X(9) X(15) X(25) X(10)
    #define ADM_COL_STAGED_RADICES(X) X(16) X(32)
#else
    #define ADM_COL_RADICES(X) X(3) X(4) X(5) X(7) X(8) X(11)
    #define ADM_COL_STAGED_RADICES(X)
#endif

#define ADM_COL_PASS(T, R)                                                                         \
    void dif_col_pass<T, R>(const T*, const T*, T*, T*, std::size_t, std::size_t, std::size_t,     \
                            const T*, const T*)
#define ADM_COL_PASS_FIRST(T, F, R)                                                                \
    void dif_col_pass_first<T, F, R>(const std::complex<T>*, std::size_t, T*, T*, std::size_t,     \
                                     std::size_t, std::size_t, const T*, const T*)
#define ADM_COL_PASS_FIRST_STAGED(T, F, R)                                                         \
    void dif_col_pass_first_staged<T, F, R>(const std::complex<T>*, std::size_t, T*, T*,           \
                                            std::size_t, std::size_t, std::size_t, const T*,       \
                                            const T*)

namespace admiral {
namespace detail {

template<std::size_t... R>
constexpr std::size_t col_staged_count(std::integer_sequence<std::size_t, R...>) {
    return (std::size_t{dif_staged_radix<R>} + ...);
}

#define ADM_COL_SEQ(R) , R
#define ADM_COL_ONE(R) +1
#define ADM_COL_IS_STAGED(R) &&dif_staged_radix<R>
static_assert(std::is_same_v<dif_radix_set,
                             std::integer_sequence<std::size_t, 2 ADM_COL_RADICES(ADM_COL_SEQ)>>);
static_assert(col_staged_count(dif_radix_set{}) == 0 ADM_COL_STAGED_RADICES(ADM_COL_ONE) &&
              (true ADM_COL_STAGED_RADICES(ADM_COL_IS_STAGED)));

}
}
