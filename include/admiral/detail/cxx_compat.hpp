#pragma once

#include <array>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <memory>
#include <type_traits>
#include <utility>

#if defined(_MSC_VER) && !defined(__clang__)
#  define ADM_CPLUSPLUS _MSVC_LANG
#else
#  define ADM_CPLUSPLUS __cplusplus
#endif

#define ADM_CXX20 (ADM_CPLUSPLUS >= 202002L)

// gcc 13 rejects a consteval function reached from a static data member initializer of a class
// template as "used before its definition" even when the definition precedes it lexically
// (codelet.hpp's `flat_leaf` calling `cofactor_simd_profitable`). gcc 14 accepts it. Every call
// site is already a constant expression, so constexpr evaluates identically there.
#if defined(__cpp_consteval) && __cpp_consteval >= 201811L \
    && !(defined(__GNUC__) && !defined(__clang__) && __GNUC__ < 14)
#  define ADM_CONSTEVAL consteval
#else
#  define ADM_CONSTEVAL constexpr
#endif

#if defined(__cpp_lib_is_constant_evaluated) && __cpp_lib_is_constant_evaluated >= 201811L
#  define ADM_IS_CONSTANT_EVALUATED() std::is_constant_evaluated()
#elif defined(__GNUC__) || defined(__clang__)
#  define ADM_IS_CONSTANT_EVALUATED() __builtin_is_constant_evaluated()
#else
#  define ADM_IS_CONSTANT_EVALUATED() (false)
#endif

#if ADM_CXX20 || (defined(__GNUC__) && !defined(__clang__))
#  define ADM_UNLIKELY [[unlikely]]
#else
#  define ADM_UNLIKELY
#endif

#if defined(__cpp_lib_unreachable) && __cpp_lib_unreachable >= 202202L
#  define ADM_UNREACHABLE() std::unreachable()
#elif defined(__GNUC__) || defined(__clang__)
#  define ADM_UNREACHABLE() __builtin_unreachable()
#else
#  define ADM_UNREACHABLE() std::abort()
#endif

#if __has_include(<span>)
#  include <span>
#endif
#if __has_include(<bit>)
#  include <bit>
#endif
#if __has_include(<numbers>)
#  include <numbers>
#endif

namespace admiral {

#if defined(__cpp_lib_span) && __cpp_lib_span >= 202002L
using std::dynamic_extent;
using std::span;
#else
inline constexpr std::size_t dynamic_extent = std::numeric_limits<std::size_t>::max();

template <class T, std::size_t Extent>
class span;
template <class C>
struct is_span : std::false_type {};
template <class T, std::size_t E>
struct is_span<span<T, E>> : std::true_type {};

template <class T, std::size_t Extent = dynamic_extent>
class span {
    static_assert(Extent == dynamic_extent,
                  "admiral::span: the C++17 polyfill implements dynamic extent only");

    T* p_ = nullptr;
    std::size_t n_ = 0;

    template <class C, class U>
    using enable_container_t = std::enable_if_t<
        !is_span<std::decay_t<C>>::value && !std::is_array<C>::value &&
            std::is_convertible_v<decltype(std::declval<C&>().data()), U*>,
        int>;

public:
    using element_type = T;
    using value_type = std::remove_cv_t<T>;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using iterator = T*;

    constexpr span() noexcept = default;
    constexpr span(T* p, std::size_t n) noexcept : p_(p), n_(n) {}
    constexpr span(T* first, T* last) noexcept
        : p_(first), n_(static_cast<std::size_t>(last - first)) {}
    template <std::size_t N>
    constexpr span(T (&a)[N]) noexcept : p_(a), n_(N) {}   // NOLINT: exact `admiral::span` shape
    template <class U, std::size_t N,
              std::enable_if_t<std::is_convertible_v<U (*)[], T (*)[]>, int> = 0>
    constexpr span(std::array<U, N>& a) noexcept : p_(a.data()), n_(N) {}
    template <class U, std::size_t N,
              std::enable_if_t<std::is_convertible_v<const U (*)[], T (*)[]>, int> = 0>
    constexpr span(const std::array<U, N>& a) noexcept : p_(a.data()), n_(N) {}
    template <class C, enable_container_t<C, T> = 0>
    constexpr span(C& c) noexcept : p_(c.data()), n_(c.size()) {}
    template <class C, class D = decltype(std::declval<const C&>().data()),
              std::enable_if_t<!is_span<std::decay_t<C>>::value && !std::is_array<C>::value &&
                                   std::is_convertible_v<D, T*>,
                               int> = 0>
    constexpr span(const C& c) noexcept : p_(c.data()), n_(c.size()) {}
    template <class U, std::enable_if_t<std::is_convertible_v<U (*)[], T (*)[]>, int> = 0>
    constexpr span(const span<U, dynamic_extent>& o) noexcept : p_(o.data()), n_(o.size()) {}

    constexpr span(const span&) noexcept = default;
    constexpr span& operator=(const span&) noexcept = default;

    constexpr iterator begin() const noexcept { return p_; }
    constexpr iterator end() const noexcept { return p_ + n_; }
    constexpr T* data() const noexcept { return p_; }
    constexpr std::size_t size() const noexcept { return n_; }
    constexpr std::size_t size_bytes() const noexcept { return n_ * sizeof(T); }
    constexpr bool empty() const noexcept { return n_ == 0; }
    constexpr T& back() const noexcept { return p_[n_ - 1]; }
    constexpr T& operator[](std::size_t i) const noexcept { return p_[i]; }
};

template <class T, std::size_t N>
span(T (&)[N]) -> span<T>;
template <class T, std::size_t N>
span(std::array<T, N>&) -> span<T>;
template <class T, std::size_t N>
span(const std::array<T, N>&) -> span<const T>;
template <class T>
span(T*, std::size_t) -> span<T>;
template <class C>
span(C&) -> span<std::remove_reference_t<decltype(*std::declval<C&>().data())>>;
#endif

namespace detail {

#if defined(__cpp_lib_bitops) && __cpp_lib_bitops >= 201907L
using std::bit_ceil;
using std::bit_floor;
using std::bit_width;
using std::countr_zero;
using std::has_single_bit;
#else
template <class U>
[[nodiscard]] constexpr std::enable_if_t<std::is_unsigned_v<U>, int>
bit_width(U x) noexcept {
    if (x == 0) return 0;
#if defined(__GNUC__) || defined(__clang__)
    if constexpr (sizeof(U) <= sizeof(unsigned int)) return 32 - __builtin_clz(x);
    else if constexpr (sizeof(U) <= sizeof(unsigned long)) return 64 - __builtin_clzl(x);
    else return 64 - __builtin_clzll(x);
#else
    int w = 1;
    while (x >>= 1) ++w;
    return w;
#endif
}

template <class U>
[[nodiscard]] constexpr std::enable_if_t<std::is_unsigned_v<U>, U>
bit_floor(U x) noexcept {
    return x == 0 ? U{0} : (U{1} << (bit_width(x) - 1));
}

template <class U>
[[nodiscard]] constexpr std::enable_if_t<std::is_unsigned_v<U>, U>
bit_ceil(U x) noexcept {
    if (x <= 1) return 1;
    return U{1} << bit_width(static_cast<U>(x - 1));
}

template <class U>
[[nodiscard]] constexpr std::enable_if_t<std::is_unsigned_v<U>, int>
countr_zero(U x) noexcept {
    if (x == 0) return std::numeric_limits<U>::digits;
#if defined(__GNUC__) || defined(__clang__)
    if constexpr (sizeof(U) <= sizeof(unsigned int)) return __builtin_ctz(x);
    else if constexpr (sizeof(U) <= sizeof(unsigned long)) return __builtin_ctzl(x);
    else return __builtin_ctzll(x);
#else
    int n = 0;
    while ((x & 1) == 0) { x >>= 1; ++n; }
    return n;
#endif
}

template <class U>
[[nodiscard]] constexpr std::enable_if_t<std::is_unsigned_v<U>, bool>
has_single_bit(U x) noexcept {
    return x != 0 && (x & (x - 1)) == 0;
}
#endif

#if defined(__cpp_lib_math_constants) && __cpp_lib_math_constants >= 201907L
namespace numbers = std::numbers;
#else
namespace numbers {
namespace impl {
inline constexpr long double pi_ld = 3.141592653589793238462643383279502884L;
inline constexpr long double sqrt2_ld = 1.414213562373095048801688724209698079L;
inline constexpr long double ln2_ld = 0.693147180559945309417232121458176568L;
inline constexpr long double log2e_ld = 1.442695040888963407359924681001892137L;
}
inline constexpr double pi = static_cast<double>(impl::pi_ld);
inline constexpr double ln2 = static_cast<double>(impl::ln2_ld);
inline constexpr double log2e = static_cast<double>(impl::log2e_ld);
template <class T>
inline constexpr T pi_v = static_cast<T>(impl::pi_ld);
template <class T>
inline constexpr T sqrt2_v = static_cast<T>(impl::sqrt2_ld);
}
#endif

#if defined(__cpp_lib_type_identity) && __cpp_lib_type_identity >= 201806L
using std::type_identity_t;
#else
template <class T>
struct type_identity {
    using type = T;
};
template <class T>
using type_identity_t = typename type_identity<T>::type;
#endif

#if defined(__cpp_lib_remove_cvref) && __cpp_lib_remove_cvref >= 201711L
using std::remove_cvref_t;
#else
template <class T>
using remove_cvref_t = std::remove_cv_t<std::remove_reference_t<T>>;
#endif

#if defined(__cpp_lib_smart_ptr_for_overwrite) && \
    __cpp_lib_smart_ptr_for_overwrite >= 202002L
using std::make_unique_for_overwrite;
#else
template <class T>
[[nodiscard]] std::unique_ptr<T> make_unique_for_overwrite(std::size_t n) {
    static_assert(std::is_array_v<T> && std::extent_v<T> == 0, "admiral: T must be U[]");
    return std::unique_ptr<T>(new std::remove_extent_t<T>[n]);
}
#endif

#if defined(__cpp_lib_integer_comparison_functions) && \
    __cpp_lib_integer_comparison_functions >= 202002L
using std::cmp_less;
#else
template <class A, class B>
[[nodiscard]] constexpr bool cmp_less(A a, B b) noexcept {
    if constexpr (std::is_signed_v<A> == std::is_signed_v<B>) return a < b;
    else if constexpr (std::is_signed_v<A>)
        return a < 0 || static_cast<std::make_unsigned_t<A>>(a) < b;
    else
        return b >= 0 && a < static_cast<std::make_unsigned_t<B>>(b);
}
#endif

template <class T>
inline constexpr bool is_simd_precision_v =
    std::is_same_v<T, float> || std::is_same_v<T, double>;
template <class T>
inline constexpr bool is_precision_v = is_simd_precision_v<T> || std::is_same_v<T, long double>;

template <class It, class V>
[[nodiscard]] constexpr It const_find(It first, It last, const V& v) {
    while (first != last && !(*first == v)) ++first;
    return first;
}

}
}
