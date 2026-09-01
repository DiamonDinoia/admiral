#pragma once

#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <type_traits>
#include <utility>

#include <admiral/detail/codelet_max.hpp>
#include "cxx_compat.hpp"

namespace admiral {
namespace detail {

[[nodiscard]] constexpr bool is_codelet_supported(std::size_t N) {
    if (N == 0) return false;
    for (unsigned p : {2u, 3u, 5u, 7u, 11u}) {
        while (N % p == 0) N /= p;
    }
    return N == 1;
}

template<typename T>
void scale_inplace(std::complex<T>* p, std::size_t n, T s) noexcept {
    for (std::size_t i = 0; i < n; ++i) p[i] *= s;
}

[[nodiscard]] constexpr bool is_codelet_catalog(std::size_t N) {
    for (const std::size_t n : CODELET_CATALOG_SIZES) {
        if (N == n) return true;
    }
    return false;
}

template<std::size_t... N>
[[nodiscard]] constexpr bool in_seq(std::integer_sequence<std::size_t, N...>, std::size_t v) {
    return ((v == N) || ...);
}

inline constexpr std::size_t kFourStepLeafMax = 64;

// Measured cycles for one codelet call at length n, indexed by n. Absolute, not relative weights:
// codelet-versus-iterative_dif turns on their magnitude, so never import another host's numbers.
inline constexpr std::array<double, kFourStepLeafMax + 1> codelet_cost_cyc_f64 = {
    0, 0, 11.3, 13.6, 14.8, 52.9, 20.9, 63.0, 37.7, 101.4, 57.9, 114.2, 118.7, 134.6, 71.7,
    141.1, 40.4, 210.4, 114.3, 327.3, 87.5, 193.3, 127.3, 387.9, 173.8, 197.8, 149.5, 219.7,
    135.8, 384.5, 162.0, 443.9, 173.7, 202.6, 250.3, 202.9, 162.5, 424.4, 380.9, 259.2, 218.4,
    528.8, 217.9, 553.9, 219.0, 287.6, 455.9, 1047.5, 157.9, 243.4, 254.3, 438.9, 245.5,
    683.5, 275.7, 310.9, 240.6, 508.0, 526.2, 1286.5, 223.0, 646.7, 640.7, 313.0, 182.5};

inline constexpr std::array<double, kFourStepLeafMax + 1> codelet_cost_cyc_f32 = {
    0, 0, 12.9, 13.0, 14.8, 43.7, 19.2, 49.4, 32.3, 93.7, 97.9, 126.2, 61.0, 152.2, 129.5,
    105.1, 33.9, 233.9, 169.1, 425.4, 75.4, 165.3, 252.9, 578.2, 185.7, 205.1, 245.5, 158.4,
    140.0, 408.2, 236.4, 637.5, 115.5, 194.2, 455.4, 235.8, 146.2, 430.9, 832.8, 210.3, 138.8,
    514.8, 361.3, 808.5, 176.5, 230.1, 1104.2, 2294.3, 133.0, 280.2, 397.2, 301.9, 172.2, 685.0,
    186.9, 241.7, 216.4, 479.2, 785.9, 1744.2, 214.1, 682.4, 1272.3, 276.1, 137.5};

// Frozen leaf cycles the compile-time N > 512 gates were fitted against. Refreshing it
// de-calibrates that band; it is deliberately not the live table above.
inline constexpr std::array<double, kFourStepLeafMax + 1> gate_leaf_cyc_ref = {
    0, 0, 13.8, 45.0, 24.4, 62.9, 114.8, 74.7, 92.4, 103.8, 102.0, 102.6, 130.7,
    272.6, 133.1, 136.0, 119.1, 294.6, 193.1, 436.2, 168.0, 177.0, 185.1, 437.1,
    249.9, 182.9, 274.9, 265.8, 217.6, 496.6, 253.5, 629.4, 269.0, 286.4, 335.8,
    293.0, 337.8, 759.8, 470.5, 408.3, 331.7, 765.0, 375.7, 784.6, 388.2, 405.7,
    663.5, 1526.7, 402.8, 398.0, 384.4, 522.2, 431.0, 997.7, 575.4, 491.3, 505.7,
    691.6, 726.3, 1584.7, 500.8, 1124.1, 807.4, 574.8, 457.2};

// Leaf cycles for the gates: the frozen table where it has an entry, else n as a placeholder.
[[nodiscard]] constexpr double gate_leaf_cyc(std::size_t n) {
    return n <= kFourStepLeafMax && gate_leaf_cyc_ref[n] > 0.0 ? gate_leaf_cyc_ref[n]
                                                               : double(n);
}

// {n, f64 cycles, f32 cycles} for the catalog lengths past kFourStepLeafMax.
inline constexpr std::array<std::array<double, 3>, 6> codelet_cost_cyc_extra = {{
    {  65.0,  452.5,  355.6},
    {  85.0,  487.4,  549.7},
    { 100.0,  409.4,  333.6},
    { 120.0,  463.7,  361.6},
    { 143.0, 1266.4,  700.0},
    { 360.0, 1173.9,  872.0},
}};

template<typename T>
// Measured cycles for the codelet at n, or 0 when n is not in the catalog.
[[nodiscard]] constexpr double catalog_leaf_cyc(std::size_t n) {
    const auto& tab = sizeof(T) == 4 ? codelet_cost_cyc_f32 : codelet_cost_cyc_f64;
    if (n <= kFourStepLeafMax) return tab[n];
    for (const auto& e : codelet_cost_cyc_extra)
        if (std::size_t(e[0]) == n) return e[sizeof(T) == 4 ? 2 : 1];
    return 0.0;
}

template<typename T>
// catalog_leaf_cyc with n itself as the fallback price for a non-catalog length.
[[nodiscard]] constexpr double leaf_cost_cyc(std::size_t n) {
    const double cyc = catalog_leaf_cyc<T>(n);
    return cyc > 0.0 ? cyc : double(n);
}

template<typename T>
// Cycles per element per octave of log2 n, read off the length-64 codelet.
[[nodiscard]] constexpr double stream_octave_cyc() {
    return (sizeof(T) == 4 ? codelet_cost_cyc_f32 : codelet_cost_cyc_f64)[kFourStepLeafMax] /
           (double(kFourStepLeafMax) * 6.0);
}

template<typename T>
// Cycles for one transform of length n: the measured leaf below the catalog top, else
// n log2(n) at the octave rate. This is the W the auto-thread law minimises.
[[nodiscard]] inline double line_work_cyc(std::size_t n) {
    if (n <= 1) return 0.0;
    if (n <= kFourStepLeafMax) {
        if (const double c = catalog_leaf_cyc<T>(n); c > 0.0) return c;
    }
    return double(n) * std::log2(double(n)) * stream_octave_cyc<T>();
}

static_assert([] {
    for (std::size_t n = 2; n <= kFourStepLeafMax; ++n)
        if (is_codelet_catalog(n) &&
            !(codelet_cost_cyc_f64[n] > 0.0 && codelet_cost_cyc_f32[n] > 0.0 &&
              gate_leaf_cyc_ref[n] > 0.0))
            return false;
    double prev = double(kFourStepLeafMax);
    for (const auto& e : codelet_cost_cyc_extra) {
        if (!(e[0] > prev && e[1] > 0.0 && e[2] > 0.0)) return false;
        prev = e[0];
    }
    return true;
}());

inline constexpr double kFourStepOverhead = 1.13;

template<typename T>
// Cycles for four_step at N = n1*n2: n1 leaves of length n2 plus n2 of length n1, times the
// transpose and twist overhead.
[[nodiscard]] inline constexpr double four_step_cost(std::size_t n1, std::size_t n2) {
    return kFourStepOverhead *
           (double(n1) * leaf_cost_cyc<T>(n2) + double(n2) * leaf_cost_cyc<T>(n1));
}

// four_step_cost against the frozen gate table instead of the live leaf tables.
[[nodiscard]] inline constexpr double gate_four_step_cost(std::size_t n1, std::size_t n2) {
    return kFourStepOverhead * (double(n1) * gate_leaf_cyc(n2) + double(n2) * gate_leaf_cyc(n1));
}

// {largest prime factor of n, count of prime factors with multiplicity}.
[[nodiscard]] constexpr std::array<std::size_t, 2> lpf_nfac(std::size_t n) {
    std::size_t m = n, lpf = 1, nfac = 0;
    for (std::size_t d = 2; d * d <= m; ++d)
        while (m % d == 0) { m /= d; ++nfac; lpf = d; }
    if (m > 1) { lpf = m; ++nfac; }
    return {lpf, nfac};
}

// The factor pair n1*n2 == n with n1 <= n2 and n1 as large as it can be.
[[nodiscard]] constexpr std::array<std::size_t, 2> balanced_split(std::size_t n) {
    std::size_t n1 = 1;
    for (std::size_t a = 1; a * a <= n; ++a) if (n % a == 0) n1 = a;
    return {n1, n / n1};
}

inline constexpr std::size_t kChainRadices[11] = {2, 3, 4, 5, 7, 8, 11, 9, 15, 16, 32};

// Vector-lane work of the cheapest DIF radix chain for n: a shortest path over n's divisors,
// each pass priced ceil(divisor/w) lanes. regs >= 32 unlocks radices 9, 15, 16 and 32.
[[nodiscard]] constexpr double chain_work(std::size_t n, std::size_t w, std::size_t regs) {
    if (n <= 1) return 0.0;
    const std::size_t n_radices = regs >= 32 ? 11u : 7u;
    const auto vec = [w](std::size_t m) { return double((m + w - 1) / w); };
    std::array<std::size_t, 11> rad{};
    std::array<std::size_t, 11> cur{};
    std::size_t nr = 0;
    for (std::size_t k = 0; k < n_radices; ++k)
        if (n % kChainRadices[k] == 0) rad[nr++] = kChainRadices[k];
    std::array<std::size_t, 64> dv{};
    std::array<std::size_t, 32> hi{};
    std::size_t nd = 0, nh = 0;
    for (std::size_t d = 1; d * d <= n; ++d) {
        if (n % d) continue;
        dv[nd++] = d;
        if (d != n / d) hi[nh++] = n / d;
    }
    while (nh > 0 && nd < dv.size()) dv[nd++] = hi[--nh];
    std::array<double, 64> work{};
    for (std::size_t i = 1; i < nd; ++i) work[i] = -1.0;
    for (std::size_t j = 0; j < nd; ++j) {
        if (work[j] < 0.0) continue;
        const double reach = vec(dv[j]) + work[j];
        for (std::size_t k = 0; k < nr; ++k) {
            const std::size_t p = dv[j] * rad[k];
            std::size_t idx = cur[k];
            while (idx < nd && dv[idx] < p) ++idx;
            cur[k] = idx;
            if (idx >= nd || dv[idx] != p) continue;
            if (work[idx] < 0.0 || reach < work[idx]) work[idx] = reach;
        }
    }
    return work[nd - 1] < 0.0 ? 0.0 : work[nd - 1];
}

inline constexpr unsigned kPadMaxV3 = 4;
inline constexpr unsigned kPadMaxV7 = 2;

// Smallest 7-smooth pad >= 2n-1 within the 3^4 and 7^2 caps, else the power of two above it.
[[nodiscard]] constexpr std::size_t bluestein_choose_pad(std::size_t n) {
    const std::size_t need = 2 * n - 1;
    const std::size_t ceil2 = detail::bit_ceil(need);
    for (std::size_t m = need; m < ceil2; ++m) {
        std::size_t q = m;
        unsigned v3 = 0, v7 = 0;
        while (q % 2 == 0) q /= 2;
        while (q % 3 == 0) { q /= 3; ++v3; }
        while (q % 5 == 0) q /= 5;
        while (q % 7 == 0) { q /= 7; ++v7; }
        if (q == 1 && v3 <= kPadMaxV3 && v7 <= kPadMaxV7) return m;
    }
    return ceil2;
}

inline constexpr double kBluesteinCostPerPadLog = 1.53;

// Cycles for Bluestein at N, priced pad*log2(pad) on the POWER-OF-TWO pad. base_cost_model's
// bluestein form prices the smooth pad from bluestein_choose_pad instead.
[[nodiscard]] constexpr double bluestein_model_cost(std::size_t N) {
    const std::size_t pad = detail::bit_ceil(2 * N - 1);
    return kBluesteinCostPerPadLog * double(pad) * double(detail::countr_zero(pad));
}

[[nodiscard]] constexpr double ct_log2(double x) {
    if (!(x > 0.0)) return 0.0;
    if (!ADM_IS_CONSTANT_EVALUATED()) return std::log2(x);
    int e = 0;
    while (x >= 2.0) { x *= 0.5; ++e; }
    while (x < 1.0) { x *= 2.0; --e; }
    const double t = (x - 1.0) / (x + 1.0);
    const double t2 = t * t;
    double term = t, sum = t;
    for (int k = 3; k <= 33; k += 2) {
        term *= t2;
        sum += term / double(k);
    }
    return double(e) + 2.0 * sum * detail::numbers::log2e;
}

[[nodiscard]] constexpr double ct_exp(double x) {
    constexpr double ln2 = detail::numbers::ln2;
    if (x > 700.0) return 1e308;
    if (x < -700.0) return 0.0;
    if (!ADM_IS_CONSTANT_EVALUATED()) return std::exp(x);
    const double kf = x / ln2;
    const auto k = static_cast<long long>(kf >= 0.0 ? kf + 0.5 : kf - 0.5);
    const double r = x - double(k) * ln2;
    double term = 1.0, sum = 1.0;
    for (int i = 1; i <= 16; ++i) {
        term *= r / double(i);
        sum += term;
    }
    double p = 1.0;
    for (long long i = 0; i < (k < 0 ? -k : k); ++i) p *= (k < 0 ? 0.5 : 2.0);
    return sum * p;
}

template<typename T, bool Forward>
void codelet_dispatch(const std::complex<T>* in, std::complex<T>* out, std::size_t N);

extern template void codelet_dispatch<float,  true >(const std::complex<float>*,  std::complex<float>*,  std::size_t);
extern template void codelet_dispatch<float,  false>(const std::complex<float>*,  std::complex<float>*,  std::size_t);
extern template void codelet_dispatch<double, true >(const std::complex<double>*, std::complex<double>*, std::size_t);
extern template void codelet_dispatch<double, false>(const std::complex<double>*, std::complex<double>*, std::size_t);

template<typename T, bool Forward>
void codelet_dispatch_many(std::complex<T>* data, std::size_t nlines, std::size_t stride,
                           std::size_t N, T fct);

extern template void codelet_dispatch_many<float,  true >(std::complex<float>*,  std::size_t, std::size_t, std::size_t, float);
extern template void codelet_dispatch_many<float,  false>(std::complex<float>*,  std::size_t, std::size_t, std::size_t, float);
extern template void codelet_dispatch_many<double, true >(std::complex<double>*, std::size_t, std::size_t, std::size_t, double);
extern template void codelet_dispatch_many<double, false>(std::complex<double>*, std::size_t, std::size_t, std::size_t, double);

template<typename T, bool Forward>
void codelet_dispatch_many_oop(const std::complex<T>* in, std::complex<T>* out,
                               std::size_t nlines, std::size_t in_stride,
                               std::size_t out_stride, std::size_t N, T fct);

extern template void codelet_dispatch_many_oop<float,  true >(const std::complex<float>*,  std::complex<float>*,  std::size_t, std::size_t, std::size_t, std::size_t, float);
extern template void codelet_dispatch_many_oop<float,  false>(const std::complex<float>*,  std::complex<float>*,  std::size_t, std::size_t, std::size_t, std::size_t, float);
extern template void codelet_dispatch_many_oop<double, true >(const std::complex<double>*, std::complex<double>*, std::size_t, std::size_t, std::size_t, std::size_t, double);
extern template void codelet_dispatch_many_oop<double, false>(const std::complex<double>*, std::complex<double>*, std::size_t, std::size_t, std::size_t, std::size_t, double);

template<typename T>
void col_codelet_dispatch(bool forward, const std::complex<T>* in, std::size_t in_inner,
                          std::complex<T>* out, std::size_t out_inner, std::size_t ncols,
                          std::size_t N, T scale);

extern template void col_codelet_dispatch<float>(bool, const std::complex<float>*,  std::size_t, std::complex<float>*,  std::size_t, std::size_t, std::size_t, float);
extern template void col_codelet_dispatch<double>(bool, const std::complex<double>*, std::size_t, std::complex<double>*, std::size_t, std::size_t, std::size_t, double);

}
}
