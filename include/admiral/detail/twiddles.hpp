#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <ranges>
#include <stdexcept>
#include <utility>
#include <vector>

#include <admiral/errors.hpp>

#include <poet/poet.hpp>
#include "simd.hpp"

#include "cache.hpp"
#include "ct_math.hpp"
#include "cxx_compat.hpp"
#include "math.hpp"
#include "twiddle_row.hpp"
#include "portable_trig.hpp"

namespace admiral {
namespace detail {

struct dif_factor_plan {
    static constexpr std::size_t max_passes = 32;

    std::array<std::size_t, max_passes> radices{};
    std::size_t count = 0;

    constexpr dif_factor_plan() = default;
    constexpr dif_factor_plan(std::initializer_list<std::size_t> rs) {
        for (const std::size_t r : rs) push(r);
    }

    constexpr void push(std::size_t radix) {
        if (count < max_passes) {
            radices[count++] = radix;
        }
    }

    [[nodiscard]] constexpr std::size_t operator[](std::size_t index) const {
        return radices[index];
    }
};

inline constexpr bool dif_wide_radices = poet::vector_register_count() >= 32;
using dif_radix_set = std::conditional_t<
    dif_wide_radices,
    std::integer_sequence<std::size_t, 2, 3, 4, 5, 7, 8, 11, 16, 32, 9, 15, 25, 10>,
    std::integer_sequence<std::size_t, 2, 3, 4, 5, 7, 8, 11>>;

template<std::size_t... Rs>
constexpr auto radix_seq_to_array(std::integer_sequence<std::size_t, Rs...>) {
    return std::array{Rs...};
}
inline constexpr auto dif_candidate_radices = radix_seq_to_array(dif_radix_set{});

using dif_generic_radix_seq =
    std::integer_sequence<std::size_t, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59,
                          61, 67, 71, 73, 79, 83, 89, 97>;
inline constexpr auto dif_generic_radices = radix_seq_to_array(dif_generic_radix_seq{});
[[nodiscard]] constexpr bool dif_is_generic_radix(std::size_t r) {
    const std::size_t* const first = dif_generic_radices.data();
    const std::size_t* const last = first + dif_generic_radices.size();
    return detail::const_find(first, last, r) != last;
}

using dif_ip_radix_set = dif_radix_set;

struct dif_cost_row { double vec, valley, last; };

inline constexpr double kL2BwBytesPerCyc = 12.6;

template<typename T>
[[nodiscard]] constexpr dif_cost_row dif_analytical_cost(std::size_t radix) {
    constexpr std::size_t W = xsimd::batch<T>::size;
    constexpr std::size_t regs = poet::vector_register_count();
    constexpr double bytes_pe = 4.0 * sizeof(T);
    const bool pow2 = detail::has_single_bit(radix);
    const double flops = radix == 2 ? 5 : radix == 3 ? 8 : radix == 4 ? 8.5
                       : radix == 5 ? 12 : radix == 7 ? 16 : radix == 8 ? 12.5
                       : radix == 9 ? 19 : radix == 11 ? 22 : radix == 15 ? 34
                       : radix == 16 ? 17 : radix == 25 ? 24 : 21.5;
    const double r = static_cast<double>(radix);
    const double liv = !pow2 ? r + 9.0
                     : radix <= 8 ? 2.0 * r + 6.0
                     : radix == 16 ? 24.0 : 26.0;
    const double bw = bytes_pe / kL2BwBytesPerCyc;
    const double traffic = bw * (1.0 + 0.056 * (radix > 4 ? r - 4.0 : 0.0));
    const double comp = flops / (0.77 * 4.0 * static_cast<double>(W));
    const double spill = 1.5 * (liv > static_cast<double>(regs) ? liv - static_cast<double>(regs) : 0.0)
                       / static_cast<double>(W);
    const double vec = (comp > traffic ? comp : traffic) + spill;
    const double vall = vec * 1.25;
    const double lastc = (comp > bw ? comp : bw) + 0.5 * spill;
    return {vec, vall, lastc};
}

inline constexpr std::array<std::size_t, 12> dif_cost_radices{2, 3, 4, 5, 7, 8, 11, 16, 32, 9, 15, 25};

[[nodiscard]] constexpr std::size_t dif_cost_index(std::size_t radix) {
    const std::size_t* const first = dif_cost_radices.data();
    return static_cast<std::size_t>(
        const_find(first, first + dif_cost_radices.size(), radix) - first);
}

static_assert(dif_cost_index(25) < dif_cost_radices.size(),
              "dif_cost_radices lost a measured radix");

template<std::size_t Bytes, std::size_t W, std::size_t Regs>
struct dif_cost_table {};

template<> struct dif_cost_table<8, 4, 16> {
    static constexpr dif_cost_row t[12] = {
        {2.23, 2.23, 2.28}, {2.10, 2.10, 2.08}, {2.62, 2.62, 2.39},
        {2.53, 2.53, 2.38}, {3.50, 3.50, 3.04}, {6.10, 6.10, 2.70},
        {4.19, 4.19, 4.26}, {6.01, 6.01, 4.66}, {7.00, 7.00, 5.84},
        {99.0, 99.0, 99.0}, {99.0, 99.0, 99.0}, {99.0, 99.0, 99.0}};
};
template<> struct dif_cost_table<8, 2, 16> {
    static constexpr dif_cost_row t[12] = {
        {2.76, 2.76, 2.43}, {2.18, 2.18, 2.16}, {2.84, 2.84, 2.34},
        {2.73, 2.73, 2.22}, {3.26, 3.26, 3.11}, {9.99, 9.99, 3.84},
        {4.83, 4.83, 4.51}, {10.2, 10.2, 7.77}, {11.2, 11.2, 8.70},
        {99.0, 99.0, 99.0}, {99.0, 99.0, 99.0}, {99.0, 99.0, 99.0}};
};
template<> struct dif_cost_table<4, 8, 16> {
    static constexpr dif_cost_row t[12] = {
        {1.10, 1.79, 1.68}, {1.07, 3.61, 1.68}, {1.33, 3.96, 1.69},
        {1.17, 2.65, 1.27}, {1.52, 5.91, 1.50}, {3.00, 3.23, 1.51},
        {2.05, 9.22, 1.81}, {2.98, 20.1, 2.31}, {3.39, 22.5, 2.47},
        {99.0, 99.0, 99.0}, {99.0, 99.0, 99.0}, {99.0, 99.0, 99.0}};
};
template<> struct dif_cost_table<4, 4, 16> {
    static constexpr dif_cost_row t[12] = {
        {1.42, 1.42, 1.26}, {1.20, 1.20, 1.18}, {1.55, 1.55, 1.06},
        {1.42, 1.42, 1.38}, {1.72, 1.72, 1.84}, {2.60, 2.60, 1.15},
        {2.38, 2.38, 2.46}, {5.07, 5.07, 3.79}, {5.78, 5.78, 3.86},
        {99.0, 99.0, 99.0}, {99.0, 99.0, 99.0}, {99.0, 99.0, 99.0}};
};

#if !ADM_CXX20
template<typename, typename = void>
struct dif_cost_table_has_t : std::false_type {};
template<typename X>
struct dif_cost_table_has_t<X, std::void_t<decltype(X::t)>> : std::true_type {};
template<typename, typename = void>
struct dif_surface_has_c : std::false_type {};
template<typename X>
struct dif_surface_has_c<X, std::void_t<decltype(X::c)>> : std::true_type {};
#endif

template<typename T>
[[nodiscard]] constexpr dif_cost_row dif_measured_cost(std::size_t radix) {
    using table = dif_cost_table<sizeof(T), xsimd::batch<T>::size, poet::vector_register_count()>;
#if ADM_CXX20
    if constexpr (requires { table::t; }) {
#else
    if constexpr (dif_cost_table_has_t<table>::value) {
#endif
        if (dif_cost_index(radix) < dif_cost_radices.size()) return table::t[dif_cost_index(radix)];
        if (const auto s = coprime_split(radix); s.first != 0) {
            const dif_cost_row a = dif_measured_cost<T>(s.first), b = dif_measured_cost<T>(s.second);
            return {a.vec + b.vec, a.valley + b.valley, a.last + b.last};
        }
    }
    return dif_analytical_cost<T>(radix);
}

template<std::size_t Bytes, std::size_t W, std::size_t Regs>
struct dif_surface {};

template<> struct dif_surface<8, 8, 32> {
    static constexpr std::array<double, 9> c{
        0.4132, 0.0185, 7.4518, 1.871, 2.246, 0.0365, 29.3, 0.0324, 0.818};
    static constexpr std::array<double, 12> arith{206, 568, 570, 875, 1465, 1512, 3131, 4610, 10375,
                                                  2246, 4698, 7654};
};
template<> struct dif_surface<4, 16, 32> {
    static constexpr std::array<double, 9> c{
        0.1582, 0.0087, 5.5862, 0.992, 0.441, 0.0272, 24.8, 0.0293, 0.440};
    static constexpr std::array<double, 12> arith{246, 602, 620, 1053, 1764, 1637, 3485, 4812, 10525,
                                                  2444, 5484, 8225};
};

template<typename T>
using dif_surface_t = dif_surface<sizeof(T), xsimd::batch<T>::size, poet::vector_register_count()>;
#if ADM_CXX20
template<typename T>
inline constexpr bool dif_surface_is_analytic = requires { dif_surface_t<T>::c; };
#else
template<typename T>
inline constexpr bool dif_surface_is_analytic = dif_surface_has_c<dif_surface_t<T>>::value;
#endif

template<typename T>
[[nodiscard]] constexpr std::size_t dif_pass_footprint_bytes(std::size_t span,
                                                             std::size_t radix,
                                                             std::size_t ido) {
    return (4u * span + (ido > 1u ? 2u * (radix - 1u) * ido : 0u)) * sizeof(T);
}

template<typename T>
[[nodiscard]] constexpr double cyc_per_pt_l1_resident(std::size_t radix) {
    using S = dif_surface_t<T>;
    return S::c[0] + S::c[1] * std::sqrt(S::arith[dif_cost_index(radix)]);
}

template<typename T>
[[nodiscard]] constexpr double valley_underfill_cyc(std::size_t ido) {
    constexpr std::size_t W = xsimd::batch<T>::size;
    if (ido <= 1 || ido >= W) return 0.0;
    return dif_surface<sizeof(T), xsimd::batch<T>::size,
                       poet::vector_register_count()>::c[2]
           / static_cast<double>(ido);
}

inline constexpr std::size_t kSurfaceL1dBytes = 48u * 1024u;
inline constexpr std::size_t kSurfaceL2Bytes  = 2u * 1024u * 1024u;

template<typename T>
[[nodiscard]] constexpr double footprint_cache_level_cpe(std::size_t radix,
                                                         std::size_t bytes) {
    if (bytes <= kSurfaceL1dBytes) return 0.0;
    if (bytes <= kSurfaceL2Bytes) return dif_surface_t<T>::c[3];
    const std::size_t idx = dif_cost_index(radix);
    return dif_surface_t<T>::c[4]
         + (idx < dif_cost_radices.size()
                ? dif_surface_t<T>::c[5] * std::sqrt(dif_surface_t<T>::arith[idx])
                : 0.0);
}

template<typename T>
[[nodiscard]] constexpr double amortized_launch_cpe(std::size_t span) {
    return dif_surface<sizeof(T), xsimd::batch<T>::size,
                       poet::vector_register_count()>::c[6]
           / static_cast<double>(span);
}

template<typename T>
[[nodiscard]] constexpr double terminal_pass_cpe(std::size_t radix, std::size_t span) {
    using S = dif_surface_t<T>;
    return S::c[8] + S::c[7] * S::arith[dif_cost_index(radix)] / static_cast<double>(span);
}

inline constexpr std::size_t kResidentBytes = 128u * 1024u;

[[nodiscard]] constexpr double dif_footprint_mult(std::size_t radix, std::size_t ido,
                                                  std::size_t side_bytes) {
    if (ido < 512) return 1.0;
    const double m1 = radix >= 32 ? 2.05 : radix >= 16 ? 1.57 : 1.42;
    const double m4 = radix >= 32 ? 2.33 : 1.55;
    constexpr double kRef = double(kResidentBytes), kMid = 1024.0 * 1024.0, kBig = 4.0 * kMid;
    const double b = static_cast<double>(side_bytes);
    if (b <= kRef) return 1.0;
    if (b <= kMid) return 1.0 + (m1 - 1.0) * (b - kRef) / (kMid - kRef);
    if (b <= kBig) return m1 + (m4 - m1) * (b - kMid) / (kBig - kMid);
    return m4;
}

inline constexpr double kValleyPenalty = 1e9;

[[nodiscard]] constexpr double dif_interior_kernel_mult(std::size_t radix) {
    constexpr double regs = static_cast<double>(poet::vector_register_count());
    if (detail::has_single_bit(radix) && radix >= 4) {
        const double live = 2.0 * static_cast<double>(radix);
        return live > regs ? 1.0 + 0.5 * (live - regs) / regs : 1.0;
    }
    return 1.0;
}

template<typename T>
[[nodiscard]] constexpr double dif_placement_mult(std::size_t N, std::size_t ido,
                                                  std::size_t radix) {
    if (ido == 1 || 2 * sizeof(T) * N > kResidentBytes) return 1.0;
    return dif_interior_kernel_mult(radix);
}

[[nodiscard]] constexpr bool is_pentanomial(std::size_t N) {
    while (N % 2u == 0u) N /= 2u;
    while (N % 5u == 0u) N /= 5u;
    return N == 1u;
}

[[nodiscard]] constexpr bool dif_radix_admissible(std::size_t N, std::size_t radix) {
    return radix != 25 || is_pentanomial(N);
}

inline constexpr std::size_t kValleyWideRadix = 15;

template<typename T>
[[nodiscard]] constexpr double dif_valley_penalty(std::size_t ido, std::size_t radix) {
    constexpr std::size_t W = xsimd::batch<T>::size;
    if (ido <= 1 || ido >= W) return 0.0;
    return radix >= kValleyWideRadix ? 2.0 * kValleyPenalty : kValleyPenalty;
}

inline constexpr std::size_t kGenericStarvedTailMinRadix = 47;

template<typename T>
[[nodiscard]] constexpr bool dif_generic_tail_starved(std::size_t g, std::size_t ido) {
    return g >= kGenericStarvedTailMinRadix && ido < xsimd::batch<T>::size;
}

inline constexpr double kConvCyc = 2.891;

template<typename T>
[[nodiscard]] constexpr double generic_prime_cost(std::size_t g) {
    constexpr std::size_t W = xsimd::batch<T>::size;
    std::size_t L = g - 1, sumf = 0, m = L;
    for (std::size_t d = 2; d * d <= m; ++d)
        while (m % d == 0) { sumf += d; m /= d; }
    if (m > 1) sumf += m;
    return kConvCyc * static_cast<double>(L * sumf)
           / (static_cast<double>(g) * static_cast<double>(W));
}
template<typename T>
[[nodiscard]] constexpr double dif_generic_stage_cost(std::size_t N, std::size_t n,
                                                      std::size_t g) {
    const std::size_t ido = n / g;
    if (dif_generic_tail_starved<T>(g, ido))
        return 2.0 * kValleyPenalty * static_cast<double>(N);
    constexpr std::size_t W = xsimd::batch<T>::size;
    const double mask_waste = static_cast<double>((ido + W - 1u) / W * W)
                              / static_cast<double>(ido);
    if constexpr (dif_surface_is_analytic<T>) {
        const std::size_t B = dif_pass_footprint_bytes<T>(N, g, ido);
        const double per_elem = generic_prime_cost<T>(g) * mask_waste
                                + footprint_cache_level_cpe<T>(g, B)
                                + amortized_launch_cpe<T>(N);
        return (per_elem + dif_valley_penalty<T>(ido, g)) * static_cast<double>(N);
    } else {
        return (generic_prime_cost<T>(g) * mask_waste
                  * dif_footprint_mult(g, ido, N * 2u * sizeof(T))
                + dif_valley_penalty<T>(ido, g))
               * static_cast<double>(N);
    }
}

[[nodiscard]] constexpr double dif_order_eps(std::size_t N, std::size_t n, std::size_t radix) {
    if (poet::vector_register_count() >= 32 && n != N) return 0.0;
    return 1e-9 * static_cast<double>(radix) * static_cast<double>(n);
}

template<typename T>
[[nodiscard]] constexpr double dif_stage_cost_tape(std::size_t N, std::size_t n,
                                                   std::size_t radix) {
    const std::size_t ido = n / radix;
    constexpr std::size_t W = xsimd::batch<T>::size;
    const dif_cost_row row = dif_measured_cost<T>(radix);
    const bool pentanomial = is_pentanomial(N);
    constexpr double kValleyLaneCap = 2.0;
    const double lane_mult = static_cast<double>(W) / static_cast<double>(ido);
    const double valley = (!pentanomial || row.valley >= row.vec)
                              ? row.valley
                              : row.vec * (lane_mult < kValleyLaneCap ? lane_mult : kValleyLaneCap);
    const double per_elem = (ido == 1) ? row.last
                          : (ido < W)  ? valley
                                       : row.vec * dif_footprint_mult(radix, ido, N * 2u * sizeof(T));
    const std::size_t tail = (ido > 1) ? (ido % W) : ((N / n) % W);
    const double tail_mult = tail ? 1.0 + 0.025 * static_cast<double>(tail) / static_cast<double>(W)
                                  : 1.0;
    return (per_elem * tail_mult * dif_placement_mult<T>(N, ido, radix)
            + dif_valley_penalty<T>(ido, radix))
               * static_cast<double>(N) + dif_order_eps(N, n, radix);
}

template<typename T>
[[nodiscard]] constexpr double dif_stage_cost(std::size_t N, std::size_t n, std::size_t radix) {
    if constexpr (dif_surface_is_analytic<T>) {
        if (dif_cost_index(radix) >= dif_cost_radices.size()) {
            const auto s = coprime_split(radix);
            if (s.first == 0)
                return dif_stage_cost_tape<T>(N, n, radix);
            const std::size_t ido = n / radix;
            const std::size_t B = dif_pass_footprint_bytes<T>(N, radix, ido);
            const double kernel =
                ido == 1 ? terminal_pass_cpe<T>(s.first, N) + terminal_pass_cpe<T>(s.second, N)
                         : (cyc_per_pt_l1_resident<T>(s.first)
                            + cyc_per_pt_l1_resident<T>(s.second)
                            + valley_underfill_cyc<T>(ido));
            const double per_elem = kernel + footprint_cache_level_cpe<T>(radix, B)
                                  + (ido == 1 ? 0.0 : amortized_launch_cpe<T>(N));
            return (per_elem + dif_valley_penalty<T>(ido, radix)) * static_cast<double>(N)
                 + dif_order_eps(N, n, radix);
        }
        const std::size_t ido = n / radix;
        const std::size_t B = dif_pass_footprint_bytes<T>(N, radix, ido);
        const double kernel =
            ido == 1 ? terminal_pass_cpe<T>(radix, N)
                     : (cyc_per_pt_l1_resident<T>(radix) + valley_underfill_cyc<T>(ido))
                           * (B <= kSurfaceL2Bytes ? dif_interior_kernel_mult(radix) : 1.0);
        const double per_elem =
            kernel + footprint_cache_level_cpe<T>(radix, B)
            + (ido == 1 ? 0.0 : amortized_launch_cpe<T>(N));
        return (per_elem + dif_valley_penalty<T>(ido, radix)) * static_cast<double>(N)
             + dif_order_eps(N, n, radix);
    } else {
        return dif_stage_cost_tape<T>(N, n, radix);
    }
}

template<typename T>
[[nodiscard]] inline double dif_chain_cost(std::size_t N, const dif_factor_plan& p) {
    double      total = 0.0;
    std::size_t n     = N;
    for (std::size_t i = 0; i < p.count; ++i) {
        total += dif_is_generic_radix(p[i]) ? dif_generic_stage_cost<T>(N, n, p[i])
                                           : dif_stage_cost<T>(N, n, p[i]);
        n /= p[i];
    }
    return total;
}

enum class dif_fuse : std::uint8_t { plain = 0, f2head, f2tail, f3head, f3tail };

using dif_fused_pair_set = std::integer_sequence<std::size_t, 4, 5, 8>;

[[nodiscard]] constexpr bool dif_fusable_radix(std::size_t r) noexcept {
    return in_seq(dif_fused_pair_set{}, r);
}

template<typename T>
inline constexpr std::size_t kDifFuseMinN =
    poet::vector_register_count() <= 16 ? (sizeof(T) == 8 ? 2048 : 4096) : 8192;

inline constexpr std::size_t kDifFused3MaxNF64 = 256u * 1024u / (2 * sizeof(double));

template<typename T>
constexpr void dif_fusion_schedule_into(std::size_t N, const std::size_t* radices,
                                        std::size_t n, dif_fuse* sched) {
    constexpr std::size_t W = xsimd::batch<T>::size;
    for (std::size_t i = 0; i < n; ++i) sched[i] = dif_fuse::plain;
    if (N < kDifFuseMinN<T> || n < 3) return;
    const bool fused3_ok = sizeof(T) == 8 && N <= kDifFused3MaxNF64;
    std::size_t l1 = radices[0];
    for (std::size_t p = 1; p + 1 < n; ++p) {
        const std::size_t ip = radices[p];
        const std::size_t ido = N / (l1 * ip);
        if (fused3_ok && p + 3 < n && ip == 4u && radices[p + 1] == 4u
            && radices[p + 2] == 4u && ido % (16u * W) == 0u) {
            sched[p] = dif_fuse::f3head;
            sched[p + 1] = sched[p + 2] = dif_fuse::f3tail;
            l1 *= 64u;
            p += 2;
            continue;
        }
        if (p + 2 < n) {
            const std::size_t r2 = radices[p + 1];
            if (dif_fusable_radix(ip) && dif_fusable_radix(r2) && ido % (r2 * W) == 0u) {
                sched[p] = dif_fuse::f2head;
                sched[p + 1] = dif_fuse::f2tail;
                l1 *= ip * r2;
                p += 1;
                continue;
            }
        }
        l1 *= ip;
    }
}

template<typename T>
[[nodiscard]] inline std::vector<dif_fuse>
dif_fusion_schedule(std::size_t N, const std::vector<std::size_t>& radices) {
    std::vector<dif_fuse> sched(radices.size());
    dif_fusion_schedule_into<T>(N, radices.data(), radices.size(), sched.data());
    return sched;
}

inline constexpr std::size_t kDifFuseDiscountMaxNF64 = 16384;
inline constexpr std::size_t kDifFuseDiscountMaxNF32 = 131072;
inline constexpr double kDifFuseDiscountFactor = 0.40;

template<typename T>
[[nodiscard]] constexpr double dif_fuse_discount(std::size_t N) {
    const std::size_t cap = sizeof(T) == 8 ? kDifFuseDiscountMaxNF64 : kDifFuseDiscountMaxNF32;
    return N <= cap ? kDifFuseDiscountFactor : 1.0;
}

inline constexpr std::size_t kDifCandidates = 16;
inline constexpr std::size_t kDifBeam = kDifCandidates;

[[nodiscard]] constexpr bool dif_same_multiset(const dif_factor_plan& a,
                                               const dif_factor_plan& b) {
    if (a.count != b.count) return false;
    auto x = a.radices, y = b.radices;
    std::sort(x.begin(), x.begin() + static_cast<std::ptrdiff_t>(a.count));
    std::sort(y.begin(), y.begin() + static_cast<std::ptrdiff_t>(b.count));
    return x == y;
}

struct dif_chain_list {
    std::array<dif_factor_plan, kDifCandidates> chain{};
    std::size_t count = 0;
    std::size_t cap = kDifCandidates;

    constexpr void push(const dif_factor_plan& p) {
        if (count < cap && find(p) == count) chain[count++] = p;
    }
    constexpr void offer(const dif_factor_plan& p, double c) {
        if (const std::size_t j = find(p); j < count) {
            if (cost[j] <= c) return;
            for (std::size_t m = j + 1; m < count; ++m) {
                chain[m - 1] = chain[m];
                cost[m - 1] = cost[m];
            }
            --count;
        }
        std::size_t k = count;
        while (k > 0 && c < cost[k - 1]) --k;
        if (k >= cap) return;
        for (std::size_t j = (std::min)(count, cap - 1); j > k; --j) {
            chain[j] = chain[j - 1];
            cost[j] = cost[j - 1];
        }
        chain[k] = p;
        cost[k] = c;
        count = (std::min)(count + 1, cap);
    }
    [[nodiscard]] constexpr const dif_factor_plan& operator[](std::size_t i) const {
        return chain[i];
    }

private:
    [[nodiscard]] constexpr std::size_t find(const dif_factor_plan& p) const {
        std::size_t j = 0;
        while (j < count && !dif_same_multiset(chain[j], p)) ++j;
        return j;
    }
    std::array<double, kDifCandidates> cost{};
};

template<typename T>
[[nodiscard]] constexpr dif_chain_list enumerate_pow2_dif_chains(std::size_t N,
                                                                std::size_t want = kDifCandidates) {
    std::size_t chain[dif_factor_plan::max_passes]{};
    dif_chain_list out;
    out.cap = (std::max)(std::size_t{1}, (std::min)(want, kDifCandidates));
    const double fuse_g = dif_fuse_discount<T>(N);
    const auto score = [&](std::size_t len) {
        dif_fuse sched[dif_factor_plan::max_passes]{};
        dif_fusion_schedule_into<T>(N, chain, len, sched);
        double cost = 0.0;
        std::size_t n = N;
        for (std::size_t i = 0; i < len; ++i) {
            const double pc = dif_stage_cost<T>(N, n, chain[i]);
            cost += (sched[i] == dif_fuse::plain) ? pc : fuse_g * pc;
            n /= chain[i];
        }
        dif_factor_plan p;
        for (std::size_t i = 0; i < len; ++i) p.push(chain[i]);
        out.offer(p, cost);
    };
    auto rec = [&](auto&& self, std::size_t rem, std::size_t len) -> void {
        if (rem == 1) {
            score(len);
            return;
        }
        constexpr std::size_t cands[] = {4, 8, 16, 32};
        constexpr std::size_t n_cands = dif_wide_radices ? 4u : 2u;
        for (std::size_t ci = 0; ci < n_cands; ++ci) {
            const std::size_t r = cands[ci];
            if (rem % r == 0) {
                chain[len] = r;
                self(self, rem / r, len + 1);
            }
        }
    };
    rec(rec, N, 0);
    return out;
}
template<typename T>
[[nodiscard]] constexpr dif_factor_plan enumerate_pow2_dif_plan(std::size_t N) {
    return enumerate_pow2_dif_chains<T>(N, 1)[0];
}

template<typename T>
[[nodiscard]] inline bool dif_chain_shape_ok(std::size_t N, const dif_factor_plan& p) {
    if (p.count == 0) return false;
    std::size_t prefix = 1;
    for (std::size_t i = 0; i < p.count; ++i) {
        const std::size_t r = p[i];
        if (!in_seq(dif_radix_set{}, r)) {
            if (!dif_is_generic_radix(r)) return false;
            if (i == 0 || i + 1 == p.count) return false;
            if (dif_generic_tail_starved<T>(r, N / (prefix * r))) return false;
        }
        prefix *= r;
    }
    return prefix == N;
}

[[nodiscard]] inline std::vector<std::size_t> ascending_divisors(std::size_t n) {
    std::vector<std::size_t> d{1};
    for (std::size_t p = 2; p * p <= n; ++p) {
        if (n % p != 0) continue;
        const std::size_t known = d.size();
        for (std::size_t q = p; n % p == 0; q *= p) {
            n /= p;
            for (std::size_t i = 0; i < known; ++i) d.push_back(d[i] * q);
        }
    }
    if (n > 1) {
        const std::size_t known = d.size();
        for (std::size_t i = 0; i < known; ++i) d.push_back(d[i] * n);
    }
    std::sort(d.begin(), d.end());
    return d;
}

// Namespace scope, not function scope: MSVC 14.51 treats a function-local class carrying
// default member initializers as having no default constructor, so a `std::vector` of it sized
// by count fails inside `std::construct_at` (C2672) and a plain `dif_chain_entry{}` inside a
// lambda fails as C2512. Neither reproduces on a class declared here.
constexpr double kDifUnreachable = 1e300;
struct dif_chain_entry {
    double cost = kDifUnreachable;
    std::size_t radix = 0;
    std::size_t next = 0;
    std::size_t key = 0;
};

template<typename T>
[[nodiscard]] inline dif_chain_list dif_chain_candidates(std::size_t N,
                                                         std::size_t want = kDifCandidates) {
    const std::size_t beam = want <= 1 ? 1 : kDifBeam;
    if (N >= kDifFuseMinN<T> && (N & (N - 1u)) == 0u) return enumerate_pow2_dif_chains<T>(N, want);

    const auto rmix = [](std::size_t r) {
        std::size_t x = r * 0x9E3779B97F4A7C15ull;
        x ^= x >> 31;
        return x * 0xBF58476D1CE4E5B9ull;
    };

    const std::vector<std::size_t> divisors = ascending_divisors(N);
    const auto state = [&divisors](std::size_t n) {
        const auto* const first = divisors.data();
        const auto* const at = std::lower_bound(first, first + divisors.size(), n);
        return static_cast<std::size_t>(at - first);
    };

    std::vector<dif_chain_entry> dp(divisors.size() * beam);
    const auto row = [&dp, beam](std::size_t i) {
        return span<dif_chain_entry>(&dp[i * beam], beam);
    };

    const auto before = [](const dif_chain_entry& a, const dif_chain_entry& b) {
        return a.cost < b.cost || (a.cost == b.cost && a.radix > b.radix);
    };
    const auto sorted_chain = [&row, &state](std::size_t n, dif_chain_entry e) {
        std::array<std::size_t, dif_factor_plan::max_passes> rs{};
        std::size_t c = 0;
        while (n > 1 && c < rs.size()) {
            rs[c++] = e.radix;
            n /= e.radix;
            if (n > 1) e = row(state(n))[e.next];
        }
        std::sort(rs.begin(), rs.begin() + static_cast<std::ptrdiff_t>(c));
        return std::pair{rs, c};
    };
    const auto offer = [beam, &before, &sorted_chain](std::size_t n, span<dif_chain_entry> r,
                                                      const dif_chain_entry e) {
        for (std::size_t m = 0; beam > 1 && m < beam && r[m].cost < kDifUnreachable; ++m) {
            if (r[m].key != e.key || sorted_chain(n, r[m]) != sorted_chain(n, e)) continue;
            if (!before(e, r[m])) return;
            for (std::size_t j = m + 1; j < beam; ++j) r[j - 1] = r[j];
            r[beam - 1] = {};
            break;
        }
        std::size_t k = beam;
        while (k > 0 && before(e, r[k - 1])) --k;
        if (k >= beam) return;
        for (std::size_t j = beam - 1; j > k; --j) r[j] = r[j - 1];
        r[k] = e;
    };

    std::array<std::size_t, dif_candidate_radices.size()> radices{};
    std::size_t nrad = 0;
    for (const std::size_t r : dif_candidate_radices)
        if (N % r == 0 && dif_radix_admissible(N, r)) radices[nrad++] = r;
    std::array<std::size_t, dif_generic_radices.size()> generics{};
    std::size_t ngen = 0;
    for (const std::size_t g : dif_generic_radices)
        if (N % g == 0) generics[ngen++] = g;

    for (std::size_t i = 1; i < divisors.size(); ++i) {
        const std::size_t n = divisors[i];
        const auto expand = [&](const std::size_t radix, const double stage) {
            const std::size_t rest = n / radix;
            if (rest == 1) return offer(n, row(i), {stage, radix, 0, rmix(radix)});
            const span<const dif_chain_entry> sub = row(state(rest));
            for (std::size_t j = 0; j < beam && sub[j].cost < kDifUnreachable; ++j) {
                if (stage + sub[j].cost > row(i)[beam - 1].cost) break;
                offer(n, row(i), {stage + sub[j].cost, radix, j, sub[j].key + rmix(radix)});
            }
        };
        for (std::size_t k = 0; k < nrad; ++k)
            if (n % radices[k] == 0) expand(radices[k], dif_stage_cost<T>(N, n, radices[k]));
        if (n != N)
            for (std::size_t k = 0; k < ngen; ++k)
                if (n % generics[k] == 0 && n / generics[k] != 1)
                    expand(generics[k], dif_generic_stage_cost<T>(N, n, generics[k]));
    }

    const auto rebuild = [&](std::size_t k) {
        dif_factor_plan p;
        for (std::size_t n = N, idx = k; n > 1;) {
            const dif_chain_entry e = row(state(n))[idx];
            if (e.cost >= kDifUnreachable || e.radix == 0 || n % e.radix != 0)
                return dif_factor_plan{};
            p.push(e.radix);
            n /= e.radix;
            idx = e.next;
        }
        return p;
    };

    dif_chain_list out;
    out.cap = (std::max)(std::size_t{1}, (std::min)(want, kDifCandidates));
    for (std::size_t k = 0; k < beam && out.count < out.cap; ++k) {
        const dif_factor_plan p = rebuild(k);
        if (p.count == 0) break;
        out.push(p);
    }
    if (out.count == 0) {
        dif_factor_plan p;
        for (std::size_t n = N; n > 1;) {
            const std::size_t fallback = smallest_radix(n);
            p.push(fallback);
            n /= fallback;
        }
        out.push(p);
    }
    return out;
}

template<typename T>
[[nodiscard]] inline dif_factor_plan dif_elected_chain(std::size_t N) {
    const dif_factor_plan best = dif_chain_candidates<T>(N, 1)[0];
    if (dif_chain_shape_ok<T>(N, best)) return best;
    const dif_chain_list c = dif_chain_candidates<T>(N);
    for (std::size_t i = 0; i < c.count; ++i)
        if (dif_chain_shape_ok<T>(N, c[i])) return c[i];
    return best;
}

template<typename T>
[[nodiscard]] inline dif_factor_plan build_dif_factor_plan(std::size_t N) {
    return dif_chain_candidates<T>(N, 1)[0];
}

template<typename T>
struct dif_rt;

template<typename T>
struct dif_step {
    using fn_t = void (*)(const T*, const T*, T*, T*, const dif_step&, const dif_rt<T>&);
    fn_t fn = nullptr;
    std::size_t p = 0;
    std::size_t l1 = 0;
    std::size_t ido = 0;
    std::size_t n = 0;
    std::uint8_t src = 0, dst = 0, sim = 0, dim = 0, es = 0;
};

template<typename T>
struct dif_tape_pair {
    std::vector<dif_step<T>> blk, flat;
};

template<typename T>
struct dif_twiddle_set {
    std::vector<std::pair<std::vector<T>, std::vector<T>>> passes;
    std::vector<std::size_t> radices;
    std::vector<dif_fuse> sched;
    std::vector<std::vector<T>> packed_pair;
    std::uint32_t ip_mask = 0;
    std::vector<std::uint32_t> rowperm;
    std::size_t p0_block = 0;
    dif_tape_pair<T> tape[2];
};

template<typename T, bool Forward>
void dif_build_tape(dif_twiddle_set<T>& s, std::size_t N);

template<typename T>
[[nodiscard]] dif_twiddle_set<T> build_dif_twiddle_set(std::size_t N,
                                         const dif_factor_plan* override_plan = nullptr,
                                         bool fuse_packed = true) {
    dif_twiddle_set<T> s;
    const dif_factor_plan planned = override_plan ? *override_plan : dif_elected_chain<T>(N);
    if (override_plan)
        for (std::size_t p = 0; p < planned.count; ++p) {
            if (!in_seq(dif_radix_set{}, planned[p]) && !dif_is_generic_radix(planned[p]))
                throw unsupported_error("forced dif: radix is not in dif_radix_set");
            if (dif_is_generic_radix(planned[p]) &&
                (p == 0 || p + 1 == planned.count))
                throw unsupported_error("forced dif: generic radix must be a middle pass");
        }

    std::size_t l1 = 1;
    for (std::size_t pass = 0; pass < planned.count; ++pass) {
        const std::size_t ip = planned[pass];
        const std::size_t ido = N / (l1 * ip);

        s.radices.push_back(ip);

        const std::size_t sz = (ip - 1) * ido;
        std::vector<T> re(sz), im(sz);
        for (std::size_t j = 1; j < ip; ++j)
            geom_twiddle_row<T, true>(j * l1, N, ido,
                                                  re.data() + (j - 1) * ido,
                                                  im.data() + (j - 1) * ido);
        s.passes.emplace_back(std::move(re), std::move(im));

        l1 *= ip;
    }

    if (fuse_packed && !s.radices.empty()) {
        constexpr std::size_t W = xsimd::batch<T>::size;
        const std::size_t ip0 = s.radices[0], ido0 = N / ip0;
        const std::size_t flat = 2 * (ip0 - 1) * ido0 * sizeof(T);
        if (8 * N * sizeof(T) + flat > cpu_cache().l2) {
            std::size_t bw = W;
            while (bw * bw < ido0) bw *= 2;
            const std::size_t nb = (ido0 + bw - 1) / bw;
            std::vector<T> re((ip0 - 1) * (bw + nb)), im(re.size());
            for (std::size_t j = 1; j < ip0; ++j) {
                geom_twiddle_row<T, true>(j, N, bw,
                                                      re.data() + (j - 1) * bw,
                                                      im.data() + (j - 1) * bw);
                geom_twiddle_row<T, true>(j * bw, N, nb,
                                                      re.data() + (ip0 - 1) * bw + (j - 1) * nb,
                                                      im.data() + (ip0 - 1) * bw + (j - 1) * nb);
            }
            s.passes[0] = {std::move(re), std::move(im)};
            s.p0_block = bw;
        }
    }

    s.packed_pair.resize(s.passes.size());
    s.sched = fuse_packed ? dif_fusion_schedule<T>(N, s.radices)
                          : std::vector<dif_fuse>(s.radices.size(), dif_fuse::plain);
    {
        constexpr std::size_t W = xsimd::batch<T>::size;
        std::size_t lp = 1;
        for (std::size_t p = 0; p < s.radices.size(); ++p) {
            const std::size_t ip1 = s.radices[p];
            const std::size_t ido_p = N / (lp * ip1);
            if (s.sched[p] == dif_fuse::f2head) {
                const std::size_t ip2 = s.radices[p + 1u];
                const std::size_t ido2_p = ido_p / ip2;
                const std::size_t sz1 = (ip1 - 1u) * 2u * ido_p;
                const std::size_t sz2 = (ip2 - 1u) * 2u * ido2_p;
                std::vector<T> packed(sz1 + sz2);
                const auto& tw1 = s.passes[p];
                const auto& tw2 = s.passes[p + 1u];
                std::size_t off = 0;
                const std::size_t n_chunks1 = ido_p / W;
                for (std::size_t ac = 0; ac < n_chunks1; ++ac) {
                    for (std::size_t k = 1; k < ip1; ++k) {
                        const std::size_t base = (k - 1) * ido_p + ac * W;
                        for (std::size_t i = 0; i < W; ++i) packed[off++] = tw1.first[base + i];
                        for (std::size_t i = 0; i < W; ++i) packed[off++] = tw1.second[base + i];
                    }
                }
                const std::size_t n_chunks2 = ido2_p / W;
                for (std::size_t ac = 0; ac < n_chunks2; ++ac) {
                    for (std::size_t k2 = 1; k2 < ip2; ++k2) {
                        const std::size_t base2 = (k2 - 1) * ido2_p + ac * W;
                        for (std::size_t i = 0; i < W; ++i) packed[off++] = tw2.first[base2 + i];
                        for (std::size_t i = 0; i < W; ++i) packed[off++] = tw2.second[base2 + i];
                    }
                }
                s.packed_pair[p] = std::move(packed);
                s.passes[p] = {};
                s.passes[p + 1u] = {};
            }
            lp *= ip1;
        }
    }

    if (fuse_packed && s.radices.size() >= 3) {
        constexpr std::size_t kGatherWindowBytes = 256u * 1024u;
        const std::size_t r_last = s.radices.back();
        std::size_t L = s.radices[0];
        std::size_t ip_prefix = 0;
        for (std::size_t p = 1; p + 1 < s.radices.size(); ++p) {
            if (s.sched[p] != dif_fuse::plain) break;
            if (dif_is_generic_radix(s.radices[p])) break;
            if (butterfly_wants_reload(s.radices[p], poet::vector_register_count())) break;
            const std::size_t Ln = L * s.radices[p];
            if (Ln * r_last * 2u * sizeof(T) > kGatherWindowBytes) break;
            L = Ln;
            ip_prefix = p;
        }
        for (std::size_t p = 1; p <= ip_prefix; ++p) s.ip_mask |= 1u << p;

        for (std::size_t p = 1, l1p = s.radices[0]; p + 2 < s.radices.size();
             l1p *= s.radices[p], ++p) {
            const std::size_t ip = s.radices[p];
            if (s.sched[p] == dif_fuse::plain &&
                butterfly_wants_reload(ip, poet::vector_register_count()) &&
                (N / (l1p * ip)) * ip * 2u * sizeof(T) <= kIpTileBytes)
                s.ip_mask |= 1u << p;
        }
    }
    if (s.ip_mask != 0) {
        const std::size_t K = s.radices.size() - 1;
        const std::size_t l1_last = N / s.radices.back();
        s.rowperm.resize(l1_last);
        std::size_t k[dif_factor_plan::max_passes]{};
        for (std::size_t n = 0; n < l1_last; ++n) {
            std::size_t lg = 0, ph = 0, w = 1;
            for (std::size_t i = 0; i < K; ++i) {
                const std::size_t r = s.radices[i];
                lg += w * k[i];
                if (s.ip_mask >> i & 1u)
                    ph = r * ph + (butterfly_wants_reload(r, poet::vector_register_count())
                                       ? (k[i] & 1u) * (r / 2u) + k[i] / 2u
                                       : k[i]);
                else
                    ph += w * k[i];
                w *= r;
            }
            s.rowperm[lg] = static_cast<std::uint32_t>(ph);
            for (std::size_t i = 0; i < K; ++i)
                if (++k[i] < s.radices[i]) break;
                else k[i] = 0;
        }
    }

    if (fuse_packed && !s.radices.empty()) {
        dif_build_tape<T, true>(s, N);
        dif_build_tape<T, false>(s, N);
    }

    return s;
}

}
}
