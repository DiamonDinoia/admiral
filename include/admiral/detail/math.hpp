#pragma once

// ============================================================================
// Small integer / size-class utilities shared across the FFT detail headers.
// Pure, header-only, no SIMD or twiddle dependencies.
// ============================================================================

#include <array>
#include <bit>
#include <cmath>
#include <complex>
#include <cstddef>
#include <numbers>
#include <type_traits>
#include <utility>

// CMake-generated codelet catalog sizes (shared with the per-N TUs; see src/CodeletCatalog.cmake).
#include <admiral/detail/codelet_max.hpp>

namespace admiral {
namespace detail {

// True iff N >= 1 and every prime factor of N <= 11 (11-smooth).
// iterative DIF and codelet paths handle these without Bluestein.
[[nodiscard]] constexpr bool is_codelet_supported(std::size_t N) {
    if (N == 0) return false;
    for (unsigned p : {2u, 3u, 5u, 7u, 11u}) {
        while (N % p == 0) N /= p;
    }
    return N == 1;
}

// In-place scale of n AoS complex samples by s (single normalization pass).
// The one scale loop shared by every route's inverse normalization.
template<typename T>
void scale_inplace(std::complex<T>* p, std::size_t n, T s) noexcept {
    for (std::size_t i = 0; i < n; ++i) p[i] *= s;
}

// True iff N is in the compiled straight-line codelet catalog.
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

// Largest leaf a four-step split may use. Bounded above by the contiguous
// spill-free catalog range [2..64]; extras beyond the range (e.g. 120) are
// deliberately not four-step leaves. It also bounds codelet_cost_cyc's
// index and four_step_execute's leaf buffer.
inline constexpr std::size_t kFourStepLeafMax = 64;

// Measured forward codelet_dispatch cycles, indexed by leaf size (0/1 and non-catalog
// unused). Per-precision, not one table plus a scale: the per-entry f32/f64 ratio carries
// structure in n that a single scale cannot, and it misprices one precision exactly at the
// sizes that decide routes. It is also the only place recording that a prime leaf is
// rader_apply<P> over kernel<P-1> and so costs far more than a neighbouring composite,
// which is why the routing model and its fitter read it instead of fitting a polynomial.
// Regenerate with `admiral_benchmark --codelet-sweep --prec=both --no-ducc`; take
// per-entry medians across sweeps, one sweep moves an entry too much.
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

// FROZEN calibration reference: the leaf costs the compile-time gates above the model's
// domain (four_step_beats_bluestein, rader_beats_bluestein) were fitted against, together
// with kDifCostPerNLogN and kBluesteinCostPerPadLog. Only their relative scale decides a
// route, so pointing the gates at the refreshed tables above de-calibrates them. Do not
// refresh this without refitting every one of them; the structural fix is to extend the
// fitted model past BASE_MODEL_NMAX and delete the gates.
inline constexpr std::array<double, kFourStepLeafMax + 1> gate_leaf_cyc_ref = {
    0, 0, 13.8, 45.0, 24.4, 62.9, 114.8, 74.7, 92.4, 103.8, 102.0, 102.6, 130.7,
    272.6, 133.1, 136.0, 119.1, 294.6, 193.1, 436.2, 168.0, 177.0, 185.1, 437.1,
    249.9, 182.9, 274.9, 265.8, 217.6, 496.6, 253.5, 629.4, 269.0, 286.4, 335.8,
    293.0, 337.8, 759.8, 470.5, 408.3, 331.7, 765.0, 375.7, 784.6, 388.2, 405.7,
    663.5, 1526.7, 402.8, 398.0, 384.4, 522.2, 431.0, 997.7, 575.4, 491.3, 505.7,
    691.6, 726.3, 1584.7, 500.8, 1124.1, 807.4, 574.8, 457.2};

[[nodiscard]] constexpr double gate_leaf_cyc(std::size_t n) {
    return n <= kFourStepLeafMax && gate_leaf_cyc_ref[n] > 0.0 ? gate_leaf_cyc_ref[n]
                                                               : double(n);
}

// The catalog extras sit above the contiguous range, so they carry (n, f64, f32) rows
// instead of widening two mostly-empty arrays. Each value is the shipped table's scale
// times a measured ratio, keeping it in frame with the 63 leaves the model was fitted on.
//
// A missing row is not a small error: the double(n) fallback under-prices an extra leaf
// by enough that the model elects the codelet at every N reading that entry.
inline constexpr std::array<std::array<double, 3>, 6> codelet_cost_cyc_extra = {{
    //  n      f64      f32
    {  65.0,  452.5,  355.6},
    {  85.0,  487.4,  549.7},
    { 100.0,  409.4,  333.6},
    { 120.0,  463.7,  361.6},
    { 143.0, 1266.4,  700.0},
    { 360.0, 1173.9,  872.0},
}};

// Measured cycles for a catalog leaf, 0 for a size the catalog does not carry.
template<typename T>
[[nodiscard]] constexpr double catalog_leaf_cyc(std::size_t n) {
    const auto& tab = sizeof(T) == 4 ? codelet_cost_cyc_f32 : codelet_cost_cyc_f64;
    if (n <= kFourStepLeafMax) return tab[n];
    for (const auto& e : codelet_cost_cyc_extra)
        if (std::size_t(e[0]) == n) return e[sizeof(T) == 4 ? 2 : 1];
    return 0.0;
}

// Measured leaf cost, falling back to n for a size no table covers. Every price goes
// through here rather than indexing the table: the routing model scores four_step at
// EVERY n it models, and balanced_split(n) hands it {1, n} for a prime, so a direct
// index read runs past the table for all n > 64.
template<typename T>
[[nodiscard]] constexpr double leaf_cost_cyc(std::size_t n) {
    const double cyc = catalog_leaf_cyc<T>(n);
    return cyc > 0.0 ? cyc : double(n);
}

// Every leaf in the dense range is measured in all three tables, so the fallback above is
// unreachable there: every caller gates on is_codelet_catalog first, which is what makes
// routing through leaf_cost_cyc identical to indexing the table.

// The extras cannot join this assert in either direction, because
// ADM_CODELET_EXTRA_SIZES is build-configurable. An unpriced extra falls back.

// Demanding a price for every catalog member breaks a build that adds one (validate.sh's
// catalog arm adds 66); demanding a member for every row breaks the sanitize preset,
// which clears the extras.
static_assert([] {
    for (std::size_t n = 2; n <= kFourStepLeafMax; ++n)
        if (is_codelet_catalog(n) &&
            !(codelet_cost_cyc_f64[n] > 0.0 && codelet_cost_cyc_f32[n] > 0.0 &&
              gate_leaf_cyc_ref[n] > 0.0))
            return false;
    // What the extras CAN be held to: strictly ascending n, above the dense range, both
    // prices positive. That catches a reordered or duplicated row, which is the edit a
    // linear-scan lookup would otherwise answer with whichever row comes first.
    double prev = double(kFourStepLeafMax);
    for (const auto& e : codelet_cost_cyc_extra) {
        if (!(e[0] > prev && e[1] > 0.0 && e[2] > 0.0)) return false;
        prev = e[0];
    }
    return true;
}());

// Scalar gather/twist overhead a four-step split pays on top of its leaves.
inline constexpr double kFourStepOverhead = 1.13;

// Modeled cost: n1 size-n2 codelets + n2 size-n1 codelets. Two callers, two tables, on
// purpose: <T> is the cost model's price, gate_ is the frozen calibration reference the
// compile-time gates were fitted against (see gate_leaf_cyc_ref).
template<typename T>
[[nodiscard]] inline constexpr double four_step_cost(std::size_t n1, std::size_t n2) {
    return kFourStepOverhead *
           (double(n1) * leaf_cost_cyc<T>(n2) + double(n2) * leaf_cost_cyc<T>(n1));
}

[[nodiscard]] inline constexpr double gate_four_step_cost(std::size_t n1, std::size_t n2) {
    return kFourStepOverhead * (double(n1) * gate_leaf_cyc(n2) + double(n2) * gate_leaf_cyc(n1));
}

// The three structural quantities the routing cost model featurises N with. Here for the
// same reason bluestein_choose_pad is: the generated header scores a plan with them and
// tools/fit_cost_model.cpp fits the coefficients with them, so they must agree by sharing
// one definition, not by inspection. W and the register count are runtime arguments, not
// <T>: the fitter pools receipts from many (precision, W, regs) targets in one process,
// and build_id.hpp is not reachable from it.

// Largest prime factor and prime-factor count. The cofactor left after the small primes
// need not be prime (289 = 17^2), so the loop finishes the job. n <= 1 gives {1, 0}.
[[nodiscard]] constexpr std::array<std::size_t, 2> lpf_nfac(std::size_t n) {
    std::size_t m = n, lpf = 1, nfac = 0;
    for (std::size_t d = 2; d * d <= m; ++d)
        while (m % d == 0) { m /= d; ++nfac; lpf = d; }
    if (m > 1) { lpf = m; ++nfac; }
    return {lpf, nfac};
}

// n = n1*n2 with n1 the largest divisor <= sqrt(n): the four_step split. A prime gives
// {1, n}, which the cost model still has to score.
[[nodiscard]] constexpr std::array<std::size_t, 2> balanced_split(std::size_t n) {
    std::size_t n1 = 1;
    for (std::size_t a = 1; a * a <= n; ++a) if (n % a == 0) n1 = a;
    return {n1, n / n1};
}

// Admissible iterative-DIF radices: the first seven are the narrow (16-reg) set, wide
// (32-reg) ISAs enumerate the last four too, one array with a prefix each. Namespace scope,
// not function-local: a constexpr local cannot be `static` before C++23, and without
// static storage gcc re-materializes the table per call rather than indexing one .rodata
// copy.
inline constexpr std::size_t kChainRadices[11] = {2, 3, 4, 5, 7, 8, 11, 9, 15, 16, 32};

// Cheapest sum of ceil((n/r)/W) over admissible radix chains: the lane-work of an
// iterative DIF. Purely structural, with no fitted constant. 0.0 means "no chain reaches n",
// which callers read as such (n > 1 with zero work is not a transform).
//
// A DP over the divisors of n, relaxed FORWARD: reaching divisor d with cost c lets every
// d*r be reached for c + vec(d). Forward because the inner loop then needs no division
// and no search: for a fixed radix the product d*r is ascending in d, so one cursor per
// radix walks the divisor list once.
//
// Not bounded by BASE_MODEL_NMAX: the bluestein feature passes the convolution pad. A
// product past the end of the divisor list is not relaxed, so an argument beyond
// the margin degrades to a larger cost estimate instead of walking off the end.
[[nodiscard]] constexpr double chain_work(std::size_t n, std::size_t w, std::size_t regs) {
    if (n <= 1) return 0.0;
    const std::size_t n_radices = regs >= 32 ? 11u : 7u;
    const auto vec = [w](std::size_t m) { return double((m + w - 1) / w); };
    // Only radices dividing n can divide any divisor of n: filter once, not per divisor.
    std::array<std::size_t, 11> rad{};
    std::array<std::size_t, 11> cur{};   // rad[k]'s monotone cursor into dv
    std::size_t nr = 0;
    for (std::size_t k = 0; k < n_radices; ++k)
        if (n % kChainRadices[k] == 0) rad[nr++] = kChainRadices[k];
    // Divisors ascending without a sort: the small half comes out ascending, and its
    // cofactors come out descending, so reading them back gives the large half ascending.
    std::array<std::size_t, 64> dv{};
    std::array<std::size_t, 32> hi{};
    std::size_t nd = 0, nh = 0;
    for (std::size_t d = 1; d * d <= n; ++d) {
        if (n % d) continue;
        dv[nd++] = d;
        if (d != n / d) hi[nh++] = n / d;
    }
    while (nh > 0 && nd < dv.size()) dv[nd++] = hi[--nh];
    // -1 is "no chain reaches this divisor", and it propagates: a source left at -1 never
    // relaxes anything, so a partial chain is not a chain.
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

// Past these exponents a 3/7-heavy pad loses to the larger bit_ceil it undercuts,
// because the engine's 3/7-heavy inner DIF chains are weak.
inline constexpr unsigned kPadMaxV3 = 4;
inline constexpr unsigned kPadMaxV7 = 2;

// Bluestein convolution length: first {2,3,5,7}-smooth >= 2n-1 whose inner chain is
// decent, capped at bit_ceil. Lives here rather than in bluestein.hpp because it is a
// price, not an engine: the dif-vs-bluestein gate, the routing cost model and the model's
// offline fitter all have to featurise the pad the transform will run.
[[nodiscard]] constexpr std::size_t bluestein_choose_pad(std::size_t n) {
    const std::size_t need = 2 * n - 1;
    const std::size_t ceil2 = std::bit_ceil(need);
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

// Bluestein: 3 pow2 FFTs of the pad, fitted per pad*log2(pad).
inline constexpr double kBluesteinCostPerPadLog = 1.53;

// Shared by the four-step and Rader cost gates. Prices the *bit_ceil* phantom, not the
// smoothed pad bluestein_choose_pad runs: both gates' ratios were fitted against THIS
// form, so it and they move together (plan.hpp's dif-vs-blue gate prices the real pad).
// Inside 2..512 the fitted model prices the real pad and is unaffected.
[[nodiscard]] constexpr double bluestein_model_cost(std::size_t N) {
    const std::size_t pad = std::bit_ceil(2 * N - 1);
    return kBluesteinCostPerPadLog * double(pad) * double(std::countr_zero(pad));
}

// ----------------------------------------------------------------------------
// constexpr log2 / exp. <cmath> is not constexpr before C++26, and the routing
// cost model needs log-scale features (and exponentiates its log-cycle score)
// during constant evaluation. Cost-model arithmetic only; twiddles use
// ct_sincos_turn at full precision.
// ----------------------------------------------------------------------------
[[nodiscard]] constexpr double ct_log2(double x) {
    if (!(x > 0.0)) return 0.0;  // features never take log of 0; stay total
    // The series exists for constant evaluation only. At RUNTIME, libm does the same job
    // in one instruction sequence and every cost-model feature vector is log-scale, so
    // plan code must take std::log2.
    if (!std::is_constant_evaluated()) return std::log2(x);
    int e = 0;
    while (x >= 2.0) { x *= 0.5; ++e; }
    while (x < 1.0) { x *= 2.0; --e; }
    // ln(x) = 2*atanh(t), t = (x-1)/(x+1); |t| <= 1/3 on [1,2) so t^2 <= 1/9
    // and t^33/33 < 1e-17, converged well past double for any x.
    const double t = (x - 1.0) / (x + 1.0);
    const double t2 = t * t;
    double term = t, sum = t;
    for (int k = 3; k <= 33; k += 2) {
        term *= t2;
        sum += term / double(k);
    }
    return double(e) + 2.0 * sum * std::numbers::log2e;
}

[[nodiscard]] constexpr double ct_exp(double x) {
    constexpr double ln2 = std::numbers::ln2;
    if (x > 700.0) return 1e308;   // cost scores only; saturate instead of inf
    if (x < -700.0) return 0.0;
    if (!std::is_constant_evaluated()) return std::exp(x);
    const double kf = x / ln2;
    const auto k = static_cast<long long>(kf >= 0.0 ? kf + 0.5 : kf - 0.5);
    const double r = x - double(k) * ln2;  // |r| <= ln2/2
    double term = 1.0, sum = 1.0;
    for (int i = 1; i <= 16; ++i) {
        term *= r / double(i);
        sum += term;
    }
    double p = 1.0;
    for (long long i = 0; i < (k < 0 ? -k : k); ++i) p *= (k < 0 ? 0.5 : 2.0);
    return sum * p;
}

// ============================================================================
// Codelet catalog dispatch.
//
// Catalog sizes have twiddles baked into .rodata (prime N = straight-line DFT).
// Heavy kernel<N> bodies (radix-31/61 ≈ P² ops, ×4 {f32,f64}×{fwd,inv}) are
// compiled ONCE into admiral_codelets, one TU per N; consumers see only the
// codelet_dispatch declaration + extern-template, instantiating nothing heavy.
// (Definition: src/codelet_apply.hpp, codelet_instance.cpp.in, codelet_dispatch.cpp.in.)
// ============================================================================

// AoS DFT of a catalog size. in==out: in-place; in!=out: reads `in`, writes `out`.
// UN-normalized: caller applies 1/N for inverse. Defined in admiral_codelets; declared-only here.
template<typename T, bool Forward>
void codelet_dispatch(const std::complex<T>* in, std::complex<T>* out, std::size_t N);

extern template void codelet_dispatch<float,  true >(const std::complex<float>*,  std::complex<float>*,  std::size_t);
extern template void codelet_dispatch<float,  false>(const std::complex<float>*,  std::complex<float>*,  std::size_t);
extern template void codelet_dispatch<double, true >(const std::complex<double>*, std::complex<double>*, std::size_t);
extern template void codelet_dispatch<double, false>(const std::complex<double>*, std::complex<double>*, std::size_t);

// `nlines` in-place AoS DFTs of a catalog size at uniform `stride`, with SIMD lanes
// carrying LINES rather than elements, the layout that fills the vector units a
// single small-N codelet cannot. Why a run prefers it over any per-line route:
// plan_impl::execute_many. How the tile works: src/codelet_apply.hpp.
// SCALED, unlike codelet_dispatch: `fct` folds into the output interleave, which
// costs nothing and saves the run a second pass over the data.
template<typename T, bool Forward>
void codelet_dispatch_many(std::complex<T>* data, std::size_t nlines, std::size_t stride,
                           std::size_t N, T fct);

extern template void codelet_dispatch_many<float,  true >(std::complex<float>*,  std::size_t, std::size_t, std::size_t, float);
extern template void codelet_dispatch_many<float,  false>(std::complex<float>*,  std::size_t, std::size_t, std::size_t, float);
extern template void codelet_dispatch_many<double, true >(std::complex<double>*, std::size_t, std::size_t, std::size_t, double);
extern template void codelet_dispatch_many<double, false>(std::complex<double>*, std::size_t, std::size_t, std::size_t, double);

} // namespace detail
} // namespace admiral

