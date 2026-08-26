#pragma once

// ============================================================================
// Rader prime route for primes the codelet catalog does not cover.
// Turns a size-p DFT into a length-(p-1) cyclic convolution:
//
//   X[0]      = sum_n x[n]
//   X[g^-m]   = x[0] + ( a (*) b )[m],   a[q] = x[g^q],   b[j] = exp(s 2pi i g^-j / p)
//
// g is a primitive root mod p; (*) is length-L=(p-1) cyclic convolution via
// c = IDFT_L( DFT_L(a) .* DFT_L(b) ). DFT_L(b) is precomputed. Inner transforms
// use codelet / iterative_dif / four_step, with no chirp-z zero-padding.
//
// Runtime analogue of compile-time rader_apply<P> in codelet.hpp (uses kernel<P-1>);
// here the inner transform is a runtime plan path, not an instantiated codelet.
// Rader vs Bluestein is gated (rader_beats_bluestein): 2 size-(p-1) FFTs vs 3 FFTs
// of ~4p.
//
// Ref: Rader, "Discrete Fourier transforms when the number of data samples is
// prime", Proc. IEEE 56 (1968) 1107. DOI 10.1109/PROC.1968.6477
// ============================================================================

#include <cmath>
#include <complex>
#include <cstddef>
#include <vector>
#include "cxx_compat.hpp"  // detail::has_single_bit

#include "ct_math.hpp"       // ct_is_prime, ct_powmod, ct_primitive_root
#include "dif_driver.hpp"    // iterative_dif_execute_ws, dif_execute_in_place
#include "four_step.hpp"     // four_step_*, choose_four_step_split
#include "math.hpp"          // is_codelet_supported/catalog, codelet_dispatch
#include "portable_trig.hpp" // sincos_turns
#include "scratch.hpp"       // soa_scratch
#include "twiddles.hpp"      // dif_twiddle_set, build_dif_twiddle_set

namespace admiral {
namespace detail {

// How the inner length-L=(p-1) transform is executed. Mirrors select_route's
// first three (non-Bluestein) tiers so Rader never recurses into Bluestein.
enum class rader_inner_kind { codelet, iterative_dif, four_step };

[[nodiscard]] inline bool rader_inner_supported(std::size_t L) {
    return is_codelet_catalog(L) || detail::has_single_bit(L) || is_codelet_supported(L)
           || four_step_supported(L);
}

// Rader is available for p iff p is a prime above the codelet catalog whose
// p-1 the inner kernel paths can execute without falling back to Bluestein.
[[nodiscard]] inline bool rader_supported(std::size_t p) {
    if (is_codelet_catalog(p)) return false;
    if (!ct_is_prime(p)) return false;
    return rader_inner_supported(p - 1);
}

// ----------------------------------------------------------------------------
// Recursive cost model for the planner.
// estimated_plan_cost(N): modeled cycle cost of select_route's choice for N,
// recursing for Rader's inner transform. Same calibration as four_step_cost /
// codelet_cost_cyc. Gates Rader vs Bluestein to prevent regressions on primes
// with costly p-1.
// bluestein_model_cost: shared gate (math.hpp).
// ----------------------------------------------------------------------------

// iterative_dif cost model (pow2 / 11-smooth), cycles per N log2 N.
inline constexpr double kDifCostPerNLogN = 0.95;

// Rader's O(N) gather/twist/scatter, cycles per element.
inline constexpr double kRaderCostPerElement = 17.0;

[[nodiscard]] inline double dif_model_cost(std::size_t N) {
    return kDifCostPerNLogN * double(N) * std::log2(double(N));
}

[[nodiscard]] inline double estimated_plan_cost(std::size_t N);

// Two inner transforms of length p-1, plus the gather/twist/scatter around them.
[[nodiscard]] inline double rader_model_cost(std::size_t p) {
    return 2.0 * estimated_plan_cost(p - 1) + kRaderCostPerElement * double(p);
}

[[nodiscard]] inline double estimated_plan_cost(std::size_t N) {
    if (N <= 1) return 0.0;
    if (N <= kFourStepLeafMax && is_codelet_catalog(N)) return gate_leaf_cyc(N);
    if (detail::has_single_bit(N) || is_codelet_supported(N)) return dif_model_cost(N);
    const four_step_split s = choose_four_step_split(N);
    if (s.valid()) {
        const double c = gate_four_step_cost(s.n1, s.n2);
        if (c < bluestein_model_cost(N)) return c;
    }
    if (rader_supported(N)) return rader_model_cost(N);
    return bluestein_model_cost(N);
}

// The precision-invariant model over-estimates Rader for f64, keeping primes on Bluestein
// that Rader wins outright, so the gate admits a per-(prec,W) multiple: f32 stays at 1
// because Bluestein's pow2 pad vectorizes well there, f64 loosens, and loosens further at
// narrow W where Bluestein gains least. These and gate_leaf_cyc_ref are one calibrated set.
inline constexpr double kRaderGateF32 = 1.0;
inline constexpr double kRaderGateF64Wide = 1.06;
inline constexpr double kRaderGateF64Narrow = 1.30;

// Route a prime above the codelet catalog through Rader only when it beats Bluestein.
template<typename T>
[[nodiscard]] inline bool rader_beats_bluestein(std::size_t p) {
    constexpr std::size_t W = xsimd::batch<T>::size;
    constexpr double ratio =
        sizeof(T) == 4 ? kRaderGateF32 : (W >= 8 ? kRaderGateF64Wide : kRaderGateF64Narrow);
    return rader_model_cost(p) < ratio * bluestein_model_cost(p);
}

template<typename T>
class rader_plan {
public:
    rader_plan(std::size_t p, bool is_forward) {
        m.p = p;
        m.L = p - 1;
        const std::size_t g = ct_primitive_root(p);

        m.gpow.resize(m.L);
        m.ginvpow.resize(m.L);
        for (std::size_t q = 0; q < m.L; ++q) m.gpow[q] = ct_powmod(g, q, p);
        // g^L == 1, so g^{-q} == g^{L-q}: the inverse powers are a reversal of gpow.
        for (std::size_t q = 0; q < m.L; ++q) m.ginvpow[q] = m.gpow[(m.L - q) % m.L];

        // Build inner-transform route and twiddle tables (size L), once.
        m.inner = pick_inner(m.L);
        if (m.inner == rader_inner_kind::iterative_dif) {
            m.inner_tw = build_dif_twiddle_set<T>(m.L, nullptr);
        } else if (m.inner == rader_inner_kind::four_step) {
            m.inner_split = choose_four_step_split_exec<T>(m.L);
            m.inner_fs_fwd = build_four_step_twiddles<T, true>(m.inner_split.n1, m.inner_split.n2);
            m.inner_fs_inv = build_four_step_twiddles<T, false>(m.inner_split.n1, m.inner_split.n2);
        }

        // b'[j] = exp(s 2pi i g^{-j}/p), s=-1 fwd/+1 inv (same sign convention as
        // compile-time make_rader_bhat). Bhat = forward DFT_L(b').
        std::vector<std::complex<T>> bp(m.L);
        for (std::size_t j = 0; j < m.L; ++j) {
            const std::size_t ee = m.ginvpow[j];  // g^{-j} mod p
            const auto [sn, cs] = portable_trig::sincos_turns(ee, p, is_forward);
            bp[j] = std::complex<T>(static_cast<T>(cs), static_cast<T>(sn));
        }
        run_inner<true>(bp.data());  // Bhat = DFT_L(b')
        m.bhat = std::move(bp);
    }

    // in==out: in-place. in!=out: reads `in` fully before any write.
    void execute(const std::complex<T>* in, std::complex<T>* out) const {
        const std::complex<T> x0 = in[0];
        std::complex<T> sum = x0;

        // Uninitialized: the gather below writes every one of the L entries, so a
        // std::vector's value-init would be a memset that is immediately overwritten.
        soa_scratch<T, 1> scratch(2 * m.L);
        auto* const a = reinterpret_cast<std::complex<T>*>(scratch.buf(0));
        for (std::size_t q = 0; q < m.L; ++q) {
            a[q] = in[m.gpow[q]];
            sum += a[q];
        }

        run_inner<true>(a);                  // A = DFT_L(a)
        for (std::size_t k = 0; k < m.L; ++k)            // pointwise A .* Bhat
            a[k] *= m.bhat[k];
        run_inner<false>(a);                 // c = IDFT_L(.) / L

        out[0] = sum;
        for (std::size_t mm = 0; mm < m.L; ++mm)
            out[m.ginvpow[mm]] = x0 + a[mm];
    }

    [[nodiscard]] std::size_t size() const noexcept { return m.p; }

private:
    struct State {
        std::size_t p = 0, L = 0;
        rader_inner_kind inner = rader_inner_kind::codelet;
        std::vector<std::size_t> gpow, ginvpow;
        std::vector<std::complex<T>> bhat;
        dif_twiddle_set<T> inner_tw;  // direction-free (twiddles.hpp): one set, both ways
        four_step_split inner_split{};
        std::vector<std::complex<T>> inner_fs_fwd, inner_fs_inv;
    } m;

    static rader_inner_kind pick_inner(std::size_t L) {
        if (is_codelet_catalog(L)) return rader_inner_kind::codelet;
        if (detail::has_single_bit(L) || is_codelet_supported(L)) return rader_inner_kind::iterative_dif;
        return rader_inner_kind::four_step;
    }

    // In-place length-L inner transform. Forward: un-normalized. Inverse: scaled by 1/L.
    template<bool Forward>
    void run_inner(std::complex<T>* buf) const {
        // Inverse normalization 1/L: folded into the DIF last pass's store;
        // applied as one scale pass for the codelet/four-step branches (no fold hook).
        if (m.inner == rader_inner_kind::codelet) {
            codelet_dispatch<T, Forward>(buf, buf, m.L);
            if constexpr (!Forward) scale_inplace(buf, m.L, T(1) / T(m.L));
        } else if (m.inner == rader_inner_kind::iterative_dif) {
            dif_execute_in_place<T>(Forward, buf, buf, m.L, m.inner_tw,
                                    Forward ? T(1) : T(1) / T(m.L));
        } else {  // four_step
            std::vector<std::complex<T>> G(m.L);
            if constexpr (Forward)
                four_step_execute<T, true>(buf, buf, m.inner_split.n1, m.inner_split.n2,
                                           m.inner_fs_fwd.data(), G.data());
            else
                four_step_execute<T, false>(buf, buf, m.inner_split.n1, m.inner_split.n2,
                                            m.inner_fs_inv.data(), G.data());
            if constexpr (!Forward) scale_inplace(buf, m.L, T(1) / T(m.L));
        }
    }
};

} // namespace detail
} // namespace admiral
