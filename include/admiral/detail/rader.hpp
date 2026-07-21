#pragma once

// ============================================================================
// Rader prime route for isolated primes p > 64 (the catalog tops out at 64, so
// primes above it would otherwise fall to Bluestein). Rader turns a size-p DFT
// (p prime) into a length-(p-1) CYCLIC CONVOLUTION:
//
//   X[0]      = sum_n x[n]
//   X[g^-m]   = x[0] + ( a (*) b )[m],   a[q] = x[g^q],   b[j] = exp(s 2pi i g^-j / p)
//
// where g is a primitive root mod p and (*) is length-L=(p-1) cyclic convolution,
// evaluated by the convolution theorem: c = IDFT_L( DFT_L(a) .* DFT_L(b) ). The
// two length-L transforms are run by the ordinary kernel paths (codelet for
// L<=64, iterative_dif for 11-smooth L, four-step for two-<=64-factor L), so the
// whole prime is a COMPOSITION OF CODELET KERNELS over the composite size p-1 --
// no chirp-z zero-padding. DFT_L(b) is data-independent and precomputed once.
//
// This is the runtime analogue of the compile-time rader_apply<P> in codelet.hpp
// (which uses kernel<P-1>); here P-1 > 64 so the inner transform is a runtime
// plan path instead of a straight-line codelet.
//
// Cost vs Bluestein: Rader runs 2 size-(p-1) FFTs; Bluestein runs 3 FFTs of size
// next_pow2(2p-1) ~ 4p. Since p-1 < 4p and FFT cost grows ~ N log N, Rader is
// essentially always cheaper.
// ============================================================================

#include <bit>
#include <cmath>
#include <complex>
#include <cstddef>
#include <span>
#include <vector>

#include "ct_math.hpp"       // ct_is_prime, ct_powmod, ct_primitive_root
#include "dif_driver.hpp"    // iterative_dif_execute_ws
#include "four_step.hpp"     // four_step_*, choose_four_step_split
#include "math.hpp"          // is_codelet_supported/catalog, codelet_dispatch
#include "portable_trig.hpp" // sincos_turns
#include "scratch.hpp"       // soa_scratch
#include "twiddles.hpp"      // dif_twiddle_set, build_dif_twiddle_set

namespace admiral {
namespace detail {

// How the inner length-L = (p-1) transform is executed. Mirrors select_route's
// first three (non-Bluestein) tiers so Rader never recurses into Bluestein.
enum class rader_inner_kind { codelet, iterative_dif, four_step };

[[nodiscard]] inline bool rader_inner_supported(std::size_t L) {
    return is_codelet_catalog(L) || std::has_single_bit(L) || is_codelet_supported(L)
           || four_step_supported(L);
}

// Rader is available for p iff p is a prime above the codelet catalog whose
// p-1 the inner kernel paths can execute without falling back to Bluestein.
[[nodiscard]] inline bool rader_supported(std::size_t p) {
    if (p <= 64) return false;
    if (!ct_is_prime(static_cast<unsigned>(p))) return false;
    return rader_inner_supported(p - 1);
}

// ----------------------------------------------------------------------------
// Unified recursive cost model (the calibrated planner cost).
// estimated_plan_cost(N) returns the modeled cycle cost of whichever route
// select_route picks for N, recursing for Rader's inner transform. Same
// calibration family as four_step_cost / codelet_cost_cyc. Used to cost-gate the
// Rader route so it is taken only when it actually beats Bluestein (no regression
// on sizes whose p-1 transform is dear, e.g. 79/83/127/191/251/283).
// bluestein_model_cost is the shared Bluestein cost gate (math.hpp).
// ----------------------------------------------------------------------------

// iterative_dif (pow2 / 11-smooth): ~0.95 * N log2 N.
[[nodiscard]] inline double dif_model_cost(std::size_t N) {
    return 0.95 * double(N) * std::log2(double(N));
}

[[nodiscard]] inline double estimated_plan_cost(std::size_t N) {
    if (N <= 1) return 0.0;
    if (N <= 64 && is_codelet_catalog(N)) return codelet_cost_cyc[N];
    if (std::has_single_bit(N) || is_codelet_supported(N)) return dif_model_cost(N);
    const four_step_split s = choose_four_step_split(N);
    if (s.valid() && four_step_cost(s.n1, s.n2) < bluestein_model_cost(N))
        return four_step_cost(s.n1, s.n2);
    if (rader_supported(N))  // 2 inner size-(N-1) transforms + O(N) gather/twist/scatter
        return 2.0 * estimated_plan_cost(N - 1) + 17.0 * double(N);
    return bluestein_model_cost(N);
}

// Cost gate: route a prime p>64 through Rader only when the modeled Rader cost
// (2 inner transforms of size p-1 + gather/pointwise/scatter) beats Bluestein.
[[nodiscard]] inline bool rader_beats_bluestein(std::size_t p) {
    return 2.0 * estimated_plan_cost(p - 1) + 17.0 * double(p) < bluestein_model_cost(p);
}

template<typename T>
class rader_plan {
public:
    rader_plan(std::size_t p, bool is_forward) {
        m.p = p;
        m.L = p - 1;
        const unsigned P = static_cast<unsigned>(p);
        const unsigned g = ct_primitive_root(P);
        const unsigned ginv = ct_powmod(g, P - 2, P);  // g^{-1} mod p

        m.gpow.resize(m.L);
        m.ginvpow.resize(m.L);
        for (std::size_t q = 0; q < m.L; ++q) {
            m.gpow[q] = ct_powmod(g, static_cast<unsigned>(q), P);
            m.ginvpow[q] = ct_powmod(ginv, static_cast<unsigned>(q), P);
        }

        // Inner-transform route + its twiddle tables (size L), built once.
        m.inner = pick_inner(m.L);
        if (m.inner == rader_inner_kind::iterative_dif) {
            m.inner_tw_fwd = build_dif_twiddle_set<T, true>(m.L, nullptr);
            m.inner_tw_inv = build_dif_twiddle_set<T, false>(m.L, nullptr);
        } else if (m.inner == rader_inner_kind::four_step) {
            m.inner_split = choose_four_step_split(m.L);
            m.inner_fs_fwd = build_four_step_twiddles<T, true>(m.inner_split.n1, m.inner_split.n2);
            m.inner_fs_inv = build_four_step_twiddles<T, false>(m.inner_split.n1, m.inner_split.n2);
        }

        // b'[j] = exp(s 2pi i g^{-j}/p), s = -1 (forward) / +1 (inverse); the
        // transform direction is encoded entirely in b's twiddle sign (same
        // convention as the compile-time make_rader_bhat). Bhat = DFT_L(b').
        std::vector<std::complex<T>> bp(m.L);
        for (std::size_t j = 0; j < m.L; ++j) {
            const unsigned ee = ct_powmod(ginv, static_cast<unsigned>(j), P);
            const auto [sn, cs] = portable_trig::sincos_turns(
                static_cast<long long>(ee), static_cast<long long>(P), is_forward);
            bp[j] = std::complex<T>(static_cast<T>(cs), static_cast<T>(sn));
        }
        run_inner<true>(bp.data());  // Bhat = forward DFT of b'
        m.bhat = std::move(bp);
    }

    // in == out is in-place; in != out reads `in` (fully gathered before any
    // write) and writes `out`.
    void execute(const std::complex<T>* in, std::complex<T>* out) const {
        const std::complex<T> x0 = in[0];
        std::complex<T> sum = x0;

        std::vector<std::complex<T>> a(m.L);
        for (std::size_t q = 0; q < m.L; ++q) {
            a[q] = in[m.gpow[q]];
            sum += a[q];
        }

        run_inner<true>(a.data());          // A = DFT_L(a)
        for (std::size_t k = 0; k < m.L; ++k)            // pointwise A .* Bhat
            a[k] *= m.bhat[k];
        run_inner<false>(a.data());          // c = (1/L) IDFT_L(.)

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
        dif_twiddle_set<T> inner_tw_fwd, inner_tw_inv;
        four_step_split inner_split{};
        std::vector<std::complex<T>> inner_fs_fwd, inner_fs_inv;
    } m;

    static rader_inner_kind pick_inner(std::size_t L) {
        if (is_codelet_catalog(L)) return rader_inner_kind::codelet;
        if (std::has_single_bit(L) || is_codelet_supported(L)) return rader_inner_kind::iterative_dif;
        return rader_inner_kind::four_step;
    }

    // Run the length-L inner transform in place. UN-normalized forward; inverse
    // is normalized by 1/L (so c = (1/L) IDFT = cyclic convolution, as Rader needs).
    template<bool Forward>
    void run_inner(std::complex<T>* buf) const {
        if (m.inner == rader_inner_kind::codelet) {
            codelet_dispatch<T, Forward>(buf, buf, m.L);
        } else if (m.inner == rader_inner_kind::iterative_dif) {
            soa_scratch<T, 4> sc(m.L);
            if constexpr (Forward)
                iterative_dif_execute_ws<T, true>(buf, buf, m.L, sc.buf(0), sc.buf(1),
                                                  sc.buf(2), sc.buf(3), m.inner_tw_fwd);
            else
                iterative_dif_execute_ws<T, false>(buf, buf, m.L, sc.buf(0), sc.buf(1),
                                                   sc.buf(2), sc.buf(3), m.inner_tw_inv);
        } else {  // four_step
            std::vector<std::complex<T>> G(m.L);
            if constexpr (Forward)
                four_step_execute<T, true>(buf, buf, m.inner_split.n1, m.inner_split.n2,
                                           m.inner_fs_fwd.data(), G.data());
            else
                four_step_execute<T, false>(buf, buf, m.inner_split.n1, m.inner_split.n2,
                                            m.inner_fs_inv.data(), G.data());
        }
        if constexpr (!Forward) {
            const T scale = T(1) / T(m.L);
            for (std::size_t i = 0; i < m.L; ++i) buf[i] *= scale;
        }
    }
};

} // namespace detail
} // namespace admiral
