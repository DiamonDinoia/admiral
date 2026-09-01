#pragma once

// Prime N as a cyclic convolution of length N-1 over the multiplicative group mod N.
// Rader, Proc. IEEE 56 (1968) 1107: DFTs when the number of data samples is prime.

#include <cmath>
#include <complex>
#include <cstddef>
#include <vector>
#include "cxx_compat.hpp"

#include "ct_math.hpp"
#include "dif_driver.hpp"
#include "four_step.hpp"
#include "math.hpp"
#include "portable_trig.hpp"
#include "scratch.hpp"
#include "twiddles.hpp"

namespace admiral {
namespace detail {

enum class rader_inner_kind { codelet, iterative_dif, four_step };

[[nodiscard]] inline bool rader_inner_supported(std::size_t L) {
    return is_codelet_catalog(L) || detail::has_single_bit(L) || is_codelet_supported(L)
           || four_step_supported(L);
}

[[nodiscard]] inline bool rader_supported(std::size_t p) {
    if (is_codelet_catalog(p)) return false;
    if (!ct_is_prime(p)) return false;
    return rader_inner_supported(p - 1);
}

inline constexpr double kDifCostPerNLogN = 0.95;

inline constexpr double kRaderCostPerElement = 17.0;

[[nodiscard]] inline double dif_model_cost(std::size_t N) {
    return kDifCostPerNLogN * double(N) * std::log2(double(N));
}

[[nodiscard]] inline double estimated_plan_cost(std::size_t N);

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

inline constexpr double kRaderGateF32 = 1.0;
inline constexpr double kRaderGateF64Wide = 1.06;
inline constexpr double kRaderGateF64Narrow = 1.30;

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
        for (std::size_t q = 0; q < m.L; ++q) m.ginvpow[q] = m.gpow[(m.L - q) % m.L];

        m.inner = pick_inner(m.L);
        if (m.inner == rader_inner_kind::iterative_dif) {
            m.inner_tw = build_dif_twiddle_set<T>(m.L, nullptr);
        } else if (m.inner == rader_inner_kind::four_step) {
            m.inner_split = choose_four_step_split_exec<T>(m.L);
            m.inner_fs_fwd = build_four_step_twiddles<T, true>(m.inner_split.n1, m.inner_split.n2);
            m.inner_fs_inv = build_four_step_twiddles<T, false>(m.inner_split.n1, m.inner_split.n2);
        }

        std::vector<std::complex<T>> bp(m.L);
        for (std::size_t j = 0; j < m.L; ++j) {
            const std::size_t ee = m.ginvpow[j];
            const auto [sn, cs] = portable_trig::sincos_turns(ee, p, is_forward);
            bp[j] = std::complex<T>(static_cast<T>(cs), static_cast<T>(sn));
        }
        run_inner<true>(bp.data());
        m.bhat = std::move(bp);
    }

    void execute(const std::complex<T>* in, std::complex<T>* out) const {
        const std::complex<T> x0 = in[0];
        std::complex<T> sum = x0;

        soa_scratch<T, 1> scratch(2 * m.L);
        auto* const a = reinterpret_cast<std::complex<T>*>(scratch.buf(0));
        for (std::size_t q = 0; q < m.L; ++q) {
            a[q] = in[m.gpow[q]];
            sum += a[q];
        }

        run_inner<true>(a);
        for (std::size_t k = 0; k < m.L; ++k)
            a[k] *= m.bhat[k];
        run_inner<false>(a);

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
        dif_twiddle_set<T> inner_tw;
        four_step_split inner_split{};
        std::vector<std::complex<T>> inner_fs_fwd, inner_fs_inv;
    } m;

    static rader_inner_kind pick_inner(std::size_t L) {
        if (is_codelet_catalog(L)) return rader_inner_kind::codelet;
        if (detail::has_single_bit(L) || is_codelet_supported(L)) return rader_inner_kind::iterative_dif;
        return rader_inner_kind::four_step;
    }

    template<bool Forward>
    void run_inner(std::complex<T>* buf) const {
        if (m.inner == rader_inner_kind::codelet) {
            codelet_dispatch<T, Forward>(buf, buf, m.L);
            if constexpr (!Forward) scale_inplace(buf, m.L, T(1) / T(m.L));
        } else if (m.inner == rader_inner_kind::iterative_dif) {
            dif_execute_in_place<T>(Forward, buf, buf, m.L, m.inner_tw,
                                    Forward ? T(1) : T(1) / T(m.L));
        } else {
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

}
}
