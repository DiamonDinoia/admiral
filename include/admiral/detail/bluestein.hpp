#pragma once

// Any N as a chirp-modulated convolution of length >= 2N-1, evaluated by a power-of-two FFT.
// Bluestein, IEEE Trans. AU-18 (1970) 451; Rabiner, Schafer and Rader, ibid. AU-17 (1969) 86.

#include <algorithm>
#include <complex>
#include <cstddef>
#include <optional>
#include <vector>
#include "cxx_compat.hpp"

#include "dif_driver.hpp"
#include "four_step_large.hpp"
#include "base_cost_model.hpp"
#include "math.hpp"
#include "scratch.hpp"
#include "twiddles.hpp"
#include "portable_trig.hpp"

namespace admiral {
namespace detail {

// The Bluestein inner uses the `route_kind` that a standalone plan of the pad length
// and type T picks: the `route_kind::four_step_large` decision of plan.hpp
// (`select_route`), at the serial `nthreads` count the inner uses.
template<typename T>
class plan_impl;

template<typename T>
[[nodiscard]] bool bluestein_inner_six_step_admits(std::size_t pad) {
    return plan_impl<T>::large_route_admits(pad, 1) &&
           four_step_large_fused_shape<T>(pad);
}

// Pick the Bluestein inner pad. The planner defaults to the first smooth pad
// (bluestein_choose_pad). This also weighs the power-of-two pad and adopts it when the
// planner's iterative-DIF cost ranks the power of two lower. Only sizes where a power of
// two genuinely wins change pad; every other size keeps its default smooth pad.
template<typename T>
[[nodiscard]] std::size_t bluestein_pick_pad(std::size_t size) {
    const std::size_t base = bluestein_choose_pad(size);
    const std::size_t pow2 = bit_ceil(2 * size - 1);
    if (pow2 == base) return base;
    const double base_cost = base_model::score<base_model::iterative_dif, T>(base);
    const double pow2_cost = base_model::score<base_model::iterative_dif, T>(pow2);
    return pow2_cost < base_cost ? pow2 : base;
}

template<typename T>
class bluestein_plan {
    struct M {
        std::size_t size;
        bool is_forward;
        std::size_t padded_size;
        std::vector<std::complex<T>> chirp;
        std::vector<std::complex<T>> kernel_fft;
        dif_twiddle_set<T> bl_dif;
        std::optional<four_step_large_plan<T>> six_fwd, six_inv;
    } m;

public:
    bluestein_plan(std::size_t size, bool is_forward)
        : m{size,
            is_forward,
            bluestein_pick_pad<T>(size),
            {},
            {},
            {},
            {},
            {}}
    {
        if (!is_codelet_catalog(m.padded_size)) {
            if (bluestein_inner_six_step_admits<T>(m.padded_size)) {
                m.six_fwd.emplace(m.padded_size, true);
                m.six_inv.emplace(m.padded_size, false);
            } else {
                m.bl_dif = build_dif_twiddle_set<T>(m.padded_size);
            }
        }

        m.chirp.resize(m.size);
        // e steps through n * n mod 2N without a divide: the step 2n+1 is below 2N and e
        // stays below 2N, so the sum is under 4N and one conditional subtract wraps it.
        const std::size_t two_n = 2 * m.size;
        std::size_t e = 0;
        portable_trig::turns_memo memo;
        for (std::size_t n = 0; n < m.size; ++n) {
            const auto [s, c] = portable_trig::sincos_turns(e, two_n, m.is_forward, &memo);
            m.chirp[n] = std::complex<T>(static_cast<T>(c), static_cast<T>(s));
            e += 2 * n + 1;
            if (e >= two_n) e -= two_n;
        }

        std::vector<std::complex<T>> kernel(std::max<std::size_t>(m.padded_size, 1));
        kernel[0] = std::conj(m.chirp[0]);
        for (std::size_t n = 1; n < m.size; ++n) {
            kernel[n] = std::conj(m.chirp[n]);
            kernel[m.padded_size - n] = std::conj(m.chirp[n]);
        }

        pad_fft<true>(span(kernel));
        m.kernel_fft = std::move(kernel);
    }

    void execute(const std::complex<T>* in, std::complex<T>* out, T fct) const {
        const std::size_t N = m.size;

        soa_scratch<T, 1> scratch(2 * m.padded_size);
        T* const raw = scratch.buf(0);
        auto* const a = reinterpret_cast<std::complex<T>*>(raw);
        for (std::size_t n = 0; n < N; ++n) {
            a[n] = in[n] * m.chirp[n];
        }
        std::fill(raw + 2 * N, raw + 2 * m.padded_size, T(0));

        const span buf{a, m.padded_size};
        pad_fft<true>(buf);

        for (std::size_t i = 0; i < m.padded_size; ++i) {
            a[i] *= m.kernel_fft[i];
        }

        pad_fft<false>(buf);

        if (fct == T(1)) {
            for (std::size_t n = 0; n < N; ++n) out[n] = a[n] * m.chirp[n];
        } else {
            for (std::size_t n = 0; n < N; ++n) out[n] = a[n] * m.chirp[n] * fct;
        }
    }

private:
    template<bool Forward>
    void pad_fft(span<std::complex<T>> buf) const {
        const std::size_t pad = m.padded_size;
        if (is_codelet_catalog(pad)) {
            codelet_dispatch<T, Forward>(buf.data(), buf.data(), pad);
            if constexpr (!Forward) scale_inplace(buf.data(), pad, T(1) / T(pad));
            return;
        }
        const auto& six = Forward ? m.six_fwd : m.six_inv;
        if (six) {
            six->execute(buf.data(), buf.data(), Forward ? T(1) : T(1) / T(pad));
            return;
        }
        dif_execute_in_place<T>(Forward, buf.data(), buf.data(), pad, m.bl_dif,
                                Forward ? T(1) : T(1) / T(pad));
    }
};

}
}
