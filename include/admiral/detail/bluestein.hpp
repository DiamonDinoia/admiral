#pragma once

#include <bit>
#include <complex>
#include <cstddef>
#include <span>
#include <vector>

#include "dif_driver.hpp"   // iterative_dif_execute_ws
#include "math.hpp"         // codelet_dispatch, is_codelet_catalog
#include "scratch.hpp"    // soa_scratch
#include "twiddles.hpp"     // dif_twiddle_set, build_dif_twiddle_set
#include "portable_trig.hpp"  // sincos_turns

namespace admiral {
namespace detail {

// ============================================================================
// Bluestein's algorithm (chirp-z) for arbitrary N (primes / non-smooth sizes).
//
// Self-contained plan: the ctor precomputes the chirp sequence, the transformed
// kernel, and the twiddle tables for the inner padded-pow2 transforms; execute()
// runs the convolution and applies 1/N for the inverse direction. plan_impl
// composes one of these ONLY on the bluestein route, so the common routes pay
// nothing for its five precomputed buffers.
// ============================================================================

template<typename T>
class bluestein_plan {
    // All instance state lives in one internal aggregate `m` (no per-member _
    // suffix). Only built/populated on the bluestein route.
    struct M {
        std::size_t size;
        bool is_forward;
        std::size_t padded_size;
        std::vector<std::complex<T>> chirp;       // Chirp sequence
        std::vector<std::complex<T>> kernel_fft;  // Pre-transformed kernel
        // Twiddles for the padded-size pow2 transforms. The inner convolution is
        // always fwd(a), fwd(kernel), inv(a) regardless of the outer direction,
        // so BOTH are needed. Only built when padded_size exceeds the catalog.
        dif_twiddle_set<T> bl_dif_fwd;
        dif_twiddle_set<T> bl_dif_inv;
    } m;

public:
    bluestein_plan(std::size_t size, bool is_forward)
        : m{.size = size,
            .is_forward = is_forward,
            .padded_size = std::bit_ceil(2 * size - 1),
            .chirp = {},
            .kernel_fft = {},
            .bl_dif_fwd = {},
            .bl_dif_inv = {}}
    {
        // Twiddle tables for the padded pow2 transforms (only when the padded
        // size is above the codelet catalog; the codelet bakes its own twiddles).
        if (!is_codelet_catalog(m.padded_size)) {
            m.bl_dif_fwd = build_dif_twiddle_set<T, true>(m.padded_size);
            m.bl_dif_inv = build_dif_twiddle_set<T, false>(m.padded_size);
        }

        // Pre-compute chirp sequence: b[n] = exp(+/- i*pi*n^2/N) = exp(+/- 2*pi*i * n^2/(2N)).
        m.chirp.resize(m.size);
        for (std::size_t n = 0; n < m.size; ++n) {
            const auto [s, c] = portable_trig::sincos_turns(n * n, 2 * m.size, m.is_forward);
            m.chirp[n] = std::complex<T>(static_cast<T>(c), static_cast<T>(s));
        }

        // Pre-compute and transform the kernel
        std::vector<std::complex<T>> kernel(m.padded_size);
        kernel[0] = std::conj(m.chirp[0]);
        for (std::size_t n = 1; n < m.size; ++n) {
            kernel[n] = std::conj(m.chirp[n]);
            kernel[m.padded_size - n] = std::conj(m.chirp[n]);
        }

        // Transform kernel once (reuse for all executions)
        pow2_fft<true>(std::span(kernel));
        m.kernel_fft = std::move(kernel);
    }

    // Execute Bluestein's algorithm using pre-computed data. `fct` scales the
    // output (folded into the final chirp sweep — no separate normalization
    // pass). in == out is in-place; in != out reads `in` (fully consumed by the
    // chirp multiply) and writes `out`.
    void execute(const std::complex<T>* in, std::complex<T>* out, T fct) const {
        const std::size_t N = m.size;

        // Multiply input by chirp and pad
        std::vector<std::complex<T>> a(m.padded_size);
        for (std::size_t n = 0; n < N; ++n) {
            a[n] = in[n] * m.chirp[n];
        }

        pow2_fft<true>(std::span(a));

        // Pointwise multiply with pre-transformed kernel
        for (std::size_t i = 0; i < m.padded_size; ++i) {
            a[i] *= m.kernel_fft[i];
        }

        pow2_fft<false>(std::span(a));

        // Extract result, multiply by chirp, and fold in the output scale.
        if (fct == T(1)) {
            for (std::size_t n = 0; n < N; ++n) out[n] = a[n] * m.chirp[n];
        } else {
            for (std::size_t n = 0; n < N; ++n) out[n] = a[n] * m.chirp[n] * fct;
        }
    }

private:
    // Fast in-place pow2 FFT of a padded Bluestein buffer (m.padded_size is always
    // a power of 2). Routes through the production vectorized path — the codelet
    // for small padded sizes, else the iterative DIF driver. Forward is
    // un-normalized; inverse scales by 1/pad.
    template<bool Forward>
    void pow2_fft(std::span<std::complex<T>> buf) const {
        const std::size_t pad = m.padded_size;
        if (is_codelet_catalog(pad)) {
            codelet_dispatch<T, Forward>(buf.data(), buf.data(), pad);
            if constexpr (!Forward) {
                const T scale = T(1) / T(pad);
                for (auto& x : buf) x *= scale;
            }
            return;
        }
        soa_scratch<T, 4> sc(pad);
        if constexpr (Forward) {
            iterative_dif_execute_ws<T, true>(buf.data(), buf.data(), pad, sc.buf(0), sc.buf(1),
                                              sc.buf(2), sc.buf(3), m.bl_dif_fwd);
        } else {
            iterative_dif_execute_ws<T, false>(buf.data(), buf.data(), pad, sc.buf(0), sc.buf(1),
                                               sc.buf(2), sc.buf(3), m.bl_dif_inv);
            const T scale = T(1) / T(pad);
            for (auto& x : buf) x *= scale;
        }
    }
};

} // namespace detail
} // namespace admiral

