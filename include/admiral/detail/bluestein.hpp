#pragma once

// ============================================================================
// Bluestein / chirp-z arbitrary-length DFT. Rewrites size-N as length-M
// (M = bluestein_choose_pad(N), the first {2,3,5,7}-smooth >= 2N-1 or bit_ceil(2N-1))
// cyclic convolution via chirp x[n]*W_N^{n^2/2}, evaluated by 3 size-M FFTs.
// General fallback for any N.
//
// Ref: Bluestein, "A linear filtering approach to the computation of discrete
// Fourier transform", IEEE Trans. Audio Electroacoust. 18 (1970) 451.
// DOI 10.1109/TAU.1970.1162132
// ============================================================================

#include <algorithm>
#include <bit>
#include <complex>
#include <cstddef>
#include <limits>
#include <optional>
#include <span>
#include <vector>

#include "dif_driver.hpp"       // iterative_dif_execute_ws, dif_execute_in_place
#include "four_step_large.hpp"  // four_step_large_plan, four_step_large_supported
#include "math.hpp"             // codelet_dispatch, is_codelet_catalog
#include "scratch.hpp"          // soa_scratch
#include "twiddles.hpp"         // dif_twiddle_set, build_dif_twiddle_set
#include "portable_trig.hpp"    // sincos_turns

namespace admiral {
namespace detail {

// bluestein_choose_pad lives in math.hpp, beside the other prices the routing cost model
// and its offline fitter have to agree on: both must featurise the pad the engine runs.

// Inner-engine delegate admission: the padded transform delegates to the six-step
// engine when the pad crosses the public large route's serial byte line
// (kLargeRouteSerialF64Bytes; f64 only) AND the pad's split is the band-fusable shape
// (four_step_large_fused_shape, the engine's own fused-sweep guard, so the gate tracks
// W across ISAs). Below the byte line the six-step arm loses, and n2 % n1 != 0 falls
// into four_step_transpose_cycles, which is not a fast path at all.
template<typename T>
[[nodiscard]] constexpr bool bluestein_inner_six_step_admits(std::size_t pad) {
    constexpr std::size_t line = sizeof(T) == 8 ? kLargeRouteSerialF64Bytes
                                                : std::numeric_limits<std::size_t>::max();
    return four_step_large_supported(pad, sizeof(std::complex<T>), line) &&
           four_step_large_fused_shape<T>(pad);
}

// ============================================================================
// Bluestein plan: ctor precomputes chirp, transformed kernel, and inner twiddles.
// execute() runs the convolution and applies 1/N for inverse. Built only on the
// bluestein route; common routes pay nothing for the precomputed buffers.
// ============================================================================

template<typename T>
class bluestein_plan {
    // Instance state in one aggregate; only populated on the bluestein route.
    struct M {
        std::size_t size;
        bool is_forward;
        std::size_t padded_size;
        std::vector<std::complex<T>> chirp;       // chirp W_N^{n^2/2}
        std::vector<std::complex<T>> kernel_fft;  // DFT_M(conj(chirp))
        // Twiddles for the inner padded pow2 transforms. ONE set for both directions:
        // the tables are direction-free (twiddles.hpp), and the convolution runs
        // fwd(a), fwd(kernel), inv(a) regardless of the outer direction. Built only
        // when the inner transforms run the in-place DIF arm (the six-step delegates
        // below replace it entirely at DRAM-scale pads).
        dif_twiddle_set<T> bl_dif;
        // Inner six-step delegates for pads past bluestein_inner_six_step_admits
        // (mutually exclusive with bl_dif: one inner engine per plan). execute() is
        // const with no plan-owned mutable state (four_step_large.hpp contract), so
        // the plan stays copyable and re-entrant. The inverse delegate's DEFAULT
        // scale is exactly 1/pad (P2 folds 1/n2, P4 1/n1 — see four_step_large),
        // matching the in-place arm's last-pass 1/pad.
        std::optional<four_step_large_plan<T>> six_fwd, six_inv;
    } m;

public:
    bluestein_plan(std::size_t size, bool is_forward)
        // Every member is designated even where {} is the default: -Wmissing-field-initializers
        // is an error here.
        : m{.size = size,
            .is_forward = is_forward,
            .padded_size = bluestein_choose_pad(size),
            .chirp = {},
            .kernel_fft = {},
            .bl_dif = {},
            .six_fwd = {},
            .six_inv = {}}
    {
        // Twiddle tables for padded pow2 transforms (only above codelet catalog).
        if (!is_codelet_catalog(m.padded_size)) {
            if (bluestein_inner_six_step_admits<T>(m.padded_size)) {
                m.six_fwd.emplace(m.padded_size, true);
                m.six_inv.emplace(m.padded_size, false);
            } else {
                m.bl_dif = build_dif_twiddle_set<T>(m.padded_size);
            }
        }

        // Chirp: b[n] = exp(+/- i*pi*n^2/N) = exp(+/- 2*pi*i * n^2/(2N)).
        m.chirp.resize(m.size);
        for (std::size_t n = 0; n < m.size; ++n) {
            const auto [s, c] = portable_trig::sincos_turns(n * n, 2 * m.size, m.is_forward);
            m.chirp[n] = std::complex<T>(static_cast<T>(c), static_cast<T>(s));
        }

        // Build and forward-transform the convolution kernel (reused for all executions).
        // max(...,1): padded_size is always >= 2*size, but sizing it so kernel[0] is
        // provably in bounds is what lets -Wnull-dereference see it (gcc-14 -Werror).
        std::vector<std::complex<T>> kernel(std::max<std::size_t>(m.padded_size, 1));
        kernel[0] = std::conj(m.chirp[0]);
        for (std::size_t n = 1; n < m.size; ++n) {
            kernel[n] = std::conj(m.chirp[n]);
            kernel[m.padded_size - n] = std::conj(m.chirp[n]);
        }

        // Forward-transform the kernel
        pad_fft<true>(std::span(kernel));
        m.kernel_fft = std::move(kernel);
    }

    // Run Bluestein. `fct` scales the output (folded into the final chirp sweep).
    // in==out: in-place. in!=out: reads `in` fully (chirp multiply) then writes `out`.
    void execute(const std::complex<T>* in, std::complex<T>* out, T fct) const {
        const std::size_t N = m.size;

        // Multiply input by chirp and zero-pad. soa_scratch is uninitialized, so
        // only the pad tail [N, padded) is zeroed -- a std::vector value-inits all
        // of it and the chirp sweep then overwrites [0, N), making that half of the
        // memset pure waste.
        soa_scratch<T, 1> scratch(2 * m.padded_size);
        T* const raw = scratch.buf(0);
        auto* const a = reinterpret_cast<std::complex<T>*>(raw);
        for (std::size_t n = 0; n < N; ++n) {
            a[n] = in[n] * m.chirp[n];
        }
        std::fill(raw + 2 * N, raw + 2 * m.padded_size, T(0));

        const std::span buf{a, m.padded_size};
        pad_fft<true>(buf);

        // Pointwise multiply with pre-transformed kernel.
        for (std::size_t i = 0; i < m.padded_size; ++i) {
            a[i] *= m.kernel_fft[i];
        }

        pad_fft<false>(buf);

        // Extract, multiply by chirp, fold in output scale.
        if (fct == T(1)) {
            for (std::size_t n = 0; n < N; ++n) out[n] = a[n] * m.chirp[n];
        } else {
            for (std::size_t n = 0; n < N; ++n) out[n] = a[n] * m.chirp[n] * fct;
        }
    }

private:
    // In-place FFT of the Bluestein buffer at padded_size ({2,3,5,7}-smooth,
    // not always pow2). Routes to codelet (small), the six-step delegate
    // (DRAM-scale pads), or the iterative DIF driver. Forward: un-normalized;
    // inverse: scaled by 1/pad.
    template<bool Forward>
    void pad_fft(std::span<std::complex<T>> buf) const {
        const std::size_t pad = m.padded_size;
        if (is_codelet_catalog(pad)) {
            codelet_dispatch<T, Forward>(buf.data(), buf.data(), pad);
            if constexpr (!Forward) scale_inplace(buf.data(), pad, T(1) / T(pad));
            return;
        }
        // Both engines need the inverse scaled by exactly 1/pad: the six-step
        // inverse's default (1/n2 in P2, 1/n1 in P4) equals the dif last-pass fold.
        const auto& six = Forward ? m.six_fwd : m.six_inv;
        if (six) {
            six->execute(buf.data(), buf.data(), Forward ? T(1) : T(1) / T(pad));
            return;
        }
        dif_execute_in_place<T>(Forward, buf.data(), buf.data(), pad, m.bl_dif,
                                Forward ? T(1) : T(1) / T(pad));
    }
};

} // namespace detail
} // namespace admiral

