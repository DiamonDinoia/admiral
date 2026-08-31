#pragma once

// ============================================================================
// Bluestein chirp-z arbitrary-length DFT: size-N as a length-M cyclic convolution via
// the chirp x[n]*W_N^{n^2/2}, evaluated by 3 size-M FFTs. M = `bluestein_choose_pad`(N).
// General fallback for any N.
// Ref: Bluestein, IEEE Trans. Audio Electroacoust. 18 (1970) 451.
// DOI 10.1109/TAU.1970.1162132
// ============================================================================

#include <algorithm>
#include <complex>
#include <cstddef>
#include <limits>
#include <optional>
#include <vector>
#include "cxx_compat.hpp"  // span, detail::bit_ceil

#include "dif_driver.hpp"       // `iterative_dif_execute_ws`, `dif_execute_in_place`
#include "four_step_large.hpp"  // `four_step_large_plan`, `four_step_large_supported`
#include "math.hpp"             // `codelet_dispatch`, `is_codelet_catalog`
#include "scratch.hpp"          // `soa_scratch`
#include "twiddles.hpp"         // `dif_twiddle_set`, `build_dif_twiddle_set`
#include "portable_trig.hpp"    // `sincos_turns`

namespace admiral {
namespace detail {

// `bluestein_choose_pad` lives in `math.hpp`: the routing model and fitter must
// featurise the pad the engine runs.

// The six-step engine takes the padded transform only past the public large route's
// serial byte line (f64 only) and only for the band-fusable split shape. Below
// the line six-step loses; n2 % n1 != 0 falls into `four_step_transpose_cycles`.
template<typename T>
[[nodiscard]] constexpr bool bluestein_inner_six_step_admits(std::size_t pad) {
    constexpr std::size_t line = sizeof(T) == 8 ? kLargeRouteSerialF64Bytes
                                                : std::numeric_limits<std::size_t>::max();
    return four_step_large_supported(pad, sizeof(std::complex<T>), line) &&
           four_step_large_fused_shape<T>(pad);
}

// Bluestein plan: the ctor precomputes chirp, transformed kernel and inner twiddles;
// `execute()` runs the convolution and applies 1/N for inverse. Built only on the
// Bluestein route.

template<typename T>
class bluestein_plan {
    struct M {
        std::size_t size;
        bool is_forward;
        std::size_t padded_size;
        std::vector<std::complex<T>> chirp;       // chirp W_N^{n^2/2}
        std::vector<std::complex<T>> kernel_fft;  // DFT_M(conj(chirp))
        // Inner-transform twiddles: direction-free, one set for both directions, built
        // only for the in-place DIF arm. The six-step delegates replace the set entirely.
        dif_twiddle_set<T> bl_dif;
        // Six-step delegates for admitted pads, mutually exclusive with `bl_dif`. The
        // inverse delegate's default scale is exactly 1/pad, matching the DIF last-pass
        // fold.
        std::optional<four_step_large_plan<T>> six_fwd, six_inv;
    } m;

public:
    bluestein_plan(std::size_t size, bool is_forward)
        // Every member initialized: `-Wmissing-field-initializers` is an error here.
        : m{size,
            is_forward,
            bluestein_choose_pad(size),
            {},
            {},
            {},
            {},
            {}}
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

        // Forward-transform the convolution kernel once. The `std::max` with 1 keeps
        // `kernel[0]` provably in bounds for `-Wnull-dereference` (gcc-14 `-Werror`).
        std::vector<std::complex<T>> kernel(std::max<std::size_t>(m.padded_size, 1));
        kernel[0] = std::conj(m.chirp[0]);
        for (std::size_t n = 1; n < m.size; ++n) {
            kernel[n] = std::conj(m.chirp[n]);
            kernel[m.padded_size - n] = std::conj(m.chirp[n]);
        }

        pad_fft<true>(span(kernel));
        m.kernel_fft = std::move(kernel);
    }

    // `fct` folds into the final chirp sweep. If `in==out`, the call reads `in` fully
    // before any write.
    void execute(const std::complex<T>* in, std::complex<T>* out, T fct) const {
        const std::size_t N = m.size;

        // Zero only the pad tail [N, padded): `soa_scratch` is uninitialized and the
        // chirp sweep overwrites [0, N), so a full memset would be waste.
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
    // In-place FFT at `padded_size` ({2,3,5,7}-smooth, not always pow2): codelet,
    // six-step delegate, or the DIF driver. Forward un-normalized; inverse scaled by 1/pad.
    template<bool Forward>
    void pad_fft(span<std::complex<T>> buf) const {
        const std::size_t pad = m.padded_size;
        if (is_codelet_catalog(pad)) {
            codelet_dispatch<T, Forward>(buf.data(), buf.data(), pad);
            if constexpr (!Forward) scale_inplace(buf.data(), pad, T(1) / T(pad));
            return;
        }
        // Both engines scale the inverse by exactly 1/pad (six-step's default matches DIF).
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

