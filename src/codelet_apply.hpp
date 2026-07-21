#pragma once

// Private implementation header — included only by per-N TUs and codelet_dispatch.cpp.
// Defines codelet_apply<N,T,Forward>: deinterleave -> kernel<N>::apply -> reinterleave.
// in == out is in-place; in != out reads `in` (preserved) and writes `out`.
// Un-normalized (does NOT apply 1/N).  N <= CODELET_CATALOG_MAX is guaranteed by routing.

#include <bit>
#include <complex>
#include <cstddef>

#include "admiral/detail/codelet.hpp"
#include "admiral/detail/simd_swizzle.hpp"

namespace admiral {
namespace detail {

template<unsigned N, typename T, bool Forward>
void codelet_apply(const std::complex<T>* in, std::complex<T>* out) {
    // Exact-sized stack buffer: 4 planar buffers of N elements each (N is a
    // compile-time codelet size <= CODELET_CATALOG_MAX, so this is always stack).
    alignas(xsimd::batch<T>::arch_type::alignment()) T buf[4 * N];
    T* xre = buf;
    T* xim = buf + N;
    T* yre = buf + 2 * N;
    T* yim = buf + 3 * N;

    constexpr std::size_t W = xsimd::batch<T>::size;

    // Below N=64 the scalar boundary loops constant-fold/unroll better than the
    // width-descent. Keep the simple loops there.
    if constexpr (N < 64) {
        for (std::size_t i = 0; i < N; ++i) {
            xre[i] = in[i].real();
            if constexpr (Forward) xim[i] =  in[i].imag();
            else                   xim[i] = -in[i].imag();
        }
        kernel<N, T, true>::apply(xre, xim, 1, yre, yim);
        for (std::size_t i = 0; i < N; ++i) {
            if constexpr (Forward) out[i] = std::complex<T>(yre[i],  yim[i]);
            else                   out[i] = std::complex<T>(yre[i], -yim[i]);
        }
        return;
    }

    // De-interleave AoS -> SoA: native-width shuffles, then a sized-batch width
    // descent (W/2, W/4, ..., 2) and at most one scalar residue per width gap.
    const T* src = reinterpret_cast<const T*>(in);
    std::size_t i = 0;
    for (; i + W <= N; i += W) {
        xsimd::batch<T> re, im;
        aos_deinterleave(src + 2 * i, re, im);
        re.store_unaligned(xre + i);
        im.store_unaligned(xim + i);
    }
    poet::static_for<1, std::bit_width(W)>([&](auto S) {
        constexpr std::size_t Wt = W >> S;
        using Vt = xsimd::make_sized_batch_t<T, Wt>;
        if constexpr (Wt >= 2 && !std::is_void_v<Vt>) {
            if (i + Wt <= N) {
                Vt re, im;
                aos_deinterleave<T, Vt>(src + 2 * i, re, im);
                re.store_unaligned(xre + i);
                im.store_unaligned(xim + i);
                i += Wt;
            }
        }
    });
    for (; i < N; ++i) {
        xre[i] = in[i].real();
        xim[i] = in[i].imag();
    }

    // Inverse via the conjugate identity  inv(x) = conj(fwd(conj(x)))  so only
    // the forward kernel is instantiated (halves the codelet catalog's heavy
    // per-N instantiation). conj = negate the imaginary SoA lane.
    if constexpr (!Forward) for (std::size_t k = 0; k < N; ++k) xim[k] = -xim[k];
    kernel<N, T, true>::apply(xre, xim, 1, yre, yim);
    if constexpr (!Forward) for (std::size_t k = 0; k < N; ++k) yim[k] = -yim[k];

    // Re-interleave SoA -> AoS, same width descent.
    T* dst = reinterpret_cast<T*>(out);
    i = 0;
    for (; i + W <= N; i += W) {
        aos_interleave(dst + 2 * i, xsimd::batch<T>::load_unaligned(yre + i),
                       xsimd::batch<T>::load_unaligned(yim + i));
    }
    poet::static_for<1, std::bit_width(W)>([&](auto S) {
        constexpr std::size_t Wt = W >> S;
        using Vt = xsimd::make_sized_batch_t<T, Wt>;
        if constexpr (Wt >= 2 && !std::is_void_v<Vt>) {
            if (i + Wt <= N) {
                aos_interleave<T, Vt>(dst + 2 * i, Vt::load_unaligned(yre + i),
                                      Vt::load_unaligned(yim + i));
                i += Wt;
            }
        }
    });
    for (; i < N; ++i) {
        out[i] = std::complex<T>(yre[i], yim[i]);
    }
}

} // namespace detail
} // namespace admiral
