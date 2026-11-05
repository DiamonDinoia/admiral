#pragma once

#include <complex>
#include <cstddef>
#include <type_traits>

#if defined(__AVX512F__)
#include <immintrin.h>
#endif

namespace admiral {
namespace detail {

template <typename T>
inline std::complex<T> r2c_even_bin(const std::complex<T>& zk, const std::complex<T>& zb,
                                   std::complex<T> tw) {
    const T zer = (zk.real() + zb.real()) * T(0.5);
    const T zei = (zk.imag() - zb.imag()) * T(0.5);
    const T zor = (zk.imag() + zb.imag()) * T(0.5);
    const T zoi = (zb.real() - zk.real()) * T(0.5);
    return {zer + tw.real() * zor - tw.imag() * zoi,
            zei + tw.real() * zoi + tw.imag() * zor};
}

template <typename T>
inline std::complex<T> r2c_even_bin(const std::complex<T>* z, std::complex<T> tw,
                                   std::size_t M, std::size_t k) {
    const bool edge = (k == 0) || (k == M);
    return r2c_even_bin(z[edge ? 0 : k], z[edge ? 0 : M - k], tw);
}

template <typename T>
inline std::complex<T> c2r_even_bin(const std::complex<T>& Xk, const std::complex<T>& Xmk,
                                   std::complex<T> tw) {
    const T zer = (Xk.real() + Xmk.real()) * T(0.5);
    const T zei = (Xk.imag() - Xmk.imag()) * T(0.5);
    const T vor = (Xk.real() - Xmk.real()) * T(0.5);
    const T voi = (Xk.imag() + Xmk.imag()) * T(0.5);
    const T zor = tw.real() * vor + tw.imag() * voi;
    const T zoi = tw.real() * voi - tw.imag() * vor;
    return {zer - zoi, zei + zor};
}

template <typename T>
inline std::complex<T> c2r_even_bin(const std::complex<T>* X, std::complex<T> tw,
                                   std::size_t M, std::size_t k) {
    return c2r_even_bin(X[k], X[M - k], tw);
}

#if defined(__AVX512F__)
// Four mirrored bins per step. The element ops mirror r2c_even_bin/c2r_even_bin order.
inline __m512d r2c_even_bin_vec(__m512d a, __m512d b, __m512d tw) {
    const __m512d half = _mm512_set1_pd(0.5);
    const __m512d ar = _mm512_permute_pd(a, 0x00), ai = _mm512_permute_pd(a, 0xFF);
    const __m512d br = _mm512_permute_pd(b, 0x00), bi = _mm512_permute_pd(b, 0xFF);
    const __m512d wr = _mm512_permute_pd(tw, 0x00), wi = _mm512_permute_pd(tw, 0xFF);
    const __m512d zer = _mm512_mul_pd(_mm512_add_pd(ar, br), half);
    const __m512d zei = _mm512_mul_pd(_mm512_sub_pd(ai, bi), half);
    const __m512d zor = _mm512_mul_pd(_mm512_add_pd(ai, bi), half);
    const __m512d zoi = _mm512_mul_pd(_mm512_sub_pd(br, ar), half);
    const __m512d re = _mm512_sub_pd(_mm512_fmadd_pd(wr, zor, zer), _mm512_mul_pd(wi, zoi));
    const __m512d im = _mm512_fmadd_pd(wi, zor, _mm512_fmadd_pd(wr, zoi, zei));
    return _mm512_mask_blend_pd(0xAA, re, im);
}

inline __m512d c2r_even_bin_vec(__m512d a, __m512d b, __m512d tw) {
    const __m512d half = _mm512_set1_pd(0.5);
    const __m512d ar = _mm512_permute_pd(a, 0x00), ai = _mm512_permute_pd(a, 0xFF);
    const __m512d br = _mm512_permute_pd(b, 0x00), bi = _mm512_permute_pd(b, 0xFF);
    const __m512d wr = _mm512_permute_pd(tw, 0x00), wi = _mm512_permute_pd(tw, 0xFF);
    const __m512d zer = _mm512_mul_pd(_mm512_add_pd(ar, br), half);
    const __m512d zei = _mm512_mul_pd(_mm512_sub_pd(ai, bi), half);
    const __m512d vor = _mm512_mul_pd(_mm512_sub_pd(ar, br), half);
    const __m512d voi = _mm512_mul_pd(_mm512_add_pd(ai, bi), half);
    const __m512d zor = _mm512_fmadd_pd(wi, voi, _mm512_mul_pd(wr, vor));
    const __m512d zoi = _mm512_fnmadd_pd(wi, vor, _mm512_mul_pd(wr, voi));
    return _mm512_mask_blend_pd(0xAA, _mm512_sub_pd(zer, zoi), _mm512_add_pd(zei, zor));
}
#endif

// The full recombine pass: edges k=0 and k=M, mirrored pairs (k, M-k), middle bin.
template <typename T>
inline void r2c_even_recombine(const std::complex<T>* z, const std::complex<T>* tw,
                               std::complex<T>* Xr, std::size_t M) {
    Xr[0] = r2c_even_bin(z[0], z[0], tw[0]);
    Xr[M] = r2c_even_bin(z[0], z[0], tw[M]);
    const std::size_t kmax = (M - 1) / 2;
    std::size_t k = 1;
#if defined(__AVX512F__)
    if constexpr (std::is_same_v<T, double>) {
        for (; k + 3 <= kmax; k += 4) {
            const __m512d va = _mm512_loadu_pd(&z[k]);
            const __m512d vbl = _mm512_loadu_pd(&z[M - k - 3]);
            const __m512i rev = _mm512_setr_epi64(6, 7, 4, 5, 2, 3, 0, 1);
            const __m512d vb = _mm512_permutexvar_pd(rev, vbl);
            const __m512d var = _mm512_permutexvar_pd(rev, va);
            _mm512_storeu_pd(&Xr[k], r2c_even_bin_vec(va, vb, _mm512_loadu_pd(&tw[k])));
            _mm512_storeu_pd(&Xr[M - k - 3], r2c_even_bin_vec(vbl, var, _mm512_loadu_pd(&tw[M - k - 3])));
        }
    }
#endif
    for (; k <= kmax; ++k) {
        const std::complex<T> zk = z[k], zb = z[M - k];
        Xr[k] = r2c_even_bin(zk, zb, tw[k]);
        Xr[M - k] = r2c_even_bin(zb, zk, tw[M - k]);
    }
    if ((M & 1) == 0) Xr[M / 2] = r2c_even_bin(z[M / 2], z[M / 2], tw[M / 2]);
}

template <typename T>
inline void c2r_even_recombine(const std::complex<T>* X, const std::complex<T>* tw,
                               std::complex<T>* Z, std::size_t M) {
    // Edge bins: the imaginary parts of DC and Nyquist are ignored, matching
    // pocketfft's c2r. The spectrum input stays read-only for direct feeds.
    Z[0] = c2r_even_bin(std::complex<T>(X[0].real(), T(0)),
                       std::complex<T>(X[M].real(), T(0)), tw[0]);
    const std::size_t kmax = (M - 1) / 2;
    std::size_t k = 1;
#if defined(__AVX512F__)
    if constexpr (std::is_same_v<T, double>) {
        for (; k + 3 <= kmax; k += 4) {
            const __m512d va = _mm512_loadu_pd(&X[k]);
            const __m512d vbl = _mm512_loadu_pd(&X[M - k - 3]);
            const __m512i rev = _mm512_setr_epi64(6, 7, 4, 5, 2, 3, 0, 1);
            const __m512d vb = _mm512_permutexvar_pd(rev, vbl);
            const __m512d var = _mm512_permutexvar_pd(rev, va);
            _mm512_storeu_pd(&Z[k], c2r_even_bin_vec(va, vb, _mm512_loadu_pd(&tw[k])));
            _mm512_storeu_pd(&Z[M - k - 3], c2r_even_bin_vec(vbl, var, _mm512_loadu_pd(&tw[M - k - 3])));
        }
    }
#endif
    for (; k <= kmax; ++k) {
        const std::complex<T> Xk = X[k], Xm = X[M - k];
        Z[k] = c2r_even_bin(Xk, Xm, tw[k]);
        Z[M - k] = c2r_even_bin(Xm, Xk, tw[M - k]);
    }
    if ((M & 1) == 0) Z[M / 2] = c2r_even_bin(X[M / 2], X[M / 2], tw[M / 2]);
}

}
}
