#pragma once

#include <complex>
#include <cstddef>
#include <type_traits>

#include "simd.hpp"

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

// The xsimd form of the bins above: lane i of a, b and tw holds bin k + i, its mirror and its
// twiddle.
template <typename C>
inline C r2c_even_bins(const C& a, const C& b, const C& tw) {
    using R = typename C::real_batch;
    const R h(typename R::value_type(0.5));
    const R zer = (a.real() + b.real()) * h, zei = (a.imag() - b.imag()) * h;
    const R zor = (a.imag() + b.imag()) * h, zoi = (b.real() - a.real()) * h;
    return C(xsimd::fma(tw.real(), zor, zer) - tw.imag() * zoi,
             xsimd::fma(tw.imag(), zor, xsimd::fma(tw.real(), zoi, zei)));
}

template <typename C>
inline C c2r_even_bins(const C& a, const C& b, const C& tw) {
    using R = typename C::real_batch;
    const R h(typename R::value_type(0.5));
    const R zer = (a.real() + b.real()) * h, zei = (a.imag() - b.imag()) * h;
    const R vor = (a.real() - b.real()) * h, voi = (a.imag() + b.imag()) * h;
    const R zor = xsimd::fma(tw.imag(), voi, tw.real() * vor);
    const R zoi = xsimd::fnma(tw.imag(), vor, tw.real() * voi);
    return C(zer - zoi, zei + zor);
}

struct lane_reverse {
    static constexpr std::size_t get(std::size_t i, std::size_t n) { return n - 1 - i; }
};

template <typename C>
inline C reverse_lanes(const C& c) {
    using R = typename C::real_batch;
    using index = xsimd::as_unsigned_integer_t<typename R::value_type>;
    constexpr auto rev = xsimd::make_batch_constant<index, lane_reverse, typename R::arch_type>();
    return C(xsimd::swizzle(c.real(), rev), xsimd::swizzle(c.imag(), rev));
}

template <typename T>
inline constexpr bool kRecombineSimd = std::is_same_v<T, float> || std::is_same_v<T, double>;

// Pairs (k, M - k) for k = 1 .. kmax, H of them per step, while a full step fits. Returns the
// first k it did not do. Each step loads all its bins before it stores, so out may equal in.
template <bool Forward, typename T>
inline std::size_t even_recombine_simd(const std::complex<T>* in, const std::complex<T>* tw,
                                       std::complex<T>* out, std::size_t M) {
    using C = xsimd::batch<std::complex<T>>;
    constexpr std::size_t H = C::size;
    const std::size_t kend = (M - 1) / 2 + 1;
    std::size_t k = 1;
    for (; k + H <= kend; k += H) {
        const std::size_t lo = M - k - H + 1;
        const C a = C::load_unaligned(in + k), b = C::load_unaligned(in + lo);
        const C ta = C::load_unaligned(tw + k), tb = C::load_unaligned(tw + lo);
        const C ar = reverse_lanes(a), br = reverse_lanes(b);
        if constexpr (Forward) {
            r2c_even_bins(a, br, ta).store_unaligned(out + k);
            r2c_even_bins(b, ar, tb).store_unaligned(out + lo);
        } else {
            c2r_even_bins(a, br, ta).store_unaligned(out + k);
            c2r_even_bins(b, ar, tb).store_unaligned(out + lo);
        }
    }
    return k;
}

// The full recombine pass: edges k=0 and k=M, mirrored pairs (k, M-k), middle bin.
// Xr may equal z: each step loads both bins of a pair before it stores either.
template <typename T>
inline void r2c_even_recombine(const std::complex<T>* z, const std::complex<T>* tw,
                               std::complex<T>* Xr, std::size_t M) {
    const std::complex<T> z0 = z[0];
    Xr[0] = r2c_even_bin(z0, z0, tw[0]);
    Xr[M] = r2c_even_bin(z0, z0, tw[M]);
    const std::size_t kmax = (M - 1) / 2;
    std::size_t k = 1;
    if constexpr (kRecombineSimd<T>) k = even_recombine_simd<true>(z, tw, Xr, M);
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
    if constexpr (kRecombineSimd<T>) k = even_recombine_simd<false>(X, tw, Z, M);
    for (; k <= kmax; ++k) {
        const std::complex<T> Xk = X[k], Xm = X[M - k];
        Z[k] = c2r_even_bin(Xk, Xm, tw[k]);
        Z[M - k] = c2r_even_bin(Xm, Xk, tw[M - k]);
    }
    if ((M & 1) == 0) Z[M / 2] = c2r_even_bin(X[M / 2], X[M / 2], tw[M / 2]);
}

}
}
