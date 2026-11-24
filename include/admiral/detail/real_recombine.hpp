#pragma once

#include <complex>
#include <cstddef>
#include <type_traits>

#include "simd.hpp"

namespace admiral {
namespace detail {

// The recombine row is tw[k] = exp(-i pi k / M), so tw[M - k] = -conj(tw[k]) exactly: the row
// holds only k = 0 .. M / 2 and the upper half comes from this mirror.
template <typename T>
inline std::complex<T> mirror_tw(const std::complex<T>& t) {
    return {-t.real(), t.imag()};
}

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

// Two-source gather masks for even_recombine_simd over the raw interleaved halves. Word
// 2 * j of a raw half is the real part of bin j, word 2 * j + 1 its imaginary part. Mask
// values 0 .. n - 1 take the first register, n .. 2 * n - 1 the second. Lane i of the mirror
// half holds bin lo + L - 1 - i = M - k - i, the pair of lane i's bin k + i, so one indexed
// permute per component fuses the deinterleave with the lane reverse: 2 lane shuffles in
// place of the 4 of a deinterleave plus a lane reverse.
template <std::size_t Imag>
struct mirror_pair {
    static constexpr std::size_t get(std::size_t i, std::size_t n) {
        const std::size_t w = 2 * (n - 1 - i) + Imag;
        return w < n ? w + n : w - n;
    }
};

// Store masks for the two raw halves of out + lo. Word 2 * i of the store is the real part of
// bin lo + i, the mirror bin of pair L - 1 - i. One indexed permute per store half fuses the
// lane reverse with the store interleave: 2 lane shuffles in place of 4.
template <std::size_t Half>
struct mirror_store {
    static constexpr std::size_t get(std::size_t i, std::size_t n) {
        const std::size_t w = Half * n + i, e = w / 2;
        return (w & 1) == 0 ? 2 * n - 1 - e : n - 1 - e;
    }
};

template <typename T>
inline constexpr bool kRecombineSimd = std::is_same_v<T, float> || std::is_same_v<T, double>;

// Pairs (k, M - k) for k = 1 .. kmax, L of them for each step, where L is the tier width
// and M is the transform length. A step runs only while its bins stay in the range
// [k, M - k]: the steps before it wrote the bins that are not in that range and out can
// equal in. The lane that pairs the middle bin with itself writes bin M / 2 two
// times. The caller repairs that bin from the value it reads before the steps. The tail
// step reads twiddle values above k = M / 2, the mirror pad of the row. Below that, the
// same follows at L / 2 down to 4 lanes and then the scalar loop. The output is the
// first k it did not do.
template <bool Forward, typename T, std::size_t L = xsimd::batch<T>::size>
inline std::size_t even_recombine_simd(const std::complex<T>* in, const std::complex<T>* tw,
                                       std::complex<T>* out, std::size_t M, std::size_t k = 1) {
    using R = xsimd::make_sized_batch_t<T, L>;
    if constexpr (L >= 2 && !std::is_void_v<R>) {
        using C = xsimd::batch<std::complex<T>, typename R::arch_type>;
        using index = xsimd::as_unsigned_integer_t<typename R::value_type>;
        const std::size_t kend = (M - 1) / 2 + 1;
        for (; k < kend && k + L <= M - k + 1; k += L) {
            const std::size_t lo = M - k - L + 1;
            const C a = C::load_unaligned(in + k);
            // The mirror half loads raw: two indexed permutes fuse its deinterleave with the
            // lane reverse. Lane i pairs bin k + i with bin M - k - i.
            const T* blp = reinterpret_cast<const T*>(in + lo);
            const R bl = R::load_unaligned(blp), bh = R::load_unaligned(blp + L);
            const C br(
                xsimd::shuffle(bh, bl, xsimd::make_batch_constant<index, mirror_pair<0>, typename R::arch_type>()),
                xsimd::shuffle(bh, bl, xsimd::make_batch_constant<index, mirror_pair<1>, typename R::arch_type>()));
            const C ta = C::load_unaligned(tw + k);
            // The 0.5 goes on the sum and difference terms before the twiddle FMAs, as in
            // r2c_even_bin. A bin near the type maximum then does not overflow in the FMAs.
            const R h(typename R::value_type(0.5)), two(typename R::value_type(2));
            const R sre = (a.real() + br.real()) * h, si = (a.imag() + br.imag()) * h;
            const R sd = (a.imag() - br.imag()) * h, d = (br.real() - a.real()) * h;
            const R tr = ta.real(), ti = ta.imag();
            R xkr, xki;
            if constexpr (Forward) {
                xkr = xsimd::fnma(ti, d, xsimd::fma(tr, si, sre));
                xki = xsimd::fma(ti, si, xsimd::fma(tr, d, sd));
            } else {
                xkr = xsimd::fnma(ti, d, xsimd::fnma(tr, si, sre));
                xki = xsimd::fma(ti, si, xsimd::fnma(tr, d, sd));
            }
            C(xkr, xki).store_unaligned(out + k);
            // The mirror bin of pair (k + i, M - k - i) is (2 sre - xkr, xki - 2 sd) in both
            // directions: the scalar bin on the swapped inputs and the mirrored twiddle
            // tw[M - k] = -conj(tw[k]). The conjugate of the partner is a different value.
            // The product 2 sre is the exact sum a + b, so it is finite when the sum is. This
            // holds with and without FMA.
            // Two indexed permutes fuse the lane reverse with the store interleave.
            const R mre = xsimd::fms(two, sre, xkr), mim = xsimd::fnma(two, sd, xki);
            T* bop = reinterpret_cast<T*>(out + lo);
            xsimd::shuffle(mim, mre, xsimd::make_batch_constant<index, mirror_store<0>, typename R::arch_type>())
                .store_unaligned(bop);
            xsimd::shuffle(mim, mre, xsimd::make_batch_constant<index, mirror_store<1>, typename R::arch_type>())
                .store_unaligned(bop + L);
        }
        if constexpr (L / 2 >= 4) return even_recombine_simd<Forward, T, L / 2>(in, tw, out, M, k);
        else return k;
    } else {
        return k;
    }
}

// The full recombine pass: edges k=0 and k=M, mirrored pairs (k, M-k), middle bin. tw holds
// k = 0 .. M / 2.
// Xr may equal z: each step loads both bins of a pair before it stores either.
template <typename T>
inline void r2c_even_recombine(const std::complex<T>* z, const std::complex<T>* tw,
                               std::complex<T>* Xr, std::size_t M) {
    const std::complex<T> z0 = z[0];
    Xr[0] = r2c_even_bin(z0, z0, tw[0]);
    Xr[M] = r2c_even_bin(z0, z0, mirror_tw(tw[0]));
    const std::size_t kmax = (M - 1) / 2;
    // The middle bin is read before a SIMD step writes it twice.
    const std::complex<T> zm = ((M & 1) == 0) ? z[M / 2] : std::complex<T>();
    std::size_t k = 1;
    if constexpr (kRecombineSimd<T>) k = even_recombine_simd<true>(z, tw, Xr, M);
    for (; k <= kmax; ++k) {
        const std::complex<T> zk = z[k], zb = z[M - k];
        Xr[k] = r2c_even_bin(zk, zb, tw[k]);
        Xr[M - k] = r2c_even_bin(zb, zk, mirror_tw(tw[k]));
    }
    if ((M & 1) == 0) Xr[M / 2] = r2c_even_bin(zm, zm, tw[M / 2]);
}

template <typename T>
inline void c2r_even_recombine(const std::complex<T>* X, const std::complex<T>* tw,
                               std::complex<T>* Z, std::size_t M) {
    // Edge bins: the imaginary parts of DC and Nyquist are ignored, matching
    // pocketfft's c2r. The spectrum input stays read-only for direct feeds.
    Z[0] = c2r_even_bin(std::complex<T>(X[0].real(), T(0)),
                       std::complex<T>(X[M].real(), T(0)), tw[0]);
    const std::size_t kmax = (M - 1) / 2;
    // The middle bin is read before a SIMD step writes it twice.
    const std::complex<T> zm = ((M & 1) == 0) ? X[M / 2] : std::complex<T>();
    std::size_t k = 1;
    if constexpr (kRecombineSimd<T>) k = even_recombine_simd<false>(X, tw, Z, M);
    for (; k <= kmax; ++k) {
        const std::complex<T> Xk = X[k], Xm = X[M - k];
        Z[k] = c2r_even_bin(Xk, Xm, tw[k]);
        Z[M - k] = c2r_even_bin(Xm, Xk, mirror_tw(tw[k]));
    }
    if ((M & 1) == 0) Z[M / 2] = c2r_even_bin(zm, zm, tw[M / 2]);
}

}
}
