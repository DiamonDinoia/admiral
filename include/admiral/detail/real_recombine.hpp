#pragma once

// The even-N real recombination, one bin at a time: a half-length complex
// transform over the interleaved input, then bin k traded against bin M-k.
// Both backends use this recombination (`real_fft.hpp`, `scalar_fft.hpp`).
// Real arithmetic throughout: a `std::complex` product carries Annex G
// inf/nan fix-up, which gcc emits inline without `-fcx-limited-range`.

#include <complex>
#include <cstddef>

namespace admiral {
namespace detail {

// Bin k: X[k] = Ze[k] + tw[k] * Zo[k] over the M-point transform `z` of the
// interleaved input. `k` runs to `M`, so DC (`k == 0`) and Nyquist (`k == M`)
// read `z[0]`.
template <typename T>
inline std::complex<T> r2c_even_bin(const std::complex<T>* z, std::complex<T> tw,
                                   std::size_t M, std::size_t k) {
    const bool edge = (k == 0) || (k == M);
    const std::complex<T> zk = z[edge ? 0 : k];
    const std::complex<T> zb = z[edge ? 0 : M - k];   // zc == conj(zb)
    const T zer = (zk.real() + zb.real()) * T(0.5);
    const T zei = (zk.imag() - zb.imag()) * T(0.5);
    // Zo = (zk - zc) / (2i) = (zk - zc) * (-i/2)
    const T zor = (zk.imag() + zb.imag()) * T(0.5);
    const T zoi = (zb.real() - zk.real()) * T(0.5);
    return {zer + tw.real() * zor - tw.imag() * zoi,
            zei + tw.real() * zoi + tw.imag() * zor};
}

// Inverse of `r2c_even_bin`: z[k] = Ze[k] + i*Zo[k] from X (M+1 bins).
// X[M - k] enters conjugated; `k == 0` pairs DC with Nyquist through X[M].
template <typename T>
inline std::complex<T> c2r_even_bin(const std::complex<T>* X, std::complex<T> tw,
                                    std::size_t M, std::size_t k) {
    const std::complex<T> Xk = X[k], Xmk = X[M - k];
    const T zer = (Xk.real() + Xmk.real()) * T(0.5);
    const T zei = (Xk.imag() - Xmk.imag()) * T(0.5);
    const T vor = (Xk.real() - Xmk.real()) * T(0.5);   // Vo = W_N^k Zo[k]
    const T voi = (Xk.imag() + Xmk.imag()) * T(0.5);
    const T zor = tw.real() * vor + tw.imag() * voi;   // Zo = conj(tw) * Vo
    const T zoi = tw.real() * voi - tw.imag() * vor;
    return {zer - zoi, zei + zor};                     // Ze + i*Zo, i*(a+ib) = -b+ia
}

}  // namespace detail
}  // namespace admiral
