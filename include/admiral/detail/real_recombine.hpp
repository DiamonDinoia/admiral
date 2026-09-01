#pragma once

#include <complex>
#include <cstddef>

namespace admiral {
namespace detail {

template <typename T>
inline std::complex<T> r2c_even_bin(const std::complex<T>* z, std::complex<T> tw,
                                   std::size_t M, std::size_t k) {
    const bool edge = (k == 0) || (k == M);
    const std::complex<T> zk = z[edge ? 0 : k];
    const std::complex<T> zb = z[edge ? 0 : M - k];
    const T zer = (zk.real() + zb.real()) * T(0.5);
    const T zei = (zk.imag() - zb.imag()) * T(0.5);
    const T zor = (zk.imag() + zb.imag()) * T(0.5);
    const T zoi = (zb.real() - zk.real()) * T(0.5);
    return {zer + tw.real() * zor - tw.imag() * zoi,
            zei + tw.real() * zoi + tw.imag() * zor};
}

template <typename T>
inline std::complex<T> c2r_even_bin(const std::complex<T>* X, std::complex<T> tw,
                                    std::size_t M, std::size_t k) {
    const std::complex<T> Xk = X[k], Xmk = X[M - k];
    const T zer = (Xk.real() + Xmk.real()) * T(0.5);
    const T zei = (Xk.imag() - Xmk.imag()) * T(0.5);
    const T vor = (Xk.real() - Xmk.real()) * T(0.5);
    const T voi = (Xk.imag() + Xmk.imag()) * T(0.5);
    const T zor = tw.real() * vor + tw.imag() * voi;
    const T zoi = tw.real() * voi - tw.imag() * vor;
    return {zer - zoi, zei + zor};
}

}
}
