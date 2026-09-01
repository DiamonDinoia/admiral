#pragma once

// DCT-II and DST-II from one real FFT of the same length: even samples forward, odd reversed,
// then a half-sample phase twiddle. Makhoul, IEEE Trans. ASSP 28 (1980) 27.

#include <complex>
#include <cstddef>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

#include <admiral/admiral.hpp>

#include "portable_trig.hpp"
#include "real_fft.hpp"
#include "scratch.hpp"   // soa_scratch, make_aligned_buffer
#include "thread_pool.hpp"

namespace admiral {
namespace detail {

[[nodiscard]] constexpr bool r2r_forward_is_dct2(r2r_kind kind) noexcept {
    return kind == r2r_kind::dct2 || kind == r2r_kind::dst2;
}

[[nodiscard]] constexpr bool r2r_is_sine(r2r_kind kind) noexcept {
    return kind == r2r_kind::dst2 || kind == r2r_kind::dst3;
}

template<typename T>
class r2r_plan {
public:
    r2r_plan(std::size_t N, r2r_kind kind, std::size_t rows = 1,
             admiral::effort eff = admiral::effort::estimate, std::size_t nthreads = 1);

    void forward(const T* in, T* out, std::optional<T> fct = std::nullopt) const {
        run(in, out, r2r_forward_is_dct2(kind_), fct);
    }
    void inverse(const T* in, T* out, std::optional<T> fct = std::nullopt) const {
        run(in, out, !r2r_forward_is_dct2(kind_), fct);
    }

    [[nodiscard]] std::size_t size() const noexcept { return N_ * rows_; }

private:
    void dct2_rows(const T* in, T* out, T scale) const;
    void dct3_rows(const T* in, T* out, T scale) const;

    [[nodiscard]] static std::size_t checked(std::size_t N, std::size_t rows) {
        if (N == 0 || rows == 0) throw size_error("Plan size must be greater than 0");
        return N;
    }

    void run(const T* in, T* out, bool use_dct2, std::optional<T> fct) const {
        const T two_n = static_cast<T>(2 * N_);
        const T def = r2r_forward_is_dct2(kind_) ? T(1)
                      : use_dct2                 ? T(1) / two_n
                                                 : two_n;
        const T scale = fct.value_or(def);
        if (use_dct2) dct2_rows(in, out, scale);
        else          dct3_rows(in, out, scale);
    }

    std::size_t N_, rows_;
    r2r_kind kind_;
    real_adm_plan<T> rp_;
    std::vector<std::complex<T>> tw_;
    std::unique_ptr<thread_pool> pool_;
    std::size_t Nh_;
};

template<typename T>
r2r_plan<T>::r2r_plan(std::size_t N, r2r_kind kind, std::size_t rows, admiral::effort eff,
                      std::size_t nthreads)
    : N_(N), rows_(rows), kind_(kind), rp_(checked(N, rows), eff), Nh_(N / 2 + 1) {
    tw_.resize(Nh_);
    for (std::size_t k = 0; k < Nh_; ++k) {
        const auto [sn, cs] = portable_trig::sincos_turns<true>(k, 4 * N);
        tw_[k] = std::complex<T>(static_cast<T>(cs), static_cast<T>(sn));
    }
    if (nthreads > 1 && rows_ > 1) pool_ = std::make_unique<thread_pool>(nthreads);
}

template<typename T>
void r2r_plan<T>::dct2_rows(const T* in, T* out, T scale) const {
    const bool sine = r2r_is_sine(kind_);
    soa_scratch<T, 1> v_sc(rows_ * N_);
    soa_scratch<std::complex<T>, 1> spec_sc(rows_ * Nh_);
    T* v_buf = v_sc.buf(0);
    std::complex<T>* spec_buf = spec_sc.buf(0);
    for (std::size_t r = 0; r < rows_; ++r) {
        const T* x = in + r * N_;
        T* v = v_buf + r * N_;
        for (std::size_t j = 0; 2 * j < N_; ++j) v[j] = x[2 * j];
        for (std::size_t j = 0; 2 * j + 1 < N_; ++j)
            v[N_ - 1 - j] = sine ? -x[2 * j + 1] : x[2 * j + 1];
    }
    rp_.r2c(v_buf, spec_buf, rows_, pool_.get());
    const T two = scale * T(2);
    for (std::size_t r = 0; r < rows_; ++r) {
        const std::complex<T>* V = spec_buf + r * Nh_;
        T* y = out + r * N_;
        for (std::size_t k = 0; k < Nh_; ++k) {
            const std::complex<T> W = tw_[k] * V[k];
            const std::size_t lo = sine ? N_ - 1 - k : k;
            y[lo] = two * W.real();
            if (k == 0 || 2 * k == N_) continue;
            const std::size_t hi = sine ? k - 1 : N_ - k;
            y[hi] = -two * W.imag();
        }
    }
}

template<typename T>
void r2r_plan<T>::dct3_rows(const T* in, T* out, T scale) const {
    const bool sine = r2r_is_sine(kind_);
    const T half = T(0.5);
    soa_scratch<T, 1> v_sc(rows_ * N_);
    soa_scratch<std::complex<T>, 1> spec_sc(rows_ * Nh_);
    T* v_buf = v_sc.buf(0);
    std::complex<T>* spec_buf = spec_sc.buf(0);
    for (std::size_t r = 0; r < rows_; ++r) {
        const T* y = in + r * N_;
        std::complex<T>* V = spec_buf + r * Nh_;
        for (std::size_t k = 0; k < Nh_; ++k) {
            const std::size_t lo = sine ? N_ - 1 - k : k;
            const std::size_t hi = sine ? k - 1 : N_ - k;
            const T re = y[lo];
            const T im = k == 0 ? T(0) : -y[hi];
            V[k] = std::conj(tw_[k]) * std::complex<T>(re, im) * half;
        }
    }
    rp_.c2r(spec_buf, v_buf, rows_, pool_.get());
    for (std::size_t r = 0; r < rows_; ++r) {
        const T* v = v_buf + r * N_;
        T* x = out + r * N_;
        for (std::size_t j = 0; 2 * j < N_; ++j) x[2 * j] = scale * v[j];
        for (std::size_t j = 0; 2 * j + 1 < N_; ++j)
            x[2 * j + 1] = sine ? -scale * v[N_ - 1 - j] : scale * v[N_ - 1 - j];
    }
}

extern template class r2r_plan<float>;
extern template class r2r_plan<double>;

}
}
