#pragma once

// ============================================================================
// Real-to-real transforms (DCT/DST) on top of the r2c engine. Makhoul's shuffle folds
// the half-sample shift into a length-N real transform, so a DCT costs one r2c plus
// two O(N) passes:
//   v[j] = x[2j], v[N-1-j] = x[2j+1];  V = DFT_N(v)
//   Wk = exp(-i*pi*k/(2N)) * V[k]
//   DCT2[k] = 2*Re(Wk), DCT2[N-k] = -2*Im(Wk),  k = 0..N/2
// DCT-III is DCT-II transposed and DST-II is DCT-II of a sign-alternated input read
// backwards, so there are exactly two cores, not four.
// Normalization: `forward()` is FFTW's unnormalized kind and `inverse()` its exact
// inverse, so `forward()` -> `inverse()` is the identity; FFTW's own pair differs by 2N.
// Pass `fct` for that pair.
// ============================================================================

#include <complex>
#include <cstddef>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

#include <admiral/admiral.hpp>  // `effort`, `r2r_kind`

#include "portable_trig.hpp"  // `sincos_turns`
#include "real_fft.hpp"       // `real_adm_plan`
#include "scratch.hpp"        // `make_aligned_buffer`
#include "thread_pool.hpp"    // `thread_pool`

namespace admiral {
namespace detail {

// The kinds whose forward runs the DCT-II pipeline; the other two run it backwards.
[[nodiscard]] constexpr bool r2r_forward_is_dct2(r2r_kind kind) noexcept {
    return kind == r2r_kind::dct2 || kind == r2r_kind::dst2;
}

// The sine kinds reach the cosine core through the sign-flip and reversal identity.
[[nodiscard]] constexpr bool r2r_is_sine(r2r_kind kind) noexcept {
    return kind == r2r_kind::dst2 || kind == r2r_kind::dst3;
}

// 1-D DCT/DST over `rows` contiguous lines of length N. The plan owns the scratch: one
// plan, one thread.
template<typename T>
class r2r_plan {
public:
    r2r_plan(std::size_t N, r2r_kind kind, std::size_t rows = 1,
             admiral::effort eff = admiral::effort::estimate, std::size_t nthreads = 1);

    // `fct` scales the result; `std::nullopt` takes the kind's own normalization, see `run()`.
    void forward(const T* in, T* out, std::optional<T> fct = std::nullopt) const {
        run(in, out, r2r_forward_is_dct2(kind_), fct);
    }
    void inverse(const T* in, T* out, std::optional<T> fct = std::nullopt) const {
        run(in, out, !r2r_forward_is_dct2(kind_), fct);
    }

    [[nodiscard]] std::size_t size() const noexcept { return N_ * rows_; }

private:
    // Cosine core, forward: shuffle -> `r2c` -> post-twiddle. A sine kind adds the input
    // sign flip and the output reversal.
    void dct2_rows(const T* in, T* out, T scale) const;
    // Cosine core, backwards: pre-twiddle -> `c2r` -> unshuffle. Exact inverse of
    // `dct2_rows` at scale 1: `c2r` carries the 1/N.
    void dct3_rows(const T* in, T* out, T scale) const;

    // Runs from the initializer list: rejects a zero before it reaches `rp_`.
    [[nodiscard]] static std::size_t checked(std::size_t N, std::size_t rows) {
        if (N == 0 || rows == 0) throw size_error("Plan size must be greater than 0");
        return N;
    }

    void run(const T* in, T* out, bool use_dct2, std::optional<T> fct) const {
        // Unscaled, `dct2_rows` is the type-2 kind and `dct3_rows` its exact inverse
        // (`c2r` carries the 1/N), so type-2 kinds need no scale. FFTW's type-3 is its
        // type-2's inverse times 2N, and that factor is the whole difference.
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
    std::vector<std::complex<T>> tw_;      // exp(-i*pi*k/(2N)), k = 0..N/2
    aligned_buffer<T> v_;                  // rows_ * N_ shuffled reals
    aligned_buffer<std::complex<T>> spec_; // rows_ * Nh_ half-spectra
    // `pool_` threads the `r2c`/`c2r` batched tile loop over rows; the O(N) passes stay serial.
    std::unique_ptr<thread_pool> pool_;
    std::size_t Nh_;
};

template<typename T>
r2r_plan<T>::r2r_plan(std::size_t N, r2r_kind kind, std::size_t rows, admiral::effort eff,
                      std::size_t nthreads)
    : N_(N), rows_(rows), kind_(kind), rp_(checked(N, rows), eff), Nh_(N / 2 + 1) {
    // exp(-i*pi*k/(2N)) = exp(-2*pi*i*k/(4N)), built by the integer-exact helper,
    // not `std::cos`.
    tw_.resize(Nh_);
    for (std::size_t k = 0; k < Nh_; ++k) {
        const auto [sn, cs] = portable_trig::sincos_turns<true>(k, 4 * N);
        tw_[k] = std::complex<T>(static_cast<T>(cs), static_cast<T>(sn));
    }
    v_ = make_aligned_buffer<T>(rows_ * N_);
    spec_ = make_aligned_buffer<std::complex<T>>(rows_ * Nh_);
    if (nthreads > 1 && rows_ > 1) pool_ = std::make_unique<thread_pool>(nthreads);
}

template<typename T>
void r2r_plan<T>::dct2_rows(const T* in, T* out, T scale) const {
    const bool sine = r2r_is_sine(kind_);
    for (std::size_t r = 0; r < rows_; ++r) {
        const T* x = in + r * N_;
        T* v = v_.get() + r * N_;
        // Even indices ascending, odd descending. (-1)^n is +1 on the even half, so the
        // sine flip is one negation of the descending stream.
        for (std::size_t j = 0; 2 * j < N_; ++j) v[j] = x[2 * j];
        for (std::size_t j = 0; 2 * j + 1 < N_; ++j)
            v[N_ - 1 - j] = sine ? -x[2 * j + 1] : x[2 * j + 1];
    }
    rp_.r2c(v_.get(), spec_.get(), rows_, pool_.get());
    const T two = scale * T(2);
    for (std::size_t r = 0; r < rows_; ++r) {
        const std::complex<T>* V = spec_.get() + r * Nh_;
        T* y = out + r * N_;
        for (std::size_t k = 0; k < Nh_; ++k) {
            const std::complex<T> W = tw_[k] * V[k];
            // Two writes per k (index N-1-k for sine). Skipped: k == 0 (no slot N or -1)
            // and 2k == N (hi == lo; V is real and Re(W) == -Im(W)). Every slot 0..N-1
            // is written exactly once.
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
    for (std::size_t r = 0; r < rows_; ++r) {
        const T* y = in + r * N_;
        std::complex<T>* V = spec_.get() + r * Nh_;
        for (std::size_t k = 0; k < Nh_; ++k) {
            // Undo the two writes per k: W = (y[k] - i*y[N-k])/2 with y[N] := 0.
            const std::size_t lo = sine ? N_ - 1 - k : k;
            const std::size_t hi = sine ? k - 1 : N_ - k;
            const T re = y[lo];
            const T im = k == 0 ? T(0) : -y[hi];
            V[k] = std::conj(tw_[k]) * std::complex<T>(re, im) * half;
        }
    }
    rp_.c2r(spec_.get(), v_.get(), rows_, pool_.get());   // carries 1/N
    for (std::size_t r = 0; r < rows_; ++r) {
        const T* v = v_.get() + r * N_;
        T* x = out + r * N_;
        for (std::size_t j = 0; 2 * j < N_; ++j) x[2 * j] = scale * v[j];
        for (std::size_t j = 0; 2 * j + 1 < N_; ++j)
            x[2 * j + 1] = sine ? -scale * v[N_ - 1 - j] : scale * v[N_ - 1 - j];
    }
}

// Instantiated once in `src/inst_real_{f,d}.cpp`; no consumer TU rebuilds the `r2c` tree.
extern template class r2r_plan<float>;
extern template class r2r_plan<double>;

}  // namespace detail
}  // namespace admiral
