#pragma once

// ============================================================================
// Real-to-real transforms (DCT/DST) on top of the r2c engine.
//
// One length-N real FFT serves all four kinds. Makhoul's shuffle folds the
// half-sample shift into a real transform of the SAME length, not a 2N-long even
// extension, so a DCT costs one r2c plus two O(N) passes.
//
//   v[j] = x[2j], v[N-1-j] = x[2j+1]         (even indices up, odd indices down)
//   V    = DFT_N(v)                          via real_adm_plan::r2c
//   Wk   = exp(-i*pi*k/(2N)) * V[k]
//   DCT2[k] = 2*Re(Wk), DCT2[N-k] = -2*Im(Wk)     k = 0..N/2
//
// The two writes per k are what makes one half-spectrum cover all N outputs: k and
// N-k meet only at k = N/2 (N even), where V is real and both formulas agree.
//
// Kinds. DCT-III is DCT-II transposed, so the inverse pipeline IS the DCT-III
// forward and vice versa: there are exactly two cores here, not four. DST-II is
// DCT-II of a sign-alternated input read backwards:
//   RODFT10(x)[k] = REDFT10(x')[N-1-k],  x'[n] = (-1)^n x[n]
//
// Normalization follows the library, not FFTW: forward() is FFTW's unnormalized kind
// (REDFT10 / REDFT01 / RODFT10 / RODFT01) and inverse() is its exact inverse, so
// forward -> inverse is the identity. FFTW's own inverse pair differs by 2N; pass fct
// to get it.
// ============================================================================

#include <complex>
#include <cstddef>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

#include <admiral/admiral.hpp>  // effort, r2r_kind

#include "portable_trig.hpp"  // sincos_turns
#include "real_fft.hpp"       // real_adm_plan
#include "scratch.hpp"        // make_aligned_buffer
#include "thread_pool.hpp"    // thread_pool

namespace admiral {
namespace detail {

// True for the kinds whose forward direction runs the DCT-II pipeline. The other
// two run it backwards, which is exactly what their forward transform is.
[[nodiscard]] constexpr bool r2r_forward_is_dct2(r2r_kind kind) noexcept {
    return kind == r2r_kind::dct2 || kind == r2r_kind::dst2;
}

// True for the sine kinds, which reach the cosine core through the sign-flip and
// reversal identity above.
[[nodiscard]] constexpr bool r2r_is_sine(r2r_kind kind) noexcept {
    return kind == r2r_kind::dst2 || kind == r2r_kind::dst3;
}

// 1-D DCT/DST over `rows` contiguous lines of length N. Owns its scratch, so it is
// non-re-entrant for the same reason real_adm_plan is: one plan, one thread.
template<typename T>
class r2r_plan {
public:
    r2r_plan(std::size_t N, r2r_kind kind, std::size_t rows = 1,
             admiral::effort eff = admiral::effort::estimate, std::size_t nthreads = 1);

    // fct scales the result; nullopt takes the kind's own normalization, see run().
    void forward(const T* in, T* out, std::optional<T> fct = std::nullopt) const {
        run(in, out, r2r_forward_is_dct2(kind_), fct);
    }
    void inverse(const T* in, T* out, std::optional<T> fct = std::nullopt) const {
        run(in, out, !r2r_forward_is_dct2(kind_), fct);
    }

    [[nodiscard]] std::size_t size() const noexcept { return N_ * rows_; }

private:
    // The cosine core, forward: shuffle -> r2c -> post-twiddle. For a sine kind the
    // same code applies the (-1)^n input flip and the output reversal.
    void dct2_rows(const T* in, T* out, T scale) const;
    // The cosine core, backwards: pre-twiddle -> c2r -> unshuffle. Exact inverse of
    // dct2_rows at scale 1 (c2r carries the 1/N), so `scale` only carries the kind's
    // normalization.
    void dct3_rows(const T* in, T* out, T scale) const;

    // Called from the initializer list: rp_ is constructed before the constructor body
    // runs, so a zero has to be rejected before it reaches the inner real plan.
    [[nodiscard]] static std::size_t checked(std::size_t N, std::size_t rows) {
        if (N == 0 || rows == 0) throw size_error("Plan size must be greater than 0");
        return N;
    }

    void run(const T* in, T* out, bool use_dct2, std::optional<T> fct) const {
        // Unscaled, dct2_rows IS the type-2 kind and dct3_rows IS its exact inverse,
        // because c2r already carries the 1/N that makes it so. Hence the type-2 kinds need no
        // scale in either direction. The type-3 kinds are the same two cores read the
        // other way round, and FFTW's type-3 is its type-2's inverse times 2N
        // (REDFT01(REDFT10(x)) = 2N*x), so that factor is the whole difference.
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
    // Threads the r2c/c2r batched tile loop over rows; the O(N) shuffle and twiddle
    // passes stay serial, so this only pays for many rows.
    std::unique_ptr<thread_pool> pool_;
    std::size_t Nh_;
};

template<typename T>
r2r_plan<T>::r2r_plan(std::size_t N, r2r_kind kind, std::size_t rows, admiral::effort eff,
                      std::size_t nthreads)
    : N_(N), rows_(rows), kind_(kind), rp_(checked(N, rows), eff), Nh_(N / 2 + 1) {
    // exp(-i*pi*k/(2N)) = exp(-2*pi*i*k/(4N)): the half-sample shift Makhoul's
    // shuffle leaves behind, asked of the same integer-exact helper as every other
    // twiddle here rather than of std::cos.
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
        // Even indices ascending, odd indices descending. The sine kinds read a
        // sign-alternated input, and (-1)^n is +1 on the even half and -1 on the odd,
        // so the flip is one negation of the descending stream, not a pass over x.
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
            // k and N-k in one step, reversed for the sine kinds (index N-1-k). The loop
            // skips both cases where the second index is not its own slot: at k == 0
            // there is no index N (cosine) or -1 (sine), and at 2k == N (N even) hi is
            // lo again: V is real there and Re(W) == -Im(W), so it would store the
            // value already written. Every slot 0..N-1 is then written exactly once.
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
            // Undo the two-writes-per-k of dct2_rows: W = (y[k] - i*y[N-k])/2 with
            // y[N] := 0, then V[k] = conj(tw[k]) * W. Same index mapping as there.
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

// Instantiated once in src/inst_real_{f,d}.cpp, like real_adm_plan: the r2c tree
// below it is the expensive part and no consumer TU should rebuild it.
extern template class r2r_plan<float>;
extern template class r2r_plan<double>;

}  // namespace detail
}  // namespace admiral
