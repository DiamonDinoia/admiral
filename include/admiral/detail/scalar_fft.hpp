#pragma once

// Scalar backend for precisions the SIMD engine cannot represent (long
// double: no ISA has 80-bit SIMD registers). One recursive mixed-radix DIF
// covers radices {2,3,4,5,7,8}, running in place because every combine's read
// set equals its write set. A length that no radix divides runs a direct
// W-matrix pass at or below `kDirectMax`, a Bluestein chirp-z convolution
// above. Every mutable buffer (gather scratch, Bluestein wrap, nd line
// staging, leaf locals) slabs per `tid`. The radix butterflies are the
// engine's own (`butterfly.hpp`) at `V = T`, so this backend carries no
// second radix-math copy.

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include <poet/poet.hpp>

#include "butterfly.hpp"        // `sub_dft`: the engine's radix butterflies at `V = T`
#include "cxx_compat.hpp"       // `span`, `detail::numbers`
#include "real_recombine.hpp"  // `r2c_even_bin`, `c2r_even_bin`
#include "thread_pool.hpp"      // `thread_pool`, `parallel_for`, `will_thread`

namespace admiral {
namespace detail {

// Longest direct-DFT length. At or below `kDirectMax`, one accumulation over
// a precomputed W matrix has a flatter error profile than a chirp-z chain.
// The value also bounds the on-stack leaf buffer in `run_sub`.
inline constexpr std::size_t kDirectMax = 37;

// w_n^k: exp(-2 pi i k/n) forward, the conjugate backward. The trig runs in
// `T`, not `double`: a `double` twiddle would cap the whole transform at
// 2^-53. Quadrant reduction keeps the sine argument small and the quadrant
// multiples exact.
template<typename T>
inline std::complex<T> scalar_twiddle(std::size_t k, std::size_t n, bool forward) {
    k %= n;
    const std::size_t k4 = 4 * k;
    const std::size_t q = k4 / n;
    const T rem = static_cast<T>(k4 % n) / static_cast<T>(4 * n);
    const T ang = T(2) * detail::numbers::pi_v<T> * rem;
    const T c = std::cos(ang), sn = std::sin(ang);
    T re, im;
    switch (q) {
    case 0: re = c;   im = sn;  break;
    case 1: re = -sn; im = c;   break;
    case 2: re = -c;  im = -sn; break;
    default: re = sn; im = -c;  break;
    }
    return std::complex<T>(re, forward ? -im : im);
}

template<typename T>
inline std::complex<T> maybe_conj(std::complex<T> w, bool conj) {
    return conj ? std::complex<T>(w.real(), -w.imag()) : w;
}

// 1-D c2c over one contiguous line, in place. Unscaled: `fct` (default 1)
// multiplies every output element.
template<typename T>
class scalar_c2c {
public:
    explicit scalar_c2c(std::size_t n, std::size_t nthreads = 1) : n_(n), nthreads_(nthreads) {
        if (n_ > 1 && n_ <= kDirectMax) {
            build_direct(n_);
            return;
        }
        // One level state per combine stage; the chain stops at length 1 or a
        // residue `factor()` cannot split.
        for (std::size_t m = n_; m > 1;) {
            const std::size_t p = factor(m);
            if (p == 0) {
                if (m <= kDirectMax) {
                    // Residue inside a composite chain: a direct leaf beats
                    // one Bluestein convolution per sub-block.
                    build_direct(m);
                } else {
                    blue_ = std::make_unique<blue_state>(m, nthreads_);
                }
                break;
            }
            level_state ls;
            ls.p = p;
            ls.m = m;
            ls.tw.resize(m);
            for (std::size_t j = 0; j < m; ++j) ls.tw[j] = scalar_twiddle<T>(j, m, true);
            levels_.push_back(std::move(ls));
            m /= p;
        }
        if (levels_.empty()) return;   // pure Bluestein: no chain, no gather
        build_perm();
        // Gather target, one slab per `tid`, so `execute` allocates nothing.
        scratch_.assign(nthreads_, std::vector<std::complex<T>>(n_));
    }

    [[nodiscard]] std::size_t size() const { return n_; }

    template<bool forward>
    void execute(std::complex<T>* x, T fct, std::size_t tid,
                 thread_pool* pool = nullptr) const {
        if (n_ <= 1) {
            if (fct != T(1)) x[0] *= fct;
            return;
        }
        if (direct_n_ == n_) {
            direct<forward>(x);
            if (fct != T(1))
                for (std::size_t k = 0; k < n_; ++k) x[k] *= fct;
            return;
        }
        if (blue_ && levels_.empty()) {   // `bluestein` folds `fct` itself
            bluestein<forward>(x, n_, fct, tid);
            return;
        }
        // 1-D threading: the level-0 combine runs serially; the `p`
        // sub-problems then fan out over the pool, disjoint in `x`. The `tid`
        // slabs take the terminal leaves.
        if (will_thread(pool, levels_[0].p, n_)) {
            const level_state& lv0 = levels_[0];
            const std::size_t u0 = lv0.m / lv0.p;
            dft<forward>(x, 0, 0, tid, true);
            admiral::detail::parallel_for(
                pool, lv0.p, n_,
                [&](std::size_t begin, std::size_t end, std::size_t ctid) {
                    for (std::size_t s = begin; s < end; ++s)
                        run_sub<forward>(x, s * u0, u0, 1, ctid);
                });
        } else {
            // `tid`, not 0: the terminal leaves index their slabs by `tid`.
            dft<forward>(x, 0, 0, tid);
        }
        // Digit-reversal gather back to natural order, folding the scale.
        std::complex<T>* sx = scratch_[tid].data();
        for (std::size_t k = 0; k < n_; ++k) sx[k] = x[pos_of_[k]];
        if (fct != T(1))
            for (std::size_t k = 0; k < n_; ++k) x[k] = sx[k] * fct;
        else
            for (std::size_t k = 0; k < n_; ++k) x[k] = sx[k];
    }

private:
    // Radix preference: eights first (halves the streaming levels of a
    // radix-4 chain on even powers of two), then 3, 5, 7, 4, then a last 2.
    static std::size_t factor(std::size_t m) {
        for (const std::size_t p : {8u, 3u, 5u, 7u, 4u, 2u})
            if (m % p == 0) return p;
        return 0;
    }


    // One terminal sub-block at the given level (same dispatch as `dft`'s tail).
    template<bool forward>
    void run_sub(std::complex<T>* x, std::size_t off, std::size_t u, std::size_t level,
                 std::size_t tid) const {
        if (u == direct_n_)
            direct<forward>(x + off);
        else if (blue_ && u == blue_->n)
            bluestein<forward>(x + off, u, T(1), tid);
        else
            dft<forward>(x, off, level, tid);
    }

    // w_n^{jk} for a direct O(n^2) DFT, forward sign. One table serves both
    // exits (a whole length, or a chain residue). Only one exit arises per
    // plan, because a residue is strictly shorter than the length that
    // produced the residue.
    void build_direct(std::size_t n) {
        direct_n_ = n;
        direct_mat_.resize(n * n);
        for (std::size_t k = 0; k < n; ++k)
            for (std::size_t j = 0; j < n; ++j)
                direct_mat_[k * n + j] = scalar_twiddle<T>((j * k) % n, n, true);
    }

    template<bool forward>
    void direct(std::complex<T>* x) const {
        const std::size_t n = direct_n_;
        std::complex<T> t[kDirectMax];
        for (std::size_t j = 0; j < n; ++j) t[j] = x[j];
        for (std::size_t k = 0; k < n; ++k) {
            std::complex<T> acc(0, 0);
            for (std::size_t j = 0; j < n; ++j)
                acc += t[j] * maybe_conj<T>(direct_mat_[k * n + j], !forward);
            x[k] = acc;
        }
    }

    // One DIF level at compile-time radix P. With u = m/P, q < u and s,i < P:
    //   a_i               = x[off + q + i*u]
    //   x[off + q + s*u]  = w_m^{q s} * sum_i a_i w_P^{i s}
    // Read set equals write set per q, so the level runs in place.
    //
    // The butterflies are forward-only; the inverse rides the butterflies in
    // the swapped domain. In the swapped domain swap(fwd(swap x)) == inv(x),
    // and a twiddle multiply becomes a multiply by conj(w). The stage twiddle
    // therefore carries no direction.
    template<bool forward, std::size_t P>
    void combine(std::complex<T>* x, std::size_t off, std::size_t level) const {
        constexpr bool sw = !forward;
        // `level_state` is declared below, so take the state by index: a
        // member function body sees the whole class; a parameter type does not.
        const auto& lv = levels_[level];
        const std::size_t u = lv.m / P;
        const std::complex<T>* tw = lv.tw.data();
        for (std::size_t q = 0; q < u; ++q) {
            T ar[P], ai[P];
            for (std::size_t i = 0; i < P; ++i) {
                const std::complex<T> a = x[off + q + i * u];
                ar[i] = sw ? a.imag() : a.real();
                ai[i] = sw ? a.real() : a.imag();
            }
            sub_dft<T, P>(ar, ai, [&](auto sc, T yr, T yi) {
                constexpr std::size_t s = decltype(sc)::value;
                if constexpr (s != 0) {   // `w_m^0 = 1`, so output 0 takes no twiddle
                    const std::complex<T> w = tw[q * s];
                    const T r = yr * w.real() - yi * w.imag();
                    yi = yr * w.imag() + yi * w.real();
                    yr = r;
                }
                x[off + q + s * u] = sw ? std::complex<T>(yi, yr) : std::complex<T>(yr, yi);
            });
        }
    }

    // `poet::dispatch`'s compile-time radix set: one entry per butterfly in
    // `butterfly.hpp`.
    using radix_set = std::integer_sequence<std::size_t, 2, 3, 4, 5, 7, 8>;

    // `poet::dispatch` adapter: runtime radix to `combine`'s compile-time `P`.
    // A struct, not a lambda: a C++17 lambda cannot declare a templated
    // `operator()`.
    template<bool forward>
    struct combine_invoke_t {
        template<std::size_t P>
        void operator()(const scalar_c2c* self, std::complex<T>* x, std::size_t off,
                        std::size_t level) const {
            self->template combine<forward, P>(x, off, level);
        }
    };

    // One level's combine, then the `P` sub-problems the combine leaves behind.
    // `single_level`: stops after the combine (threaded level-0 fan-out).
    template<bool forward>
    void dft(std::complex<T>* x, std::size_t off, std::size_t level,
             std::size_t tid, bool single_level = false) const {
        const level_state& lv = levels_[level];
        poet::dispatch(poet::throw_on_no_match, combine_invoke_t<forward>{},
                       poet::dispatch_param<radix_set>{lv.p}, this, x, off, level);
        const std::size_t u = lv.m / lv.p;
        if (single_level || u == 1) return;
        for (std::size_t s = 0; s < lv.p; ++s) run_sub<forward>(x, off + s * u, u, level + 1, tid);
    }

    // After the recursion, position `j` holds X[natural(j)]: with base-p digits
    // s_l of `j`, natural(j) = sum_l s_l * prod_{l'<l} p_l'. `pos_of_` is the
    // inverse map.
    void build_perm() {
        pos_of_.resize(n_);
        for (std::size_t j = 0; j < n_; ++j) {
            std::size_t residual = j, stride_k = 1, natural = 0;
            for (const level_state& lv : levels_) {
                const std::size_t u = lv.m / lv.p;
                const std::size_t s = residual / u;
                residual -= s * u;
                natural += s * stride_k;
                stride_k *= lv.p;
            }
            // Leftover digits index inside the terminal block in natural order.
            natural += residual * stride_k;
            pos_of_[natural] = j;
        }
    }

    struct blue_state {
        explicit blue_state(std::size_t m, std::size_t nthreads)
            : n(m), pad(pad_size(m)), inner(pad, nthreads) {
            chirp.resize(n);
            for (std::size_t k = 0; k < n; ++k)
                chirp[k] = scalar_twiddle<T>((k * k) % (2 * n), 2 * n, true);
            // Convolution partner of a = x . chirp: b[k] = conj(chirp[k]) in
            // wrap-around. The backward transform conjugates the chirp, so
            // both tables precompute.
            std::vector<std::complex<T>> b(pad);
            for (std::size_t k = 0; k < n; ++k) {
                b[k] = std::conj(chirp[k]);
                if (k) b[pad - k] = std::conj(chirp[k]);
            }
            bfft = b;
            inner.template execute<true>(bfft.data(), T(1), 0);
            for (std::size_t i = 0; i < pad; ++i) b[i] = maybe_conj<T>(b[i], true);
            bfft_inv = std::move(b);
            inner.template execute<true>(bfft_inv.data(), T(1), 0);
            a_.assign(nthreads, std::vector<std::complex<T>>(pad));
        }
        static std::size_t pad_size(std::size_t n) {
            // Smallest {2,3,5,7}-smooth length >= 2n-1, so the inner
            // transform stays in the radix set. `n > kDirectMax` here, so the
            // scan loop is cheap.
            std::size_t cand = 2 * n - 1;
            for (;; ++cand) {
                std::size_t v = cand;
                for (std::size_t p : {4u, 3u, 5u, 7u, 2u})
                    while (v % p == 0) v /= p;
                if (v == 1) return cand;
            }
        }
        std::size_t n, pad;
        scalar_c2c inner;
        std::vector<std::complex<T>> chirp;     // w_n^{k^2/2} forward sign
        std::vector<std::complex<T>> bfft;      // FFT_pad(conj-reversed chirp)
        std::vector<std::complex<T>> bfft_inv;  // FFT_pad(reversed chirp)
        mutable std::vector<std::vector<std::complex<T>>> a_;  // wrap buffer, one slab per `tid`
    };

    template<bool forward>
    void bluestein(std::complex<T>* x, std::size_t m, T fct, std::size_t tid) const {
        const blue_state& B = *blue_;
        const std::size_t pad = B.pad;
        std::complex<T>* a = B.a_[tid].data();
        std::fill_n(a, pad, std::complex<T>(0, 0));
        for (std::size_t k = 0; k < m; ++k) a[k] = x[k] * maybe_conj<T>(B.chirp[k], !forward);
        // `inner` slabs scratch per `tid`, so the caller's `tid` must reach
        // `inner`: two threads sharing one slab corrupt each other's transform.
        B.inner.template execute<true>(a, T(1), tid);
        const auto& bfft = forward ? B.bfft : B.bfft_inv;
        for (std::size_t i = 0; i < pad; ++i) a[i] *= bfft[i];
        B.inner.template execute<false>(a, T(1), tid);  // unscaled inverse
        const T inv = fct / static_cast<T>(pad);
        for (std::size_t k = 0; k < m; ++k)
            x[k] = a[k] * maybe_conj<T>(B.chirp[k], !forward) * inv;
    }

    std::size_t n_;
    struct level_state {
        std::size_t p, m;
        std::vector<std::complex<T>> tw;
    };
    std::vector<level_state> levels_;
    std::vector<std::size_t> pos_of_;
    std::unique_ptr<blue_state> blue_;
    std::size_t direct_n_ = 0;                 // direct-DFT length (0 = no direct leaf)
    std::vector<std::complex<T>> direct_mat_;  // w_{direct_n_}^{jk}, forward sign
    std::size_t nthreads_;
    mutable std::vector<std::vector<std::complex<T>>> scratch_;  // gather target, per `tid`
};


// Public-plan states on the scalar engine (long double only): `plan`,
// `plan_r2c` and the one-shots route here; nothing else does.

// Transforms the first `n_axes` axes of a contiguous tensor (all, by default).
// Strided lines stage through a reusable scratch line.
template<typename T>
class scalar_nd_c2c {
public:
    explicit scalar_nd_c2c(span<const std::size_t> shape, std::size_t n_axes,
                           std::size_t nthreads = 1)
        : shape_(shape.begin(), shape.end()), stride_(shape_.size(), 1) {
        if (n_axes > shape_.size()) n_axes = shape_.size();
        active_ = n_axes;
        total_ = 1;
        for (std::size_t d = shape_.size(); d-- > 0;) {
            stride_[d] = total_;
            total_ *= shape_[d];
        }
        for (std::size_t d = 0; d < active_; ++d) {
            std::size_t e = 0;
            while (e < eng_.size() && eng_[e].size() != shape_[d]) ++e;
            if (e == eng_.size()) eng_.emplace_back(shape_[d], nthreads);
            axis_engine_.push_back(e);   // an index: `eng_` iterators can reallocate
        }
        // One staging buffer per `tid`, sized for the longest active line.
        std::size_t line_cap = 1;
        for (std::size_t d = 0; d < active_; ++d) line_cap = std::max(line_cap, shape_[d]);
        line_.assign(nthreads, std::vector<std::complex<T>>(line_cap));
    }

    [[nodiscard]] std::size_t size() const { return total_; }

    template<bool Forward>
    void execute(std::complex<T>* data, T fct = T(1), thread_pool* pool = nullptr) const {
        for (std::size_t d = 0; d < active_; ++d) {
            const std::size_t len = shape_[d], st = stride_[d];
            const scalar_c2c<T>& eng = eng_[axis_engine_[d]];
            // The scale rides the last axis's lines: every element of the
            // tensor crosses exactly one such line.
            const T line_fct = (d == active_ - 1) ? fct : T(1);
            const std::size_t pre = total_ / (len * st);
            const std::size_t nlines = pre * st;   // `st == 1`: one line per `p`
            // Enough lines: fan out per line, engines stay serial. Too few:
            // the engine splits its own first level.
            thread_pool* eng_pool =
                will_thread(pool, nlines, total_) ? nullptr : pool;
            admiral::detail::parallel_for(
                pool, nlines, total_,
                [&](std::size_t begin, std::size_t end, std::size_t tid) {
                    auto line = line_[tid].data();
                    for (std::size_t i = begin; i < end; ++i) {
                        const std::size_t p = st == 1 ? i : i / st;
                        const std::size_t q = st == 1 ? 0 : i - p * st;
                        std::complex<T>* base = data + p * len * st + q;
                        if (st == 1) {
                            eng.template execute<Forward>(base, line_fct, tid, eng_pool);
                            continue;
                        }
                        for (std::size_t j = 0; j < len; ++j) line[j] = base[j * st];
                        eng.template execute<Forward>(line, line_fct, tid);
                        for (std::size_t j = 0; j < len; ++j) base[j * st] = line[j];
                    }
                });
        }
    }

private:
    std::vector<std::size_t> shape_, stride_;
    std::vector<scalar_c2c<T>> eng_;
    std::vector<std::size_t> axis_engine_;
    mutable std::vector<std::vector<std::complex<T>>> line_;  // strided staging, per tid
    std::size_t total_, active_;
};

// 1-D real transform, both directions, unscaled. Even N: half-size complex
// DFT plus recombination. Odd N: full-size DFT plus spectrum completion.
template<typename T>
class scalar_r2c_1d {
public:
    explicit scalar_r2c_1d(std::size_t n, std::size_t nthreads = 1)
        : n_(n), nh_(n / 2 + 1), even_(n % 2 == 0), eng_(even_ ? n / 2 : n, nthreads) {
        tw_.resize(nh_);
        for (std::size_t k = 0; k < nh_; ++k) tw_[k] = scalar_twiddle<T>(k, n_, true);
        buf_.assign(nthreads, std::vector<std::complex<T>>(even_ ? n_ / 2 : n_));
    }

    [[nodiscard]] std::size_t real_size() const { return n_; }
    [[nodiscard]] std::size_t cplx_size() const { return nh_; }

    void forward(const T* in, std::complex<T>* out, std::size_t tid) const {
        auto buf = buf_[tid].data();
        if (!even_) {
            for (std::size_t i = 0; i < n_; ++i) buf[i] = {in[i], T(0)};
            eng_.template execute<true>(buf, T(1), tid);
            std::copy_n(buf, nh_, out);
            return;
        }
        const std::size_t M = n_ / 2;
        std::complex<T>* z = buf;
        for (std::size_t j = 0; j < M; ++j) z[j] = {in[2 * j], in[2 * j + 1]};
        eng_.template execute<true>(z, T(1), tid);
        for (std::size_t k = 0; k <= M; ++k) out[k] = r2c_even_bin(z, tw_[k], M, k);
    }

    // `extra_scale == 1` leaves the transform unscaled; the scale folds into
    // the unpack sweep.
    void inverse(std::complex<T>* spec, T* out, T extra_scale, std::size_t tid) const {
        auto buf = buf_[tid].data();
        if (!even_) {
            for (std::size_t k = 0; k < nh_; ++k) buf[k] = spec[k];
            for (std::size_t k = nh_; k < n_; ++k) buf[k] = std::conj(spec[n_ - k]);
            eng_.template execute<false>(buf, T(1), tid);
            const T inv_n = extra_scale / T(n_);   // the true IDFT_N scale folds here
            for (std::size_t i = 0; i < n_; ++i) out[i] = buf[i].real() * inv_n;
            return;
        }
        const std::size_t M = n_ / 2;
        std::complex<T>* z = buf;
        for (std::size_t k = 0; k < M; ++k) z[k] = c2r_even_bin(spec, tw_[k], M, k);
        eng_.template execute<false>(z, T(1), tid);
        const T inv_m = extra_scale / T(M);   // the true IDFT_M scale folds here
        for (std::size_t j = 0; j < M; ++j) {
            out[2 * j] = z[j].real() * inv_m;
            out[2 * j + 1] = z[j].imag() * inv_m;
        }
    }

private:
    std::size_t n_, nh_;
    bool even_;
    scalar_c2c<T> eng_;
    std::vector<std::complex<T>> tw_;                     // W_N^k forward sign
    mutable std::vector<std::vector<std::complex<T>>> buf_;  // pack buffer, per `tid`
};

// `plan_state<long double>`. The interface mirrors `plan_state<T>`; effort
// and debug settings have no counterpart here and are ignored.
template<typename T>
struct scalar_plan_state {
    scalar_plan_state(span<const std::size_t> shape, std::size_t nthreads)
        : plan(shape, axis_count(shape), nthreads),
          pool_(nthreads > 1 ? std::make_unique<thread_pool>(nthreads) : nullptr) {}
    [[nodiscard]] std::size_t size() const noexcept { return plan.size(); }
    // `fct == nullptr` takes the direction's default: 1 forward, 1/N inverse.
    void run(bool is_forward, std::complex<T>* data, const T* fct) const {
        const T s = fct ? *fct : (is_forward ? T(1) : T(1) / static_cast<T>(plan.size()));
        if (is_forward) plan.template execute<true>(data, s, pool_.get());
        else plan.template execute<false>(data, s, pool_.get());
    }
    void run(bool is_forward, const std::complex<T>* src, std::complex<T>* dst,
             const T* fct) const {
        if (src != dst) std::copy_n(src, plan.size(), dst);
        run(is_forward, dst, fct);
    }

private:
    // Transform every axis, and reject the empty shape the API cannot plan.
    static std::size_t axis_count(span<const std::size_t> shape) {
        if (shape.empty()) throw std::invalid_argument("Plan size must be greater than 0");
        return shape.size();
    }
    scalar_nd_c2c<T> plan;
    std::unique_ptr<thread_pool> pool_;
};

// `real_state<long double>`: r2c/c2r on the last axis, then c2c over the
// remaining axes of the half-spectrum, in `nd_real_plan`'s order.
template<typename T>
struct scalar_real_state {
    scalar_real_state(span<const std::size_t> shape, std::size_t nthreads)
        : n_(last_extent(shape)),
          nh_(n_ / 2 + 1),
          rows_(outer_extent_product(shape)),
          inner_(n_, nthreads),
          outer_(half_spectrum_shape(shape, nh_), shape.size() - 1, nthreads),
          pool_(nthreads > 1 ? std::make_unique<thread_pool>(nthreads) : nullptr) {}

    [[nodiscard]] std::size_t real_size() const noexcept { return rows_ * n_; }
    [[nodiscard]] std::size_t cplx_size() const noexcept { return rows_ * nh_; }

    void forward(const T* in, std::complex<T>* out, std::optional<T> fct) const {
        admiral::detail::parallel_for(
            pool_.get(), rows_, real_size(),
            [&](std::size_t begin, std::size_t end, std::size_t tid) {
                for (std::size_t r = begin; r < end; ++r)
                    inner_.forward(in + r * n_, out + r * nh_, tid);
            });
        outer_.template execute<true>(out, fct.value_or(T(1)), pool_.get());
    }

    void inverse(std::complex<T>* spec, T* out, std::optional<T> fct) const {
        // The unscaled r2c makes the inner c2r an exact inverse, so the API's
        // 1/Ntot splits across both stages. The outer axes take `1/rows_`, and
        // the inner sweep divides by the engine's own length (`n_` odd, `n_/2`
        // even).
        outer_.template execute<false>(spec, T(1) / static_cast<T>(rows_), pool_.get());
        // `fct` convention as in `nd_real_plan`: read against the 1/Ntot
        // default, so the caller's `fct` lands as `fct * real_size()` folded
        // into the per-row inner scale.
        const T s = fct ? *fct * static_cast<T>(real_size()) : T(1);
        admiral::detail::parallel_for(
            pool_.get(), rows_, real_size(),
            [&](std::size_t begin, std::size_t end, std::size_t tid) {
                for (std::size_t r = begin; r < end; ++r)
                    inner_.inverse(spec + r * nh_, out + r * n_, s, tid);
            });
    }

private:
    static std::size_t last_extent(span<const std::size_t> shape) {
        if (shape.empty() || shape.back() == 0)
            throw std::invalid_argument("Plan size must be greater than 0");
        return shape.back();
    }
    // Rows of the real tensor: every extent but the last one.
    static std::size_t outer_extent_product(span<const std::size_t> shape) {
        std::size_t rows = 1;
        for (std::size_t d = 0; d + 1 < shape.size(); ++d) rows *= shape[d];
        return rows;
    }
    // The complex tensor the r2c stage writes: the real shape with the last
    // axis halved.
    static std::vector<std::size_t> half_spectrum_shape(span<const std::size_t> shape,
                                                        std::size_t nh) {
        std::vector<std::size_t> out(shape.begin(), shape.end() - 1);
        out.push_back(nh);
        return out;
    }

    std::size_t n_, nh_, rows_;
    scalar_r2c_1d<T> inner_;
    scalar_nd_c2c<T> outer_;
    std::unique_ptr<thread_pool> pool_;
};

}  // namespace detail
}  // namespace admiral
