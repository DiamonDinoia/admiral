#pragma once

// The V-generic butterflies of butterfly.hpp at V = T, for long double, which has no SIMD lanes.

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

#include "butterfly.hpp"
#include "cxx_compat.hpp"
#include "math.hpp"
#include "real_recombine.hpp"
#include "scratch.hpp"
#include "thread_pool.hpp"

namespace admiral {
namespace detail {

inline constexpr std::size_t kDirectMax = 37;

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

template<typename T>
class scalar_c2c {
public:
    explicit scalar_c2c(std::size_t n, std::size_t nthreads = 1) : n_(n), nthreads_(nthreads) {
        if (n_ > 1 && n_ <= kDirectMax) {
            build_direct(n_);
            return;
        }
        for (std::size_t m = n_; m > 1;) {
            const std::size_t p = factor(m);
            if (p == 0) {
                if (m <= kDirectMax) {
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
        if (levels_.empty()) return;
        build_perm();
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
        if (blue_ && levels_.empty()) {
            bluestein<forward>(x, n_, fct, tid);
            return;
        }
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
            dft<forward>(x, 0, 0, tid);
        }
        soa_scratch<std::complex<T>, 1> sx_sc(n_);
        std::complex<T>* sx = sx_sc.buf(0);
        for (std::size_t k = 0; k < n_; ++k) sx[k] = x[pos_of_[k]];
        if (fct != T(1))
            for (std::size_t k = 0; k < n_; ++k) x[k] = sx[k] * fct;
        else
            for (std::size_t k = 0; k < n_; ++k) x[k] = sx[k];
    }

private:
    static std::size_t factor(std::size_t m) {
        for (const std::size_t p : {8u, 3u, 5u, 7u, 4u, 2u})
            if (m % p == 0) return p;
        return 0;
    }

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

    template<bool forward, std::size_t P>
    void combine(std::complex<T>* x, std::size_t off, std::size_t level) const {
        constexpr bool sw = !forward;
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
                if constexpr (sc != 0) {
                    const std::complex<T> w = tw[q * sc];
                    const T r = yr * w.real() - yi * w.imag();
                    yi = yr * w.imag() + yi * w.real();
                    yr = r;
                }
                x[off + q + sc * u] = sw ? std::complex<T>(yi, yr) : std::complex<T>(yr, yi);
            });
        }
    }

    using radix_set = std::integer_sequence<std::size_t, 2, 3, 4, 5, 7, 8>;

    template<bool forward>
    struct combine_invoke_t {
        template<std::size_t P>
        void operator()(const scalar_c2c* self, std::complex<T>* x, std::size_t off,
                        std::size_t level) const {
            self->template combine<forward, P>(x, off, level);
        }
    };

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
        }
        static std::size_t pad_size(std::size_t n) {
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
        std::vector<std::complex<T>> chirp;
        std::vector<std::complex<T>> bfft;
        std::vector<std::complex<T>> bfft_inv;
    };

    template<bool forward>
    void bluestein(std::complex<T>* x, std::size_t m, T fct, std::size_t tid) const {
        const blue_state& B = *blue_;
        const std::size_t pad = B.pad;
        soa_scratch<std::complex<T>, 1> a_sc(pad);
        std::complex<T>* a = a_sc.buf(0);
        std::fill_n(a, pad, std::complex<T>(0, 0));
        for (std::size_t k = 0; k < m; ++k) a[k] = x[k] * maybe_conj<T>(B.chirp[k], !forward);
        B.inner.template execute<true>(a, T(1), tid);
        const auto& bfft = forward ? B.bfft : B.bfft_inv;
        for (std::size_t i = 0; i < pad; ++i) a[i] *= bfft[i];
        B.inner.template execute<false>(a, T(1), tid);
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
    std::size_t direct_n_ = 0;
    std::vector<std::complex<T>> direct_mat_;
    std::size_t nthreads_;
};

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
            axis_engine_.push_back(e);
        }
        line_cap_ = 1;
        for (std::size_t d = 0; d < active_; ++d) line_cap_ = std::max(line_cap_, shape_[d]);
    }

    [[nodiscard]] std::size_t size() const { return total_; }

    template<bool Forward>
    void execute(std::complex<T>* data, T fct = T(1), thread_pool* pool = nullptr) const {
        for (std::size_t d = 0; d < active_; ++d) {
            const std::size_t len = shape_[d], st = stride_[d];
            const scalar_c2c<T>& eng = eng_[axis_engine_[d]];
            const T line_fct = (d == active_ - 1) ? fct : T(1);
            const std::size_t pre = total_ / (len * st);
            const std::size_t nlines = pre * st;
            thread_pool* eng_pool =
                will_thread(pool, nlines, total_) ? nullptr : pool;
            admiral::detail::parallel_for(
                pool, nlines, total_,
                [&](std::size_t begin, std::size_t end, std::size_t tid) {
                    soa_scratch<std::complex<T>, 1> line_sc(line_cap_);
                    auto line = line_sc.buf(0);
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
    std::size_t line_cap_;
    std::size_t total_, active_;
};

template<typename T>
class scalar_r2c_1d {
public:
    explicit scalar_r2c_1d(std::size_t n, std::size_t nthreads = 1)
        : n_(n), nh_(n / 2 + 1), even_(n % 2 == 0), eng_(even_ ? n / 2 : n, nthreads) {
        tw_.resize(nh_);
        for (std::size_t k = 0; k < nh_; ++k) tw_[k] = scalar_twiddle<T>(k, n_, true);
    }

    [[nodiscard]] std::size_t real_size() const { return n_; }
    [[nodiscard]] std::size_t cplx_size() const { return nh_; }

    void forward(const T* in, std::complex<T>* out, std::size_t tid) const {
        soa_scratch<std::complex<T>, 1> buf_sc(even_ ? n_ / 2 : n_);
        auto buf = buf_sc.buf(0);
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

    void inverse(std::complex<T>* spec, T* out, T extra_scale, std::size_t tid) const {
        soa_scratch<std::complex<T>, 1> buf_sc(even_ ? n_ / 2 : n_);
        auto buf = buf_sc.buf(0);
        if (!even_) {
            for (std::size_t k = 0; k < nh_; ++k) buf[k] = spec[k];
            for (std::size_t k = nh_; k < n_; ++k) buf[k] = std::conj(spec[n_ - k]);
            eng_.template execute<false>(buf, T(1), tid);
            const T inv_n = extra_scale / T(n_);
            for (std::size_t i = 0; i < n_; ++i) out[i] = buf[i].real() * inv_n;
            return;
        }
        const std::size_t M = n_ / 2;
        std::complex<T>* z = buf;
        for (std::size_t k = 0; k < M; ++k) z[k] = c2r_even_bin(spec, tw_[k], M, k);
        eng_.template execute<false>(z, T(1), tid);
        const T inv_m = extra_scale / T(M);
        for (std::size_t j = 0; j < M; ++j) {
            out[2 * j] = z[j].real() * inv_m;
            out[2 * j + 1] = z[j].imag() * inv_m;
        }
    }

private:
    std::size_t n_, nh_;
    bool even_;
    scalar_c2c<T> eng_;
    std::vector<std::complex<T>> tw_;
};

template<typename T>
[[nodiscard]] inline std::size_t scalar_resolve(span<const std::size_t> shape,
                                                std::size_t nthreads) {
    if (nthreads != 0) return nthreads;
    std::size_t total = 1, dispatches = 0;
    double work_cyc = 0.0;
    for (const std::size_t d : shape) {
        if (d == 0) return 1;
        total = sat_elems(total, d);
        if (d > 1) {
            ++dispatches;
            work_cyc += double(total / d) * line_work_cyc<T>(d);
        }
    }
    return resolve_nthreads(0, total, dispatches, work_cyc / core_cyc_per_ns(),
                            shape.size() >= 3 ? 2 : 1);
}

template<typename T>
struct scalar_plan_state {
    scalar_plan_state(span<const std::size_t> shape, std::size_t nthreads)
        : scalar_plan_state(shape, scalar_resolve<T>(shape, nthreads), resolved_tag{}) {}

private:
    struct resolved_tag {};
    scalar_plan_state(span<const std::size_t> shape, std::size_t nthreads, resolved_tag)
        : plan(shape, axis_count(shape), nthreads),
          pool_(nthreads > 1 ? std::make_unique<thread_pool>(nthreads) : nullptr) {}

public:
    [[nodiscard]] std::size_t size() const noexcept { return plan.size(); }
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
    static std::size_t axis_count(span<const std::size_t> shape) {
        if (shape.empty()) throw std::invalid_argument("Plan size must be greater than 0");
        return shape.size();
    }
    scalar_nd_c2c<T> plan;
    std::unique_ptr<thread_pool> pool_;
};

template<typename T>
[[nodiscard]] inline std::size_t scalar_real_resolve(span<const std::size_t> shape,
                                                     std::size_t nthreads) {
    if (nthreads != 0) return nthreads;
    if (shape.empty() || shape.back() == 0) return 1;
    std::size_t rows = 1;
    double work_cyc = 0.0;
    std::size_t dispatches = 1;
    for (std::size_t d = 0; d + 1 < shape.size(); ++d) {
        if (shape[d] == 0) return 1;
        rows = sat_elems(rows, shape[d]);
        if (shape[d] > 1) {
            ++dispatches;
            work_cyc += double(rows / shape[d]) * double(shape.back() / 2 + 1) *
                       line_work_cyc<T>(shape[d]);
        }
    }
    const std::size_t nh = shape.back() / 2 + 1;
    work_cyc += double(rows) * line_work_cyc<T>(nh);
    return resolve_nthreads(0, sat_elems(rows, shape.back()), dispatches,
                            work_cyc / core_cyc_per_ns(), shape.size() >= 3 ? 2 : 1);
}

template<typename T>
struct scalar_real_state {
    scalar_real_state(span<const std::size_t> shape, std::size_t nthreads)
        : scalar_real_state(shape, scalar_real_resolve<T>(shape, nthreads), resolved_tag{}) {}

private:
    struct resolved_tag {};
    scalar_real_state(span<const std::size_t> shape, std::size_t nthreads, resolved_tag)
        : n_(last_extent(shape)),
          nh_(n_ / 2 + 1),
          rows_(outer_extent_product(shape)),
          inner_(n_, nthreads),
          outer_(half_spectrum_shape(shape, nh_), shape.size() - 1, nthreads),
          pool_(nthreads > 1 ? std::make_unique<thread_pool>(nthreads) : nullptr) {}

public:

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
        outer_.template execute<false>(spec, T(1) / static_cast<T>(rows_), pool_.get());
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
    static std::size_t outer_extent_product(span<const std::size_t> shape) {
        std::size_t rows = 1;
        for (std::size_t d = 0; d + 1 < shape.size(); ++d) rows *= shape[d];
        return rows;
    }
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

}
}
