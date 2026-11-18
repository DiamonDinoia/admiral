#pragma once

// Rader's algorithm for prime N in the dd (double-double) engine.
// Replaces Bluestein when N-1 is a power of two: the two length N-1 transforms then run the
// four-step with no mixed-radix waste, and N-1 stays below Bluestein's pad.
// The route runs on one dd grid: the load and the scatter are the only long double boundaries.

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <vector>

#include "ct_math.hpp"

// Included inside namespace admiral::detail (from dd_fft.hpp).

template<typename T>
class dd_rader {
public:
    explicit dd_rader(std::size_t p) : p_(p), L_(p - 1) {
        inner_.emplace(L_, dd_engine_tag{});
        build_perms();
        build_bhat();
    }

    [[nodiscard]] std::size_t size() const { return p_; }

    // The route at the scale fct / div: the scatter store folds the scale in without a rounded
    // long double factor. The grid carries the convolution, which holds at most p times the
    // largest input.
    template<bool forward>
    void execute(std::complex<T>* x, T fct, std::size_t div, thread_pool* pool) const {
        const dd_grid g(ld_max_exponent(x, p_), p_, fct, 0, div);
        const std::size_t nb = (L_ + kDdW - 1) / kDdW;
        auto z_sc = make_aligned_buffer<double>(nb * kDdElem);
        double* z = z_sc.get();
        // The DC bin is the transform's bin 0 (the sum of the gathered inputs) plus x0.
        double bin0[4] = {};

        double x0r[2], x0i[2];
        grid_ld(x[0].real(), g.s, x0r);
        grid_ld(x[0].imag(), g.s, x0i);
        const gdd x0g_r{dd_batch(x0r[0]), dd_batch(x0r[1])};
        const gdd x0g_i{dd_batch(x0i[0]), dd_batch(x0i[1])};

        // Forward FFT of the permuted inputs; the sink multiplies by the kernel spectrum, whose
        // table carries the 1 / L of the convolution, and stages the products in z.
        const double* bhat = forward ? bhat_fwd_.get() : bhat_inv_.get();
        inner_->transform(
            [&](std::size_t j, std::size_t lanes, double* dst) {
                std::complex<T> w[kDdW];
                for (std::size_t l = 0; l < lanes; ++l) w[l] = x[gpow_[j + l]];
                dd_load_ld<true>(w, lanes, g.s, dst);
            },
            [&](std::size_t k, std::size_t, double* src) {
                if (k == 0) {
                    bin0[0] = src[0];
                    bin0[1] = src[kDdW];
                    bin0[2] = src[2 * kDdW];
                    bin0[3] = src[3 * kDdW];
                }
                dd_cmul_table(bhat + k / kDdW * kDdElem, src);
                std::copy_n(src, kDdElem, z + k / kDdW * kDdElem);
            },
            pool);
        // The unscaled inverse of the products: the engine runs forward, and the swaps at the
        // two boundaries conjugate around it. The scatter adds x0 on the grid and stores.
        inner_->transform(
            [&](std::size_t j, std::size_t, double* dst) {
                std::copy_n(z + j / kDdW * kDdElem, kDdElem, dst);
                for (std::size_t l = 0; l < kDdW; ++l) {
                    std::swap(dst[l], dst[2 * kDdW + l]);
                    std::swap(dst[kDdW + l], dst[3 * kDdW + l]);
                }
            },
            [&](std::size_t k, std::size_t lanes, double* src) {
                gdd re, im;
                dd_load(src, re, im);
                std::swap(re, im);
                dd_store(src, re + x0g_r, im + x0g_i);
                std::complex<T> w[kDdW];
                dd_store_ld<true>(src, g, w, lanes);
                for (std::size_t l = 0; l < lanes; ++l) x[ginvpow_[k + l]] = w[l];
            },
            pool);
        alignas(dd_batch::arch_type::alignment()) double sp[kDdElem] = {};
        sp[0] = bin0[0] + x0r[0];
        sp[kDdW] = bin0[1] + x0r[1];
        sp[2 * kDdW] = bin0[2] + x0i[0];
        sp[3 * kDdW] = bin0[3] + x0i[1];
        std::complex<T> dc[kDdW];
        dd_store_ld<true>(sp, g, dc, 1);
        x[0] = dc[0];
    }

private:
    // hi + lo = v exactly.
    static void split_ld(T v, double& h, double& l) {
        h = static_cast<double>(v);
        l = static_cast<double>(v - static_cast<T>(h));
    }

    // The grid pair of v: the hi half an integer number of grid units, the residual in lo, so
    // the plain pair adds of the engine stay exact. Exact as a whole.
    static void grid_ld(T v, int s, double out[2]) {
        double h, l;
        split_ld(v, h, l);
        const double a = std::ldexp(h, s), r = std::nearbyint(a);
        out[0] = r;
        out[1] = (a - r) + std::ldexp(l, s);
    }

    void build_perms() {
        const std::size_t g = ct_primitive_root(p_);
        gpow_.resize(L_);
        ginvpow_.resize(L_);
        gpow_[0] = 1 % p_;
        for (std::size_t q = 1; q < L_; ++q) gpow_[q] = ct_mul_mod(gpow_[q - 1], g, p_);
        for (std::size_t q = 0; q < L_; ++q) ginvpow_[q] = gpow_[(L_ - q) % L_];
    }

    // The kernel spectra on the build grid sb, then true values carrying the 1 / L of the
    // convolution: every product constant stays below 1, and the products of the route keep
    // the grid bound.
    void build_bhat() {
        const std::size_t nb = (L_ + kDdW - 1) / kDdW, elems = nb * kDdElem;
        bhat_fwd_ = make_aligned_buffer<double>(elems);
        bhat_inv_ = make_aligned_buffer<double>(elems);
        auto kern = make_aligned_buffer<double>(elems);
        auto conj = make_aligned_buffer<double>(elems);
        std::fill_n(kern.get(), elems, 0.0);
        std::fill_n(conj.get(), elems, 0.0);
        const int sb = 48 - bit_width(L_ - 1);
        const double sb_scale = std::ldexp(1.0, sb);
        const auto put = [sb_scale](double* p, double h, double l) {
            const double v = h * sb_scale, r = std::nearbyint(v);
            p[0] = r;
            p[kDdW] = (v - r) + l * sb_scale;
        };
        const dd_roots w(p_);
        for (std::size_t j = 0; j < L_; ++j) {
            const dd_const c = w(ginvpow_[j]);
            double* fk = kern.get() + j / kDdW * kDdElem + j % kDdW;
            double* fj = conj.get() + j / kDdW * kDdElem + j % kDdW;
            put(fk, c.rh, c.rl);
            put(fk + 2 * kDdW, c.ih, c.il);
            put(fj, c.rh, c.rl);
            put(fj + 2 * kDdW, -c.ih, -c.il);
        }
        const auto fill = [&](const aligned_buffer<double>& src, aligned_buffer<double>& dst) {
            inner_->transform(
                [&](std::size_t j, std::size_t, double* p) {
                    std::copy_n(src.get() + j / kDdW * kDdElem, kDdElem, p);
                },
                [&](std::size_t k, std::size_t, double* p) {
                    std::copy_n(p, kDdElem, dst.get() + k / kDdW * kDdElem);
                },
                nullptr);
            const ct_dd scale = ct_dd(std::ldexp(1.0, -sb)) / ct_dd(L_);
            for (std::size_t k = 0; k < L_; ++k) {
                double* e = dst.get() + k / kDdW * kDdElem + k % kDdW;
                const ct_dd re = dd_two_sum(e[0], e[kDdW]) * scale;
                const ct_dd im = dd_two_sum(e[2 * kDdW], e[3 * kDdW]) * scale;
                e[0] = re.hi;
                e[kDdW] = re.lo;
                e[2 * kDdW] = im.hi;
                e[3 * kDdW] = im.lo;
            }
        };
        fill(kern, bhat_fwd_);
        fill(conj, bhat_inv_);
    }

    std::size_t p_, L_;
    std::vector<std::size_t> gpow_, ginvpow_;
    aligned_buffer<double> bhat_fwd_, bhat_inv_;
    std::optional<dd_c2c<T>> inner_;
};
