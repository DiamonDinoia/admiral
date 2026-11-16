#pragma once

// Rader's algorithm for prime N in the dd (double-double) engine.
// Replaces Bluestein when N-1 is 7-smooth (the dd four-step handles it directly).
// Two FFT_{N-1} transforms via the inner dd_c2c replace the two FFT_{pad} of Bluestein.

#include <algorithm>
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
        scratch_.resize(L_);
        bhat_fwd_.resize(L_);
        bhat_inv_.resize(L_);
        build_bhat();
    }

    [[nodiscard]] std::size_t size() const { return p_; }

    template<bool forward>
    void execute(std::complex<T>* x, T fct, std::size_t tid, thread_pool* pool) const {
        const std::complex<T> x0 = x[0];
        std::complex<T> sum = x0;

        // Permute: scratch_[q] = x[gpow_[q]]
        for (std::size_t q = 0; q < L_; ++q) {
            scratch_[q] = x[gpow_[q]];
            sum += scratch_[q];
        }

        // Forward FFT on the permuted data
        inner_->template execute<true>(scratch_.data(), T(1), tid, pool);

        // Pointwise multiply by the Rader kernel
        const auto& bhat = forward ? bhat_fwd_ : bhat_inv_;
        for (std::size_t k = 0; k < L_; ++k)
            scratch_[k] *= bhat[k];

        // Inverse FFT (includes the 1/L scale)
        inner_->template execute<false>(scratch_.data(), T(1) / T(L_), tid, pool);

        // Scatter back: out[g^t] = (x0 + result[t]) * fct
        x[0] = sum * fct;
        for (std::size_t mm = 0; mm < L_; ++mm)
            x[ginvpow_[mm]] = (x0 + scratch_[mm]) * fct;
    }

private:
    void build_perms() {
        const std::size_t g = ct_primitive_root(p_);
        gpow_.resize(L_);
        ginvpow_.resize(L_);
        gpow_[0] = 1 % p_;
        for (std::size_t q = 1; q < L_; ++q) gpow_[q] = ct_mul_mod(gpow_[q - 1], g, p_);
        for (std::size_t q = 0; q < L_; ++q) ginvpow_[q] = gpow_[(L_ - q) % L_];
    }

    void build_bhat() {
        const dd_roots w(p_);
        for (std::size_t j = 0; j < L_; ++j) {
            const dd_const c = w(ginvpow_[j]);
            const T re = static_cast<T>(static_cast<long double>(c.rh) + static_cast<long double>(c.rl));
            const T im = static_cast<T>(static_cast<long double>(c.ih) + static_cast<long double>(c.il));
            bhat_fwd_[j] = {re, im};
            bhat_inv_[j] = {re, -im};
        }
        inner_->template execute<true>(bhat_fwd_.data(), T(1), 0, nullptr);
        inner_->template execute<true>(bhat_inv_.data(), T(1), 0, nullptr);
    }

    std::size_t p_, L_;
    std::vector<std::size_t> gpow_, ginvpow_;
    std::vector<std::complex<T>> bhat_fwd_, bhat_inv_;
    mutable std::vector<std::complex<T>> scratch_;
    std::optional<dd_c2c<T>> inner_;
};
