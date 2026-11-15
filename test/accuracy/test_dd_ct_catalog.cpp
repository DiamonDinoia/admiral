// Test: compile-time dd root tables match run-time build within 1e-30 relative.
// Tests dd_fft.hpp's ct_roots_v<N> against the run-time dd_roots_runtime (the
// original build kept verbatim). Coverage: catalog sizes 9, 18, 30, 32, 50, 60, 64, 100, 128, 256.

#include <catch2/catch_test_macros.hpp>
#include "admiral/detail/dd_fft.hpp"

namespace adm = admiral;
namespace adm_d = admiral::detail;

TEST_CASE("dd_roots: compile-time tables equal run-time build") {
    // Run-time build reference: uses dd_anchor with sincos_turns_eval<ct_dd_plain>
    // and cmul recurrence. Same interface as dd_roots.
    class dd_roots_runtime {
    public:
        explicit dd_roots_runtime(std::size_t n) : n_(n) {
            b_ = adm_d::dd_root_block(n_);
            const std::size_t nc = (n_ + b_ - 1) / b_;
            coarse_.resize(nc);
            fine_.resize(b_);
            const adm_d::ct_sincos_v<adm_d::ct_dd_plain> one{adm_d::ct_dd_plain(0), adm_d::ct_dd_plain(1)};
            coarse_[0] = fine_[0] = one;
            if (b_ > 1) {
                const adm_d::dd_anchor_pair a = adm_d::dd_anchor(n_);
                fine_[1] = a.fine;
                for (std::size_t e = 2; e < b_; ++e) fine_[e] = cmul(fine_[e - 1], fine_[1]);
                if (nc > 1) {
                    coarse_[1] = a.coarse;
                    for (std::size_t e = 2; e < nc; ++e) coarse_[e] = cmul(coarse_[e - 1], coarse_[1]);
                }
            }
        }

        [[nodiscard]] adm_d::dd_const operator()(std::size_t e) const {
            e %= n_;
            const adm_d::ct_sincos_v<adm_d::ct_dd_plain>& a = coarse_[e / b_];
            const adm_d::ct_sincos_v<adm_d::ct_dd_plain>& b = fine_[e % b_];
            return adm_d::dd_split(cmul(a, b));
        }

    private:
        static adm_d::ct_sincos_v<adm_d::ct_dd_plain> cmul(const adm_d::ct_sincos_v<adm_d::ct_dd_plain>& a,
                                                         const adm_d::ct_sincos_v<adm_d::ct_dd_plain>& b) {
            return {a.s * b.c + a.c * b.s, a.c * b.c - a.s * b.s};
        }
        std::size_t n_, b_ = 1;
        std::vector<adm_d::ct_sincos_v<adm_d::ct_dd_plain>> coarse_, fine_;
    };

    // Catalog sizes to test: 9, 18, 30, 32, 50, 60, 64, 100, 128, 256
    constexpr std::size_t catalog_sizes[] = {9, 18, 30, 32, 50, 60, 64, 100, 128, 256};

    for (std::size_t n : catalog_sizes) {
        const adm_d::dd_roots ct(n);
        const dd_roots_runtime rt(n);

        // Compare every entry
        double max_rel_diff = 0.0;
        for (std::size_t e = 0; e < n; ++e) {
            const adm_d::dd_const c = ct(e);
            const adm_d::dd_const r = rt(e);
            // Relative diff for each component
            for (int i = 0; i < 4; ++i) {
                const double* cp = reinterpret_cast<const double*>(&c);
                const double* rp = reinterpret_cast<const double*>(&r);
                const double abs_r = std::fabs(rp[i]);
                if (abs_r > 0) {
                    const double diff = std::fabs(cp[i] - rp[i]) / abs_r;
                    if (diff > max_rel_diff) max_rel_diff = diff;
                }
            }
        }

        // Check within 1e-30 relative
        CAPTURE(n, max_rel_diff);
        REQUIRE(max_rel_diff <= 1e-30);
    }
}

TEST_CASE("dd_roots: perturbed table fails comparison") {
    // Positive control: a table with one entry perturbed must fail the 1e-30 check
    // We build a small perturbation manually for one catalog size.
    
    // Build a reference for N=18 using the run-time build
    class dd_roots_runtime {
    public:
        explicit dd_roots_runtime(std::size_t n) : n_(n) {
            b_ = adm_d::dd_root_block(n_);
            const std::size_t nc = (n_ + b_ - 1) / b_;
            coarse_.resize(nc);
            fine_.resize(b_);
            const adm_d::ct_sincos_v<adm_d::ct_dd_plain> one{adm_d::ct_dd_plain(0), adm_d::ct_dd_plain(1)};
            coarse_[0] = fine_[0] = one;
            if (b_ > 1) {
                const adm_d::dd_anchor_pair a = adm_d::dd_anchor(n_);
                fine_[1] = a.fine;
                for (std::size_t e = 2; e < b_; ++e) fine_[e] = cmul(fine_[e - 1], fine_[1]);
                if (nc > 1) {
                    coarse_[1] = a.coarse;
                    for (std::size_t e = 2; e < nc; ++e) coarse_[e] = cmul(coarse_[e - 1], coarse_[1]);
                }
            }
        }
        [[nodiscard]] adm_d::dd_const operator()(std::size_t e) const {
            e %= n_;
            const adm_d::ct_sincos_v<adm_d::ct_dd_plain>& a = coarse_[e / b_];
            const adm_d::ct_sincos_v<adm_d::ct_dd_plain>& b = fine_[e % b_];
            return adm_d::dd_split(cmul(a, b));
        }
    private:
        static adm_d::ct_sincos_v<adm_d::ct_dd_plain> cmul(const adm_d::ct_sincos_v<adm_d::ct_dd_plain>& a,
                                                         const adm_d::ct_sincos_v<adm_d::ct_dd_plain>& b) {
            return {a.s * b.c + a.c * b.s, a.c * b.c - a.s * b.s};
        }
        std::size_t n_, b_ = 1;
        std::vector<adm_d::ct_sincos_v<adm_d::ct_dd_plain>> coarse_, fine_;
    };

    const dd_roots_runtime ref(18);
    const adm_d::dd_roots ct(18);
    
    // Perturb one entry of the run-time reference (add 1e-29 to rh component of entry 0)
    adm_d::dd_const perturbed[18];
    for (std::size_t e = 0; e < 18; ++e) {
        perturbed[e] = ref(e);
    }
    // Apply perturbation to entry 1: change rh by 1e-28 (large enough to exceed 1e-30 relative)
    adm_d::dd_const p1 = ref(1);
    double* pp1 = reinterpret_cast<double*>(&p1);
    double* ppert = reinterpret_cast<double*>(&perturbed[1]);
    // Assume p1.rh is around 1.0 (cos(2pi*1/18) ~ 0.8090), so 1e-28 added is ~1e-28 relative
    ppert[0] = pp1[0] + 1e-28;
    
    // Now compare: the perturbed version should fail the 1e-30 check at entry 1
    double max_rel_diff = 0.0;
    for (std::size_t e = 0; e < 18; ++e) {
        const adm_d::dd_const c = ct(e);
        const adm_d::dd_const r = perturbed[e];
        for (int i = 0; i < 4; ++i) {
            const double* cp = reinterpret_cast<const double*>(&c);
            const double* rp = reinterpret_cast<const double*>(&r);
            const double abs_r = std::fabs(rp[i]);
            if (abs_r > 0) {
                const double diff = std::fabs(cp[i] - rp[i]) / abs_r;
                if (diff > max_rel_diff) max_rel_diff = diff;
            }
        }
    }
    
    // The perturbed table should fail the 1e-30 bound
    CAPTURE(max_rel_diff);
    REQUIRE(max_rel_diff > 1e-30);
}
