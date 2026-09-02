
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "utils/reference.hpp"

#include <admiral/detail/codelet.hpp>
#include <admiral/detail/four_step.hpp>
#include <admiral/detail/math.hpp>

#include <cmath>
#include <complex>
#include <vector>

using namespace admiral::detail;

static_assert(cofactor_simd_profitable<double, 4>(cofactor_batch_width<double, 4>() + 2));
static_assert(cofactor_simd_profitable<double, 4>(3 * cofactor_batch_width<double, 4>()));
static_assert(!cofactor_simd_profitable<double, 4>(cofactor_batch_width<double, 4>() - 2));
static_assert(cofactor_simd_profitable<float, 8>(cofactor_batch_width<float, 8>() + 2));
static_assert(cofactor_simd_profitable<float, 8>(3 * cofactor_batch_width<float, 8>()));
static_assert(!cofactor_simd_profitable<float, 8>(cofactor_batch_width<float, 8>() - 2));
constexpr bool kMasked256F64 = cofactor_batch_width<double, 3>() * sizeof(double) > 16;
static_assert(kMasked256F64 == !cofactor_simd_profitable<double, 3>(8));
static_assert(cofactor_simd_profitable<double, 3>(10));
static_assert(cofactor_simd_profitable<float, 3>(cofactor_batch_width<float, 3>()));

namespace {

template<unsigned N, bool Forward, typename T>
void check_one() {
    const auto x = make_input<T>(N);
    std::vector<T> xre(N), xim(N), yre(N), yim(N);
    for (std::size_t i = 0; i < N; ++i) {
        xre[i] = x[i].real();
        xim[i] = x[i].imag();
    }

    kernel<N, T, Forward>::apply(xre.data(), xim.data(), 1, yre.data(), yim.data());

    const auto ref = reference_dft<T>(x, Forward);

    std::vector<std::complex<T>> got(N);
    for (std::size_t k = 0; k < N; ++k) got[k] = {yre[k], yim[k]};
    require_close(got, ref, fft_tol<T>());
}

template<unsigned N>
void check_size() {
    check_one<N, true, double>();
    check_one<N, false, double>();
    check_one<N, true, float>();
    check_one<N, false, float>();
}

}

TEST_CASE("codelet kernel<N> matches reference DFT", "[codelet]") {
    check_size<1>();
    check_size<2>();
    check_size<3>();
    check_size<4>();
    check_size<5>();
    check_size<6>();
    check_size<7>();
    check_size<8>();
    check_size<9>();
    check_size<10>();
    check_size<12>();
    check_size<15>();
    check_size<16>();
    check_size<24>();
    check_size<30>();
    check_size<32>();
    check_size<36>();
    check_size<48>();
    check_size<52>();
    check_size<60>();
    check_size<64>();
    check_size<120>();
    check_size<128>();
}

TEST_CASE("codelet kernel<N> small primes (direct DFT path)", "[codelet]") {
    check_size<11>();
    check_size<13>();
}

TEST_CASE("a small non-pow2 codelet takes the flat leaf", "[codelet]") {
    if constexpr (poet::vector_register_count() == 32) {
        CHECK(kernel_batched<11, double, true>::flat_leaf);
        CHECK_FALSE(kernel_batched<13, double, true>::flat_leaf);
        CHECK(kernel_batched<16, double, true>::flat_leaf);
    }
    CHECK_FALSE(kernel_batched<CODELET_CATALOG_MAX, double, true>::flat_leaf);
}

TEST_CASE("the scalar flat leaf yields to a larger cofactor", "[codelet]") {
    if constexpr (poet::vector_register_count() == 32) {
        CHECK(kernel<11, double, true>::flat_leaf);
        CHECK(kernel<12, double, true>::flat_leaf);
        CHECK_FALSE(kernel<10, double, true>::flat_leaf);
    }
    CHECK_FALSE(kernel<16, double, true>::flat_leaf);
    CHECK_FALSE(kernel<CODELET_CATALOG_MAX, double, true>::flat_leaf);
}

TEST_CASE("codelet kernel<N> prime composites (cofactor-SIMD batched path)", "[codelet]") {
    check_size<26>();
    check_size<34>();
    check_size<38>();
    check_size<39>();
    check_size<46>();
    check_size<51>();
    check_size<57>();
    check_size<58>();
    check_size<62>();
}

template<unsigned N1, unsigned N2, typename T>
void check_four_step() {
    constexpr unsigned N = N1 * N2;
    const auto x = make_input<T>(N);
    std::vector<T> in_re(N), in_im(N), out_re(N), out_im(N), Gre(N), Gim(N);
    for (std::size_t i = 0; i < N; ++i) { in_re[i] = x[i].real(); in_im[i] = x[i].imag(); }

    auto run = [&](bool fwd) {
        constexpr unsigned W = static_cast<unsigned>(xsimd::batch<T>::size);
        if (fwd) {
            const auto tw = admiral::detail::build_four_step_twiddles_v<T, true>(N1, N2, W);
            std::vector<T> twre(N), twim(N);
            for (std::size_t i = 0; i < N; ++i) { twre[i] = tw[i].real(); twim[i] = tw[i].imag(); }
            admiral::detail::four_step_batched_ct<N1, N2, T, true>(
                in_re.data(), in_im.data(), out_re.data(), out_im.data(),
                twre.data(), twim.data(), Gre.data(), Gim.data());
        } else {
            const auto tw = admiral::detail::build_four_step_twiddles_v<T, false>(N1, N2, W);
            std::vector<T> twre(N), twim(N);
            for (std::size_t i = 0; i < N; ++i) { twre[i] = tw[i].real(); twim[i] = tw[i].imag(); }
            admiral::detail::four_step_batched_ct<N1, N2, T, false>(
                in_re.data(), in_im.data(), out_re.data(), out_im.data(),
                twre.data(), twim.data(), Gre.data(), Gim.data());
        }
    };

    for (bool fwd : {true, false}) {
        run(fwd);
        const auto ref = reference_dft<T>(x, fwd);
        std::vector<std::complex<T>> got(N);
        for (std::size_t k = 0; k < N; ++k) got[k] = {out_re[k], out_im[k]};
        require_close(got, ref, fft_tol<T>());
    }

    run(true);
    std::vector<T> f_re = out_re, f_im = out_im;
    for (std::size_t i = 0; i < N; ++i) { in_re[i] = f_re[i]; in_im[i] = f_im[i]; }
    run(false);
    std::vector<std::complex<T>> back(N);
    for (std::size_t i = 0; i < N; ++i) back[i] = {out_re[i] / T(N), out_im[i] / T(N)};
    require_close(back, x, fft_tol<T>());
}

TEST_CASE("batched four-step matches reference DFT (composition of kernels, N>64)", "[codelet][fourstep]") {
    check_four_step<16, 16, double>();
    check_four_step<32, 32, double>();
    check_four_step<64, 64, double>();
    check_four_step<32, 64, double>();
    check_four_step<16, 16, float>();
    check_four_step<32, 32, float>();
    check_four_step<32, 64, float>();
}

TEST_CASE("codelet kernel<N>::apply_sink matches apply and reference DFT", "[codelet]") {
    auto check = [](auto Nc, auto forward, auto tag) {
        constexpr unsigned N = decltype(Nc)::value;
        constexpr bool Forward = decltype(forward)::value;
        using T = decltype(tag);
        const auto x = make_input<T>(N);
        std::vector<T> xre(N), xim(N), scratch_re(N), scratch_im(N);
        std::vector<T> ref_re(N), ref_im(N), got_re(N, T(1e30)), got_im(N, T(1e30));
        for (std::size_t i = 0; i < N; ++i) { xre[i] = x[i].real(); xim[i] = x[i].imag(); }

        kernel<N, T, Forward>::apply(xre.data(), xim.data(), 1,
                                     ref_re.data(), ref_im.data());
        kernel<N, T, Forward>::apply_sink(
            xre.data(), xim.data(), 1, scratch_re.data(), scratch_im.data(),
            [&](std::size_t p, auto outr, auto outi) {
                using V = decltype(outr);
                if constexpr (std::is_same_v<V, T>) {
                    got_re[p] = outr;
                    got_im[p] = outi;
                } else {
                    outr.store_unaligned(got_re.data() + p);
                    outi.store_unaligned(got_im.data() + p);
                }
            });

        std::vector<std::complex<T>> got(N), via_apply(N);
        for (std::size_t k = 0; k < N; ++k) {
            got[k] = {got_re[k], got_im[k]};
            via_apply[k] = {ref_re[k], ref_im[k]};
        }
        require_close(got, reference_dft<T>(x, Forward), fft_tol<T>());
        require_close(got, via_apply, fft_tol<T>());
    };
    auto both = [&](auto Nc) {
        check(Nc, std::true_type{}, double{});
        check(Nc, std::false_type{}, double{});
        check(Nc, std::true_type{}, float{});
        check(Nc, std::false_type{}, float{});
    };
    both(std::integral_constant<unsigned, 2>{});
    both(std::integral_constant<unsigned, 6>{});
    both(std::integral_constant<unsigned, 12>{});
    both(std::integral_constant<unsigned, 13>{});
    both(std::integral_constant<unsigned, 15>{});
    both(std::integral_constant<unsigned, 16>{});
    both(std::integral_constant<unsigned, 20>{});
    both(std::integral_constant<unsigned, 25>{});
    both(std::integral_constant<unsigned, 39>{});
    both(std::integral_constant<unsigned, 52>{});
    both(std::integral_constant<unsigned, 64>{});
    both(std::integral_constant<unsigned, 128>{});
}

TEST_CASE("codelet forward then inverse is identity (up to 1/N)", "[codelet]") {
    auto roundtrip = [](auto Nc) {
        constexpr unsigned N = decltype(Nc)::value;
        using T = double;
        const auto x = make_input<T>(N);
        std::vector<T> xre(N), xim(N), fre(N), fim(N), gre(N), gim(N);
        for (std::size_t i = 0; i < N; ++i) { xre[i] = x[i].real(); xim[i] = x[i].imag(); }

        kernel<N, T, true>::apply(xre.data(), xim.data(), 1, fre.data(), fim.data());
        kernel<N, T, false>::apply(fre.data(), fim.data(), 1, gre.data(), gim.data());

        std::vector<std::complex<T>> back(N);
        for (std::size_t i = 0; i < N; ++i) back[i] = {gre[i] / T(N), gim[i] / T(N)};
        require_close(back, x, fft_tol<T>());
    };
    roundtrip(std::integral_constant<unsigned, 12>{});
    roundtrip(std::integral_constant<unsigned, 60>{});
    roundtrip(std::integral_constant<unsigned, 64>{});
}

namespace {

// The batched leaves transform nlines lines through a blocked gather, one straight-line kernel and
// a blocked scatter, and a compile-time gate on the block count N / batch width picks a rolled or a
// fully static block loop per size. Nothing else in the tree reaches either arm, so this is the
// check on both. The single-line leaf is the reference, itself checked against the reference DFT
// above, and the guard value in the padding fails the case if a block writes past its own columns.
template<typename T, bool Forward>
void check_many(std::size_t N, std::size_t nlines, std::size_t stride, T fct) {
    REQUIRE(stride >= N);
    const std::complex<T> guard(T(-9), T(7));
    const auto x = make_input<T>(nlines * stride, 0x5EEDu);

    std::vector<std::complex<T>> want(nlines * stride, guard);
    std::vector<std::complex<T>> line(N);
    for (std::size_t r = 0; r < nlines; ++r) {
        codelet_dispatch<T, Forward>(x.data() + r * stride, line.data(), N);
        for (std::size_t k = 0; k < N; ++k) want[r * stride + k] = line[k] * fct;
    }

    std::vector<std::complex<T>> oop(nlines * stride, guard);
    codelet_dispatch_many_oop<T, Forward>(x.data(), oop.data(), nlines, stride, stride, N, fct);
    require_close(oop, want, fft_tol<T>());

    auto ip = x;
    for (std::size_t r = 0; r < nlines; ++r)
        for (std::size_t k = N; k < stride; ++k) ip[r * stride + k] = guard;
    codelet_dispatch_many<T, Forward>(ip.data(), nlines, stride, N, fct);
    require_close(ip, want, fft_tol<T>());
}

template<typename T>
void check_many_sizes() {
    // Sweep the catalog itself: ADM_CODELET_EXTRA_SIZES changes which sizes exist, so a hardcoded
    // list is a different test in every configuration. kGate mirrors kManyRollMinBlocks in
    // src/codelet_apply.hpp, and the counts below fail if a catalog stops reaching an arm it can
    // reach. Whether the rolled arm is reachable at all is a property of the catalog and the batch
    // width, not of the code: the largest size has to carry kGate blocks. Every size is under the
    // gate at the smallest catalog a sanitizer build accepts, so requiring the arm unconditionally
    // would fail there for a reason no source change can fix.
    constexpr std::size_t W = xsimd::batch<T>::size;
    constexpr std::size_t kGate = 5;
    std::size_t rolled = 0, statik = 0;

    for (const std::size_t N : CODELET_CATALOG_SIZES) {
        if (N < 2) continue;
        ++(N / W >= kGate ? rolled : statik);
        for (const bool forward : {true, false})
            for (const T fct : {T(1), T(0.5)})
                // Under, at and over the batch width, and counts that leave a partial block, so the
                // block loop, the remainder block and the fewer-than-width residual loop all run.
                for (const std::size_t nlines : {std::size_t{1}, std::size_t{3}, std::size_t{9},
                                                 std::size_t{17}})
                    for (const std::size_t stride : {N, N + 3}) {
                        if (forward) check_many<T, true>(N, nlines, stride, fct);
                        else         check_many<T, false>(N, nlines, stride, fct);
                    }
    }
    if (CODELET_CATALOG_SIZES.back() / W >= kGate) REQUIRE(rolled > 0);
    REQUIRE(statik > 0);
}

}

TEMPLATE_TEST_CASE("batched codelet leaves match the single-line leaf, both gate arms",
                   "[codelet][batched]", float, double) {
    check_many_sizes<TestType>();
}
